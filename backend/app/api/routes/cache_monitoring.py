"""
API endpoints pour monitoring cache Redis
US-002: Endpoints REST pour dashboard et alerting cache

Endpoints:
- GET /cache/status - Statut général cache
- GET /cache/metrics - Métriques détaillées
- GET /cache/dashboard - Données dashboard
- GET /cache/health - Health check complet
- POST /cache/invalidate - Invalidation manuelle
- GET /cache/report - Rapport de performance
"""

from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import logging

from app.core.cache import get_cache, CacheType
from app.core.cache_monitoring import get_cache_monitor
from app.core.dependencies import get_current_active_user
from app.models.user import User

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/cache", tags=["Cache Monitoring"])


@router.get("/status", summary="Statut général du cache")
async def get_cache_status(
    current_user: User = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """
    Statut rapide du cache Redis
    
    Retourne statut de connexion et métriques de base
    """
    try:
        cache = await get_cache()
        
        # Health check rapide
        health = await cache.health_check()
        
        # Informations de base
        cache_info = await cache.get_cache_info()
        
        return {
            "status": health.get("status", "unknown"),
            "timestamp": datetime.now().isoformat(),
            "connection_healthy": health.get("ping_success", False),
            "ping_latency_ms": health.get("ping_latency_ms", 0),
            "hit_ratio_percent": cache_info.get("cache_metrics", {}).get("hit_ratio_percent", 0),
            "memory_used_mb": cache_info.get("redis_info", {}).get("memory_used_mb", 0),
            "operations_per_second": cache_info.get("cache_metrics", {}).get("operations_per_second", 0)
        }
        
    except Exception as e:
        logger.error(f"❌ Erreur statut cache: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur cache: {str(e)}")


@router.get("/metrics", summary="Métriques détaillées du cache")
async def get_cache_metrics(
    cache_type: Optional[str] = Query(None, description="Type de cache spécifique"),
    current_user: User = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """
    Métriques complètes du cache
    
    Args:
        cache_type: Type de cache spécifique (optionnel)
    """
    try:
        cache = await get_cache()
        cache_info = await cache.get_cache_info()
        
        result = {
            "timestamp": datetime.now().isoformat(),
            "global_metrics": cache_info.get("cache_metrics", {}),
            "redis_info": cache_info.get("redis_info", {}),
            "cache_types": cache_info.get("cache_stats_by_type", {})
        }
        
        # Filtrer par type si demandé
        if cache_type and cache_type in result["cache_types"]:
            result["filtered_cache_type"] = {
                cache_type: result["cache_types"][cache_type]
            }
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Erreur métriques cache: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur métriques: {str(e)}")


@router.get("/dashboard", summary="Données dashboard monitoring")
async def get_cache_dashboard(
    current_user: User = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """
    Données complètes pour dashboard de monitoring
    
    Inclut métriques, alertes, tendances et recommandations
    """
    try:
        monitor = await get_cache_monitor()
        
        dashboard_data = await monitor.get_cache_dashboard_data()
        
        return dashboard_data
        
    except Exception as e:
        logger.error(f"❌ Erreur dashboard cache: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur dashboard: {str(e)}")


@router.get("/health", summary="Health check complet")
async def get_cache_health(
    run_performance_tests: bool = Query(False, description="Exécuter tests de performance"),
    current_user: User = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """
    Health check approfondi du cache
    
    Args:
        run_performance_tests: Si True, exécute des tests de performance
    """
    try:
        if run_performance_tests:
            monitor = await get_cache_monitor()
            health_result = await monitor.trigger_cache_health_check()
        else:
            cache = await get_cache()
            health_result = await cache.health_check()
            health_result["performance_tests"] = "skipped"
        
        return health_result
        
    except Exception as e:
        logger.error(f"❌ Erreur health check cache: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur health check: {str(e)}")


@router.post("/invalidate", summary="Invalidation manuelle du cache")
async def invalidate_cache(
    pattern: str = Query(..., description="Pattern de clés à invalider"),
    cache_type: Optional[str] = Query(None, description="Type de cache"),
    confirm: bool = Query(False, description="Confirmation d'invalidation"),
    current_user: User = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """
    Invalidation manuelle de clés cache
    
    Args:
        pattern: Pattern de clés (ex: "companies:*", "scoring:*")
        cache_type: Type de cache spécifique (optionnel)
        confirm: Confirmation requise pour éviter suppressions accidentelles
        
    Returns:
        Nombre de clés supprimées
    """
    try:
        if not confirm:
            raise HTTPException(
                status_code=400, 
                detail="Paramètre 'confirm=true' requis pour invalidation"
            )
        
        cache = await get_cache()
        
        # Validation pattern sécurisé
        dangerous_patterns = ["*", "**", "cache:*"]
        if pattern in dangerous_patterns:
            raise HTTPException(
                status_code=400,
                detail=f"Pattern dangereux non autorisé: {pattern}"
            )
        
        # Conversion cache_type
        cache_type_enum = None
        if cache_type:
            try:
                cache_type_enum = CacheType(cache_type)
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Type de cache invalide: {cache_type}"
                )
        
        # Invalidation
        deleted_count = await cache.invalidate_pattern(pattern, cache_type_enum)
        
        # Log action admin
        logger.warning(
            f"🧹 Invalidation cache manuelle par {current_user.username}: "
            f"pattern='{pattern}', type='{cache_type}', supprimées={deleted_count}"
        )
        
        return {
            "timestamp": datetime.now().isoformat(),
            "pattern": pattern,
            "cache_type": cache_type,
            "deleted_keys_count": deleted_count,
            "invalidated_by": current_user.username
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Erreur invalidation cache: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur invalidation: {str(e)}")


@router.get("/report", summary="Rapport de performance détaillé")
async def get_cache_performance_report(
    hours: int = Query(24, ge=1, le=168, description="Période d'analyse en heures"),
    format: str = Query("json", regex="^(json|summary)$", description="Format du rapport"),
    current_user: User = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """
    Rapport de performance détaillé sur période
    
    Args:
        hours: Période d'analyse (1-168 heures)
        format: Format de sortie (json ou summary)
    """
    try:
        monitor = await get_cache_monitor()
        
        # Génération rapport
        report = await monitor.get_performance_report(hours)
        
        if format == "summary":
            # Version résumée
            summary = report.get("summary", {})
            return {
                "report_type": "summary",
                "period_hours": hours,
                "generated_at": datetime.now().isoformat(),
                "key_metrics": {
                    "hit_ratio_percent": summary.get("overall_hit_ratio", 0),
                    "total_operations": summary.get("total_operations", 0),
                    "avg_latency_ms": summary.get("avg_latency_ms", 0),
                    "estimated_savings_euros": summary.get("estimated_savings_euros", 0)
                },
                "status": "healthy" if summary.get("total_alerts", 0) == 0 else "alerts_present"
            }
        else:
            # Rapport complet
            return report
        
    except Exception as e:
        logger.error(f"❌ Erreur rapport cache: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur rapport: {str(e)}")


@router.get("/alerts", summary="Alertes cache récentes")
async def get_cache_alerts(
    hours: int = Query(24, ge=1, le=168, description="Période des alertes"),
    level: Optional[str] = Query(None, regex="^(info|warning|critical)$", description="Niveau d'alerte"),
    current_user: User = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """
    Alertes cache sur période
    
    Args:
        hours: Période de recherche
        level: Filtrer par niveau d'alerte
    """
    try:
        monitor = await get_cache_monitor()
        
        # Filtrer alertes par période
        since = datetime.now() - timedelta(hours=hours)
        
        alerts = [
            {
                "level": alert.level.value,
                "message": alert.message,
                "metric_name": alert.metric_name,
                "current_value": alert.current_value,
                "threshold": alert.threshold,
                "timestamp": alert.timestamp.isoformat(),
                "cache_type": alert.cache_type.value if alert.cache_type else None
            }
            for alert in monitor.alerts_history
            if alert.timestamp > since and (not level or alert.level.value == level)
        ]
        
        # Grouper par niveau
        alerts_by_level = {}
        for alert in alerts:
            alert_level = alert["level"]
            if alert_level not in alerts_by_level:
                alerts_by_level[alert_level] = []
            alerts_by_level[alert_level].append(alert)
        
        return {
            "timestamp": datetime.now().isoformat(),
            "period_hours": hours,
            "filter_level": level,
            "total_alerts": len(alerts),
            "alerts_by_level": alerts_by_level,
            "alerts": sorted(alerts, key=lambda x: x["timestamp"], reverse=True)
        }
        
    except Exception as e:
        logger.error(f"❌ Erreur alertes cache: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur alertes: {str(e)}")


@router.post("/optimize", summary="Optimisation automatique du cache")
async def optimize_cache(
    background_tasks: BackgroundTasks,
    dry_run: bool = Query(True, description="Mode simulation (pas d'actions réelles)"),
    current_user: User = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """
    Déclenche optimisation automatique du cache
    
    Args:
        dry_run: Si True, simule les optimisations sans les appliquer
        
    Returns:
        Plan d'optimisation et actions recommandées
    """
    try:
        monitor = await get_cache_monitor()
        
        # Analyser état actuel
        dashboard_data = await monitor.get_cache_dashboard_data()
        recommendations = dashboard_data.get("recommendations", [])
        
        optimization_plan = {
            "timestamp": datetime.now().isoformat(),
            "dry_run": dry_run,
            "current_status": dashboard_data.get("overall_status", "unknown"),
            "recommendations": recommendations,
            "planned_actions": []
        }
        
        # Générer actions selon recommandations
        for recommendation in recommendations:
            if "hit ratio" in recommendation.lower():
                optimization_plan["planned_actions"].append({
                    "action": "adjust_ttl",
                    "description": "Ajuster TTL pour améliorer hit ratio",
                    "impact": "medium",
                    "estimated_improvement": "+5% hit ratio"
                })
            
            if "latence" in recommendation.lower():
                optimization_plan["planned_actions"].append({
                    "action": "optimize_serialization",
                    "description": "Optimiser sérialisation des objets volumineux",
                    "impact": "high",
                    "estimated_improvement": "-20ms latence"
                })
        
        if not dry_run:
            # En mode réel, déclencher optimisations en arrière-plan
            background_tasks.add_task(_run_cache_optimizations, optimization_plan)
            optimization_plan["status"] = "optimizations_started"
        else:
            optimization_plan["status"] = "simulation_only"
        
        return optimization_plan
        
    except Exception as e:
        logger.error(f"❌ Erreur optimisation cache: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur optimisation: {str(e)}")


async def _run_cache_optimizations(optimization_plan: Dict[str, Any]):
    """Exécution optimisations en arrière-plan"""
    try:
        logger.info("🔧 Démarrage optimisations cache automatiques...")
        
        # Ici, implémenter les optimisations réelles
        # Ex: ajuster TTL, nettoyer clés inutiles, etc.
        
        # Simulation pour l'instant
        await asyncio.sleep(5)
        
        logger.info("✅ Optimisations cache terminées")
        
    except Exception as e:
        logger.error(f"❌ Erreur optimisations cache: {e}")


@router.get("/statistics/{cache_type}", summary="Statistiques par type de cache")
async def get_cache_type_statistics(
    cache_type: str,
    period_hours: int = Query(24, ge=1, le=168),
    current_user: User = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """
    Statistiques détaillées pour un type de cache spécifique
    
    Args:
        cache_type: Type de cache (enrichment_pappers, scoring_ma, etc.)
        period_hours: Période d'analyse
    """
    try:
        # Validation type cache
        try:
            cache_type_enum = CacheType(cache_type)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Type de cache invalide: {cache_type}. "
                       f"Types disponibles: {[ct.value for ct in CacheType]}"
            )
        
        cache = await get_cache()
        cache_info = await cache.get_cache_info()
        
        # Statistiques spécifiques au type
        type_stats = cache_info.get("cache_stats_by_type", {}).get(cache_type, {})
        
        return {
            "cache_type": cache_type,
            "period_hours": period_hours,
            "timestamp": datetime.now().isoformat(),
            "statistics": {
                "key_count": type_stats.get("key_count", 0),
                "estimated_memory_usage_mb": type_stats.get("key_count", 0) * 0.001,
                "default_ttl_seconds": type_stats.get("config", 3600),
                "usage_pattern": "active" if type_stats.get("key_count", 0) > 10 else "low"
            },
            "recommendations": [
                f"Type de cache {cache_type} avec {type_stats.get('key_count', 0)} clés",
                "Performance normale" if type_stats.get("key_count", 0) > 0 else "Peu utilisé"
            ]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Erreur stats type cache: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur statistiques: {str(e)}")