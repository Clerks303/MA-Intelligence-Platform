"""
API endpoints pour dashboard monitoring temps réel
US-003: Endpoints REST pour visualisation monitoring et observabilité

Endpoints:
- GET /monitoring/overview - Vue d'ensemble système
- GET /monitoring/metrics - Métriques détaillées
- GET /monitoring/alerts - Alertes actives
- GET /monitoring/health - Health checks
- GET /monitoring/logs - Logs récents
- GET /monitoring/performance - Métriques performance
- POST /monitoring/alerts/{id}/acknowledge - Acquitter alerte
"""

from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

from app.core.dependencies import get_current_active_user
from app.core.logging_system import get_logger, LogCategory
from app.core.metrics_collector import get_metrics_collector
from app.core.health_monitor import get_health_monitor, quick_health_check
from app.core.alerting_system import get_alerting_system, AlertSeverity, AlertCategory
from app.models.user import User

logger = get_logger("monitoring_api", LogCategory.API)
router = APIRouter(prefix="/monitoring", tags=["Monitoring Dashboard"])


@router.get("/overview", summary="Vue d'ensemble système monitoring")
async def get_monitoring_overview(
    current_user: User = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """
    Vue d'ensemble complète du monitoring système
    
    Retourne:
    - Statut global du système
    - Métriques clés en temps réel
    - Alertes critiques actives
    - Performance générale
    """
    try:
        # Collecteurs de données
        metrics_collector = get_metrics_collector()
        health_monitor = await get_health_monitor()
        alerting_system = await get_alerting_system()
        
        # Métriques business importantes
        business_metrics = metrics_collector.get_business_metrics()
        
        # Santé système globale
        health_overview = await health_monitor.get_system_health_overview()
        
        # Alertes critiques
        critical_alerts = alerting_system.get_active_alerts(severity=AlertSeverity.CRITICAL)
        emergency_alerts = alerting_system.get_active_alerts(severity=AlertSeverity.EMERGENCY)
        
        # Statistiques alertes
        alert_stats = alerting_system.get_alert_statistics(hours=24)
        
        # Performance key metrics
        performance_summary = metrics_collector.get_metrics_summary(window_seconds=300)  # 5 min
        
        return {
            'timestamp': datetime.now().isoformat(),
            'system_status': {
                'overall_health': health_overview.get('overall_status', 'unknown'),
                'services_healthy': health_overview.get('summary', {}).get('healthy_services', 0),
                'services_total': health_overview.get('summary', {}).get('total_services', 0),
                'availability_percent': health_overview.get('summary', {}).get('availability_percent', 0)
            },
            'critical_issues': {
                'critical_alerts_count': len(critical_alerts),
                'emergency_alerts_count': len(emergency_alerts),
                'services_down': health_overview.get('critical_issues', []),
                'immediate_attention_required': len(critical_alerts) + len(emergency_alerts) > 0
            },
            'key_metrics': {
                'api_requests_last_hour': business_metrics.get('api_usage', {}).get('requests_last_hour', 0),
                'avg_response_time_ms': business_metrics.get('api_usage', {}).get('avg_response_time_ms', 0),
                'error_rate_percent': business_metrics.get('api_usage', {}).get('error_rate_percent', 0),
                'cache_hit_ratio': business_metrics.get('business_value', {}).get('cache_hit_ratio', 0),
                'active_users': business_metrics.get('business_value', {}).get('active_users_last_hour', 0)
            },
            'performance_summary': {
                'system_cpu_percent': business_metrics.get('system_health', {}).get('system_cpu_percent', 0),
                'system_memory_percent': business_metrics.get('system_health', {}).get('system_memory_percent', 0),
                'database_health': 'healthy' if health_overview.get('services', {}).get('database', {}).get('status') == 'healthy' else 'degraded',
                'cache_health': 'healthy' if health_overview.get('services', {}).get('redis_cache', {}).get('status') == 'healthy' else 'degraded'
            },
            'alert_trends': {
                'alerts_last_24h': alert_stats.get('total_alerts', 0),
                'alerts_last_hour': alerting_system.get_alert_statistics(hours=1).get('total_alerts', 0),
                'avg_resolution_time_minutes': alert_stats.get('avg_resolution_time_minutes', 0),
                'top_alert_rules': alert_stats.get('top_rules', [])[:3]
            },
            'business_health': {
                'companies_scraped_today': business_metrics.get('scraping_performance', {}).get('companies_scraped_last_day', 0),
                'ma_scores_calculated': business_metrics.get('business_value', {}).get('ma_scores_calculated', 0),
                'exports_generated': business_metrics.get('business_value', {}).get('exports_generated', 0),
                'scraping_success_rate': business_metrics.get('scraping_performance', {}).get('success_rate_percent', 0)
            }
        }
        
    except Exception as e:
        logger.error("Erreur récupération overview monitoring", exception=e)
        raise HTTPException(status_code=500, detail=f"Erreur monitoring: {str(e)}")


@router.get("/metrics", summary="Métriques détaillées système")
async def get_detailed_metrics(
    window_minutes: int = Query(5, ge=1, le=1440, description="Fenêtre temporelle en minutes"),
    category: Optional[str] = Query(None, description="Catégorie de métriques"),
    current_user: User = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """
    Métriques détaillées avec agrégations temporelles
    
    Args:
        window_minutes: Fenêtre d'agrégation (1 minute à 24h)
        category: Filtrer par catégorie (api, system, business, cache)
    """
    try:
        metrics_collector = get_metrics_collector()
        window_seconds = window_minutes * 60
        
        # Métriques agrégées
        metrics_summary = metrics_collector.get_metrics_summary(window_seconds)
        
        # Métriques business détaillées
        business_metrics = metrics_collector.get_business_metrics()
        
        result = {
            'timestamp': datetime.now().isoformat(),
            'window_minutes': window_minutes,
            'metrics_summary': metrics_summary,
            'business_metrics': business_metrics
        }
        
        # Filtrage par catégorie si demandé
        if category:
            if category == "api":
                result['filtered_metrics'] = {
                    k: v for k, v in metrics_summary.items() 
                    if 'api_' in k
                }
            elif category == "system":
                result['filtered_metrics'] = {
                    k: v for k, v in metrics_summary.items() 
                    if 'system_' in k
                }
            elif category == "cache":
                result['filtered_metrics'] = {
                    k: v for k, v in metrics_summary.items() 
                    if 'cache_' in k or 'redis_' in k
                }
            elif category == "business":
                result['filtered_metrics'] = business_metrics
        
        return result
        
    except Exception as e:
        logger.error("Erreur récupération métriques détaillées", exception=e)
        raise HTTPException(status_code=500, detail=f"Erreur métriques: {str(e)}")


@router.get("/health", summary="État santé services")
async def get_health_status(
    detailed: bool = Query(False, description="Inclure diagnostics détaillés"),
    service: Optional[str] = Query(None, description="Service spécifique"),
    current_user: User = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """
    État de santé des services avec diagnostics optionnels
    
    Args:
        detailed: Si True, inclut diagnostics complets
        service: Nom du service pour diagnostics spécifiques
    """
    try:
        health_monitor = await get_health_monitor()
        
        if service:
            # Diagnostics service spécifique
            diagnostics = await health_monitor.get_service_diagnostics(service)
            return {
                'timestamp': datetime.now().isoformat(),
                'service': service,
                'diagnostics': diagnostics
            }
        
        elif detailed:
            # Vue complète détaillée
            overview = await health_monitor.get_system_health_overview()
            return overview
        
        else:
            # Health check rapide
            quick_status = await quick_health_check()
            return quick_status
        
    except Exception as e:
        logger.error("Erreur récupération health status", exception=e)
        raise HTTPException(status_code=500, detail=f"Erreur health check: {str(e)}")


@router.get("/alerts", summary="Alertes système")
async def get_alerts_overview(
    status: Optional[str] = Query(None, regex="^(active|resolved|acknowledged)$"),
    severity: Optional[str] = Query(None, regex="^(info|warning|error|critical|emergency)$"),
    hours: int = Query(24, ge=1, le=168, description="Période en heures"),
    current_user: User = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """
    Vue d'ensemble des alertes avec filtres
    
    Args:
        status: Filtrer par statut
        severity: Filtrer par sévérité 
        hours: Période d'analyse
    """
    try:
        alerting_system = await get_alerting_system()
        
        # Dashboard complet alertes
        dashboard_data = await alerting_system.get_alerting_dashboard_data()
        
        # Statistiques période demandée
        period_stats = alerting_system.get_alert_statistics(hours)
        
        # Filtrage alertes actives si demandé
        active_alerts = alerting_system.get_active_alerts()
        
        if severity:
            try:
                severity_enum = AlertSeverity(severity)
                active_alerts = [a for a in active_alerts if a.severity == severity_enum]
            except ValueError:
                pass
        
        # Compléter dashboard data
        dashboard_data['period_statistics'] = period_stats
        dashboard_data['filtered_active_alerts'] = [a.to_dict() for a in active_alerts[:20]]
        
        return dashboard_data
        
    except Exception as e:
        logger.error("Erreur récupération alertes", exception=e)
        raise HTTPException(status_code=500, detail=f"Erreur alertes: {str(e)}")


@router.post("/alerts/{alert_id}/acknowledge", summary="Acquitter une alerte")
async def acknowledge_alert(
    alert_id: str,
    comment: str = Query("", description="Commentaire optionnel"),
    current_user: User = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """
    Acquitte une alerte spécifique
    
    Args:
        alert_id: ID de l'alerte
        comment: Commentaire optionnel
    """
    try:
        alerting_system = await get_alerting_system()
        
        await alerting_system.acknowledge_alert(
            alert_id=alert_id,
            user_id=current_user.username,
            comment=comment
        )
        
        logger.info(f"Alerte {alert_id} acquittée par {current_user.username}")
        
        return {
            'timestamp': datetime.now().isoformat(),
            'alert_id': alert_id,
            'status': 'acknowledged',
            'acknowledged_by': current_user.username,
            'comment': comment
        }
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Erreur acquittement alerte {alert_id}", exception=e)
        raise HTTPException(status_code=500, detail=f"Erreur acquittement: {str(e)}")


@router.post("/alerts/{alert_id}/resolve", summary="Résoudre une alerte")
async def resolve_alert(
    alert_id: str,
    comment: str = Query("", description="Commentaire résolution"),
    current_user: User = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """
    Résout manuellement une alerte
    """
    try:
        alerting_system = await get_alerting_system()
        
        await alerting_system.resolve_alert(
            alert_id=alert_id,
            user_id=current_user.username,
            comment=comment
        )
        
        logger.info(f"Alerte {alert_id} résolue par {current_user.username}")
        
        return {
            'timestamp': datetime.now().isoformat(),
            'alert_id': alert_id,
            'status': 'resolved',
            'resolved_by': current_user.username,
            'comment': comment
        }
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Erreur résolution alerte {alert_id}", exception=e)
        raise HTTPException(status_code=500, detail=f"Erreur résolution: {str(e)}")


@router.get("/performance", summary="Métriques performance détaillées")
async def get_performance_metrics(
    component: Optional[str] = Query(None, regex="^(api|database|cache|scraping|system)$"),
    timerange: str = Query("1h", regex="^(5m|15m|1h|6h|24h)$"),
    current_user: User = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """
    Métriques de performance détaillées par composant
    
    Args:
        component: Composant spécifique
        timerange: Plage temporelle (5m, 15m, 1h, 6h, 24h)
    """
    try:
        metrics_collector = get_metrics_collector()
        
        # Conversion timerange en secondes
        timerange_seconds = {
            '5m': 300,
            '15m': 900, 
            '1h': 3600,
            '6h': 21600,
            '24h': 86400
        }[timerange]
        
        # Métriques générales
        performance_data = metrics_collector.get_metrics_summary(timerange_seconds)
        business_metrics = metrics_collector.get_business_metrics()
        
        result = {
            'timestamp': datetime.now().isoformat(),
            'timerange': timerange,
            'component_filter': component,
            'performance_overview': {
                'timerange_seconds': timerange_seconds,
                'general_metrics': performance_data
            }
        }
        
        # Métriques spécifiques par composant
        if component == "api":
            result['api_performance'] = {
                'requests_per_second': business_metrics.get('api_usage', {}).get('requests_last_hour', 0) / 3600,
                'avg_response_time_ms': business_metrics.get('api_usage', {}).get('avg_response_time_ms', 0),
                'error_rate_percent': business_metrics.get('api_usage', {}).get('error_rate_percent', 0),
                'p95_response_time': 'Not available',  # À implémenter avec historique
                'endpoints_performance': {}  # À enrichir
            }
        
        elif component == "database":
            result['database_performance'] = {
                'connection_pool_usage': business_metrics.get('system_health', {}).get('database_connection_pool', 0),
                'avg_query_time_ms': 'Not available',  # À implémenter
                'active_connections': 'Not available',
                'slow_queries_count': 'Not available'
            }
        
        elif component == "cache":
            result['cache_performance'] = {
                'hit_ratio_percent': business_metrics.get('business_value', {}).get('cache_hit_ratio', 0),
                'memory_usage_mb': business_metrics.get('system_health', {}).get('redis_memory_usage_mb', 0),
                'operations_per_second': 'Not available',  # À calculer
                'latency_p95_ms': 'Not available'
            }
        
        elif component == "scraping":
            result['scraping_performance'] = {
                'companies_scraped_last_hour': business_metrics.get('scraping_performance', {}).get('companies_scraped_last_hour', 0),
                'avg_scraping_time_ms': business_metrics.get('scraping_performance', {}).get('avg_scraping_time_ms', 0),
                'success_rate_percent': business_metrics.get('scraping_performance', {}).get('success_rate_percent', 0),
                'queue_size': 'Not available',  # À implémenter
                'failed_operations': 'Not available'
            }
        
        elif component == "system":
            result['system_performance'] = {
                'cpu_usage_percent': business_metrics.get('system_health', {}).get('system_cpu_percent', 0),
                'memory_usage_percent': business_metrics.get('system_health', {}).get('system_memory_percent', 0),
                'disk_usage_percent': 'Not available',  # À ajouter
                'network_io': 'Not available',
                'load_average': 'Not available'
            }
        
        else:
            # Vue d'ensemble tous composants
            result['all_components'] = {
                'api': business_metrics.get('api_usage', {}),
                'scraping': business_metrics.get('scraping_performance', {}),
                'business': business_metrics.get('business_value', {}),
                'system': business_metrics.get('system_health', {})
            }
        
        return result
        
    except Exception as e:
        logger.error("Erreur récupération métriques performance", exception=e)
        raise HTTPException(status_code=500, detail=f"Erreur performance: {str(e)}")


@router.get("/logs", summary="Logs système récents")
async def get_recent_logs(
    level: Optional[str] = Query(None, regex="^(DEBUG|INFO|WARNING|ERROR|CRITICAL|AUDIT)$"),
    component: Optional[str] = Query(None, description="Composant système"),
    limit: int = Query(100, ge=1, le=1000, description="Nombre max de logs"),
    current_user: User = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """
    Logs système récents avec filtres
    
    Args:
        level: Niveau de log
        component: Composant spécifique
        limit: Nombre maximum de logs
    """
    try:
        # Cette implémentation nécessiterait un système de collecte centralisée des logs
        # Pour l'instant, retourner des logs simulés ou pointer vers les fichiers
        
        return {
            'timestamp': datetime.now().isoformat(),
            'filters': {
                'level': level,
                'component': component,
                'limit': limit
            },
            'logs': [
                {
                    'timestamp': datetime.now().isoformat(),
                    'level': 'INFO',
                    'component': 'api',
                    'message': 'Logs endpoint accessed',
                    'context': {'user': current_user.username}
                }
            ],
            'note': 'Log aggregation system to be implemented - check log files in logs/ directory'
        }
        
    except Exception as e:
        logger.error("Erreur récupération logs", exception=e)
        raise HTTPException(status_code=500, detail=f"Erreur logs: {str(e)}")


@router.get("/export/metrics", summary="Export métriques Prometheus")
async def export_prometheus_metrics(
    current_user: User = Depends(get_current_active_user)
) -> str:
    """
    Export des métriques au format Prometheus
    """
    try:
        metrics_collector = get_metrics_collector()
        prometheus_data = metrics_collector.export_prometheus_format()
        
        return prometheus_data
        
    except Exception as e:
        logger.error("Erreur export Prometheus", exception=e)
        raise HTTPException(status_code=500, detail=f"Erreur export: {str(e)}")


@router.post("/test/alert", summary="Tester le système d'alerting")
async def test_alerting_system(
    title: str = Query(..., description="Titre de l'alerte test"),
    severity: str = Query("warning", regex="^(info|warning|error|critical|emergency)$"),
    current_user: User = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """
    Envoie une alerte de test pour vérifier le système
    
    Uniquement disponible pour les admins
    """
    try:
        # Vérification admin (à implémenter selon système d'autorisation)
        if not getattr(current_user, 'is_admin', False):
            raise HTTPException(status_code=403, detail="Accès admin requis")
        
        alerting_system = await get_alerting_system()
        
        # Import local pour éviter dépendance circulaire
        from app.core.alerting_system import send_custom_alert, AlertSeverity, AlertCategory
        
        severity_enum = AlertSeverity(severity)
        
        await send_custom_alert(
            title=f"TEST: {title}",
            description=f"Alerte de test générée par {current_user.username}",
            severity=severity_enum,
            category=AlertCategory.SYSTEM
        )
        
        logger.info(f"Alerte de test envoyée par {current_user.username}: {title}")
        
        return {
            'timestamp': datetime.now().isoformat(),
            'status': 'test_alert_sent',
            'title': title,
            'severity': severity,
            'sent_by': current_user.username
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Erreur test alerting", exception=e)
        raise HTTPException(status_code=500, detail=f"Erreur test: {str(e)}")


@router.get("/dashboard", summary="Données complètes dashboard")
async def get_complete_dashboard_data(
    current_user: User = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """
    Toutes les données nécessaires pour le dashboard monitoring
    
    Optimisé pour affichage dashboard temps réel
    """
    try:
        # Appels parallèles pour performance
        overview_task = get_monitoring_overview(current_user)
        health_task = get_health_status(detailed=False, current_user=current_user)
        alerts_task = get_alerts_overview(current_user=current_user)
        performance_task = get_performance_metrics(timerange="15m", current_user=current_user)
        
        # Attendre tous les résultats
        overview = await overview_task
        health = await health_task
        alerts = await alerts_task
        performance = await performance_task
        
        return {
            'timestamp': datetime.now().isoformat(),
            'dashboard_sections': {
                'overview': overview,
                'health': health,
                'alerts': alerts,
                'performance': performance
            },
            'refresh_interval_seconds': 30,
            'last_updated': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error("Erreur dashboard complet", exception=e)
        raise HTTPException(status_code=500, detail=f"Erreur dashboard: {str(e)}")