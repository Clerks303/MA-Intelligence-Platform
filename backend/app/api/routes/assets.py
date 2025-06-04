"""
Routes API pour l'optimisation et gestion des assets
US-009: Endpoints pour compression, minification et cache des ressources

Endpoints:
- GET /assets/metrics - Métriques d'optimisation
- POST /assets/optimize - Optimiser assets manuellement
- GET /assets/report - Rapport d'optimisation
- POST /assets/precompress - Précompresser assets statiques
- POST /assets/clear-cache - Vider le cache des assets
- GET /assets/service-worker - Générer Service Worker
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query
from typing import Dict, List, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field

from app.core.asset_optimizer import (
    get_asset_optimizer,
    get_asset_optimization_report,
    precompress_static_assets,
    generate_service_worker
)
from app.core.dependencies import get_current_active_user
from app.models.user import User
from app.core.logging_system import get_logger, LogCategory

logger = get_logger("assets_api", LogCategory.API)

router = APIRouter(prefix="/assets", tags=["assets"])


class AssetOptimizationRequest(BaseModel):
    """Requête d'optimisation d'assets"""
    content: str = Field(..., description="Contenu à optimiser")
    content_type: str = Field("text/html", description="Type MIME du contenu")
    enable_compression: bool = Field(True, description="Activer la compression")
    enable_minification: bool = Field(True, description="Activer la minification")


class PrecompressionRequest(BaseModel):
    """Requête de précompression"""
    static_directory: str = Field(..., description="Répertoire des assets statiques")
    file_patterns: Optional[List[str]] = Field(None, description="Patterns de fichiers à traiter")


class ServiceWorkerRequest(BaseModel):
    """Requête de génération Service Worker"""
    cache_files: List[str] = Field(..., description="Fichiers à mettre en cache")
    output_path: str = Field("./sw.js", description="Chemin de sortie du Service Worker")


@router.get("/metrics")
async def get_asset_metrics(
    current_user: User = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """
    Récupère les métriques d'optimisation des assets
    
    Retourne les statistiques de compression, cache, et performance
    """
    try:
        optimizer = await get_asset_optimizer()
        metrics = optimizer.get_metrics()
        
        return {
            "success": True,
            "metrics": metrics,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Erreur récupération métriques assets: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur récupération métriques: {str(e)}"
        )


@router.post("/optimize")
async def optimize_content(
    request: AssetOptimizationRequest,
    accept_encoding: str = Query("gzip, br", description="Encodages acceptés"),
    current_user: User = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """
    Optimise du contenu (compression, minification)
    
    Permet de tester l'optimisation sur du contenu spécifique
    """
    try:
        optimizer = await get_asset_optimizer()
        
        # Créer une requête factice
        from fastapi import Request
        fake_request = Request({
            "type": "http", 
            "headers": [(b"accept-encoding", accept_encoding.encode())]
        })
        
        # Optimiser le contenu
        optimized_content, headers = await optimizer.optimize_response(
            request.content,
            request.content_type,
            fake_request
        )
        
        original_size = len(request.content.encode('utf-8'))
        optimized_size = len(optimized_content)
        
        return {
            "success": True,
            "optimization_result": {
                "original_size": original_size,
                "optimized_size": optimized_size,
                "compression_ratio": optimized_size / original_size if original_size > 0 else 1,
                "size_reduction_percent": ((original_size - optimized_size) / original_size * 100) if original_size > 0 else 0,
                "encoding_used": headers.get("Content-Encoding", "none"),
                "headers_added": headers
            },
            "content_preview": optimized_content[:200].decode('utf-8', errors='ignore') + "..." if len(optimized_content) > 200 else optimized_content.decode('utf-8', errors='ignore'),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Erreur optimisation contenu: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur optimisation contenu: {str(e)}"
        )


@router.get("/report")
async def get_optimization_report(
    current_user: User = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """
    Génère un rapport complet d'optimisation des assets
    
    Retourne analyse de performance, recommandations, et métriques
    """
    try:
        report = await get_asset_optimization_report()
        
        return {
            "success": True,
            "report": report,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Erreur génération rapport assets: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur génération rapport: {str(e)}"
        )


@router.post("/precompress")
async def start_precompression(
    request: PrecompressionRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """
    Lance la précompression des assets statiques en arrière-plan
    
    Compresse tous les fichiers éligibles d'un répertoire
    """
    try:
        import os
        
        # Vérifier que le répertoire existe
        if not os.path.exists(request.static_directory):
            raise HTTPException(
                status_code=400,
                detail=f"Répertoire non trouvé: {request.static_directory}"
            )
        
        # Lancer en arrière-plan
        background_tasks.add_task(
            precompress_static_assets,
            request.static_directory
        )
        
        logger.info(f"Précompression lancée par {current_user.username}: {request.static_directory}")
        
        return {
            "success": True,
            "message": f"Précompression lancée pour {request.static_directory}",
            "status": "started",
            "directory": request.static_directory,
            "started_by": current_user.username,
            "started_at": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur lancement précompression: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lancement précompression: {str(e)}"
        )


@router.post("/clear-cache")
async def clear_asset_cache(
    current_user: User = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """
    Vide le cache des assets optimisés
    
    Force la re-optimisation de tous les assets au prochain accès
    """
    try:
        optimizer = await get_asset_optimizer()
        
        # Sauvegarder métriques avant reset
        old_metrics = optimizer.get_metrics()
        
        # Reset du cache et métriques
        optimizer.static_cache.cache.clear()
        optimizer.reset_metrics()
        
        logger.info(f"Cache assets vidé par {current_user.username}")
        
        return {
            "success": True,
            "message": "Cache des assets vidé avec succès",
            "previous_stats": {
                "cached_items": old_metrics.get("total_requests", 0),
                "total_savings_mb": old_metrics.get("total_savings_mb", 0)
            },
            "cleared_by": current_user.username,
            "cleared_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Erreur vidage cache assets: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur vidage cache: {str(e)}"
        )


@router.post("/service-worker")
async def generate_service_worker_endpoint(
    request: ServiceWorkerRequest,
    current_user: User = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """
    Génère un Service Worker pour cache offline
    
    Crée un fichier Service Worker avec les assets spécifiés
    """
    try:
        import os
        
        # Vérifier que le répertoire de destination existe
        output_dir = os.path.dirname(request.output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        # Générer le Service Worker
        await generate_service_worker(request.output_path, request.cache_files)
        
        # Vérifier que le fichier a été créé
        if not os.path.exists(request.output_path):
            raise Exception("Échec de génération du Service Worker")
        
        file_size = os.path.getsize(request.output_path)
        
        logger.info(f"Service Worker généré par {current_user.username}: {request.output_path}")
        
        return {
            "success": True,
            "message": "Service Worker généré avec succès",
            "service_worker": {
                "path": request.output_path,
                "size_bytes": file_size,
                "cached_files_count": len(request.cache_files),
                "cached_files": request.cache_files
            },
            "generated_by": current_user.username,
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Erreur génération Service Worker: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur génération Service Worker: {str(e)}"
        )


@router.get("/config")
async def get_asset_config(
    current_user: User = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """
    Récupère la configuration actuelle d'optimisation des assets
    
    Retourne les paramètres de compression, minification, cache, etc.
    """
    try:
        optimizer = await get_asset_optimizer()
        
        config = {
            "compression": {
                "gzip_enabled": optimizer.compression_config.enable_gzip,
                "brotli_enabled": optimizer.compression_config.enable_brotli,
                "gzip_level": optimizer.compression_config.gzip_level,
                "brotli_level": optimizer.compression_config.brotli_level,
                "min_size_bytes": optimizer.compression_config.min_size_bytes,
                "max_size_bytes": optimizer.compression_config.max_size_bytes,
                "compressible_types": optimizer.compression_config.compressible_types
            },
            "optimization": {
                "minification_enabled": optimizer.optimization_config.enable_minification,
                "image_optimization_enabled": optimizer.optimization_config.enable_image_optimization,
                "caching_enabled": optimizer.optimization_config.enable_caching,
                "cache_max_age": optimizer.optimization_config.cache_max_age,
                "versioning_enabled": optimizer.optimization_config.versioning,
                "cdn_enabled": optimizer.optimization_config.cdn_enabled,
                "cdn_base_url": optimizer.optimization_config.cdn_base_url,
                "image_quality": optimizer.optimization_config.image_quality
            },
            "features_available": {
                "minification": optimizer.minifier.enabled,
                "image_optimization": optimizer.image_optimizer.enabled,
                "compression": True
            }
        }
        
        return {
            "success": True,
            "config": config,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Erreur récupération config assets: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur récupération configuration: {str(e)}"
        )


@router.get("/health")
async def get_asset_health(
    current_user: User = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """
    Vérifie la santé du système d'optimisation des assets
    
    Retourne le statut et les problèmes potentiels
    """
    try:
        optimizer = await get_asset_optimizer()
        metrics = optimizer.get_metrics()
        
        health_score = 100
        issues = []
        status = "healthy"
        
        # Vérifier les métriques
        if metrics['cache_hit_rate'] < 50:
            health_score -= 20
            issues.append("Taux de hit cache faible")
        
        if metrics['compression_rate'] < 60:
            health_score -= 15
            issues.append("Taux de compression faible")
        
        if metrics['avg_compression_time'] > 0.2:
            health_score -= 10
            issues.append("Temps de compression élevé")
        
        # Vérifier disponibilité des fonctionnalités
        if not optimizer.minifier.enabled:
            health_score -= 10
            issues.append("Minification non disponible (paquets manquants)")
        
        if not optimizer.image_optimizer.enabled:
            health_score -= 5
            issues.append("Optimisation d'images non disponible")
        
        # Déterminer statut
        if health_score >= 90:
            status = "healthy"
        elif health_score >= 70:
            status = "degraded"
        else:
            status = "unhealthy"
        
        return {
            "success": True,
            "health": {
                "status": status,
                "score": max(0, health_score),
                "issues": issues,
                "metrics_summary": {
                    "total_requests": metrics['total_requests'],
                    "cache_hit_rate": metrics['cache_hit_rate'],
                    "compression_rate": metrics['compression_rate'],
                    "total_savings_mb": metrics['total_savings_mb']
                }
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Erreur vérification santé assets: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur vérification santé: {str(e)}"
        )


@router.get("/statistics")
async def get_asset_statistics(
    hours: int = Query(24, ge=1, le=168, description="Période d'analyse en heures"),
    current_user: User = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """
    Récupère les statistiques d'optimisation sur une période
    
    Args:
        hours: Période d'analyse (1-168 heures)
    """
    try:
        optimizer = await get_asset_optimizer()
        metrics = optimizer.get_metrics()
        
        # Calculer tendances (simulation - à améliorer avec vraies données temporelles)
        trends = {
            "cache_hit_rate_trend": "stable",
            "compression_rate_trend": "increasing",
            "bandwidth_savings_trend": "increasing"
        }
        
        # Projections
        daily_requests = metrics['total_requests'] / max(1, (datetime.now() - metrics.get('last_reset', datetime.now())).days)
        projected_monthly_savings = metrics['total_savings_mb'] * 30
        
        statistics = {
            "period_hours": hours,
            "current_metrics": metrics,
            "trends": trends,
            "projections": {
                "daily_requests_estimate": daily_requests,
                "monthly_bandwidth_savings_mb": projected_monthly_savings,
                "yearly_bandwidth_savings_gb": projected_monthly_savings * 12 / 1024
            },
            "efficiency_scores": {
                "compression_efficiency": min(100, metrics['compression_rate']),
                "cache_efficiency": min(100, metrics['cache_hit_rate']),
                "overall_efficiency": (metrics['compression_rate'] + metrics['cache_hit_rate']) / 2
            }
        }
        
        return {
            "success": True,
            "statistics": statistics,
            "period": {
                "hours": hours,
                "start": (datetime.now() - timedelta(hours=hours)).isoformat(),
                "end": datetime.now().isoformat()
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Erreur statistiques assets: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur calcul statistiques: {str(e)}"
        )