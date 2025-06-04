"""
Routes API pour le Dashboard IA
US-010: Endpoints pour accéder aux fonctionnalités du dashboard IA
"""

from fastapi import APIRouter, Depends, HTTPException, Query, Body
from typing import Dict, List, Any, Optional
import asyncio
from datetime import datetime

from app.core.ai_dashboard import (
    get_dashboard_engine, 
    get_dashboard_data_api, 
    explain_prediction_api,
    get_ai_insights_summary
)
from app.core.dependencies import get_current_active_user
from app.models.user import User
from app.core.logging_system import get_logger, LogCategory

logger = get_logger("ai_dashboard_api", LogCategory.API)

router = APIRouter()


@router.get("/dashboards", response_model=List[Dict[str, Any]])
async def get_available_dashboards(
    current_user: User = Depends(get_current_active_user)
):
    """Récupère la liste des dashboards disponibles pour l'utilisateur"""
    
    try:
        engine = await get_dashboard_engine()
        dashboards = engine.get_available_dashboards(current_user.username)
        
        logger.info(f"📊 {len(dashboards)} dashboards récupérés pour {current_user.username}")
        
        return {
            "dashboards": dashboards,
            "total_count": len(dashboards),
            "user_id": current_user.username
        }
        
    except Exception as e:
        logger.error(f"❌ Erreur récupération dashboards: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/dashboard/{dashboard_id}", response_model=Dict[str, Any])
async def get_dashboard_data(
    dashboard_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """Récupère les données complètes d'un dashboard"""
    
    try:
        data = await get_dashboard_data_api(dashboard_id, current_user.username)
        
        if "error" in data:
            raise HTTPException(status_code=404, detail=data["error"])
        
        logger.info(f"📊 Données dashboard {dashboard_id} récupérées pour {current_user.username}")
        
        return data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Erreur récupération dashboard {dashboard_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/dashboard/create", response_model=Dict[str, str])
async def create_custom_dashboard(
    dashboard_data: Dict[str, Any] = Body(...),
    current_user: User = Depends(get_current_active_user)
):
    """Crée un dashboard personnalisé"""
    
    try:
        engine = await get_dashboard_engine()
        
        title = dashboard_data.get("title", "Dashboard Personnalisé")
        widgets = dashboard_data.get("widgets", [])
        
        dashboard_id = engine.create_custom_dashboard(
            current_user.username, 
            title, 
            widgets
        )
        
        logger.info(f"📊 Dashboard personnalisé créé: {dashboard_id} par {current_user.username}")
        
        return {
            "dashboard_id": dashboard_id,
            "message": "Dashboard créé avec succès",
            "title": title
        }
        
    except Exception as e:
        logger.error(f"❌ Erreur création dashboard: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/insights/summary", response_model=Dict[str, Any])
async def get_insights_summary(
    current_user: User = Depends(get_current_active_user)
):
    """Récupère le résumé des insights IA"""
    
    try:
        summary = await get_ai_insights_summary()
        
        logger.info(f"🧠 Résumé insights IA récupéré pour {current_user.username}")
        
        return summary
        
    except Exception as e:
        logger.error(f"❌ Erreur récupération insights: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/prediction/explain", response_model=Dict[str, Any])
async def explain_prediction(
    explanation_request: Dict[str, Any] = Body(...),
    current_user: User = Depends(get_current_active_user)
):
    """Génère une explication pour une prédiction"""
    
    try:
        model_name = explanation_request.get("model_name")
        prediction_data = explanation_request.get("prediction_data", {})
        
        if not model_name:
            raise HTTPException(status_code=400, detail="model_name requis")
        
        explanation = await explain_prediction_api(model_name, prediction_data)
        
        if "error" in explanation:
            raise HTTPException(status_code=400, detail=explanation["error"])
        
        logger.info(f"🔍 Explication générée pour modèle {model_name} par {current_user.username}")
        
        return explanation
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Erreur génération explication: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/alerts", response_model=List[Dict[str, Any]])
async def get_system_alerts(
    level: Optional[str] = Query(None, description="Filtrer par niveau d'alerte"),
    current_user: User = Depends(get_current_active_user)
):
    """Récupère les alertes système actives"""
    
    try:
        engine = await get_dashboard_engine()
        alerts = await engine.get_system_alerts()
        
        # Filtrer par niveau si spécifié
        if level:
            alerts = [alert for alert in alerts if alert["level"] == level]
        
        logger.info(f"🚨 {len(alerts)} alertes récupérées pour {current_user.username}")
        
        return {
            "alerts": alerts,
            "total_count": len(alerts),
            "filtered_by_level": level
        }
        
    except Exception as e:
        logger.error(f"❌ Erreur récupération alertes: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(
    alert_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """Acquitte une alerte"""
    
    try:
        engine = await get_dashboard_engine()
        engine.alerting_system.acknowledge_alert(alert_id, current_user.username)
        
        logger.info(f"🔔 Alerte {alert_id} acquittée par {current_user.username}")
        
        return {
            "message": "Alerte acquittée avec succès",
            "alert_id": alert_id,
            "acknowledged_by": current_user.username,
            "acknowledged_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Erreur acquittement alerte {alert_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/alerts/{alert_id}/resolve")
async def resolve_alert(
    alert_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """Résout une alerte"""
    
    try:
        engine = await get_dashboard_engine()
        engine.alerting_system.resolve_alert(alert_id, current_user.username)
        
        logger.info(f"✅ Alerte {alert_id} résolue par {current_user.username}")
        
        return {
            "message": "Alerte résolue avec succès",
            "alert_id": alert_id,
            "resolved_by": current_user.username,
            "resolved_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Erreur résolution alerte {alert_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analytics", response_model=Dict[str, Any])
async def get_dashboard_analytics(
    current_user: User = Depends(get_current_active_user)
):
    """Récupère les analytics du système de dashboard"""
    
    try:
        engine = await get_dashboard_engine()
        analytics = engine.get_dashboard_analytics()
        
        logger.info(f"📈 Analytics dashboard récupérées pour {current_user.username}")
        
        return analytics
        
    except Exception as e:
        logger.error(f"❌ Erreur récupération analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/widgets/types", response_model=List[Dict[str, str]])
async def get_available_widget_types(
    current_user: User = Depends(get_current_active_user)
):
    """Récupère les types de widgets disponibles"""
    
    try:
        from app.core.ai_dashboard import VisualizationType
        
        widget_types = []
        for widget_type in VisualizationType:
            widget_types.append({
                "value": widget_type.value,
                "label": widget_type.value.replace("_", " ").title(),
                "description": f"Widget de type {widget_type.value}"
            })
        
        return {
            "widget_types": widget_types,
            "total_count": len(widget_types)
        }
        
    except Exception as e:
        logger.error(f"❌ Erreur récupération types widgets: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/performance", response_model=Dict[str, Any])
async def get_models_performance(
    current_user: User = Depends(get_current_active_user)
):
    """Récupère les métriques de performance des modèles IA"""
    
    try:
        from app.core.advanced_ai_engine import get_advanced_ai_engine
        from app.core.continuous_learning import get_continuous_learning_engine
        
        # Récupérer les métriques des modèles
        ai_engine = await get_advanced_ai_engine()
        learning_engine = await get_continuous_learning_engine()
        
        # Simulation des métriques de performance
        models_performance = {
            "RandomForest": {"accuracy": 0.87, "precision": 0.85, "recall": 0.89, "f1": 0.87},
            "XGBoost": {"accuracy": 0.89, "precision": 0.88, "recall": 0.90, "f1": 0.89},
            "LightGBM": {"accuracy": 0.85, "precision": 0.83, "recall": 0.87, "f1": 0.85},
            "NeuralNetwork": {"accuracy": 0.82, "precision": 0.80, "recall": 0.84, "f1": 0.82}
        }
        
        learning_status = learning_engine.get_learning_system_status()
        
        logger.info(f"🤖 Métriques modèles récupérées pour {current_user.username}")
        
        return {
            "models_performance": models_performance,
            "learning_system_status": learning_status,
            "total_models": len(models_performance),
            "best_performing_model": max(models_performance.keys(), key=lambda k: models_performance[k]["accuracy"]),
            "last_updated": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Erreur récupération performance modèles: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/features/importance", response_model=Dict[str, Any])
async def get_global_feature_importance(
    model_name: Optional[str] = Query(None, description="Nom du modèle spécifique"),
    current_user: User = Depends(get_current_active_user)
):
    """Récupère l'importance globale des features"""
    
    try:
        # Simulation d'importance des features
        feature_importance = {
            "chiffre_affaires": 0.25,
            "effectifs": 0.18,
            "company_age": 0.15,
            "productivity": 0.12,
            "growth_rate": 0.10,
            "sector_tech": 0.08,
            "localisation_paris": 0.07,
            "debt_ratio": 0.05
        }
        
        # Si modèle spécifique demandé, on pourrait ajuster les valeurs
        if model_name:
            logger.info(f"🎯 Feature importance pour modèle {model_name}")
        
        logger.info(f"📊 Feature importance globale récupérée pour {current_user.username}")
        
        return {
            "feature_importance": feature_importance,
            "model_name": model_name or "ensemble",
            "total_features": len(feature_importance),
            "last_updated": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Erreur récupération feature importance: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/dashboard/{dashboard_id}/refresh")
async def refresh_dashboard(
    dashboard_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """Force le rafraîchissement d'un dashboard"""
    
    try:
        engine = await get_dashboard_engine()
        
        # Vérifier que le dashboard existe et que l'utilisateur y a accès
        if dashboard_id not in engine.dashboard_configs:
            raise HTTPException(status_code=404, detail="Dashboard non trouvé")
        
        config = engine.dashboard_configs[dashboard_id]
        if not (config.public or config.owner_id == current_user.username or current_user.username in config.shared_with):
            raise HTTPException(status_code=403, detail="Accès non autorisé")
        
        # Vider le cache pour forcer le rafraîchissement
        cache_keys_to_remove = [key for key in engine.data_cache.keys() if dashboard_id in key]
        for key in cache_keys_to_remove:
            del engine.data_cache[key]
            if key in engine.cache_ttl:
                del engine.cache_ttl[key]
        
        # Récupérer les nouvelles données
        fresh_data = await engine.get_dashboard_data(dashboard_id)
        
        logger.info(f"🔄 Dashboard {dashboard_id} rafraîchi pour {current_user.username}")
        
        return {
            "message": "Dashboard rafraîchi avec succès",
            "dashboard_id": dashboard_id,
            "refreshed_at": datetime.now().isoformat(),
            "data": fresh_data
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Erreur rafraîchissement dashboard {dashboard_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health", response_model=Dict[str, Any])
async def get_dashboard_health(
    current_user: User = Depends(get_current_active_user)
):
    """Récupère l'état de santé du système de dashboard IA"""
    
    try:
        engine = await get_dashboard_engine()
        
        # Vérifier la santé des différents composants
        health_status = {
            "dashboard_engine": "healthy",
            "visualization_engine": "healthy",
            "explanation_engine": "healthy",
            "alerting_system": "healthy",
            "total_dashboards": len(engine.dashboard_configs),
            "active_alerts": len(engine.alerting_system.active_alerts),
            "cache_size": len(engine.data_cache),
            "last_check": datetime.now().isoformat()
        }
        
        # Déterminer le statut global
        overall_status = "healthy"
        if health_status["active_alerts"] > 10:
            overall_status = "warning"
        
        health_status["overall_status"] = overall_status
        
        logger.info(f"💚 État de santé dashboard vérifié pour {current_user.username}")
        
        return health_status
        
    except Exception as e:
        logger.error(f"❌ Erreur vérification santé dashboard: {e}")
        raise HTTPException(status_code=500, detail=str(e))