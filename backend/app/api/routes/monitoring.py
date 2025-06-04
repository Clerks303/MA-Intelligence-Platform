"""
API Routes pour le dashboard de monitoring temps réel
US-006: Endpoints pour métriques, alertes et observabilité

Features:
- Métriques système et business en temps réel
- Dashboard interactif avec graphiques
- Alertes actives et historique
- SLA monitoring et compliance
- Métriques Prometheus export
- Health checks avancés
"""

import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from fastapi.responses import PlainTextResponse, JSONResponse
from pydantic import BaseModel, Field
from enum import Enum

from app.core.dependencies import get_current_active_user
from app.models.user import User
from app.core.advanced_monitoring import get_advanced_monitoring, AdvancedMonitoring
from app.core.intelligent_alerting import (
    get_intelligent_alerting, IntelligentAlertingSystem,
    AlertType, SeverityLevel, AlertStatus, EscalationLevel
)
from app.core.performance_monitor import get_performance_monitor
from app.core.cache_manager import get_cache_manager
from app.core.database_optimizer import get_database_optimizer
from app.core.logging_system import get_logger, LogCategory

logger = get_logger("monitoring_api", LogCategory.API)

router = APIRouter(prefix="/monitoring", tags=["monitoring"])


# Modèles Pydantic pour les réponses

class SystemMetricsResponse(BaseModel):
    """Métriques système temps réel"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    network_bytes_sent_mb: float
    network_bytes_recv_mb: float
    process_memory_mb: float
    process_cpu_percent: float
    active_connections: int


class BusinessMetricsResponse(BaseModel):
    """Métriques business"""
    timestamp: datetime
    companies_total: int
    companies_scraped_24h: int
    scraping_success_rate: float
    prospects_identified_24h: int
    conversion_rate: float
    avg_processing_time_ms: float
    data_quality_score: float


class AlertSummaryResponse(BaseModel):
    """Résumé des alertes"""
    total_active: int
    critical: int
    high: int
    medium: int
    low: int
    acknowledged: int
    escalated: int
    mttr_minutes: float


class AlertResponse(BaseModel):
    """Détails d'une alerte"""
    id: str
    title: str
    description: str
    alert_type: str
    severity: str
    status: str
    current_value: float
    threshold_value: float
    first_triggered: datetime
    last_triggered: datetime
    escalation_level: str
    acknowledgments: List[Dict[str, Any]]
    tags: List[str]


class SLAStatusResponse(BaseModel):
    """Statut SLA"""
    name: str
    current_value: float
    target_value: float
    status: str  # compliant, violated
    compliance_percentage: float
    violations_24h: int
    last_check: datetime


class PerformanceSnapshot(BaseModel):
    """Snapshot de performance"""
    timestamp: datetime
    api_response_time_p95: float
    api_response_time_p99: float
    throughput_rpm: float
    error_rate_percent: float
    cache_hit_ratio: float
    db_connection_pool_usage: float
    active_tasks: int


class HealthCheckResponse(BaseModel):
    """Health check complet"""
    status: str  # healthy, degraded, unhealthy
    timestamp: datetime
    services: Dict[str, Dict[str, Any]]
    overall_score: float
    issues: List[str]


# Endpoints du dashboard

@router.get("/dashboard", response_model=Dict[str, Any])
async def get_dashboard_overview(
    current_user: User = Depends(get_current_active_user)
):
    """Vue d'ensemble complète du dashboard"""
    try:
        monitoring = await get_advanced_monitoring()
        alerting = await get_intelligent_alerting()
        
        # Collecter toutes les données en parallèle
        dashboard_data = monitoring.get_monitoring_dashboard_data()
        alert_stats = alerting.get_alert_statistics()
        
        # Métriques système récentes
        perf_monitor = await get_performance_monitor()
        current_metrics = perf_monitor.get_current_metrics()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "system_health": dashboard_data.get("system_health", {}),
            "sla_status": dashboard_data.get("sla_status", {}),
            "alert_summary": alert_stats,
            "recent_anomalies": dashboard_data.get("recent_anomalies", []),
            "current_metrics": current_metrics,
            "services_status": await _get_services_status()
        }
        
    except Exception as e:
        logger.error(f"Erreur dashboard overview: {e}")
        raise HTTPException(status_code=500, detail="Erreur interne du serveur")


@router.get("/metrics/system", response_model=SystemMetricsResponse)
async def get_system_metrics(
    current_user: User = Depends(get_current_active_user)
):
    """Métriques système en temps réel"""
    try:
        perf_monitor = await get_performance_monitor()
        metrics = perf_monitor._collect_system_metrics()
        
        # Pool de connexions DB
        db_optimizer = await get_database_optimizer()
        pool_stats = db_optimizer.get_performance_stats().get('connection_pool', {})
        active_connections = pool_stats.get('size', 0)
        
        return SystemMetricsResponse(
            timestamp=datetime.now(),
            cpu_percent=metrics.cpu_percent,
            memory_percent=metrics.memory_percent,
            disk_percent=metrics.disk_percent,
            network_bytes_sent_mb=metrics.network_bytes_sent / 1024 / 1024,
            network_bytes_recv_mb=metrics.network_bytes_recv / 1024 / 1024,
            process_memory_mb=metrics.process_memory_rss / 1024 / 1024,
            process_cpu_percent=metrics.process_cpu_percent,
            active_connections=active_connections
        )
        
    except Exception as e:
        logger.error(f"Erreur métriques système: {e}")
        raise HTTPException(status_code=500, detail="Erreur collecte métriques")


@router.get("/metrics/business", response_model=BusinessMetricsResponse)
async def get_business_metrics(
    current_user: User = Depends(get_current_active_user)
):
    """Métriques business en temps réel"""
    try:
        # Simuler les métriques business (à connecter aux vraies données)
        # En production, récupérer depuis la base de données
        
        from app.core.cache_manager import cached
        
        @cached('business_metrics', ttl_seconds=300)  # Cache 5 minutes
        async def _get_cached_business_metrics():
            # Ici récupérer les vraies données business
            return {
                "companies_total": 15420,
                "companies_scraped_24h": 342,
                "scraping_success_rate": 94.2,
                "prospects_identified_24h": 28,
                "conversion_rate": 12.5,
                "avg_processing_time_ms": 1250.0,
                "data_quality_score": 87.3
            }
        
        metrics_data = await _get_cached_business_metrics()
        
        return BusinessMetricsResponse(
            timestamp=datetime.now(),
            **metrics_data
        )
        
    except Exception as e:
        logger.error(f"Erreur métriques business: {e}")
        raise HTTPException(status_code=500, detail="Erreur métriques business")


@router.get("/alerts", response_model=List[AlertResponse])
async def get_active_alerts(
    alert_type: Optional[str] = Query(None, description="Filtrer par type d'alerte"),
    severity: Optional[str] = Query(None, description="Filtrer par sévérité"),
    limit: int = Query(50, ge=1, le=500, description="Nombre maximum d'alertes"),
    current_user: User = Depends(get_current_active_user)
):
    """Liste des alertes actives avec filtres"""
    try:
        alerting = await get_intelligent_alerting()
        
        # Filtres
        filters = {}
        if alert_type:
            filters['alert_type'] = AlertType(alert_type)
        if severity:
            filters['severity'] = SeverityLevel(severity)
        
        alerts = alerting.get_active_alerts(filters)
        
        # Limiter et convertir
        alerts = alerts[:limit]
        
        return [
            AlertResponse(
                id=alert.id,
                title=alert.title,
                description=alert.description,
                alert_type=alert.alert_type.value,
                severity=alert.severity.value,
                status=alert.status.value,
                current_value=alert.current_value,
                threshold_value=alert.threshold_value,
                first_triggered=alert.first_triggered,
                last_triggered=alert.last_triggered,
                escalation_level=alert.escalation_level.value,
                acknowledgments=alert.acknowledgments,
                tags=list(alert.tags)
            )
            for alert in alerts
        ]
        
    except Exception as e:
        logger.error(f"Erreur récupération alertes: {e}")
        raise HTTPException(status_code=500, detail="Erreur récupération alertes")


@router.get("/alerts/summary", response_model=AlertSummaryResponse)
async def get_alerts_summary(
    current_user: User = Depends(get_current_active_user)
):
    """Résumé des alertes pour dashboard"""
    try:
        alerting = await get_intelligent_alerting()
        stats = alerting.get_alert_statistics()
        
        # Compter par sévérité
        active_alerts = alerting.get_active_alerts()
        severity_counts = {
            'critical': 0,
            'high': 0,
            'medium': 0,
            'low': 0
        }
        
        acknowledged_count = 0
        escalated_count = 0
        
        for alert in active_alerts:
            severity_counts[alert.severity.value] += 1
            
            if alert.status == AlertStatus.ACKNOWLEDGED:
                acknowledged_count += 1
            elif alert.status == AlertStatus.ESCALATED:
                escalated_count += 1
        
        return AlertSummaryResponse(
            total_active=stats['active_alerts'],
            critical=severity_counts['critical'],
            high=severity_counts['high'],
            medium=severity_counts['medium'],
            low=severity_counts['low'],
            acknowledged=acknowledged_count,
            escalated=escalated_count,
            mttr_minutes=stats['mttr_minutes']
        )
        
    except Exception as e:
        logger.error(f"Erreur résumé alertes: {e}")
        raise HTTPException(status_code=500, detail="Erreur résumé alertes")


@router.post("/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(
    alert_id: str,
    comment: str = "",
    current_user: User = Depends(get_current_active_user)
):
    """Acquitte une alerte"""
    try:
        alerting = await get_intelligent_alerting()
        
        success = await alerting.acknowledge_alert(
            alert_id, 
            current_user.username, 
            comment
        )
        
        if not success:
            raise HTTPException(status_code=404, detail="Alerte non trouvée")
        
        return {"message": "Alerte acquittée avec succès", "alert_id": alert_id}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur acquittement alerte {alert_id}: {e}")
        raise HTTPException(status_code=500, detail="Erreur acquittement alerte")


@router.post("/alerts/{alert_id}/resolve")
async def resolve_alert(
    alert_id: str,
    resolution_note: str = "",
    current_user: User = Depends(get_current_active_user)
):
    """Résout une alerte"""
    try:
        alerting = await get_intelligent_alerting()
        
        success = await alerting.resolve_alert(
            alert_id,
            current_user.username,
            resolution_note
        )
        
        if not success:
            raise HTTPException(status_code=404, detail="Alerte non trouvée")
        
        return {"message": "Alerte résolue avec succès", "alert_id": alert_id}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur résolution alerte {alert_id}: {e}")
        raise HTTPException(status_code=500, detail="Erreur résolution alerte")


@router.get("/sla", response_model=List[SLAStatusResponse])
async def get_sla_status(
    current_user: User = Depends(get_current_active_user)
):
    """Statut des SLA"""
    try:
        monitoring = await get_advanced_monitoring()
        sla_data = monitoring.sla_status
        
        response = []
        for sla_name, sla_info in sla_data.items():
            
            # Calculer compliance 24h
            violations_24h = len([
                v for v in sla_info.get('violations', [])
                if datetime.fromisoformat(v['timestamp']) > datetime.now() - timedelta(hours=24)
            ])
            
            # Compliance approximative (à améliorer avec vraies données)
            compliance_percentage = max(0, 100 - (violations_24h * 5))
            
            response.append(SLAStatusResponse(
                name=sla_name,
                current_value=sla_info['current_value'],
                target_value=sla_info['target_value'],
                status=sla_info['status'],
                compliance_percentage=compliance_percentage,
                violations_24h=violations_24h,
                last_check=sla_info['last_check']
            ))
        
        return response
        
    except Exception as e:
        logger.error(f"Erreur statut SLA: {e}")
        raise HTTPException(status_code=500, detail="Erreur statut SLA")


@router.get("/performance", response_model=PerformanceSnapshot)
async def get_performance_snapshot(
    current_user: User = Depends(get_current_active_user)
):
    """Snapshot de performance actuel"""
    try:
        # Performance monitor
        perf_monitor = await get_performance_monitor()
        function_profiles = perf_monitor.get_function_profiles(top_n=10)
        
        # Cache stats
        cache_manager = await get_cache_manager()
        cache_stats = cache_manager.get_stats()
        
        # DB stats
        db_optimizer = await get_database_optimizer()
        db_stats = db_optimizer.get_performance_stats()
        
        # Calculer métriques agrégées
        avg_response_time = 0
        if function_profiles:
            avg_response_time = sum(p['avg_time_ms'] for p in function_profiles) / len(function_profiles)
        
        cache_hit_ratio = cache_stats.get('total', {}).get('hit_ratio', 0.0)
        
        return PerformanceSnapshot(
            timestamp=datetime.now(),
            api_response_time_p95=avg_response_time * 1.5,  # Approximation P95
            api_response_time_p99=avg_response_time * 2.0,  # Approximation P99
            throughput_rpm=120.0,  # À calculer depuis métriques réelles
            error_rate_percent=2.1,  # À calculer depuis logs
            cache_hit_ratio=cache_hit_ratio * 100,
            db_connection_pool_usage=75.0,  # À récupérer du pool
            active_tasks=len(function_profiles)
        )
        
    except Exception as e:
        logger.error(f"Erreur snapshot performance: {e}")
        raise HTTPException(status_code=500, detail="Erreur snapshot performance")


@router.get("/health", response_model=HealthCheckResponse)
async def get_health_check(
    detailed: bool = Query(False, description="Health check détaillé"),
    current_user: User = Depends(get_current_active_user)
):
    """Health check complet du système"""
    try:
        services_status = await _get_services_status()
        
        # Calculer score global
        service_scores = [s.get('score', 0) for s in services_status.values()]
        overall_score = sum(service_scores) / len(service_scores) if service_scores else 0
        
        # Déterminer statut global
        if overall_score >= 90:
            status = "healthy"
        elif overall_score >= 70:
            status = "degraded"
        else:
            status = "unhealthy"
        
        # Identifier problèmes
        issues = []
        for service_name, service_info in services_status.items():
            if service_info.get('score', 0) < 80:
                issues.append(f"{service_name}: {service_info.get('message', 'Problème détecté')}")
        
        response = HealthCheckResponse(
            status=status,
            timestamp=datetime.now(),
            services=services_status if detailed else {},
            overall_score=overall_score,
            issues=issues
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Erreur health check: {e}")
        raise HTTPException(status_code=500, detail="Erreur health check")


@router.get("/metrics/history")
async def get_metrics_history(
    metric_name: str = Query(..., description="Nom de la métrique"),
    hours: int = Query(24, ge=1, le=168, description="Nombre d'heures d'historique"),
    current_user: User = Depends(get_current_active_user)
):
    """Historique d'une métrique pour graphiques"""
    try:
        perf_monitor = await get_performance_monitor()
        
        # Récupérer historique
        history = perf_monitor.get_metrics_history(hours=hours)
        
        # Filtrer par métrique demandée
        metric_history = []
        for metrics in history:
            if metric_name in metrics:
                metric_history.append({
                    'timestamp': metrics['timestamp'],
                    'value': metrics.get(metric_name, 0)
                })
        
        return {
            'metric_name': metric_name,
            'timerange_hours': hours,
            'data_points': len(metric_history),
            'history': metric_history
        }
        
    except Exception as e:
        logger.error(f"Erreur historique métrique {metric_name}: {e}")
        raise HTTPException(status_code=500, detail="Erreur historique métrique")


@router.get("/prometheus", response_class=PlainTextResponse)
async def get_prometheus_metrics():
    """Export des métriques au format Prometheus"""
    try:
        monitoring = await get_advanced_monitoring()
        metrics_text = monitoring.prometheus.get_metrics_text()
        
        return PlainTextResponse(
            content=metrics_text,
            media_type="text/plain; version=0.0.4"
        )
        
    except Exception as e:
        logger.error(f"Erreur export Prometheus: {e}")
        raise HTTPException(status_code=500, detail="Erreur export métriques")


@router.post("/test-alert")
async def create_test_alert(
    title: str = "Test Alert",
    severity: str = "medium",
    current_user: User = Depends(get_current_active_user)
):
    """Crée une alerte de test (développement)"""
    try:
        from app.core.intelligent_alerting import create_custom_alert, AlertType, SeverityLevel
        
        alert = await create_custom_alert(
            title=f"[TEST] {title}",
            description=f"Alerte de test créée par {current_user.username}",
            alert_type=AlertType.SYSTEM,
            severity=SeverityLevel(severity),
            current_value=100.0,
            threshold_value=80.0,
            metadata={'test': True, 'created_by': current_user.username}
        )
        
        return {
            "message": "Alerte de test créée",
            "alert_id": alert.id,
            "alert": alert.to_dict()
        }
        
    except Exception as e:
        logger.error(f"Erreur création alerte test: {e}")
        raise HTTPException(status_code=500, detail="Erreur création alerte test")


# Fonctions utilitaires

async def _get_services_status() -> Dict[str, Dict[str, Any]]:
    """Vérifie le statut de tous les services"""
    
    services = {}
    
    # Cache Redis
    try:
        cache_manager = await get_cache_manager()
        cache_stats = cache_manager.get_stats()
        
        redis_connected = cache_stats.get('redis_cache', {}).get('connected', False)
        
        services['redis'] = {
            'status': 'healthy' if redis_connected else 'unhealthy',
            'score': 100 if redis_connected else 0,
            'message': 'Redis opérationnel' if redis_connected else 'Redis non disponible',
            'details': cache_stats
        }
    except Exception as e:
        services['redis'] = {
            'status': 'unhealthy',
            'score': 0,
            'message': f'Erreur Redis: {str(e)[:100]}',
            'details': {}
        }
    
    # Base de données
    try:
        db_optimizer = await get_database_optimizer()
        db_stats = db_optimizer.get_performance_stats()
        
        # Score basé sur temps de réponse
        avg_query_time = db_stats.get('query_performance', {}).get('avg_query_time_ms', 0)
        db_score = max(0, 100 - (avg_query_time / 10))  # Pénalité si > 1000ms
        
        services['database'] = {
            'status': 'healthy' if db_score > 70 else 'degraded' if db_score > 30 else 'unhealthy',
            'score': db_score,
            'message': f'DB performance: {avg_query_time:.1f}ms moyenne',
            'details': db_stats
        }
    except Exception as e:
        services['database'] = {
            'status': 'unhealthy',
            'score': 0,
            'message': f'Erreur DB: {str(e)[:100]}',
            'details': {}
        }
    
    # Monitoring system
    try:
        monitoring = await get_advanced_monitoring()
        
        services['monitoring'] = {
            'status': 'healthy',
            'score': 100,
            'message': 'Système de monitoring opérationnel',
            'details': {'prometheus_available': True, 'tracing_available': True}
        }
    except Exception as e:
        services['monitoring'] = {
            'status': 'unhealthy',
            'score': 0,
            'message': f'Erreur monitoring: {str(e)[:100]}',
            'details': {}
        }
    
    return services


# WebSocket pour temps réel (optionnel)
@router.websocket("/ws/metrics")
async def websocket_metrics_stream():
    """WebSocket pour stream de métriques temps réel"""
    # TODO: Implémenter WebSocket pour dashboard temps réel
    pass