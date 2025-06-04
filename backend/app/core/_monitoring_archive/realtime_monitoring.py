"""
Système de monitoring performance temps réel pour M&A Intelligence Platform
US-009: Monitoring et alertes en temps réel avec métriques détaillées

Ce module fournit:
- Monitoring en temps réel des métriques système
- Alertes automatiques sur seuils critiques  
- Dashboard de métriques en live
- Collecte de métriques applicatives
- Intégration Prometheus/Grafana
- Notification Slack/Email automatique
"""

import asyncio
import time
import json
import psutil
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Set
from dataclasses import dataclass, field
from functools import wraps
from enum import Enum
from collections import defaultdict, deque
import statistics
import logging
from concurrent.futures import ThreadPoolExecutor

# Monitoring libraries
try:
    from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, generate_latest
    from prometheus_client.exposition import MetricsHandler
    from prometheus_client.core import REGISTRY
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logging.warning("Prometheus client non disponible")

import numpy as np
import redis
import aiohttp

from app.core.logging_system import get_logger, LogCategory
from app.core.cache_manager import get_cache_manager
from app.core.performance_analyzer import get_performance_analyzer

logger = get_logger("realtime_monitoring", LogCategory.PERFORMANCE)


class AlertSeverity(str, Enum):
    """Niveaux de sévérité des alertes"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class MetricType(str, Enum):
    """Types de métriques"""
    COUNTER = "counter"         # Toujours croissant
    GAUGE = "gauge"            # Valeur instantanée
    HISTOGRAM = "histogram"     # Distribution de valeurs
    RATE = "rate"              # Taux par seconde


@dataclass
class MetricDefinition:
    """Définition d'une métrique"""
    name: str
    metric_type: MetricType
    description: str
    unit: str = ""
    labels: List[str] = field(default_factory=list)
    help_text: str = ""
    

@dataclass
class AlertRule:
    """Règle d'alerte"""
    name: str
    metric_name: str
    condition: str  # ">=", "<=", ">", "<", "=="
    threshold: float
    severity: AlertSeverity
    duration_seconds: int = 60  # Durée avant déclenchement
    cooldown_seconds: int = 300  # Pause entre alertes
    message_template: str = ""
    enabled: bool = True
    last_triggered: Optional[datetime] = None


@dataclass
class Alert:
    """Alerte déclenchée"""
    rule_name: str
    severity: AlertSeverity
    message: str
    metric_value: float
    threshold: float
    triggered_at: datetime
    resolved_at: Optional[datetime] = None
    acknowledged: bool = False
    

@dataclass
class SystemMetrics:
    """Métriques système en temps réel"""
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    memory_available_gb: float
    disk_usage: float
    disk_free_gb: float
    network_bytes_sent: int
    network_bytes_recv: int
    active_connections: int
    load_average: Tuple[float, float, float]
    process_count: int


@dataclass
class ApplicationMetrics:
    """Métriques applicatives en temps réel"""
    timestamp: datetime
    http_requests_total: int
    http_request_duration_avg: float
    http_errors_5xx: int
    http_errors_4xx: int
    database_connections_active: int
    database_query_duration_avg: float
    cache_hit_rate: float
    background_jobs_pending: int
    background_jobs_running: int
    active_users: int


class RealtimeMonitor:
    """Système de monitoring temps réel principal"""
    
    def __init__(self):
        self.metrics_registry: Dict[str, MetricDefinition] = {}
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.metrics_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.monitoring_active = False
        self.collection_interval = 5  # secondes
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Prometheus metrics si disponible
        if PROMETHEUS_AVAILABLE:
            self.prometheus_registry = CollectorRegistry()
            self.prometheus_metrics: Dict[str, Any] = {}
            self._setup_prometheus_metrics()
        
        # Notification handlers
        self.notification_handlers: List[Callable] = []
        
        # WebSocket connections pour dashboard temps réel
        self.websocket_connections: Set[Any] = set()
        
        # Métriques système et applicatives
        self.last_system_metrics: Optional[SystemMetrics] = None
        self.last_app_metrics: Optional[ApplicationMetrics] = None
        
        # Setup métriques par défaut
        self._setup_default_metrics()
        self._setup_default_alerts()
        
        logger.info("📊 Système de monitoring temps réel initialisé")
    
    def _setup_default_metrics(self):
        """Configure les métriques par défaut"""
        
        default_metrics = [
            # Métriques système
            MetricDefinition("cpu_usage_percent", MetricType.GAUGE, "Utilisation CPU", "%"),
            MetricDefinition("memory_usage_percent", MetricType.GAUGE, "Utilisation mémoire", "%"),
            MetricDefinition("disk_usage_percent", MetricType.GAUGE, "Utilisation disque", "%"),
            MetricDefinition("network_io_bytes", MetricType.COUNTER, "I/O réseau", "bytes", ["direction"]),
            MetricDefinition("load_average", MetricType.GAUGE, "Charge système", "", ["timeframe"]),
            
            # Métriques applicatives
            MetricDefinition("http_requests_total", MetricType.COUNTER, "Requêtes HTTP totales", "", ["method", "status"]),
            MetricDefinition("http_request_duration", MetricType.HISTOGRAM, "Durée requêtes HTTP", "seconds"),
            MetricDefinition("database_connections", MetricType.GAUGE, "Connexions DB actives", ""),
            MetricDefinition("database_query_duration", MetricType.HISTOGRAM, "Durée requêtes DB", "seconds"),
            MetricDefinition("cache_hit_rate", MetricType.GAUGE, "Taux de hit cache", "%"),
            MetricDefinition("background_jobs", MetricType.GAUGE, "Jobs arrière-plan", "", ["status"]),
            MetricDefinition("active_users", MetricType.GAUGE, "Utilisateurs actifs", ""),
            
            # Métriques business
            MetricDefinition("companies_scraped_total", MetricType.COUNTER, "Entreprises scrapées", ""),
            MetricDefinition("ai_scoring_requests", MetricType.COUNTER, "Requêtes scoring IA", ""),
            MetricDefinition("reports_generated", MetricType.COUNTER, "Rapports générés", ""),
            MetricDefinition("api_rate_limit_hits", MetricType.COUNTER, "Rate limit atteint", "", ["endpoint"]),
        ]
        
        for metric in default_metrics:
            self.metrics_registry[metric.name] = metric
    
    def _setup_default_alerts(self):
        """Configure les alertes par défaut"""
        
        default_alerts = [
            AlertRule(
                name="high_cpu_usage",
                metric_name="cpu_usage_percent",
                condition=">=",
                threshold=80.0,
                severity=AlertSeverity.WARNING,
                duration_seconds=120,
                message_template="Utilisation CPU élevée: {value:.1f}% (seuil: {threshold}%)"
            ),
            AlertRule(
                name="critical_cpu_usage", 
                metric_name="cpu_usage_percent",
                condition=">=",
                threshold=95.0,
                severity=AlertSeverity.CRITICAL,
                duration_seconds=60,
                message_template="⚠️ CRITIQUE: CPU saturé à {value:.1f}%"
            ),
            AlertRule(
                name="high_memory_usage",
                metric_name="memory_usage_percent",
                condition=">=",
                threshold=85.0,
                severity=AlertSeverity.WARNING,
                duration_seconds=180,
                message_template="Utilisation mémoire élevée: {value:.1f}% (seuil: {threshold}%)"
            ),
            AlertRule(
                name="low_disk_space",
                metric_name="disk_usage_percent", 
                condition=">=",
                threshold=90.0,
                severity=AlertSeverity.ERROR,
                duration_seconds=60,
                message_template="Espace disque faible: {value:.1f}% utilisé"
            ),
            AlertRule(
                name="high_error_rate",
                metric_name="http_errors_5xx",
                condition=">=",
                threshold=10.0,
                severity=AlertSeverity.ERROR,
                duration_seconds=300,
                message_template="Taux d'erreur HTTP 5xx élevé: {value} erreurs"
            ),
            AlertRule(
                name="slow_response_time",
                metric_name="http_request_duration_avg",
                condition=">=",
                threshold=2.0,
                severity=AlertSeverity.WARNING,
                duration_seconds=300,
                message_template="Temps de réponse lent: {value:.2f}s (seuil: {threshold}s)"
            ),
            AlertRule(
                name="database_connection_exhaustion",
                metric_name="database_connections_active",
                condition=">=",
                threshold=18.0,  # 90% of max 20 connections
                severity=AlertSeverity.ERROR,
                duration_seconds=60,
                message_template="Pool de connexions DB saturé: {value} connexions actives"
            ),
            AlertRule(
                name="low_cache_hit_rate",
                metric_name="cache_hit_rate",
                condition="<=",
                threshold=70.0,
                severity=AlertSeverity.WARNING,
                duration_seconds=600,
                message_template="Taux de hit cache faible: {value:.1f}% (seuil: {threshold}%)"
            )
        ]
        
        for alert in default_alerts:
            self.alert_rules[alert.name] = alert
    
    def _setup_prometheus_metrics(self):
        """Configure les métriques Prometheus"""
        if not PROMETHEUS_AVAILABLE:
            return
        
        # Métriques système
        self.prometheus_metrics.update({
            'cpu_usage': Gauge('cpu_usage_percent', 'CPU usage percentage', registry=self.prometheus_registry),
            'memory_usage': Gauge('memory_usage_percent', 'Memory usage percentage', registry=self.prometheus_registry),
            'disk_usage': Gauge('disk_usage_percent', 'Disk usage percentage', registry=self.prometheus_registry),
            'network_io': Counter('network_io_bytes_total', 'Network I/O bytes', ['direction'], registry=self.prometheus_registry),
            
            # Métriques HTTP
            'http_requests': Counter('http_requests_total', 'HTTP requests', ['method', 'status'], registry=self.prometheus_registry),
            'http_duration': Histogram('http_request_duration_seconds', 'HTTP request duration', registry=self.prometheus_registry),
            
            # Métriques DB
            'db_connections': Gauge('database_connections_active', 'Active database connections', registry=self.prometheus_registry),
            'db_query_duration': Histogram('database_query_duration_seconds', 'Database query duration', registry=self.prometheus_registry),
            
            # Métriques cache
            'cache_hit_rate': Gauge('cache_hit_rate_percent', 'Cache hit rate', registry=self.prometheus_registry),
            
            # Métriques jobs
            'background_jobs': Gauge('background_jobs_count', 'Background jobs', ['status'], registry=self.prometheus_registry),
        })
    
    async def start_monitoring(self):
        """Démarre le monitoring en temps réel"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        logger.info("🚀 Démarrage monitoring temps réel")
        
        # Démarrer collecte métrique en parallèle
        asyncio.create_task(self._metrics_collection_loop())
        asyncio.create_task(self._alert_evaluation_loop())
        asyncio.create_task(self._websocket_broadcast_loop())
        
        # Cleanup périodique
        asyncio.create_task(self._cleanup_old_data_loop())
    
    def stop_monitoring(self):
        """Arrête le monitoring"""
        self.monitoring_active = False
        self.executor.shutdown(wait=False)
        logger.info("⏹️ Monitoring arrêté")
    
    async def _metrics_collection_loop(self):
        """Boucle de collecte des métriques"""
        while self.monitoring_active:
            try:
                start_time = time.time()
                
                # Collecter métriques système et applicatives
                system_metrics = await self._collect_system_metrics()
                app_metrics = await self._collect_application_metrics()
                
                # Stocker dans l'historique
                await self._store_metrics(system_metrics, app_metrics)
                
                # Mettre à jour Prometheus si disponible
                if PROMETHEUS_AVAILABLE:
                    self._update_prometheus_metrics(system_metrics, app_metrics)
                
                # Calculer temps de collection
                collection_time = time.time() - start_time
                if collection_time > 1.0:
                    logger.warning(f"Collecte métrique lente: {collection_time:.2f}s")
                
                # Attendre prochain cycle
                await asyncio.sleep(max(0, self.collection_interval - collection_time))
                
            except Exception as e:
                logger.error(f"Erreur collecte métriques: {e}")
                await asyncio.sleep(self.collection_interval)
    
    async def _alert_evaluation_loop(self):
        """Boucle d'évaluation des alertes"""
        while self.monitoring_active:
            try:
                await self._evaluate_alerts()
                await asyncio.sleep(30)  # Évaluer toutes les 30 secondes
                
            except Exception as e:
                logger.error(f"Erreur évaluation alertes: {e}")
                await asyncio.sleep(30)
    
    async def _websocket_broadcast_loop(self):
        """Diffusion temps réel via WebSocket"""
        while self.monitoring_active:
            try:
                if self.websocket_connections:
                    # Préparer données pour dashboard
                    dashboard_data = await self._prepare_dashboard_data()
                    
                    # Diffuser à toutes les connexions
                    await self._broadcast_to_websockets(dashboard_data)
                
                await asyncio.sleep(2)  # Diffusion toutes les 2 secondes
                
            except Exception as e:
                logger.error(f"Erreur diffusion WebSocket: {e}")
                await asyncio.sleep(2)
    
    async def _cleanup_old_data_loop(self):
        """Nettoyage périodique des anciennes données"""
        while self.monitoring_active:
            try:
                # Nettoyer toutes les 10 minutes
                await asyncio.sleep(600)
                
                # Supprimer alertes résolues anciennes
                cutoff_time = datetime.now() - timedelta(hours=24)
                alerts_to_remove = [
                    alert_id for alert_id, alert in self.active_alerts.items()
                    if alert.resolved_at and alert.resolved_at < cutoff_time
                ]
                
                for alert_id in alerts_to_remove:
                    del self.active_alerts[alert_id]
                
                # Limiter taille historique métriques
                for metric_name, history in self.metrics_history.items():
                    if len(history) > 10000:  # Garder max 10k points
                        while len(history) > 5000:
                            history.popleft()
                
                logger.debug(f"Nettoyage: {len(alerts_to_remove)} alertes supprimées")
                
            except Exception as e:
                logger.error(f"Erreur nettoyage: {e}")
    
    async def _collect_system_metrics(self) -> SystemMetrics:
        """Collecte les métriques système"""
        
        # Utiliser executor pour opérations bloquantes
        def get_system_info():
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            network = psutil.net_io_counters()
            load_avg = psutil.getloadavg() if hasattr(psutil, 'getloadavg') else (0, 0, 0)
            
            return {
                'cpu_usage': cpu_percent,
                'memory_usage': memory.percent,
                'memory_available_gb': memory.available / (1024**3),
                'disk_usage': disk.percent,
                'disk_free_gb': disk.free / (1024**3),
                'network_bytes_sent': network.bytes_sent,
                'network_bytes_recv': network.bytes_recv,
                'load_average': load_avg,
                'process_count': len(psutil.pids())
            }
        
        loop = asyncio.get_event_loop()
        system_info = await loop.run_in_executor(self.executor, get_system_info)
        
        return SystemMetrics(
            timestamp=datetime.now(),
            active_connections=0,  # À implémenter avec vraies connexions
            **system_info
        )
    
    async def _collect_application_metrics(self) -> ApplicationMetrics:
        """Collecte les métriques applicatives"""
        
        # Métriques depuis performance analyzer
        performance_analyzer = get_performance_analyzer()
        
        # Métriques cache
        cache_stats = {'hit_ratio': 0.75}  # Default
        try:
            cache_manager = await get_cache_manager()
            cache_stats = cache_manager.get_stats()
        except:
            pass
        
        # Métriques background jobs
        job_stats = {'pending': 0, 'running': 0}
        try:
            from app.core.background_jobs import get_background_job_manager
            job_manager = await get_background_job_manager()
            queue_status = await job_manager.get_queue_status()
            job_stats = {
                'pending': queue_status.get('total_scheduled', 0),
                'running': queue_status.get('total_active', 0)
            }
        except:
            pass
        
        # Calculer métriques HTTP depuis analyzer
        recent_metrics = list(performance_analyzer.metrics_history)[-100:] if performance_analyzer.metrics_history else []
        http_metrics = [m for m in recent_metrics if 'http' in m.operation.lower() or 'api' in m.operation.lower()]
        
        avg_duration = statistics.mean([m.duration_ms for m in http_metrics]) / 1000 if http_metrics else 0.1
        error_count_5xx = len([m for m in http_metrics if m.error])
        error_count_4xx = 0  # À implémenter avec vraies métriques
        
        return ApplicationMetrics(
            timestamp=datetime.now(),
            http_requests_total=len(recent_metrics),
            http_request_duration_avg=avg_duration,
            http_errors_5xx=error_count_5xx,
            http_errors_4xx=error_count_4xx,
            database_connections_active=5,  # À implémenter
            database_query_duration_avg=0.05,  # À implémenter
            cache_hit_rate=cache_stats.get('total', {}).get('hit_ratio', 0.75) * 100,
            background_jobs_pending=job_stats['pending'],
            background_jobs_running=job_stats['running'],
            active_users=1  # À implémenter avec vraies sessions
        )
    
    async def _store_metrics(self, system_metrics: SystemMetrics, app_metrics: ApplicationMetrics):
        """Stocke les métriques dans l'historique"""
        
        timestamp = datetime.now().isoformat()
        
        # Métriques système
        self.metrics_history['cpu_usage_percent'].append((timestamp, system_metrics.cpu_usage))
        self.metrics_history['memory_usage_percent'].append((timestamp, system_metrics.memory_usage))
        self.metrics_history['disk_usage_percent'].append((timestamp, system_metrics.disk_usage))
        self.metrics_history['network_bytes_sent'].append((timestamp, system_metrics.network_bytes_sent))
        self.metrics_history['network_bytes_recv'].append((timestamp, system_metrics.network_bytes_recv))
        
        # Métriques applicatives
        self.metrics_history['http_request_duration_avg'].append((timestamp, app_metrics.http_request_duration_avg))
        self.metrics_history['cache_hit_rate'].append((timestamp, app_metrics.cache_hit_rate))
        self.metrics_history['background_jobs_pending'].append((timestamp, app_metrics.background_jobs_pending))
        self.metrics_history['background_jobs_running'].append((timestamp, app_metrics.background_jobs_running))
        self.metrics_history['http_errors_5xx'].append((timestamp, app_metrics.http_errors_5xx))
        
        # Sauvegarder dernières métriques
        self.last_system_metrics = system_metrics
        self.last_app_metrics = app_metrics
    
    def _update_prometheus_metrics(self, system_metrics: SystemMetrics, app_metrics: ApplicationMetrics):
        """Met à jour les métriques Prometheus"""
        if not PROMETHEUS_AVAILABLE:
            return
        
        try:
            # Métriques système
            self.prometheus_metrics['cpu_usage'].set(system_metrics.cpu_usage)
            self.prometheus_metrics['memory_usage'].set(system_metrics.memory_usage)
            self.prometheus_metrics['disk_usage'].set(system_metrics.disk_usage)
            
            # Métriques applicatives
            self.prometheus_metrics['cache_hit_rate'].set(app_metrics.cache_hit_rate)
            self.prometheus_metrics['db_connections'].set(app_metrics.database_connections_active)
            
            # Jobs arrière-plan
            self.prometheus_metrics['background_jobs'].labels(status='pending').set(app_metrics.background_jobs_pending)
            self.prometheus_metrics['background_jobs'].labels(status='running').set(app_metrics.background_jobs_running)
            
        except Exception as e:
            logger.warning(f"Erreur mise à jour Prometheus: {e}")
    
    async def _evaluate_alerts(self):
        """Évalue les règles d'alerte"""
        
        current_time = datetime.now()
        
        for rule_name, rule in self.alert_rules.items():
            if not rule.enabled:
                continue
            
            try:
                # Récupérer valeur métrique actuelle
                metric_value = await self._get_current_metric_value(rule.metric_name)
                if metric_value is None:
                    continue
                
                # Évaluer condition
                condition_met = self._evaluate_condition(metric_value, rule.condition, rule.threshold)
                
                # Vérifier si alerte déjà active
                alert_id = f"{rule_name}_{rule.metric_name}"
                existing_alert = self.active_alerts.get(alert_id)
                
                if condition_met:
                    if not existing_alert:
                        # Nouvelle alerte potentielle - vérifier durée
                        if not hasattr(rule, '_first_breach'):
                            rule._first_breach = current_time
                        elif (current_time - rule._first_breach).total_seconds() >= rule.duration_seconds:
                            # Déclencher alerte
                            await self._trigger_alert(rule, metric_value)
                            rule._first_breach = None
                    else:
                        # Alerte déjà active, mettre à jour valeur
                        existing_alert.metric_value = metric_value
                else:
                    # Condition non remplie
                    if hasattr(rule, '_first_breach'):
                        rule._first_breach = None
                    
                    if existing_alert and not existing_alert.resolved_at:
                        # Résoudre alerte
                        await self._resolve_alert(alert_id)
                        
            except Exception as e:
                logger.error(f"Erreur évaluation alerte {rule_name}: {e}")
    
    async def _get_current_metric_value(self, metric_name: str) -> Optional[float]:
        """Récupère la valeur actuelle d'une métrique"""
        
        if metric_name not in self.metrics_history:
            return None
        
        history = self.metrics_history[metric_name]
        if not history:
            return None
        
        # Retourner la dernière valeur
        return history[-1][1]
    
    def _evaluate_condition(self, value: float, condition: str, threshold: float) -> bool:
        """Évalue une condition d'alerte"""
        
        if condition == ">=":
            return value >= threshold
        elif condition == "<=":
            return value <= threshold
        elif condition == ">":
            return value > threshold
        elif condition == "<":
            return value < threshold
        elif condition == "==":
            return abs(value - threshold) < 0.001  # Égalité approximative
        else:
            return False
    
    async def _trigger_alert(self, rule: AlertRule, metric_value: float):
        """Déclenche une alerte"""
        
        # Vérifier cooldown
        if rule.last_triggered:
            time_since_last = (datetime.now() - rule.last_triggered).total_seconds()
            if time_since_last < rule.cooldown_seconds:
                return
        
        # Créer alerte
        alert_id = f"{rule.name}_{rule.metric_name}"
        message = rule.message_template.format(
            value=metric_value,
            threshold=rule.threshold,
            metric=rule.metric_name
        )
        
        alert = Alert(
            rule_name=rule.name,
            severity=rule.severity,
            message=message,
            metric_value=metric_value,
            threshold=rule.threshold,
            triggered_at=datetime.now()
        )
        
        self.active_alerts[alert_id] = alert
        rule.last_triggered = datetime.now()
        
        # Envoyer notifications
        await self._send_alert_notifications(alert)
        
        logger.warning(f"🚨 ALERTE {rule.severity.value.upper()}: {message}")
    
    async def _resolve_alert(self, alert_id: str):
        """Résout une alerte"""
        
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.resolved_at = datetime.now()
            
            logger.info(f"✅ Alerte résolue: {alert.rule_name}")
    
    async def _send_alert_notifications(self, alert: Alert):
        """Envoie les notifications d'alerte"""
        
        for handler in self.notification_handlers:
            try:
                await handler(alert)
            except Exception as e:
                logger.error(f"Erreur notification alerte: {e}")
    
    async def _prepare_dashboard_data(self) -> Dict[str, Any]:
        """Prépare les données pour le dashboard temps réel"""
        
        dashboard_data = {
            'timestamp': datetime.now().isoformat(),
            'system_metrics': {},
            'application_metrics': {},
            'alerts': [],
            'trends': {}
        }
        
        # Métriques système actuelles
        if self.last_system_metrics:
            dashboard_data['system_metrics'] = {
                'cpu_usage': self.last_system_metrics.cpu_usage,
                'memory_usage': self.last_system_metrics.memory_usage,
                'disk_usage': self.last_system_metrics.disk_usage,
                'load_average': self.last_system_metrics.load_average,
                'active_connections': self.last_system_metrics.active_connections
            }
        
        # Métriques applicatives actuelles
        if self.last_app_metrics:
            dashboard_data['application_metrics'] = {
                'http_duration_avg': self.last_app_metrics.http_request_duration_avg,
                'cache_hit_rate': self.last_app_metrics.cache_hit_rate,
                'background_jobs_pending': self.last_app_metrics.background_jobs_pending,
                'background_jobs_running': self.last_app_metrics.background_jobs_running,
                'database_connections': self.last_app_metrics.database_connections_active,
                'active_users': self.last_app_metrics.active_users
            }
        
        # Alertes actives
        dashboard_data['alerts'] = [
            {
                'id': alert_id,
                'severity': alert.severity.value,
                'message': alert.message,
                'triggered_at': alert.triggered_at.isoformat(),
                'resolved': alert.resolved_at is not None
            }
            for alert_id, alert in self.active_alerts.items()
            if not alert.resolved_at
        ]
        
        # Tendances (dernière heure)
        dashboard_data['trends'] = await self._calculate_trends()
        
        return dashboard_data
    
    async def _calculate_trends(self) -> Dict[str, Any]:
        """Calcule les tendances des métriques"""
        
        trends = {}
        cutoff_time = datetime.now() - timedelta(hours=1)
        
        for metric_name, history in self.metrics_history.items():
            if len(history) < 10:  # Pas assez de données
                continue
            
            # Filtrer dernière heure
            recent_data = [
                (timestamp, value) for timestamp, value in history
                if datetime.fromisoformat(timestamp) > cutoff_time
            ]
            
            if len(recent_data) < 5:
                continue
            
            values = [value for _, value in recent_data]
            
            # Calculer tendance
            if len(values) >= 2:
                first_half = values[:len(values)//2]
                second_half = values[len(values)//2:]
                
                avg_first = statistics.mean(first_half)
                avg_second = statistics.mean(second_half)
                
                trend_direction = "up" if avg_second > avg_first else "down" if avg_second < avg_first else "stable"
                trend_percentage = ((avg_second - avg_first) / avg_first * 100) if avg_first > 0 else 0
                
                trends[metric_name] = {
                    'direction': trend_direction,
                    'percentage': round(trend_percentage, 2),
                    'current_value': values[-1],
                    'avg_last_hour': round(statistics.mean(values), 2)
                }
        
        return trends
    
    async def _broadcast_to_websockets(self, data: Dict[str, Any]):
        """Diffuse les données aux connexions WebSocket"""
        
        if not self.websocket_connections:
            return
        
        message = json.dumps(data, default=str)
        disconnected = set()
        
        for websocket in self.websocket_connections:
            try:
                await websocket.send_text(message)
            except:
                disconnected.add(websocket)
        
        # Nettoyer connexions fermées
        self.websocket_connections -= disconnected
    
    def add_websocket_connection(self, websocket):
        """Ajoute une connexion WebSocket"""
        self.websocket_connections.add(websocket)
        logger.debug(f"Connexion WebSocket ajoutée: {len(self.websocket_connections)} actives")
    
    def remove_websocket_connection(self, websocket):
        """Supprime une connexion WebSocket"""
        self.websocket_connections.discard(websocket)
        logger.debug(f"Connexion WebSocket supprimée: {len(self.websocket_connections)} actives")
    
    def add_notification_handler(self, handler: Callable):
        """Ajoute un gestionnaire de notification"""
        self.notification_handlers.append(handler)
        logger.info(f"Gestionnaire de notification ajouté: {len(self.notification_handlers)} actifs")
    
    def get_prometheus_metrics(self) -> str:
        """Retourne les métriques au format Prometheus"""
        if not PROMETHEUS_AVAILABLE:
            return ""
        
        return generate_latest(self.prometheus_registry)
    
    def get_current_status(self) -> Dict[str, Any]:
        """Retourne le statut actuel du système"""
        
        status = {
            'monitoring_active': self.monitoring_active,
            'metrics_collected': len(self.metrics_history),
            'active_alerts': len([a for a in self.active_alerts.values() if not a.resolved_at]),
            'websocket_connections': len(self.websocket_connections),
            'collection_interval': self.collection_interval,
            'timestamp': datetime.now().isoformat()
        }
        
        if self.last_system_metrics:
            status['system_health'] = {
                'cpu_usage': self.last_system_metrics.cpu_usage,
                'memory_usage': self.last_system_metrics.memory_usage,
                'disk_usage': self.last_system_metrics.disk_usage
            }
        
        return status


# Instance globale
_realtime_monitor: Optional[RealtimeMonitor] = None


async def get_realtime_monitor() -> RealtimeMonitor:
    """Factory pour obtenir le monitor temps réel"""
    global _realtime_monitor
    
    if _realtime_monitor is None:
        _realtime_monitor = RealtimeMonitor()
        await _realtime_monitor.start_monitoring()
    
    return _realtime_monitor


# Gestionnaires de notification

async def slack_notification_handler(alert: Alert):
    """Envoie une notification Slack"""
    
    # Configuration Slack (à adapter selon environnement)
    webhook_url = "YOUR_SLACK_WEBHOOK_URL"
    
    if not webhook_url or webhook_url == "YOUR_SLACK_WEBHOOK_URL":
        logger.debug("Webhook Slack non configuré")
        return
    
    color = {
        AlertSeverity.INFO: "#36a64f",
        AlertSeverity.WARNING: "#ff9500", 
        AlertSeverity.ERROR: "#ff4444",
        AlertSeverity.CRITICAL: "#ff0000"
    }.get(alert.severity, "#808080")
    
    payload = {
        "attachments": [{
            "color": color,
            "title": f"🚨 Alerte {alert.severity.value.upper()}",
            "text": alert.message,
            "fields": [
                {"title": "Métrique", "value": alert.rule_name, "short": True},
                {"title": "Valeur", "value": f"{alert.metric_value:.2f}", "short": True},
                {"title": "Seuil", "value": f"{alert.threshold:.2f}", "short": True},
                {"title": "Déclenché", "value": alert.triggered_at.strftime("%H:%M:%S"), "short": True}
            ]
        }]
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(webhook_url, json=payload, timeout=10) as response:
                if response.status == 200:
                    logger.info("Notification Slack envoyée")
                else:
                    logger.warning(f"Erreur notification Slack: {response.status}")
    except Exception as e:
        logger.error(f"Erreur envoi Slack: {e}")


async def email_notification_handler(alert: Alert):
    """Envoie une notification par email (simulation)"""
    
    # Simulation d'envoi email
    logger.info(f"📧 Email simulé envoyé pour alerte {alert.severity.value}: {alert.message}")


# Décorateurs pour instrumentation

def monitor_endpoint(endpoint_name: str = None):
    """Décorateur pour monitorer les endpoints API"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            method = "GET"  # À détecter depuis le contexte FastAPI
            status = "200"
            
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                status = "500"
                raise
            finally:
                duration = time.time() - start_time
                
                # Enregistrer métriques si monitor disponible
                try:
                    monitor = await get_realtime_monitor()
                    if PROMETHEUS_AVAILABLE and monitor.prometheus_metrics:
                        monitor.prometheus_metrics['http_requests'].labels(method=method, status=status).inc()
                        monitor.prometheus_metrics['http_duration'].observe(duration)
                except:
                    pass
        
        return wrapper
    return decorator


def monitor_function(function_name: str = None):
    """Décorateur pour monitorer les fonctions critiques"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            fname = function_name or f"{func.__module__}.{func.__name__}"
            
            try:
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                logger.debug(f"⏱️ {fname}: {duration:.3f}s")
        
        return wrapper
    return decorator


# Fonctions utilitaires pour l'API

async def get_monitoring_dashboard_data() -> Dict[str, Any]:
    """Récupère les données pour le dashboard de monitoring"""
    
    monitor = await get_realtime_monitor()
    return await monitor._prepare_dashboard_data()


async def get_monitoring_alerts() -> List[Dict[str, Any]]:
    """Récupère la liste des alertes actives"""
    
    monitor = await get_realtime_monitor()
    return [
        {
            'id': alert_id,
            'rule_name': alert.rule_name,
            'severity': alert.severity.value,
            'message': alert.message,
            'metric_value': alert.metric_value,
            'threshold': alert.threshold,
            'triggered_at': alert.triggered_at.isoformat(),
            'resolved_at': alert.resolved_at.isoformat() if alert.resolved_at else None,
            'acknowledged': alert.acknowledged
        }
        for alert_id, alert in monitor.active_alerts.items()
    ]


async def acknowledge_alert(alert_id: str) -> bool:
    """Acquitte une alerte"""
    
    monitor = await get_realtime_monitor()
    if alert_id in monitor.active_alerts:
        monitor.active_alerts[alert_id].acknowledged = True
        logger.info(f"Alerte acquittée: {alert_id}")
        return True
    return False


async def get_metric_history(metric_name: str, hours: int = 1) -> List[Tuple[str, float]]:
    """Récupère l'historique d'une métrique"""
    
    monitor = await get_realtime_monitor()
    if metric_name not in monitor.metrics_history:
        return []
    
    cutoff_time = datetime.now() - timedelta(hours=hours)
    history = monitor.metrics_history[metric_name]
    
    return [
        (timestamp, value) for timestamp, value in history
        if datetime.fromisoformat(timestamp) > cutoff_time
    ]


# Configuration pour démarrage automatique
async def setup_monitoring():
    """Configure et démarre le monitoring"""
    
    monitor = await get_realtime_monitor()
    
    # Ajouter gestionnaires de notification
    monitor.add_notification_handler(slack_notification_handler)
    monitor.add_notification_handler(email_notification_handler)
    
    logger.info("✅ Système de monitoring configuré et démarré")