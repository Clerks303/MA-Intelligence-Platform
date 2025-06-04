"""
Système de collecte et analyse de métriques pour M&A Intelligence Platform
US-003: Métriques business et techniques en temps réel

Features:
- Métriques business (conversions, usage, ROI)
- Métriques techniques (performance, erreurs, infrastructure)
- Agrégation temporelle (temps réel, horaire, quotidienne)
- Export Prometheus/Grafana compatible
- Alerting sur seuils configurables
- Analytics prédictives
"""

import asyncio
import time
import statistics
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict, deque
import json
import threading
from concurrent.futures import ThreadPoolExecutor

from app.core.logging_system import get_logger, LogCategory
from app.config import settings

logger = get_logger("metrics", LogCategory.PERFORMANCE)


class MetricType(str, Enum):
    """Types de métriques collectées"""
    COUNTER = "counter"          # Compteur incrémental (ex: requests totales)
    GAUGE = "gauge"             # Valeur instantanée (ex: utilisateurs connectés)  
    HISTOGRAM = "histogram"     # Distribution de valeurs (ex: temps de réponse)
    TIMER = "timer"            # Durée d'opérations
    BUSINESS = "business"      # Métriques métier spécifiques


class MetricCategory(str, Enum):
    """Catégories de métriques pour organisation"""
    SYSTEM = "system"
    API = "api"
    DATABASE = "database"
    CACHE = "cache"
    BUSINESS = "business"
    SECURITY = "security"
    SCRAPING = "scraping"
    USER_EXPERIENCE = "user_experience"


@dataclass
class MetricPoint:
    """Point de métrique horodaté"""
    timestamp: datetime
    value: Union[int, float]
    labels: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'value': self.value,
            'labels': self.labels
        }


@dataclass
class MetricDefinition:
    """Définition d'une métrique"""
    name: str
    type: MetricType
    category: MetricCategory
    description: str
    unit: str = ""
    labels: List[str] = field(default_factory=list)
    retention_days: int = 30
    alert_thresholds: Dict[str, float] = field(default_factory=dict)


class MetricBuffer:
    """Buffer circulaire pour stockage efficace des métriques"""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.data = deque(maxlen=max_size)
        self._lock = threading.Lock()
    
    def add(self, point: MetricPoint):
        with self._lock:
            self.data.append(point)
    
    def get_recent(self, since: datetime = None, limit: int = None) -> List[MetricPoint]:
        with self._lock:
            if since is None:
                recent_data = list(self.data)
            else:
                recent_data = [p for p in self.data if p.timestamp >= since]
            
            if limit:
                recent_data = recent_data[-limit:]
                
            return recent_data
    
    def clear_old(self, before: datetime):
        with self._lock:
            # Convertir en liste pour filtrer
            filtered = [p for p in self.data if p.timestamp >= before]
            self.data.clear()
            self.data.extend(filtered)


class MetricsCollector:
    """
    Collecteur central de métriques avec agrégation et alerting
    
    Fonctionnalités:
    - Collecte métriques temps réel avec buffering
    - Agrégation par fenêtres temporelles
    - Calcul de percentiles et statistiques
    - Détection d'anomalies et alerting
    - Export compatible Prometheus
    """
    
    def __init__(self):
        self.metrics_definitions: Dict[str, MetricDefinition] = {}
        self.metrics_buffers: Dict[str, MetricBuffer] = {}
        self.aggregated_metrics: Dict[str, Dict[str, Any]] = defaultdict(dict)
        
        # Configuration
        self.collection_interval = 60  # secondes
        self.aggregation_windows = [300, 3600, 86400]  # 5min, 1h, 24h
        self.max_buffer_size = 50000
        
        # Threading pour agrégation en arrière-plan
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.running = False
        
        # Métriques système intégrées
        self._register_system_metrics()
        
        logger.info("🔢 Collecteur de métriques initialisé")
    
    def register_metric(self, definition: MetricDefinition):
        """Enregistre une nouvelle métrique"""
        self.metrics_definitions[definition.name] = definition
        self.metrics_buffers[definition.name] = MetricBuffer(self.max_buffer_size)
        
        logger.debug(f"Métrique enregistrée: {definition.name} ({definition.type.value})")
    
    def record(self, metric_name: str, value: Union[int, float], 
              labels: Dict[str, str] = None, timestamp: datetime = None):
        """Enregistre un point de métrique"""
        
        if metric_name not in self.metrics_definitions:
            logger.warning(f"Métrique non définie: {metric_name}")
            return
            
        point = MetricPoint(
            timestamp=timestamp or datetime.now(timezone.utc),
            value=value,
            labels=labels or {}
        )
        
        self.metrics_buffers[metric_name].add(point)
        
        # Mise à jour en temps réel des agrégations
        self._update_realtime_aggregation(metric_name, point)
    
    def increment(self, metric_name: str, value: int = 1, labels: Dict[str, str] = None):
        """Incrémente un compteur"""
        current = self.get_current_value(metric_name, labels) or 0
        self.record(metric_name, current + value, labels)
    
    def set_gauge(self, metric_name: str, value: Union[int, float], labels: Dict[str, str] = None):
        """Définit la valeur d'une gauge"""
        self.record(metric_name, value, labels)
    
    def time_operation(self, metric_name: str, labels: Dict[str, str] = None):
        """Context manager pour mesurer la durée d'une opération"""
        return TimerContext(self, metric_name, labels)
    
    def histogram(self, metric_name: str, value: float, labels: Dict[str, str] = None):
        """Enregistre une valeur dans un histogramme"""
        self.record(metric_name, value, labels)
        
        # Calcul percentiles en temps réel
        self._calculate_histogram_stats(metric_name, labels)
    
    def get_current_value(self, metric_name: str, labels: Dict[str, str] = None) -> Optional[float]:
        """Récupère la valeur actuelle d'une métrique"""
        if metric_name not in self.metrics_buffers:
            return None
            
        recent_points = self.metrics_buffers[metric_name].get_recent(limit=1)
        
        if not recent_points:
            return None
            
        # Filtrer par labels si spécifiés
        if labels:
            for point in reversed(recent_points):
                if all(point.labels.get(k) == v for k, v in labels.items()):
                    return point.value
            return None
        
        return recent_points[-1].value
    
    def get_metrics_summary(self, window_seconds: int = 300) -> Dict[str, Any]:
        """Résumé des métriques sur une fenêtre temporelle"""
        since = datetime.now(timezone.utc) - timedelta(seconds=window_seconds)
        summary = {}
        
        for metric_name, buffer in self.metrics_buffers.items():
            points = buffer.get_recent(since=since)
            
            if not points:
                continue
                
            values = [p.value for p in points]
            definition = self.metrics_definitions[metric_name]
            
            if definition.type == MetricType.COUNTER:
                summary[metric_name] = {
                    'type': 'counter',
                    'current': values[-1] if values else 0,
                    'increase': values[-1] - values[0] if len(values) > 1 else 0,
                    'rate_per_second': (values[-1] - values[0]) / window_seconds if len(values) > 1 else 0
                }
            elif definition.type == MetricType.GAUGE:
                summary[metric_name] = {
                    'type': 'gauge',
                    'current': values[-1] if values else 0,
                    'min': min(values),
                    'max': max(values),
                    'avg': statistics.mean(values)
                }
            elif definition.type in [MetricType.HISTOGRAM, MetricType.TIMER]:
                summary[metric_name] = {
                    'type': definition.type.value,
                    'count': len(values),
                    'min': min(values),
                    'max': max(values),
                    'avg': statistics.mean(values),
                    'p50': statistics.median(values),
                    'p95': statistics.quantiles(values, n=20)[18] if len(values) >= 20 else max(values),
                    'p99': statistics.quantiles(values, n=100)[98] if len(values) >= 100 else max(values)
                }
        
        return summary
    
    def get_business_metrics(self) -> Dict[str, Any]:
        """Métriques business spécifiques à la plateforme"""
        now = datetime.now(timezone.utc)
        last_hour = now - timedelta(hours=1)
        last_day = now - timedelta(days=1)
        
        return {
            'api_usage': {
                'requests_last_hour': self._count_metric('api_requests', last_hour),
                'requests_last_day': self._count_metric('api_requests', last_day),
                'avg_response_time_ms': self._avg_metric('api_response_time', last_hour),
                'error_rate_percent': self._error_rate('api_requests', 'api_errors', last_hour)
            },
            'scraping_performance': {
                'companies_scraped_last_hour': self._count_metric('companies_scraped', last_hour),
                'companies_scraped_last_day': self._count_metric('companies_scraped', last_day),
                'avg_scraping_time_ms': self._avg_metric('scraping_duration', last_hour),
                'success_rate_percent': self._success_rate('scraping_operations', last_hour)
            },
            'business_value': {
                'ma_scores_calculated': self._count_metric('ma_scores_calculated', last_day),
                'exports_generated': self._count_metric('exports_generated', last_day),
                'active_users_last_hour': self._unique_metric('active_users', last_hour),
                'cache_hit_ratio': self._avg_metric('cache_hit_ratio', last_hour)
            },
            'system_health': {
                'database_connection_pool': self.get_current_value('db_connection_pool_usage'),
                'redis_memory_usage_mb': self.get_current_value('redis_memory_usage'),
                'system_cpu_percent': self.get_current_value('system_cpu_usage'),
                'system_memory_percent': self.get_current_value('system_memory_usage')
            }
        }
    
    def export_prometheus_format(self) -> str:
        """Export des métriques au format Prometheus"""
        lines = []
        
        for metric_name, definition in self.metrics_definitions.items():
            # Métadonnées
            lines.append(f"# HELP {metric_name} {definition.description}")
            lines.append(f"# TYPE {metric_name} {self._prometheus_type(definition.type)}")
            
            # Points de données récents
            recent_points = self.metrics_buffers[metric_name].get_recent(limit=1)
            
            for point in recent_points:
                labels_str = ""
                if point.labels:
                    labels_pairs = [f'{k}="{v}"' for k, v in point.labels.items()]
                    labels_str = "{" + ",".join(labels_pairs) + "}"
                
                timestamp_ms = int(point.timestamp.timestamp() * 1000)
                lines.append(f"{metric_name}{labels_str} {point.value} {timestamp_ms}")
        
        return "\n".join(lines)
    
    async def start_background_aggregation(self):
        """Démarre l'agrégation en arrière-plan"""
        self.running = True
        logger.info("🔄 Démarrage agrégation métriques en arrière-plan")
        
        while self.running:
            try:
                await self._run_aggregation_cycle()
                await asyncio.sleep(self.collection_interval)
            except Exception as e:
                logger.error("Erreur agrégation métriques", exception=e)
                await asyncio.sleep(30)  # Attendre plus longtemps en cas d'erreur
    
    def stop_background_aggregation(self):
        """Arrête l'agrégation en arrière-plan"""
        self.running = False
        logger.info("⏹️ Arrêt agrégation métriques")
    
    async def _run_aggregation_cycle(self):
        """Cycle d'agrégation des métriques"""
        current_time = datetime.now(timezone.utc)
        
        # Nettoyage des anciennes données
        cutoff_time = current_time - timedelta(days=7)  # Garder 7 jours
        
        for buffer in self.metrics_buffers.values():
            buffer.clear_old(cutoff_time)
        
        # Agrégation par fenêtres temporelles
        for window_seconds in self.aggregation_windows:
            await self._aggregate_window(window_seconds, current_time)
        
        # Détection d'anomalies
        await self._detect_anomalies()
        
        logger.debug(f"Cycle d'agrégation terminé: {len(self.metrics_buffers)} métriques traitées")
    
    def _register_system_metrics(self):
        """Enregistre les métriques système de base"""
        
        system_metrics = [
            # API Metrics
            MetricDefinition("api_requests", MetricType.COUNTER, MetricCategory.API,
                           "Nombre total de requêtes API", "requests",
                           labels=["method", "endpoint", "status"],
                           alert_thresholds={"rate_per_minute": 1000}),
            
            MetricDefinition("api_response_time", MetricType.HISTOGRAM, MetricCategory.API,
                           "Temps de réponse API", "milliseconds",
                           labels=["method", "endpoint"],
                           alert_thresholds={"p95": 2000, "p99": 5000}),
            
            MetricDefinition("api_errors", MetricType.COUNTER, MetricCategory.API,
                           "Nombre d'erreurs API", "errors",
                           labels=["method", "endpoint", "error_type"],
                           alert_thresholds={"rate_per_minute": 10}),
            
            # Business Metrics
            MetricDefinition("companies_scraped", MetricType.COUNTER, MetricCategory.BUSINESS,
                           "Entreprises scrapées avec succès", "companies"),
            
            MetricDefinition("ma_scores_calculated", MetricType.COUNTER, MetricCategory.BUSINESS,
                           "Scores M&A calculés", "scores"),
            
            MetricDefinition("exports_generated", MetricType.COUNTER, MetricCategory.BUSINESS,
                           "Exports générés par les utilisateurs", "exports",
                           labels=["format", "user_type"]),
            
            MetricDefinition("active_users", MetricType.GAUGE, MetricCategory.BUSINESS,
                           "Utilisateurs actifs", "users",
                           labels=["time_window"]),
            
            # System Metrics
            MetricDefinition("system_cpu_usage", MetricType.GAUGE, MetricCategory.SYSTEM,
                           "Usage CPU système", "percent"),
            
            MetricDefinition("system_memory_usage", MetricType.GAUGE, MetricCategory.SYSTEM,
                           "Usage mémoire système", "percent"),
            
            MetricDefinition("db_connection_pool_usage", MetricType.GAUGE, MetricCategory.DATABASE,
                           "Usage du pool de connexions DB", "connections"),
            
            # Cache Metrics (integration avec US-002)
            MetricDefinition("cache_hit_ratio", MetricType.GAUGE, MetricCategory.CACHE,
                           "Ratio de hit du cache", "percent",
                           labels=["cache_type"],
                           alert_thresholds={"min": 80}),
            
            MetricDefinition("redis_memory_usage", MetricType.GAUGE, MetricCategory.CACHE,
                           "Usage mémoire Redis", "megabytes",
                           alert_thresholds={"max": 1000}),
            
            # Scraping Metrics
            MetricDefinition("scraping_duration", MetricType.HISTOGRAM, MetricCategory.SCRAPING,
                           "Durée des opérations de scraping", "milliseconds",
                           labels=["source", "operation_type"],
                           alert_thresholds={"p95": 30000}),
            
            MetricDefinition("scraping_operations", MetricType.COUNTER, MetricCategory.SCRAPING,
                           "Opérations de scraping", "operations",
                           labels=["source", "status"]),
        ]
        
        for metric in system_metrics:
            self.register_metric(metric)
    
    def _update_realtime_aggregation(self, metric_name: str, point: MetricPoint):
        """Mise à jour en temps réel des agrégations"""
        # Mise à jour simple pour les métriques en temps réel
        if metric_name not in self.aggregated_metrics:
            self.aggregated_metrics[metric_name] = {}
        
        self.aggregated_metrics[metric_name]['last_value'] = point.value
        self.aggregated_metrics[metric_name]['last_updated'] = point.timestamp
    
    def _calculate_histogram_stats(self, metric_name: str, labels: Dict[str, str] = None):
        """Calcul des statistiques d'histogramme"""
        # Récupérer les points récents pour calcul des percentiles
        recent_points = self.metrics_buffers[metric_name].get_recent(
            since=datetime.now(timezone.utc) - timedelta(minutes=5)
        )
        
        if labels:
            recent_points = [p for p in recent_points 
                           if all(p.labels.get(k) == v for k, v in labels.items())]
        
        if len(recent_points) < 2:
            return
        
        values = [p.value for p in recent_points]
        
        # Stockage des statistiques calculées
        stats_key = f"{metric_name}_stats"
        if labels:
            stats_key += "_" + "_".join(f"{k}_{v}" for k, v in labels.items())
        
        self.aggregated_metrics[stats_key] = {
            'count': len(values),
            'min': min(values),
            'max': max(values),
            'avg': statistics.mean(values),
            'p50': statistics.median(values),
            'calculated_at': datetime.now(timezone.utc)
        }
    
    async def _aggregate_window(self, window_seconds: int, current_time: datetime):
        """Agrégation sur une fenêtre temporelle"""
        since = current_time - timedelta(seconds=window_seconds)
        
        for metric_name, buffer in self.metrics_buffers.items():
            points = buffer.get_recent(since=since)
            
            if not points:
                continue
            
            definition = self.metrics_definitions[metric_name]
            window_key = f"{metric_name}_{window_seconds}s"
            
            values = [p.value for p in points]
            
            if definition.type == MetricType.COUNTER:
                self.aggregated_metrics[window_key] = {
                    'type': 'counter',
                    'total_increase': values[-1] - values[0] if len(values) > 1 else 0,
                    'rate_per_second': (values[-1] - values[0]) / window_seconds if len(values) > 1 else 0,
                    'window_seconds': window_seconds,
                    'aggregated_at': current_time
                }
            elif definition.type == MetricType.GAUGE:
                self.aggregated_metrics[window_key] = {
                    'type': 'gauge',
                    'current': values[-1],
                    'min': min(values),
                    'max': max(values),
                    'avg': statistics.mean(values),
                    'window_seconds': window_seconds,
                    'aggregated_at': current_time
                }
    
    async def _detect_anomalies(self):
        """Détection d'anomalies basée sur les seuils configurés"""
        for metric_name, definition in self.metrics_definitions.items():
            if not definition.alert_thresholds:
                continue
            
            current_value = self.get_current_value(metric_name)
            if current_value is None:
                continue
            
            for threshold_type, threshold_value in definition.alert_thresholds.items():
                if self._check_threshold_exceeded(metric_name, threshold_type, 
                                                threshold_value, current_value):
                    await self._trigger_alert(metric_name, threshold_type, 
                                            current_value, threshold_value)
    
    def _check_threshold_exceeded(self, metric_name: str, threshold_type: str, 
                                threshold_value: float, current_value: float) -> bool:
        """Vérifie si un seuil est dépassé"""
        if threshold_type == "max":
            return current_value > threshold_value
        elif threshold_type == "min":
            return current_value < threshold_value
        elif threshold_type == "rate_per_minute":
            # Calculer le taux sur la dernière minute
            since = datetime.now(timezone.utc) - timedelta(minutes=1)
            points = self.metrics_buffers[metric_name].get_recent(since=since)
            if len(points) >= 2:
                rate = (points[-1].value - points[0].value)
                return rate > threshold_value
        return False
    
    async def _trigger_alert(self, metric_name: str, threshold_type: str, 
                           current_value: float, threshold_value: float):
        """Déclenche une alerte"""
        alert_data = {
            'metric_name': metric_name,
            'threshold_type': threshold_type,
            'current_value': current_value,
            'threshold_value': threshold_value,
            'severity': 'warning' if abs(current_value - threshold_value) / threshold_value < 0.5 else 'critical'
        }
        
        logger.warning(f"🚨 ALERTE MÉTRIQUE: {metric_name} {threshold_type} dépassé", 
                      extra=alert_data)
        
        # Ici on pourrait intégrer avec un système d'alerting externe
        # (Slack, email, PagerDuty, etc.)
    
    def _count_metric(self, metric_name: str, since: datetime) -> int:
        """Compte les occurrences d'une métrique depuis une date"""
        if metric_name not in self.metrics_buffers:
            return 0
        
        points = self.metrics_buffers[metric_name].get_recent(since=since)
        return len(points)
    
    def _avg_metric(self, metric_name: str, since: datetime) -> float:
        """Moyenne d'une métrique depuis une date"""
        if metric_name not in self.metrics_buffers:
            return 0.0
        
        points = self.metrics_buffers[metric_name].get_recent(since=since)
        if not points:
            return 0.0
        
        values = [p.value for p in points]
        return statistics.mean(values)
    
    def _error_rate(self, total_metric: str, error_metric: str, since: datetime) -> float:
        """Calcule le taux d'erreur"""
        total = self._count_metric(total_metric, since)
        errors = self._count_metric(error_metric, since)
        
        if total == 0:
            return 0.0
        
        return (errors / total) * 100
    
    def _success_rate(self, metric_name: str, since: datetime) -> float:
        """Calcule le taux de succès basé sur les labels de status"""
        if metric_name not in self.metrics_buffers:
            return 0.0
        
        points = self.metrics_buffers[metric_name].get_recent(since=since)
        if not points:
            return 0.0
        
        total = len(points)
        success = sum(1 for p in points if p.labels.get('status') == 'success')
        
        return (success / total) * 100 if total > 0 else 0.0
    
    def _unique_metric(self, metric_name: str, since: datetime) -> int:
        """Compte les valeurs uniques d'une métrique"""
        if metric_name not in self.metrics_buffers:
            return 0
        
        points = self.metrics_buffers[metric_name].get_recent(since=since)
        unique_values = set(p.value for p in points)
        return len(unique_values)
    
    def _prometheus_type(self, metric_type: MetricType) -> str:
        """Convertit le type de métrique vers Prometheus"""
        mapping = {
            MetricType.COUNTER: "counter",
            MetricType.GAUGE: "gauge", 
            MetricType.HISTOGRAM: "histogram",
            MetricType.TIMER: "histogram",
            MetricType.BUSINESS: "gauge"
        }
        return mapping.get(metric_type, "gauge")


class TimerContext:
    """Context manager pour mesure automatique de durée"""
    
    def __init__(self, collector: MetricsCollector, metric_name: str, labels: Dict[str, str] = None):
        self.collector = collector
        self.metric_name = metric_name
        self.labels = labels or {}
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration_ms = (time.time() - self.start_time) * 1000
            
            # Ajouter status selon succès/échec
            labels = self.labels.copy()
            labels['status'] = 'success' if exc_type is None else 'error'
            
            self.collector.histogram(self.metric_name, duration_ms, labels)


# Instance globale du collecteur
_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Factory pour obtenir le collecteur de métriques"""
    global _metrics_collector
    
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    
    return _metrics_collector


# Utilitaires et décorateurs

def record_metric(metric_name: str, value: Union[int, float], labels: Dict[str, str] = None):
    """Enregistre une métrique"""
    collector = get_metrics_collector()
    collector.record(metric_name, value, labels)


def increment_counter(metric_name: str, labels: Dict[str, str] = None):
    """Incrémente un compteur"""
    collector = get_metrics_collector()
    collector.increment(metric_name, 1, labels)


def set_gauge_value(metric_name: str, value: Union[int, float], labels: Dict[str, str] = None):
    """Définit la valeur d'une gauge"""
    collector = get_metrics_collector()
    collector.set_gauge(metric_name, value, labels)


def time_function(metric_name: str, labels: Dict[str, str] = None):
    """Décorateur pour mesurer la durée d'une fonction"""
    def decorator(func):
        import asyncio
        from functools import wraps
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            collector = get_metrics_collector()
            with collector.time_operation(metric_name, labels):
                return await func(*args, **kwargs)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            collector = get_metrics_collector()
            with collector.time_operation(metric_name, labels):
                return func(*args, **kwargs)
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator