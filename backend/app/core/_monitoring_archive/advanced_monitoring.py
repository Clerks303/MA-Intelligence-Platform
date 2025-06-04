"""
Système de monitoring avancé pour M&A Intelligence Platform
US-006: Monitoring et observabilité avec Prometheus, OpenTelemetry et ML

Features:
- Métriques Prometheus avec labels dynamiques
- Tracing distribué OpenTelemetry
- Détection d'anomalies ML avec scikit-learn
- Monitoring business et technique intégré
- SLA tracking et availability monitoring
- Custom metrics et alerting automatique
"""

import asyncio
import time
import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from functools import wraps
from collections import defaultdict, deque
import statistics
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import threading

# Prometheus
from prometheus_client import (
    Counter, Histogram, Gauge, Summary, Info, Enum as PrometheusEnum,
    CollectorRegistry, generate_latest, push_to_gateway, start_http_server
)

# OpenTelemetry
from opentelemetry import trace, metrics
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

from app.config import settings
from app.core.logging_system import get_logger, LogCategory
from app.core.cache_manager import get_cache_manager

logger = get_logger("advanced_monitoring", LogCategory.MONITORING)


class MetricType(str, Enum):
    """Types de métriques"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class SeverityLevel(str, Enum):
    """Niveaux de sévérité pour anomalies"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class MetricDefinition:
    """Définition d'une métrique"""
    name: str
    description: str
    metric_type: MetricType
    labels: List[str] = field(default_factory=list)
    buckets: List[float] = field(default_factory=list)  # Pour histogrammes
    unit: str = ""
    namespace: str = "ma_intelligence"


@dataclass
class BusinessMetric:
    """Métrique business avec contexte"""
    name: str
    value: float
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)
    category: str = "business"
    description: str = ""


@dataclass
class AnomalyDetection:
    """Résultat de détection d'anomalie"""
    metric_name: str
    current_value: float
    expected_range: Tuple[float, float]
    anomaly_score: float
    severity: SeverityLevel
    timestamp: datetime
    context: Dict[str, Any] = field(default_factory=dict)


class PrometheusMetrics:
    """Gestionnaire de métriques Prometheus"""
    
    def __init__(self):
        self.registry = CollectorRegistry()
        self.metrics: Dict[str, Any] = {}
        
        # Métriques système prédéfinies
        self._setup_system_metrics()
        
        # Métriques business
        self._setup_business_metrics()
        
        logger.info("📊 Prometheus metrics initialisé")
    
    def _setup_system_metrics(self):
        """Configure les métriques système"""
        
        # API Performance
        self.metrics['http_requests_total'] = Counter(
            'http_requests_total',
            'Total HTTP requests',
            ['method', 'endpoint', 'status_code'],
            registry=self.registry
        )
        
        self.metrics['http_request_duration'] = Histogram(
            'http_request_duration_seconds',
            'HTTP request duration',
            ['method', 'endpoint'],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
            registry=self.registry
        )
        
        # Database
        self.metrics['db_connections_active'] = Gauge(
            'db_connections_active',
            'Active database connections',
            registry=self.registry
        )
        
        self.metrics['db_query_duration'] = Histogram(
            'db_query_duration_seconds',
            'Database query duration',
            ['query_type', 'table'],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
            registry=self.registry
        )
        
        # Cache
        self.metrics['cache_operations_total'] = Counter(
            'cache_operations_total',
            'Total cache operations',
            ['operation', 'cache_level', 'result'],
            registry=self.registry
        )
        
        self.metrics['cache_hit_ratio'] = Gauge(
            'cache_hit_ratio',
            'Cache hit ratio',
            ['cache_level'],
            registry=self.registry
        )
        
        # System Resources
        self.metrics['system_cpu_usage'] = Gauge(
            'system_cpu_usage_percent',
            'System CPU usage percentage',
            registry=self.registry
        )
        
        self.metrics['system_memory_usage'] = Gauge(
            'system_memory_usage_percent',
            'System memory usage percentage',
            registry=self.registry
        )
        
        self.metrics['system_disk_usage'] = Gauge(
            'system_disk_usage_percent',
            'System disk usage percentage',
            ['mountpoint'],
            registry=self.registry
        )
    
    def _setup_business_metrics(self):
        """Configure les métriques business"""
        
        # Scraping Performance
        self.metrics['companies_scraped_total'] = Counter(
            'companies_scraped_total',
            'Total companies scraped',
            ['source', 'status'],
            registry=self.registry
        )
        
        self.metrics['scraping_success_rate'] = Gauge(
            'scraping_success_rate',
            'Scraping success rate',
            ['source'],
            registry=self.registry
        )
        
        self.metrics['scraping_duration'] = Histogram(
            'scraping_duration_seconds',
            'Scraping operation duration',
            ['source', 'batch_size'],
            buckets=[1, 5, 10, 30, 60, 300, 600, 1800, 3600],
            registry=self.registry
        )
        
        # Data Quality
        self.metrics['data_quality_score'] = Gauge(
            'data_quality_score',
            'Data quality score (0-100)',
            ['data_type'],
            registry=self.registry
        )
        
        self.metrics['enrichment_rate'] = Gauge(
            'enrichment_rate',
            'Data enrichment success rate',
            ['enrichment_type'],
            registry=self.registry
        )
        
        # Business KPIs
        self.metrics['prospects_identified'] = Counter(
            'prospects_identified_total',
            'Total prospects identified',
            ['score_range', 'sector'],
            registry=self.registry
        )
        
        self.metrics['conversion_rate'] = Gauge(
            'conversion_rate',
            'Prospect to customer conversion rate',
            ['source'],
            registry=self.registry
        )
        
        # User Activity
        self.metrics['active_users'] = Gauge(
            'active_users',
            'Number of active users',
            ['time_period'],
            registry=self.registry
        )
        
        self.metrics['user_actions_total'] = Counter(
            'user_actions_total',
            'Total user actions',
            ['action_type', 'user_role'],
            registry=self.registry
        )
    
    def increment_counter(self, metric_name: str, labels: Dict[str, str] = None, value: float = 1):
        """Incrémente un counter"""
        if metric_name in self.metrics:
            metric = self.metrics[metric_name]
            if labels:
                metric.labels(**labels).inc(value)
            else:
                metric.inc(value)
    
    def set_gauge(self, metric_name: str, value: float, labels: Dict[str, str] = None):
        """Définit la valeur d'une gauge"""
        if metric_name in self.metrics:
            metric = self.metrics[metric_name]
            if labels:
                metric.labels(**labels).set(value)
            else:
                metric.set(value)
    
    def observe_histogram(self, metric_name: str, value: float, labels: Dict[str, str] = None):
        """Observe une valeur dans un histogramme"""
        if metric_name in self.metrics:
            metric = self.metrics[metric_name]
            if labels:
                metric.labels(**labels).observe(value)
            else:
                metric.observe(value)
    
    def get_metrics_text(self) -> str:
        """Retourne les métriques au format Prometheus"""
        return generate_latest(self.registry).decode('utf-8')


class OpenTelemetryTracing:
    """Gestionnaire de tracing distribué"""
    
    def __init__(self):
        self.tracer_provider = None
        self.tracer = None
        self.meter_provider = None
        self.meter = None
        
        self._setup_tracing()
        self._setup_metrics()
        
        logger.info("🔍 OpenTelemetry tracing initialisé")
    
    def _setup_tracing(self):
        """Configure le tracing"""
        try:
            # Configuration TracerProvider
            self.tracer_provider = TracerProvider(
                resource=self._create_resource()
            )
            trace.set_tracer_provider(self.tracer_provider)
            
            # Exporteur Jaeger
            jaeger_exporter = JaegerExporter(
                agent_host_name=getattr(settings, 'JAEGER_HOST', 'localhost'),
                agent_port=getattr(settings, 'JAEGER_PORT', 6831),
            )
            
            span_processor = BatchSpanProcessor(jaeger_exporter)
            self.tracer_provider.add_span_processor(span_processor)
            
            self.tracer = trace.get_tracer(__name__)
            
        except Exception as e:
            logger.warning(f"Jaeger non disponible: {e}")
    
    def _setup_metrics(self):
        """Configure les métriques OpenTelemetry"""
        try:
            self.meter_provider = MeterProvider(
                resource=self._create_resource()
            )
            metrics.set_meter_provider(self.meter_provider)
            
            self.meter = metrics.get_meter(__name__)
            
        except Exception as e:
            logger.warning(f"OpenTelemetry metrics non disponibles: {e}")
    
    def _create_resource(self):
        """Crée la ressource OpenTelemetry"""
        from opentelemetry.sdk.resources import Resource
        
        return Resource.create({
            "service.name": "ma-intelligence-api",
            "service.version": "1.0.0",
            "environment": getattr(settings, 'ENVIRONMENT', 'development'),
            "instance.id": f"api-{int(time.time())}"
        })
    
    def start_span(self, name: str, attributes: Dict[str, Any] = None):
        """Démarre un nouveau span"""
        if self.tracer:
            span = self.tracer.start_span(name)
            if attributes:
                span.set_attributes(attributes)
            return span
        return None
    
    def trace_function(self, operation_name: str = None):
        """Décorateur pour tracer une fonction"""
        def decorator(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                span_name = operation_name or f"{func.__module__}.{func.__name__}"
                
                if self.tracer:
                    with self.tracer.start_as_current_span(span_name) as span:
                        span.set_attribute("function.name", func.__name__)
                        span.set_attribute("function.module", func.__module__)
                        
                        try:
                            result = await func(*args, **kwargs)
                            span.set_attribute("success", True)
                            return result
                        except Exception as e:
                            span.set_attribute("success", False)
                            span.set_attribute("error.message", str(e))
                            span.set_attribute("error.type", type(e).__name__)
                            raise
                else:
                    return await func(*args, **kwargs)
            
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                span_name = operation_name or f"{func.__module__}.{func.__name__}"
                
                if self.tracer:
                    with self.tracer.start_as_current_span(span_name) as span:
                        span.set_attribute("function.name", func.__name__)
                        span.set_attribute("function.module", func.__module__)
                        
                        try:
                            result = func(*args, **kwargs)
                            span.set_attribute("success", True)
                            return result
                        except Exception as e:
                            span.set_attribute("success", False)
                            span.set_attribute("error.message", str(e))
                            span.set_attribute("error.type", type(e).__name__)
                            raise
                else:
                    return func(*args, **kwargs)
            
            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        
        return decorator


class AnomalyDetector:
    """Détecteur d'anomalies basé sur ML"""
    
    def __init__(self, window_size: int = 100, contamination: float = 0.1):
        self.window_size = window_size
        self.contamination = contamination
        self.models: Dict[str, IsolationForest] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.data_windows: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self.baselines: Dict[str, Dict[str, float]] = {}
        
        logger.info("🤖 Détecteur d'anomalies ML initialisé")
    
    def add_metric_value(self, metric_name: str, value: float, timestamp: datetime = None):
        """Ajoute une valeur de métrique pour analyse"""
        timestamp = timestamp or datetime.now()
        
        self.data_windows[metric_name].append({
            'value': value,
            'timestamp': timestamp
        })
        
        # Entraîner le modèle si suffisamment de données
        if len(self.data_windows[metric_name]) >= min(50, self.window_size):
            self._train_model(metric_name)
    
    def _train_model(self, metric_name: str):
        """Entraîne le modèle d'anomalie pour une métrique"""
        try:
            data = self.data_windows[metric_name]
            values = np.array([d['value'] for d in data]).reshape(-1, 1)
            
            # Normalisation
            if metric_name not in self.scalers:
                self.scalers[metric_name] = StandardScaler()
            
            scaled_values = self.scalers[metric_name].fit_transform(values)
            
            # Entraînement modèle
            if metric_name not in self.models:
                self.models[metric_name] = IsolationForest(
                    contamination=self.contamination,
                    random_state=42,
                    n_estimators=100
                )
            
            self.models[metric_name].fit(scaled_values)
            
            # Calcul baseline
            self.baselines[metric_name] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'p95': float(np.percentile(values, 95)),
                'p99': float(np.percentile(values, 99))
            }
            
        except Exception as e:
            logger.error(f"Erreur entraînement modèle {metric_name}: {e}")
    
    def detect_anomaly(self, metric_name: str, value: float) -> Optional[AnomalyDetection]:
        """Détecte une anomalie pour une valeur"""
        if metric_name not in self.models or metric_name not in self.scalers:
            return None
        
        try:
            # Normalisation
            scaled_value = self.scalers[metric_name].transform([[value]])
            
            # Prédiction anomalie
            anomaly_score = self.models[metric_name].decision_function(scaled_value)[0]
            is_anomaly = self.models[metric_name].predict(scaled_value)[0] == -1
            
            if is_anomaly:
                baseline = self.baselines[metric_name]
                
                # Calcul sévérité
                severity = self._calculate_severity(value, baseline, anomaly_score)
                
                # Range attendu (±2 std autour de la moyenne)
                expected_range = (
                    baseline['mean'] - 2 * baseline['std'],
                    baseline['mean'] + 2 * baseline['std']
                )
                
                return AnomalyDetection(
                    metric_name=metric_name,
                    current_value=value,
                    expected_range=expected_range,
                    anomaly_score=anomaly_score,
                    severity=severity,
                    timestamp=datetime.now(),
                    context={
                        'baseline': baseline,
                        'deviation_from_mean': abs(value - baseline['mean']) / baseline['std']
                    }
                )
        
        except Exception as e:
            logger.error(f"Erreur détection anomalie {metric_name}: {e}")
        
        return None
    
    def _calculate_severity(self, value: float, baseline: Dict[str, float], anomaly_score: float) -> SeverityLevel:
        """Calcule la sévérité d'une anomalie"""
        
        # Distance par rapport à la moyenne en unités d'écart-type
        std_distance = abs(value - baseline['mean']) / baseline['std'] if baseline['std'] > 0 else 0
        
        # Score d'anomalie normalisé (plus négatif = plus anormal)
        normalized_score = abs(anomaly_score)
        
        if std_distance > 4 or normalized_score > 0.8:
            return SeverityLevel.CRITICAL
        elif std_distance > 3 or normalized_score > 0.6:
            return SeverityLevel.HIGH
        elif std_distance > 2 or normalized_score > 0.4:
            return SeverityLevel.MEDIUM
        else:
            return SeverityLevel.LOW
    
    def get_baseline(self, metric_name: str) -> Optional[Dict[str, float]]:
        """Retourne la baseline d'une métrique"""
        return self.baselines.get(metric_name)


class AdvancedMonitoring:
    """Système de monitoring avancé intégré"""
    
    def __init__(self):
        self.prometheus = PrometheusMetrics()
        self.tracing = OpenTelemetryTracing()
        self.anomaly_detector = AnomalyDetector()
        
        # Business metrics tracking
        self.business_metrics: List[BusinessMetric] = []
        self.anomalies: List[AnomalyDetection] = []
        
        # SLA tracking
        self.sla_targets = {
            'api_availability': 99.9,  # 99.9% uptime
            'response_time_p95': 500,  # 500ms P95
            'scraping_success_rate': 95.0,  # 95% success
            'data_freshness_hours': 24  # Données < 24h
        }
        
        self.sla_status: Dict[str, Dict[str, Any]] = {}
        
        # Monitoring thread
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None
        
        logger.info("🚀 Advanced monitoring system initialisé")
    
    async def start_monitoring(self, interval_seconds: int = 60):
        """Démarre le monitoring continu"""
        self.monitoring_active = True
        
        # Thread de monitoring
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval_seconds,),
            daemon=True
        )
        self.monitoring_thread.start()
        
        # Démarrer serveur Prometheus metrics
        try:
            start_http_server(8001)  # Port pour métriques Prometheus
            logger.info("📊 Serveur métriques Prometheus démarré sur :8001")
        except Exception as e:
            logger.warning(f"Serveur Prometheus non démarré: {e}")
        
        logger.info(f"📈 Monitoring avancé démarré (interval: {interval_seconds}s)")
    
    def stop_monitoring(self):
        """Arrête le monitoring"""
        self.monitoring_active = False
        
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        
        logger.info("⏹️ Monitoring avancé arrêté")
    
    def _monitoring_loop(self, interval_seconds: int):
        """Boucle de monitoring"""
        while self.monitoring_active:
            try:
                # Collecter métriques système
                asyncio.create_task(self._collect_system_metrics())
                
                # Analyser anomalies
                asyncio.create_task(self._analyze_anomalies())
                
                # Vérifier SLA
                asyncio.create_task(self._check_sla_compliance())
                
                # Nettoyer anciennes données
                self._cleanup_old_data()
                
            except Exception as e:
                logger.error(f"Erreur monitoring loop: {e}")
            
            time.sleep(interval_seconds)
    
    async def _collect_system_metrics(self):
        """Collecte les métriques système"""
        try:
            import psutil
            
            # CPU
            cpu_percent = psutil.cpu_percent(interval=1)
            self.prometheus.set_gauge('system_cpu_usage', cpu_percent)
            self.anomaly_detector.add_metric_value('system_cpu_usage', cpu_percent)
            
            # Mémoire
            memory = psutil.virtual_memory()
            self.prometheus.set_gauge('system_memory_usage', memory.percent)
            self.anomaly_detector.add_metric_value('system_memory_usage', memory.percent)
            
            # Disque
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            self.prometheus.set_gauge('system_disk_usage', disk_percent, {'mountpoint': '/'})
            self.anomaly_detector.add_metric_value('system_disk_usage', disk_percent)
            
        except Exception as e:
            logger.error(f"Erreur collecte métriques système: {e}")
    
    async def _analyze_anomalies(self):
        """Analyse les anomalies récentes"""
        try:
            # Analyser les métriques importantes
            important_metrics = [
                'system_cpu_usage',
                'system_memory_usage', 
                'http_request_duration',
                'scraping_success_rate'
            ]
            
            for metric_name in important_metrics:
                if metric_name in self.anomaly_detector.data_windows:
                    recent_data = list(self.anomaly_detector.data_windows[metric_name])
                    if recent_data:
                        latest_value = recent_data[-1]['value']
                        anomaly = self.anomaly_detector.detect_anomaly(metric_name, latest_value)
                        
                        if anomaly:
                            self.anomalies.append(anomaly)
                            logger.warning(
                                f"🚨 Anomalie détectée: {anomaly.metric_name} = {anomaly.current_value:.2f} "
                                f"(sévérité: {anomaly.severity.value})"
                            )
        
        except Exception as e:
            logger.error(f"Erreur analyse anomalies: {e}")
    
    async def _check_sla_compliance(self):
        """Vérifie la conformité SLA"""
        try:
            current_time = datetime.now()
            
            for sla_name, target_value in self.sla_targets.items():
                if sla_name not in self.sla_status:
                    self.sla_status[sla_name] = {
                        'current_value': 0,
                        'target_value': target_value,
                        'status': 'unknown',
                        'last_check': current_time,
                        'violations': []
                    }
                
                # Simuler vérification SLA (à remplacer par vraies métriques)
                current_value = await self._get_sla_metric_value(sla_name)
                
                if current_value is not None:
                    self.sla_status[sla_name]['current_value'] = current_value
                    self.sla_status[sla_name]['last_check'] = current_time
                    
                    # Vérifier conformité
                    if sla_name in ['api_availability', 'scraping_success_rate']:
                        # Pour les pourcentages (plus haut = mieux)
                        compliant = current_value >= target_value
                    else:
                        # Pour les latences (plus bas = mieux)
                        compliant = current_value <= target_value
                    
                    self.sla_status[sla_name]['status'] = 'compliant' if compliant else 'violated'
                    
                    if not compliant:
                        violation = {
                            'timestamp': current_time,
                            'current_value': current_value,
                            'target_value': target_value
                        }
                        self.sla_status[sla_name]['violations'].append(violation)
                        
                        logger.warning(f"🚨 SLA violation: {sla_name} = {current_value} (target: {target_value})")
        
        except Exception as e:
            logger.error(f"Erreur vérification SLA: {e}")
    
    async def _get_sla_metric_value(self, sla_name: str) -> Optional[float]:
        """Récupère la valeur actuelle d'une métrique SLA"""
        # Ici on simule - en production, récupérer depuis Prometheus/cache
        import random
        
        if sla_name == 'api_availability':
            return random.uniform(99.0, 100.0)
        elif sla_name == 'response_time_p95':
            return random.uniform(200, 800)
        elif sla_name == 'scraping_success_rate':
            return random.uniform(90.0, 98.0)
        elif sla_name == 'data_freshness_hours':
            return random.uniform(1, 48)
        
        return None
    
    def _cleanup_old_data(self):
        """Nettoie les anciennes données"""
        cutoff_time = datetime.now() - timedelta(hours=24)
        
        # Nettoyer business metrics
        self.business_metrics = [
            m for m in self.business_metrics 
            if m.timestamp > cutoff_time
        ]
        
        # Nettoyer anomalies
        self.anomalies = [
            a for a in self.anomalies 
            if a.timestamp > cutoff_time
        ]
        
        # Nettoyer violations SLA
        for sla_data in self.sla_status.values():
            sla_data['violations'] = [
                v for v in sla_data['violations']
                if v['timestamp'] > cutoff_time
            ]
    
    # Méthodes publiques pour instrumenter l'application
    
    def track_business_metric(self, name: str, value: float, labels: Dict[str, str] = None, category: str = "business"):
        """Enregistre une métrique business"""
        metric = BusinessMetric(
            name=name,
            value=value,
            timestamp=datetime.now(),
            labels=labels or {},
            category=category
        )
        
        self.business_metrics.append(metric)
        self.anomaly_detector.add_metric_value(f"business_{name}", value)
        
        # Prometheus
        if f"business_{name}" in self.prometheus.metrics:
            self.prometheus.set_gauge(f"business_{name}", value, labels)
    
    def track_api_request(self, method: str, endpoint: str, status_code: int, duration_seconds: float):
        """Enregistre une requête API"""
        # Prometheus
        self.prometheus.increment_counter(
            'http_requests_total',
            {'method': method, 'endpoint': endpoint, 'status_code': str(status_code)}
        )
        
        self.prometheus.observe_histogram(
            'http_request_duration',
            duration_seconds,
            {'method': method, 'endpoint': endpoint}
        )
        
        # Anomaly detection
        self.anomaly_detector.add_metric_value('http_request_duration', duration_seconds * 1000)  # ms
    
    def track_scraping_operation(self, source: str, success: bool, duration_seconds: float, companies_count: int):
        """Enregistre une opération de scraping"""
        status = 'success' if success else 'failure'
        
        # Prometheus
        self.prometheus.increment_counter(
            'companies_scraped_total',
            {'source': source, 'status': status},
            companies_count
        )
        
        self.prometheus.observe_histogram(
            'scraping_duration',
            duration_seconds,
            {'source': source, 'batch_size': str(companies_count)}
        )
        
        # Business metric
        self.track_business_metric(
            f'scraping_operations',
            1,
            {'source': source, 'status': status},
            'scraping'
        )
    
    def get_monitoring_dashboard_data(self) -> Dict[str, Any]:
        """Retourne les données pour le dashboard"""
        return {
            'system_health': {
                'cpu_usage': self.anomaly_detector.get_baseline('system_cpu_usage'),
                'memory_usage': self.anomaly_detector.get_baseline('system_memory_usage'),
                'disk_usage': self.anomaly_detector.get_baseline('system_disk_usage')
            },
            'sla_status': self.sla_status,
            'recent_anomalies': [
                asdict(a) for a in self.anomalies[-10:]  # 10 dernières
            ],
            'business_metrics': [
                asdict(m) for m in self.business_metrics[-50:]  # 50 dernières
            ],
            'prometheus_metrics_url': 'http://localhost:8001/metrics'
        }


# Instance globale
_advanced_monitoring: Optional[AdvancedMonitoring] = None


async def get_advanced_monitoring() -> AdvancedMonitoring:
    """Factory pour obtenir le système de monitoring"""
    global _advanced_monitoring
    
    if _advanced_monitoring is None:
        _advanced_monitoring = AdvancedMonitoring()
    
    return _advanced_monitoring


# Décorateurs pour instrumenter facilement

def monitor_performance(operation_name: str = None):
    """Décorateur pour monitorer la performance d'une fonction"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            monitoring = await get_advanced_monitoring()
            start_time = time.time()
            
            operation = operation_name or f"{func.__module__}.{func.__name__}"
            
            # Tracing
            tracer_decorator = monitoring.tracing.trace_function(operation)
            traced_func = tracer_decorator(func)
            
            try:
                result = await traced_func(*args, **kwargs)
                success = True
            except Exception as e:
                success = False
                raise
            finally:
                duration = time.time() - start_time
                
                # Métriques Prometheus
                monitoring.prometheus.observe_histogram(
                    'operation_duration',
                    duration,
                    {'operation': operation, 'success': str(success)}
                )
                
                # Business metric
                monitoring.track_business_metric(
                    f'operation_{operation.replace(".", "_")}',
                    duration * 1000,  # ms
                    {'success': str(success)},
                    'performance'
                )
            
            return result
        
        return wrapper
    return decorator


# Fonctions utilitaires

async def setup_fastapi_monitoring(app):
    """Configure le monitoring pour FastAPI"""
    monitoring = await get_advanced_monitoring()
    
    # Instrumenter FastAPI avec OpenTelemetry
    FastAPIInstrumentor.instrument_app(app)
    
    # Middleware pour métriques Prometheus
    @app.middleware("http")
    async def prometheus_middleware(request, call_next):
        start_time = time.time()
        
        response = await call_next(request)
        
        duration = time.time() - start_time
        
        monitoring.track_api_request(
            method=request.method,
            endpoint=request.url.path,
            status_code=response.status_code,
            duration_seconds=duration
        )
        
        return response
    
    logger.info("📊 Monitoring FastAPI configuré")


async def start_monitoring_services():
    """Démarre tous les services de monitoring"""
    monitoring = await get_advanced_monitoring()
    await monitoring.start_monitoring(interval_seconds=30)  # Collecte toutes les 30s