"""
Analyseur de performance pour identifier les goulots d'étranglement
US-009: Module d'analyse et optimisation des performances système

Ce module fournit:
- Détection automatique des goulots d'étranglement
- Profiling des opérations critiques 
- Métriques de performance en temps réel
- Recommandations d'optimisation automatisées
- Monitoring des ressources système
"""

import asyncio
import time
import psutil
import asyncpg
import aioredis
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from functools import wraps
import inspect
import sys
import tracemalloc
import gc
from collections import defaultdict, deque
import json
import numpy as np

from app.core.logging_system import get_logger, LogCategory

logger = get_logger("performance_analyzer", LogCategory.PERFORMANCE)


@dataclass
class PerformanceMetric:
    """Métrique de performance individuelle"""
    operation: str
    duration_ms: float
    memory_usage_mb: float
    cpu_percent: float
    timestamp: datetime
    context: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class BottleneckAnalysis:
    """Analyse d'un goulot d'étranglement identifié"""
    component: str
    severity: str  # 'critical', 'high', 'medium', 'low'
    impact_score: float  # 0-100
    description: str
    current_performance: Dict[str, float]
    recommendations: List[str]
    estimated_improvement: str
    implementation_effort: str


@dataclass
class SystemHealthMetrics:
    """Métriques de santé système globales"""
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_io: Dict[str, int]
    database_connections: int
    cache_hit_rate: float
    active_requests: int
    average_response_time: float
    error_rate: float


class PerformanceAnalyzer:
    """Analyseur principal de performance système"""
    
    def __init__(self):
        self.metrics_history: deque = deque(maxlen=10000)
        self.operation_stats = defaultdict(list)
        self.bottlenecks: List[BottleneckAnalysis] = []
        self.monitoring_active = False
        self.baseline_metrics: Optional[Dict[str, float]] = None
        
        # Seuils de performance
        self.performance_thresholds = {
            'api_response_time_ms': 1000,  # < 1 seconde
            'database_query_ms': 100,      # < 100ms
            'cache_hit_rate': 0.85,        # > 85%
            'memory_usage_percent': 80,     # < 80%
            'cpu_usage_percent': 70,        # < 70%
            'error_rate_percent': 1.0       # < 1%
        }
        
        # Compteurs en temps réel
        self.active_operations = {}
        self.request_counter = 0
        self.error_counter = 0
        
        logger.info("📊 Analyseur de performance initialisé")
    
    def performance_monitor(self, operation_name: str = None, 
                          track_memory: bool = True,
                          track_cpu: bool = True):
        """Décorateur pour monitorer automatiquement la performance d'une fonction"""
        def decorator(func: Callable) -> Callable:
            func_name = operation_name or f"{func.__module__}.{func.__name__}"
            
            if asyncio.iscoroutinefunction(func):
                @wraps(func)
                async def async_wrapper(*args, **kwargs):
                    return await self._monitor_async_operation(
                        func, func_name, track_memory, track_cpu, args, kwargs
                    )
                return async_wrapper
            else:
                @wraps(func)
                def sync_wrapper(*args, **kwargs):
                    return self._monitor_sync_operation(
                        func, func_name, track_memory, track_cpu, args, kwargs
                    )
                return sync_wrapper
        
        return decorator
    
    async def _monitor_async_operation(self, func: Callable, operation_name: str,
                                     track_memory: bool, track_cpu: bool,
                                     args: tuple, kwargs: dict) -> Any:
        """Monitore une opération asynchrone"""
        start_time = time.time()
        start_memory = self._get_memory_usage() if track_memory else 0
        start_cpu = psutil.cpu_percent() if track_cpu else 0
        
        operation_id = f"{operation_name}_{id(asyncio.current_task())}"
        self.active_operations[operation_id] = {
            'start_time': start_time,
            'operation': operation_name
        }
        
        error = None
        
        try:
            result = await func(*args, **kwargs)
            return result
            
        except Exception as e:
            error = str(e)
            self.error_counter += 1
            raise
            
        finally:
            end_time = time.time()
            duration_ms = (end_time - start_time) * 1000
            
            # Calculer métriques finales
            memory_usage = self._get_memory_usage() - start_memory if track_memory else 0
            cpu_usage = psutil.cpu_percent() - start_cpu if track_cpu else 0
            
            # Enregistrer métrique
            metric = PerformanceMetric(
                operation=operation_name,
                duration_ms=duration_ms,
                memory_usage_mb=memory_usage,
                cpu_percent=cpu_usage,
                timestamp=datetime.now(),
                context={
                    'args_count': len(args),
                    'kwargs_count': len(kwargs),
                    'thread_id': id(asyncio.current_task())
                },
                error=error
            )
            
            self._record_metric(metric)
            
            # Nettoyer opération active
            self.active_operations.pop(operation_id, None)
    
    def _monitor_sync_operation(self, func: Callable, operation_name: str,
                              track_memory: bool, track_cpu: bool,
                              args: tuple, kwargs: dict) -> Any:
        """Monitore une opération synchrone"""
        start_time = time.time()
        start_memory = self._get_memory_usage() if track_memory else 0
        start_cpu = psutil.cpu_percent() if track_cpu else 0
        
        error = None
        
        try:
            result = func(*args, **kwargs)
            return result
            
        except Exception as e:
            error = str(e)
            self.error_counter += 1
            raise
            
        finally:
            end_time = time.time()
            duration_ms = (end_time - start_time) * 1000
            
            memory_usage = self._get_memory_usage() - start_memory if track_memory else 0
            cpu_usage = psutil.cpu_percent() - start_cpu if track_cpu else 0
            
            metric = PerformanceMetric(
                operation=operation_name,
                duration_ms=duration_ms,
                memory_usage_mb=memory_usage,
                cpu_percent=cpu_usage,
                timestamp=datetime.now(),
                context={
                    'args_count': len(args),
                    'kwargs_count': len(kwargs)
                },
                error=error
            )
            
            self._record_metric(metric)
    
    def _record_metric(self, metric: PerformanceMetric):
        """Enregistre une métrique de performance"""
        self.metrics_history.append(metric)
        self.operation_stats[metric.operation].append(metric)
        
        # Limiter l'historique par opération
        if len(self.operation_stats[metric.operation]) > 1000:
            self.operation_stats[metric.operation] = self.operation_stats[metric.operation][-500:]
        
        # Analyser en temps réel si nécessaire
        if metric.duration_ms > self.performance_thresholds.get('api_response_time_ms', 1000):
            logger.warning(f"⚠️ Performance lente détectée: {metric.operation} ({metric.duration_ms:.1f}ms)")
    
    async def analyze_bottlenecks(self) -> List[BottleneckAnalysis]:
        """Analyse les goulots d'étranglement système"""
        logger.info("🔍 Analyse des goulots d'étranglement en cours...")
        
        bottlenecks = []
        
        # Analyser performance des opérations
        bottlenecks.extend(await self._analyze_operation_bottlenecks())
        
        # Analyser utilisation ressources
        bottlenecks.extend(await self._analyze_resource_bottlenecks())
        
        # Analyser performance base de données
        bottlenecks.extend(await self._analyze_database_bottlenecks())
        
        # Analyser cache performance
        bottlenecks.extend(await self._analyze_cache_bottlenecks())
        
        # Trier par score d'impact
        bottlenecks.sort(key=lambda x: x.impact_score, reverse=True)
        
        self.bottlenecks = bottlenecks
        logger.info(f"📊 Analyse terminée: {len(bottlenecks)} goulots identifiés")
        
        return bottlenecks
    
    async def _analyze_operation_bottlenecks(self) -> List[BottleneckAnalysis]:
        """Analyse les goulots d'étranglement d'opérations"""
        bottlenecks = []
        
        for operation, metrics in self.operation_stats.items():
            if len(metrics) < 10:  # Pas assez de données
                continue
            
            # Calculer statistiques
            durations = [m.duration_ms for m in metrics[-100:]]  # 100 dernières mesures
            avg_duration = np.mean(durations)
            p95_duration = np.percentile(durations, 95)
            error_rate = len([m for m in metrics[-100:] if m.error]) / len(metrics[-100:]) * 100
            
            # Identifier problèmes
            issues = []
            severity = 'low'
            impact_score = 0
            
            if avg_duration > self.performance_thresholds.get('api_response_time_ms', 1000):
                issues.append(f"Temps de réponse moyen élevé: {avg_duration:.1f}ms")
                severity = 'high'
                impact_score += 30
            
            if p95_duration > avg_duration * 3:
                issues.append(f"Variabilité importante (P95: {p95_duration:.1f}ms)")
                severity = max(severity, 'medium', key=lambda x: ['low', 'medium', 'high', 'critical'].index(x))
                impact_score += 20
            
            if error_rate > self.performance_thresholds.get('error_rate_percent', 1.0):
                issues.append(f"Taux d'erreur élevé: {error_rate:.1f}%")
                severity = 'critical'
                impact_score += 40
            
            if issues:
                # Générer recommandations
                recommendations = self._generate_operation_recommendations(operation, avg_duration, error_rate)
                
                bottleneck = BottleneckAnalysis(
                    component=f"Operation: {operation}",
                    severity=severity,
                    impact_score=impact_score,
                    description=f"Performance dégradée: {', '.join(issues)}",
                    current_performance={
                        'avg_duration_ms': avg_duration,
                        'p95_duration_ms': p95_duration,
                        'error_rate_percent': error_rate,
                        'call_count': len(metrics)
                    },
                    recommendations=recommendations,
                    estimated_improvement=self._estimate_improvement(severity),
                    implementation_effort=self._estimate_effort(recommendations)
                )
                
                bottlenecks.append(bottleneck)
        
        return bottlenecks
    
    async def _analyze_resource_bottlenecks(self) -> List[BottleneckAnalysis]:
        """Analyse les goulots d'étranglement de ressources système"""
        bottlenecks = []
        
        # Métriques actuelles
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # CPU
        if cpu_percent > self.performance_thresholds.get('cpu_usage_percent', 70):
            severity = 'critical' if cpu_percent > 90 else 'high'
            impact_score = min(100, cpu_percent)
            
            bottleneck = BottleneckAnalysis(
                component="System: CPU",
                severity=severity,
                impact_score=impact_score,
                description=f"Utilisation CPU élevée: {cpu_percent:.1f}%",
                current_performance={'cpu_usage_percent': cpu_percent},
                recommendations=[
                    "Optimiser les algorithmes CPU-intensifs",
                    "Implémenter la parallélisation",
                    "Considérer l'ajout de workers/processus",
                    "Profiler les fonctions les plus coûteuses"
                ],
                estimated_improvement="30-50% réduction temps réponse",
                implementation_effort="Moyen (1-2 semaines)"
            )
            bottlenecks.append(bottleneck)
        
        # Mémoire
        if memory.percent > self.performance_thresholds.get('memory_usage_percent', 80):
            severity = 'critical' if memory.percent > 95 else 'high'
            impact_score = min(100, memory.percent)
            
            bottleneck = BottleneckAnalysis(
                component="System: Memory",
                severity=severity,
                impact_score=impact_score,
                description=f"Utilisation mémoire élevée: {memory.percent:.1f}%",
                current_performance={
                    'memory_usage_percent': memory.percent,
                    'available_gb': memory.available / (1024**3)
                },
                recommendations=[
                    "Implémenter cache avec expiration",
                    "Optimiser chargement des données",
                    "Utiliser lazy loading",
                    "Nettoyer objets non utilisés",
                    "Configurer garbage collection"
                ],
                estimated_improvement="40-60% réduction utilisation mémoire",
                implementation_effort="Faible (3-5 jours)"
            )
            bottlenecks.append(bottleneck)
        
        # Disque
        if disk.percent > 85:
            severity = 'high' if disk.percent > 95 else 'medium'
            impact_score = min(100, disk.percent * 0.8)  # Moins critique que CPU/mémoire
            
            bottleneck = BottleneckAnalysis(
                component="System: Disk",
                severity=severity,
                impact_score=impact_score,
                description=f"Espace disque faible: {disk.percent:.1f}%",
                current_performance={
                    'disk_usage_percent': disk.percent,
                    'free_gb': disk.free / (1024**3)
                },
                recommendations=[
                    "Nettoyer logs et fichiers temporaires",
                    "Implémenter rotation des logs",
                    "Archiver anciennes données",
                    "Configurer monitoring espace disque"
                ],
                estimated_improvement="Prévention pannes système",
                implementation_effort="Faible (1-2 jours)"
            )
            bottlenecks.append(bottleneck)
        
        return bottlenecks
    
    async def _analyze_database_bottlenecks(self) -> List[BottleneckAnalysis]:
        """Analyse les goulots d'étranglement base de données"""
        bottlenecks = []
        
        # Analyser métriques DB des opérations
        db_operations = [op for op in self.operation_stats.keys() if 'db' in op.lower() or 'query' in op.lower()]
        
        for operation in db_operations:
            metrics = self.operation_stats[operation]
            if len(metrics) < 5:
                continue
            
            durations = [m.duration_ms for m in metrics[-50:]]
            avg_duration = np.mean(durations)
            
            if avg_duration > self.performance_thresholds.get('database_query_ms', 100):
                severity = 'high' if avg_duration > 500 else 'medium'
                impact_score = min(100, avg_duration / 10)
                
                bottleneck = BottleneckAnalysis(
                    component=f"Database: {operation}",
                    severity=severity,
                    impact_score=impact_score,
                    description=f"Requête DB lente: {avg_duration:.1f}ms moyenne",
                    current_performance={'avg_query_time_ms': avg_duration},
                    recommendations=[
                        "Ajouter index sur colonnes fréquemment utilisées",
                        "Optimiser requêtes SQL",
                        "Implémenter pagination",
                        "Utiliser connection pooling",
                        "Considérer requêtes asynchrones",
                        "Analyser plan d'exécution des requêtes"
                    ],
                    estimated_improvement="50-80% réduction temps requête",
                    implementation_effort="Moyen (1 semaine)"
                )
                bottlenecks.append(bottleneck)
        
        return bottlenecks
    
    async def _analyze_cache_bottlenecks(self) -> List[BottleneckAnalysis]:
        """Analyse les goulots d'étranglement de cache"""
        bottlenecks = []
        
        # Simuler analyse cache (à connecter avec Redis réel plus tard)
        cache_operations = [op for op in self.operation_stats.keys() if 'cache' in op.lower()]
        
        if cache_operations:
            # Analyser hit rate simulé
            simulated_hit_rate = 0.65  # Exemple de hit rate faible
            
            if simulated_hit_rate < self.performance_thresholds.get('cache_hit_rate', 0.85):
                impact_score = (0.85 - simulated_hit_rate) * 100
                
                bottleneck = BottleneckAnalysis(
                    component="Cache System",
                    severity='medium',
                    impact_score=impact_score,
                    description=f"Taux de hit cache faible: {simulated_hit_rate:.1%}",
                    current_performance={'cache_hit_rate': simulated_hit_rate},
                    recommendations=[
                        "Réviser stratégie de cache",
                        "Augmenter TTL pour données stables",
                        "Implémenter cache prédictif",
                        "Optimiser clés de cache",
                        "Ajouter cache warm-up",
                        "Monitorer patterns d'accès"
                    ],
                    estimated_improvement="25-40% réduction temps réponse",
                    implementation_effort="Moyen (1 semaine)"
                )
                bottlenecks.append(bottleneck)
        
        return bottlenecks
    
    def _generate_operation_recommendations(self, operation: str, avg_duration: float, error_rate: float) -> List[str]:
        """Génère des recommandations spécifiques à une opération"""
        recommendations = []
        
        if 'api' in operation.lower():
            recommendations.extend([
                "Implémenter cache de réponse",
                "Optimiser sérialisation JSON",
                "Utiliser compression gzip"
            ])
        
        if 'scraping' in operation.lower():
            recommendations.extend([
                "Implémenter parallélisation",
                "Optimiser rate limiting",
                "Utiliser session HTTP réutilisable"
            ])
        
        if 'ml' in operation.lower() or 'ai' in operation.lower():
            recommendations.extend([
                "Cache des résultats de modèle",
                "Optimisation calculs vectoriels",
                "Utiliser batch processing"
            ])
        
        if avg_duration > 1000:
            recommendations.extend([
                "Diviser en opérations plus petites",
                "Implémenter traitement asynchrone",
                "Optimiser algorithmes"
            ])
        
        if error_rate > 5:
            recommendations.extend([
                "Améliorer gestion d'erreurs",
                "Implémenter retry avec backoff",
                "Ajouter circuit breaker"
            ])
        
        return recommendations[:6]  # Limiter à 6 recommandations
    
    def _estimate_improvement(self, severity: str) -> str:
        """Estime l'amélioration potentielle"""
        improvements = {
            'critical': "60-80% amélioration performance",
            'high': "40-60% amélioration performance", 
            'medium': "20-40% amélioration performance",
            'low': "10-20% amélioration performance"
        }
        return improvements.get(severity, "Amélioration modérée")
    
    def _estimate_effort(self, recommendations: List[str]) -> str:
        """Estime l'effort d'implémentation"""
        effort_keywords = {
            'cache': 'Faible',
            'index': 'Faible',
            'optimiser': 'Moyen',
            'implémenter': 'Moyen',
            'parallélisation': 'Élevé',
            'refactor': 'Élevé'
        }
        
        max_effort = 'Faible'
        for rec in recommendations:
            for keyword, effort in effort_keywords.items():
                if keyword in rec.lower():
                    if effort == 'Élevé' or (effort == 'Moyen' and max_effort == 'Faible'):
                        max_effort = effort
        
        effort_details = {
            'Faible': 'Faible (2-5 jours)',
            'Moyen': 'Moyen (1-2 semaines)',
            'Élevé': 'Élevé (3-4 semaines)'
        }
        
        return effort_details.get(max_effort, 'Moyen (1-2 semaines)')
    
    def get_system_health(self) -> SystemHealthMetrics:
        """Retourne les métriques de santé système actuelles"""
        # Métriques système
        cpu_usage = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        network = psutil.net_io_counters()
        
        # Métriques application
        recent_metrics = list(self.metrics_history)[-100:] if self.metrics_history else []
        avg_response_time = np.mean([m.duration_ms for m in recent_metrics]) if recent_metrics else 0
        error_rate = (self.error_counter / max(1, self.request_counter)) * 100
        
        return SystemHealthMetrics(
            timestamp=datetime.now(),
            cpu_usage=cpu_usage,
            memory_usage=memory.percent,
            disk_usage=disk.percent,
            network_io={
                'bytes_sent': network.bytes_sent,
                'bytes_recv': network.bytes_recv
            },
            database_connections=0,  # À implémenter avec pool DB
            cache_hit_rate=0.75,    # À implémenter avec Redis
            active_requests=len(self.active_operations),
            average_response_time=avg_response_time,
            error_rate=error_rate
        )
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Génère un rapport de performance complet"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'system_health': self.get_system_health().__dict__,
            'bottlenecks': [b.__dict__ for b in self.bottlenecks],
            'operation_stats': {},
            'recommendations_summary': [],
            'performance_score': 0
        }
        
        # Statistiques par opération
        for operation, metrics in self.operation_stats.items():
            if len(metrics) >= 5:
                durations = [m.duration_ms for m in metrics[-50:]]
                report['operation_stats'][operation] = {
                    'avg_duration_ms': np.mean(durations),
                    'median_duration_ms': np.median(durations),
                    'p95_duration_ms': np.percentile(durations, 95),
                    'total_calls': len(metrics),
                    'error_count': len([m for m in metrics if m.error])
                }
        
        # Recommandations prioritaires
        critical_bottlenecks = [b for b in self.bottlenecks if b.severity == 'critical']
        high_bottlenecks = [b for b in self.bottlenecks if b.severity == 'high']
        
        report['recommendations_summary'] = (
            [f"CRITIQUE: {b.description}" for b in critical_bottlenecks[:3]] +
            [f"HAUTE: {b.description}" for b in high_bottlenecks[:3]]
        )
        
        # Score de performance global (0-100)
        base_score = 100
        for bottleneck in self.bottlenecks:
            severity_penalty = {'critical': 25, 'high': 15, 'medium': 8, 'low': 3}
            base_score -= severity_penalty.get(bottleneck.severity, 0)
        
        report['performance_score'] = max(0, base_score)
        
        return report
    
    def _get_memory_usage(self) -> float:
        """Retourne l'utilisation mémoire actuelle en MB"""
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)
    
    async def start_monitoring(self, interval_seconds: int = 60):
        """Démarre le monitoring continu en arrière-plan"""
        self.monitoring_active = True
        logger.info("📊 Démarrage monitoring performance continu")
        
        while self.monitoring_active:
            try:
                # Analyser goulots d'étranglement
                await self.analyze_bottlenecks()
                
                # Générer rapport si problèmes critiques
                critical_bottlenecks = [b for b in self.bottlenecks if b.severity == 'critical']
                if critical_bottlenecks:
                    logger.warning(f"🚨 {len(critical_bottlenecks)} goulots critiques détectés!")
                    for bottleneck in critical_bottlenecks[:3]:
                        logger.warning(f"   - {bottleneck.description}")
                
                await asyncio.sleep(interval_seconds)
                
            except Exception as e:
                logger.error(f"Erreur monitoring performance: {e}")
                await asyncio.sleep(interval_seconds)
    
    def stop_monitoring(self):
        """Arrête le monitoring continu"""
        self.monitoring_active = False
        logger.info("⏹️ Arrêt monitoring performance")


# Instance globale
_performance_analyzer: Optional[PerformanceAnalyzer] = None


def get_performance_analyzer() -> PerformanceAnalyzer:
    """Factory pour obtenir l'instance de l'analyseur de performance"""
    global _performance_analyzer
    
    if _performance_analyzer is None:
        _performance_analyzer = PerformanceAnalyzer()
    
    return _performance_analyzer


# Décorateur global pour faciliter l'usage
def monitor_performance(operation_name: str = None, track_memory: bool = True, track_cpu: bool = True):
    """Décorateur global pour monitorer la performance"""
    analyzer = get_performance_analyzer()
    return analyzer.performance_monitor(operation_name, track_memory, track_cpu)