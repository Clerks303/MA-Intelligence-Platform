"""
Moniteur de performance système pour M&A Intelligence Platform
US-005: Monitoring performance temps réel avec métriques et profiling

Features:
- Monitoring CPU, mémoire, I/O en temps réel
- Profiling des fonctions critiques
- Détection des bottlenecks automatique
- Alertes performance intelligentes
- Recommandations d'optimisation
- Benchmarks et comparaisons historiques
"""

import asyncio
import psutil
import time
import threading
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Callable, Tuple
from dataclasses import dataclass, field, asdict
from functools import wraps
from collections import deque, defaultdict
import tracemalloc
import gc
import sys
import json

from app.core.logging_system import get_logger, LogCategory, performance_logger
from app.core.metrics_collector import get_metrics_collector
from app.core.alerting_system import get_alerting_system, AlertSeverity, AlertCategory

logger = get_logger("performance_monitor", LogCategory.PERFORMANCE)


@dataclass
class SystemMetrics:
    """Métriques système instantanées"""
    timestamp: datetime = field(default_factory=datetime.now)
    
    # CPU
    cpu_percent: float = 0.0
    cpu_count: int = 0
    load_average: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    
    # Mémoire
    memory_total: int = 0
    memory_available: int = 0
    memory_percent: float = 0.0
    memory_used: int = 0
    
    # Swap
    swap_total: int = 0
    swap_used: int = 0
    swap_percent: float = 0.0
    
    # Disque
    disk_total: int = 0
    disk_used: int = 0
    disk_free: int = 0
    disk_percent: float = 0.0
    
    # Réseau
    network_bytes_sent: int = 0
    network_bytes_recv: int = 0
    network_packets_sent: int = 0
    network_packets_recv: int = 0
    
    # Process Python
    process_memory_rss: int = 0
    process_memory_vms: int = 0
    process_cpu_percent: float = 0.0
    process_threads: int = 0
    process_open_files: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Conversion en dictionnaire"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'cpu': {
                'percent': self.cpu_percent,
                'count': self.cpu_count,
                'load_average': self.load_average
            },
            'memory': {
                'total_mb': self.memory_total // 1024 // 1024,
                'available_mb': self.memory_available // 1024 // 1024,
                'used_mb': self.memory_used // 1024 // 1024,
                'percent': self.memory_percent
            },
            'swap': {
                'total_mb': self.swap_total // 1024 // 1024,
                'used_mb': self.swap_used // 1024 // 1024,
                'percent': self.swap_percent
            },
            'disk': {
                'total_gb': self.disk_total // 1024 // 1024 // 1024,
                'used_gb': self.disk_used // 1024 // 1024 // 1024,
                'free_gb': self.disk_free // 1024 // 1024 // 1024,
                'percent': self.disk_percent
            },
            'network': {
                'bytes_sent_mb': self.network_bytes_sent // 1024 // 1024,
                'bytes_recv_mb': self.network_bytes_recv // 1024 // 1024,
                'packets_sent': self.network_packets_sent,
                'packets_recv': self.network_packets_recv
            },
            'process': {
                'memory_rss_mb': self.process_memory_rss // 1024 // 1024,
                'memory_vms_mb': self.process_memory_vms // 1024 // 1024,
                'cpu_percent': self.process_cpu_percent,
                'threads': self.process_threads,
                'open_files': self.process_open_files
            }
        }


@dataclass
class FunctionProfile:
    """Profil de performance d'une fonction"""
    name: str
    calls: int = 0
    total_time: float = 0.0
    min_time: float = float('inf')
    max_time: float = 0.0
    avg_time: float = 0.0
    last_call: Optional[datetime] = None
    
    def add_call(self, duration: float):
        """Ajoute une mesure d'appel"""
        self.calls += 1
        self.total_time += duration
        self.min_time = min(self.min_time, duration)
        self.max_time = max(self.max_time, duration)
        self.avg_time = self.total_time / self.calls
        self.last_call = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'calls': self.calls,
            'total_time_ms': round(self.total_time * 1000, 2),
            'avg_time_ms': round(self.avg_time * 1000, 2),
            'min_time_ms': round(self.min_time * 1000, 2),
            'max_time_ms': round(self.max_time * 1000, 2),
            'last_call': self.last_call.isoformat() if self.last_call else None
        }


@dataclass
class PerformanceAlert:
    """Alerte de performance"""
    type: str
    message: str
    severity: AlertSeverity
    metric_value: float
    threshold: float
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': self.type,
            'message': self.message,
            'severity': self.severity.value,
            'metric_value': self.metric_value,
            'threshold': self.threshold,
            'timestamp': self.timestamp.isoformat()
        }


class MemoryProfiler:
    """Profileur mémoire avec tracemalloc"""
    
    def __init__(self):
        self.enabled = False
        self.snapshots: List[Any] = []
        self.baseline_snapshot = None
    
    def start(self):
        """Démarre le profiling mémoire"""
        if not self.enabled:
            tracemalloc.start()
            self.enabled = True
            self.baseline_snapshot = tracemalloc.take_snapshot()
            logger.info("🔍 Memory profiling démarré")
    
    def stop(self):
        """Arrête le profiling mémoire"""
        if self.enabled:
            tracemalloc.stop()
            self.enabled = False
            logger.info("⏹️ Memory profiling arrêté")
    
    def take_snapshot(self) -> Dict[str, Any]:
        """Prend un snapshot mémoire"""
        if not self.enabled:
            return {}
        
        try:
            current = tracemalloc.take_snapshot()
            
            # Comparaison avec baseline
            if self.baseline_snapshot:
                top_stats = current.compare_to(self.baseline_snapshot, 'lineno')
                
                # Top 10 des allocations
                top_allocations = []
                for stat in top_stats[:10]:
                    top_allocations.append({
                        'filename': stat.traceback.format()[0] if stat.traceback.format() else 'unknown',
                        'size_mb': stat.size / 1024 / 1024,
                        'size_diff_mb': stat.size_diff / 1024 / 1024,
                        'count': stat.count,
                        'count_diff': stat.count_diff
                    })
                
                return {
                    'timestamp': datetime.now().isoformat(),
                    'total_memory_mb': sum(stat.size for stat in top_stats) / 1024 / 1024,
                    'top_allocations': top_allocations
                }
            
            return {'timestamp': datetime.now().isoformat()}
            
        except Exception as e:
            logger.error(f"Erreur snapshot mémoire: {e}")
            return {}


class PerformanceMonitor:
    """Moniteur de performance système complet"""
    
    def __init__(self):
        self.running = False
        self.monitoring_thread: Optional[threading.Thread] = None
        self.system_metrics_history: deque = deque(maxlen=1440)  # 24h si collecte par minute
        self.function_profiles: Dict[str, FunctionProfile] = {}
        self.performance_alerts: List[PerformanceAlert] = []
        self.memory_profiler = MemoryProfiler()
        
        # Configuration seuils d'alerte
        self.alert_thresholds = {
            'cpu_percent': 80.0,
            'memory_percent': 85.0,
            'disk_percent': 90.0,
            'swap_percent': 50.0,
            'response_time_ms': 5000.0,
            'error_rate_percent': 10.0
        }
        
        # Statistiques performance
        self.stats = {
            'monitoring_start_time': None,
            'total_metrics_collected': 0,
            'alerts_generated': 0,
            'functions_profiled': 0
        }
        
        logger.info("📊 PerformanceMonitor initialisé")
    
    async def start_monitoring(self, interval_seconds: int = 60):
        """Démarre le monitoring continu"""
        if self.running:
            logger.warning("Monitoring déjà en cours")
            return
        
        self.running = True
        self.stats['monitoring_start_time'] = datetime.now()
        
        # Démarrer memory profiling
        self.memory_profiler.start()
        
        # Thread de monitoring système
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval_seconds,),
            daemon=True
        )
        self.monitoring_thread.start()
        
        logger.info(f"🚀 Monitoring performance démarré (interval: {interval_seconds}s)")
    
    def stop_monitoring(self):
        """Arrête le monitoring"""
        self.running = False
        
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        
        self.memory_profiler.stop()
        
        logger.info("⏹️ Monitoring performance arrêté")
    
    def _monitoring_loop(self, interval_seconds: int):
        """Boucle de monitoring système"""
        while self.running:
            try:
                # Collecter métriques système
                metrics = self._collect_system_metrics()
                self.system_metrics_history.append(metrics)
                self.stats['total_metrics_collected'] += 1
                
                # Vérifier seuils d'alerte
                asyncio.create_task(self._check_performance_alerts(metrics))
                
                # Publier métriques vers collecteur
                asyncio.create_task(self._publish_metrics(metrics))
                
            except Exception as e:
                logger.error(f"Erreur monitoring loop: {e}")
            
            time.sleep(interval_seconds)
    
    def _collect_system_metrics(self) -> SystemMetrics:
        """Collecte les métriques système actuelles"""
        try:
            # CPU
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            load_avg = psutil.getloadavg() if hasattr(psutil, 'getloadavg') else (0, 0, 0)
            
            # Mémoire
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            # Disque (partition racine)
            disk = psutil.disk_usage('/')
            
            # Réseau
            network = psutil.net_io_counters()
            
            # Process actuel
            process = psutil.Process()
            process_memory = process.memory_info()
            
            try:
                process_open_files = len(process.open_files())
            except (psutil.AccessDenied, OSError):
                process_open_files = 0
            
            return SystemMetrics(
                cpu_percent=cpu_percent,
                cpu_count=cpu_count,
                load_average=load_avg,
                memory_total=memory.total,
                memory_available=memory.available,
                memory_percent=memory.percent,
                memory_used=memory.used,
                swap_total=swap.total,
                swap_used=swap.used,
                swap_percent=swap.percent,
                disk_total=disk.total,
                disk_used=disk.used,
                disk_free=disk.free,
                disk_percent=(disk.used / disk.total * 100) if disk.total > 0 else 0,
                network_bytes_sent=network.bytes_sent,
                network_bytes_recv=network.bytes_recv,
                network_packets_sent=network.packets_sent,
                network_packets_recv=network.packets_recv,
                process_memory_rss=process_memory.rss,
                process_memory_vms=process_memory.vms,
                process_cpu_percent=process.cpu_percent(),
                process_threads=process.num_threads(),
                process_open_files=process_open_files
            )
            
        except Exception as e:
            logger.error(f"Erreur collecte métriques système: {e}")
            return SystemMetrics()
    
    async def _check_performance_alerts(self, metrics: SystemMetrics):
        """Vérifie les seuils d'alerte de performance"""
        try:
            alerts_to_send = []
            
            # CPU
            if metrics.cpu_percent > self.alert_thresholds['cpu_percent']:
                alert = PerformanceAlert(
                    type='high_cpu',
                    message=f"CPU usage élevé: {metrics.cpu_percent:.1f}%",
                    severity=AlertSeverity.WARNING if metrics.cpu_percent < 90 else AlertSeverity.CRITICAL,
                    metric_value=metrics.cpu_percent,
                    threshold=self.alert_thresholds['cpu_percent']
                )
                alerts_to_send.append(alert)
            
            # Mémoire
            if metrics.memory_percent > self.alert_thresholds['memory_percent']:
                alert = PerformanceAlert(
                    type='high_memory',
                    message=f"Memory usage élevé: {metrics.memory_percent:.1f}%",
                    severity=AlertSeverity.WARNING if metrics.memory_percent < 95 else AlertSeverity.CRITICAL,
                    metric_value=metrics.memory_percent,
                    threshold=self.alert_thresholds['memory_percent']
                )
                alerts_to_send.append(alert)
            
            # Disque
            if metrics.disk_percent > self.alert_thresholds['disk_percent']:
                alert = PerformanceAlert(
                    type='high_disk',
                    message=f"Disk usage élevé: {metrics.disk_percent:.1f}%",
                    severity=AlertSeverity.ERROR,
                    metric_value=metrics.disk_percent,
                    threshold=self.alert_thresholds['disk_percent']
                )
                alerts_to_send.append(alert)
            
            # Swap
            if metrics.swap_percent > self.alert_thresholds['swap_percent']:
                alert = PerformanceAlert(
                    type='high_swap',
                    message=f"Swap usage détecté: {metrics.swap_percent:.1f}%",
                    severity=AlertSeverity.WARNING,
                    metric_value=metrics.swap_percent,
                    threshold=self.alert_thresholds['swap_percent']
                )
                alerts_to_send.append(alert)
            
            # Envoyer alertes
            for alert in alerts_to_send:
                await self._send_performance_alert(alert)
                self.performance_alerts.append(alert)
                self.stats['alerts_generated'] += 1
            
            # Nettoyer anciennes alertes (garder 100 dernières)
            if len(self.performance_alerts) > 100:
                self.performance_alerts = self.performance_alerts[-100:]
                
        except Exception as e:
            logger.error(f"Erreur vérification alertes performance: {e}")
    
    async def _send_performance_alert(self, alert: PerformanceAlert):
        """Envoie une alerte de performance"""
        try:
            # Intégration avec système d'alerting
            alerting_system = await get_alerting_system()
            
            # Import local pour éviter dépendance circulaire
            from app.core.alerting_system import send_custom_alert
            
            await send_custom_alert(
                title=f"Performance Alert: {alert.type}",
                description=alert.message,
                severity=alert.severity,
                category=AlertCategory.PERFORMANCE
            )
            
        except Exception as e:
            logger.error(f"Erreur envoi alerte performance: {e}")
    
    async def _publish_metrics(self, metrics: SystemMetrics):
        """Publie les métriques vers le collecteur"""
        try:
            metrics_collector = await get_metrics_collector()
            
            # Métriques système
            metrics_collector.set_gauge("system_cpu_percent", metrics.cpu_percent)
            metrics_collector.set_gauge("system_memory_percent", metrics.memory_percent)
            metrics_collector.set_gauge("system_disk_percent", metrics.disk_percent)
            metrics_collector.set_gauge("system_swap_percent", metrics.swap_percent)
            
            # Métriques process
            metrics_collector.set_gauge("process_memory_rss_mb", metrics.process_memory_rss / 1024 / 1024)
            metrics_collector.set_gauge("process_cpu_percent", metrics.process_cpu_percent)
            metrics_collector.set_gauge("process_threads", metrics.process_threads)
            
        except Exception as e:
            logger.error(f"Erreur publication métriques: {e}")
    
    def profile_function(self, name: str = None):
        """Décorateur pour profiler une fonction"""
        def decorator(func):
            function_name = name or f"{func.__module__}.{func.__name__}"
            
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = await func(*args, **kwargs)
                    return result
                finally:
                    duration = time.time() - start_time
                    self._record_function_call(function_name, duration)
            
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    duration = time.time() - start_time
                    self._record_function_call(function_name, duration)
            
            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        
        return decorator
    
    def _record_function_call(self, function_name: str, duration: float):
        """Enregistre un appel de fonction"""
        if function_name not in self.function_profiles:
            self.function_profiles[function_name] = FunctionProfile(function_name)
            self.stats['functions_profiled'] += 1
        
        profile = self.function_profiles[function_name]
        profile.add_call(duration)
        
        # Log si fonction lente
        if duration > 1.0:  # > 1 seconde
            performance_logger.performance(
                operation=function_name,
                duration_ms=duration * 1000,
                success=True,
                details={'slow_function': True}
            )
    
    def get_current_metrics(self) -> Optional[Dict[str, Any]]:
        """Retourne les métriques système actuelles"""
        if self.system_metrics_history:
            return self.system_metrics_history[-1].to_dict()
        return None
    
    def get_metrics_history(self, hours: int = 1) -> List[Dict[str, Any]]:
        """Retourne l'historique des métriques"""
        if not self.system_metrics_history:
            return []
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_metrics = [
            metrics for metrics in self.system_metrics_history
            if metrics.timestamp > cutoff_time
        ]
        
        return [metrics.to_dict() for metrics in recent_metrics]
    
    def get_function_profiles(self, top_n: int = 20) -> List[Dict[str, Any]]:
        """Retourne les profils de fonctions"""
        # Trier par temps total décroissant
        sorted_profiles = sorted(
            self.function_profiles.values(),
            key=lambda p: p.total_time,
            reverse=True
        )
        
        return [profile.to_dict() for profile in sorted_profiles[:top_n]]
    
    def get_performance_alerts(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Retourne les alertes de performance"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_alerts = [
            alert for alert in self.performance_alerts
            if alert.timestamp > cutoff_time
        ]
        
        return [alert.to_dict() for alert in recent_alerts]
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Retourne un résumé de performance"""
        current_metrics = self.get_current_metrics()
        recent_alerts = self.get_performance_alerts(hours=1)
        top_functions = self.get_function_profiles(top_n=10)
        
        # Analyse tendances
        recommendations = self._generate_recommendations()
        
        return {
            'current_system': current_metrics,
            'monitoring_stats': self.stats,
            'recent_alerts': recent_alerts,
            'top_functions_by_time': top_functions,
            'memory_profile': self.memory_profiler.take_snapshot(),
            'recommendations': recommendations,
            'alert_thresholds': self.alert_thresholds
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Génère des recommandations d'optimisation"""
        recommendations = []
        
        if not self.system_metrics_history:
            return recommendations
        
        # Analyse des 10 dernières métriques
        recent_metrics = list(self.system_metrics_history)[-10:]
        
        # CPU moyen
        avg_cpu = sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics)
        if avg_cpu > 70:
            recommendations.append("CPU usage élevé - considérer optimisation ou scaling horizontal")
        
        # Mémoire moyenne
        avg_memory = sum(m.memory_percent for m in recent_metrics) / len(recent_metrics)
        if avg_memory > 80:
            recommendations.append("Memory usage élevé - analyser les fuites mémoire et optimiser le cache")
        
        # Swap usage
        if any(m.swap_percent > 0 for m in recent_metrics):
            recommendations.append("Swap détecté - augmenter la RAM ou optimiser l'usage mémoire")
        
        # Fonctions lentes
        slow_functions = [
            p for p in self.function_profiles.values()
            if p.avg_time > 1.0 and p.calls > 10
        ]
        if slow_functions:
            recommendations.append(f"{len(slow_functions)} fonctions lentes détectées - optimiser le code critique")
        
        # Garbage collection
        gc_stats = gc.get_stats()
        if gc_stats:
            total_collections = sum(stat['collections'] for stat in gc_stats)
            if total_collections > 1000:
                recommendations.append("GC fréquent détecté - revoir la gestion des objets Python")
        
        if not recommendations:
            recommendations.append("Performance système optimale")
        
        return recommendations
    
    async def run_performance_benchmark(self) -> Dict[str, Any]:
        """Exécute un benchmark de performance"""
        logger.info("🏃 Démarrage benchmark performance...")
        
        benchmark_results = {}
        
        try:
            # Test CPU
            start_time = time.time()
            # Simple calcul intensif
            result = sum(i * i for i in range(100000))
            cpu_benchmark_time = time.time() - start_time
            benchmark_results['cpu_benchmark_ms'] = cpu_benchmark_time * 1000
            
            # Test mémoire
            start_time = time.time()
            # Allocation/libération mémoire
            test_data = [list(range(1000)) for _ in range(1000)]
            del test_data
            gc.collect()
            memory_benchmark_time = time.time() - start_time
            benchmark_results['memory_benchmark_ms'] = memory_benchmark_time * 1000
            
            # Test I/O (écriture fichier temporaire)
            start_time = time.time()
            test_file = "/tmp/perf_test.json"
            test_data = {'test': list(range(10000))}
            with open(test_file, 'w') as f:
                json.dump(test_data, f)
            with open(test_file, 'r') as f:
                json.load(f)
            import os
            os.remove(test_file)
            io_benchmark_time = time.time() - start_time
            benchmark_results['io_benchmark_ms'] = io_benchmark_time * 1000
            
            # Score global (plus bas = mieux)
            total_score = (
                cpu_benchmark_time * 1000 +
                memory_benchmark_time * 500 +
                io_benchmark_time * 200
            )
            benchmark_results['total_score'] = total_score
            benchmark_results['timestamp'] = datetime.now().isoformat()
            
            logger.info(f"✅ Benchmark terminé - Score: {total_score:.2f}")
            
        except Exception as e:
            logger.error(f"Erreur benchmark: {e}")
            benchmark_results['error'] = str(e)
        
        return benchmark_results


# Instance globale
_performance_monitor: Optional[PerformanceMonitor] = None


async def get_performance_monitor() -> PerformanceMonitor:
    """Factory pour obtenir le moniteur de performance"""
    global _performance_monitor
    
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor()
    
    return _performance_monitor


# Décorateur de profiling
def profile_performance(name: str = None):
    """Décorateur pour profiler la performance d'une fonction"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            monitor = await get_performance_monitor()
            return await monitor.profile_function(name)(func)(*args, **kwargs)
        
        return wrapper
    return decorator


# Utilitaires

async def start_system_monitoring():
    """Démarre le monitoring système"""
    monitor = await get_performance_monitor()
    await monitor.start_monitoring(interval_seconds=60)


def stop_system_monitoring():
    """Arrête le monitoring système"""
    if _performance_monitor:
        _performance_monitor.stop_monitoring()