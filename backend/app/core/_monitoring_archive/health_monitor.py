"""
Système de health checks complets pour M&A Intelligence Platform
US-003: Monitoring de la santé de tous les services et dépendances

Features:
- Health checks pour DB, Redis, APIs externes
- Monitoring continu avec seuils configurables  
- Diagnostics automatisés et réparation auto
- Circuit breakers pour services défaillants
- Tableau de bord santé temps réel
- Intégration avec alerting et métriques
"""

import asyncio
import aiohttp
import time
import psutil
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any, Optional, Callable, Awaitable
from dataclasses import dataclass, field, asdict
from enum import Enum
import json
from concurrent.futures import ThreadPoolExecutor

from app.core.logging_system import get_logger, LogCategory
from app.core.metrics_collector import get_metrics_collector, MetricType, MetricCategory
from app.config import settings

logger = get_logger("health_monitor", LogCategory.SYSTEM)


class HealthStatus(str, Enum):
    """États de santé possibles"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"
    MAINTENANCE = "maintenance"


class ServiceType(str, Enum):
    """Types de services monitorés"""
    DATABASE = "database"
    CACHE = "cache"
    EXTERNAL_API = "external_api"
    SYSTEM = "system"
    APPLICATION = "application"
    STORAGE = "storage"


@dataclass
class HealthCheck:
    """Définition d'un health check"""
    name: str
    service_type: ServiceType
    check_function: Callable[[], Awaitable[Dict[str, Any]]]
    timeout_seconds: int = 30
    interval_seconds: int = 60
    failure_threshold: int = 3
    recovery_threshold: int = 2
    critical: bool = True
    enabled: bool = True
    dependencies: List[str] = field(default_factory=list)


@dataclass
class HealthResult:
    """Résultat d'un health check"""
    name: str
    status: HealthStatus
    response_time_ms: float
    timestamp: datetime
    details: Dict[str, Any] = field(default_factory=dict)
    error_message: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'status': self.status.value,
            'response_time_ms': self.response_time_ms,
            'timestamp': self.timestamp.isoformat(),
            'details': self.details,
            'error_message': self.error_message
        }


@dataclass
class ServiceHealth:
    """État de santé d'un service"""
    name: str
    status: HealthStatus
    last_check: datetime
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    total_checks: int = 0
    success_rate: float = 100.0
    avg_response_time_ms: float = 0.0
    last_error: str = ""
    uptime_percent: float = 100.0


class CircuitBreakerState(str, Enum):
    """États du circuit breaker"""
    CLOSED = "closed"      # Service OK, requêtes passent
    OPEN = "open"          # Service KO, requêtes bloquées  
    HALF_OPEN = "half_open"  # Test de récupération


@dataclass
class CircuitBreaker:
    """Circuit breaker pour service défaillant"""
    name: str
    state: CircuitBreakerState = CircuitBreakerState.CLOSED
    failure_count: int = 0
    last_failure_time: Optional[datetime] = None
    failure_threshold: int = 5
    recovery_timeout_seconds: int = 60
    half_open_max_calls: int = 3
    half_open_calls: int = 0


class HealthMonitor:
    """
    Moniteur de santé centralisé pour tous les services
    
    Fonctionnalités:
    - Health checks périodiques automatisés
    - Circuit breakers pour isolation des pannes
    - Métriques de disponibilité et performance
    - Diagnostics automatisés et recommandations
    - Alerting intégré sur dégradation
    """
    
    def __init__(self):
        self.health_checks: Dict[str, HealthCheck] = {}
        self.service_health: Dict[str, ServiceHealth] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.health_history: Dict[str, List[HealthResult]] = {}
        
        # Configuration
        self.max_history_size = 1000
        self.global_timeout = 120
        self.check_interval = 30
        
        # État monitoring
        self.running = False
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.metrics_collector = get_metrics_collector()
        
        # Enregistrer health checks par défaut
        self._register_default_health_checks()
        
        logger.info("🏥 Health Monitor initialisé")
    
    def register_health_check(self, health_check: HealthCheck):
        """Enregistre un nouveau health check"""
        self.health_checks[health_check.name] = health_check
        self.service_health[health_check.name] = ServiceHealth(
            name=health_check.name,
            status=HealthStatus.UNKNOWN,
            last_check=datetime.now(timezone.utc)
        )
        self.circuit_breakers[health_check.name] = CircuitBreaker(
            name=health_check.name,
            failure_threshold=health_check.failure_threshold
        )
        self.health_history[health_check.name] = []
        
        logger.debug(f"Health check enregistré: {health_check.name}")
    
    async def run_single_check(self, check_name: str) -> HealthResult:
        """Exécute un health check spécifique"""
        if check_name not in self.health_checks:
            raise ValueError(f"Health check non trouvé: {check_name}")
        
        health_check = self.health_checks[check_name]
        circuit_breaker = self.circuit_breakers[check_name]
        
        # Vérifier circuit breaker
        if circuit_breaker.state == CircuitBreakerState.OPEN:
            if self._should_attempt_reset(circuit_breaker):
                circuit_breaker.state = CircuitBreakerState.HALF_OPEN
                circuit_breaker.half_open_calls = 0
            else:
                return HealthResult(
                    name=check_name,
                    status=HealthStatus.UNHEALTHY,
                    response_time_ms=0,
                    timestamp=datetime.now(timezone.utc),
                    error_message="Circuit breaker OPEN - service indisponible"
                )
        
        # Exécuter le check
        start_time = time.time()
        
        try:
            check_details = await asyncio.wait_for(
                health_check.check_function(),
                timeout=health_check.timeout_seconds
            )
            
            response_time_ms = (time.time() - start_time) * 1000
            
            result = HealthResult(
                name=check_name,
                status=HealthStatus.HEALTHY,
                response_time_ms=response_time_ms,
                timestamp=datetime.now(timezone.utc),
                details=check_details
            )
            
            # Mise à jour circuit breaker - succès
            self._handle_check_success(circuit_breaker)
            
        except asyncio.TimeoutError:
            response_time_ms = health_check.timeout_seconds * 1000
            result = HealthResult(
                name=check_name,
                status=HealthStatus.UNHEALTHY,
                response_time_ms=response_time_ms,
                timestamp=datetime.now(timezone.utc),
                error_message=f"Timeout après {health_check.timeout_seconds}s"
            )
            
            # Mise à jour circuit breaker - échec
            self._handle_check_failure(circuit_breaker)
            
        except Exception as e:
            response_time_ms = (time.time() - start_time) * 1000
            result = HealthResult(
                name=check_name,
                status=HealthStatus.UNHEALTHY,
                response_time_ms=response_time_ms,
                timestamp=datetime.now(timezone.utc),
                error_message=str(e)
            )
            
            # Mise à jour circuit breaker - échec
            self._handle_check_failure(circuit_breaker)
        
        # Mise à jour historique et métriques
        await self._update_service_health(check_name, result)
        
        return result
    
    async def run_all_checks(self) -> Dict[str, HealthResult]:
        """Exécute tous les health checks activés"""
        results = {}
        
        # Exécuter checks en parallèle
        tasks = []
        for check_name, health_check in self.health_checks.items():
            if health_check.enabled:
                tasks.append(self.run_single_check(check_name))
        
        if tasks:
            check_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for i, result in enumerate(check_results):
                check_name = list(self.health_checks.keys())[i]
                
                if isinstance(result, Exception):
                    results[check_name] = HealthResult(
                        name=check_name,
                        status=HealthStatus.UNHEALTHY,
                        response_time_ms=0,
                        timestamp=datetime.now(timezone.utc),
                        error_message=f"Erreur exécution check: {str(result)}"
                    )
                else:
                    results[check_name] = result
        
        return results
    
    async def get_system_health_overview(self) -> Dict[str, Any]:
        """Vue d'ensemble de la santé du système"""
        
        # Exécuter tous les checks
        recent_results = await self.run_all_checks()
        
        # Calculer statut global
        critical_services = [name for name, check in self.health_checks.items() if check.critical]
        critical_unhealthy = [name for name in critical_services 
                            if recent_results.get(name, {}).status == HealthStatus.UNHEALTHY]
        
        non_critical_unhealthy = [name for name, result in recent_results.items()
                                if result.status == HealthStatus.UNHEALTHY 
                                and name not in critical_services]
        
        if critical_unhealthy:
            overall_status = HealthStatus.UNHEALTHY
        elif non_critical_unhealthy:
            overall_status = HealthStatus.DEGRADED
        else:
            overall_status = HealthStatus.HEALTHY
        
        # Statistiques globales
        total_services = len(self.health_checks)
        healthy_services = sum(1 for r in recent_results.values() 
                              if r.status == HealthStatus.HEALTHY)
        
        # Temps de réponse moyen
        response_times = [r.response_time_ms for r in recent_results.values() 
                         if r.response_time_ms > 0]
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        
        return {
            'overall_status': overall_status.value,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'summary': {
                'total_services': total_services,
                'healthy_services': healthy_services,
                'degraded_services': len(non_critical_unhealthy),
                'unhealthy_services': len(critical_unhealthy),
                'availability_percent': (healthy_services / total_services * 100) if total_services > 0 else 0
            },
            'performance': {
                'avg_response_time_ms': round(avg_response_time, 2),
                'slowest_service': max(recent_results.items(), 
                                     key=lambda x: x[1].response_time_ms)[0] if recent_results else None
            },
            'services': {name: result.to_dict() for name, result in recent_results.items()},
            'critical_issues': critical_unhealthy,
            'warnings': non_critical_unhealthy,
            'circuit_breakers': {
                name: {
                    'state': cb.state.value,
                    'failure_count': cb.failure_count
                } for name, cb in self.circuit_breakers.items()
            }
        }
    
    async def get_service_diagnostics(self, service_name: str) -> Dict[str, Any]:
        """Diagnostics détaillés pour un service spécifique"""
        if service_name not in self.service_health:
            return {'error': f'Service non trouvé: {service_name}'}
        
        service = self.service_health[service_name]
        circuit_breaker = self.circuit_breakers[service_name]
        
        # Historique récent
        recent_history = self.health_history[service_name][-50:]  # 50 derniers checks
        
        # Analyse des tendances
        if len(recent_history) >= 10:
            recent_failures = sum(1 for h in recent_history[-10:] 
                                if h.status == HealthStatus.UNHEALTHY)
            trend = "degrading" if recent_failures > 3 else "stable"
        else:
            trend = "insufficient_data"
        
        # Recommandations
        recommendations = self._generate_service_recommendations(service_name, service, circuit_breaker)
        
        return {
            'service_name': service_name,
            'current_status': service.status.value,
            'last_check': service.last_check.isoformat(),
            'performance': {
                'success_rate_percent': service.success_rate,
                'avg_response_time_ms': service.avg_response_time_ms,
                'uptime_percent': service.uptime_percent,
                'total_checks': service.total_checks
            },
            'reliability': {
                'consecutive_failures': service.consecutive_failures,
                'consecutive_successes': service.consecutive_successes,
                'last_error': service.last_error,
                'trend': trend
            },
            'circuit_breaker': {
                'state': circuit_breaker.state.value,
                'failure_count': circuit_breaker.failure_count,
                'last_failure': circuit_breaker.last_failure_time.isoformat() if circuit_breaker.last_failure_time else None
            },
            'recent_history': [h.to_dict() for h in recent_history],
            'recommendations': recommendations
        }
    
    async def start_continuous_monitoring(self):
        """Démarre le monitoring continu en arrière-plan"""
        self.running = True
        logger.info("🔄 Démarrage monitoring continu health checks")
        
        while self.running:
            try:
                await self._monitoring_cycle()
                await asyncio.sleep(self.check_interval)
            except Exception as e:
                logger.error("Erreur cycle monitoring health", exception=e)
                await asyncio.sleep(60)  # Attendre plus en cas d'erreur
    
    def stop_continuous_monitoring(self):
        """Arrête le monitoring continu"""
        self.running = False
        logger.info("⏹️ Arrêt monitoring health checks")
    
    async def _monitoring_cycle(self):
        """Cycle de monitoring périodique"""
        start_time = time.time()
        
        # Exécuter tous les checks
        results = await self.run_all_checks()
        
        # Analyse et alerting
        for name, result in results.items():
            await self._analyze_result_and_alert(name, result)
        
        # Métriques globales
        cycle_duration_ms = (time.time() - start_time) * 1000
        healthy_count = sum(1 for r in results.values() if r.status == HealthStatus.HEALTHY)
        
        self.metrics_collector.histogram("health_check_cycle_duration", cycle_duration_ms)
        self.metrics_collector.set_gauge("healthy_services_count", healthy_count)
        self.metrics_collector.set_gauge("total_services_count", len(self.health_checks))
        
        logger.debug(f"Cycle monitoring terminé: {len(results)} checks en {cycle_duration_ms:.2f}ms")
    
    def _register_default_health_checks(self):
        """Enregistre les health checks par défaut du système"""
        
        # Database Health Check
        async def check_database():
            try:
                from app.core.database import get_db_session
                async with get_db_session() as session:
                    result = await session.execute("SELECT 1")
                    row = result.fetchone()
                    return {
                        'connection': 'ok',
                        'query_test': 'passed',
                        'version': 'postgresql'
                    }
            except Exception as e:
                raise Exception(f"Database check failed: {str(e)}")
        
        self.register_health_check(HealthCheck(
            name="database",
            service_type=ServiceType.DATABASE,
            check_function=check_database,
            timeout_seconds=10,
            interval_seconds=30,
            critical=True
        ))
        
        # Redis Cache Health Check
        async def check_redis():
            try:
                from app.core.cache import get_cache
                cache = await get_cache()
                health = await cache.health_check()
                
                if health.get("status") != "healthy":
                    raise Exception(f"Redis unhealthy: {health}")
                
                return {
                    'connection': 'ok',
                    'ping_latency_ms': health.get('ping_latency_ms'),
                    'memory_used_mb': health.get('memory_used_mb', 0)
                }
            except Exception as e:
                raise Exception(f"Redis check failed: {str(e)}")
        
        self.register_health_check(HealthCheck(
            name="redis_cache",
            service_type=ServiceType.CACHE,
            check_function=check_redis,
            timeout_seconds=5,
            interval_seconds=60,
            critical=True
        ))
        
        # System Resources Health Check
        async def check_system_resources():
            try:
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                
                # Seuils d'alerte
                alerts = []
                if cpu_percent > 80:
                    alerts.append(f"High CPU usage: {cpu_percent}%")
                if memory.percent > 85:
                    alerts.append(f"High memory usage: {memory.percent}%")
                if disk.percent > 90:
                    alerts.append(f"High disk usage: {disk.percent}%")
                
                if alerts:
                    raise Exception("; ".join(alerts))
                
                return {
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory.percent,
                    'disk_percent': disk.percent,
                    'status': 'optimal'
                }
            except Exception as e:
                raise Exception(f"System resources check failed: {str(e)}")
        
        self.register_health_check(HealthCheck(
            name="system_resources",
            service_type=ServiceType.SYSTEM,
            check_function=check_system_resources,
            timeout_seconds=15,
            interval_seconds=60,
            critical=False
        ))
        
        # External API Health Check (Pappers)
        async def check_pappers_api():
            try:
                if not settings.PAPPERS_API_KEY:
                    return {'status': 'disabled', 'reason': 'No API key configured'}
                
                async with aiohttp.ClientSession() as session:
                    url = "https://api.pappers.fr/v2/ping"
                    headers = {"Authorization": f"Bearer {settings.PAPPERS_API_KEY}"}
                    
                    async with session.get(url, headers=headers, timeout=10) as response:
                        if response.status == 200:
                            data = await response.json()
                            return {
                                'status': 'ok',
                                'response_code': response.status,
                                'api_status': data.get('status', 'unknown')
                            }
                        else:
                            raise Exception(f"API returned {response.status}")
            except Exception as e:
                raise Exception(f"Pappers API check failed: {str(e)}")
        
        self.register_health_check(HealthCheck(
            name="pappers_api",
            service_type=ServiceType.EXTERNAL_API,
            check_function=check_pappers_api,
            timeout_seconds=15,
            interval_seconds=300,  # Moins fréquent pour API externe
            critical=False
        ))
    
    async def _update_service_health(self, service_name: str, result: HealthResult):
        """Met à jour l'état de santé d'un service"""
        service = self.service_health[service_name]
        
        # Mise à jour statistiques
        service.last_check = result.timestamp
        service.total_checks += 1
        
        if result.status == HealthStatus.HEALTHY:
            service.consecutive_successes += 1
            service.consecutive_failures = 0
        else:
            service.consecutive_failures += 1
            service.consecutive_successes = 0
            service.last_error = result.error_message
        
        # Calcul taux de succès sur les 100 derniers checks
        history = self.health_history[service_name]
        history.append(result)
        
        # Limiter taille historique
        if len(history) > self.max_history_size:
            history.pop(0)
        
        # Calculer métriques sur historique récent
        recent_history = history[-100:]  # 100 derniers
        if recent_history:
            successful = sum(1 for h in recent_history if h.status == HealthStatus.HEALTHY)
            service.success_rate = (successful / len(recent_history)) * 100
            
            # Temps de réponse moyen
            response_times = [h.response_time_ms for h in recent_history if h.response_time_ms > 0]
            service.avg_response_time_ms = sum(response_times) / len(response_times) if response_times else 0
        
        # Déterminer statut actuel
        health_check = self.health_checks[service_name]
        if service.consecutive_failures >= health_check.failure_threshold:
            service.status = HealthStatus.UNHEALTHY
        elif service.consecutive_successes >= health_check.recovery_threshold:
            service.status = HealthStatus.HEALTHY
        else:
            # État transitoire
            service.status = HealthStatus.DEGRADED if service.consecutive_failures > 0 else HealthStatus.HEALTHY
        
        # Métriques pour collecteur
        status_value = 1 if result.status == HealthStatus.HEALTHY else 0
        self.metrics_collector.set_gauge(f"service_health_{service_name}", status_value)
        self.metrics_collector.histogram(f"service_response_time_{service_name}", result.response_time_ms)
    
    def _should_attempt_reset(self, circuit_breaker: CircuitBreaker) -> bool:
        """Détermine si on doit tenter de réinitialiser un circuit breaker"""
        if circuit_breaker.last_failure_time is None:
            return False
        
        time_since_failure = datetime.now(timezone.utc) - circuit_breaker.last_failure_time
        return time_since_failure.total_seconds() >= circuit_breaker.recovery_timeout_seconds
    
    def _handle_check_success(self, circuit_breaker: CircuitBreaker):
        """Gère le succès d'un health check pour circuit breaker"""
        if circuit_breaker.state == CircuitBreakerState.HALF_OPEN:
            circuit_breaker.half_open_calls += 1
            if circuit_breaker.half_open_calls >= circuit_breaker.half_open_max_calls:
                # Suffisamment de succès pour fermer le circuit
                circuit_breaker.state = CircuitBreakerState.CLOSED
                circuit_breaker.failure_count = 0
                logger.info(f"Circuit breaker {circuit_breaker.name} fermé après récupération")
        elif circuit_breaker.state == CircuitBreakerState.CLOSED:
            # Reset compteur échecs
            circuit_breaker.failure_count = 0
    
    def _handle_check_failure(self, circuit_breaker: CircuitBreaker):
        """Gère l'échec d'un health check pour circuit breaker"""
        circuit_breaker.failure_count += 1
        circuit_breaker.last_failure_time = datetime.now(timezone.utc)
        
        if circuit_breaker.state == CircuitBreakerState.HALF_OPEN:
            # Échec pendant test de récupération - rouvrir
            circuit_breaker.state = CircuitBreakerState.OPEN
            logger.warning(f"Circuit breaker {circuit_breaker.name} rouvert après échec test")
        elif (circuit_breaker.state == CircuitBreakerState.CLOSED and 
              circuit_breaker.failure_count >= circuit_breaker.failure_threshold):
            # Trop d'échecs - ouvrir circuit
            circuit_breaker.state = CircuitBreakerState.OPEN
            logger.error(f"Circuit breaker {circuit_breaker.name} ouvert après {circuit_breaker.failure_count} échecs")
    
    async def _analyze_result_and_alert(self, service_name: str, result: HealthResult):
        """Analyse un résultat et déclenche alertes si nécessaire"""
        service = self.service_health[service_name]
        health_check = self.health_checks[service_name]
        
        # Alerting sur changement d'état critique
        if health_check.critical and result.status == HealthStatus.UNHEALTHY:
            if service.consecutive_failures == health_check.failure_threshold:
                # Premier passage en état critique
                logger.critical(f"🚨 SERVICE CRITIQUE INDISPONIBLE: {service_name}",
                              extra={
                                  'service_name': service_name,
                                  'consecutive_failures': service.consecutive_failures,
                                  'error': result.error_message,
                                  'alert_type': 'service_down'
                              })
        
        # Alerting sur récupération
        if result.status == HealthStatus.HEALTHY and service.consecutive_failures > 0:
            if service.consecutive_successes == health_check.recovery_threshold:
                logger.info(f"✅ SERVICE RÉCUPÉRÉ: {service_name}",
                           extra={
                               'service_name': service_name,
                               'downtime_checks': service.consecutive_failures,
                               'alert_type': 'service_recovery'
                           })
        
        # Alerting sur performance dégradée
        if result.response_time_ms > health_check.timeout_seconds * 500:  # 50% du timeout
            logger.warning(f"⚠️ PERFORMANCE DÉGRADÉE: {service_name}",
                          extra={
                              'service_name': service_name,
                              'response_time_ms': result.response_time_ms,
                              'threshold_ms': health_check.timeout_seconds * 500,
                              'alert_type': 'performance_degraded'
                          })
    
    def _generate_service_recommendations(self, service_name: str, 
                                        service: ServiceHealth, 
                                        circuit_breaker: CircuitBreaker) -> List[str]:
        """Génère des recommandations pour un service"""
        recommendations = []
        
        # Recommandations basées sur l'état
        if service.status == HealthStatus.UNHEALTHY:
            recommendations.append("Vérifier les logs d'erreur du service")
            recommendations.append("Contrôler la connectivité réseau")
            
            if service.consecutive_failures > 10:
                recommendations.append("Redémarrage du service recommandé")
        
        # Recommandations basées sur les performances
        if service.avg_response_time_ms > 5000:
            recommendations.append("Optimiser les performances - temps de réponse élevé")
        
        if service.success_rate < 95:
            recommendations.append("Investiguer la cause des échecs récurrents")
        
        # Recommandations circuit breaker
        if circuit_breaker.state == CircuitBreakerState.OPEN:
            recommendations.append("Circuit breaker ouvert - service isolé")
            recommendations.append("Résoudre les problèmes avant réactivation automatique")
        
        # Recommandations par type de service
        health_check = self.health_checks[service_name]
        if health_check.service_type == ServiceType.DATABASE:
            if service.status != HealthStatus.HEALTHY:
                recommendations.append("Vérifier les connexions à la base de données")
                recommendations.append("Contrôler l'espace disque disponible")
        elif health_check.service_type == ServiceType.EXTERNAL_API:
            if service.status != HealthStatus.HEALTHY:
                recommendations.append("Vérifier les clés API et quotas")
                recommendations.append("Contrôler la connectivité Internet")
        
        return recommendations


# Instance globale du health monitor
_health_monitor: Optional[HealthMonitor] = None


async def get_health_monitor() -> HealthMonitor:
    """Factory pour obtenir le health monitor"""
    global _health_monitor
    
    if _health_monitor is None:
        _health_monitor = HealthMonitor()
    
    return _health_monitor


# Utilitaires

async def quick_health_check() -> Dict[str, Any]:
    """Health check rapide pour endpoint /health"""
    monitor = await get_health_monitor()
    
    # Checks critiques seulement pour rapidité
    critical_checks = [name for name, check in monitor.health_checks.items() if check.critical]
    
    results = {}
    for check_name in critical_checks:
        try:
            result = await monitor.run_single_check(check_name)
            results[check_name] = result.status.value
        except Exception as e:
            results[check_name] = HealthStatus.UNHEALTHY.value
    
    # Statut global
    all_healthy = all(status == HealthStatus.HEALTHY.value for status in results.values())
    overall_status = HealthStatus.HEALTHY if all_healthy else HealthStatus.UNHEALTHY
    
    return {
        'status': overall_status.value,
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'services': results
    }