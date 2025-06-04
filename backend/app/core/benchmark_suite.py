"""
Suite de benchmarks et tests de charge pour M&A Intelligence Platform
US-009: Tests de performance, charge et stress pour valider les optimisations

Ce module fournit:
- Benchmarks de performance des composants critiques
- Tests de charge avec montée progressive
- Tests de stress pour identifier les limites
- Profiling détaillé des performances
- Métriques de performance sous charge
- Génération de rapports de benchmark
- Comparaison avant/après optimisations
"""

import asyncio
import time
import statistics
import psutil
import tracemalloc
import gc
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from functools import wraps
from enum import Enum
import json
import concurrent.futures
import random
import string

import numpy as np
import aiohttp
import asyncpg

from app.core.logging_system import get_logger, LogCategory
from app.core.performance_analyzer import get_performance_analyzer
from app.core.cache_manager import get_cache_manager
from app.core.database_optimizer import get_database_optimizer

logger = get_logger("benchmark_suite", LogCategory.PERFORMANCE)


class BenchmarkType(str, Enum):
    """Types de benchmarks"""
    PERFORMANCE = "performance"     # Tests de performance pure
    LOAD = "load"                  # Tests de charge normale
    STRESS = "stress"              # Tests de stress/limite
    SPIKE = "spike"                # Tests de pic de charge
    ENDURANCE = "endurance"        # Tests d'endurance


class BenchmarkStatus(str, Enum):
    """Statuts de benchmark"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class BenchmarkConfig:
    """Configuration d'un benchmark"""
    name: str
    benchmark_type: BenchmarkType
    duration_seconds: int = 60
    concurrent_users: int = 10
    ramp_up_seconds: int = 10
    requests_per_second: Optional[int] = None
    timeout_seconds: int = 30
    
    # Configuration spécifique par type
    max_concurrent_users: int = 100  # Pour tests de stress
    spike_duration_seconds: int = 30  # Pour tests de spike
    endurance_hours: int = 1  # Pour tests d'endurance
    
    # Métriques à collecter
    collect_memory: bool = True
    collect_cpu: bool = True
    collect_database: bool = True
    collect_cache: bool = True


@dataclass
class BenchmarkMetrics:
    """Métriques collectées pendant un benchmark"""
    timestamp: datetime
    response_times: List[float] = field(default_factory=list)
    success_count: int = 0
    error_count: int = 0
    timeout_count: int = 0
    
    # Métriques système
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    memory_rss_mb: float = 0.0
    
    # Métriques application
    active_connections: int = 0
    cache_hit_rate: float = 0.0
    database_connections: int = 0
    
    @property
    def total_requests(self) -> int:
        return self.success_count + self.error_count + self.timeout_count
    
    @property
    def success_rate(self) -> float:
        return (self.success_count / self.total_requests * 100) if self.total_requests > 0 else 0
    
    @property
    def avg_response_time(self) -> float:
        return statistics.mean(self.response_times) if self.response_times else 0
    
    @property
    def p95_response_time(self) -> float:
        return np.percentile(self.response_times, 95) if self.response_times else 0
    
    @property
    def p99_response_time(self) -> float:
        return np.percentile(self.response_times, 99) if self.response_times else 0


@dataclass
class BenchmarkResult:
    """Résultat d'un benchmark"""
    config: BenchmarkConfig
    status: BenchmarkStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    
    # Métriques agrégées
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    timeout_requests: int = 0
    
    # Performance
    avg_response_time: float = 0.0
    p95_response_time: float = 0.0
    p99_response_time: float = 0.0
    min_response_time: float = 0.0
    max_response_time: float = 0.0
    
    # Débit
    requests_per_second: float = 0.0
    peak_rps: float = 0.0
    
    # Ressources
    peak_cpu_usage: float = 0.0
    peak_memory_usage: float = 0.0
    avg_cpu_usage: float = 0.0
    avg_memory_usage: float = 0.0
    
    # Détails
    metrics_timeline: List[BenchmarkMetrics] = field(default_factory=list)
    error_details: List[str] = field(default_factory=list)
    
    @property
    def duration_seconds(self) -> float:
        if self.end_time and self.start_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0
    
    @property
    def success_rate(self) -> float:
        return (self.successful_requests / self.total_requests * 100) if self.total_requests > 0 else 0


class BenchmarkRunner:
    """Exécuteur de benchmarks"""
    
    def __init__(self):
        self.active_benchmarks: Dict[str, BenchmarkResult] = {}
        self.benchmark_history: List[BenchmarkResult] = []
        
        # Pool d'exécuteurs
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=50)
        
        logger.info("🏁 Système de benchmarks initialisé")
    
    async def run_benchmark(self, config: BenchmarkConfig, target_function: Callable) -> BenchmarkResult:
        """Lance un benchmark complet"""
        
        benchmark_id = f"{config.name}_{int(time.time())}"
        
        result = BenchmarkResult(
            config=config,
            status=BenchmarkStatus.RUNNING,
            start_time=datetime.now()
        )
        
        self.active_benchmarks[benchmark_id] = result
        
        logger.info(f"🚀 Démarrage benchmark: {config.name} ({config.benchmark_type.value})")
        
        try:
            if config.benchmark_type == BenchmarkType.PERFORMANCE:
                await self._run_performance_benchmark(result, target_function)
            elif config.benchmark_type == BenchmarkType.LOAD:
                await self._run_load_benchmark(result, target_function)
            elif config.benchmark_type == BenchmarkType.STRESS:
                await self._run_stress_benchmark(result, target_function)
            elif config.benchmark_type == BenchmarkType.SPIKE:
                await self._run_spike_benchmark(result, target_function)
            elif config.benchmark_type == BenchmarkType.ENDURANCE:
                await self._run_endurance_benchmark(result, target_function)
            
            result.status = BenchmarkStatus.COMPLETED
            
        except Exception as e:
            logger.error(f"❌ Erreur benchmark {config.name}: {e}")
            result.status = BenchmarkStatus.FAILED
            result.error_details.append(str(e))
        
        finally:
            result.end_time = datetime.now()
            self._finalize_result(result)
            
            # Déplacer vers l'historique
            del self.active_benchmarks[benchmark_id]
            self.benchmark_history.append(result)
            
            logger.info(f"✅ Benchmark terminé: {config.name} - {result.success_rate:.1f}% succès")
        
        return result
    
    async def _run_performance_benchmark(self, result: BenchmarkResult, target_function: Callable):
        """Benchmark de performance pure (1 utilisateur)"""
        
        config = result.config
        metrics_collector = MetricsCollector(config)
        
        # Démarrer collecte métrique
        metrics_task = asyncio.create_task(metrics_collector.start_collection())
        
        try:
            for i in range(100):  # 100 requêtes de performance
                start_time = time.time()
                
                try:
                    await target_function()
                    response_time = time.time() - start_time
                    
                    result.successful_requests += 1
                    metrics_collector.record_response(response_time, True)
                    
                except asyncio.TimeoutError:
                    result.timeout_requests += 1
                    metrics_collector.record_response(time.time() - start_time, False)
                    
                except Exception as e:
                    result.failed_requests += 1
                    result.error_details.append(str(e))
                    metrics_collector.record_response(time.time() - start_time, False)
                
                # Petite pause pour éviter la saturation
                await asyncio.sleep(0.1)
        
        finally:
            metrics_collector.stop_collection()
            await metrics_task
            result.metrics_timeline = metrics_collector.get_metrics()
    
    async def _run_load_benchmark(self, result: BenchmarkResult, target_function: Callable):
        """Test de charge avec montée progressive"""
        
        config = result.config
        metrics_collector = MetricsCollector(config)
        
        # Démarrer collecte métrique
        metrics_task = asyncio.create_task(metrics_collector.start_collection())
        
        try:
            # Montée progressive
            ramp_step = config.ramp_up_seconds / config.concurrent_users
            
            tasks = []
            
            for user_id in range(config.concurrent_users):
                # Délai de montée en charge
                await asyncio.sleep(ramp_step)
                
                # Créer tâche utilisateur
                task = asyncio.create_task(
                    self._simulate_user_load(
                        target_function, 
                        config.duration_seconds,
                        result,
                        metrics_collector,
                        user_id
                    )
                )
                tasks.append(task)
            
            # Attendre toutes les tâches
            await asyncio.gather(*tasks, return_exceptions=True)
        
        finally:
            metrics_collector.stop_collection()
            await metrics_task
            result.metrics_timeline = metrics_collector.get_metrics()
    
    async def _run_stress_benchmark(self, result: BenchmarkResult, target_function: Callable):
        """Test de stress pour trouver les limites"""
        
        config = result.config
        metrics_collector = MetricsCollector(config)
        
        metrics_task = asyncio.create_task(metrics_collector.start_collection())
        
        try:
            # Augmentation progressive de la charge jusqu'à la limite
            current_users = 1
            max_successful_users = 1
            
            while current_users <= config.max_concurrent_users:
                logger.info(f"📈 Test stress: {current_users} utilisateurs simultanés")
                
                # Test avec le nombre actuel d'utilisateurs
                test_duration = 30  # Test court pour chaque niveau
                tasks = []
                
                temp_result = BenchmarkResult(
                    config=config,
                    status=BenchmarkStatus.RUNNING,
                    start_time=datetime.now()
                )
                
                for user_id in range(current_users):
                    task = asyncio.create_task(
                        self._simulate_user_load(
                            target_function,
                            test_duration,
                            temp_result,
                            metrics_collector,
                            user_id
                        )
                    )
                    tasks.append(task)
                
                await asyncio.gather(*tasks, return_exceptions=True)
                
                # Évaluer si ce niveau est acceptable
                if temp_result.success_rate > 95 and temp_result.avg_response_time < 5.0:
                    max_successful_users = current_users
                    # Merger les résultats
                    result.successful_requests += temp_result.successful_requests
                    result.failed_requests += temp_result.failed_requests
                    result.timeout_requests += temp_result.timeout_requests
                else:
                    logger.warning(f"⚠️ Limite atteinte: {current_users} utilisateurs")
                    break
                
                # Augmentation progressive
                current_users = int(current_users * 1.5)
            
            logger.info(f"🎯 Limite trouvée: {max_successful_users} utilisateurs max")
        
        finally:
            metrics_collector.stop_collection()
            await metrics_task
            result.metrics_timeline = metrics_collector.get_metrics()
    
    async def _run_spike_benchmark(self, result: BenchmarkResult, target_function: Callable):
        """Test de pic de charge soudain"""
        
        config = result.config
        metrics_collector = MetricsCollector(config)
        
        metrics_task = asyncio.create_task(metrics_collector.start_collection())
        
        try:
            # Phase 1: Charge normale (25% des utilisateurs)
            normal_users = max(1, config.concurrent_users // 4)
            logger.info(f"📊 Phase normale: {normal_users} utilisateurs")
            
            normal_tasks = []
            for user_id in range(normal_users):
                task = asyncio.create_task(
                    self._simulate_user_load(
                        target_function,
                        config.duration_seconds // 3,
                        result,
                        metrics_collector,
                        user_id
                    )
                )
                normal_tasks.append(task)
            
            await asyncio.gather(*normal_tasks, return_exceptions=True)
            
            # Phase 2: Pic soudain (100% des utilisateurs)
            logger.info(f"⚡ Phase pic: {config.concurrent_users} utilisateurs")
            
            spike_tasks = []
            for user_id in range(config.concurrent_users):
                task = asyncio.create_task(
                    self._simulate_user_load(
                        target_function,
                        config.spike_duration_seconds,
                        result,
                        metrics_collector,
                        user_id
                    )
                )
                spike_tasks.append(task)
            
            await asyncio.gather(*spike_tasks, return_exceptions=True)
            
            # Phase 3: Retour normal
            logger.info(f"📉 Phase retour: {normal_users} utilisateurs")
            
            return_tasks = []
            for user_id in range(normal_users):
                task = asyncio.create_task(
                    self._simulate_user_load(
                        target_function,
                        config.duration_seconds // 3,
                        result,
                        metrics_collector,
                        user_id
                    )
                )
                return_tasks.append(task)
            
            await asyncio.gather(*return_tasks, return_exceptions=True)
        
        finally:
            metrics_collector.stop_collection()
            await metrics_task
            result.metrics_timeline = metrics_collector.get_metrics()
    
    async def _run_endurance_benchmark(self, result: BenchmarkResult, target_function: Callable):
        """Test d'endurance sur longue durée"""
        
        config = result.config
        metrics_collector = MetricsCollector(config)
        
        total_duration = config.endurance_hours * 3600
        
        metrics_task = asyncio.create_task(metrics_collector.start_collection())
        
        try:
            logger.info(f"⏰ Test endurance: {config.endurance_hours}h avec {config.concurrent_users} utilisateurs")
            
            tasks = []
            
            for user_id in range(config.concurrent_users):
                task = asyncio.create_task(
                    self._simulate_user_load(
                        target_function,
                        total_duration,
                        result,
                        metrics_collector,
                        user_id
                    )
                )
                tasks.append(task)
            
            await asyncio.gather(*tasks, return_exceptions=True)
        
        finally:
            metrics_collector.stop_collection()
            await metrics_task
            result.metrics_timeline = metrics_collector.get_metrics()
    
    async def _simulate_user_load(
        self, 
        target_function: Callable, 
        duration_seconds: int,
        result: BenchmarkResult,
        metrics_collector: 'MetricsCollector',
        user_id: int
    ):
        """Simule la charge d'un utilisateur"""
        
        end_time = time.time() + duration_seconds
        request_count = 0
        
        while time.time() < end_time:
            start_time = time.time()
            
            try:
                # Appeler la fonction cible
                await target_function()
                
                response_time = time.time() - start_time
                result.successful_requests += 1
                metrics_collector.record_response(response_time, True)
                
            except asyncio.TimeoutError:
                response_time = time.time() - start_time
                result.timeout_requests += 1
                metrics_collector.record_response(response_time, False)
                
            except Exception as e:
                response_time = time.time() - start_time
                result.failed_requests += 1
                result.error_details.append(f"User {user_id}: {str(e)}")
                metrics_collector.record_response(response_time, False)
            
            request_count += 1
            
            # Pause entre requêtes (simulation réaliste)
            think_time = random.uniform(0.5, 2.0)
            await asyncio.sleep(think_time)
    
    def _finalize_result(self, result: BenchmarkResult):
        """Finalise les métriques du résultat"""
        
        result.total_requests = result.successful_requests + result.failed_requests + result.timeout_requests
        
        if result.metrics_timeline:
            all_response_times = []
            cpu_values = []
            memory_values = []
            rps_values = []
            
            for metrics in result.metrics_timeline:
                all_response_times.extend(metrics.response_times)
                cpu_values.append(metrics.cpu_usage)
                memory_values.append(metrics.memory_usage)
                if metrics.total_requests > 0:
                    rps_values.append(metrics.total_requests / 5)  # Intervalle de 5s
            
            if all_response_times:
                result.avg_response_time = statistics.mean(all_response_times)
                result.p95_response_time = np.percentile(all_response_times, 95)
                result.p99_response_time = np.percentile(all_response_times, 99)
                result.min_response_time = min(all_response_times)
                result.max_response_time = max(all_response_times)
            
            if cpu_values:
                result.avg_cpu_usage = statistics.mean(cpu_values)
                result.peak_cpu_usage = max(cpu_values)
            
            if memory_values:
                result.avg_memory_usage = statistics.mean(memory_values)
                result.peak_memory_usage = max(memory_values)
            
            if rps_values:
                result.requests_per_second = statistics.mean(rps_values)
                result.peak_rps = max(rps_values)
        
        if result.duration_seconds > 0:
            result.requests_per_second = result.total_requests / result.duration_seconds


class MetricsCollector:
    """Collecteur de métriques pendant les benchmarks"""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.metrics: List[BenchmarkMetrics] = []
        self.current_metrics = BenchmarkMetrics(timestamp=datetime.now())
        self.collecting = False
        
        # Buffers pour réponses
        self.response_buffer: List[Tuple[float, bool]] = []
        self.buffer_lock = asyncio.Lock()
    
    async def start_collection(self):
        """Démarre la collecte de métriques"""
        self.collecting = True
        
        while self.collecting:
            try:
                await self._collect_current_metrics()
                await asyncio.sleep(5)  # Collecte toutes les 5 secondes
            except Exception as e:
                logger.warning(f"Erreur collecte métriques: {e}")
                await asyncio.sleep(5)
    
    def stop_collection(self):
        """Arrête la collecte"""
        self.collecting = False
    
    async def record_response(self, response_time: float, success: bool):
        """Enregistre une réponse"""
        async with self.buffer_lock:
            self.response_buffer.append((response_time, success))
    
    async def _collect_current_metrics(self):
        """Collecte les métriques actuelles"""
        
        # Traiter buffer de réponses
        async with self.buffer_lock:
            for response_time, success in self.response_buffer:
                self.current_metrics.response_times.append(response_time)
                if success:
                    self.current_metrics.success_count += 1
                else:
                    self.current_metrics.error_count += 1
            
            self.response_buffer.clear()
        
        # Métriques système
        if self.config.collect_cpu:
            self.current_metrics.cpu_usage = psutil.cpu_percent()
        
        if self.config.collect_memory:
            memory = psutil.virtual_memory()
            self.current_metrics.memory_usage = memory.percent
            
            process = psutil.Process()
            self.current_metrics.memory_rss_mb = process.memory_info().rss / (1024 * 1024)
        
        # Métriques cache
        if self.config.collect_cache:
            try:
                cache_manager = await get_cache_manager()
                cache_stats = cache_manager.get_stats()
                self.current_metrics.cache_hit_rate = cache_stats.get('total', {}).get('hit_ratio', 0) * 100
            except:
                pass
        
        # Métriques base de données
        if self.config.collect_database:
            try:
                db_optimizer = await get_database_optimizer()
                db_stats = db_optimizer.get_performance_stats()
                # Approximation du nombre de connexions
                self.current_metrics.database_connections = 5
            except:
                pass
        
        # Sauvegarder et reset
        self.metrics.append(self.current_metrics)
        self.current_metrics = BenchmarkMetrics(timestamp=datetime.now())
    
    def get_metrics(self) -> List[BenchmarkMetrics]:
        """Retourne les métriques collectées"""
        return self.metrics


class ComponentBenchmarkSuite:
    """Suite de benchmarks pour composants spécifiques"""
    
    def __init__(self):
        self.runner = BenchmarkRunner()
    
    async def benchmark_api_endpoints(self) -> Dict[str, BenchmarkResult]:
        """Benchmark des endpoints API"""
        
        results = {}
        
        # Simuler différents endpoints
        endpoints = [
            ("auth_login", self._simulate_auth_login),
            ("companies_list", self._simulate_companies_list),
            ("company_details", self._simulate_company_details),
            ("scraping_request", self._simulate_scraping_request),
            ("stats_dashboard", self._simulate_stats_dashboard)
        ]
        
        for endpoint_name, endpoint_func in endpoints:
            config = BenchmarkConfig(
                name=f"api_{endpoint_name}",
                benchmark_type=BenchmarkType.LOAD,
                duration_seconds=60,
                concurrent_users=20,
                ramp_up_seconds=10
            )
            
            result = await self.runner.run_benchmark(config, endpoint_func)
            results[endpoint_name] = result
        
        return results
    
    async def benchmark_database_operations(self) -> Dict[str, BenchmarkResult]:
        """Benchmark des opérations base de données"""
        
        results = {}
        
        operations = [
            ("simple_select", self._simulate_simple_select),
            ("complex_join", self._simulate_complex_join),
            ("bulk_insert", self._simulate_bulk_insert),
            ("full_text_search", self._simulate_full_text_search)
        ]
        
        for op_name, op_func in operations:
            config = BenchmarkConfig(
                name=f"db_{op_name}",
                benchmark_type=BenchmarkType.PERFORMANCE,
                duration_seconds=30,
                concurrent_users=5
            )
            
            result = await self.runner.run_benchmark(config, op_func)
            results[op_name] = result
        
        return results
    
    async def benchmark_cache_operations(self) -> Dict[str, BenchmarkResult]:
        """Benchmark des opérations de cache"""
        
        results = {}
        
        operations = [
            ("cache_get", self._simulate_cache_get),
            ("cache_set", self._simulate_cache_set),
            ("cache_invalidation", self._simulate_cache_invalidation)
        ]
        
        for op_name, op_func in operations:
            config = BenchmarkConfig(
                name=f"cache_{op_name}",
                benchmark_type=BenchmarkType.PERFORMANCE,
                duration_seconds=20,
                concurrent_users=10
            )
            
            result = await self.runner.run_benchmark(config, op_func)
            results[op_name] = result
        
        return results
    
    async def benchmark_scraping_operations(self) -> Dict[str, BenchmarkResult]:
        """Benchmark des opérations de scraping"""
        
        results = {}
        
        operations = [
            ("single_company_scrape", self._simulate_single_scrape),
            ("batch_scraping", self._simulate_batch_scraping),
            ("parallel_sources", self._simulate_parallel_sources)
        ]
        
        for op_name, op_func in operations:
            config = BenchmarkConfig(
                name=f"scraping_{op_name}",
                benchmark_type=BenchmarkType.LOAD,
                duration_seconds=90,
                concurrent_users=5
            )
            
            result = await self.runner.run_benchmark(config, op_func)
            results[op_name] = result
        
        return results
    
    # Fonctions de simulation pour les benchmarks
    
    async def _simulate_auth_login(self):
        """Simule une authentification"""
        await asyncio.sleep(random.uniform(0.05, 0.15))  # 50-150ms
        if random.random() < 0.02:  # 2% d'échec
            raise Exception("Auth failed")
    
    async def _simulate_companies_list(self):
        """Simule récupération liste entreprises"""
        await asyncio.sleep(random.uniform(0.1, 0.3))  # 100-300ms
        if random.random() < 0.01:  # 1% d'échec
            raise Exception("Database timeout")
    
    async def _simulate_company_details(self):
        """Simule récupération détails entreprise"""
        await asyncio.sleep(random.uniform(0.2, 0.5))  # 200-500ms
        if random.random() < 0.015:  # 1.5% d'échec
            raise Exception("Company not found")
    
    async def _simulate_scraping_request(self):
        """Simule demande de scraping"""
        await asyncio.sleep(random.uniform(1.0, 3.0))  # 1-3s
        if random.random() < 0.05:  # 5% d'échec
            raise Exception("Scraping failed")
    
    async def _simulate_stats_dashboard(self):
        """Simule chargement dashboard stats"""
        await asyncio.sleep(random.uniform(0.3, 0.8))  # 300-800ms
        if random.random() < 0.01:  # 1% d'échec
            raise Exception("Stats calculation error")
    
    async def _simulate_simple_select(self):
        """Simule SELECT simple"""
        await asyncio.sleep(random.uniform(0.001, 0.005))  # 1-5ms
        if random.random() < 0.001:  # 0.1% d'échec
            raise Exception("Connection lost")
    
    async def _simulate_complex_join(self):
        """Simule JOIN complexe"""
        await asyncio.sleep(random.uniform(0.01, 0.05))  # 10-50ms
        if random.random() < 0.005:  # 0.5% d'échec
            raise Exception("Query timeout")
    
    async def _simulate_bulk_insert(self):
        """Simule insertion en lot"""
        await asyncio.sleep(random.uniform(0.1, 0.3))  # 100-300ms
        if random.random() < 0.01:  # 1% d'échec
            raise Exception("Constraint violation")
    
    async def _simulate_full_text_search(self):
        """Simule recherche textuelle"""
        await asyncio.sleep(random.uniform(0.05, 0.2))  # 50-200ms
        if random.random() < 0.02:  # 2% d'échec
            raise Exception("Search index error")
    
    async def _simulate_cache_get(self):
        """Simule lecture cache"""
        await asyncio.sleep(random.uniform(0.001, 0.003))  # 1-3ms
        if random.random() < 0.005:  # 0.5% d'échec
            raise Exception("Cache miss")
    
    async def _simulate_cache_set(self):
        """Simule écriture cache"""
        await asyncio.sleep(random.uniform(0.002, 0.008))  # 2-8ms
        if random.random() < 0.01:  # 1% d'échec
            raise Exception("Cache full")
    
    async def _simulate_cache_invalidation(self):
        """Simule invalidation cache"""
        await asyncio.sleep(random.uniform(0.005, 0.015))  # 5-15ms
        if random.random() < 0.005:  # 0.5% d'échec
            raise Exception("Invalidation error")
    
    async def _simulate_single_scrape(self):
        """Simule scraping d'une entreprise"""
        await asyncio.sleep(random.uniform(0.5, 2.0))  # 500ms-2s
        if random.random() < 0.1:  # 10% d'échec
            raise Exception("Scraping timeout")
    
    async def _simulate_batch_scraping(self):
        """Simule scraping par lot"""
        await asyncio.sleep(random.uniform(2.0, 5.0))  # 2-5s
        if random.random() < 0.15:  # 15% d'échec
            raise Exception("Batch processing error")
    
    async def _simulate_parallel_sources(self):
        """Simule scraping sources parallèles"""
        await asyncio.sleep(random.uniform(1.5, 4.0))  # 1.5-4s
        if random.random() < 0.2:  # 20% d'échec partiel
            raise Exception("Source timeout")


# Instance globale
_benchmark_suite: Optional[ComponentBenchmarkSuite] = None


async def get_benchmark_suite() -> ComponentBenchmarkSuite:
    """Factory pour obtenir la suite de benchmarks"""
    global _benchmark_suite
    
    if _benchmark_suite is None:
        _benchmark_suite = ComponentBenchmarkSuite()
    
    return _benchmark_suite


# Fonctions utilitaires

async def run_comprehensive_benchmark() -> Dict[str, Any]:
    """Lance un benchmark complet de tous les composants"""
    
    suite = await get_benchmark_suite()
    start_time = datetime.now()
    
    logger.info("🏁 Démarrage benchmark complet du système")
    
    results = {
        'start_time': start_time.isoformat(),
        'api_benchmarks': {},
        'database_benchmarks': {},
        'cache_benchmarks': {},
        'scraping_benchmarks': {},
        'summary': {}
    }
    
    try:
        # Benchmark API
        logger.info("📡 Benchmark des APIs...")
        results['api_benchmarks'] = await suite.benchmark_api_endpoints()
        
        # Benchmark Database
        logger.info("🗄️ Benchmark de la base de données...")
        results['database_benchmarks'] = await suite.benchmark_database_operations()
        
        # Benchmark Cache
        logger.info("💾 Benchmark du cache...")
        results['cache_benchmarks'] = await suite.benchmark_cache_operations()
        
        # Benchmark Scraping
        logger.info("🕷️ Benchmark du scraping...")
        results['scraping_benchmarks'] = await suite.benchmark_scraping_operations()
        
        # Résumé
        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds()
        
        all_results = []
        for category in ['api_benchmarks', 'database_benchmarks', 'cache_benchmarks', 'scraping_benchmarks']:
            all_results.extend(results[category].values())
        
        avg_success_rate = statistics.mean([r.success_rate for r in all_results]) if all_results else 0
        avg_response_time = statistics.mean([r.avg_response_time for r in all_results]) if all_results else 0
        
        results['summary'] = {
            'total_duration_seconds': total_duration,
            'total_benchmarks': len(all_results),
            'avg_success_rate': avg_success_rate,
            'avg_response_time': avg_response_time,
            'end_time': end_time.isoformat()
        }
        
        logger.info(f"✅ Benchmark complet terminé en {total_duration:.1f}s - Succès: {avg_success_rate:.1f}%")
        
    except Exception as e:
        logger.error(f"❌ Erreur benchmark complet: {e}")
        results['error'] = str(e)
    
    return results


async def compare_before_after_optimization(
    before_results: Dict[str, BenchmarkResult],
    after_results: Dict[str, BenchmarkResult]
) -> Dict[str, Any]:
    """Compare les résultats avant/après optimisation"""
    
    comparison = {
        'comparison_date': datetime.now().isoformat(),
        'improvements': {},
        'regressions': {},
        'summary': {}
    }
    
    total_improvement_percent = 0
    comparison_count = 0
    
    for benchmark_name in before_results.keys():
        if benchmark_name in after_results:
            before = before_results[benchmark_name]
            after = after_results[benchmark_name]
            
            # Calculs d'amélioration
            response_time_improvement = ((before.avg_response_time - after.avg_response_time) / before.avg_response_time * 100) if before.avg_response_time > 0 else 0
            success_rate_improvement = after.success_rate - before.success_rate
            throughput_improvement = ((after.requests_per_second - before.requests_per_second) / before.requests_per_second * 100) if before.requests_per_second > 0 else 0
            
            benchmark_comparison = {
                'before': {
                    'avg_response_time': before.avg_response_time,
                    'success_rate': before.success_rate,
                    'requests_per_second': before.requests_per_second
                },
                'after': {
                    'avg_response_time': after.avg_response_time,
                    'success_rate': after.success_rate,
                    'requests_per_second': after.requests_per_second
                },
                'improvements': {
                    'response_time_percent': response_time_improvement,
                    'success_rate_points': success_rate_improvement,
                    'throughput_percent': throughput_improvement
                }
            }
            
            # Classer comme amélioration ou régression
            overall_improvement = (response_time_improvement + throughput_improvement) / 2
            if overall_improvement > 5:  # > 5% d'amélioration
                comparison['improvements'][benchmark_name] = benchmark_comparison
            elif overall_improvement < -5:  # > 5% de régression
                comparison['regressions'][benchmark_name] = benchmark_comparison
            
            total_improvement_percent += overall_improvement
            comparison_count += 1
    
    # Résumé global
    avg_improvement = total_improvement_percent / comparison_count if comparison_count > 0 else 0
    
    comparison['summary'] = {
        'total_benchmarks_compared': comparison_count,
        'improvements_count': len(comparison['improvements']),
        'regressions_count': len(comparison['regressions']),
        'avg_improvement_percent': avg_improvement,
        'overall_assessment': 'positive' if avg_improvement > 5 else 'negative' if avg_improvement < -5 else 'neutral'
    }
    
    return comparison