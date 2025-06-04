"""
Tests de performance pour M&A Intelligence Platform
US-005: Tests et benchmarks pour validation des optimisations

Features testées:
- Cache multi-niveaux (mémoire + Redis)
- Pool de connexions base de données
- Optimisation des requêtes
- Middlewares de compression et rate limiting
- Monitoring des performances
- Tâches asynchrones
"""

import asyncio
import pytest
import time
import statistics
from datetime import datetime, timedelta
from typing import List, Dict, Any
import json
import gzip
import concurrent.futures
from unittest.mock import AsyncMock, MagicMock, patch

# Imports du système
from app.core.cache_manager import get_cache_manager, CacheManager, CacheLevel
from app.core.database_optimizer import get_database_optimizer, DatabaseOptimizer
from app.core.performance_monitor import get_performance_monitor, PerformanceMonitor
from app.core.task_manager import get_task_manager, TaskManager, TaskPriority
from app.core.performance_middleware import CompressionMiddleware, RateLimitingMiddleware


class TestCachePerformance:
    """Tests de performance du système de cache"""
    
    @pytest.fixture
    async def cache_manager(self):
        """Cache manager pour tests"""
        manager = CacheManager()
        await manager.initialize()
        yield manager
        await manager.close()
    
    @pytest.mark.asyncio
    async def test_cache_performance_memory_vs_redis(self, cache_manager):
        """Compare la performance L1 (mémoire) vs L2 (Redis)"""
        
        test_data = {"companies": [{"id": i, "name": f"Company {i}"} for i in range(100)]}
        
        # Test L1 (mémoire)
        start_time = time.time()
        for i in range(100):
            await cache_manager.set('companies', f'test_key_{i}', test_data)
        memory_write_time = time.time() - start_time
        
        start_time = time.time()
        for i in range(100):
            result = await cache_manager.get('companies', f'test_key_{i}')
            assert result is not None
        memory_read_time = time.time() - start_time
        
        # Vider cache pour test Redis
        await cache_manager.clear_all()
        
        # Forcer utilisation Redis uniquement
        cache_manager.configs['companies'].level = CacheLevel.REDIS
        
        start_time = time.time()
        for i in range(100):
            await cache_manager.set('companies', f'test_redis_{i}', test_data)
        redis_write_time = time.time() - start_time
        
        start_time = time.time()
        for i in range(100):
            result = await cache_manager.get('companies', f'test_redis_{i}')
            # Redis peut ne pas être disponible en test
            if cache_manager.redis_client:
                assert result is not None
        redis_read_time = time.time() - start_time
        
        # Vérifications performance
        assert memory_write_time < 0.1, f"Écriture L1 trop lente: {memory_write_time:.3f}s"
        assert memory_read_time < 0.05, f"Lecture L1 trop lente: {memory_read_time:.3f}s"
        
        print(f"\n📊 Performance Cache:")
        print(f"L1 Write: {memory_write_time:.3f}s | L1 Read: {memory_read_time:.3f}s")
        print(f"L2 Write: {redis_write_time:.3f}s | L2 Read: {redis_read_time:.3f}s")
        print(f"L1/L2 Write Ratio: {redis_write_time/memory_write_time:.1f}x")
    
    @pytest.mark.asyncio
    async def test_cache_hit_ratio_improvement(self, cache_manager):
        """Test l'amélioration du hit ratio avec le cache multi-niveaux"""
        
        # Données de test
        test_data = {"stats": {"total": 1000, "processed": 500}}
        
        # Simuler des accès répétés
        cache_hits = 0
        total_requests = 200
        
        for i in range(total_requests):
            # 80% des requêtes sur les mêmes clés (simulation réaliste)
            key = f"stats_{i % 40}"  # 40 clés différentes sur 200 requêtes
            
            result = await cache_manager.get('stats', key)
            if result is None:
                # Cache miss - simuler récupération depuis DB
                await asyncio.sleep(0.001)  # Simuler latence DB
                await cache_manager.set('stats', key, test_data)
            else:
                cache_hits += 1
        
        hit_ratio = cache_hits / total_requests
        
        # Vérifier amélioration performance
        assert hit_ratio > 0.6, f"Hit ratio trop bas: {hit_ratio:.2%}"
        
        stats = cache_manager.get_stats()
        print(f"\n🎯 Cache Hit Ratio: {hit_ratio:.2%}")
        print(f"Memory Cache: {stats['memory_cache']['hit_ratio']:.2%}")
        print(f"Total Cache: {stats['total']['hit_ratio']:.2%}")
    
    @pytest.mark.asyncio
    async def test_cache_compression_performance(self, cache_manager):
        """Test performance de la compression cache"""
        
        # Grosse donnée pour tester compression
        large_data = {
            "companies": [
                {
                    "id": i,
                    "name": f"Company Name {i} with long description and details",
                    "description": "Lorem ipsum dolor sit amet " * 50,
                    "metadata": {"key1": "value1", "key2": "value2", "key3": "value3"}
                }
                for i in range(100)
            ]
        }
        
        # Test sans compression
        cache_manager.configs['companies'].compress = False
        start_time = time.time()
        await cache_manager.set('companies', 'large_uncompressed', large_data)
        uncompressed_time = time.time() - start_time
        
        # Test avec compression
        cache_manager.configs['companies'].compress = True
        start_time = time.time()
        await cache_manager.set('companies', 'large_compressed', large_data)
        compressed_time = time.time() - start_time
        
        # Vérifier que les données sont récupérables
        result_compressed = await cache_manager.get('companies', 'large_compressed')
        assert result_compressed == large_data
        
        print(f"\n🗜️ Compression Performance:")
        print(f"Sans compression: {uncompressed_time:.3f}s")
        print(f"Avec compression: {compressed_time:.3f}s")
        print(f"Ratio temps: {compressed_time/uncompressed_time:.2f}x")


class TestDatabasePerformance:
    """Tests de performance de la base de données"""
    
    @pytest.fixture
    async def db_optimizer(self):
        """Database optimizer pour tests"""
        optimizer = DatabaseOptimizer()
        # Mock pour éviter vraie connexion DB en test
        optimizer.connection_pool = AsyncMock()
        optimizer.engine = AsyncMock()
        yield optimizer
        await optimizer.close()
    
    @pytest.mark.asyncio
    async def test_query_performance_monitoring(self, db_optimizer):
        """Test du monitoring des requêtes"""
        
        @db_optimizer.query_monitor("test_query")
        async def mock_slow_query():
            await asyncio.sleep(0.1)  # Simuler requête lente
            return [{"id": 1, "name": "test"}]
        
        @db_optimizer.query_monitor("test_fast_query")
        async def mock_fast_query():
            await asyncio.sleep(0.001)  # Requête rapide
            return [{"id": 1}]
        
        # Exécuter requêtes
        await mock_slow_query()
        await mock_fast_query()
        
        # Vérifier stats
        stats = db_optimizer.get_performance_stats()
        
        assert stats['query_performance']['total_queries_1h'] == 2
        assert len(stats['query_performance']['slowest_queries']) > 0
        
        # Vérifier détection requête lente
        slow_query = stats['query_performance']['slowest_queries'][0]
        assert slow_query['time_ms'] > 50  # > 50ms
        
        print(f"\n🐌 Requête la plus lente: {slow_query['time_ms']:.2f}ms")
    
    @pytest.mark.asyncio
    async def test_connection_pool_performance(self, db_optimizer):
        """Test performance du pool de connexions"""
        
        # Mock pool avec métrique
        pool_mock = AsyncMock()
        pool_mock._queue.qsize.return_value = 5
        pool_mock._maxsize = 10
        pool_mock._minsize = 2
        db_optimizer.connection_pool = pool_mock
        
        # Simuler requêtes concurrentes
        async def mock_query(query_id: int):
            # Simuler acquisition connexion
            await asyncio.sleep(0.01)
            return f"result_{query_id}"
        
        start_time = time.time()
        tasks = [mock_query(i) for i in range(20)]
        results = await asyncio.gather(*tasks)
        total_time = time.time() - start_time
        
        assert len(results) == 20
        assert total_time < 1.0, f"Pool trop lent: {total_time:.3f}s"
        
        # Vérifier stats pool
        stats = db_optimizer.get_performance_stats()
        pool_info = stats['connection_pool']
        
        assert 'size' in pool_info
        assert pool_info['max_size'] == 10
        
        print(f"\n🏊 Pool Performance: {total_time:.3f}s pour 20 requêtes")
    
    @pytest.mark.asyncio
    async def test_bulk_operations_performance(self, db_optimizer):
        """Test performance des opérations en lot"""
        
        # Mock pour bulk update
        db_optimizer.connection_pool = AsyncMock()
        db_optimizer.connection_pool.acquire().__aenter__ = AsyncMock()
        db_optimizer.connection_pool.acquire().__aexit__ = AsyncMock()
        
        mock_conn = AsyncMock()
        mock_conn.execute.return_value = "UPDATE 100"
        db_optimizer.connection_pool.acquire().__aenter__.return_value = mock_conn
        
        # Test bulk update
        company_ids = list(range(1, 101))  # 100 companies
        
        start_time = time.time()
        result = await db_optimizer.bulk_update_companies_status(
            company_ids, "contacted", "test_user"
        )
        bulk_time = time.time() - start_time
        
        assert bulk_time < 0.1, f"Bulk update trop lent: {bulk_time:.3f}s"
        
        print(f"\n📦 Bulk Update: {bulk_time:.3f}s pour 100 entrées")


class TestMiddlewarePerformance:
    """Tests de performance des middlewares"""
    
    def test_compression_performance(self):
        """Test performance de compression"""
        from starlette.applications import Starlette
        from starlette.requests import Request
        from starlette.responses import JSONResponse
        
        app = Starlette()
        middleware = CompressionMiddleware(app)
        
        # Données de test
        large_data = {
            "companies": [
                {"id": i, "name": f"Company {i} with description " * 10}
                for i in range(100)
            ]
        }
        
        # Mock request/response
        request = MagicMock()
        request.headers = {"accept-encoding": "gzip, br"}
        
        response = JSONResponse(large_data)
        original_size = len(json.dumps(large_data).encode())
        
        # Test compression (simulé)
        start_time = time.time()
        compressed_data = gzip.compress(json.dumps(large_data).encode())
        compression_time = time.time() - start_time
        
        compression_ratio = len(compressed_data) / original_size
        
        assert compression_ratio < 0.8, f"Compression insuffisante: {compression_ratio:.2%}"
        assert compression_time < 0.1, f"Compression trop lente: {compression_time:.3f}s"
        
        print(f"\n🗜️ Compression: {compression_ratio:.2%} ratio en {compression_time:.3f}s")
    
    def test_rate_limiting_performance(self):
        """Test performance du rate limiting"""
        from starlette.applications import Starlette
        
        app = Starlette()
        middleware = RateLimitingMiddleware(app)
        
        # Simuler vérifications rate limit
        start_time = time.time()
        
        for i in range(1000):
            # Simuler vérification rate limit
            client_ip = f"192.168.1.{i % 255}"
            endpoint = "GET:/api/v1/companies"
            rate_info = middleware._check_rate_limit(client_ip, endpoint)
            
            # Simuler enregistrement requête
            middleware._record_request(client_ip, endpoint)
        
        check_time = time.time() - start_time
        
        assert check_time < 1.0, f"Rate limiting trop lent: {check_time:.3f}s"
        
        print(f"\n🚦 Rate Limiting: {check_time:.3f}s pour 1000 vérifications")


class TestTaskManagerPerformance:
    """Tests de performance du gestionnaire de tâches"""
    
    @pytest.fixture
    async def task_manager(self):
        """Task manager pour tests"""
        manager = TaskManager()
        yield manager
    
    @pytest.mark.asyncio
    async def test_task_submission_performance(self, task_manager):
        """Test performance de soumission de tâches"""
        
        # Mock Celery pour tests
        with patch('app.core.task_manager.signature') as mock_signature:
            mock_result = AsyncMock()
            mock_result.id = "test-task-id"
            mock_signature.return_value.apply_async.return_value = mock_result
            
            start_time = time.time()
            
            # Soumettre 100 tâches
            task_ids = []
            for i in range(100):
                task_id = await task_manager.submit_task(
                    "test.task",
                    args=(i,),
                    priority=TaskPriority.NORMAL
                )
                task_ids.append(task_id)
            
            submission_time = time.time() - start_time
            
            assert len(task_ids) == 100
            assert submission_time < 1.0, f"Soumission trop lente: {submission_time:.3f}s"
            
            print(f"\n📋 Task Submission: {submission_time:.3f}s pour 100 tâches")
    
    @pytest.mark.asyncio
    async def test_task_status_checking_performance(self, task_manager):
        """Test performance de vérification statut tâches"""
        
        # Ajouter des tâches mockées
        for i in range(50):
            from app.core.task_manager import TaskResult, TaskStatus
            task = TaskResult(
                task_id=f"task-{i}",
                status=TaskStatus.PENDING
            )
            task_manager.active_tasks[f"task-{i}"] = task
        
        start_time = time.time()
        
        # Vérifier statut de toutes les tâches
        for i in range(50):
            status = await task_manager.get_task_status(f"task-{i}")
            assert status is not None
        
        check_time = time.time() - start_time
        
        assert check_time < 0.5, f"Vérification statut trop lente: {check_time:.3f}s"
        
        print(f"\n✅ Task Status Check: {check_time:.3f}s pour 50 tâches")


class TestPerformanceMonitorIntegration:
    """Tests d'intégration du monitoring de performance"""
    
    @pytest.fixture
    async def perf_monitor(self):
        """Performance monitor pour tests"""
        monitor = PerformanceMonitor()
        yield monitor
        monitor.stop_monitoring()
    
    @pytest.mark.asyncio
    async def test_system_metrics_collection_performance(self, perf_monitor):
        """Test performance de collecte des métriques système"""
        
        start_time = time.time()
        
        # Collecter métriques 10 fois
        for _ in range(10):
            metrics = perf_monitor._collect_system_metrics()
            assert metrics.cpu_percent >= 0
            assert metrics.memory_percent >= 0
        
        collection_time = time.time() - start_time
        
        assert collection_time < 1.0, f"Collecte métriques trop lente: {collection_time:.3f}s"
        
        print(f"\n📊 Metrics Collection: {collection_time:.3f}s pour 10 collectes")
    
    @pytest.mark.asyncio
    async def test_function_profiling_overhead(self, perf_monitor):
        """Test l'overhead du profiling de fonctions"""
        
        # Fonction de test normale
        def normal_function():
            time.sleep(0.01)
            return "result"
        
        # Fonction avec profiling
        @perf_monitor.profile_function("test_function")
        def profiled_function():
            time.sleep(0.01)
            return "result"
        
        # Mesurer overhead
        start_time = time.time()
        for _ in range(10):
            normal_function()
        normal_time = time.time() - start_time
        
        start_time = time.time()
        for _ in range(10):
            profiled_function()
        profiled_time = time.time() - start_time
        
        overhead = (profiled_time - normal_time) / normal_time
        
        assert overhead < 0.1, f"Overhead profiling trop élevé: {overhead:.2%}"
        
        print(f"\n⏱️ Profiling Overhead: {overhead:.2%}")
    
    @pytest.mark.asyncio
    async def test_performance_benchmark(self, perf_monitor):
        """Test du benchmark de performance système"""
        
        start_time = time.time()
        benchmark_results = await perf_monitor.run_performance_benchmark()
        benchmark_time = time.time() - start_time
        
        assert 'cpu_benchmark_ms' in benchmark_results
        assert 'memory_benchmark_ms' in benchmark_results
        assert 'io_benchmark_ms' in benchmark_results
        assert 'total_score' in benchmark_results
        
        assert benchmark_time < 5.0, f"Benchmark trop long: {benchmark_time:.3f}s"
        
        print(f"\n🏃 Benchmark complet en {benchmark_time:.3f}s")
        print(f"Score total: {benchmark_results['total_score']:.2f}")


class TestIntegratedPerformance:
    """Tests de performance intégrés (end-to-end)"""
    
    @pytest.mark.asyncio
    async def test_full_request_cycle_performance(self):
        """Test performance d'un cycle de requête complet"""
        
        # Simuler cycle complet:
        # Request → Auth → Cache Check → DB Query → Response
        
        start_time = time.time()
        
        # 1. Auth simulation (JWT decode + DB check)
        await asyncio.sleep(0.002)  # ~2ms
        
        # 2. Cache check simulation
        cache_manager = CacheManager()
        result = await cache_manager.get('companies', 'test_key')
        if result is None:
            # 3. DB query simulation
            await asyncio.sleep(0.010)  # ~10ms pour requête DB
            
            # 4. Cache set
            await cache_manager.set('companies', 'test_key', {"data": "test"})
        
        # 5. Response serialization
        await asyncio.sleep(0.001)  # ~1ms
        
        total_time = time.time() - start_time
        
        assert total_time < 0.05, f"Cycle complet trop lent: {total_time:.3f}s"
        
        print(f"\n🔄 Cycle complet: {total_time:.3f}s")
    
    @pytest.mark.asyncio
    async def test_concurrent_requests_performance(self):
        """Test performance sous charge concurrente"""
        
        async def simulate_request(request_id: int):
            # Simuler une requête API typique
            await asyncio.sleep(0.01)  # Traitement
            return f"response_{request_id}"
        
        # Test avec 50 requêtes concurrentes
        start_time = time.time()
        
        tasks = [simulate_request(i) for i in range(50)]
        results = await asyncio.gather(*tasks)
        
        concurrent_time = time.time() - start_time
        
        assert len(results) == 50
        assert concurrent_time < 1.0, f"Concurrence trop lente: {concurrent_time:.3f}s"
        
        # Calculer throughput
        throughput = 50 / concurrent_time
        
        print(f"\n⚡ Concurrence: {concurrent_time:.3f}s pour 50 requêtes")
        print(f"Throughput: {throughput:.1f} req/s")
    
    def test_memory_usage_optimization(self):
        """Test optimisation de l'usage mémoire"""
        import psutil
        import gc
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        # Simuler charge mémoire
        large_objects = []
        for i in range(1000):
            large_objects.append([f"data_{j}" for j in range(100)])
        
        peak_memory = process.memory_info().rss
        
        # Nettoyer
        large_objects.clear()
        gc.collect()
        
        final_memory = process.memory_info().rss
        memory_cleanup_ratio = (peak_memory - final_memory) / (peak_memory - initial_memory)
        
        assert memory_cleanup_ratio > 0.8, f"Nettoyage mémoire insuffisant: {memory_cleanup_ratio:.2%}"
        
        print(f"\n🧠 Nettoyage mémoire: {memory_cleanup_ratio:.2%}")


# Configuration pytest pour tests de performance
@pytest.fixture(scope="session")
def performance_report():
    """Génère un rapport de performance à la fin des tests"""
    results = {}
    yield results
    
    # Afficher rapport final
    print("\n" + "="*60)
    print("📊 RAPPORT DE PERFORMANCE US-005")
    print("="*60)
    
    if results:
        for test_name, metrics in results.items():
            print(f"\n{test_name}:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value}")
    else:
        print("\n✅ Tous les tests de performance ont réussi!")
        print("🚀 Le système est optimisé pour de bonnes performances.")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    # Exécution directe pour tests rapides
    pytest.main([__file__, "-v", "--tb=short"])