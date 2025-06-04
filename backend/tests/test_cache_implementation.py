"""
Tests complets pour US-002: Cache Redis multi-niveaux
Tests de validation de l'implémentation cache

Tests:
- Module cache principal
- Intégration scrapers cachés
- Scoring avec cache
- Monitoring et métriques
- Performance et hit ratio
"""

import pytest
import asyncio
import time
import json
from datetime import datetime, timedelta
from typing import Dict, Any

from app.core.cache import DistributedCache, CacheType, get_cache
from app.core.cache_monitoring import CacheMonitor, get_cache_monitor
from app.scrapers.cached_pappers import CachedPappersAPIClient
from app.scrapers.cached_kaspr import CachedKasprAPIClient
from app.services.cached_ma_scoring import CachedMAScoring


@pytest.fixture
async def cache_instance():
    """Fixture cache pour tests"""
    cache = DistributedCache("redis://localhost:6379")
    await cache.connect()
    yield cache
    await cache.disconnect()


@pytest.fixture
async def monitor_instance():
    """Fixture moniteur cache"""
    monitor = CacheMonitor()
    yield monitor


@pytest.fixture
def sample_company_data():
    """Données entreprise pour tests"""
    return {
        "siren": "123456789",
        "nom_entreprise": "Test Company SAS",
        "chiffre_affaires": 5000000,
        "effectif": 50,
        "resultat": 500000,
        "code_naf": "6920Z",
        "ville": "Paris",
        "code_postal": "75001"
    }


class TestCacheCore:
    """Tests du module cache principal"""
    
    async def test_cache_connection(self, cache_instance):
        """Test connexion Redis"""
        assert cache_instance._connected
        
        # Test ping
        health = await cache_instance.health_check()
        assert health["status"] == "healthy"
        assert health["ping_success"] is True
    
    async def test_cache_basic_operations(self, cache_instance):
        """Test opérations cache de base"""
        # Test SET
        key = "test_key"
        value = {"test": True, "timestamp": time.time()}
        
        result = await cache_instance.set(key, value, CacheType.API_EXTERNAL, ttl=60)
        assert result is True
        
        # Test GET
        retrieved = await cache_instance.get(key, CacheType.API_EXTERNAL)
        assert retrieved is not None
        assert retrieved["test"] is True
        
        # Test DELETE
        deleted = await cache_instance.delete(key, CacheType.API_EXTERNAL)
        assert deleted is True
        
        # Vérifier suppression
        retrieved_after_delete = await cache_instance.get(key, CacheType.API_EXTERNAL)
        assert retrieved_after_delete is None
    
    async def test_cache_ttl(self, cache_instance):
        """Test TTL et expiration"""
        key = "test_ttl"
        value = {"data": "test"}
        
        # Cache avec TTL court
        await cache_instance.set(key, value, CacheType.API_EXTERNAL, ttl=1)
        
        # Vérifier présence immédiate
        retrieved = await cache_instance.get(key, CacheType.API_EXTERNAL)
        assert retrieved is not None
        
        # Attendre expiration
        await asyncio.sleep(2)
        
        # Vérifier expiration
        expired = await cache_instance.get(key, CacheType.API_EXTERNAL)
        assert expired is None
    
    async def test_cache_compression(self, cache_instance):
        """Test compression des données"""
        # Données volumineuses pour déclencher compression
        large_data = {"data": "x" * 10000, "numbers": list(range(1000))}
        
        key = "test_compression"
        
        # Stocker avec compression
        result = await cache_instance.set(key, large_data, CacheType.ENRICHMENT_PAPPERS)
        assert result is True
        
        # Récupérer et vérifier intégrité
        retrieved = await cache_instance.get(key, CacheType.ENRICHMENT_PAPPERS)
        assert retrieved is not None
        assert retrieved["data"] == large_data["data"]
        assert len(retrieved["numbers"]) == 1000
    
    async def test_cache_get_or_compute(self, cache_instance):
        """Test pattern cache-aside"""
        compute_called = False
        
        async def expensive_computation():
            nonlocal compute_called
            compute_called = True
            await asyncio.sleep(0.1)  # Simulation calcul coûteux
            return {"computed": True, "timestamp": time.time()}
        
        key = "test_compute"
        
        # Premier appel - doit calculer
        result1 = await cache_instance.get_or_compute(
            key, expensive_computation, CacheType.API_EXTERNAL, ttl=60
        )
        assert compute_called is True
        assert result1["computed"] is True
        
        # Reset flag
        compute_called = False
        
        # Deuxième appel - doit utiliser cache
        result2 = await cache_instance.get_or_compute(
            key, expensive_computation, CacheType.API_EXTERNAL, ttl=60
        )
        assert compute_called is False  # Pas de calcul
        assert result2["computed"] is True
        assert result1["timestamp"] == result2["timestamp"]  # Même données
    
    async def test_cache_invalidation_pattern(self, cache_instance):
        """Test invalidation par pattern"""
        # Créer plusieurs clés avec pattern
        keys_data = [
            ("test:company:123", {"data": "company 123"}),
            ("test:company:456", {"data": "company 456"}),
            ("test:other:789", {"data": "other 789"})
        ]
        
        for key, data in keys_data:
            await cache_instance.set(key, data, CacheType.API_EXTERNAL, ttl=300)
        
        # Invalider pattern "company"
        deleted_count = await cache_instance.invalidate_pattern("test:company:*")
        assert deleted_count == 2
        
        # Vérifier invalidation sélective
        company_123 = await cache_instance.get("test:company:123", CacheType.API_EXTERNAL)
        company_456 = await cache_instance.get("test:company:456", CacheType.API_EXTERNAL)
        other_789 = await cache_instance.get("test:other:789", CacheType.API_EXTERNAL)
        
        assert company_123 is None
        assert company_456 is None
        assert other_789 is not None  # Pas invalidé
    
    async def test_cache_metrics(self, cache_instance):
        """Test collecte métriques"""
        # Générer activité cache
        for i in range(10):
            await cache_instance.set(f"metric_test_{i}", {"value": i}, CacheType.API_EXTERNAL)
        
        for i in range(5):
            await cache_instance.get(f"metric_test_{i}", CacheType.API_EXTERNAL)
        
        # Vérifier métriques
        metrics = cache_instance.metrics
        assert metrics.hits >= 5
        assert metrics.sets >= 10
        assert metrics.hit_ratio > 0
        
        # Test info cache
        cache_info = await cache_instance.get_cache_info()
        assert "redis_info" in cache_info
        assert "cache_metrics" in cache_info


class TestScrapersCache:
    """Tests intégration cache dans scrapers"""
    
    @pytest.fixture
    def mock_db_client(self):
        """Mock client DB pour tests"""
        class MockDB:
            def table(self, name):
                return self
            
            def select(self, fields):
                return self
            
            def execute(self):
                return type('MockResponse', (), {'data': []})()
        
        return MockDB()
    
    async def test_pappers_cache_integration(self, mock_db_client, sample_company_data):
        """Test cache Pappers"""
        async with CachedPappersAPIClient(mock_db_client) as client:
            # Test cache company details
            siren = sample_company_data["siren"]
            
            # Mock API response
            client._api_get_company_details = lambda s: sample_company_data
            
            # Premier appel - devrait mettre en cache
            result1 = await client.get_company_details_cached(siren)
            assert result1["siren"] == siren
            
            # Deuxième appel - devrait utiliser cache
            result2 = await client.get_company_details_cached(siren)
            assert result2 == result1
            
            # Vérifier statistiques cache
            stats = await client.get_cache_statistics()
            assert stats["session_stats"]["hits"] >= 1
    
    async def test_kaspr_cache_integration(self, sample_company_data):
        """Test cache Kaspr"""
        async with CachedKasprAPIClient() as client:
            # Forcer mode mock pour tests
            client.mock_mode = True
            
            # Test cache contacts
            contacts1 = await client.get_company_contacts_cached(sample_company_data)
            assert len(contacts1) > 0
            
            # Deuxième appel - devrait utiliser cache
            contacts2 = await client.get_company_contacts_cached(sample_company_data)
            assert len(contacts2) == len(contacts1)
            
            # Vérifier cache hit
            stats = await client.get_cache_statistics()
            assert stats["cache_hit_ratio"] > 0
    
    async def test_batch_operations_cache(self, mock_db_client, sample_company_data):
        """Test opérations par lot avec cache"""
        companies = [
            {**sample_company_data, "siren": f"12345678{i}"}
            for i in range(5)
        ]
        
        async with CachedPappersAPIClient(mock_db_client) as client:
            # Mock API pour chaque SIREN
            async def mock_api_call(siren):
                return {**sample_company_data, "siren": siren}
            
            client._api_get_company_details = mock_api_call
            
            # Premier batch - tout en cache
            results1 = await client.batch_get_companies_cached([c["siren"] for c in companies])
            assert len(results1) == 5
            
            # Deuxième batch - devrait utiliser cache
            results2 = await client.batch_get_companies_cached([c["siren"] for c in companies])
            assert results1 == results2
            
            # Vérifier économies
            stats = await client.get_cache_statistics()
            assert stats["session_stats"]["api_calls_saved"] > 0


class TestScoringCache:
    """Tests cache scoring M&A"""
    
    async def test_scoring_cache_basic(self, sample_company_data):
        """Test cache scoring de base"""
        scorer = CachedMAScoring()
        
        # Premier calcul
        result1 = await scorer.calculate_ma_score_cached(sample_company_data)
        assert result1.final_score > 0
        
        # Deuxième calcul - devrait utiliser cache
        result2 = await scorer.calculate_ma_score_cached(sample_company_data)
        assert result1.final_score == result2.final_score
        
        # Vérifier métadonnées cache
        assert result2.metadata.get("computation_time_saved") is True
    
    async def test_scoring_cache_invalidation(self, sample_company_data):
        """Test invalidation cache scoring"""
        scorer = CachedMAScoring()
        siren = sample_company_data["siren"]
        
        # Calculer score initial
        await scorer.calculate_ma_score_cached(sample_company_data)
        
        # Invalider cache entreprise
        invalidated = await scorer.invalidate_company_scoring(siren)
        assert invalidated > 0
        
        # Nouveau calcul après invalidation
        result_after = await scorer.calculate_ma_score_cached(sample_company_data)
        assert result_after.final_score > 0
    
    async def test_scoring_batch_cache(self, sample_company_data):
        """Test scoring par lot avec cache"""
        companies = [
            {**sample_company_data, "siren": f"12345678{i}", "chiffre_affaires": 1000000 * (i + 1)}
            for i in range(3)
        ]
        
        scorer = CachedMAScoring()
        
        # Premier batch
        results1 = await scorer.batch_score_companies_cached(companies)
        assert len(results1) == 3
        
        # Deuxième batch - devrait utiliser cache
        results2 = await scorer.batch_score_companies_cached(companies)
        
        # Vérifier cohérence
        for siren in results1.keys():
            assert results1[siren].final_score == results2[siren].final_score
        
        # Vérifier métriques
        metrics = await scorer.get_cache_performance_metrics()
        assert metrics["computations_saved_count"] > 0


class TestCacheMonitoring:
    """Tests monitoring cache"""
    
    async def test_monitor_initialization(self, monitor_instance):
        """Test initialisation moniteur"""
        assert monitor_instance.monitoring_active is False
        assert len(monitor_instance.alerts_history) == 0
        
        # Test configuration
        assert "hit_ratio_warning" in monitor_instance.alert_thresholds
        assert monitor_instance.config["collection_interval_seconds"] > 0
    
    async def test_cache_dashboard_data(self, monitor_instance):
        """Test génération données dashboard"""
        dashboard_data = await monitor_instance.get_cache_dashboard_data()
        
        assert "timestamp" in dashboard_data
        assert "overall_status" in dashboard_data
        assert "recommendations" in dashboard_data
        
        # Vérifier structure
        assert isinstance(dashboard_data["recommendations"], list)
    
    async def test_performance_report(self, monitor_instance):
        """Test rapport de performance"""
        report = await monitor_instance.get_performance_report(hours=1)
        
        assert "report_period_hours" in report
        assert "generated_at" in report
        assert "summary" in report
        
        # Vérifier métriques
        summary = report["summary"]
        assert "overall_hit_ratio" in summary
        assert "total_operations" in summary
    
    async def test_health_check(self, monitor_instance):
        """Test health check complet"""
        health_result = await monitor_instance.trigger_cache_health_check()
        
        assert "timestamp" in health_result
        assert "health_status" in health_result
        
        # Doit inclure tests performance si disponible
        if "performance_tests" in health_result:
            perf_tests = health_result["performance_tests"]
            assert "set_latency_ms" in perf_tests
            assert "get_latency_ms" in perf_tests


class TestCachePerformance:
    """Tests de performance cache"""
    
    async def test_cache_latency(self, cache_instance):
        """Test latence opérations cache"""
        data = {"test": "performance", "size": "medium"}
        
        # Mesurer latence SET
        start_time = time.time()
        await cache_instance.set("perf_test", data, CacheType.API_EXTERNAL)
        set_latency = (time.time() - start_time) * 1000
        
        # Mesurer latence GET
        start_time = time.time()
        await cache_instance.get("perf_test", CacheType.API_EXTERNAL)
        get_latency = (time.time() - start_time) * 1000
        
        # Assertions performance
        assert set_latency < 50  # < 50ms
        assert get_latency < 20  # < 20ms
    
    async def test_cache_throughput(self, cache_instance):
        """Test débit opérations cache"""
        operations_count = 100
        start_time = time.time()
        
        # Test throughput SET
        tasks = []
        for i in range(operations_count):
            task = cache_instance.set(f"throughput_{i}", {"value": i}, CacheType.API_EXTERNAL)
            tasks.append(task)
        
        await asyncio.gather(*tasks)
        
        # Test throughput GET
        tasks = []
        for i in range(operations_count):
            task = cache_instance.get(f"throughput_{i}", CacheType.API_EXTERNAL)
            tasks.append(task)
        
        await asyncio.gather(*tasks)
        
        total_time = time.time() - start_time
        ops_per_second = (operations_count * 2) / total_time  # SET + GET
        
        # Au moins 100 ops/sec
        assert ops_per_second > 100
    
    async def test_cache_hit_ratio_target(self, cache_instance):
        """Test atteinte objectif hit ratio"""
        # Simuler pattern d'utilisation réaliste
        
        # Phase 1: Populate cache
        for i in range(50):
            await cache_instance.set(f"popular_{i}", {"data": i}, CacheType.API_EXTERNAL, ttl=300)
        
        # Phase 2: Mix hits/misses (80% hits attendus)
        hits = 0
        total = 100
        
        for i in range(total):
            if i < 80:  # 80% de hits
                key = f"popular_{i % 50}"
            else:  # 20% de misses
                key = f"rare_{i}"
            
            result = await cache_instance.get(key, CacheType.API_EXTERNAL)
            if result is not None:
                hits += 1
        
        hit_ratio = (hits / total) * 100
        
        # Objectif: hit ratio > 75%
        assert hit_ratio > 75


class TestCacheIntegration:
    """Tests d'intégration complète"""
    
    async def test_full_workflow_cache(self, mock_db_client, sample_company_data):
        """Test workflow complet avec cache"""
        siren = sample_company_data["siren"]
        
        # 1. Enrichissement Pappers avec cache
        async with CachedPappersAPIClient(mock_db_client) as pappers:
            pappers._api_get_company_details = lambda s: sample_company_data
            company_details = await pappers.get_company_details_cached(siren)
        
        # 2. Enrichissement Kaspr avec cache
        async with CachedKasprAPIClient() as kaspr:
            kaspr.mock_mode = True
            contacts = await kaspr.get_company_contacts_cached(sample_company_data)
        
        # 3. Scoring avec cache
        scorer = CachedMAScoring()
        scoring_result = await scorer.calculate_ma_score_cached(sample_company_data)
        
        # 4. Vérifications
        assert company_details["siren"] == siren
        assert len(contacts) > 0
        assert scoring_result.final_score > 0
        
        # 5. Test invalidation cascade
        from app.services.cached_ma_scoring import invalidate_company_cache_cascade
        invalidated = await invalidate_company_cache_cascade(siren)
        assert invalidated > 0
    
    async def test_cache_api_endpoints(self):
        """Test endpoints API cache (simulation)"""
        # Simulation des endpoints sans serveur FastAPI
        cache = await get_cache()
        monitor = await get_cache_monitor()
        
        # Test équivalent /cache/status
        health = await cache.health_check()
        assert health["status"] in ["healthy", "degraded"]
        
        # Test équivalent /cache/metrics
        cache_info = await cache.get_cache_info()
        assert "cache_metrics" in cache_info
        
        # Test équivalent /cache/dashboard
        dashboard = await monitor.get_cache_dashboard_data()
        assert "overall_status" in dashboard


# Helpers et utilitaires tests

def generate_test_data(count: int = 10) -> list:
    """Génère données de test"""
    return [
        {
            "siren": f"12345678{i:01d}",
            "nom_entreprise": f"Test Company {i}",
            "chiffre_affaires": 1000000 * (i + 1),
            "effectif": 10 * (i + 1)
        }
        for i in range(count)
    ]


@pytest.mark.asyncio
async def test_cache_us002_validation():
    """Test de validation globale US-002"""
    
    # Critères de succès US-002
    success_criteria = {
        "cache_hit_ratio_target": 80,  # > 80%
        "api_calls_reduction_target": 80,  # 80% de réduction
        "response_time_target": 500,   # < 500ms
        "redis_stable_target": 1000,   # 1GB+ stable
    }
    
    # Simulation validation
    cache = await get_cache()
    
    # Test connexion stable
    health = await cache.health_check()
    assert health["status"] == "healthy"
    
    # Test performance basique
    start_time = time.time()
    await cache.set("validation_test", {"us": "002"}, CacheType.API_EXTERNAL)
    result = await cache.get("validation_test", CacheType.API_EXTERNAL)
    latency_ms = (time.time() - start_time) * 1000
    
    assert result is not None
    assert latency_ms < success_criteria["response_time_target"]
    
    print(f"✅ US-002 Validation: Latence {latency_ms:.2f}ms < {success_criteria['response_time_target']}ms")
    
    await cache.disconnect()


if __name__ == "__main__":
    # Exécution tests standalone
    asyncio.run(test_cache_us002_validation())