#!/usr/bin/env python3
"""
Script de test de performance PostgreSQL pour M&A Intelligence Platform
Tests des optimisations US-001 : connection pooling, indexes, requêtes

Usage:
    python scripts/test_database_performance.py
    python scripts/test_database_performance.py --full-test
    python scripts/test_database_performance.py --baseline
"""

import asyncio
import time
import statistics
import argparse
import logging
from typing import List, Dict, Any
from contextlib import asynccontextmanager
import json
from datetime import datetime, timedelta
import random

# Configuration logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from sqlalchemy import text, create_engine
    from sqlalchemy.pool import QueuePool
    import asyncpg
    from app.core.database import get_async_db_session, async_engine, engine
    from app.models.company import Company
    from app.models.user import User
except ImportError as e:
    logger.error(f"Import error: {e}")
    logger.error("Make sure to run from backend directory: cd backend && python scripts/test_database_performance.py")
    exit(1)


class DatabasePerformanceTester:
    """Testeur de performance pour optimisations PostgreSQL"""
    
    def __init__(self):
        self.results = {}
        self.test_data_created = False
        
    async def run_all_tests(self, full_test: bool = False, baseline: bool = False):
        """Lance tous les tests de performance"""
        logger.info("🚀 Démarrage des tests de performance PostgreSQL")
        
        # Test connexion base
        if not await self.test_database_connection():
            logger.error("❌ Connexion base échouée - arrêt des tests")
            return
        
        # Créer données de test si nécessaire
        if not baseline:
            await self.setup_test_data(full_test)
        
        # Tests de base
        await self.test_connection_pool_performance()
        await self.test_index_performance()
        await self.test_query_optimization()
        
        if full_test:
            await self.test_concurrent_connections()
            await self.test_large_dataset_queries()
            await self.test_partitioning_benefits()
        
        # Monitoring et statistiques
        await self.collect_database_stats()
        await self.analyze_slow_queries()
        
        # Génération rapport
        self.generate_performance_report()
        
        logger.info("✅ Tests de performance terminés")
    
    async def test_database_connection(self) -> bool:
        """Test de connexion basique"""
        try:
            async with get_async_db_session() as session:
                result = await session.execute(text("SELECT version()"))
                version = result.scalar()
                logger.info(f"✅ Connexion PostgreSQL OK: {version}")
                
                # Vérifier extensions
                result = await session.execute(text("SELECT name FROM pg_available_extensions WHERE installed_version IS NOT NULL"))
                extensions = [row[0] for row in result.fetchall()]
                logger.info(f"📊 Extensions installées: {', '.join(extensions)}")
                
                return True
        except Exception as e:
            logger.error(f"❌ Erreur connexion base: {e}")
            return False
    
    async def setup_test_data(self, full_test: bool = False):
        """Création données de test pour les benchmarks"""
        target_companies = 10000 if full_test else 1000
        
        try:
            async with get_async_db_session() as session:
                # Vérifier si données déjà présentes
                result = await session.execute(text("SELECT COUNT(*) FROM companies"))
                current_count = result.scalar()
                
                if current_count >= target_companies:
                    logger.info(f"📊 Données suffisantes présentes: {current_count} entreprises")
                    return
                
                logger.info(f"🔄 Création de {target_companies - current_count} entreprises de test...")
                
                # Générer données de test
                companies_to_create = target_companies - current_count
                batch_size = 500
                
                for batch_start in range(0, companies_to_create, batch_size):
                    batch_end = min(batch_start + batch_size, companies_to_create)
                    
                    # Préparer le batch d'insertion
                    companies_batch = []
                    for i in range(batch_start, batch_end):
                        company_data = self._generate_test_company(current_count + i)
                        companies_batch.append(company_data)
                    
                    # Insertion batch
                    insert_query = text("""
                        INSERT INTO companies (
                            siren, siret, nom_entreprise, forme_juridique, 
                            adresse, ville, code_postal, chiffre_affaires, 
                            effectif, code_naf, ma_score, last_scraped_at, created_at
                        ) VALUES (
                            :siren, :siret, :nom_entreprise, :forme_juridique,
                            :adresse, :ville, :code_postal, :chiffre_affaires,
                            :effectif, :code_naf, :ma_score, :last_scraped_at, :created_at
                        )
                    """)
                    
                    await session.execute(insert_query, companies_batch)
                    await session.commit()
                    
                    logger.info(f"✅ Batch {batch_end}/{companies_to_create} inséré")
                
                logger.info(f"✅ Données de test créées: {target_companies} entreprises")
                self.test_data_created = True
                
        except Exception as e:
            logger.error(f"❌ Erreur création données test: {e}")
            raise
    
    def _generate_test_company(self, index: int) -> Dict[str, Any]:
        """Génère une entreprise de test réaliste"""
        base_siren = 100000000 + index
        
        villes = ['Paris', 'Lyon', 'Marseille', 'Toulouse', 'Nice', 'Nantes', 'Strasbourg', 'Montpellier']
        formes_juridiques = ['SARL', 'SAS', 'SA', 'EURL', 'SNC']
        codes_naf = ['6920Z', '7022Z', '6619B', '7010Z', '6202A', '6311Z']
        
        # Score M&A avec distribution réaliste
        ma_score = max(0, min(100, random.normalvariate(55, 25)))
        
        # CA avec distribution log-normale
        ca_base = random.lognormvariate(15, 1.5)  # Moyenne ~3M€
        chiffre_affaires = int(ca_base * 1000)
        
        # Date de scraping récente avec distribution
        days_ago = random.expovariate(0.1)  # Plus de données récentes
        last_scraped = datetime.now() - timedelta(days=min(days_ago, 365))
        
        return {
            'siren': str(base_siren),
            'siret': f"{base_siren}00{index % 100:02d}",
            'nom_entreprise': f"Entreprise Test {index:06d}",
            'forme_juridique': random.choice(formes_juridiques),
            'adresse': f"{random.randint(1, 200)} Rue de Test",
            'ville': random.choice(villes),
            'code_postal': f"{random.randint(1000, 99000):05d}",
            'chiffre_affaires': chiffre_affaires,
            'effectif': random.randint(1, 500),
            'code_naf': random.choice(codes_naf),
            'ma_score': round(ma_score, 1),
            'last_scraped_at': last_scraped,
            'created_at': datetime.now()
        }
    
    async def test_connection_pool_performance(self):
        """Test performance du pool de connexions"""
        logger.info("🔄 Test performance connection pool...")
        
        async def single_query():
            async with get_async_db_session() as session:
                result = await session.execute(text("SELECT COUNT(*) FROM companies"))
                return result.scalar()
        
        # Test séquentiel
        start_time = time.time()
        for _ in range(10):
            await single_query()
        sequential_time = time.time() - start_time
        
        # Test concurrent
        start_time = time.time()
        tasks = [single_query() for _ in range(10)]
        await asyncio.gather(*tasks)
        concurrent_time = time.time() - start_time
        
        speedup = sequential_time / concurrent_time
        
        self.results['connection_pool'] = {
            'sequential_time': round(sequential_time, 3),
            'concurrent_time': round(concurrent_time, 3),
            'speedup': round(speedup, 2),
            'status': 'GOOD' if speedup > 2 else 'POOR'
        }
        
        logger.info(f"📊 Pool performance: {speedup:.2f}x speedup (concurrent vs sequential)")
    
    async def test_index_performance(self):
        """Test efficacité des index"""
        logger.info("🔄 Test performance des index...")
        
        queries = [
            {
                'name': 'search_by_ma_score',
                'query': "SELECT * FROM companies WHERE ma_score >= 80 ORDER BY ma_score DESC LIMIT 50",
                'expected_index': 'idx_companies_ma_search'
            },
            {
                'name': 'search_by_location',
                'query': "SELECT * FROM companies WHERE ville = 'Paris' AND ma_score >= 60 LIMIT 20",
                'expected_index': 'idx_companies_location_score'
            },
            {
                'name': 'search_recent_data',
                'query': "SELECT * FROM companies WHERE last_scraped_at >= CURRENT_DATE - INTERVAL '30 days' LIMIT 100",
                'expected_index': 'idx_companies_recent_data'
            }
        ]
        
        for query_test in queries:
            await self._test_single_query_performance(query_test)
    
    async def _test_single_query_performance(self, query_test: Dict[str, str]):
        """Test performance d'une requête spécifique"""
        async with get_async_db_session() as session:
            # Exécution avec EXPLAIN ANALYZE
            explain_query = f"EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) {query_test['query']}"
            
            start_time = time.time()
            result = await session.execute(text(explain_query))
            execution_time = time.time() - start_time
            
            explain_result = result.scalar()
            plan = explain_result[0]['Plan']
            
            # Extraction métriques importantes
            actual_time = plan.get('Actual Total Time', 0)
            rows_returned = plan.get('Actual Rows', 0)
            buffers_hit = plan.get('Buffers', {}).get('Hit', 0)
            buffers_read = plan.get('Buffers', {}).get('Read', 0)
            
            # Vérifier utilisation d'index
            uses_index = 'Index' in str(plan)
            index_name = self._extract_index_name(str(plan))
            
            self.results[f"query_{query_test['name']}"] = {
                'execution_time_ms': round(actual_time, 2),
                'rows_returned': rows_returned,
                'uses_index': uses_index,
                'index_used': index_name,
                'buffer_hit_ratio': round(buffers_hit / (buffers_hit + buffers_read + 1), 3),
                'status': 'GOOD' if actual_time < 100 and uses_index else 'POOR'
            }
            
            logger.info(f"📊 {query_test['name']}: {actual_time:.2f}ms, index: {uses_index}")
    
    def _extract_index_name(self, plan_str: str) -> str:
        """Extrait le nom de l'index utilisé du plan"""
        lines = plan_str.split('\n')
        for line in lines:
            if 'Index Name' in line:
                return line.split(':')[-1].strip().strip('"')
        return 'unknown'
    
    async def test_query_optimization(self):
        """Test optimisations requêtes métier"""
        logger.info("🔄 Test optimisations requêtes métier...")
        
        # Requêtes typiques de l'application
        business_queries = [
            {
                'name': 'dashboard_stats',
                'query': """
                    SELECT 
                        COUNT(*) as total,
                        COUNT(CASE WHEN ma_score >= 70 THEN 1 END) as high_potential,
                        AVG(ma_score) as avg_score
                    FROM companies 
                    WHERE last_scraped_at >= CURRENT_DATE - INTERVAL '90 days'
                """
            },
            {
                'name': 'top_prospects',
                'query': """
                    SELECT siren, nom_entreprise, ma_score, chiffre_affaires
                    FROM companies 
                    WHERE ma_score >= 75 
                    AND chiffre_affaires >= 3000000
                    ORDER BY ma_score DESC, chiffre_affaires DESC 
                    LIMIT 50
                """
            },
            {
                'name': 'sector_analysis',
                'query': """
                    SELECT 
                        LEFT(code_naf, 2) as secteur,
                        COUNT(*) as nb_entreprises,
                        AVG(ma_score) as score_moyen
                    FROM companies 
                    WHERE code_naf IS NOT NULL 
                    AND ma_score IS NOT NULL
                    GROUP BY LEFT(code_naf, 2)
                    ORDER BY AVG(ma_score) DESC
                    LIMIT 20
                """
            }
        ]
        
        for query in business_queries:
            await self._benchmark_query(query)
    
    async def _benchmark_query(self, query: Dict[str, str]):
        """Benchmark d'une requête avec métriques détaillées"""
        async with get_async_db_session() as session:
            # Mesure multiple pour stabilité
            times = []
            for _ in range(5):
                start_time = time.time()
                await session.execute(text(query['query']))
                execution_time = (time.time() - start_time) * 1000  # en ms
                times.append(execution_time)
            
            avg_time = statistics.mean(times)
            p95_time = statistics.quantiles(times, n=20)[18]  # 95e percentile
            
            self.results[f"business_query_{query['name']}"] = {
                'avg_time_ms': round(avg_time, 2),
                'p95_time_ms': round(p95_time, 2),
                'min_time_ms': round(min(times), 2),
                'max_time_ms': round(max(times), 2),
                'status': 'GOOD' if avg_time < 500 else 'POOR'
            }
            
            logger.info(f"📊 {query['name']}: {avg_time:.2f}ms avg, {p95_time:.2f}ms p95")
    
    async def test_concurrent_connections(self):
        """Test charge avec connexions concurrentes"""
        logger.info("🔄 Test connexions concurrentes...")
        
        async def worker_query(worker_id: int):
            async with get_async_db_session() as session:
                result = await session.execute(
                    text("SELECT COUNT(*) FROM companies WHERE ma_score >= :score"),
                    {"score": random.randint(50, 90)}
                )
                return result.scalar()
        
        # Test avec différents niveaux de concurrence
        concurrency_levels = [10, 25, 50]
        
        for concurrency in concurrency_levels:
            start_time = time.time()
            tasks = [worker_query(i) for i in range(concurrency)]
            results = await asyncio.gather(*tasks)
            total_time = time.time() - start_time
            
            success_rate = len([r for r in results if r is not None]) / len(results)
            
            self.results[f'concurrent_{concurrency}'] = {
                'total_time': round(total_time, 2),
                'avg_time_per_query': round(total_time * 1000 / concurrency, 2),
                'success_rate': round(success_rate, 3),
                'status': 'GOOD' if success_rate > 0.95 and total_time < 10 else 'POOR'
            }
            
            logger.info(f"📊 Concurrence {concurrency}: {total_time:.2f}s total, {success_rate:.1%} succès")
    
    async def test_large_dataset_queries(self):
        """Test performance sur gros volumes"""
        logger.info("🔄 Test requêtes gros volumes...")
        
        large_queries = [
            {
                'name': 'full_table_scan',
                'query': "SELECT COUNT(*) FROM companies"
            },
            {
                'name': 'aggregation_heavy',
                'query': """
                    SELECT 
                        ville,
                        COUNT(*) as nb_entreprises,
                        AVG(chiffre_affaires) as ca_moyen,
                        PERCENTILE_CONT(0.9) WITHIN GROUP (ORDER BY ma_score) as p90_score
                    FROM companies 
                    WHERE ma_score IS NOT NULL
                    GROUP BY ville
                    HAVING COUNT(*) >= 10
                    ORDER BY AVG(chiffre_affaires) DESC
                """
            }
        ]
        
        for query in large_queries:
            await self._benchmark_query(query)
    
    async def test_partitioning_benefits(self):
        """Test efficacité du pseudo-partitioning via index partiels"""
        logger.info("🔄 Test bénéfices partitioning...")
        
        # Requête sur données récentes (devrait utiliser index partiel)
        recent_query = """
            SELECT COUNT(*) 
            FROM companies 
            WHERE last_scraped_at >= CURRENT_DATE - INTERVAL '30 days'
            AND ma_score >= 70
        """
        
        # Requête sur données anciennes (devrait utiliser index différent)
        old_query = """
            SELECT COUNT(*) 
            FROM companies 
            WHERE last_scraped_at < CURRENT_DATE - INTERVAL '1 year'
            OR last_scraped_at IS NULL
        """
        
        await self._benchmark_query({'name': 'recent_data_partition', 'query': recent_query})
        await self._benchmark_query({'name': 'old_data_partition', 'query': old_query})
    
    async def collect_database_stats(self):
        """Collecte statistiques générales de la base"""
        logger.info("🔄 Collecte statistiques base...")
        
        async with get_async_db_session() as session:
            # Stats générales
            stats_queries = {
                'total_size': "SELECT pg_size_pretty(pg_database_size(current_database()))",
                'companies_count': "SELECT COUNT(*) FROM companies",
                'companies_size': "SELECT pg_size_pretty(pg_total_relation_size('companies'))",
                'active_connections': "SELECT count(*) FROM pg_stat_activity WHERE state = 'active'",
                'cache_hit_ratio': """
                    SELECT round(
                        100.0 * sum(blks_hit) / nullif(sum(blks_hit) + sum(blks_read), 0), 2
                    ) FROM pg_stat_database WHERE datname = current_database()
                """,
                'index_usage': """
                    SELECT round(
                        100.0 * sum(idx_scan) / nullif(sum(idx_scan) + sum(seq_scan), 0), 2
                    ) FROM pg_stat_user_tables WHERE relname = 'companies'
                """
            }
            
            stats = {}
            for stat_name, query in stats_queries.items():
                try:
                    result = await session.execute(text(query))
                    stats[stat_name] = result.scalar()
                except Exception as e:
                    logger.warning(f"⚠️ Erreur collecte stat {stat_name}: {e}")
                    stats[stat_name] = 'unknown'
            
            self.results['database_stats'] = stats
            
            logger.info(f"📊 Stats: {stats['companies_count']} entreprises, cache hit: {stats['cache_hit_ratio']}%")
    
    async def analyze_slow_queries(self):
        """Analyse des requêtes lentes via pg_stat_statements"""
        logger.info("🔄 Analyse requêtes lentes...")
        
        try:
            async with get_async_db_session() as session:
                slow_queries = await session.execute(text("""
                    SELECT 
                        query,
                        calls,
                        total_exec_time::numeric(10,2) as total_time_ms,
                        mean_exec_time::numeric(10,2) as mean_time_ms,
                        max_exec_time::numeric(10,2) as max_time_ms,
                        rows
                    FROM pg_stat_statements 
                    WHERE query NOT LIKE '%pg_stat_statements%'
                    AND mean_exec_time > 100  -- Plus de 100ms en moyenne
                    ORDER BY mean_exec_time DESC 
                    LIMIT 10
                """))
                
                slow_query_list = []
                for row in slow_queries.fetchall():
                    slow_query_list.append({
                        'query': row[0][:100] + '...' if len(row[0]) > 100 else row[0],
                        'calls': row[1],
                        'total_time_ms': float(row[2]),
                        'mean_time_ms': float(row[3]),
                        'max_time_ms': float(row[4]),
                        'rows': row[5]
                    })
                
                self.results['slow_queries'] = slow_query_list
                logger.info(f"📊 {len(slow_query_list)} requêtes lentes identifiées")
                
        except Exception as e:
            logger.warning(f"⚠️ Erreur analyse requêtes lentes: {e}")
            self.results['slow_queries'] = []
    
    def generate_performance_report(self):
        """Génère un rapport de performance détaillé"""
        logger.info("📝 Génération rapport de performance...")
        
        # Calcul score global
        scores = []
        for key, value in self.results.items():
            if isinstance(value, dict) and 'status' in value:
                scores.append(1 if value['status'] == 'GOOD' else 0)
        
        global_score = sum(scores) / len(scores) if scores else 0
        
        # Génération rapport
        report = {
            'timestamp': datetime.now().isoformat(),
            'global_score': round(global_score * 100, 1),
            'summary': {
                'total_tests': len(scores),
                'passed_tests': sum(scores),
                'failed_tests': len(scores) - sum(scores)
            },
            'detailed_results': self.results,
            'recommendations': self._generate_recommendations()
        }
        
        # Sauvegarde rapport
        report_file = f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # Affichage résumé
        logger.info(f"✅ Rapport sauvegardé: {report_file}")
        logger.info(f"📊 Score global: {global_score * 100:.1f}% ({sum(scores)}/{len(scores)} tests OK)")
        
        # Afficher résumé des problèmes
        problems = [k for k, v in self.results.items() 
                   if isinstance(v, dict) and v.get('status') == 'POOR']
        if problems:
            logger.warning(f"⚠️ Tests échoués: {', '.join(problems)}")
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Génère des recommandations d'optimisation"""
        recommendations = []
        
        # Analyse résultats pour recommandations
        if 'database_stats' in self.results:
            stats = self.results['database_stats']
            
            if isinstance(stats.get('cache_hit_ratio'), (int, float)) and stats['cache_hit_ratio'] < 95:
                recommendations.append("Augmenter shared_buffers pour améliorer le cache hit ratio")
            
            if isinstance(stats.get('index_usage'), (int, float)) and stats['index_usage'] < 90:
                recommendations.append("Ajouter des index pour réduire les scans séquentiels")
        
        # Vérifier requêtes lentes
        if 'slow_queries' in self.results and self.results['slow_queries']:
            recommendations.append("Optimiser les requêtes lentes identifiées")
        
        # Vérifier performance connexions
        for key, value in self.results.items():
            if 'concurrent' in key and value.get('status') == 'POOR':
                recommendations.append("Ajuster la configuration du pool de connexions")
                break
        
        if not recommendations:
            recommendations.append("Performance optimale - aucune amélioration nécessaire")
        
        return recommendations


async def main():
    """Point d'entrée principal"""
    parser = argparse.ArgumentParser(description='Test performance PostgreSQL')
    parser.add_argument('--full-test', action='store_true', help='Tests complets avec gros volumes')
    parser.add_argument('--baseline', action='store_true', help='Tests sans création de données')
    
    args = parser.parse_args()
    
    tester = DatabasePerformanceTester()
    await tester.run_all_tests(full_test=args.full_test, baseline=args.baseline)


if __name__ == "__main__":
    asyncio.run(main())