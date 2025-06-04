#!/usr/bin/env python3
"""
Monitoring des requêtes lentes PostgreSQL en temps réel
Pour surveiller les performances après optimisation US-001

Usage:
    python scripts/monitor_slow_queries.py
    python scripts/monitor_slow_queries.py --threshold 500
    python scripts/monitor_slow_queries.py --watch
"""

import asyncio
import argparse
import logging
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any
import time

# Configuration logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from sqlalchemy import text
    from app.core.database import get_async_db_session
except ImportError as e:
    logger.error(f"Import error: {e}")
    logger.error("Make sure to run from backend directory: cd backend && python scripts/monitor_slow_queries.py")
    exit(1)


class SlowQueryMonitor:
    """Moniteur de requêtes lentes PostgreSQL"""
    
    def __init__(self, threshold_ms: int = 1000):
        self.threshold_ms = threshold_ms
        self.alerts_sent = set()
        self.baseline_stats = {}
        
    async def start_monitoring(self, watch_mode: bool = False):
        """Démarre le monitoring des requêtes lentes"""
        logger.info(f"🔍 Démarrage monitoring requêtes lentes (seuil: {self.threshold_ms}ms)")
        
        # Vérifier pg_stat_statements
        if not await self.check_pg_stat_statements():
            logger.error("❌ pg_stat_statements non disponible - monitoring impossible")
            return
        
        # Établir baseline
        await self.establish_baseline()
        
        if watch_mode:
            await self.continuous_monitoring()
        else:
            await self.single_report()
    
    async def check_pg_stat_statements(self) -> bool:
        """Vérifie si pg_stat_statements est disponible"""
        try:
            async with get_async_db_session() as session:
                result = await session.execute(text("""
                    SELECT 1 FROM pg_extension WHERE extname = 'pg_stat_statements'
                """))
                return result.scalar() is not None
        except Exception as e:
            logger.error(f"❌ Erreur vérification pg_stat_statements: {e}")
            return False
    
    async def establish_baseline(self):
        """Établit les métriques de référence"""
        logger.info("📊 Établissement baseline de performance...")
        
        try:
            async with get_async_db_session() as session:
                # Stats générales
                baseline_queries = {
                    'total_queries': "SELECT calls FROM pg_stat_statements WHERE query = 'SELECT $1'",
                    'avg_duration': "SELECT AVG(mean_exec_time) FROM pg_stat_statements",
                    'total_time': "SELECT SUM(total_exec_time) FROM pg_stat_statements",
                    'slow_queries_count': f"""
                        SELECT COUNT(*) FROM pg_stat_statements 
                        WHERE mean_exec_time > {self.threshold_ms}
                    """
                }
                
                for stat_name, query in baseline_queries.items():
                    try:
                        result = await session.execute(text(query))
                        self.baseline_stats[stat_name] = result.scalar() or 0
                    except Exception as e:
                        logger.warning(f"⚠️ Erreur baseline {stat_name}: {e}")
                        self.baseline_stats[stat_name] = 0
                
                logger.info(f"✅ Baseline établi: {self.baseline_stats['slow_queries_count']} requêtes lentes")
        
        except Exception as e:
            logger.error(f"❌ Erreur établissement baseline: {e}")
    
    async def single_report(self):
        """Génère un rapport unique des requêtes lentes"""
        slow_queries = await self.get_slow_queries()
        db_stats = await self.get_database_stats()
        connection_stats = await self.get_connection_stats()
        index_stats = await self.get_index_efficiency()
        
        # Génération rapport
        report = {
            'timestamp': datetime.now().isoformat(),
            'threshold_ms': self.threshold_ms,
            'summary': {
                'slow_queries_count': len(slow_queries),
                'total_queries_analyzed': db_stats.get('total_queries', 0),
                'avg_query_time_ms': round(db_stats.get('avg_duration', 0), 2)
            },
            'database_stats': db_stats,
            'connection_stats': connection_stats,
            'index_efficiency': index_stats,
            'slow_queries': slow_queries[:20],  # Top 20
            'recommendations': self.generate_recommendations(slow_queries, db_stats)
        }
        
        # Sauvegarde et affichage
        report_file = f"slow_queries_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        self.display_report_summary(report)
        logger.info(f"📄 Rapport détaillé sauvegardé: {report_file}")
    
    async def continuous_monitoring(self):
        """Mode monitoring continu"""
        logger.info("🔄 Mode monitoring continu activé (Ctrl+C pour arrêter)")
        
        try:
            while True:
                current_time = datetime.now()
                slow_queries = await self.get_slow_queries(limit=5)
                
                if slow_queries:
                    logger.warning(f"⚠️ {len(slow_queries)} requêtes lentes détectées:")
                    for query in slow_queries:
                        self.log_slow_query(query)
                        
                        # Alerting si nouvelle requête lente critique
                        if query['mean_time_ms'] > self.threshold_ms * 2:
                            await self.send_alert(query)
                
                # Stats périodiques
                if current_time.minute % 5 == 0:  # Toutes les 5 minutes
                    stats = await self.get_database_stats()
                    logger.info(f"📊 Stats: {stats['active_connections']} connexions, "
                              f"{stats['cache_hit_ratio']:.1f}% cache hit")
                
                await asyncio.sleep(30)  # Check toutes les 30 secondes
                
        except KeyboardInterrupt:
            logger.info("👋 Arrêt monitoring demandé")
    
    async def get_slow_queries(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Récupère les requêtes les plus lentes"""
        try:
            async with get_async_db_session() as session:
                result = await session.execute(text(f"""
                    SELECT 
                        query,
                        calls,
                        total_exec_time::numeric(10,2) as total_time_ms,
                        mean_exec_time::numeric(10,2) as mean_time_ms,
                        max_exec_time::numeric(10,2) as max_time_ms,
                        min_exec_time::numeric(10,2) as min_time_ms,
                        stddev_exec_time::numeric(10,2) as stddev_time_ms,
                        rows,
                        100.0 * shared_blks_hit / nullif(shared_blks_hit + shared_blks_read, 0) as hit_percent
                    FROM pg_stat_statements 
                    WHERE mean_exec_time > {self.threshold_ms}
                    AND query NOT LIKE '%pg_stat_statements%'
                    AND query NOT LIKE 'EXPLAIN%'
                    ORDER BY mean_exec_time DESC 
                    LIMIT {limit}
                """))
                
                slow_queries = []
                for row in result.fetchall():
                    query_info = {
                        'query': self.normalize_query(row[0]),
                        'query_hash': hash(row[0]) % 10000,  # Hash simple pour identification
                        'calls': row[1],
                        'total_time_ms': float(row[2]),
                        'mean_time_ms': float(row[3]),
                        'max_time_ms': float(row[4]),
                        'min_time_ms': float(row[5]),
                        'stddev_time_ms': float(row[6]),
                        'rows_returned': row[7],
                        'cache_hit_percent': round(float(row[8] or 0), 1),
                        'performance_score': self.calculate_performance_score(row)
                    }
                    slow_queries.append(query_info)
                
                return slow_queries
        
        except Exception as e:
            logger.error(f"❌ Erreur récupération requêtes lentes: {e}")
            return []
    
    def normalize_query(self, query: str) -> str:
        """Normalise une requête pour l'affichage"""
        # Nettoyer et tronquer la requête
        query = ' '.join(query.split())  # Normaliser les espaces
        if len(query) > 200:
            query = query[:200] + '...'
        return query
    
    def calculate_performance_score(self, row) -> str:
        """Calcule un score de performance (CRITICAL, HIGH, MEDIUM, LOW)"""
        mean_time = float(row[3])
        calls = row[1]
        total_time = float(row[2])
        
        if mean_time > 5000 or (calls > 100 and total_time > 10000):
            return 'CRITICAL'
        elif mean_time > 2000 or (calls > 50 and total_time > 5000):
            return 'HIGH'
        elif mean_time > 1000:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    async def get_database_stats(self) -> Dict[str, Any]:
        """Récupère les statistiques générales de la base"""
        try:
            async with get_async_db_session() as session:
                stats_query = text("""
                    SELECT 
                        (SELECT count(*) FROM pg_stat_statements) as total_queries,
                        (SELECT AVG(mean_exec_time) FROM pg_stat_statements) as avg_duration,
                        (SELECT SUM(calls) FROM pg_stat_statements) as total_calls,
                        (SELECT count(*) FROM pg_stat_activity WHERE state = 'active') as active_connections,
                        (SELECT count(*) FROM pg_stat_activity) as total_connections,
                        (SELECT round(100.0 * sum(blks_hit) / nullif(sum(blks_hit) + sum(blks_read), 0), 2)
                         FROM pg_stat_database WHERE datname = current_database()) as cache_hit_ratio,
                        (SELECT pg_size_pretty(pg_database_size(current_database()))) as db_size
                """)
                
                result = await session.execute(stats_query)
                row = result.fetchone()
                
                return {
                    'total_queries': row[0] or 0,
                    'avg_duration': round(float(row[1] or 0), 2),
                    'total_calls': row[2] or 0,
                    'active_connections': row[3] or 0,
                    'total_connections': row[4] or 0,
                    'cache_hit_ratio': float(row[5] or 0),
                    'db_size': row[6] or 'unknown'
                }
        
        except Exception as e:
            logger.error(f"❌ Erreur stats base: {e}")
            return {}
    
    async def get_connection_stats(self) -> Dict[str, Any]:
        """Statistiques sur les connexions"""
        try:
            async with get_async_db_session() as session:
                result = await session.execute(text("""
                    SELECT 
                        state,
                        count(*) as count,
                        max(extract(epoch from (now() - state_change))) as max_duration_seconds
                    FROM pg_stat_activity 
                    WHERE pid != pg_backend_pid()
                    GROUP BY state
                    ORDER BY count DESC
                """))
                
                connections = {}
                for row in result.fetchall():
                    connections[row[0] or 'unknown'] = {
                        'count': row[1],
                        'max_duration_seconds': round(float(row[2] or 0), 2)
                    }
                
                return connections
        
        except Exception as e:
            logger.error(f"❌ Erreur stats connexions: {e}")
            return {}
    
    async def get_index_efficiency(self) -> Dict[str, Any]:
        """Efficacité des index"""
        try:
            async with get_async_db_session() as session:
                # Index usage sur table companies
                result = await session.execute(text("""
                    SELECT 
                        indexrelname,
                        idx_scan,
                        idx_tup_read,
                        idx_tup_fetch,
                        pg_size_pretty(pg_relation_size(indexrelid)) as size
                    FROM pg_stat_user_indexes 
                    WHERE relname = 'companies'
                    ORDER BY idx_scan DESC
                """))
                
                indexes = []
                for row in result.fetchall():
                    indexes.append({
                        'index_name': row[0],
                        'scans': row[1],
                        'tuples_read': row[2],
                        'tuples_fetched': row[3],
                        'size': row[4],
                        'efficiency': round((row[3] / max(row[2], 1)) * 100, 1)
                    })
                
                # Stats générales d'utilisation des index
                general_result = await session.execute(text("""
                    SELECT 
                        sum(idx_scan) as total_index_scans,
                        sum(seq_scan) as total_seq_scans,
                        round(100.0 * sum(idx_scan) / nullif(sum(idx_scan) + sum(seq_scan), 0), 2) as index_usage_ratio
                    FROM pg_stat_user_tables 
                    WHERE relname = 'companies'
                """))
                
                general_row = general_result.fetchone()
                
                return {
                    'indexes_detail': indexes,
                    'total_index_scans': general_row[0] or 0,
                    'total_seq_scans': general_row[1] or 0,
                    'index_usage_ratio': float(general_row[2] or 0)
                }
        
        except Exception as e:
            logger.error(f"❌ Erreur stats index: {e}")
            return {}
    
    def log_slow_query(self, query: Dict[str, Any]):
        """Log d'une requête lente"""
        severity = "🚨" if query['performance_score'] == 'CRITICAL' else "⚠️"
        logger.warning(
            f"{severity} [{query['performance_score']}] "
            f"{query['mean_time_ms']:.0f}ms avg ({query['calls']} calls) - "
            f"{query['query'][:100]}..."
        )
    
    async def send_alert(self, query: Dict[str, Any]):
        """Envoie une alerte pour une requête critique"""
        query_id = f"{query['query_hash']}_{query['performance_score']}"
        
        if query_id not in self.alerts_sent:
            logger.error(
                f"🚨 ALERTE CRITIQUE: Requête lente détectée!\n"
                f"   Temps moyen: {query['mean_time_ms']:.0f}ms\n"
                f"   Appels: {query['calls']}\n"
                f"   Requête: {query['query'][:150]}..."
            )
            self.alerts_sent.add(query_id)
    
    def display_report_summary(self, report: Dict[str, Any]):
        """Affiche un résumé du rapport"""
        logger.info("=" * 80)
        logger.info("📊 RAPPORT DE MONITORING DES REQUÊTES LENTES")
        logger.info("=" * 80)
        
        summary = report['summary']
        logger.info(f"⏱️  Seuil monitoring: {self.threshold_ms}ms")
        logger.info(f"🐌 Requêtes lentes: {summary['slow_queries_count']}")
        logger.info(f"📈 Temps moyen: {summary['avg_query_time_ms']}ms")
        
        db_stats = report.get('database_stats', {})
        if db_stats:
            logger.info(f"🔗 Connexions actives: {db_stats.get('active_connections', 0)}")
            logger.info(f"💾 Cache hit ratio: {db_stats.get('cache_hit_ratio', 0):.1f}%")
            logger.info(f"📦 Taille base: {db_stats.get('db_size', 'unknown')}")
        
        # Top 5 requêtes les plus lentes
        slow_queries = report.get('slow_queries', [])
        if slow_queries:
            logger.info("\n🔝 TOP 5 REQUÊTES LES PLUS LENTES:")
            for i, query in enumerate(slow_queries[:5], 1):
                logger.info(
                    f"  {i}. [{query['performance_score']}] {query['mean_time_ms']:.0f}ms "
                    f"({query['calls']} calls) - {query['query'][:80]}..."
                )
        
        # Recommandations
        recommendations = report.get('recommendations', [])
        if recommendations:
            logger.info("\n💡 RECOMMANDATIONS:")
            for i, rec in enumerate(recommendations, 1):
                logger.info(f"  {i}. {rec}")
        
        logger.info("=" * 80)
    
    def generate_recommendations(self, slow_queries: List[Dict], db_stats: Dict) -> List[str]:
        """Génère des recommandations d'optimisation"""
        recommendations = []
        
        # Analyse cache hit ratio
        cache_hit = db_stats.get('cache_hit_ratio', 100)
        if cache_hit < 95:
            recommendations.append(f"Cache hit ratio faible ({cache_hit:.1f}%) - augmenter shared_buffers")
        
        # Analyse requêtes critiques
        critical_queries = [q for q in slow_queries if q['performance_score'] == 'CRITICAL']
        if critical_queries:
            recommendations.append(f"{len(critical_queries)} requêtes critiques nécessitent optimisation immédiate")
        
        # Analyse patterns de requêtes
        select_queries = [q for q in slow_queries if 'SELECT' in q['query'].upper()]
        if len(select_queries) > len(slow_queries) * 0.8:
            recommendations.append("Majoritairement des SELECT lents - vérifier les index")
        
        # Analyse connexions
        active_conn = db_stats.get('active_connections', 0)
        if active_conn > 50:
            recommendations.append(f"Nombre élevé de connexions actives ({active_conn}) - vérifier pool de connexions")
        
        # Analyse requêtes fréquentes lentes
        frequent_slow = [q for q in slow_queries if q['calls'] > 100 and q['mean_time_ms'] > 1000]
        if frequent_slow:
            recommendations.append("Requêtes fréquentes et lentes détectées - priorité optimisation")
        
        if not recommendations:
            recommendations.append("Performances correctes - pas d'optimisation urgente nécessaire")
        
        return recommendations


async def main():
    """Point d'entrée principal"""
    parser = argparse.ArgumentParser(description='Monitor PostgreSQL slow queries')
    parser.add_argument('--threshold', type=int, default=1000, 
                      help='Seuil en ms pour considérer une requête comme lente (défaut: 1000ms)')
    parser.add_argument('--watch', action='store_true', 
                      help='Mode monitoring continu')
    
    args = parser.parse_args()
    
    monitor = SlowQueryMonitor(threshold_ms=args.threshold)
    await monitor.start_monitoring(watch_mode=args.watch)


if __name__ == "__main__":
    asyncio.run(main())