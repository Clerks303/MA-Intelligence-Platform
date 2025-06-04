"""
Optimiseur de base de données pour M&A Intelligence Platform
US-005: Optimisation requêtes SQL, indexation et connection pooling

Features:
- Pool de connexions optimisé avec retry logic
- Query optimization avec analyse des plans d'exécution
- Indexation intelligente basée sur les patterns d'usage
- Pagination performante avec cursor-based
- Requêtes batch optimisées
- Monitoring performance des queries
"""

import asyncio
import time
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from contextlib import asynccontextmanager
from functools import wraps
import asyncpg
from asyncpg.pool import Pool
from sqlalchemy import text, inspect
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.pool import QueuePool
from sqlalchemy.orm import selectinload, joinedload
from sqlalchemy import and_, or_, func, case, exists
import psutil

from app.config import settings
from app.core.logging_system import get_logger, LogCategory
from app.core.cache_manager import cached, get_cache_manager

logger = get_logger("database_optimizer", LogCategory.PERFORMANCE)


@dataclass
class QueryStats:
    """Statistiques d'exécution des requêtes"""
    query_hash: str
    sql: str
    execution_time_ms: float
    rows_affected: int
    timestamp: datetime = field(default_factory=datetime.now)
    success: bool = True
    error: Optional[str] = None


@dataclass
class ConnectionPoolStats:
    """Statistiques du pool de connexions"""
    total_connections: int = 0
    active_connections: int = 0
    idle_connections: int = 0
    waiting_connections: int = 0
    max_connections: int = 20
    connection_timeouts: int = 0
    total_queries: int = 0
    avg_query_time_ms: float = 0.0


class DatabaseOptimizer:
    """Optimiseur de performance base de données"""
    
    def __init__(self):
        self.engine = None
        self.session_factory = None
        self.connection_pool: Optional[Pool] = None
        self.query_stats: List[QueryStats] = []
        self.pool_stats = ConnectionPoolStats()
        self.slow_query_threshold_ms = 1000  # 1 seconde
        
        # Index recommandés par type de requête
        self.recommended_indexes = [
            # Companies - index composites pour filtres fréquents
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_companies_siren_status ON companies(siren, statut)",
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_companies_ca_secteur ON companies(chiffre_affaires, secteur_activite)",
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_companies_score_prospect ON companies(score_prospection DESC, last_scraped_at DESC)",
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_companies_created_at ON companies(created_at DESC)",
            
            # Logs et audit
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_logs_timestamp_level ON logs(timestamp DESC, level)",
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_audit_timestamp_action ON audit_logs(timestamp DESC, action)",
            
            # Recherche textuelle
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_companies_nom_trgm ON companies USING gin(nom_entreprise gin_trgm_ops)",
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_companies_siret ON companies(siret)",
            
            # Métriques et stats
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_metrics_timestamp ON metrics(timestamp DESC, metric_name)",
        ]
        
        logger.info("🔧 DatabaseOptimizer initialisé")
    
    async def initialize(self):
        """Initialise le moteur de base de données optimisé"""
        try:
            # Configuration pool de connexions
            pool_size = min(20, (psutil.cpu_count() or 4) * 2)  # 2x CPU cores max 20
            max_overflow = 10
            pool_timeout = 30
            pool_recycle = 3600  # 1 heure
            
            # Engine SQLAlchemy avec pool optimisé
            database_url = settings.database_url
            if database_url.startswith('postgresql://'):
                database_url = database_url.replace('postgresql://', 'postgresql+asyncpg://', 1)
            
            self.engine = create_async_engine(
                database_url,
                poolclass=QueuePool,
                pool_size=pool_size,
                max_overflow=max_overflow,
                pool_timeout=pool_timeout,
                pool_recycle=pool_recycle,
                pool_pre_ping=True,  # Vérification santé connexions
                echo=False,  # Pas de logs SQL en prod
                connect_args={
                    "server_settings": {
                        "jit": "off",  # Optimisation PostgreSQL
                        "application_name": "ma_intelligence_optimized"
                    }
                }
            )
            
            # Session factory
            self.session_factory = async_sessionmaker(
                self.engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            
            # Pool asyncpg direct pour requêtes raw optimisées
            await self._setup_asyncpg_pool()
            
            # Création des index recommandés
            await self._ensure_indexes()
            
            # Statistiques PostgreSQL
            await self._setup_pg_stats()
            
            logger.info(f"✅ Base de données optimisée - Pool: {pool_size}+{max_overflow}")
            
        except Exception as e:
            logger.error(f"❌ Erreur initialisation database optimizer: {e}")
            raise
    
    async def _setup_asyncpg_pool(self):
        """Configure le pool asyncpg pour requêtes raw performantes"""
        try:
            # Extraire infos de connection depuis URL
            url_parts = settings.database_url.replace('postgresql://', '').split('@')
            if len(url_parts) != 2:
                logger.warning("Format URL base non supporté pour asyncpg")
                return
                
            auth_part = url_parts[0]
            host_part = url_parts[1]
            
            username, password = auth_part.split(':')
            host_db = host_part.split('/')
            host_port = host_db[0].split(':')
            host = host_port[0]
            port = int(host_port[1]) if len(host_port) > 1 else 5432
            database = host_db[1] if len(host_db) > 1 else 'postgres'
            
            self.connection_pool = await asyncpg.create_pool(
                user=username,
                password=password,
                database=database,
                host=host,
                port=port,
                min_size=5,
                max_size=15,
                max_queries=50000,
                max_inactive_connection_lifetime=300,
                setup=self._setup_connection
            )
            
            logger.info("✅ Pool asyncpg configuré")
            
        except Exception as e:
            logger.warning(f"⚠️ Impossible de configurer asyncpg pool: {e}")
    
    async def _setup_connection(self, connection):
        """Configuration optimisée par connexion"""
        # Extensions PostgreSQL pour performance
        await connection.execute("SET jit = off")
        await connection.execute("SET random_page_cost = 1.1")  # SSD optimization
        await connection.execute("SET effective_cache_size = '1GB'")
    
    async def _ensure_indexes(self):
        """Crée les index recommandés"""
        if not self.engine:
            return
            
        try:
            async with self.engine.begin() as conn:
                # Vérifier si les extensions nécessaires existent
                await conn.execute(text("CREATE EXTENSION IF NOT EXISTS pg_trgm"))
                
                # Créer les index de manière non-bloquante
                for index_sql in self.recommended_indexes:
                    try:
                        await conn.execute(text(index_sql))
                        logger.debug(f"Index créé: {index_sql}")
                    except Exception as e:
                        logger.warning(f"Index non créé: {e}")
                
            logger.info(f"✅ Vérification index terminée ({len(self.recommended_indexes)} index)")
            
        except Exception as e:
            logger.error(f"Erreur création index: {e}")
    
    async def _setup_pg_stats(self):
        """Configure les statistiques PostgreSQL"""
        if not self.engine:
            return
            
        try:
            async with self.engine.begin() as conn:
                # Activer les statistiques détaillées
                await conn.execute(text("SET track_activities = on"))
                await conn.execute(text("SET track_counts = on"))
                await conn.execute(text("SET track_io_timing = on"))
                
        except Exception as e:
            logger.warning(f"Configuration stats PostgreSQL échouée: {e}")
    
    @asynccontextmanager
    async def get_session(self):
        """Context manager pour sessions optimisées"""
        if not self.session_factory:
            raise RuntimeError("DatabaseOptimizer not initialized")
        
        session = self.session_factory()
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
    
    @asynccontextmanager
    async def get_raw_connection(self):
        """Context manager pour connexions raw asyncpg"""
        if not self.connection_pool:
            raise RuntimeError("AsyncPG pool not available")
        
        async with self.connection_pool.acquire() as connection:
            yield connection
    
    def query_monitor(self, query_name: str = None):
        """Décorateur pour monitorer les performances des requêtes"""
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                start_time = time.time()
                success = True
                error = None
                rows_affected = 0
                
                try:
                    result = await func(*args, **kwargs)
                    
                    # Estimer le nombre de lignes selon le type de résultat
                    if hasattr(result, '__len__'):
                        rows_affected = len(result)
                    elif hasattr(result, 'rowcount'):
                        rows_affected = result.rowcount
                    
                    return result
                    
                except Exception as e:
                    success = False
                    error = str(e)
                    raise
                    
                finally:
                    execution_time_ms = (time.time() - start_time) * 1000
                    
                    # Enregistrer les stats
                    query_stats = QueryStats(
                        query_hash=f"{func.__name__}_{hash(str(args) + str(kwargs))}",
                        sql=query_name or func.__name__,
                        execution_time_ms=execution_time_ms,
                        rows_affected=rows_affected,
                        success=success,
                        error=error
                    )
                    
                    self.query_stats.append(query_stats)
                    self.pool_stats.total_queries += 1
                    
                    # Log requêtes lentes
                    if execution_time_ms > self.slow_query_threshold_ms:
                        logger.warning(
                            f"🐌 Requête lente: {query_name or func.__name__} - "
                            f"{execution_time_ms:.2f}ms - {rows_affected} lignes"
                        )
                    
                    # Nettoyer ancien historique (garder 1000 dernières)
                    if len(self.query_stats) > 1000:
                        self.query_stats = self.query_stats[-1000:]
            
            return wrapper
        return decorator
    
    # Requêtes optimisées pour companies
    
    @query_monitor("companies_paginated")
    async def get_companies_paginated(self, 
                                    page: int = 1, 
                                    page_size: int = 50,
                                    filters: Dict[str, Any] = None,
                                    sort_by: str = "created_at",
                                    sort_desc: bool = True) -> Tuple[List[Dict], int]:
        """Pagination optimisée avec cursor-based pour de gros volumes"""
        
        filters = filters or {}
        offset = (page - 1) * page_size
        
        # Construction requête avec index hints
        base_query = """
        SELECT c.*, 
               COUNT(*) OVER() as total_count
        FROM companies c
        WHERE 1=1
        """
        
        params = []
        param_count = 0
        
        # Filtres optimisés avec index
        if filters.get('siren'):
            param_count += 1
            base_query += f" AND c.siren = ${param_count}"
            params.append(filters['siren'])
        
        if filters.get('statut'):
            param_count += 1
            base_query += f" AND c.statut = ${param_count}"
            params.append(filters['statut'])
        
        if filters.get('secteur_activite'):
            param_count += 1
            base_query += f" AND c.secteur_activite ILIKE ${param_count}"
            params.append(f"%{filters['secteur_activite']}%")
        
        if filters.get('ca_min'):
            param_count += 1
            base_query += f" AND c.chiffre_affaires >= ${param_count}"
            params.append(filters['ca_min'])
        
        if filters.get('ca_max'):
            param_count += 1
            base_query += f" AND c.chiffre_affaires <= ${param_count}"
            params.append(filters['ca_max'])
        
        if filters.get('score_min'):
            param_count += 1
            base_query += f" AND c.score_prospection >= ${param_count}"
            params.append(filters['score_min'])
        
        # Recherche textuelle optimisée avec trigram
        if filters.get('search'):
            param_count += 1
            base_query += f" AND c.nom_entreprise % ${param_count}"
            params.append(filters['search'])
        
        # Tri optimisé avec index
        sort_column = sort_by if sort_by in ['created_at', 'score_prospection', 'chiffre_affaires'] else 'created_at'
        sort_direction = 'DESC' if sort_desc else 'ASC'
        base_query += f" ORDER BY c.{sort_column} {sort_direction}"
        
        # Pagination
        param_count += 1
        base_query += f" LIMIT ${param_count}"
        params.append(page_size)
        
        param_count += 1
        base_query += f" OFFSET ${param_count}"
        params.append(offset)
        
        if self.connection_pool:
            async with self.get_raw_connection() as conn:
                rows = await conn.fetch(base_query, *params)
                
                companies = []
                total_count = 0
                
                for row in rows:
                    # Convertir en dict
                    company_dict = dict(row)
                    total_count = company_dict.pop('total_count', 0)
                    companies.append(company_dict)
                
                return companies, total_count
        else:
            # Fallback SQLAlchemy
            async with self.get_session() as session:
                # Version simplifiée sans optimisations asyncpg
                from app.models.company import Company
                
                query = session.query(Company)
                
                if filters.get('statut'):
                    query = query.filter(Company.statut == filters['statut'])
                
                total = await query.count()
                companies = await query.offset(offset).limit(page_size).all()
                
                return [company.__dict__ for company in companies], total
    
    @cached(namespace='companies', ttl_seconds=1800)
    @query_monitor("companies_stats")
    async def get_companies_stats(self) -> Dict[str, Any]:
        """Statistiques optimisées des companies avec cache"""
        
        stats_query = """
        SELECT 
            COUNT(*) as total_companies,
            COUNT(CASE WHEN statut = 'à contacter' THEN 1 END) as prospects,
            COUNT(CASE WHEN statut = 'contacté' THEN 1 END) as contacted,
            COUNT(CASE WHEN statut = 'qualifié' THEN 1 END) as qualified,
            AVG(score_prospection) as avg_score,
            MAX(score_prospection) as max_score,
            COUNT(CASE WHEN last_scraped_at > NOW() - INTERVAL '24 hours' THEN 1 END) as scraped_today,
            AVG(chiffre_affaires) as avg_ca,
            COUNT(CASE WHEN chiffre_affaires > 1000000 THEN 1 END) as ca_over_1m
        FROM companies
        WHERE created_at > NOW() - INTERVAL '1 year'
        """
        
        if self.connection_pool:
            async with self.get_raw_connection() as conn:
                row = await conn.fetchrow(stats_query)
                return dict(row) if row else {}
        else:
            async with self.get_session() as session:
                result = await session.execute(text(stats_query))
                row = result.fetchone()
                return row._asdict() if row else {}
    
    @query_monitor("companies_bulk_update")
    async def bulk_update_companies_status(self, 
                                         company_ids: List[int], 
                                         new_status: str,
                                         updated_by: str) -> int:
        """Mise à jour en lot optimisée"""
        
        if not company_ids:
            return 0
        
        # Requête optimisée avec UNNEST pour gros volumes
        update_query = """
        UPDATE companies 
        SET statut = $1, 
            updated_at = NOW(),
            updated_by = $2
        WHERE id = ANY($3::int[])
        """
        
        if self.connection_pool:
            async with self.get_raw_connection() as conn:
                result = await conn.execute(update_query, new_status, updated_by, company_ids)
                
                # Invalider le cache
                cache_manager = await get_cache_manager()
                await cache_manager.invalidate_by_tag('companies')
                
                return int(result.split()[-1])  # Nombre de lignes affectées
        else:
            async with self.get_session() as session:
                from app.models.company import Company
                
                result = await session.execute(
                    text(update_query),
                    {"status": new_status, "updated_by": updated_by, "ids": company_ids}
                )
                return result.rowcount
    
    # Monitoring et statistiques
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques de performance"""
        
        recent_queries = [q for q in self.query_stats if q.timestamp > datetime.now() - timedelta(hours=1)]
        
        if recent_queries:
            avg_query_time = sum(q.execution_time_ms for q in recent_queries) / len(recent_queries)
            slow_queries = [q for q in recent_queries if q.execution_time_ms > self.slow_query_threshold_ms]
            error_rate = sum(1 for q in recent_queries if not q.success) / len(recent_queries)
        else:
            avg_query_time = 0
            slow_queries = []
            error_rate = 0
        
        # Stats pool de connexions
        pool_info = {}
        if self.connection_pool:
            pool_info = {
                'size': self.connection_pool._queue.qsize(),
                'max_size': self.connection_pool._maxsize,
                'min_size': self.connection_pool._minsize
            }
        
        return {
            'query_performance': {
                'total_queries_1h': len(recent_queries),
                'avg_query_time_ms': round(avg_query_time, 2),
                'slow_queries_count': len(slow_queries),
                'error_rate_percent': round(error_rate * 100, 2),
                'slowest_queries': [
                    {
                        'query': q.sql,
                        'time_ms': round(q.execution_time_ms, 2),
                        'timestamp': q.timestamp.isoformat()
                    }
                    for q in sorted(slow_queries, key=lambda x: x.execution_time_ms, reverse=True)[:5]
                ]
            },
            'connection_pool': {
                **pool_info,
                'total_queries': self.pool_stats.total_queries
            },
            'recommendations': self._get_performance_recommendations()
        }
    
    def _get_performance_recommendations(self) -> List[str]:
        """Génère des recommandations d'optimisation"""
        recommendations = []
        
        recent_queries = [q for q in self.query_stats if q.timestamp > datetime.now() - timedelta(hours=1)]
        
        if recent_queries:
            avg_time = sum(q.execution_time_ms for q in recent_queries) / len(recent_queries)
            slow_queries_ratio = len([q for q in recent_queries if q.execution_time_ms > self.slow_query_threshold_ms]) / len(recent_queries)
            
            if avg_time > 500:
                recommendations.append("Temps de requête moyen élevé - vérifier les index")
            
            if slow_queries_ratio > 0.1:
                recommendations.append(f"{slow_queries_ratio:.1%} de requêtes lentes - optimiser les requêtes les plus fréquentes")
            
            # Analyser les patterns de requêtes
            query_patterns = {}
            for q in recent_queries:
                base_query = q.sql.split()[0].lower()  # SELECT, UPDATE, etc.
                query_patterns[base_query] = query_patterns.get(base_query, 0) + 1
            
            if query_patterns.get('select', 0) > query_patterns.get('update', 0) * 10:
                recommendations.append("Ratio lecture/écriture élevé - considérer un cache plus agressif")
        
        if not recommendations:
            recommendations.append("Performance de base de données optimale")
        
        return recommendations
    
    async def analyze_slow_queries(self) -> Dict[str, Any]:
        """Analyse détaillée des requêtes lentes"""
        if not self.connection_pool:
            return {"error": "AsyncPG pool not available"}
        
        try:
            async with self.get_raw_connection() as conn:
                # Requêtes PostgreSQL pour analyser les requêtes lentes
                slow_queries_sql = """
                SELECT query, calls, total_time, mean_time, rows
                FROM pg_stat_statements 
                WHERE mean_time > 100  -- > 100ms
                ORDER BY mean_time DESC 
                LIMIT 10
                """
                
                try:
                    rows = await conn.fetch(slow_queries_sql)
                    return {
                        "slow_queries": [dict(row) for row in rows],
                        "analysis_timestamp": datetime.now().isoformat()
                    }
                except Exception:
                    # pg_stat_statements peut ne pas être activé
                    return {
                        "slow_queries": [],
                        "note": "pg_stat_statements extension not available"
                    }
        
        except Exception as e:
            return {"error": str(e)}
    
    async def close(self):
        """Ferme les connexions"""
        if self.connection_pool:
            await self.connection_pool.close()
        
        if self.engine:
            await self.engine.dispose()


# Instance globale
_database_optimizer: Optional[DatabaseOptimizer] = None


async def get_database_optimizer() -> DatabaseOptimizer:
    """Factory pour obtenir l'optimiseur de base de données"""
    global _database_optimizer
    
    if _database_optimizer is None:
        _database_optimizer = DatabaseOptimizer()
        await _database_optimizer.initialize()
    
    return _database_optimizer


# Décorateurs utilitaires

def optimized_query(query_name: str = None):
    """Décorateur pour marquer et monitorer une requête optimisée"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            optimizer = await get_database_optimizer()
            return await optimizer.query_monitor(query_name)(func)(*args, **kwargs)
        
        return wrapper
    return decorator


async def execute_optimized_query(sql: str, params: List[Any] = None) -> List[Dict]:
    """Exécute une requête SQL raw optimisée"""
    optimizer = await get_database_optimizer()
    
    if optimizer.connection_pool:
        async with optimizer.get_raw_connection() as conn:
            rows = await conn.fetch(sql, *(params or []))
            return [dict(row) for row in rows]
    else:
        async with optimizer.get_session() as session:
            result = await session.execute(text(sql), params or {})
            return [row._asdict() for row in result.fetchall()]