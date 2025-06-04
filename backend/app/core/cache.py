"""
Module de cache distribué Redis multi-niveaux pour M&A Intelligence Platform
US-002: Implémentation cache sophistiqué avec TTL adaptatif et invalidation intelligente

Features:
- Cache multi-niveaux par type de données (enrichissement, scoring, exports)
- TTL adaptatif selon la fraîcheur et l'importance des données
- Pattern cache-aside avec fallback automatique
- Métriques détaillées (hit ratio, latence, usage mémoire)
- Invalidation intelligente avec dependencies tracking
- Compression automatique pour optimiser la mémoire
- Support async/await complet
"""

import asyncio
import json
import hashlib
import pickle
import gzip
import time
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Callable, Awaitable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from enum import Enum
import redis.asyncio as redis
from redis.asyncio import Redis
from redis.exceptions import RedisError, ConnectionError as RedisConnectionError

from app.config import settings
from app.core.constants import CACHE_TTL

# Configuration logging
logger = logging.getLogger(__name__)


class CacheLevel(str, Enum):
    """Niveaux de cache avec priorités différentes"""
    L1_MEMORY = "l1_memory"        # Cache mémoire ultra-rapide (non implémenté dans cette version)
    L2_REDIS = "l2_redis"          # Cache Redis principal
    L3_DATABASE = "l3_database"    # Fallback vers base de données


class CacheType(str, Enum):
    """Types de cache avec TTL et stratégies différentes"""
    # Enrichissement de données
    ENRICHMENT_PAPPERS = "enrichment_pappers"      # TTL: 24h - données légales stables
    ENRICHMENT_KASPR = "enrichment_kaspr"          # TTL: 6h - contacts peuvent changer
    ENRICHMENT_SOCIETE = "enrichment_societe"      # TTL: 12h - données financières
    ENRICHMENT_INFOGREFFE = "enrichment_infogreffe" # TTL: 48h - registre officiel
    
    # Scoring et analytics
    SCORING_MA = "scoring_ma"                      # TTL: 1h - calculs dynamiques
    SCORING_ML = "scoring_ml"                      # TTL: 4h - modèle ML
    
    # Exports et rapports
    EXPORT_CSV = "export_csv"                      # TTL: 30min - fichiers volumineux
    EXPORT_STATS = "export_stats"                 # TTL: 15min - statistiques dashboard
    
    # API externes (cache des réponses)
    API_EXTERNAL = "api_external"                 # TTL: 2h - réponses APIs tierces
    
    # Sessions et utilisateurs
    USER_SESSION = "user_session"                 # TTL: 8h - sessions utilisateur
    USER_PREFERENCES = "user_preferences"         # TTL: 24h - préférences UI


@dataclass
class CacheConfig:
    """Configuration d'un type de cache"""
    default_ttl: int              # TTL par défaut en secondes
    max_ttl: int                  # TTL maximum
    min_ttl: int                  # TTL minimum
    compress: bool = False        # Compression gzip
    serialize_method: str = "json" # json, pickle
    invalidation_dependencies: List[str] = None  # Clés à invalider ensemble
    adaptive_ttl: bool = False    # TTL adaptatif selon fréquence d'accès


class CacheMetrics:
    """Métriques de performance du cache"""
    
    def __init__(self):
        self.hits = 0
        self.misses = 0
        self.sets = 0
        self.deletes = 0
        self.errors = 0
        self.total_latency_ms = 0.0
        self.operations_count = 0
        self.memory_usage_bytes = 0
        self.start_time = time.time()
    
    @property
    def hit_ratio(self) -> float:
        """Calcule le hit ratio"""
        total = self.hits + self.misses
        return (self.hits / total * 100) if total > 0 else 0.0
    
    @property
    def average_latency_ms(self) -> float:
        """Latence moyenne des opérations"""
        return (self.total_latency_ms / self.operations_count) if self.operations_count > 0 else 0.0
    
    @property
    def uptime_seconds(self) -> int:
        """Temps de fonctionnement en secondes"""
        return int(time.time() - self.start_time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Export métriques en dictionnaire"""
        return {
            'hits': self.hits,
            'misses': self.misses,
            'sets': self.sets,
            'deletes': self.deletes,
            'errors': self.errors,
            'hit_ratio_percent': round(self.hit_ratio, 2),
            'average_latency_ms': round(self.average_latency_ms, 2),
            'memory_usage_mb': round(self.memory_usage_bytes / (1024 * 1024), 2),
            'uptime_seconds': self.uptime_seconds,
            'operations_per_second': round(self.operations_count / max(self.uptime_seconds, 1), 2)
        }


class DistributedCache:
    """
    Cache distribué Redis multi-niveaux avec fonctionnalités avancées
    
    Fonctionnalités:
    - TTL adaptatif par type de données
    - Compression automatique des gros objets
    - Métriques de performance détaillées
    - Invalidation en cascade avec dependencies
    - Fallback automatique en cas d'erreur Redis
    - Pattern cache-aside optimisé
    """
    
    def __init__(self, redis_url: str = None):
        self.redis_url = redis_url or settings.REDIS_URL or "redis://localhost:6379"
        self.redis: Optional[Redis] = None
        self.metrics = CacheMetrics()
        self._connected = False
        
        # Configuration des types de cache
        self.cache_configs = {
            CacheType.ENRICHMENT_PAPPERS: CacheConfig(
                default_ttl=86400,  # 24h
                max_ttl=172800,     # 48h
                min_ttl=3600,       # 1h
                compress=True,
                serialize_method="json",
                adaptive_ttl=True
            ),
            CacheType.ENRICHMENT_KASPR: CacheConfig(
                default_ttl=21600,  # 6h
                max_ttl=43200,      # 12h  
                min_ttl=1800,       # 30min
                compress=True,
                serialize_method="json",
                adaptive_ttl=True
            ),
            CacheType.SCORING_MA: CacheConfig(
                default_ttl=3600,   # 1h
                max_ttl=7200,       # 2h
                min_ttl=300,        # 5min
                compress=False,
                serialize_method="json",
                adaptive_ttl=True,
                invalidation_dependencies=["company_data_updated"]
            ),
            CacheType.EXPORT_CSV: CacheConfig(
                default_ttl=1800,   # 30min
                max_ttl=3600,       # 1h
                min_ttl=300,        # 5min
                compress=True,
                serialize_method="pickle"
            ),
            CacheType.API_EXTERNAL: CacheConfig(
                default_ttl=7200,   # 2h
                max_ttl=14400,      # 4h
                min_ttl=900,        # 15min
                compress=True,
                serialize_method="json"
            )
        }
    
    async def connect(self) -> bool:
        """Établit la connexion Redis"""
        try:
            self.redis = redis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=False,  # Pour supporter les données binaires
                max_connections=20,
                retry_on_timeout=True,
                retry_on_error=[RedisConnectionError],
                health_check_interval=30
            )
            
            # Test de connexion
            await self.redis.ping()
            self._connected = True
            
            logger.info(f"✅ Cache Redis connecté: {self.redis_url}")
            
            # Initialiser métriques depuis Redis si disponibles
            await self._load_metrics_from_redis()
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur connexion Redis: {e}")
            self._connected = False
            return False
    
    async def disconnect(self):
        """Ferme la connexion Redis"""
        if self.redis:
            await self._save_metrics_to_redis()
            await self.redis.close()
            self._connected = False
            logger.info("🔌 Cache Redis déconnecté")
    
    async def get(self, key: str, cache_type: CacheType = CacheType.API_EXTERNAL) -> Optional[Any]:
        """
        Récupère une valeur du cache avec métriques
        
        Args:
            key: Clé de cache
            cache_type: Type de cache pour configuration TTL
            
        Returns:
            Valeur désérialisée ou None si non trouvée
        """
        start_time = time.time()
        
        try:
            if not self._connected:
                await self.connect()
            
            if not self._connected:
                self.metrics.errors += 1
                return None
            
            # Génération clé complète avec préfixe
            full_key = self._generate_cache_key(key, cache_type)
            
            # Récupération depuis Redis
            cached_data = await self.redis.get(full_key)
            
            if cached_data is None:
                self.metrics.misses += 1
                logger.debug(f"🔍 Cache MISS: {full_key}")
                return None
            
            # Désérialisation
            config = self.cache_configs.get(cache_type)
            value = self._deserialize(cached_data, config)
            
            self.metrics.hits += 1
            logger.debug(f"✅ Cache HIT: {full_key}")
            
            # Mettre à jour TTL adaptatif si activé
            if config and config.adaptive_ttl:
                await self._update_adaptive_ttl(full_key, cache_type)
            
            return value
            
        except Exception as e:
            self.metrics.errors += 1
            logger.error(f"❌ Erreur cache GET {key}: {e}")
            return None
            
        finally:
            # Métriques de latence
            latency_ms = (time.time() - start_time) * 1000
            self.metrics.total_latency_ms += latency_ms
            self.metrics.operations_count += 1
    
    async def set(self, 
                  key: str, 
                  value: Any, 
                  cache_type: CacheType = CacheType.API_EXTERNAL,
                  ttl: Optional[int] = None) -> bool:
        """
        Stocke une valeur dans le cache
        
        Args:
            key: Clé de cache
            value: Valeur à stocker
            cache_type: Type de cache
            ttl: TTL personnalisé (optionnel)
            
        Returns:
            True si succès
        """
        start_time = time.time()
        
        try:
            if not self._connected:
                await self.connect()
            
            if not self._connected:
                self.metrics.errors += 1
                return False
            
            # Configuration cache
            config = self.cache_configs.get(cache_type)
            if not config:
                logger.warning(f"⚠️ Type cache inconnu: {cache_type}")
                config = self.cache_configs[CacheType.API_EXTERNAL]
            
            # Calcul TTL
            final_ttl = ttl or config.default_ttl
            final_ttl = max(config.min_ttl, min(config.max_ttl, final_ttl))
            
            # Sérialisation
            serialized_data = self._serialize(value, config)
            
            # Génération clé
            full_key = self._generate_cache_key(key, cache_type)
            
            # Stockage Redis avec TTL
            await self.redis.setex(full_key, final_ttl, serialized_data)
            
            self.metrics.sets += 1
            logger.debug(f"💾 Cache SET: {full_key} (TTL: {final_ttl}s)")
            
            # Gestion des dépendances d'invalidation
            if config.invalidation_dependencies:
                await self._register_dependencies(full_key, config.invalidation_dependencies)
            
            return True
            
        except Exception as e:
            self.metrics.errors += 1
            logger.error(f"❌ Erreur cache SET {key}: {e}")
            return False
            
        finally:
            latency_ms = (time.time() - start_time) * 1000
            self.metrics.total_latency_ms += latency_ms
            self.metrics.operations_count += 1
    
    async def delete(self, key: str, cache_type: CacheType = CacheType.API_EXTERNAL) -> bool:
        """Supprime une clé du cache"""
        try:
            if not self._connected:
                return False
            
            full_key = self._generate_cache_key(key, cache_type)
            result = await self.redis.delete(full_key)
            
            if result > 0:
                self.metrics.deletes += 1
                logger.debug(f"🗑️ Cache DELETE: {full_key}")
                return True
            
            return False
            
        except Exception as e:
            self.metrics.errors += 1
            logger.error(f"❌ Erreur cache DELETE {key}: {e}")
            return False
    
    async def invalidate_pattern(self, pattern: str, cache_type: CacheType = None) -> int:
        """
        Invalide toutes les clés correspondant à un pattern
        
        Args:
            pattern: Pattern de clés (ex: "companies:*")
            cache_type: Type de cache (optionnel)
            
        Returns:
            Nombre de clés supprimées
        """
        try:
            if not self._connected:
                return 0
            
            # Construction pattern complet
            if cache_type:
                full_pattern = f"ma_cache:{cache_type.value}:{pattern}"
            else:
                full_pattern = f"ma_cache:*:{pattern}"
            
            # Recherche clés
            keys = await self.redis.keys(full_pattern)
            
            if not keys:
                return 0
            
            # Suppression batch
            deleted = await self.redis.delete(*keys)
            
            self.metrics.deletes += deleted
            logger.info(f"🧹 Cache invalidation pattern '{pattern}': {deleted} clés supprimées")
            
            return deleted
            
        except Exception as e:
            self.metrics.errors += 1
            logger.error(f"❌ Erreur invalidation pattern {pattern}: {e}")
            return 0
    
    async def get_or_compute(self,
                           key: str,
                           compute_func: Callable[[], Awaitable[Any]],
                           cache_type: CacheType = CacheType.API_EXTERNAL,
                           ttl: Optional[int] = None,
                           force_refresh: bool = False) -> Any:
        """
        Pattern cache-aside: récupère du cache ou calcule la valeur
        
        Args:
            key: Clé de cache
            compute_func: Fonction async pour calculer la valeur
            cache_type: Type de cache
            ttl: TTL personnalisé
            force_refresh: Forcer le recalcul même si en cache
            
        Returns:
            Valeur du cache ou calculée
        """
        # Vérifier cache sauf si force_refresh
        if not force_refresh:
            cached_value = await self.get(key, cache_type)
            if cached_value is not None:
                return cached_value
        
        # Calculer valeur
        try:
            computed_value = await compute_func()
            
            # Stocker en cache
            if computed_value is not None:
                await self.set(key, computed_value, cache_type, ttl)
            
            return computed_value
            
        except Exception as e:
            logger.error(f"❌ Erreur compute function pour clé {key}: {e}")
            # Tentative de fallback cache en cas d'erreur compute
            if not force_refresh:
                return await self.get(key, cache_type)
            raise
    
    async def get_cache_info(self) -> Dict[str, Any]:
        """Informations détaillées sur le cache"""
        try:
            if not self._connected:
                return {"status": "disconnected"}
            
            # Info Redis
            redis_info = await self.redis.info()
            
            # Statistiques par type de cache
            cache_stats = {}
            for cache_type in CacheType:
                pattern = f"ma_cache:{cache_type.value}:*"
                keys = await self.redis.keys(pattern)
                cache_stats[cache_type.value] = {
                    'key_count': len(keys),
                    'config': self.cache_configs.get(cache_type, {}).default_ttl if self.cache_configs.get(cache_type) else 'unknown'
                }
            
            return {
                'status': 'connected',
                'redis_info': {
                    'version': redis_info.get('redis_version'),
                    'memory_used_mb': round(redis_info.get('used_memory', 0) / (1024 * 1024), 2),
                    'memory_peak_mb': round(redis_info.get('used_memory_peak', 0) / (1024 * 1024), 2),
                    'connected_clients': redis_info.get('connected_clients', 0),
                    'keyspace_hits': redis_info.get('keyspace_hits', 0),
                    'keyspace_misses': redis_info.get('keyspace_misses', 0),
                    'uptime_seconds': redis_info.get('uptime_in_seconds', 0)
                },
                'cache_metrics': self.metrics.to_dict(),
                'cache_stats_by_type': cache_stats
            }
            
        except Exception as e:
            logger.error(f"❌ Erreur info cache: {e}")
            return {"status": "error", "error": str(e)}
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check du cache"""
        try:
            start_time = time.time()
            
            if not self._connected:
                await self.connect()
            
            if not self._connected:
                return {
                    'status': 'unhealthy',
                    'error': 'Connection failed'
                }
            
            # Test ping
            pong = await self.redis.ping()
            ping_latency = (time.time() - start_time) * 1000
            
            # Test set/get
            test_key = "health_check_test"
            test_value = {"timestamp": time.time()}
            
            await self.set(test_key, test_value, CacheType.API_EXTERNAL, ttl=60)
            retrieved_value = await self.get(test_key, CacheType.API_EXTERNAL)
            await self.delete(test_key, CacheType.API_EXTERNAL)
            
            operations_ok = retrieved_value is not None
            
            status = 'healthy' if pong and operations_ok else 'degraded'
            
            return {
                'status': status,
                'ping_success': bool(pong),
                'ping_latency_ms': round(ping_latency, 2),
                'operations_test': operations_ok,
                'hit_ratio_percent': round(self.metrics.hit_ratio, 2),
                'error_count': self.metrics.errors
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e)
            }
    
    # Méthodes privées utilitaires
    
    def _generate_cache_key(self, key: str, cache_type: CacheType) -> str:
        """Génère une clé de cache complète avec préfixe"""
        # Format: ma_cache:type:hashed_key
        key_hash = hashlib.md5(key.encode()).hexdigest()[:16]
        return f"ma_cache:{cache_type.value}:{key_hash}"
    
    def _serialize(self, value: Any, config: CacheConfig) -> bytes:
        """Sérialise une valeur selon la configuration"""
        try:
            if config.serialize_method == "pickle":
                serialized = pickle.dumps(value)
            else:  # JSON par défaut
                serialized = json.dumps(value, ensure_ascii=False, default=str).encode('utf-8')
            
            # Compression si activée
            if config.compress:
                serialized = gzip.compress(serialized)
            
            return serialized
            
        except Exception as e:
            logger.error(f"❌ Erreur sérialisation: {e}")
            raise
    
    def _deserialize(self, data: bytes, config: CacheConfig) -> Any:
        """Désérialise une valeur selon la configuration"""
        try:
            # Décompression si nécessaire
            if config.compress:
                data = gzip.decompress(data)
            
            # Désérialisation
            if config.serialize_method == "pickle":
                return pickle.loads(data)
            else:  # JSON
                return json.loads(data.decode('utf-8'))
                
        except Exception as e:
            logger.error(f"❌ Erreur désérialisation: {e}")
            raise
    
    async def _update_adaptive_ttl(self, key: str, cache_type: CacheType):
        """Met à jour le TTL de façon adaptative selon la fréquence d'accès"""
        try:
            config = self.cache_configs.get(cache_type)
            if not config or not config.adaptive_ttl:
                return
            
            # Incrémenter compteur d'accès
            access_key = f"{key}:access_count"
            access_count = await self.redis.incr(access_key)
            await self.redis.expire(access_key, 3600)  # Reset chaque heure
            
            # Ajuster TTL selon fréquence
            if access_count > 10:  # Très fréquent
                new_ttl = min(config.max_ttl, config.default_ttl * 2)
            elif access_count > 5:  # Fréquent
                new_ttl = int(config.default_ttl * 1.5)
            else:  # Normal
                new_ttl = config.default_ttl
            
            await self.redis.expire(key, new_ttl)
            
        except Exception as e:
            logger.warning(f"⚠️ Erreur TTL adaptatif: {e}")
    
    async def _register_dependencies(self, key: str, dependencies: List[str]):
        """Enregistre les dépendances pour invalidation en cascade"""
        try:
            for dep in dependencies:
                dep_set_key = f"ma_cache:dependencies:{dep}"
                await self.redis.sadd(dep_set_key, key)
                await self.redis.expire(dep_set_key, 86400)  # 24h
        except Exception as e:
            logger.warning(f"⚠️ Erreur enregistrement dépendances: {e}")
    
    async def _save_metrics_to_redis(self):
        """Sauvegarde les métriques dans Redis pour persistance"""
        try:
            metrics_key = "ma_cache:metrics"
            metrics_data = self.metrics.to_dict()
            await self.redis.setex(metrics_key, 86400, json.dumps(metrics_data))
        except Exception as e:
            logger.warning(f"⚠️ Erreur sauvegarde métriques: {e}")
    
    async def _load_metrics_from_redis(self):
        """Charge les métriques depuis Redis"""
        try:
            metrics_key = "ma_cache:metrics"
            metrics_data = await self.redis.get(metrics_key)
            if metrics_data:
                saved_metrics = json.loads(metrics_data)
                # Restaurer certaines métriques (pas toutes pour éviter les doublons)
                self.metrics.start_time = time.time() - saved_metrics.get('uptime_seconds', 0)
        except Exception as e:
            logger.warning(f"⚠️ Erreur chargement métriques: {e}")


# Instance globale du cache
_cache_instance: Optional[DistributedCache] = None


async def get_cache() -> DistributedCache:
    """Factory pour obtenir l'instance de cache (singleton)"""
    global _cache_instance
    
    if _cache_instance is None:
        _cache_instance = DistributedCache()
        await _cache_instance.connect()
    
    return _cache_instance


@asynccontextmanager
async def cache_context():
    """Context manager pour gestion automatique du cache"""
    cache = await get_cache()
    try:
        yield cache
    finally:
        # Optionnel: on peut laisser la connexion ouverte pour réutilisation
        pass


# Décorateurs utilitaires

def cached(cache_type: CacheType = CacheType.API_EXTERNAL, ttl: Optional[int] = None):
    """
    Décorateur pour mettre en cache automatiquement le résultat d'une fonction
    
    Usage:
        @cached(CacheType.SCORING_MA, ttl=3600)
        async def calculate_ma_score(siren: str) -> dict:
            # calculs coûteux
            return score_data
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Générer clé de cache à partir des arguments
            cache_key = f"{func.__name__}:{hash(str(args) + str(sorted(kwargs.items())))}"
            
            cache = await get_cache()
            
            async def compute():
                return await func(*args, **kwargs)
            
            return await cache.get_or_compute(cache_key, compute, cache_type, ttl)
        
        return wrapper
    return decorator