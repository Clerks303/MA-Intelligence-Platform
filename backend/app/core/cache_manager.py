"""
Gestionnaire de cache multi-niveaux pour M&A Intelligence Platform
US-005: Cache strategy avec Redis + mémoire pour performance optimale

Features:
- Cache multi-niveaux (mémoire L1 + Redis L2)
- TTL configurables par type de données
- Compression automatique pour gros objets
- Invalidation intelligente par tags
- Métriques de performance intégrées
- Patterns de cache adaptatifs
"""

import asyncio
import json
import pickle
import hashlib
import gzip
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Callable, Set
from functools import wraps
from dataclasses import dataclass, field
from enum import Enum
import redis.asyncio as aioredis
from redis.asyncio.connection import ConnectionPool
import logging
from collections import defaultdict, OrderedDict

from app.config import settings
from app.core.logging_system import get_logger, LogCategory

logger = get_logger("cache_manager", LogCategory.PERFORMANCE)


class CacheLevel(str, Enum):
    """Niveaux de cache"""
    MEMORY = "memory"      # L1 - Cache mémoire local
    REDIS = "redis"        # L2 - Cache Redis distribué
    BOTH = "both"          # L1 + L2


class CacheStrategy(str, Enum):
    """Stratégies de cache"""
    WRITE_THROUGH = "write_through"      # Write to cache and database
    WRITE_BEHIND = "write_behind"        # Write to cache, async to database
    CACHE_ASIDE = "cache_aside"          # Load from database on miss
    READ_THROUGH = "read_through"        # Load through cache layer


@dataclass
class CacheConfig:
    """Configuration du cache par type de données"""
    ttl_seconds: int = 300              # 5 minutes par défaut
    level: CacheLevel = CacheLevel.BOTH
    strategy: CacheStrategy = CacheStrategy.CACHE_ASIDE
    compress: bool = False              # Compression pour objets > 1KB
    max_size_mb: float = 1.0           # Taille max avant compression
    invalidation_tags: Set[str] = field(default_factory=set)
    priority: int = 1                   # Priorité éviction (1=low, 5=high)
    refresh_ahead: bool = False         # Refresh async avant expiration


@dataclass
class CacheStats:
    """Statistiques du cache"""
    hits: int = 0
    misses: int = 0
    sets: int = 0
    deletes: int = 0
    size_bytes: int = 0
    last_reset: datetime = field(default_factory=datetime.now)
    
    @property
    def hit_ratio(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class MemoryCache:
    """Cache mémoire L1 avec LRU et gestion TTL"""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache: OrderedDict = OrderedDict()
        self.expiry: Dict[str, float] = {}
        self.stats = CacheStats()
        self.tags: Dict[str, Set[str]] = defaultdict(set)
    
    def _is_expired(self, key: str) -> bool:
        """Vérifie si la clé est expirée"""
        if key not in self.expiry:
            return False
        return time.time() > self.expiry[key]
    
    def _evict_expired(self):
        """Supprime les entrées expirées"""
        current_time = time.time()
        expired_keys = [
            key for key, expiry in self.expiry.items()
            if current_time > expiry
        ]
        for key in expired_keys:
            self._remove_key(key)
    
    def _remove_key(self, key: str):
        """Supprime une clé du cache"""
        if key in self.cache:
            del self.cache[key]
        if key in self.expiry:
            del self.expiry[key]
        
        # Nettoyer tags
        for tag_keys in self.tags.values():
            tag_keys.discard(key)
    
    def _evict_lru(self):
        """Éviction LRU si nécessaire"""
        while len(self.cache) >= self.max_size:
            oldest_key = next(iter(self.cache))
            self._remove_key(oldest_key)
    
    def get(self, key: str) -> Optional[Any]:
        """Récupère une valeur du cache"""
        self._evict_expired()
        
        if key not in self.cache or self._is_expired(key):
            self.stats.misses += 1
            return None
        
        # Déplacer à la fin (LRU)
        value = self.cache.pop(key)
        self.cache[key] = value
        self.stats.hits += 1
        return value
    
    def set(self, key: str, value: Any, ttl_seconds: int = 300, tags: Set[str] = None):
        """Stocke une valeur dans le cache"""
        self._evict_expired()
        self._evict_lru()
        
        self.cache[key] = value
        self.expiry[key] = time.time() + ttl_seconds
        self.stats.sets += 1
        
        # Gestion des tags
        if tags:
            for tag in tags:
                self.tags[tag].add(key)
    
    def delete(self, key: str):
        """Supprime une clé"""
        if key in self.cache:
            self._remove_key(key)
            self.stats.deletes += 1
    
    def invalidate_by_tag(self, tag: str):
        """Invalide toutes les clés avec un tag"""
        if tag in self.tags:
            keys_to_remove = list(self.tags[tag])
            for key in keys_to_remove:
                self.delete(key)
            del self.tags[tag]
    
    def clear(self):
        """Vide le cache"""
        self.cache.clear()
        self.expiry.clear()
        self.tags.clear()
        self.stats = CacheStats()


class CacheManager:
    """Gestionnaire de cache multi-niveaux avec Redis et mémoire"""
    
    def __init__(self):
        self.memory_cache = MemoryCache(max_size=1000)
        self.redis_pool: Optional[ConnectionPool] = None
        self.redis_client: Optional[aioredis.Redis] = None
        self.configs: Dict[str, CacheConfig] = {}
        self.stats = {
            'memory': CacheStats(),
            'redis': CacheStats(),
            'total': CacheStats()
        }
        
        # Configuration par défaut par type de données
        self._setup_default_configs()
        
        logger.info("🚀 CacheManager initialisé")
    
    async def initialize(self):
        """Initialise les connexions Redis"""
        try:
            redis_url = getattr(settings, 'REDIS_URL', 'redis://localhost:6379/0')
            
            self.redis_pool = ConnectionPool.from_url(
                redis_url,
                max_connections=20,
                retry_on_timeout=True,
                health_check_interval=30
            )
            
            self.redis_client = aioredis.Redis(
                connection_pool=self.redis_pool,
                decode_responses=False  # Pour supporter pickle
            )
            
            # Test connexion
            await self.redis_client.ping()
            logger.info("✅ Redis connecté avec succès")
            
        except Exception as e:
            logger.warning(f"⚠️ Redis non disponible: {e}")
            self.redis_client = None
    
    def _setup_default_configs(self):
        """Configuration par défaut des types de cache"""
        
        self.configs.update({
            # Données companies (cache long)
            'companies': CacheConfig(
                ttl_seconds=3600,  # 1 heure
                level=CacheLevel.BOTH,
                compress=True,
                invalidation_tags={'companies', 'company_data'},
                priority=3
            ),
            
            # Stats dashboard (cache moyen)
            'stats': CacheConfig(
                ttl_seconds=300,  # 5 minutes
                level=CacheLevel.BOTH,
                invalidation_tags={'stats', 'dashboard'},
                priority=4
            ),
            
            # Résultats scraping (cache court)
            'scraping': CacheConfig(
                ttl_seconds=1800,  # 30 minutes
                level=CacheLevel.REDIS,  # Partagé entre workers
                compress=True,
                invalidation_tags={'scraping', 'company_data'},
                priority=2
            ),
            
            # Recherches (cache très court)
            'search': CacheConfig(
                ttl_seconds=600,  # 10 minutes
                level=CacheLevel.MEMORY,  # Local seulement
                invalidation_tags={'search'},
                priority=1
            ),
            
            # Sessions utilisateur
            'sessions': CacheConfig(
                ttl_seconds=1800,  # 30 minutes
                level=CacheLevel.REDIS,
                invalidation_tags={'auth', 'sessions'},
                priority=5
            ),
            
            # API external responses
            'api_responses': CacheConfig(
                ttl_seconds=7200,  # 2 heures
                level=CacheLevel.BOTH,
                compress=True,
                invalidation_tags={'external_api'},
                priority=2
            )
        })
    
    def _get_cache_key(self, namespace: str, key: str) -> str:
        """Génère une clé de cache préfixée"""
        return f"ma_intelligence:{namespace}:{key}"
    
    def _serialize_value(self, value: Any, compress: bool = False) -> bytes:
        """Sérialise une valeur pour le stockage"""
        try:
            # Sérialisation pickle pour performance
            data = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Compression si demandée et si gain significatif
            if compress and len(data) > 1024:  # > 1KB
                compressed = gzip.compress(data)
                if len(compressed) < len(data) * 0.8:  # Gain > 20%
                    return b'gzip:' + compressed
            
            return b'raw:' + data
            
        except Exception as e:
            logger.error(f"Erreur sérialisation: {e}")
            return b'json:' + json.dumps(str(value)).encode()
    
    def _deserialize_value(self, data: bytes) -> Any:
        """Désérialise une valeur stockée"""
        try:
            if data.startswith(b'gzip:'):
                decompressed = gzip.decompress(data[5:])
                return pickle.loads(decompressed)
            elif data.startswith(b'raw:'):
                return pickle.loads(data[4:])
            elif data.startswith(b'json:'):
                return json.loads(data[5:].decode())
            else:
                # Fallback pickle direct
                return pickle.loads(data)
                
        except Exception as e:
            logger.error(f"Erreur désérialisation: {e}")
            return None
    
    async def get(self, namespace: str, key: str) -> Optional[Any]:
        """Récupère une valeur du cache (L1 puis L2)"""
        config = self.configs.get(namespace, CacheConfig())
        cache_key = self._get_cache_key(namespace, key)
        
        # Essai cache mémoire L1
        if config.level in [CacheLevel.MEMORY, CacheLevel.BOTH]:
            value = self.memory_cache.get(cache_key)
            if value is not None:
                self.stats['memory'].hits += 1
                self.stats['total'].hits += 1
                return value
            self.stats['memory'].misses += 1
        
        # Essai cache Redis L2
        if config.level in [CacheLevel.REDIS, CacheLevel.BOTH] and self.redis_client:
            try:
                data = await self.redis_client.get(cache_key)
                if data:
                    value = self._deserialize_value(data)
                    if value is not None:
                        # Remplir L1 si configuré
                        if config.level == CacheLevel.BOTH:
                            self.memory_cache.set(
                                cache_key, 
                                value, 
                                config.ttl_seconds,
                                config.invalidation_tags
                            )
                        
                        self.stats['redis'].hits += 1
                        self.stats['total'].hits += 1
                        return value
                
                self.stats['redis'].misses += 1
                
            except Exception as e:
                logger.error(f"Erreur lecture Redis: {e}")
        
        self.stats['total'].misses += 1
        return None
    
    async def set(self, namespace: str, key: str, value: Any, ttl_seconds: Optional[int] = None):
        """Stocke une valeur dans le cache"""
        config = self.configs.get(namespace, CacheConfig())
        cache_key = self._get_cache_key(namespace, key)
        ttl = ttl_seconds or config.ttl_seconds
        
        # Stockage L1
        if config.level in [CacheLevel.MEMORY, CacheLevel.BOTH]:
            self.memory_cache.set(
                cache_key, 
                value, 
                ttl,
                config.invalidation_tags
            )
            self.stats['memory'].sets += 1
        
        # Stockage L2
        if config.level in [CacheLevel.REDIS, CacheLevel.BOTH] and self.redis_client:
            try:
                data = self._serialize_value(value, config.compress)
                await self.redis_client.setex(cache_key, ttl, data)
                
                # Tags pour invalidation
                if config.invalidation_tags:
                    for tag in config.invalidation_tags:
                        tag_key = f"tag:{tag}"
                        await self.redis_client.sadd(tag_key, cache_key)
                        await self.redis_client.expire(tag_key, ttl + 3600)  # +1h pour cleanup
                
                self.stats['redis'].sets += 1
                
            except Exception as e:
                logger.error(f"Erreur écriture Redis: {e}")
        
        self.stats['total'].sets += 1
    
    async def delete(self, namespace: str, key: str):
        """Supprime une clé du cache"""
        cache_key = self._get_cache_key(namespace, key)
        
        # Suppression L1
        self.memory_cache.delete(cache_key)
        self.stats['memory'].deletes += 1
        
        # Suppression L2
        if self.redis_client:
            try:
                await self.redis_client.delete(cache_key)
                self.stats['redis'].deletes += 1
            except Exception as e:
                logger.error(f"Erreur suppression Redis: {e}")
        
        self.stats['total'].deletes += 1
    
    async def invalidate_by_tag(self, tag: str):
        """Invalide toutes les clés avec un tag"""
        
        # Invalidation L1
        self.memory_cache.invalidate_by_tag(tag)
        
        # Invalidation L2
        if self.redis_client:
            try:
                tag_key = f"tag:{tag}"
                keys = await self.redis_client.smembers(tag_key)
                
                if keys:
                    # Suppression par batch pour performance
                    pipe = self.redis_client.pipeline()
                    for key in keys:
                        pipe.delete(key)
                    await pipe.execute()
                    
                    # Nettoyer le tag
                    await self.redis_client.delete(tag_key)
                    
                logger.info(f"🗑️ Invalidation tag '{tag}': {len(keys)} clés supprimées")
                
            except Exception as e:
                logger.error(f"Erreur invalidation tag: {e}")
    
    async def clear_all(self):
        """Vide complètement tous les caches"""
        
        # Clear L1
        self.memory_cache.clear()
        
        # Clear L2 
        if self.redis_client:
            try:
                # Supprimer toutes les clés avec notre préfixe
                keys = await self.redis_client.keys("ma_intelligence:*")
                if keys:
                    await self.redis_client.delete(*keys)
                    
                # Supprimer tous les tags
                tag_keys = await self.redis_client.keys("tag:*")
                if tag_keys:
                    await self.redis_client.delete(*tag_keys)
                    
                logger.info(f"🗑️ Cache Redis vidé: {len(keys)} clés supprimées")
                
            except Exception as e:
                logger.error(f"Erreur clear Redis: {e}")
        
        # Reset stats
        self.stats = {
            'memory': CacheStats(),
            'redis': CacheStats(), 
            'total': CacheStats()
        }
        
        logger.info("🧹 Tous les caches vidés")
    
    def get_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques de cache"""
        redis_info = {}
        
        # Stats Redis si disponible
        if self.redis_client:
            try:
                # Infos Redis synchrones (approximatives)
                redis_info = {
                    'connected': True,
                    'memory_usage': 'unknown',  # Nécessiterait une requête async
                    'keys_count': 'unknown'
                }
            except:
                redis_info = {'connected': False}
        else:
            redis_info = {'connected': False}
        
        return {
            'memory_cache': {
                'hits': self.stats['memory'].hits,
                'misses': self.stats['memory'].misses,
                'hit_ratio': self.stats['memory'].hit_ratio,
                'size': len(self.memory_cache.cache),
                'max_size': self.memory_cache.max_size
            },
            'redis_cache': {
                'hits': self.stats['redis'].hits,
                'misses': self.stats['redis'].misses,
                'hit_ratio': self.stats['redis'].hit_ratio,
                **redis_info
            },
            'total': {
                'hits': self.stats['total'].hits,
                'misses': self.stats['total'].misses,
                'hit_ratio': self.stats['total'].hit_ratio,
                'sets': self.stats['total'].sets,
                'deletes': self.stats['total'].deletes
            },
            'configs': {
                name: {
                    'ttl_seconds': config.ttl_seconds,
                    'level': config.level.value,
                    'compress': config.compress,
                    'tags': list(config.invalidation_tags)
                }
                for name, config in self.configs.items()
            }
        }
    
    async def close(self):
        """Ferme les connexions"""
        if self.redis_client:
            await self.redis_client.close()
        if self.redis_pool:
            await self.redis_pool.disconnect()


# Instance globale du cache manager
_cache_manager: Optional[CacheManager] = None


async def get_cache_manager() -> CacheManager:
    """Factory pour obtenir le gestionnaire de cache"""
    global _cache_manager
    
    if _cache_manager is None:
        _cache_manager = CacheManager()
        await _cache_manager.initialize()
    
    return _cache_manager


# Décorateurs pour faciliter l'utilisation

def cached(namespace: str, key_func: Callable = None, ttl_seconds: Optional[int] = None):
    """Décorateur pour mettre en cache le résultat d'une fonction"""
    
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            cache_manager = await get_cache_manager()
            
            # Génération de la clé
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                # Clé par défaut basée sur nom fonction + hash des arguments
                args_str = str(args) + str(sorted(kwargs.items()))
                cache_key = f"{func.__name__}:{hashlib.md5(args_str.encode()).hexdigest()}"
            
            # Essai récupération cache
            result = await cache_manager.get(namespace, cache_key)
            if result is not None:
                return result
            
            # Exécution fonction et mise en cache
            result = await func(*args, **kwargs)
            await cache_manager.set(namespace, cache_key, result, ttl_seconds)
            
            return result
        
        return wrapper
    return decorator


def cache_invalidate(namespace: str, key: str = None, tag: str = None):
    """Décorateur pour invalider le cache après exécution"""
    
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            result = await func(*args, **kwargs)
            
            cache_manager = await get_cache_manager()
            
            if tag:
                await cache_manager.invalidate_by_tag(tag)
            elif key:
                await cache_manager.delete(namespace, key)
            
            return result
        
        return wrapper
    return decorator


# Fonctions utilitaires

async def warm_cache():
    """Préchauffe le cache avec les données fréquemment utilisées"""
    logger.info("🔥 Préchauffage du cache...")
    
    try:
        # Ici on pourrait précharger:
        # - Stats dashboard
        # - Companies les plus consultées
        # - Configuration système
        
        # Exemple avec stats
        cache_manager = await get_cache_manager()
        
        # Simuler des données stats pour le préchauffage
        mock_stats = {
            'total_companies': 0,
            'last_scraping': None,
            'avg_processing_time': 0,
            'success_rate': 100.0
        }
        
        await cache_manager.set('stats', 'dashboard_overview', mock_stats)
        
        logger.info("✅ Préchauffage terminé")
        
    except Exception as e:
        logger.error(f"Erreur préchauffage cache: {e}")


async def cleanup_expired_cache():
    """Nettoie le cache expiré (tâche de maintenance)"""
    cache_manager = await get_cache_manager()
    
    # Le cleanup L1 se fait automatiquement
    # Pour L2 Redis, on fait confiance à l'expiration automatique
    
    logger.info("🧹 Nettoyage cache expiré terminé")