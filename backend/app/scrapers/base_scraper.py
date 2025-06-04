"""
Base scraper avec cache intégré pour tous les scrapers
Évite la duplication de code et standardise l'interface
"""

import asyncio
import aiohttp
import logging
import json
import hashlib
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class CacheManager:
    """Gestionnaire de cache simple en mémoire (remplacer par Redis en production)"""
    
    def __init__(self):
        self.cache = {}
        self.cache_ttl = {}
        
    def get(self, key: str) -> Optional[Any]:
        """Récupère une valeur du cache"""
        if key in self.cache:
            # Vérifier TTL
            if datetime.now() > self.cache_ttl.get(key, datetime.min):
                del self.cache[key]
                if key in self.cache_ttl:
                    del self.cache_ttl[key]
                return None
            return self.cache[key]
        return None
    
    def set(self, key: str, value: Any, ttl_seconds: int = 3600):
        """Stocke une valeur dans le cache"""
        self.cache[key] = value
        self.cache_ttl[key] = datetime.now() + timedelta(seconds=ttl_seconds)
    
    def delete(self, key: str):
        """Supprime une clé du cache"""
        if key in self.cache:
            del self.cache[key]
        if key in self.cache_ttl:
            del self.cache_ttl[key]
    
    def clear(self):
        """Vide le cache"""
        self.cache.clear()
        self.cache_ttl.clear()

# Instance globale du cache
cache_manager = CacheManager()

class BaseScraper(ABC):
    """Classe de base pour tous les scrapers avec cache intégré"""
    
    def __init__(self, db_client=None, cache_ttl: int = 3600):
        self.db = db_client
        self.cache_ttl = cache_ttl
        self.session = None
        self.stats = {
            'requests_total': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'errors': 0
        }
    
    async def __aenter__(self):
        """Initialisation asynchrone"""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Nettoyage asynchrone"""
        if self.session:
            await self.session.close()
    
    def _cache_key(self, identifier: str, suffix: str = "") -> str:
        """Génère une clé de cache"""
        base_key = f"{self.__class__.__name__}:{identifier}"
        if suffix:
            base_key += f":{suffix}"
        return base_key
    
    async def _cached_request(self, identifier: str, fetch_func, *args, **kwargs) -> Optional[Dict]:
        """Effectue une requête avec cache"""
        cache_key = self._cache_key(identifier)
        
        # Vérifier le cache
        cached_result = cache_manager.get(cache_key)
        if cached_result is not None:
            self.stats['cache_hits'] += 1
            logger.debug(f"Cache hit for {cache_key}")
            return cached_result
        
        # Pas en cache, effectuer la requête
        self.stats['cache_misses'] += 1
        self.stats['requests_total'] += 1
        
        try:
            result = await fetch_func(*args, **kwargs)
            if result:
                cache_manager.set(cache_key, result, self.cache_ttl)
                logger.debug(f"Cached result for {cache_key}")
            return result
        except Exception as e:
            self.stats['errors'] += 1
            logger.error(f"Error in {self.__class__.__name__}: {e}")
            return None
    
    @abstractmethod
    async def search_companies(self, **criteria) -> List[Dict]:
        """Recherche d'entreprises - à implémenter dans les sous-classes"""
        pass
    
    @abstractmethod
    async def get_company_details(self, identifier: str) -> Optional[Dict]:
        """Détails d'une entreprise - à implémenter dans les sous-classes"""
        pass
    
    def get_stats(self) -> Dict:
        """Retourne les statistiques du scraper"""
        total_requests = self.stats['cache_hits'] + self.stats['cache_misses']
        cache_ratio = (self.stats['cache_hits'] / total_requests * 100) if total_requests > 0 else 0
        
        return {
            **self.stats,
            'cache_ratio_percent': round(cache_ratio, 2),
            'efficiency': 'high' if cache_ratio > 70 else 'medium' if cache_ratio > 30 else 'low'
        }
    
    async def health_check(self) -> Dict:
        """Vérification de santé du scraper"""
        return {
            'status': 'healthy',
            'scraper': self.__class__.__name__,
            'cache_status': 'active',
            'session_active': self.session is not None,
            'stats': self.get_stats()
        }

class RateLimitedScraper(BaseScraper):
    """Scraper avec rate limiting intégré"""
    
    def __init__(self, db_client=None, cache_ttl: int = 3600, 
                 rate_limit: int = 10, rate_window: int = 60):
        super().__init__(db_client, cache_ttl)
        self.rate_limit = rate_limit
        self.rate_window = rate_window
        self.request_times = []
    
    async def _rate_limited_request(self, fetch_func, *args, **kwargs):
        """Effectue une requête avec rate limiting"""
        now = datetime.now()
        
        # Nettoyer les anciens timestamps
        cutoff = now - timedelta(seconds=self.rate_window)
        self.request_times = [t for t in self.request_times if t > cutoff]
        
        # Vérifier le rate limit
        if len(self.request_times) >= self.rate_limit:
            sleep_time = self.rate_window - (now - self.request_times[0]).total_seconds()
            if sleep_time > 0:
                logger.info(f"Rate limit reached, sleeping for {sleep_time:.2f}s")
                await asyncio.sleep(sleep_time)
        
        # Enregistrer cette requête
        self.request_times.append(now)
        
        return await fetch_func(*args, **kwargs)

# Utility functions
def normalize_siren(siren: str) -> str:
    """Normalise un numéro SIREN"""
    return ''.join(filter(str.isdigit, str(siren)))

def is_valid_siren(siren: str) -> bool:
    """Vérifie la validité d'un SIREN"""
    normalized = normalize_siren(siren)
    return len(normalized) == 9 and normalized.isdigit()

def hash_query(query_params: Dict) -> str:
    """Hash les paramètres de requête pour le cache"""
    query_str = json.dumps(query_params, sort_keys=True)
    return hashlib.md5(query_str.encode()).hexdigest()

__all__ = [
    'BaseScraper', 'RateLimitedScraper', 'CacheManager', 
    'cache_manager', 'normalize_siren', 'is_valid_siren', 'hash_query'
]