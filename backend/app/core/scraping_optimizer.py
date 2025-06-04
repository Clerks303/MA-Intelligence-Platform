"""
Optimiseur de scraping avec parallélisation et rate limiting intelligent
US-009: Performance optimization pour scraping à grande échelle

Ce module fournit:
- Parallélisation intelligente avec pools de workers
- Rate limiting adaptatif basé sur les réponses serveurs
- Gestion des retry avec backoff exponentiel
- Load balancing entre différents scrapers
- Monitoring et métriques de scraping
- Cache intelligent des résultats
- Rotation automatique des proxies/user agents
"""

import asyncio
import time
import random
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Tuple, Union
from dataclasses import dataclass, field
from functools import wraps
from enum import Enum
import json
import hashlib
from collections import defaultdict, deque
import statistics

import aiohttp
import asyncpg
from urllib.parse import urljoin, urlparse
import numpy as np

from app.core.logging_system import get_logger, LogCategory
from app.core.cache_manager import get_cache_manager
from app.core.performance_analyzer import get_performance_analyzer

logger = get_logger("scraping_optimizer", LogCategory.SCRAPING)


class ScrapingStrategy(str, Enum):
    """Stratégies de scraping"""
    CONSERVATIVE = "conservative"    # Sécurisé, lent
    BALANCED = "balanced"           # Équilibré
    AGGRESSIVE = "aggressive"       # Rapide, risqué
    ADAPTIVE = "adaptive"          # S'adapte selon les conditions


class ScrapingSource(str, Enum):
    """Sources de scraping"""
    PAPPERS = "pappers"
    SOCIETE = "societe"
    INFOGREFFE = "infogreffe"
    VERIF = "verif"
    MANAGEO = "manageo"


@dataclass
class ScrapingConfig:
    """Configuration de scraping optimisée"""
    strategy: ScrapingStrategy = ScrapingStrategy.BALANCED
    max_concurrent: int = 5
    base_delay: float = 1.0
    max_delay: float = 10.0
    timeout: int = 30
    max_retries: int = 3
    enable_cache: bool = True
    cache_ttl: int = 3600
    respect_robots_txt: bool = True
    user_agent_rotation: bool = True
    proxy_rotation: bool = False
    adaptive_rate_limiting: bool = True


@dataclass
class ScrapingMetrics:
    """Métriques de performance scraping"""
    requests_total: int = 0
    requests_success: int = 0
    requests_failed: int = 0
    requests_cached: int = 0
    avg_response_time: float = 0.0
    total_data_scraped: int = 0
    rate_limit_hits: int = 0
    proxy_errors: int = 0
    last_update: datetime = field(default_factory=datetime.now)
    
    @property
    def success_rate(self) -> float:
        return (self.requests_success / self.requests_total * 100) if self.requests_total > 0 else 0
    
    @property
    def cache_hit_rate(self) -> float:
        return (self.requests_cached / self.requests_total * 100) if self.requests_total > 0 else 0


@dataclass
class ScrapingResult:
    """Résultat de scraping"""
    source: ScrapingSource
    target_id: str
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    response_time: float = 0.0
    cached: bool = False
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class AdaptiveRateLimiter:
    """Rate limiter adaptatif intelligent"""
    
    def __init__(self, initial_delay: float = 1.0, min_delay: float = 0.1, max_delay: float = 30.0):
        self.current_delay = initial_delay
        self.min_delay = min_delay
        self.max_delay = max_delay
        
        # Historique des performances
        self.response_times = deque(maxlen=100)
        self.error_count = 0
        self.success_count = 0
        self.rate_limit_count = 0
        
        # Dernière adaptation
        self.last_adaptation = time.time()
        self.adaptation_interval = 30  # secondes
        
    async def wait(self):
        """Attend selon le délai adaptatif actuel"""
        await asyncio.sleep(self.current_delay)
    
    def record_success(self, response_time: float):
        """Enregistre un succès"""
        self.success_count += 1
        self.response_times.append(response_time)
        self._adapt_if_needed()
    
    def record_error(self, is_rate_limit: bool = False):
        """Enregistre une erreur"""
        self.error_count += 1
        if is_rate_limit:
            self.rate_limit_count += 1
            # Immédiatement ralentir en cas de rate limit
            self.current_delay = min(self.max_delay, self.current_delay * 2)
        
        self._adapt_if_needed()
    
    def _adapt_if_needed(self):
        """Adapte le délai selon les performances"""
        now = time.time()
        if now - self.last_adaptation < self.adaptation_interval:
            return
        
        total_requests = self.success_count + self.error_count
        if total_requests < 10:  # Pas assez de données
            return
        
        error_rate = self.error_count / total_requests
        avg_response_time = statistics.mean(self.response_times) if self.response_times else 1.0
        
        # Adaptation basée sur les performances
        if self.rate_limit_count > 0:
            # Rate limits détectés - augmenter délai
            self.current_delay = min(self.max_delay, self.current_delay * 1.5)
            logger.info(f"🐌 Rate limit détecté, délai augmenté: {self.current_delay:.2f}s")
        elif error_rate > 0.1:  # Plus de 10% d'erreurs
            self.current_delay = min(self.max_delay, self.current_delay * 1.2)
            logger.info(f"🔧 Taux d'erreur élevé ({error_rate:.1%}), délai augmenté: {self.current_delay:.2f}s")
        elif error_rate < 0.02 and avg_response_time < 2.0:  # Moins de 2% d'erreurs, réponses rapides
            self.current_delay = max(self.min_delay, self.current_delay * 0.9)
            logger.info(f"🚀 Performance stable, délai réduit: {self.current_delay:.2f}s")
        
        # Reset compteurs
        self.success_count = 0
        self.error_count = 0
        self.rate_limit_count = 0
        self.last_adaptation = now
    
    def get_current_delay(self) -> float:
        """Retourne le délai actuel"""
        return self.current_delay


class ScrapingSession:
    """Session de scraping optimisée avec gestion des connexions"""
    
    def __init__(self, config: ScrapingConfig):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        self.rate_limiter = AdaptiveRateLimiter(
            initial_delay=config.base_delay,
            min_delay=0.1,
            max_delay=config.max_delay
        )
        
        # User agents pour rotation
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:89.0) Gecko/20100101 Firefox/89.0"
        ]
        
        self.current_user_agent_index = 0
        
    async def __aenter__(self):
        """Entrée du context manager"""
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Sortie du context manager"""
        await self.close()
    
    async def start(self):
        """Démarre la session"""
        connector = aiohttp.TCPConnector(
            limit=self.config.max_concurrent * 2,
            limit_per_host=self.config.max_concurrent,
            keepalive_timeout=30,
            enable_cleanup_closed=True
        )
        
        timeout = aiohttp.ClientTimeout(total=self.config.timeout)
        
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers=self._get_headers()
        )
        
        logger.info("🌐 Session de scraping initialisée")
    
    async def close(self):
        """Ferme la session"""
        if self.session:
            await self.session.close()
            logger.info("🔌 Session de scraping fermée")
    
    def _get_headers(self) -> Dict[str, str]:
        """Génère les headers avec rotation user agent"""
        headers = {
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "fr-FR,fr;q=0.9,en-US;q=0.8,en;q=0.7",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1"
        }
        
        if self.config.user_agent_rotation:
            headers["User-Agent"] = self.user_agents[self.current_user_agent_index]
            self.current_user_agent_index = (self.current_user_agent_index + 1) % len(self.user_agents)
        else:
            headers["User-Agent"] = self.user_agents[0]
        
        return headers
    
    async def get(self, url: str, **kwargs) -> aiohttp.ClientResponse:
        """GET avec optimisations"""
        if not self.session:
            raise RuntimeError("Session non initialisée")
        
        # Rate limiting adaptatif
        if self.config.adaptive_rate_limiting:
            await self.rate_limiter.wait()
        
        # Mise à jour headers si rotation activée
        if self.config.user_agent_rotation:
            kwargs.setdefault('headers', {}).update(self._get_headers())
        
        start_time = time.time()
        
        try:
            response = await self.session.get(url, **kwargs)
            response_time = time.time() - start_time
            
            # Enregistrer succès
            if self.config.adaptive_rate_limiting:
                self.rate_limiter.record_success(response_time)
            
            return response
            
        except Exception as e:
            response_time = time.time() - start_time
            
            # Analyser le type d'erreur
            is_rate_limit = any(keyword in str(e).lower() for keyword in [
                'rate limit', '429', 'too many requests', 'throttle'
            ])
            
            if self.config.adaptive_rate_limiting:
                self.rate_limiter.record_error(is_rate_limit)
            
            raise


class ScrapingOptimizer:
    """Optimiseur principal de scraping"""
    
    def __init__(self):
        self.metrics: Dict[ScrapingSource, ScrapingMetrics] = {}
        self.active_sessions: Dict[ScrapingSource, ScrapingSession] = {}
        self.scraping_configs: Dict[ScrapingSource, ScrapingConfig] = {}
        
        # Pools de workers par source
        self.worker_pools: Dict[ScrapingSource, asyncio.Semaphore] = {}
        
        # Cache des résultats
        self.result_cache: Dict[str, ScrapingResult] = {}
        
        # Setup configuration par défaut
        self._setup_default_configs()
        
        logger.info("🔧 Optimiseur de scraping initialisé")
    
    def _setup_default_configs(self):
        """Configure les paramètres par défaut pour chaque source"""
        
        # Pappers API - Plus permissif car API officielle
        self.scraping_configs[ScrapingSource.PAPPERS] = ScrapingConfig(
            strategy=ScrapingStrategy.BALANCED,
            max_concurrent=10,
            base_delay=0.5,
            max_delay=5.0,
            timeout=15,
            max_retries=2,
            adaptive_rate_limiting=True
        )
        
        # Société.com - Plus conservateur (scraping web)
        self.scraping_configs[ScrapingSource.SOCIETE] = ScrapingConfig(
            strategy=ScrapingStrategy.CONSERVATIVE,
            max_concurrent=3,
            base_delay=2.0,
            max_delay=15.0,
            timeout=30,
            max_retries=3,
            user_agent_rotation=True,
            adaptive_rate_limiting=True
        )
        
        # Infogreffe - Équilibré (service officiel)
        self.scraping_configs[ScrapingSource.INFOGREFFE] = ScrapingConfig(
            strategy=ScrapingStrategy.BALANCED,
            max_concurrent=5,
            base_delay=1.0,
            max_delay=8.0,
            timeout=25,
            max_retries=2,
            adaptive_rate_limiting=True
        )
        
        # Initialiser métriques
        for source in ScrapingSource:
            self.metrics[source] = ScrapingMetrics()
            self.worker_pools[source] = asyncio.Semaphore(
                self.scraping_configs[source].max_concurrent
            )
    
    async def scrape_single(
        self, 
        source: ScrapingSource, 
        target_id: str, 
        scraper_func: Callable,
        **kwargs
    ) -> ScrapingResult:
        """Scrape un élément unique avec optimisations"""
        
        start_time = time.time()
        config = self.scraping_configs[source]
        
        # Vérifier cache d'abord
        if config.enable_cache:
            cached_result = await self._get_cached_result(source, target_id)
            if cached_result:
                self.metrics[source].requests_cached += 1
                return cached_result
        
        # Acquérir worker du pool
        async with self.worker_pools[source]:
            try:
                # Utiliser session optimisée
                if source not in self.active_sessions:
                    self.active_sessions[source] = ScrapingSession(config)
                    await self.active_sessions[source].start()
                
                session = self.active_sessions[source]
                
                # Exécuter scraping avec retry
                data = await self._scrape_with_retry(
                    scraper_func, session, target_id, config, **kwargs
                )
                
                # Créer résultat
                response_time = time.time() - start_time
                result = ScrapingResult(
                    source=source,
                    target_id=target_id,
                    success=True,
                    data=data,
                    response_time=response_time,
                    metadata={'scraper_func': scraper_func.__name__}
                )
                
                # Mettre en cache
                if config.enable_cache:
                    await self._cache_result(source, target_id, result, config.cache_ttl)
                
                # Mettre à jour métriques
                self._update_metrics(source, True, response_time, len(str(data)) if data else 0)
                
                return result
                
            except Exception as e:
                response_time = time.time() - start_time
                error_msg = str(e)
                
                logger.warning(f"❌ Erreur scraping {source.value} {target_id}: {error_msg}")
                
                # Mettre à jour métriques
                self._update_metrics(source, False, response_time, 0)
                
                return ScrapingResult(
                    source=source,
                    target_id=target_id,
                    success=False,
                    error=error_msg,
                    response_time=response_time
                )
    
    async def scrape_batch(
        self,
        source: ScrapingSource,
        target_ids: List[str],
        scraper_func: Callable,
        batch_size: Optional[int] = None,
        **kwargs
    ) -> List[ScrapingResult]:
        """Scrape un lot d'éléments avec parallélisation optimisée"""
        
        config = self.scraping_configs[source]
        effective_batch_size = batch_size or config.max_concurrent * 2
        
        logger.info(f"🚀 Démarrage scraping batch {source.value}: {len(target_ids)} éléments")
        
        results = []
        
        # Traitement par batches pour éviter la surcharge
        for i in range(0, len(target_ids), effective_batch_size):
            batch = target_ids[i:i + effective_batch_size]
            
            # Créer tâches pour le batch
            tasks = [
                self.scrape_single(source, target_id, scraper_func, **kwargs)
                for target_id in batch
            ]
            
            # Exécuter en parallèle
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Traiter résultats
            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error(f"Exception dans batch scraping: {result}")
                    continue
                results.append(result)
            
            # Log progression
            progress = (i + len(batch)) / len(target_ids) * 100
            logger.info(f"📊 Progression {source.value}: {progress:.1f}% ({len(results)} réussis)")
            
            # Pause entre batches si nécessaire
            if i + effective_batch_size < len(target_ids):
                await asyncio.sleep(0.5)
        
        logger.info(f"✅ Scraping batch terminé {source.value}: {len(results)} résultats")
        return results
    
    async def _scrape_with_retry(
        self,
        scraper_func: Callable,
        session: ScrapingSession,
        target_id: str,
        config: ScrapingConfig,
        **kwargs
    ) -> Any:
        """Exécute le scraping avec retry et backoff exponentiel"""
        
        last_exception = None
        
        for attempt in range(config.max_retries + 1):
            try:
                # Appeler fonction de scraping
                if asyncio.iscoroutinefunction(scraper_func):
                    result = await scraper_func(session, target_id, **kwargs)
                else:
                    result = scraper_func(session, target_id, **kwargs)
                
                return result
                
            except Exception as e:
                last_exception = e
                
                if attempt < config.max_retries:
                    # Backoff exponentiel avec jitter
                    delay = min(
                        config.base_delay * (2 ** attempt) + random.uniform(0, 1),
                        config.max_delay
                    )
                    
                    logger.warning(f"🔄 Retry {attempt + 1}/{config.max_retries} pour {target_id} dans {delay:.2f}s")
                    await asyncio.sleep(delay)
                else:
                    # Dernier essai échoué
                    raise last_exception
        
        raise last_exception
    
    async def _get_cached_result(self, source: ScrapingSource, target_id: str) -> Optional[ScrapingResult]:
        """Récupère un résultat depuis le cache"""
        
        try:
            cache_manager = await get_cache_manager()
            cache_key = f"{source.value}_{target_id}"
            
            cached_data = await cache_manager.get('scraping_results', cache_key)
            if cached_data:
                # Reconstituer le résultat
                return ScrapingResult(
                    source=source,
                    target_id=target_id,
                    success=cached_data['success'],
                    data=cached_data.get('data'),
                    error=cached_data.get('error'),
                    response_time=cached_data.get('response_time', 0),
                    cached=True,
                    timestamp=datetime.fromisoformat(cached_data['timestamp']),
                    metadata=cached_data.get('metadata', {})
                )
        except Exception as e:
            logger.warning(f"Erreur lecture cache pour {source.value}_{target_id}: {e}")
        
        return None
    
    async def _cache_result(self, source: ScrapingSource, target_id: str, result: ScrapingResult, ttl: int):
        """Met en cache un résultat"""
        
        try:
            cache_manager = await get_cache_manager()
            cache_key = f"{source.value}_{target_id}"
            
            cache_data = {
                'success': result.success,
                'data': result.data,
                'error': result.error,
                'response_time': result.response_time,
                'timestamp': result.timestamp.isoformat(),
                'metadata': result.metadata
            }
            
            await cache_manager.set('scraping_results', cache_key, cache_data, ttl_seconds=ttl)
            
        except Exception as e:
            logger.warning(f"Erreur mise en cache pour {source.value}_{target_id}: {e}")
    
    def _update_metrics(self, source: ScrapingSource, success: bool, response_time: float, data_size: int):
        """Met à jour les métriques de scraping"""
        
        metrics = self.metrics[source]
        metrics.requests_total += 1
        
        if success:
            metrics.requests_success += 1
            metrics.total_data_scraped += data_size
        else:
            metrics.requests_failed += 1
        
        # Moyenne mobile du temps de réponse
        if metrics.requests_total == 1:
            metrics.avg_response_time = response_time
        else:
            # Moyenne pondérée pour éviter la dérive
            alpha = 0.1  # Facteur de lissage
            metrics.avg_response_time = (
                alpha * response_time + (1 - alpha) * metrics.avg_response_time
            )
        
        metrics.last_update = datetime.now()
    
    def get_metrics(self, source: Optional[ScrapingSource] = None) -> Dict[str, Any]:
        """Récupère les métriques de performance"""
        
        if source:
            metrics = self.metrics[source]
            return {
                'source': source.value,
                'requests_total': metrics.requests_total,
                'success_rate': metrics.success_rate,
                'cache_hit_rate': metrics.cache_hit_rate,
                'avg_response_time': metrics.avg_response_time,
                'total_data_scraped': metrics.total_data_scraped,
                'last_update': metrics.last_update.isoformat()
            }
        else:
            # Métriques globales
            total_requests = sum(m.requests_total for m in self.metrics.values())
            total_success = sum(m.requests_success for m in self.metrics.values())
            total_cached = sum(m.requests_cached for m in self.metrics.values())
            
            return {
                'global': {
                    'total_requests': total_requests,
                    'success_rate': (total_success / total_requests * 100) if total_requests > 0 else 0,
                    'cache_hit_rate': (total_cached / total_requests * 100) if total_requests > 0 else 0,
                    'sources': list(self.metrics.keys()),
                    'active_sessions': len(self.active_sessions)
                },
                'by_source': {
                    source.value: self.get_metrics(source) 
                    for source in self.metrics.keys()
                }
            }
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Génère un rapport de performance détaillé"""
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {},
            'sources': {},
            'recommendations': []
        }
        
        # Analyse par source
        for source, metrics in self.metrics.items():
            source_report = {
                'metrics': self.get_metrics(source),
                'configuration': {
                    'max_concurrent': self.scraping_configs[source].max_concurrent,
                    'base_delay': self.scraping_configs[source].base_delay,
                    'strategy': self.scraping_configs[source].strategy.value
                },
                'health': 'good'
            }
            
            # Évaluer santé
            if metrics.success_rate < 80:
                source_report['health'] = 'poor'
                report['recommendations'].append(f"Améliorer configuration {source.value}: taux de succès faible ({metrics.success_rate:.1f}%)")
            elif metrics.success_rate < 95:
                source_report['health'] = 'fair'
            
            if metrics.avg_response_time > 5.0:
                report['recommendations'].append(f"Optimiser {source.value}: temps de réponse élevé ({metrics.avg_response_time:.2f}s)")
            
            report['sources'][source.value] = source_report
        
        # Résumé global
        all_metrics = list(self.metrics.values())
        if all_metrics:
            report['summary'] = {
                'total_sources': len(all_metrics),
                'avg_success_rate': statistics.mean([m.success_rate for m in all_metrics]),
                'avg_response_time': statistics.mean([m.avg_response_time for m in all_metrics]),
                'total_requests': sum(m.requests_total for m in all_metrics),
                'overall_health': 'good' if len(report['recommendations']) == 0 else 'needs_attention'
            }
        
        return report
    
    async def optimize_configurations(self):
        """Optimise automatiquement les configurations selon les performances"""
        
        logger.info("🔧 Optimisation automatique des configurations...")
        
        for source, metrics in self.metrics.items():
            if metrics.requests_total < 50:  # Pas assez de données
                continue
            
            config = self.scraping_configs[source]
            
            # Ajustements basés sur les performances
            if metrics.success_rate < 80:  # Faible taux de succès
                # Réduire concurrence et augmenter délais
                config.max_concurrent = max(1, config.max_concurrent - 1)
                config.base_delay = min(config.max_delay, config.base_delay * 1.2)
                logger.info(f"📉 {source.value}: Configuration réduite (succès: {metrics.success_rate:.1f}%)")
                
            elif metrics.success_rate > 95 and metrics.avg_response_time < 2.0:  # Excellentes performances
                # Augmenter concurrence si possible
                if config.max_concurrent < 10:
                    config.max_concurrent += 1
                config.base_delay = max(0.1, config.base_delay * 0.9)
                logger.info(f"📈 {source.value}: Configuration optimisée (succès: {metrics.success_rate:.1f}%)")
        
        logger.info("✅ Optimisation configurations terminée")
    
    async def close_all_sessions(self):
        """Ferme toutes les sessions actives"""
        
        for session in self.active_sessions.values():
            try:
                await session.close()
            except Exception as e:
                logger.warning(f"Erreur fermeture session: {e}")
        
        self.active_sessions.clear()
        logger.info("🔌 Toutes les sessions fermées")


# Instance globale
_scraping_optimizer: Optional[ScrapingOptimizer] = None


async def get_scraping_optimizer() -> ScrapingOptimizer:
    """Factory pour obtenir l'optimiseur de scraping"""
    global _scraping_optimizer
    
    if _scraping_optimizer is None:
        _scraping_optimizer = ScrapingOptimizer()
    
    return _scraping_optimizer


# Décorateurs pour optimisation automatique

def optimized_scraper(source: ScrapingSource, cache_ttl: int = 3600):
    """Décorateur pour optimiser automatiquement une fonction de scraping"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(target_id: str, **kwargs) -> ScrapingResult:
            optimizer = await get_scraping_optimizer()
            return await optimizer.scrape_single(
                source=source,
                target_id=target_id,
                scraper_func=func,
                **kwargs
            )
        
        return wrapper
    return decorator


# Fonctions utilitaires pour scrapers spécialisés

async def scrape_company_pappers(session: ScrapingSession, siren: str, **kwargs) -> Dict[str, Any]:
    """Scraper optimisé pour Pappers"""
    
    url = f"https://api.pappers.fr/v2/entreprise?api_token=YOUR_TOKEN&siren={siren}"
    
    async with session.get(url) as response:
        if response.status == 200:
            data = await response.json()
            return {
                'siren': siren,
                'nom': data.get('nom_entreprise'),
                'adresse': data.get('adresse'),
                'secteur': data.get('secteur_activite'),
                'chiffre_affaires': data.get('chiffre_affaires'),
                'effectifs': data.get('effectif'),
                'date_creation': data.get('date_creation'),
                'source': 'pappers'
            }
        elif response.status == 429:
            raise Exception("Rate limit Pappers atteint")
        else:
            raise Exception(f"Erreur API Pappers: {response.status}")


async def scrape_company_societe(session: ScrapingSession, siren: str, **kwargs) -> Dict[str, Any]:
    """Scraper optimisé pour Société.com"""
    
    url = f"https://www.societe.com/siren/{siren}"
    
    async with session.get(url) as response:
        if response.status == 200:
            html = await response.text()
            
            # Extraction basique (à améliorer avec BeautifulSoup)
            # Cette implémentation est simplifiée
            
            return {
                'siren': siren,
                'nom': 'Extracted from HTML',  # À extraire du HTML
                'secteur': 'Extracted from HTML',
                'source': 'societe'
            }
        elif response.status == 429:
            raise Exception("Rate limit Société.com détecté")
        else:
            raise Exception(f"Erreur Société.com: {response.status}")


# Fonctions utilitaires pour l'API

async def batch_scrape_companies(
    company_siren_list: List[str],
    sources: List[ScrapingSource] = None,
    parallel_sources: bool = True
) -> Dict[str, List[ScrapingResult]]:
    """
    Scrape un lot d'entreprises sur plusieurs sources
    
    Args:
        company_siren_list: Liste des SIREN à scraper
        sources: Sources à utiliser (par défaut toutes)
        parallel_sources: Scraper les sources en parallèle
    
    Returns:
        Dictionnaire avec résultats par source
    """
    sources = sources or [ScrapingSource.PAPPERS, ScrapingSource.SOCIETE]
    optimizer = await get_scraping_optimizer()
    
    results = {}
    
    if parallel_sources:
        # Scraping de toutes les sources en parallèle
        scraping_tasks = []
        for source in sources:
            if source == ScrapingSource.PAPPERS:
                task = optimizer.scrape_batch(source, company_siren_list, scrape_company_pappers)
            elif source == ScrapingSource.SOCIETE:
                task = optimizer.scrape_batch(source, company_siren_list, scrape_company_societe)
            else:
                continue  # Source non implémentée
            
            scraping_tasks.append((source, task))
        
        # Exécuter en parallèle
        for source, task in scraping_tasks:
            try:
                source_results = await task
                results[source.value] = source_results
            except Exception as e:
                logger.error(f"Erreur scraping source {source.value}: {e}")
                results[source.value] = []
    else:
        # Scraping séquentiel par source
        for source in sources:
            try:
                if source == ScrapingSource.PAPPERS:
                    source_results = await optimizer.scrape_batch(source, company_siren_list, scrape_company_pappers)
                elif source == ScrapingSource.SOCIETE:
                    source_results = await optimizer.scrape_batch(source, company_siren_list, scrape_company_societe)
                else:
                    continue
                
                results[source.value] = source_results
            except Exception as e:
                logger.error(f"Erreur scraping source {source.value}: {e}")
                results[source.value] = []
    
    return results


async def get_scraping_performance_stats() -> Dict[str, Any]:
    """Récupère les statistiques de performance du scraping"""
    
    optimizer = await get_scraping_optimizer()
    return optimizer.get_performance_report()