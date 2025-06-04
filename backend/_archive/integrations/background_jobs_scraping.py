"""
Jobs Celery spécialisés pour le scraping parallèle
US-009: Optimisation scraping avec parallélisation et rate limiting intelligent

Ce module contient:
- Jobs de scraping batch avec parallélisation
- Rate limiting intelligent adaptatif
- Gestion des erreurs et retry logic
- Monitoring temps réel du scraping
- Optimisation mémoire pour gros volumes
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
import aiohttp
from urllib.parse import urljoin
import random

from app.core.background_jobs import (
    celery_task, JobPriority, JobType, get_background_job_manager
)
from app.core.logging_system import get_logger, LogCategory
from app.core.cache_manager import get_cache_manager
from app.core.performance_analyzer import get_performance_analyzer
from app.scrapers.pappers import PappersClient
from app.scrapers.societe import SocieteComScraper
from app.scrapers.infogreffe import InfogreffeScraper

logger = get_logger("background_scraping", LogCategory.SCRAPING)


@dataclass
class ScrapingJobConfig:
    """Configuration d'un job de scraping"""
    batch_size: int = 50
    max_concurrent: int = 10
    rate_limit_per_second: float = 2.0
    retry_delay: int = 30
    max_retries: int = 3
    timeout: int = 30
    adaptive_rate_limiting: bool = True
    memory_limit_mb: int = 512


@dataclass
class ScrapingProgress:
    """Progression d'un scraping"""
    total_items: int
    completed_items: int = 0
    failed_items: int = 0
    skipped_items: int = 0
    current_rate: float = 0.0
    estimated_remaining_time: Optional[int] = None
    last_update: datetime = field(default_factory=datetime.now)
    
    @property
    def progress_percentage(self) -> float:
        return (self.completed_items / self.total_items * 100) if self.total_items > 0 else 0
    
    @property
    def success_rate(self) -> float:
        processed = self.completed_items + self.failed_items
        return (self.completed_items / processed * 100) if processed > 0 else 0


class AdaptiveRateLimiter:
    """Rate limiter adaptatif basé sur les performances"""
    
    def __init__(self, initial_rate: float = 2.0, min_rate: float = 0.5, max_rate: float = 10.0):
        self.current_rate = initial_rate
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.success_count = 0
        self.error_count = 0
        self.last_adjustment = time.time()
        self.adjustment_interval = 30  # secondes
    
    async def wait(self):
        """Attend selon le rate limiting actuel"""
        await asyncio.sleep(1.0 / self.current_rate)
    
    def record_success(self):
        """Enregistre un succès"""
        self.success_count += 1
        self._adjust_rate_if_needed()
    
    def record_error(self):
        """Enregistre une erreur"""
        self.error_count += 1
        self._adjust_rate_if_needed()
    
    def _adjust_rate_if_needed(self):
        """Ajuste le rate selon les performances"""
        now = time.time()
        if now - self.last_adjustment < self.adjustment_interval:
            return
        
        total_requests = self.success_count + self.error_count
        if total_requests < 10:  # Pas assez de données
            return
        
        error_rate = self.error_count / total_requests
        
        if error_rate > 0.2:  # Plus de 20% d'erreurs, ralentir
            self.current_rate = max(self.min_rate, self.current_rate * 0.8)
            logger.info(f"🐌 Rate limiting réduit: {self.current_rate:.2f} req/s (erreurs: {error_rate:.1%})")
        elif error_rate < 0.05:  # Moins de 5% d'erreurs, accélérer
            self.current_rate = min(self.max_rate, self.current_rate * 1.2)
            logger.info(f"🚀 Rate limiting augmenté: {self.current_rate:.2f} req/s (erreurs: {error_rate:.1%})")
        
        # Reset compteurs
        self.success_count = 0
        self.error_count = 0
        self.last_adjustment = now


@celery_task(priority=JobPriority.NORMAL)
async def scrape_companies_batch(
    company_ids: List[str],
    scraper_type: str = "pappers",
    config: Optional[Dict[str, Any]] = None,
    job_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Scraping en lot d'entreprises avec parallélisation
    
    Args:
        company_ids: Liste des identifiants (SIREN)
        scraper_type: Type de scraper (pappers, societe, infogreffe)
        config: Configuration du scraping
        job_id: ID du job pour tracking
    """
    start_time = time.time()
    
    # Configuration par défaut
    scraping_config = ScrapingJobConfig()
    if config:
        for key, value in config.items():
            if hasattr(scraping_config, key):
                setattr(scraping_config, key, value)
    
    # Initialisation
    progress = ScrapingProgress(total_items=len(company_ids))
    rate_limiter = AdaptiveRateLimiter(
        initial_rate=scraping_config.rate_limit_per_second
    ) if scraping_config.adaptive_rate_limiting else None
    
    results = []
    errors = []
    
    logger.info(f"🏁 Démarrage scraping batch: {len(company_ids)} entreprises ({scraper_type})")
    
    try:
        # Initialiser scraper
        scraper = await _get_scraper(scraper_type)
        
        # Traitement par batches
        for i in range(0, len(company_ids), scraping_config.batch_size):
            batch = company_ids[i:i + scraping_config.batch_size]
            
            # Traitement parallèle du batch
            batch_results = await _process_batch_parallel(
                batch, scraper, scraping_config, rate_limiter, progress
            )
            
            results.extend(batch_results['results'])
            errors.extend(batch_results['errors'])
            
            # Mise à jour progression
            progress.completed_items += len(batch_results['results'])
            progress.failed_items += len(batch_results['errors'])
            
            # Calcul vitesse et estimation
            elapsed = time.time() - start_time
            progress.current_rate = progress.completed_items / elapsed if elapsed > 0 else 0
            remaining_items = len(company_ids) - (progress.completed_items + progress.failed_items)
            if progress.current_rate > 0:
                progress.estimated_remaining_time = int(remaining_items / progress.current_rate)
            
            # Log progression
            logger.info(f"📊 Progression: {progress.progress_percentage:.1f}% "
                       f"({progress.completed_items}/{len(company_ids)}) "
                       f"- {progress.current_rate:.1f} items/s")
            
            # Gestion mémoire
            if i % (scraping_config.batch_size * 5) == 0:  # Tous les 5 batches
                await _cleanup_memory()
        
        # Finalisation
        execution_time = time.time() - start_time
        
        # Cache des résultats réussis
        cache_manager = await get_cache_manager()
        for result in results:
            if result.get('siren'):
                await cache_manager.set(
                    'scraping', 
                    f"{scraper_type}_{result['siren']}", 
                    result,
                    ttl_seconds=3600  # 1 heure
                )
        
        final_result = {
            'success': True,
            'job_id': job_id,
            'scraper_type': scraper_type,
            'total_requested': len(company_ids),
            'successful_scrapes': len(results),
            'failed_scrapes': len(errors),
            'success_rate': (len(results) / len(company_ids) * 100) if company_ids else 0,
            'execution_time': execution_time,
            'average_rate': len(results) / execution_time if execution_time > 0 else 0,
            'results': results,
            'errors': errors[:50],  # Limiter les erreurs retournées
            'progress': {
                'completed': progress.completed_items,
                'failed': progress.failed_items,
                'success_rate': progress.success_rate
            },
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"✅ Scraping terminé: {len(results)}/{len(company_ids)} réussis "
                   f"en {execution_time:.1f}s ({final_result['average_rate']:.1f} items/s)")
        
        return final_result
        
    except Exception as e:
        logger.error(f"❌ Erreur scraping batch: {e}")
        return {
            'success': False,
            'job_id': job_id,
            'error': str(e),
            'partial_results': results,
            'errors': errors,
            'timestamp': datetime.now().isoformat()
        }


@celery_task(priority=JobPriority.HIGH)
async def scrape_single_company_detailed(
    siren: str,
    scraper_types: List[str] = None,
    force_refresh: bool = False
) -> Dict[str, Any]:
    """
    Scraping détaillé d'une seule entreprise avec tous les scrapers
    
    Args:
        siren: Numéro SIREN de l'entreprise
        scraper_types: Liste des scrapers à utiliser
        force_refresh: Forcer le refresh du cache
    """
    scraper_types = scraper_types or ['pappers', 'societe', 'infogreffe']
    
    results = {}
    errors = {}
    
    logger.info(f"🔍 Scraping détaillé: {siren} avec {len(scraper_types)} scrapers")
    
    for scraper_type in scraper_types:
        try:
            # Vérifier cache sauf si refresh forcé
            if not force_refresh:
                cache_manager = await get_cache_manager()
                cached_result = await cache_manager.get('scraping', f"{scraper_type}_{siren}")
                if cached_result:
                    results[scraper_type] = {
                        'data': cached_result,
                        'source': 'cache',
                        'timestamp': datetime.now().isoformat()
                    }
                    continue
            
            # Scraping
            scraper = await _get_scraper(scraper_type)
            start_time = time.time()
            
            data = await _scrape_single_company(scraper, siren, scraper_type)
            
            execution_time = time.time() - start_time
            
            results[scraper_type] = {
                'data': data,
                'source': 'scraping',
                'execution_time': execution_time,
                'timestamp': datetime.now().isoformat()
            }
            
            # Cache du résultat
            if not force_refresh and data:
                cache_manager = await get_cache_manager()
                await cache_manager.set(
                    'scraping',
                    f"{scraper_type}_{siren}",
                    data,
                    ttl_seconds=3600
                )
            
            logger.info(f"✅ {scraper_type}: {siren} scraped en {execution_time:.2f}s")
            
        except Exception as e:
            errors[scraper_type] = str(e)
            logger.error(f"❌ Erreur {scraper_type} pour {siren}: {e}")
    
    # Fusionner les données
    merged_data = _merge_scraping_results(results)
    
    return {
        'siren': siren,
        'success': len(results) > 0,
        'scrapers_used': list(results.keys()),
        'scrapers_failed': list(errors.keys()),
        'individual_results': results,
        'merged_data': merged_data,
        'errors': errors,
        'timestamp': datetime.now().isoformat()
    }


@celery_task(priority=JobPriority.LOW)
async def monitor_scraping_performance():
    """Monitoring des performances de scraping"""
    try:
        performance_analyzer = get_performance_analyzer()
        
        # Analyser métriques de scraping
        scraping_operations = [
            op for op in performance_analyzer.operation_stats.keys() 
            if 'scrap' in op.lower()
        ]
        
        performance_data = {}
        for operation in scraping_operations:
            metrics = performance_analyzer.operation_stats[operation]
            if len(metrics) >= 5:
                durations = [m.duration_ms for m in metrics[-50:]]
                performance_data[operation] = {
                    'avg_duration_ms': np.mean(durations),
                    'median_duration_ms': np.median(durations),
                    'p95_duration_ms': np.percentile(durations, 95),
                    'total_calls': len(metrics),
                    'success_rate': len([m for m in metrics if not m.error]) / len(metrics) * 100
                }
        
        # Recommandations d'optimisation
        recommendations = []
        for operation, data in performance_data.items():
            if data['avg_duration_ms'] > 5000:  # > 5 secondes
                recommendations.append(f"Optimiser {operation}: temps moyen {data['avg_duration_ms']:.0f}ms")
            if data['success_rate'] < 90:
                recommendations.append(f"Améliorer fiabilité {operation}: {data['success_rate']:.1f}% succès")
        
        return {
            'timestamp': datetime.now().isoformat(),
            'performance_metrics': performance_data,
            'recommendations': recommendations,
            'total_scraping_operations': len(scraping_operations)
        }
        
    except Exception as e:
        logger.error(f"Erreur monitoring scraping: {e}")
        return {'error': str(e)}


async def _get_scraper(scraper_type: str):
    """Factory pour obtenir un scraper"""
    if scraper_type == 'pappers':
        return PappersClient()
    elif scraper_type == 'societe':
        return SocieteComScraper()
    elif scraper_type == 'infogreffe':
        return InfogreffeScraper()
    else:
        raise ValueError(f"Scraper non supporté: {scraper_type}")


async def _process_batch_parallel(
    batch: List[str],
    scraper: Any,
    config: ScrapingJobConfig,
    rate_limiter: Optional[AdaptiveRateLimiter],
    progress: ScrapingProgress
) -> Dict[str, List]:
    """Traite un batch en parallèle avec rate limiting"""
    
    semaphore = asyncio.Semaphore(config.max_concurrent)
    results = []
    errors = []
    
    async def process_single(siren: str) -> Tuple[Optional[Dict], Optional[str]]:
        async with semaphore:
            try:
                if rate_limiter:
                    await rate_limiter.wait()
                
                data = await _scrape_single_company(scraper, siren, scraper.__class__.__name__)
                
                if rate_limiter:
                    rate_limiter.record_success()
                
                return data, None
                
            except Exception as e:
                if rate_limiter:
                    rate_limiter.record_error()
                
                error_msg = f"SIREN {siren}: {str(e)}"
                logger.warning(f"⚠️ {error_msg}")
                return None, error_msg
    
    # Traitement parallèle
    tasks = [process_single(siren) for siren in batch]
    batch_results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Séparer résultats et erreurs
    for i, result in enumerate(batch_results):
        if isinstance(result, Exception):
            errors.append(f"SIREN {batch[i]}: {str(result)}")
        else:
            data, error = result
            if data:
                results.append(data)
            if error:
                errors.append(error)
    
    return {'results': results, 'errors': errors}


async def _scrape_single_company(scraper: Any, siren: str, scraper_type: str) -> Dict[str, Any]:
    """Scrape une seule entreprise"""
    try:
        if scraper_type.lower() == 'pappers':
            # Utiliser l'API Pappers
            company_data = await scraper.get_company_info(siren)
            return {
                'siren': siren,
                'source': 'pappers',
                'data': company_data,
                'scraped_at': datetime.now().isoformat()
            }
        
        elif scraper_type.lower() == 'societe':
            # Scraping Société.com
            company_data = await scraper.scrape_company(siren)
            return {
                'siren': siren,
                'source': 'societe',
                'data': company_data,
                'scraped_at': datetime.now().isoformat()
            }
        
        elif scraper_type.lower() == 'infogreffe':
            # Scraping Infogreffe
            company_data = await scraper.scrape_company(siren)
            return {
                'siren': siren,
                'source': 'infogreffe', 
                'data': company_data,
                'scraped_at': datetime.now().isoformat()
            }
        
        else:
            raise ValueError(f"Scraper non supporté: {scraper_type}")
    
    except Exception as e:
        logger.error(f"Erreur scraping {siren} avec {scraper_type}: {e}")
        raise


def _merge_scraping_results(results: Dict[str, Dict]) -> Dict[str, Any]:
    """Fusionne les résultats de plusieurs scrapers"""
    merged = {
        'sources': list(results.keys()),
        'merged_at': datetime.now().isoformat()
    }
    
    # Fusionner les données par priorité (Pappers > Infogreffe > Société)
    priority_order = ['pappers', 'infogreffe', 'societe']
    
    all_data = {}
    for scraper_type in priority_order:
        if scraper_type in results and results[scraper_type].get('data'):
            data = results[scraper_type]['data']
            if isinstance(data, dict):
                # Fusionner sans écraser les données existantes
                for key, value in data.items():
                    if key not in all_data and value is not None:
                        all_data[key] = value
    
    merged.update(all_data)
    return merged


async def _cleanup_memory():
    """Nettoyage mémoire périodique"""
    import gc
    collected = gc.collect()
    if collected > 0:
        logger.debug(f"🧹 Mémoire nettoyée: {collected} objets collectés")


# Fonctions utilitaires pour l'API

async def submit_scraping_job(
    company_ids: List[str],
    scraper_type: str = "pappers",
    user_id: Optional[str] = None,
    priority: JobPriority = JobPriority.NORMAL,
    config: Optional[Dict[str, Any]] = None
) -> str:
    """Soumet un job de scraping en arrière-plan"""
    
    job_manager = await get_background_job_manager()
    
    estimated_duration = len(company_ids) * 2  # 2 secondes par entreprise estimé
    
    job_id = await job_manager.submit_job(
        task_name='scrape_companies_batch',
        args=[company_ids, scraper_type, config],
        priority=priority,
        job_type=JobType.SCRAPING,
        estimated_duration=estimated_duration,
        user_id=user_id,
        context={
            'company_count': len(company_ids),
            'scraper_type': scraper_type,
            'config': config or {}
        }
    )
    
    logger.info(f"📤 Job scraping soumis: {job_id} ({len(company_ids)} entreprises)")
    return job_id


async def get_scraping_statistics() -> Dict[str, Any]:
    """Récupère les statistiques de scraping"""
    try:
        job_manager = await get_background_job_manager()
        
        # Jobs de scraping
        scraping_jobs = [
            job for job in job_manager.active_jobs.values()
            if job.metadata and job.metadata.job_type == JobType.SCRAPING
        ]
        
        # Statistiques
        total_jobs = len(scraping_jobs)
        completed_jobs = len([j for j in scraping_jobs if j.status.value == 'success'])
        failed_jobs = len([j for j in scraping_jobs if j.status.value == 'failure'])
        active_jobs = len([j for j in scraping_jobs if j.status.value in ['pending', 'started', 'progress']])
        
        return {
            'total_scraping_jobs': total_jobs,
            'completed_jobs': completed_jobs,
            'failed_jobs': failed_jobs,
            'active_jobs': active_jobs,
            'success_rate': (completed_jobs / total_jobs * 100) if total_jobs > 0 else 0,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Erreur stats scraping: {e}")
        return {'error': str(e)}