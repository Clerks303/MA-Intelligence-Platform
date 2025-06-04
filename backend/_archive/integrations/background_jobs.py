"""
Système de background jobs avec Celery pour M&A Intelligence Platform
US-009: Tâches asynchrones pour scraping, analyse ML, rapports et traitement batch

Ce module fournit:
- Configuration Celery optimisée
- Jobs de scraping parallèle
- Traitement ML en arrière-plan
- Génération de rapports
- Monitoring et retry logic
- Gestion des files de priorité
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from functools import wraps
from enum import Enum
import logging
import traceback
import uuid
from pathlib import Path

import pandas as pd
import numpy as np
from celery import Celery, Task, group, chain, chord
from celery.result import AsyncResult
from celery.exceptions import Retry, WorkerLostError
from celery.signals import task_prerun, task_postrun, task_failure, task_success
from kombu import Queue, Exchange
import redis

from app.config import settings
from app.core.logging_system import get_logger, LogCategory
from app.core.cache_manager import get_cache_manager
from app.core.performance_analyzer import get_performance_analyzer

logger = get_logger("background_jobs", LogCategory.PERFORMANCE)


class JobPriority(str, Enum):
    """Priorités des jobs"""
    CRITICAL = "critical"      # < 1 minute
    HIGH = "high"             # < 5 minutes  
    NORMAL = "normal"         # < 15 minutes
    LOW = "low"               # < 1 heure
    BATCH = "batch"           # Peut attendre


class JobStatus(str, Enum):
    """États des jobs"""
    PENDING = "pending"
    STARTED = "started"
    PROGRESS = "progress"
    SUCCESS = "success"
    FAILURE = "failure"
    RETRY = "retry"
    REVOKED = "revoked"


class JobType(str, Enum):
    """Types de jobs"""
    SCRAPING = "scraping"
    ML_ANALYSIS = "ml_analysis"
    REPORT_GENERATION = "report_generation"
    DATA_PROCESSING = "data_processing"
    MAINTENANCE = "maintenance"
    NOTIFICATION = "notification"


@dataclass
class JobMetadata:
    """Métadonnées d'un job"""
    job_id: str
    job_type: JobType
    priority: JobPriority
    created_at: datetime
    estimated_duration: Optional[int] = None  # en secondes
    retry_count: int = 0
    max_retries: int = 3
    context: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    user_id: Optional[str] = None


@dataclass
class JobResult:
    """Résultat d'un job"""
    job_id: str
    status: JobStatus
    result: Optional[Any] = None
    error: Optional[str] = None
    execution_time: Optional[float] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress: float = 0.0
    metadata: Optional[JobMetadata] = None


class CustomCeleryTask(Task):
    """Classe de base personnalisée pour les tâches Celery"""
    
    def __call__(self, *args, **kwargs):
        """Wrapper pour ajouter monitoring et gestion d'erreurs"""
        job_id = self.request.id
        start_time = time.time()
        
        try:
            # Enregistrer début job
            self._record_job_start(job_id, start_time)
            
            # Exécuter tâche
            result = super().__call__(*args, **kwargs)
            
            # Enregistrer succès
            execution_time = time.time() - start_time
            self._record_job_success(job_id, result, execution_time)
            
            return result
            
        except Exception as exc:
            execution_time = time.time() - start_time
            self._record_job_failure(job_id, exc, execution_time)
            raise
    
    def _record_job_start(self, job_id: str, start_time: float):
        """Enregistre le début d'un job"""
        logger.info(f"🚀 Job démarré: {job_id} ({self.name})")
    
    def _record_job_success(self, job_id: str, result: Any, execution_time: float):
        """Enregistre le succès d'un job"""
        logger.info(f"✅ Job terminé: {job_id} en {execution_time:.2f}s")
    
    def _record_job_failure(self, job_id: str, exc: Exception, execution_time: float):
        """Enregistre l'échec d'un job"""
        logger.error(f"❌ Job échoué: {job_id} après {execution_time:.2f}s - {exc}")


class BackgroundJobManager:
    """Gestionnaire principal des jobs Celery"""
    
    def __init__(self):
        self.celery_app = None
        self.redis_client = None
        self.job_registry: Dict[str, JobMetadata] = {}
        self.active_jobs: Dict[str, JobResult] = {}
        
        # Configuration des files de priorité
        self.queue_config = {
            JobPriority.CRITICAL: {
                'name': 'critical',
                'routing_key': 'critical',
                'concurrency': 4,
                'max_retries': 1
            },
            JobPriority.HIGH: {
                'name': 'high_priority',
                'routing_key': 'high',
                'concurrency': 8,
                'max_retries': 2
            },
            JobPriority.NORMAL: {
                'name': 'normal',
                'routing_key': 'normal',
                'concurrency': 6,
                'max_retries': 3
            },
            JobPriority.LOW: {
                'name': 'low_priority',
                'routing_key': 'low',
                'concurrency': 2,
                'max_retries': 5
            },
            JobPriority.BATCH: {
                'name': 'batch',
                'routing_key': 'batch',
                'concurrency': 1,
                'max_retries': 2
            }
        }
        
        logger.info("📋 BackgroundJobManager initialisé")
    
    async def initialize(self):
        """Initialise le gestionnaire de jobs"""
        try:
            # Configuration Celery
            redis_url = getattr(settings, 'REDIS_URL', 'redis://localhost:6379/1')
            
            self.celery_app = Celery(
                'ma_intelligence_jobs',
                broker=redis_url,
                backend=redis_url,
                include=[
                    'app.core.background_jobs.scraping_jobs',
                    'app.core.background_jobs.ml_jobs',
                    'app.core.background_jobs.report_jobs'
                ]
            )
            
            # Configuration Celery
            self.celery_app.conf.update(
                # Sérialisation
                task_serializer='json',
                accept_content=['json'],
                result_serializer='json',
                timezone='Europe/Paris',
                enable_utc=True,
                
                # Performance
                worker_prefetch_multiplier=1,
                task_acks_late=True,
                worker_max_tasks_per_child=1000,
                
                # Retry configuration
                task_retry_delay=60,
                task_max_retries=3,
                
                # Monitoring
                worker_send_task_events=True,
                task_send_sent_event=True,
                
                # Files de priorité
                task_routes={
                    'scraping.*': {'queue': 'normal'},
                    'ml_analysis.*': {'queue': 'high_priority'},
                    'reports.*': {'queue': 'low_priority'},
                    'critical.*': {'queue': 'critical'},
                    'batch.*': {'queue': 'batch'}
                },
                
                # Configuration des files
                task_default_queue='normal',
                task_default_exchange='tasks',
                task_default_exchange_type='direct',
                task_default_routing_key='normal'
            )
            
            # Créer les files avec priorités
            self._setup_queues()
            
            # Client Redis pour metadata
            self.redis_client = redis.Redis.from_url(redis_url, decode_responses=True)
            
            # Test connexion
            await asyncio.get_event_loop().run_in_executor(None, self.redis_client.ping)
            
            # Configurer signaux Celery
            self._setup_celery_signals()
            
            logger.info("✅ Système de background jobs initialisé")
            
        except Exception as e:
            logger.error(f"❌ Erreur initialisation jobs: {e}")
            raise
    
    def _setup_queues(self):
        """Configure les files de priorité"""
        task_queues = []
        
        for priority, config in self.queue_config.items():
            exchange = Exchange('tasks', type='direct')
            queue = Queue(
                config['name'],
                exchange,
                routing_key=config['routing_key']
            )
            task_queues.append(queue)
        
        self.celery_app.conf.task_queues = task_queues
    
    def _setup_celery_signals(self):
        """Configure les signaux Celery pour monitoring"""
        
        @task_prerun.connect
        def task_prerun_handler(sender=None, task_id=None, task=None, args=None, kwargs=None, **kwds):
            """Signal avant exécution"""
            logger.debug(f"🏃 Démarrage tâche: {task_id} ({task.name})")
        
        @task_postrun.connect  
        def task_postrun_handler(sender=None, task_id=None, task=None, args=None, kwargs=None, retval=None, state=None, **kwds):
            """Signal après exécution"""
            logger.debug(f"🏁 Fin tâche: {task_id} - État: {state}")
        
        @task_failure.connect
        def task_failure_handler(sender=None, task_id=None, exception=None, einfo=None, **kwds):
            """Signal d'échec"""
            logger.error(f"💥 Échec tâche: {task_id} - {exception}")
        
        @task_success.connect
        def task_success_handler(sender=None, result=None, **kwargs):
            """Signal de succès"""
            logger.debug(f"🎉 Succès tâche: {kwargs.get('task_id')}")
    
    async def submit_job(self, 
                        task_name: str,
                        args: List[Any] = None,
                        kwargs: Dict[str, Any] = None,
                        priority: JobPriority = JobPriority.NORMAL,
                        job_type: JobType = JobType.DATA_PROCESSING,
                        estimated_duration: Optional[int] = None,
                        dependencies: List[str] = None,
                        user_id: Optional[str] = None,
                        context: Dict[str, Any] = None) -> str:
        """Soumet un job en arrière-plan"""
        
        job_id = str(uuid.uuid4())
        args = args or []
        kwargs = kwargs or {}
        dependencies = dependencies or []
        context = context or {}
        
        # Créer métadonnées
        metadata = JobMetadata(
            job_id=job_id,
            job_type=job_type,
            priority=priority,
            created_at=datetime.now(),
            estimated_duration=estimated_duration,
            dependencies=dependencies,
            user_id=user_id,
            context=context
        )
        
        try:
            # Vérifier dépendances
            if dependencies:
                await self._check_dependencies(dependencies)
            
            # Sélectionner file selon priorité
            queue_name = self.queue_config[priority]['name']
            
            # Soumettre à Celery avec ID personnalisé
            task = self.celery_app.send_task(
                task_name,
                args=args,
                kwargs=kwargs,
                task_id=job_id,
                queue=queue_name,
                routing_key=self.queue_config[priority]['routing_key']
            )
            
            # Enregistrer métadonnées
            self.job_registry[job_id] = metadata
            await self._save_job_metadata(job_id, metadata)
            
            # Créer résultat initial
            result = JobResult(
                job_id=job_id,
                status=JobStatus.PENDING,
                metadata=metadata
            )
            self.active_jobs[job_id] = result
            
            logger.info(f"📤 Job soumis: {job_id} ({task_name}) - File: {queue_name}")
            return job_id
            
        except Exception as e:
            logger.error(f"❌ Erreur soumission job {job_id}: {e}")
            raise
    
    async def get_job_status(self, job_id: str) -> Optional[JobResult]:
        """Récupère le statut d'un job"""
        try:
            # Vérifier cache local
            if job_id in self.active_jobs:
                result = self.active_jobs[job_id]
            else:
                # Charger depuis Redis
                result = await self._load_job_result(job_id)
                if not result:
                    return None
            
            # Mettre à jour avec Celery
            celery_result = AsyncResult(job_id, app=self.celery_app)
            
            if celery_result.state == 'PENDING':
                result.status = JobStatus.PENDING
            elif celery_result.state == 'STARTED':
                result.status = JobStatus.STARTED
            elif celery_result.state == 'SUCCESS':
                result.status = JobStatus.SUCCESS
                result.result = celery_result.result
                result.completed_at = datetime.now()
            elif celery_result.state == 'FAILURE':
                result.status = JobStatus.FAILURE
                result.error = str(celery_result.info)
                result.completed_at = datetime.now()
            elif celery_result.state == 'RETRY':
                result.status = JobStatus.RETRY
            elif celery_result.state == 'REVOKED':
                result.status = JobStatus.REVOKED
            
            # Mettre à jour cache
            self.active_jobs[job_id] = result
            await self._save_job_result(job_id, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Erreur récupération statut job {job_id}: {e}")
            return None
    
    async def cancel_job(self, job_id: str) -> bool:
        """Annule un job"""
        try:
            celery_result = AsyncResult(job_id, app=self.celery_app)
            celery_result.revoke(terminate=True)
            
            # Mettre à jour statut
            if job_id in self.active_jobs:
                self.active_jobs[job_id].status = JobStatus.REVOKED
                await self._save_job_result(job_id, self.active_jobs[job_id])
            
            logger.info(f"🚫 Job annulé: {job_id}")
            return True
            
        except Exception as e:
            logger.error(f"Erreur annulation job {job_id}: {e}")
            return False
    
    async def get_queue_status(self) -> Dict[str, Any]:
        """Récupère le statut des files de jobs"""
        try:
            inspect = self.celery_app.control.inspect()
            
            # Tâches actives
            active_tasks = inspect.active()
            
            # Tâches en attente
            scheduled_tasks = inspect.scheduled()
            
            # Stats par file
            queue_stats = {}
            for priority, config in self.queue_config.items():
                queue_name = config['name']
                queue_stats[queue_name] = {
                    'priority': priority.value,
                    'active_tasks': len(active_tasks.get(queue_name, [])) if active_tasks else 0,
                    'scheduled_tasks': len(scheduled_tasks.get(queue_name, [])) if scheduled_tasks else 0,
                    'max_concurrency': config['concurrency']
                }
            
            return {
                'queues': queue_stats,
                'total_active': sum(len(tasks) for tasks in (active_tasks or {}).values()),
                'total_scheduled': sum(len(tasks) for tasks in (scheduled_tasks or {}).values()),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Erreur statut files: {e}")
            return {'error': str(e)}
    
    async def get_user_jobs(self, user_id: str, status: Optional[JobStatus] = None) -> List[JobResult]:
        """Récupère les jobs d'un utilisateur"""
        user_jobs = []
        
        for job_id, metadata in self.job_registry.items():
            if metadata.user_id == user_id:
                job_result = await self.get_job_status(job_id)
                if job_result and (not status or job_result.status == status):
                    user_jobs.append(job_result)
        
        return sorted(user_jobs, key=lambda x: x.metadata.created_at, reverse=True)
    
    async def cleanup_completed_jobs(self, older_than_days: int = 7):
        """Nettoie les jobs terminés anciens"""
        cutoff_date = datetime.now() - timedelta(days=older_than_days)
        cleaned_count = 0
        
        jobs_to_remove = []
        for job_id, metadata in self.job_registry.items():
            if metadata.created_at < cutoff_date:
                job_result = await self.get_job_status(job_id)
                if job_result and job_result.status in [JobStatus.SUCCESS, JobStatus.FAILURE, JobStatus.REVOKED]:
                    jobs_to_remove.append(job_id)
        
        for job_id in jobs_to_remove:
            await self._cleanup_job(job_id)
            cleaned_count += 1
        
        logger.info(f"🧹 Nettoyage: {cleaned_count} jobs supprimés")
        return cleaned_count
    
    async def _check_dependencies(self, dependencies: List[str]):
        """Vérifie que les dépendances sont satisfaites"""
        for dep_job_id in dependencies:
            dep_result = await self.get_job_status(dep_job_id)
            if not dep_result or dep_result.status != JobStatus.SUCCESS:
                raise ValueError(f"Dépendance non satisfaite: {dep_job_id}")
    
    async def _save_job_metadata(self, job_id: str, metadata: JobMetadata):
        """Sauvegarde les métadonnées d'un job"""
        if self.redis_client:
            key = f"job_metadata:{job_id}"
            data = {
                'job_id': metadata.job_id,
                'job_type': metadata.job_type.value,
                'priority': metadata.priority.value,
                'created_at': metadata.created_at.isoformat(),
                'estimated_duration': metadata.estimated_duration,
                'retry_count': metadata.retry_count,
                'max_retries': metadata.max_retries,
                'context': json.dumps(metadata.context),
                'dependencies': json.dumps(metadata.dependencies),
                'user_id': metadata.user_id
            }
            await asyncio.get_event_loop().run_in_executor(
                None, self.redis_client.hmset, key, data
            )
            await asyncio.get_event_loop().run_in_executor(
                None, self.redis_client.expire, key, 86400 * 30  # 30 jours
            )
    
    async def _save_job_result(self, job_id: str, result: JobResult):
        """Sauvegarde le résultat d'un job"""
        if self.redis_client:
            key = f"job_result:{job_id}"
            data = {
                'job_id': result.job_id,
                'status': result.status.value,
                'result': json.dumps(result.result) if result.result else None,
                'error': result.error,
                'execution_time': result.execution_time,
                'started_at': result.started_at.isoformat() if result.started_at else None,
                'completed_at': result.completed_at.isoformat() if result.completed_at else None,
                'progress': result.progress
            }
            await asyncio.get_event_loop().run_in_executor(
                None, self.redis_client.hmset, key, data
            )
    
    async def _load_job_result(self, job_id: str) -> Optional[JobResult]:
        """Charge le résultat d'un job depuis Redis"""
        if not self.redis_client:
            return None
            
        try:
            key = f"job_result:{job_id}"
            data = await asyncio.get_event_loop().run_in_executor(
                None, self.redis_client.hgetall, key
            )
            
            if not data:
                return None
            
            return JobResult(
                job_id=data['job_id'],
                status=JobStatus(data['status']),
                result=json.loads(data['result']) if data.get('result') else None,
                error=data.get('error'),
                execution_time=float(data['execution_time']) if data.get('execution_time') else None,
                started_at=datetime.fromisoformat(data['started_at']) if data.get('started_at') else None,
                completed_at=datetime.fromisoformat(data['completed_at']) if data.get('completed_at') else None,
                progress=float(data.get('progress', 0))
            )
            
        except Exception as e:
            logger.error(f"Erreur chargement résultat job {job_id}: {e}")
            return None
    
    async def _cleanup_job(self, job_id: str):
        """Nettoie complètement un job"""
        # Supprimer du registre
        self.job_registry.pop(job_id, None)
        self.active_jobs.pop(job_id, None)
        
        # Supprimer de Redis
        if self.redis_client:
            await asyncio.get_event_loop().run_in_executor(
                None, self.redis_client.delete, f"job_metadata:{job_id}", f"job_result:{job_id}"
            )


# Instance globale
_background_job_manager: Optional[BackgroundJobManager] = None


async def get_background_job_manager() -> BackgroundJobManager:
    """Factory pour obtenir le gestionnaire de jobs"""
    global _background_job_manager
    
    if _background_job_manager is None:
        _background_job_manager = BackgroundJobManager()
        await _background_job_manager.initialize()
    
    return _background_job_manager


# Décorateurs pour faciliter l'utilisation

def background_task(priority: JobPriority = JobPriority.NORMAL,
                   job_type: JobType = JobType.DATA_PROCESSING,
                   estimated_duration: Optional[int] = None):
    """Décorateur pour créer une tâche Celery avec métadonnées"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            job_manager = await get_background_job_manager()
            
            job_id = await job_manager.submit_job(
                task_name=f"{func.__module__}.{func.__name__}",
                args=args,
                kwargs=kwargs,
                priority=priority,
                job_type=job_type,
                estimated_duration=estimated_duration
            )
            
            return job_id
        
        return wrapper
    return decorator


def celery_task(priority: JobPriority = JobPriority.NORMAL, 
               bind=True, 
               base=CustomCeleryTask):
    """Décorateur pour créer des tâches Celery avec base personnalisée"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Cette fonction sera remplacée par la définition de tâche Celery
            return func(*args, **kwargs)
        
        # Configuration de la tâche
        queue_config = {
            JobPriority.CRITICAL: 'critical',
            JobPriority.HIGH: 'high_priority', 
            JobPriority.NORMAL: 'normal',
            JobPriority.LOW: 'low_priority',
            JobPriority.BATCH: 'batch'
        }
        
        wrapper._celery_config = {
            'bind': bind,
            'base': base,
            'queue': queue_config.get(priority, 'normal')
        }
        
        return wrapper
    return decorator


# Tâches utilitaires

@celery_task(priority=JobPriority.LOW)
async def cleanup_old_jobs_task(older_than_days: int = 7):
    """Tâche de nettoyage des anciens jobs"""
    job_manager = await get_background_job_manager()
    cleaned_count = await job_manager.cleanup_completed_jobs(older_than_days)
    
    return {
        'cleaned_jobs': cleaned_count,
        'timestamp': datetime.now().isoformat()
    }


@celery_task(priority=JobPriority.NORMAL)
async def health_check_task():
    """Tâche de vérification de santé du système"""
    try:
        job_manager = await get_background_job_manager()
        queue_status = await job_manager.get_queue_status()
        
        return {
            'status': 'healthy',
            'queue_status': queue_status,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }


# Fonctions utilitaires pour l'API

async def schedule_periodic_cleanup():
    """Programme le nettoyage périodique"""
    try:
        job_manager = await get_background_job_manager()
        
        # Nettoyer quotidiennement
        job_id = await job_manager.submit_job(
            task_name='cleanup_old_jobs_task',
            args=[7],  # 7 jours
            priority=JobPriority.LOW,
            job_type=JobType.MAINTENANCE
        )
        
        logger.info(f"📅 Nettoyage programmé: {job_id}")
        return job_id
        
    except Exception as e:
        logger.error(f"Erreur programmation nettoyage: {e}")
        return None


async def get_system_job_stats() -> Dict[str, Any]:
    """Récupère les statistiques système des jobs"""
    try:
        job_manager = await get_background_job_manager()
        queue_status = await job_manager.get_queue_status()
        
        # Compter jobs par statut
        status_counts = {status.value: 0 for status in JobStatus}
        for job_result in job_manager.active_jobs.values():
            status_counts[job_result.status.value] += 1
        
        return {
            'queue_status': queue_status,
            'job_counts_by_status': status_counts,
            'total_registered_jobs': len(job_manager.job_registry),
            'active_jobs': len(job_manager.active_jobs),
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Erreur stats jobs: {e}")
        return {'error': str(e)}