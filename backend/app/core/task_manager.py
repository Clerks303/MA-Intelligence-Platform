"""
Gestionnaire de tâches asynchrones pour M&A Intelligence Platform
US-005: Task manager avec Celery pour traitement asynchrone des opérations longues

Features:
- Celery avec Redis comme broker
- Tâches de scraping asynchrones
- Export CSV en arrière-plan
- Traitement par lots optimisé
- Monitoring et retry intelligents
- Progress tracking temps réel
- Rate limiting intégré
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Callable, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from functools import wraps
import traceback

# Celery imports
from celery import Celery, Task, signature
from celery.result import AsyncResult
from celery.signals import task_prerun, task_postrun, task_failure
from kombu import Queue

from app.config import settings
from app.core.logging_system import get_logger, LogCategory, audit_logger
from app.core.cache_manager import get_cache_manager
from app.core.database_optimizer import get_database_optimizer

logger = get_logger("task_manager", LogCategory.PERFORMANCE)


class TaskStatus(str, Enum):
    """États des tâches"""
    PENDING = "pending"
    STARTED = "started"
    PROGRESS = "progress"
    SUCCESS = "success"
    FAILURE = "failure"
    REVOKED = "revoked"
    RETRY = "retry"


class TaskPriority(str, Enum):
    """Priorités des tâches"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class TaskProgress:
    """Progrès d'une tâche"""
    current: int = 0
    total: int = 0
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    start_time: datetime = field(default_factory=datetime.now)
    
    @property
    def percentage(self) -> float:
        return (self.current / self.total * 100) if self.total > 0 else 0.0
    
    @property
    def elapsed_seconds(self) -> float:
        return (datetime.now() - self.start_time).total_seconds()
    
    @property
    def eta_seconds(self) -> Optional[float]:
        if self.current > 0 and self.total > self.current:
            rate = self.current / self.elapsed_seconds
            remaining = self.total - self.current
            return remaining / rate if rate > 0 else None
        return None


@dataclass
class TaskResult:
    """Résultat d'une tâche"""
    task_id: str
    status: TaskStatus
    result: Any = None
    error: Optional[str] = None
    progress: Optional[TaskProgress] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'task_id': self.task_id,
            'status': self.status.value,
            'result': self.result,
            'error': self.error,
            'progress': asdict(self.progress) if self.progress else None,
            'metadata': self.metadata,
            'created_at': self.created_at.isoformat(),
            'completed_at': self.completed_at.isoformat() if self.completed_at else None
        }


class ProgressTask(Task):
    """Tâche Celery avec support progress tracking"""
    
    def update_progress(self, current: int, total: int, message: str = "", **details):
        """Met à jour le progrès de la tâche"""
        progress = TaskProgress(
            current=current,
            total=total,
            message=message,
            details=details
        )
        
        self.update_state(
            state=TaskStatus.PROGRESS.value,
            meta={
                'progress': asdict(progress)
            }
        )
        
        logger.debug(f"Task {self.request.id} progress: {progress.percentage:.1f}% - {message}")


# Configuration Celery
def create_celery_app() -> Celery:
    """Crée et configure l'application Celery"""
    
    # URL Redis pour broker et backend
    redis_url = getattr(settings, 'REDIS_URL', 'redis://localhost:6379/0')
    
    celery_app = Celery(
        'ma_intelligence_tasks',
        broker=redis_url,
        backend=redis_url,
        include=[
            'app.core.task_manager',
            'app.tasks.scraping_tasks',
            'app.tasks.export_tasks'
        ]
    )
    
    # Configuration Celery
    celery_app.conf.update(
        # Sérialisation
        task_serializer='json',
        accept_content=['json'],
        result_serializer='json',
        timezone='UTC',
        enable_utc=True,
        
        # Queues et routing
        task_routes={
            'app.tasks.scraping_tasks.*': {'queue': 'scraping'},
            'app.tasks.export_tasks.*': {'queue': 'exports'},
            'app.core.task_manager.*': {'queue': 'system'}
        },
        
        # Worker configuration
        worker_prefetch_multiplier=1,
        task_acks_late=True,
        worker_max_tasks_per_child=1000,
        
        # Retry policy
        task_annotations={
            '*': {
                'rate_limit': '10/s',
                'time_limit': 3600,  # 1 heure max
                'soft_time_limit': 3300,  # 55 minutes
                'max_retries': 3,
                'default_retry_delay': 60
            },
            'app.tasks.scraping_tasks.scrape_companies': {
                'rate_limit': '5/s',  # Plus lent pour scraping
                'time_limit': 7200   # 2 heures pour gros volumes
            }
        },
        
        # Monitoring
        task_send_sent_event=True,
        task_track_started=True,
        worker_send_task_events=True,
        
        # Résultats
        result_expires=3600,  # 1 heure
        result_backend_transport_options={
            'master_name': 'mymaster'
        }
    )
    
    # Queues personnalisées
    celery_app.conf.task_routes = {
        'scraping_queue': Queue('scraping', routing_key='scraping'),
        'export_queue': Queue('exports', routing_key='exports'),
        'system_queue': Queue('system', routing_key='system')
    }
    
    return celery_app


# Instance Celery globale
celery_app = create_celery_app()


class TaskManager:
    """Gestionnaire de tâches asynchrones"""
    
    def __init__(self):
        self.active_tasks: Dict[str, TaskResult] = {}
        self.task_history: List[TaskResult] = []
        self.stats = {
            'total_tasks': 0,
            'successful_tasks': 0,
            'failed_tasks': 0,
            'avg_execution_time': 0.0
        }
        
        logger.info("📋 TaskManager initialisé")
    
    async def submit_task(self, 
                         task_name: str, 
                         args: tuple = (), 
                         kwargs: dict = None,
                         priority: TaskPriority = TaskPriority.NORMAL,
                         queue: str = None,
                         user_id: str = None) -> str:
        """Soumet une tâche asynchrone"""
        
        kwargs = kwargs or {}
        
        # Métadonnées de la tâche
        metadata = {
            'submitted_by': user_id,
            'submitted_at': datetime.now().isoformat(),
            'priority': priority.value,
            'queue': queue or 'default'
        }
        
        try:
            # Options de la tâche selon la priorité
            task_options = {
                'priority': self._get_priority_level(priority),
                'queue': queue
            }
            
            # Soumission via Celery
            task_signature = signature(task_name, args=args, kwargs=kwargs, **task_options)
            async_result = task_signature.apply_async()
            
            task_id = async_result.id
            
            # Enregistrer la tâche
            task_result = TaskResult(
                task_id=task_id,
                status=TaskStatus.PENDING,
                metadata=metadata
            )
            
            self.active_tasks[task_id] = task_result
            self.stats['total_tasks'] += 1
            
            # Audit log
            audit_logger.audit(
                action="task_submitted",
                resource_type="task",
                resource_id=task_id,
                success=True,
                details={
                    'task_name': task_name,
                    'priority': priority.value,
                    'user_id': user_id
                }
            )
            
            logger.info(f"📤 Tâche soumise: {task_name} ({task_id}) - Priorité: {priority.value}")
            
            return task_id
            
        except Exception as e:
            logger.error(f"❌ Erreur soumission tâche {task_name}: {e}")
            raise
    
    def _get_priority_level(self, priority: TaskPriority) -> int:
        """Convertit la priorité en niveau numérique"""
        priority_levels = {
            TaskPriority.LOW: 1,
            TaskPriority.NORMAL: 5,
            TaskPriority.HIGH: 8,
            TaskPriority.CRITICAL: 10
        }
        return priority_levels.get(priority, 5)
    
    async def get_task_status(self, task_id: str) -> Optional[TaskResult]:
        """Récupère le statut d'une tâche"""
        
        # Vérifier cache local
        if task_id in self.active_tasks:
            task_result = self.active_tasks[task_id]
            
            # Mettre à jour depuis Celery
            async_result = AsyncResult(task_id, app=celery_app)
            
            # Mapper statut Celery
            celery_status = async_result.status
            if celery_status == 'PENDING':
                task_result.status = TaskStatus.PENDING
            elif celery_status == 'STARTED':
                task_result.status = TaskStatus.STARTED
            elif celery_status == 'PROGRESS':
                task_result.status = TaskStatus.PROGRESS
                # Récupérer progrès depuis meta
                if async_result.info and 'progress' in async_result.info:
                    progress_data = async_result.info['progress']
                    task_result.progress = TaskProgress(**progress_data)
            elif celery_status == 'SUCCESS':
                task_result.status = TaskStatus.SUCCESS
                task_result.result = async_result.result
                task_result.completed_at = datetime.now()
                
                # Déplacer vers historique
                self._move_to_history(task_id)
                self.stats['successful_tasks'] += 1
                
            elif celery_status == 'FAILURE':
                task_result.status = TaskStatus.FAILURE
                task_result.error = str(async_result.info)
                task_result.completed_at = datetime.now()
                
                # Déplacer vers historique
                self._move_to_history(task_id)
                self.stats['failed_tasks'] += 1
                
            elif celery_status == 'REVOKED':
                task_result.status = TaskStatus.REVOKED
                task_result.completed_at = datetime.now()
                self._move_to_history(task_id)
            
            return task_result
        
        # Chercher dans l'historique
        for task in self.task_history:
            if task.task_id == task_id:
                return task
        
        return None
    
    def _move_to_history(self, task_id: str):
        """Déplace une tâche vers l'historique"""
        if task_id in self.active_tasks:
            task = self.active_tasks.pop(task_id)
            self.task_history.append(task)
            
            # Nettoyer historique (garder 1000 dernières)
            if len(self.task_history) > 1000:
                self.task_history = self.task_history[-1000:]
    
    async def cancel_task(self, task_id: str, user_id: str = None) -> bool:
        """Annule une tâche"""
        try:
            celery_app.control.revoke(task_id, terminate=True)
            
            if task_id in self.active_tasks:
                task = self.active_tasks[task_id]
                task.status = TaskStatus.REVOKED
                task.completed_at = datetime.now()
                self._move_to_history(task_id)
            
            # Audit log
            audit_logger.audit(
                action="task_cancelled",
                resource_type="task",
                resource_id=task_id,
                success=True,
                details={'cancelled_by': user_id}
            )
            
            logger.info(f"❌ Tâche annulée: {task_id}")
            return True
            
        except Exception as e:
            logger.error(f"Erreur annulation tâche {task_id}: {e}")
            return False
    
    async def get_active_tasks(self, user_id: str = None) -> List[TaskResult]:
        """Récupère les tâches actives"""
        tasks = list(self.active_tasks.values())
        
        if user_id:
            tasks = [
                task for task in tasks 
                if task.metadata.get('submitted_by') == user_id
            ]
        
        return tasks
    
    async def get_task_history(self, 
                              user_id: str = None, 
                              limit: int = 50,
                              status: TaskStatus = None) -> List[TaskResult]:
        """Récupère l'historique des tâches"""
        tasks = self.task_history.copy()
        
        if user_id:
            tasks = [
                task for task in tasks 
                if task.metadata.get('submitted_by') == user_id
            ]
        
        if status:
            tasks = [task for task in tasks if task.status == status]
        
        # Tri par date (plus récent en premier)
        tasks.sort(key=lambda x: x.created_at, reverse=True)
        
        return tasks[:limit]
    
    def get_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques des tâches"""
        
        # Stats temps réel
        active_count = len(self.active_tasks)
        recent_tasks = [
            task for task in self.task_history 
            if task.created_at > datetime.now() - timedelta(hours=24)
        ]
        
        # Calcul temps d'exécution moyen
        completed_tasks = [
            task for task in recent_tasks 
            if task.completed_at and task.status == TaskStatus.SUCCESS
        ]
        
        avg_execution_time = 0.0
        if completed_tasks:
            execution_times = [
                (task.completed_at - task.created_at).total_seconds()
                for task in completed_tasks
            ]
            avg_execution_time = sum(execution_times) / len(execution_times)
        
        # Stats par queue
        queue_stats = {}
        for task in list(self.active_tasks.values()) + recent_tasks:
            queue = task.metadata.get('queue', 'default')
            if queue not in queue_stats:
                queue_stats[queue] = {'active': 0, 'completed': 0, 'failed': 0}
            
            if task.task_id in self.active_tasks:
                queue_stats[queue]['active'] += 1
            elif task.status == TaskStatus.SUCCESS:
                queue_stats[queue]['completed'] += 1
            elif task.status == TaskStatus.FAILURE:
                queue_stats[queue]['failed'] += 1
        
        return {
            'overview': {
                'active_tasks': active_count,
                'total_tasks': self.stats['total_tasks'],
                'successful_tasks': self.stats['successful_tasks'],
                'failed_tasks': self.stats['failed_tasks'],
                'success_rate': (
                    self.stats['successful_tasks'] / max(1, self.stats['total_tasks']) * 100
                )
            },
            'performance': {
                'avg_execution_time_seconds': round(avg_execution_time, 2),
                'tasks_last_24h': len(recent_tasks)
            },
            'queues': queue_stats,
            'active_tasks_by_status': {
                status.value: len([
                    task for task in self.active_tasks.values() 
                    if task.status == status
                ])
                for status in TaskStatus
            }
        }
    
    async def cleanup_old_tasks(self, days: int = 7):
        """Nettoie les anciennes tâches"""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        initial_count = len(self.task_history)
        self.task_history = [
            task for task in self.task_history 
            if task.created_at > cutoff_date
        ]
        
        cleaned_count = initial_count - len(self.task_history)
        
        if cleaned_count > 0:
            logger.info(f"🧹 Nettoyage historique tâches: {cleaned_count} tâches supprimées")


# Instance globale du gestionnaire
_task_manager: Optional[TaskManager] = None


async def get_task_manager() -> TaskManager:
    """Factory pour obtenir le gestionnaire de tâches"""
    global _task_manager
    
    if _task_manager is None:
        _task_manager = TaskManager()
    
    return _task_manager


# Décorateurs pour créer des tâches

def celery_task(name: str = None, queue: str = None, **celery_options):
    """Décorateur pour créer une tâche Celery avec monitoring"""
    
    def decorator(func):
        
        @celery_app.task(
            name=name or f"{func.__module__}.{func.__name__}",
            base=ProgressTask,
            bind=True,
            queue=queue,
            **celery_options
        )
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            start_time = time.time()
            
            try:
                logger.info(f"🚀 Démarrage tâche: {self.name} ({self.request.id})")
                
                # Exécution de la fonction
                result = func(self, *args, **kwargs)
                
                execution_time = time.time() - start_time
                logger.info(f"✅ Tâche terminée: {self.name} ({self.request.id}) - {execution_time:.2f}s")
                
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                error_msg = f"Erreur tâche {self.name}: {str(e)}"
                
                logger.error(f"❌ {error_msg} - {execution_time:.2f}s")
                logger.error(f"Traceback: {traceback.format_exc()}")
                
                # Retry selon la configuration
                if self.request.retries < self.max_retries:
                    logger.info(f"🔄 Retry tâche {self.name} ({self.request.retries + 1}/{self.max_retries})")
                    raise self.retry(countdown=60 * (self.request.retries + 1))
                
                raise
        
        return wrapper
    return decorator


# Tâches système

@celery_task(name="system.cleanup_cache", queue="system")
def cleanup_cache_task(self):
    """Tâche de nettoyage du cache"""
    
    async def _cleanup():
        cache_manager = await get_cache_manager()
        
        # Nettoyer cache expiré
        # (Le nettoyage automatique se fait déjà dans le cache manager)
        
        # Stats cache
        stats = cache_manager.get_stats()
        
        return {
            'cache_stats': stats,
            'cleanup_time': datetime.now().isoformat()
        }
    
    # Exécuter dans une boucle d'événements
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(_cleanup())
    finally:
        loop.close()


@celery_task(name="system.database_maintenance", queue="system")
def database_maintenance_task(self):
    """Tâche de maintenance base de données"""
    
    async def _maintenance():
        db_optimizer = await get_database_optimizer()
        
        # Analyser les requêtes lentes
        slow_queries = await db_optimizer.analyze_slow_queries()
        
        # Stats performance
        perf_stats = db_optimizer.get_performance_stats()
        
        return {
            'slow_queries_analysis': slow_queries,
            'performance_stats': perf_stats,
            'maintenance_time': datetime.now().isoformat()
        }
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(_maintenance())
    finally:
        loop.close()


# Fonctions utilitaires

async def schedule_periodic_tasks():
    """Programme les tâches périodiques"""
    
    # Nettoyage cache toutes les heures
    celery_app.conf.beat_schedule = {
        'cleanup-cache': {
            'task': 'system.cleanup_cache',
            'schedule': 3600.0,  # 1 heure
        },
        'database-maintenance': {
            'task': 'system.database_maintenance',
            'schedule': 7200.0,  # 2 heures
        }
    }
    
    logger.info("⏰ Tâches périodiques programmées")


# Signal handlers pour monitoring

@task_prerun.connect
def task_prerun_handler(task_id, task, *args, **kwargs):
    """Handler avant exécution tâche"""
    logger.debug(f"⏳ Démarrage tâche: {task.name} ({task_id})")


@task_postrun.connect  
def task_postrun_handler(task_id, task, *args, **kwargs):
    """Handler après exécution tâche"""
    logger.debug(f"✅ Fin tâche: {task.name} ({task_id})")


@task_failure.connect
def task_failure_handler(task_id, exception, einfo, *args, **kwargs):
    """Handler échec tâche"""
    logger.error(f"❌ Échec tâche: {task_id} - {exception}")


# Point d'entrée worker
if __name__ == '__main__':
    celery_app.start()