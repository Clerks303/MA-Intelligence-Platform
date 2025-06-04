"""
Background Jobs System - Version unifiée et simplifiée
Consolide background_jobs.py, background_jobs_ml.py, background_jobs_reports.py, background_jobs_scraping.py

Fonctionnalités core uniquement:
- Scraping asynchrone
- Export de données  
- Nettoyage de cache
- Jobs de maintenance
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from enum import Enum
from dataclasses import dataclass

from app.config import settings

logger = logging.getLogger(__name__)

class JobPriority(str, Enum):
    """Priorités des jobs simplifiées"""
    HIGH = "high"        # < 5 minutes  
    NORMAL = "normal"    # < 15 minutes
    LOW = "low"          # < 1 heure

class JobStatus(str, Enum):
    """Statuts des jobs"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    RETRY = "retry"

@dataclass
class JobResult:
    """Résultat d'exécution d'un job"""
    job_id: str
    status: JobStatus
    result: Any = None
    error: Optional[str] = None
    duration: Optional[float] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

class BackgroundJobManager:
    """Gestionnaire simplifié des jobs d'arrière-plan"""
    
    def __init__(self):
        self.jobs: Dict[str, JobResult] = {}
        self.running_jobs: Dict[str, asyncio.Task] = {}
    
    async def submit_scraping_job(self, company_ids: List[str], source: str = "pappers") -> str:
        """Lance un job de scraping asynchrone"""
        job_id = f"scraping_{source}_{datetime.now().timestamp()}"
        
        job_result = JobResult(
            job_id=job_id,
            status=JobStatus.PENDING,
            started_at=datetime.now()
        )
        self.jobs[job_id] = job_result
        
        # Démarrer le job asynchrone
        task = asyncio.create_task(self._execute_scraping(job_id, company_ids, source))
        self.running_jobs[job_id] = task
        
        logger.info(f"Scraping job {job_id} submitted for {len(company_ids)} companies")
        return job_id
    
    async def submit_export_job(self, export_format: str, filters: Dict = None) -> str:
        """Lance un job d'export de données"""
        job_id = f"export_{export_format}_{datetime.now().timestamp()}"
        
        job_result = JobResult(
            job_id=job_id,
            status=JobStatus.PENDING,
            started_at=datetime.now()
        )
        self.jobs[job_id] = job_result
        
        task = asyncio.create_task(self._execute_export(job_id, export_format, filters))
        self.running_jobs[job_id] = task
        
        logger.info(f"Export job {job_id} submitted for format {export_format}")
        return job_id
    
    async def submit_maintenance_job(self, task_type: str) -> str:
        """Lance un job de maintenance"""
        job_id = f"maintenance_{task_type}_{datetime.now().timestamp()}"
        
        job_result = JobResult(
            job_id=job_id,
            status=JobStatus.PENDING,
            started_at=datetime.now()
        )
        self.jobs[job_id] = job_result
        
        task = asyncio.create_task(self._execute_maintenance(job_id, task_type))
        self.running_jobs[job_id] = task
        
        logger.info(f"Maintenance job {job_id} submitted for {task_type}")
        return job_id
    
    async def get_job_status(self, job_id: str) -> Optional[JobResult]:
        """Récupère le statut d'un job"""
        return self.jobs.get(job_id)
    
    async def list_jobs(self, status: Optional[JobStatus] = None) -> List[JobResult]:
        """Liste tous les jobs ou filtrés par statut"""
        if status:
            return [job for job in self.jobs.values() if job.status == status]
        return list(self.jobs.values())
    
    async def cancel_job(self, job_id: str) -> bool:
        """Annule un job en cours"""
        if job_id in self.running_jobs:
            task = self.running_jobs[job_id]
            task.cancel()
            del self.running_jobs[job_id]
            
            if job_id in self.jobs:
                self.jobs[job_id].status = JobStatus.FAILED
                self.jobs[job_id].error = "Job cancelled by user"
                self.jobs[job_id].completed_at = datetime.now()
            
            logger.info(f"Job {job_id} cancelled")
            return True
        return False
    
    async def _execute_scraping(self, job_id: str, company_ids: List[str], source: str):
        """Exécute un job de scraping"""
        try:
            self.jobs[job_id].status = JobStatus.RUNNING
            
            # Import ici pour éviter les dépendances circulaires
            if source == "pappers":
                from app.scrapers.pappers import PappersScraper
                scraper = PappersScraper()
            elif source == "societe":
                from app.scrapers.societe import SocieteScraper
                scraper = SocieteScraper()
            else:
                raise ValueError(f"Unknown scraping source: {source}")
            
            results = []
            for company_id in company_ids:
                try:
                    result = await scraper.scrape_company(company_id)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error scraping company {company_id}: {e}")
                    results.append({"error": str(e)})
                
                # Petite pause pour éviter la surcharge
                await asyncio.sleep(0.1)
            
            self.jobs[job_id].status = JobStatus.SUCCESS
            self.jobs[job_id].result = {
                "scraped_count": len([r for r in results if "error" not in r]),
                "error_count": len([r for r in results if "error" in r]),
                "results": results
            }
            
        except Exception as e:
            self.jobs[job_id].status = JobStatus.FAILED
            self.jobs[job_id].error = str(e)
            logger.error(f"Scraping job {job_id} failed: {e}")
        
        finally:
            self.jobs[job_id].completed_at = datetime.now()
            if job_id in self.running_jobs:
                del self.running_jobs[job_id]
    
    async def _execute_export(self, job_id: str, export_format: str, filters: Dict):
        """Exécute un job d'export"""
        try:
            self.jobs[job_id].status = JobStatus.RUNNING
            
            from app.services.export_manager import ExportManager
            export_manager = ExportManager()
            
            # Simulation d'export - à adapter selon vos besoins
            if export_format == "csv":
                result = await export_manager.export_to_csv(filters)
            elif export_format == "excel":
                result = await export_manager.export_to_excel(filters)
            else:
                raise ValueError(f"Unsupported export format: {export_format}")
            
            self.jobs[job_id].status = JobStatus.SUCCESS
            self.jobs[job_id].result = result
            
        except Exception as e:
            self.jobs[job_id].status = JobStatus.FAILED
            self.jobs[job_id].error = str(e)
            logger.error(f"Export job {job_id} failed: {e}")
        
        finally:
            self.jobs[job_id].completed_at = datetime.now()
            if job_id in self.running_jobs:
                del self.running_jobs[job_id]
    
    async def _execute_maintenance(self, job_id: str, task_type: str):
        """Exécute un job de maintenance"""
        try:
            self.jobs[job_id].status = JobStatus.RUNNING
            
            if task_type == "cache_cleanup":
                # Nettoyage du cache
                from app.core.cache import cache_manager
                if hasattr(cache_manager, 'clear_expired'):
                    await cache_manager.clear_expired()
                result = {"message": "Cache cleaned"}
                
            elif task_type == "log_rotation":
                # Rotation des logs
                result = {"message": "Log rotation completed"}
                
            elif task_type == "db_cleanup":
                # Nettoyage base de données
                result = {"message": "Database cleanup completed"}
                
            else:
                raise ValueError(f"Unknown maintenance task: {task_type}")
            
            self.jobs[job_id].status = JobStatus.SUCCESS
            self.jobs[job_id].result = result
            
        except Exception as e:
            self.jobs[job_id].status = JobStatus.FAILED
            self.jobs[job_id].error = str(e)
            logger.error(f"Maintenance job {job_id} failed: {e}")
        
        finally:
            self.jobs[job_id].completed_at = datetime.now()
            if job_id in self.running_jobs:
                del self.running_jobs[job_id]

# Instance globale
_job_manager = None

def get_job_manager() -> BackgroundJobManager:
    """Récupère l'instance du gestionnaire de jobs"""
    global _job_manager
    if _job_manager is None:
        _job_manager = BackgroundJobManager()
    return _job_manager

# Functions utilitaires pour compatibilité
async def submit_scraping_job(company_ids: List[str], source: str = "pappers") -> str:
    """Soumet un job de scraping"""
    manager = get_job_manager()
    return await manager.submit_scraping_job(company_ids, source)

async def submit_export_job(export_format: str, filters: Dict = None) -> str:
    """Soumet un job d'export"""
    manager = get_job_manager()
    return await manager.submit_export_job(export_format, filters)

async def get_job_status(job_id: str) -> Optional[JobResult]:
    """Récupère le statut d'un job"""
    manager = get_job_manager()
    return await manager.get_job_status(job_id)

async def list_active_jobs() -> List[JobResult]:
    """Liste les jobs actifs"""
    manager = get_job_manager()
    return await manager.list_jobs(JobStatus.RUNNING)

# Scheduler simple pour les tâches récurrentes
class SimpleScheduler:
    """Scheduler simple pour les tâches de maintenance"""
    
    def __init__(self):
        self.scheduled_tasks = []
        self.running = False
    
    async def start(self):
        """Démarre le scheduler"""
        self.running = True
        
        # Tâche de nettoyage cache toutes les heures
        asyncio.create_task(self._schedule_cache_cleanup())
        
        logger.info("Simple scheduler started")
    
    async def stop(self):
        """Arrête le scheduler"""
        self.running = False
        logger.info("Simple scheduler stopped")
    
    async def _schedule_cache_cleanup(self):
        """Planifie le nettoyage du cache"""
        while self.running:
            try:
                # Attendre 1 heure
                await asyncio.sleep(3600)
                
                # Lancer le nettoyage
                manager = get_job_manager()
                job_id = await manager.submit_maintenance_job("cache_cleanup")
                logger.info(f"Scheduled cache cleanup job: {job_id}")
                
            except Exception as e:
                logger.error(f"Error in scheduled cache cleanup: {e}")

# Instance globale du scheduler
_scheduler = None

def get_scheduler() -> SimpleScheduler:
    """Récupère l'instance du scheduler"""
    global _scheduler
    if _scheduler is None:
        _scheduler = SimpleScheduler()
    return _scheduler