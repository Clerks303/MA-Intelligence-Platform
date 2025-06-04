"""
Celery Tasks - ML Scheduler
M&A Intelligence Platform
"""

import subprocess
import logging
import sys
from pathlib import Path
from datetime import datetime
from celery import current_task
from celery_app import app

logger = logging.getLogger(__name__)

@app.task(bind=True, name='tasks.run_ml_scoring')
def run_ml_scoring(self, batch_size: int = 1000, specific_companies: list = None):
    """
    Tâche Celery pour lancer le scoring ML
    
    Args:
        batch_size: Taille des batches à traiter
        specific_companies: Liste d'IDs d'entreprises spécifiques (optionnel)
    """
    logger.info(f"🚀 Démarrage tâche ML scoring - Task ID: {self.request.id}")
    
    try:
        # Mise à jour du statut
        self.update_state(
            state='PROGRESS',
            meta={'status': 'Démarrage du scoring ML', 'progress': 0}
        )
        
        # Préparation de la commande
        ml_service_path = Path(__file__).parent.parent / 'ml-service'
        script_path = ml_service_path / 'batch_scoring.py'
        
        cmd = ['python', str(script_path), '--batch-size', str(batch_size)]
        
        if specific_companies:
            cmd.extend(['--companies'] + [str(c) for c in specific_companies])
        
        # Exécution du script ML
        logger.info(f"📋 Commande: {' '.join(cmd)}")
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(ml_service_path)
        )
        
        # Monitoring du processus
        self.update_state(
            state='PROGRESS',
            meta={'status': 'Calcul des scores en cours...', 'progress': 50}
        )
        
        stdout, stderr = process.communicate()
        
        if process.returncode == 0:
            logger.info("✅ Scoring ML terminé avec succès")
            return {
                'status': 'SUCCESS',
                'message': 'Scoring ML terminé avec succès',
                'output': stdout,
                'processed_at': datetime.utcnow().isoformat()
            }
        else:
            logger.error(f"❌ Erreur scoring ML: {stderr}")
            raise Exception(f"Scoring ML échoué: {stderr}")
            
    except Exception as e:
        logger.error(f"❌ Erreur tâche scoring ML: {e}")
        self.update_state(
            state='FAILURE',
            meta={'status': f'Erreur: {str(e)}', 'error': str(e)}
        )
        raise

@app.task(bind=True, name='tasks.train_models')
def train_models(self, model_types: list = None):
    """
    Tâche Celery pour ré-entraîner les modèles ML
    
    Args:
        model_types: Types de modèles à entraîner (optionnel)
    """
    logger.info(f"🧠 Démarrage entraînement modèles - Task ID: {self.request.id}")
    
    try:
        self.update_state(
            state='PROGRESS',
            meta={'status': 'Démarrage entraînement des modèles', 'progress': 0}
        )
        
        # Préparation de la commande
        ml_service_path = Path(__file__).parent.parent / 'ml-service'
        script_path = ml_service_path / 'train_models.py'
        
        cmd = ['python', str(script_path)]
        
        if model_types:
            cmd.extend(['--models'] + model_types)
        
        # Exécution de l'entraînement
        logger.info(f"🏋️ Commande entraînement: {' '.join(cmd)}")
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(ml_service_path)
        )
        
        self.update_state(
            state='PROGRESS',
            meta={'status': 'Entraînement en cours...', 'progress': 50}
        )
        
        stdout, stderr = process.communicate()
        
        if process.returncode == 0:
            logger.info("✅ Entraînement modèles terminé")
            return {
                'status': 'SUCCESS',
                'message': 'Entraînement terminé avec succès',
                'output': stdout,
                'trained_at': datetime.utcnow().isoformat()
            }
        else:
            logger.error(f"❌ Erreur entraînement: {stderr}")
            raise Exception(f"Entraînement échoué: {stderr}")
            
    except Exception as e:
        logger.error(f"❌ Erreur tâche entraînement: {e}")
        self.update_state(
            state='FAILURE',
            meta={'status': f'Erreur: {str(e)}', 'error': str(e)}
        )
        raise

@app.task(bind=True, name='tasks.data_quality_check')
def data_quality_check(self):
    """
    Tâche Celery pour vérifier la qualité des données
    """
    logger.info(f"🔍 Démarrage vérification qualité données - Task ID: {self.request.id}")
    
    try:
        self.update_state(
            state='PROGRESS',
            meta={'status': 'Vérification qualité des données', 'progress': 0}
        )
        
        # TODO: Implémenter la vérification qualité
        # - Vérifier les données manquantes
        # - Détecter les outliers
        # - Vérifier la cohérence des données
        # - Générer un rapport
        
        self.update_state(
            state='PROGRESS',
            meta={'status': 'Analyse en cours...', 'progress': 50}
        )
        
        # Simulation de vérification
        import time
        time.sleep(5)
        
        logger.info("✅ Vérification qualité terminée")
        return {
            'status': 'SUCCESS',
            'message': 'Vérification qualité terminée',
            'checked_at': datetime.utcnow().isoformat(),
            'quality_score': 85  # Exemple
        }
        
    except Exception as e:
        logger.error(f"❌ Erreur vérification qualité: {e}")
        self.update_state(
            state='FAILURE',
            meta={'status': f'Erreur: {str(e)}', 'error': str(e)}
        )
        raise

@app.task(name='tasks.trigger_manual_scoring')
def trigger_manual_scoring(company_ids: list = None, user_id: int = None):
    """
    Tâche pour déclencher un scoring manuel spécifique
    
    Args:
        company_ids: Liste des IDs d'entreprises à scorer
        user_id: ID de l'utilisateur qui déclenche le scoring
    """
    logger.info(f"👤 Scoring manuel déclenché par utilisateur {user_id}")
    
    # Appel de la tâche principale avec les entreprises spécifiques
    return run_ml_scoring.delay(
        batch_size=len(company_ids) if company_ids else 1000,
        specific_companies=company_ids
    )

# Fonction utilitaire pour monitoring
@app.task(name='tasks.get_task_status')
def get_task_status(task_id: str):
    """Récupère le statut d'une tâche"""
    result = app.AsyncResult(task_id)
    return {
        'task_id': task_id,
        'status': result.status,
        'result': result.result,
        'info': result.info
    }