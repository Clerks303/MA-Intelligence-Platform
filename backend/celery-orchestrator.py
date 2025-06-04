#!/usr/bin/env python3
"""
Celery Orchestrator - ML Scoring
M&A Intelligence Platform

Configuration et tâches Celery pour orchestrer le scoring ML
"""

import os
import sys
import subprocess
import logging
from datetime import datetime, timedelta
from typing import List, Optional

from celery import Celery
from celery.schedules import crontab
from celery import current_task
from kombu import Queue

# Configuration
from dotenv import load_dotenv
load_dotenv()

# Configuration logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration Celery
REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
BROKER_URL = os.getenv('CELERY_BROKER_URL', REDIS_URL)
RESULT_BACKEND = os.getenv('CELERY_RESULT_BACKEND', REDIS_URL)

app = Celery('ml_orchestrator')

# Configuration avancée
app.conf.update(
    broker_url=BROKER_URL,
    result_backend=RESULT_BACKEND,
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='Europe/Paris',
    enable_utc=True,
    result_expires=3600,  # 1 heure
    
    # Configuration des queues
    task_routes={
        'ml_orchestrator.run_daily_scoring': {'queue': 'ml_scoring'},
        'ml_orchestrator.run_retrain_models': {'queue': 'ml_training'},
        'ml_orchestrator.run_health_check': {'queue': 'monitoring'},
        'ml_orchestrator.run_custom_scoring': {'queue': 'ml_scoring'},
    },
    
    # Définition des queues avec priorités
    task_queues=(
        Queue('ml_scoring', routing_key='ml_scoring'),
        Queue('ml_training', routing_key='ml_training'), 
        Queue('monitoring', routing_key='monitoring'),
        Queue('urgent', routing_key='urgent'),
    ),
    
    # Configuration avancée
    worker_send_task_events=True,
    task_send_sent_event=True,
    worker_hijack_root_logger=False,
    worker_log_format="[%(asctime)s: %(levelname)s/%(processName)s] %(message)s",
    worker_task_log_format="[%(asctime)s: %(levelname)s/%(processName)s][%(task_name)s(%(task_id)s)] %(message)s",
)

# Planification des tâches
app.conf.beat_schedule = {
    # Scoring quotidien à 2h du matin
    'daily-ml-scoring': {
        'task': 'ml_orchestrator.run_daily_scoring',
        'schedule': crontab(hour=2, minute=0),
        'options': {'queue': 'ml_scoring', 'retry': True, 'retry_policy': {
            'max_retries': 3,
            'interval_start': 0,
            'interval_step': 0.2,
            'interval_max': 0.2,
        }}
    },
    
    # Ré-entraînement hebdomadaire (dimanche 1h)
    'weekly-model-retrain': {
        'task': 'ml_orchestrator.run_retrain_models', 
        'schedule': crontab(hour=1, minute=0, day_of_week=0),
        'options': {'queue': 'ml_training', 'retry': True}
    },
    
    # Monitoring quotidien (8h)
    'daily-health-check': {
        'task': 'ml_orchestrator.run_health_check',
        'schedule': crontab(hour=8, minute=0),
        'options': {'queue': 'monitoring'}
    },
    
    # Nettoyage hebdomadaire (lundi 3h)
    'weekly-cleanup': {
        'task': 'ml_orchestrator.run_cleanup',
        'schedule': crontab(hour=3, minute=0, day_of_week=1),
        'options': {'queue': 'monitoring'}
    },
}

# ============================================================================
# TÂCHES PRINCIPALES
# ============================================================================

@app.task(bind=True, name='ml_orchestrator.run_daily_scoring')
def run_daily_scoring(self, batch_size: int = 1000):
    """Tâche de scoring quotidien"""
    logger.info(f"🌅 Démarrage scoring quotidien - Task ID: {self.request.id}")
    
    try:
        # Mise à jour du statut
        self.update_state(
            state='PROGRESS',
            meta={'status': 'Initialisation du scoring quotidien', 'progress': 0}
        )
        
        # Exécution du script scoring
        cmd = [
            sys.executable, 'scoring.py',
            '--batch-size', str(batch_size),
            '--log-level', 'INFO'
        ]
        
        logger.info(f"📋 Commande: {' '.join(cmd)}")
        
        # Lancement avec monitoring du progrès
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        
        # Monitoring du processus
        stdout_lines = []
        stderr_lines = []
        
        while True:
            stdout_line = process.stdout.readline()
            if stdout_line:
                stdout_lines.append(stdout_line.strip())
                logger.info(f"📊 {stdout_line.strip()}")
                
                # Mise à jour du progrès basé sur les logs
                if "entreprises traitées" in stdout_line:
                    try:
                        progress = min(80, len(stdout_lines) * 2)
                        self.update_state(
                            state='PROGRESS',
                            meta={'status': stdout_line.strip(), 'progress': progress}
                        )
                    except:
                        pass
            
            stderr_line = process.stderr.readline()
            if stderr_line:
                stderr_lines.append(stderr_line.strip())
                logger.error(f"❌ {stderr_line.strip()}")
            
            if process.poll() is not None:
                break
        
        # Récupération des dernières sorties
        remaining_stdout, remaining_stderr = process.communicate()
        if remaining_stdout:
            stdout_lines.extend(remaining_stdout.strip().split('\n'))
        if remaining_stderr:
            stderr_lines.extend(remaining_stderr.strip().split('\n'))
        
        # Vérification du résultat
        if process.returncode == 0:
            logger.info("✅ Scoring quotidien terminé avec succès")
            
            # Extraction des statistiques
            stats = extract_stats_from_output(stdout_lines)
            
            return {
                'status': 'SUCCESS',
                'message': 'Scoring quotidien terminé avec succès',
                'stats': stats,
                'processed_at': datetime.utcnow().isoformat(),
                'task_id': self.request.id
            }
        else:
            error_msg = '\n'.join(stderr_lines[-5:])  # Dernières 5 erreurs
            logger.error(f"❌ Scoring quotidien échoué: {error_msg}")
            raise Exception(f"Scoring échoué (code {process.returncode}): {error_msg}")
            
    except Exception as e:
        logger.error(f"❌ Erreur tâche scoring quotidien: {e}")
        self.update_state(
            state='FAILURE',
            meta={'status': f'Erreur: {str(e)}', 'error': str(e)}
        )
        
        # Notification d'erreur (Slack, email, etc.)
        send_error_notification.delay(
            task_name='Scoring Quotidien',
            error=str(e),
            task_id=self.request.id
        )
        raise

@app.task(bind=True, name='ml_orchestrator.run_retrain_models')
def run_retrain_models(self):
    """Tâche de ré-entraînement hebdomadaire des modèles"""
    logger.info(f"🧠 Démarrage ré-entraînement - Task ID: {self.request.id}")
    
    try:
        self.update_state(
            state='PROGRESS',
            meta={'status': 'Démarrage ré-entraînement des modèles', 'progress': 0}
        )
        
        # Sauvegarde des modèles actuels
        backup_result = backup_current_models.delay()
        backup_result.get(timeout=300)  # Attendre 5 minutes max
        
        # Ré-entraînement
        cmd = [
            sys.executable, 'scoring.py',
            '--force-retrain',
            '--batch-size', '500',
            '--log-level', 'INFO'
        ]
        
        process = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600,  # 1 heure max
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        
        if process.returncode == 0:
            logger.info("✅ Ré-entraînement terminé avec succès")
            
            # Validation des nouveaux modèles
            validation_result = validate_models.delay()
            validation_result.get(timeout=300)
            
            return {
                'status': 'SUCCESS',
                'message': 'Ré-entraînement terminé avec succès',
                'retrained_at': datetime.utcnow().isoformat(),
                'output': process.stdout[-1000:]  # Derniers 1000 caractères
            }
        else:
            raise Exception(f"Ré-entraînement échoué: {process.stderr}")
            
    except Exception as e:
        logger.error(f"❌ Erreur ré-entraînement: {e}")
        self.update_state(
            state='FAILURE', 
            meta={'status': f'Erreur: {str(e)}', 'error': str(e)}
        )
        
        send_error_notification.delay(
            task_name='Ré-entraînement Modèles',
            error=str(e),
            task_id=self.request.id
        )
        raise

@app.task(bind=True, name='ml_orchestrator.run_custom_scoring')
def run_custom_scoring(self, company_ids: List[int] = None, batch_size: int = 100, user_id: str = None):
    """Tâche de scoring personnalisé (déclenché manuellement)"""
    logger.info(f"🎯 Scoring personnalisé - Task ID: {self.request.id}, User: {user_id}")
    
    try:
        self.update_state(
            state='PROGRESS',
            meta={'status': 'Démarrage scoring personnalisé', 'progress': 10}
        )
        
        cmd = [sys.executable, 'scoring.py', '--batch-size', str(batch_size)]
        
        if company_ids:
            cmd.extend(['--company-ids', ','.join(map(str, company_ids))])
            logger.info(f"🏢 Scoring de {len(company_ids)} entreprises spécifiques")
        
        cmd.extend(['--log-level', 'INFO'])
        
        process = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1800,  # 30 minutes max
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        
        if process.returncode == 0:
            stats = extract_stats_from_output(process.stdout.split('\n'))
            
            # Notification de succès à l'utilisateur
            if user_id:
                send_success_notification.delay(
                    user_id=user_id,
                    task_type='Scoring Personnalisé',
                    stats=stats
                )
            
            return {
                'status': 'SUCCESS',
                'message': 'Scoring personnalisé terminé',
                'stats': stats,
                'user_id': user_id,
                'companies_processed': len(company_ids) if company_ids else 'all'
            }
        else:
            raise Exception(process.stderr)
            
    except Exception as e:
        logger.error(f"❌ Erreur scoring personnalisé: {e}")
        
        if user_id:
            send_error_notification.delay(
                task_name='Scoring Personnalisé',
                error=str(e),
                user_id=user_id
            )
        raise

@app.task(name='ml_orchestrator.run_health_check')
def run_health_check():
    """Vérification de santé quotidienne"""
    logger.info("🏥 Vérification de santé du système ML")
    
    try:
        from supabase import create_client
        
        # Connexion à la base
        supabase = create_client(
            os.getenv('SUPABASE_URL'),
            os.getenv('SUPABASE_KEY')
        )
        
        health_status = {
            'timestamp': datetime.utcnow().isoformat(),
            'database_connection': False,
            'recent_scores': False,
            'models_available': False,
            'issues': []
        }
        
        # Test connexion base
        try:
            result = supabase.table('companies').select('id').limit(1).execute()
            health_status['database_connection'] = True
        except Exception as e:
            health_status['issues'].append(f"Connexion base: {e}")
        
        # Vérification scores récents
        try:
            result = supabase.table('ml_scores')\
                .select('calculated_at')\
                .order('calculated_at', desc=True)\
                .limit(1)\
                .execute()
            
            if result.data:
                last_score_date = datetime.fromisoformat(
                    result.data[0]['calculated_at'].replace('Z', '+00:00')
                )
                hours_since = (datetime.now() - last_score_date.replace(tzinfo=None)).total_seconds() / 3600
                
                if hours_since < 48:  # Moins de 2 jours
                    health_status['recent_scores'] = True
                else:
                    health_status['issues'].append(f"Derniers scores: il y a {hours_since:.1f}h")
            else:
                health_status['issues'].append("Aucun score trouvé en base")
        except Exception as e:
            health_status['issues'].append(f"Vérification scores: {e}")
        
        # Vérification modèles
        model_files = ['xgb_ma_model.joblib', 'lgb_croissance_model.joblib', 'catboost_stabilite_model.joblib']
        available_models = [f for f in model_files if os.path.exists(f)]
        
        if len(available_models) >= 3:
            health_status['models_available'] = True
        else:
            health_status['issues'].append(f"Modèles manquants: {3 - len(available_models)}")
        
        # Notification si problèmes
        if health_status['issues']:
            send_health_alert.delay(health_status)
        
        logger.info(f"🏥 Santé système: {len(health_status['issues'])} problèmes détectés")
        return health_status
        
    except Exception as e:
        logger.error(f"❌ Erreur vérification santé: {e}")
        raise

# ============================================================================
# TÂCHES UTILITAIRES
# ============================================================================

@app.task(name='ml_orchestrator.backup_current_models')
def backup_current_models():
    """Sauvegarde les modèles actuels"""
    try:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_dir = 'model_backups'
        os.makedirs(backup_dir, exist_ok=True)
        
        model_files = [f for f in os.listdir('.') if f.endswith('.joblib')]
        
        if model_files:
            import tarfile
            backup_file = f"{backup_dir}/models_backup_{timestamp}.tar.gz"
            
            with tarfile.open(backup_file, 'w:gz') as tar:
                for file in model_files:
                    tar.add(file)
            
            logger.info(f"💾 Modèles sauvegardés: {backup_file}")
            return {'backup_file': backup_file, 'files_count': len(model_files)}
        else:
            logger.warning("⚠️ Aucun modèle à sauvegarder")
            return {'backup_file': None, 'files_count': 0}
            
    except Exception as e:
        logger.error(f"❌ Erreur sauvegarde modèles: {e}")
        raise

@app.task(name='ml_orchestrator.validate_models')
def validate_models():
    """Valide les modèles après ré-entraînement"""
    try:
        import joblib
        
        model_files = {
            'xgb_ma_model.joblib': 'XGBoost M&A',
            'lgb_croissance_model.joblib': 'LightGBM Croissance', 
            'catboost_stabilite_model.joblib': 'CatBoost Stabilité'
        }
        
        validation_results = {}
        
        for file, name in model_files.items():
            if os.path.exists(file):
                try:
                    model = joblib.load(file)
                    # Test basique de prédiction
                    import numpy as np
                    test_features = np.random.rand(1, 10)
                    prediction = model.predict(test_features)
                    
                    validation_results[name] = {
                        'loaded': True,
                        'prediction_test': True,
                        'file_size': os.path.getsize(file)
                    }
                    logger.info(f"✅ {name}: Validation OK")
                    
                except Exception as e:
                    validation_results[name] = {
                        'loaded': False,
                        'error': str(e)
                    }
                    logger.error(f"❌ {name}: {e}")
            else:
                validation_results[name] = {'loaded': False, 'error': 'Fichier non trouvé'}
        
        return validation_results
        
    except Exception as e:
        logger.error(f"❌ Erreur validation modèles: {e}")
        raise

@app.task(name='ml_orchestrator.run_cleanup')
def run_cleanup():
    """Nettoyage des fichiers temporaires et anciens logs"""
    try:
        import shutil
        from pathlib import Path
        
        cleanup_results = {'deleted_files': 0, 'freed_space': 0}
        
        # Nettoyage logs anciens (>30 jours)
        log_files = Path('.').glob('*.log')
        for log_file in log_files:
            if log_file.stat().st_mtime < (datetime.now() - timedelta(days=30)).timestamp():
                size = log_file.stat().st_size
                log_file.unlink()
                cleanup_results['deleted_files'] += 1
                cleanup_results['freed_space'] += size
        
        # Nettoyage backups anciens (>60 jours)
        if os.path.exists('model_backups'):
            backup_files = Path('model_backups').glob('*.tar.gz')
            for backup_file in backup_files:
                if backup_file.stat().st_mtime < (datetime.now() - timedelta(days=60)).timestamp():
                    size = backup_file.stat().st_size
                    backup_file.unlink()
                    cleanup_results['deleted_files'] += 1
                    cleanup_results['freed_space'] += size
        
        # Nettoyage cache Python
        cache_dirs = list(Path('.').rglob('__pycache__'))
        for cache_dir in cache_dirs:
            shutil.rmtree(cache_dir, ignore_errors=True)
            cleanup_results['deleted_files'] += 1
        
        logger.info(f"🧹 Nettoyage: {cleanup_results['deleted_files']} fichiers, "
                   f"{cleanup_results['freed_space']/1024/1024:.1f}MB libérés")
        
        return cleanup_results
        
    except Exception as e:
        logger.error(f"❌ Erreur nettoyage: {e}")
        raise

# ============================================================================
# TÂCHES DE NOTIFICATION
# ============================================================================

@app.task(name='ml_orchestrator.send_error_notification')
def send_error_notification(task_name: str, error: str, task_id: str = None, user_id: str = None):
    """Envoie une notification d'erreur"""
    try:
        # Notification Slack (exemple)
        slack_webhook = os.getenv('SLACK_WEBHOOK_URL')
        if slack_webhook:
            import requests
            
            message = {
                "text": f"🚨 Erreur ML Scoring",
                "attachments": [{
                    "color": "danger",
                    "fields": [
                        {"title": "Tâche", "value": task_name, "short": True},
                        {"title": "Erreur", "value": error[:500], "short": False},
                        {"title": "Task ID", "value": task_id or "N/A", "short": True},
                        {"title": "Timestamp", "value": datetime.now().strftime('%Y-%m-%d %H:%M:%S'), "short": True}
                    ]
                }]
            }
            
            requests.post(slack_webhook, json=message, timeout=10)
            logger.info("📬 Notification Slack envoyée")
        
        # Log local
        logger.error(f"🚨 NOTIFICATION ERREUR: {task_name} - {error}")
        
    except Exception as e:
        logger.error(f"❌ Erreur envoi notification: {e}")

@app.task(name='ml_orchestrator.send_success_notification')  
def send_success_notification(user_id: str, task_type: str, stats: dict):
    """Envoie une notification de succès"""
    try:
        # Notification email/Slack pour l'utilisateur
        logger.info(f"✅ {task_type} terminé avec succès pour utilisateur {user_id}")
        logger.info(f"📊 Stats: {stats}")
        
    except Exception as e:
        logger.error(f"❌ Erreur notification succès: {e}")

@app.task(name='ml_orchestrator.send_health_alert')
def send_health_alert(health_status: dict):
    """Envoie une alerte de santé système"""
    try:
        issues = health_status.get('issues', [])
        if issues:
            logger.warning(f"🏥 ALERTE SANTÉ: {len(issues)} problèmes détectés")
            for issue in issues:
                logger.warning(f"  - {issue}")
        
    except Exception as e:
        logger.error(f"❌ Erreur alerte santé: {e}")

# ============================================================================
# UTILITAIRES
# ============================================================================

def extract_stats_from_output(output_lines: List[str]) -> dict:
    """Extrait les statistiques depuis la sortie du script"""
    stats = {}
    
    for line in output_lines:
        if "entreprises traitées:" in line:
            try:
                stats['companies_processed'] = int(line.split(':')[1].strip())
            except:
                pass
        elif "Score Composite moyen:" in line:
            try:
                stats['avg_composite_score'] = float(line.split(':')[1].split('(')[0].strip())
            except:
                pass
        elif "Confiance moyenne:" in line:
            try:
                stats['avg_confidence'] = float(line.split(':')[1].replace('%', '').strip())
            except:
                pass
    
    return stats

# ============================================================================
# API D'INTERFACE
# ============================================================================

@app.task(name='ml_orchestrator.trigger_manual_scoring')
def trigger_manual_scoring(company_ids: List[int] = None, user_id: str = None):
    """Interface pour déclencher un scoring manuel"""
    return run_custom_scoring.delay(
        company_ids=company_ids,
        batch_size=len(company_ids) if company_ids else 1000,
        user_id=user_id
    )

@app.task(name='ml_orchestrator.get_task_status')
def get_task_status(task_id: str):
    """Récupère le statut d'une tâche"""
    result = app.AsyncResult(task_id)
    return {
        'task_id': task_id,
        'status': result.status,
        'result': result.result,
        'info': result.info,
        'traceback': result.traceback
    }

if __name__ == '__main__':
    app.start()