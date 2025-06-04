"""
Celery App Configuration - ML Scheduler
M&A Intelligence Platform
"""

import os
from celery import Celery
from celery.schedules import crontab
from kombu import Queue

# Configuration Redis
redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0')

# Configuration Celery
app = Celery('ml_scheduler')
app.conf.update(
    broker_url=redis_url,
    result_backend=redis_url,
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='Europe/Paris',
    enable_utc=True,
    result_expires=3600,  # 1 heure
    
    # Configuration des queues
    task_routes={
        'tasks.run_ml_scoring': {'queue': 'ml_scoring'},
        'tasks.train_models': {'queue': 'ml_training'},
        'tasks.data_quality_check': {'queue': 'data_quality'},
    },
    
    # Définition des queues
    task_queues=(
        Queue('ml_scoring', routing_key='ml_scoring'),
        Queue('ml_training', routing_key='ml_training'),
        Queue('data_quality', routing_key='data_quality'),
        Queue('default', routing_key='default'),
    ),
    task_default_queue='default',
    task_default_exchange='default',
    task_default_routing_key='default',
)

# Planification des tâches automatiques
app.conf.beat_schedule = {
    # Scoring ML nocturne quotidien
    'nightly-ml-scoring': {
        'task': 'tasks.run_ml_scoring',
        'schedule': crontab(hour=2, minute=0),  # Tous les jours à 2h
        'options': {'queue': 'ml_scoring'}
    },
    
    # Vérification qualité des données (hebdomadaire)
    'weekly-data-quality-check': {
        'task': 'tasks.data_quality_check',
        'schedule': crontab(hour=1, minute=0, day_of_week=1),  # Lundi 1h
        'options': {'queue': 'data_quality'}
    },
    
    # Ré-entraînement mensuel des modèles
    'monthly-model-retraining': {
        'task': 'tasks.train_models',
        'schedule': crontab(hour=3, minute=0, day_of_month=1),  # 1er du mois à 3h
        'options': {'queue': 'ml_training'}
    },
}

# Configuration du monitoring
app.conf.worker_send_task_events = True
app.conf.task_send_sent_event = True

if __name__ == '__main__':
    app.start()