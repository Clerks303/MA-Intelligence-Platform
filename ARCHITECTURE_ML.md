# Architecture Modulaire ML - M&A Intelligence Platform

## 🏗️ Vue d'ensemble

Cette architecture sépare clairement les responsabilités entre l'API backend et le calcul des scores ML, permettant une scalabilité et une maintenabilité optimales.

```
┌─────────────────────────────────────────────────────────────────┐
│                    M&A Intelligence Platform                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────┐    ┌──────────────────┐    ┌───────────┐  │
│  │   🌐 Frontend    │◄───┤   🚀 API Backend │◄───┤💾 Supabase│   │
│  │   (React)        │    │   (FastAPI)      │    │           │  │
│  └──────────────────┘    └──────────────────┘    └───────────┘  │
│                                   │                   ▲         │
│                                   │                   │         │
│                          ┌────────▼────────┐          │         │
│                          │  ⏰ Scheduler   │          │         │
│                          │  (Celery/Cron)  │          │         │
│                          └────────┬────────┘          │         │
│                                   │                   │         │
│                                   ▼                   │         │
│                          ┌────────────────────┐       │         │
│                          │  🤖 ML Service     │───────┘         │
│                          │  (LightGBM/XGBoost)│                 │
│                          └────────────────────┘                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 🚀 Composants Principaux

### 1. 🌐 API Backend (FastAPI)
**Responsabilités :**
- Authentification et autorisation
- CRUD des entreprises
- Scraping de données
- **Lecture des scores pré-calculés**
- Interface REST/GraphQL

**Technologies :**
- FastAPI + Uvicorn
- SQLAlchemy + Alembic
- Redis (cache)
- Supabase client

### 2. 🤖 ML Scoring Service
**Responsabilités :**
- Entraînement des modèles ML
- Calcul des scores M&A
- Feature engineering
- **Écriture des scores dans la base**

**Technologies :**
- LightGBM, XGBoost, CatBoost
- scikit-learn, pandas
- MLflow (model registry)
- Supabase client (écriture)

### 3. ⏰ Scheduler Service
**Responsabilités :**
- Déclenchement manuel des calculs
- Planification nocturne automatique
- Monitoring des tâches ML
- Gestion des queues

**Technologies :**
- Celery + Redis
- Cron jobs
- APScheduler

### 4. 💾 Base de Données (Supabase)
**Tables principales :**
```sql
-- Table principale des entreprises
companies (
  id, siren, nom_entreprise, 
  chiffre_affaires, secteur, ...
)

-- Table des scores ML (nouvelle)
ml_scores (
  id, company_id, 
  score_ma, score_croissance, score_stabilite,
  model_version, calculated_at, confidence
)

-- Table des modèles ML (nouvelle)
ml_models (
  id, name, version, 
  model_file_path, metrics, created_at
)
```

## 📁 Structure de Fichiers Proposée

```
backend/
├── api/                          # 🌐 API Backend (actuel)
│   ├── app/
│   │   ├── api/routes/
│   │   ├── core/
│   │   ├── models/
│   │   └── services/
│   ├── requirements-api.txt      # Dépendances API uniquement
│   └── main.py
│
├── ml-service/                   # 🤖 ML Service (nouveau)
│   ├── app/
│   │   ├── models/              # Modèles ML
│   │   ├── features/            # Feature engineering
│   │   ├── training/            # Scripts d'entraînement
│   │   ├── scoring/             # Calcul des scores
│   │   └── utils/
│   ├── requirements-ml.txt      # Dépendances ML uniquement
│   ├── batch_scoring.py         # Script principal
│   └── train_models.py          # Entraînement
│
├── scheduler/                   # ⏰ Scheduler (nouveau)
│   ├── celery_app.py
│   ├── tasks.py
│   ├── cron_jobs.sh
│   └── requirements-scheduler.txt
│
├── shared/                      # 📦 Code partagé
│   ├── database/
│   │   ├── models.py           # Modèles SQLAlchemy partagés
│   │   └── supabase_client.py  # Client Supabase
│   ├── config/
│   │   └── settings.py         # Configuration partagée
│   └── utils/
│
└── docker-compose.yml           # Orchestration complète
```

## 🔄 Flux de Données

### Flux Principal (Lecture des Scores)
```
1. 👤 Utilisateur → API Backend
2. 🚀 API Backend → Supabase (SELECT ml_scores)
3. 💾 Supabase → API Backend (scores + metadata)
4. 🚀 API Backend → Frontend (JSON avec scores)
```

### Flux ML (Calcul des Scores)
```
1. ⏰ Scheduler → ML Service (trigger)
2. 🤖 ML Service → Supabase (SELECT companies)
3. 🤖 ML Service → Feature Engineering
4. 🤖 ML Service → Modèles ML (LightGBM/XGBoost)
5. 🤖 ML Service → Supabase (INSERT/UPDATE ml_scores)
```

## 🛠️ Implémentation Technique

### 1. Modification de l'API Backend

**Nouveau service de scores :**
```python
# app/services/scoring_service.py
class ScoringService:
    async def get_company_scores(self, company_id: int):
        """Récupère les scores pré-calculés"""
        return await self.db.fetch_one(
            "SELECT * FROM ml_scores WHERE company_id = ?", 
            company_id
        )
    
    async def get_scores_batch(self, company_ids: List[int]):
        """Récupère les scores pour plusieurs entreprises"""
        pass
```

**Route API modifiée :**
```python
# app/api/routes/companies.py
@router.get("/{company_id}/scores")
async def get_company_scores(company_id: int):
    scores = await scoring_service.get_company_scores(company_id)
    return {
        "company_id": company_id,
        "scores": scores,
        "last_calculated": scores.calculated_at,
        "model_version": scores.model_version
    }
```

### 2. ML Scoring Service

**Script principal :**
```python
# ml-service/batch_scoring.py
class MLScoringService:
    def __init__(self):
        self.models = self.load_models()
        self.supabase = create_supabase_client()
    
    async def calculate_scores_batch(self, batch_size=1000):
        """Calcule les scores pour toutes les entreprises"""
        companies = await self.get_companies_to_score()
        
        for batch in self.chunk(companies, batch_size):
            features = self.engineer_features(batch)
            scores = self.predict_scores(features)
            await self.save_scores(scores)
    
    def predict_scores(self, features):
        """Applique les modèles ML"""
        return {
            'score_ma': self.models['lightgbm'].predict(features),
            'score_croissance': self.models['xgboost'].predict(features),
            'score_stabilite': self.models['catboost'].predict(features)
        }
```

### 3. Scheduler avec Celery

**Configuration Celery :**
```python
# scheduler/celery_app.py
from celery import Celery
from celery.schedules import crontab

app = Celery('ml_scheduler')

app.conf.beat_schedule = {
    'nightly-ml-scoring': {
        'task': 'tasks.run_ml_scoring',
        'schedule': crontab(hour=2, minute=0),  # 2h du matin
    },
}

# scheduler/tasks.py
@app.task
def run_ml_scoring():
    """Tâche Celery pour lancer le scoring ML"""
    import subprocess
    result = subprocess.run([
        'python', '/app/ml-service/batch_scoring.py'
    ])
    return result.returncode
```

## 🐳 Déploiement Docker

```yaml
# docker-compose.yml
version: '3.8'
services:
  api-backend:
    build: ./api
    ports: ["8000:8000"]
    environment:
      - SUPABASE_URL=${SUPABASE_URL}
    depends_on: [redis]
  
  ml-service:
    build: ./ml-service
    environment:
      - SUPABASE_URL=${SUPABASE_URL}
    volumes:
      - ./models:/app/models
  
  scheduler:
    build: ./scheduler
    depends_on: [redis, ml-service]
  
  redis:
    image: redis:7-alpine
```

## 📊 Gestion des Modèles ML

### MLflow Integration
```python
# ml-service/app/models/model_manager.py
import mlflow

class ModelManager:
    def load_production_models(self):
        """Charge les modèles en production depuis MLflow"""
        models = {}
        for model_name in ['lightgbm_ma', 'xgboost_growth']:
            model_uri = f"models:/{model_name}/Production"
            models[model_name] = mlflow.pyfunc.load_model(model_uri)
        return models
    
    def promote_model(self, model_name: str, version: str):
        """Promeut un modèle en production"""
        client = mlflow.MlflowClient()
        client.transition_model_version_stage(
            name=model_name, version=version, stage="Production"
        )
```

## 🔄 API de Déclenchement Manuel

```python
# api/app/api/routes/ml.py
@router.post("/ml/trigger-scoring")
async def trigger_ml_scoring(
    current_user: User = Depends(get_admin_user)
):
    """Déclenche le calcul ML manuellement"""
    task = run_ml_scoring.delay()
    return {
        "task_id": task.id,
        "status": "started",
        "message": "ML scoring task initiated"
    }

@router.get("/ml/scoring-status/{task_id}")
async def get_scoring_status(task_id: str):
    """Vérifie le statut d'une tâche ML"""
    task = run_ml_scoring.AsyncResult(task_id)
    return {
        "task_id": task_id,
        "status": task.status,
        "result": task.result
    }
```

## 📈 Monitoring et Observabilité

### Métriques ML
```python
# ml-service/app/monitoring/metrics.py
from prometheus_client import Counter, Histogram

ml_predictions_total = Counter(
    'ml_predictions_total', 
    'Total ML predictions made'
)

ml_prediction_duration = Histogram(
    'ml_prediction_duration_seconds',
    'Time spent on ML predictions'
)
```

## 🔧 Configuration par Environnement

```python
# shared/config/settings.py
class Settings:
    # API Backend
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    
    # ML Service
    ML_BATCH_SIZE: int = 1000
    ML_MODEL_PATH: str = "/app/models"
    
    # Scheduler
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    ML_SCHEDULE_HOUR: int = 2  # 2h du matin
    
    # Database
    SUPABASE_URL: str
    SUPABASE_KEY: str
```

## 🚀 Avantages de cette Architecture

1. **🔄 Séparation des préoccupations** - API rapide, ML optimisé
2. **📈 Scalabilité** - Services indépendants scalables
3. **🛠️ Maintenabilité** - Code modulaire et testable
4. **⚡ Performance** - API sans latence ML
5. **🔧 Flexibilité** - Déploiement et mise à jour indépendants
6. **📊 Monitoring** - Observabilité granulaire par service

Cette architecture permet de démarrer simple et d'évoluer vers une solution microservices complète selon les besoins de scaling.