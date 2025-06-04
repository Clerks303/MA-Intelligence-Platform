"""
Shared Configuration Settings
M&A Intelligence Platform
"""

import os
from typing import List, Optional
from pydantic import BaseSettings, Field
from functools import lru_cache

class BaseConfig(BaseSettings):
    """Configuration de base partagée"""
    
    # Database
    SUPABASE_URL: str = Field(..., env="SUPABASE_URL")
    SUPABASE_KEY: str = Field(..., env="SUPABASE_KEY")
    
    # Redis
    REDIS_URL: str = Field("redis://localhost:6379/0", env="REDIS_URL")
    
    # Environment
    ENVIRONMENT: str = Field("development", env="ENVIRONMENT")
    DEBUG: bool = Field(False, env="DEBUG")
    
    class Config:
        env_file = ".env"
        case_sensitive = True

class APISettings(BaseConfig):
    """Configuration pour l'API Backend"""
    
    # FastAPI
    API_HOST: str = Field("0.0.0.0", env="API_HOST")
    API_PORT: int = Field(8000, env="API_PORT")
    
    # Security
    SECRET_KEY: str = Field(..., env="SECRET_KEY")
    ALGORITHM: str = Field("HS256", env="ALGORITHM")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    
    # CORS
    ALLOWED_ORIGINS: List[str] = Field(
        ["http://localhost:3000"], 
        env="ALLOWED_ORIGINS"
    )
    
    # External APIs
    OPENAI_API_KEY: Optional[str] = Field(None, env="OPENAI_API_KEY")
    PAPPERS_API_KEY: Optional[str] = Field(None, env="PAPPERS_API_KEY")

class MLSettings(BaseConfig):
    """Configuration pour le ML Service"""
    
    # ML Processing
    ML_BATCH_SIZE: int = Field(1000, env="ML_BATCH_SIZE")
    ML_MODEL_PATH: str = Field("/app/models", env="ML_MODEL_PATH")
    ML_CACHE_TTL: int = Field(3600, env="ML_CACHE_TTL")  # 1 heure
    
    # MLflow
    MLFLOW_TRACKING_URI: Optional[str] = Field(None, env="MLFLOW_TRACKING_URI")
    MLFLOW_EXPERIMENT_NAME: str = Field("ma_intelligence", env="MLFLOW_EXPERIMENT_NAME")
    
    # Feature Engineering
    FEATURE_STORE_PATH: str = Field("/app/features", env="FEATURE_STORE_PATH")
    MAX_MISSING_FEATURES: float = Field(0.3, env="MAX_MISSING_FEATURES")  # 30%
    
    # Model Performance
    MIN_MODEL_CONFIDENCE: float = Field(0.7, env="MIN_MODEL_CONFIDENCE")
    RETRAIN_THRESHOLD_DAYS: int = Field(30, env="RETRAIN_THRESHOLD_DAYS")

class SchedulerSettings(BaseConfig):
    """Configuration pour le Scheduler"""
    
    # Celery
    CELERY_BROKER_URL: str = Field("redis://localhost:6379/0", env="CELERY_BROKER_URL")
    CELERY_RESULT_BACKEND: str = Field("redis://localhost:6379/0", env="CELERY_RESULT_BACKEND")
    
    # Scheduling
    ML_SCHEDULE_HOUR: int = Field(2, env="ML_SCHEDULE_HOUR")  # 2h du matin
    ML_SCHEDULE_MINUTE: int = Field(0, env="ML_SCHEDULE_MINUTE")
    
    # Queue Configuration
    ML_SCORING_QUEUE: str = Field("ml_scoring", env="ML_SCORING_QUEUE")
    ML_TRAINING_QUEUE: str = Field("ml_training", env="ML_TRAINING_QUEUE")
    DATA_QUALITY_QUEUE: str = Field("data_quality", env="DATA_QUALITY_QUEUE")
    
    # Monitoring
    TASK_TIMEOUT: int = Field(3600, env="TASK_TIMEOUT")  # 1 heure
    MAX_RETRIES: int = Field(3, env="MAX_RETRIES")

# Fonctions d'accès aux configurations
@lru_cache()
def get_api_settings() -> APISettings:
    """Récupère la configuration API"""
    return APISettings()

@lru_cache()
def get_ml_settings() -> MLSettings:
    """Récupère la configuration ML"""
    return MLSettings()

@lru_cache()
def get_scheduler_settings() -> SchedulerSettings:
    """Récupère la configuration Scheduler"""
    return SchedulerSettings()

# Configuration commune pour les logs
def setup_logging(service_name: str = "ma_intelligence"):
    """Configure le logging pour un service"""
    import logging
    import structlog
    
    # Configuration structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Configuration logging standard
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=logging.INFO,
    )
    
    # Logger pour le service
    logger = structlog.get_logger(service_name)
    return logger

# Validation de configuration
def validate_configuration():
    """Valide que toutes les configurations nécessaires sont présentes"""
    required_env_vars = [
        "SUPABASE_URL",
        "SUPABASE_KEY"
    ]
    
    missing_vars = []
    for var in required_env_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        raise ValueError(f"Variables d'environnement manquantes: {', '.join(missing_vars)}")
    
    return True