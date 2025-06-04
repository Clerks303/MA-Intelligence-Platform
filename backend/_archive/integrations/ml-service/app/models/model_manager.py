"""
Model Manager - Gestion des modèles ML
M&A Intelligence Platform
"""

import pickle
import joblib
import logging
from pathlib import Path
from typing import Dict, Any
import mlflow
import mlflow.lightgbm
import mlflow.xgboost
import mlflow.catboost

logger = logging.getLogger(__name__)

class ModelManager:
    """Gestionnaire des modèles ML"""
    
    def __init__(self, model_path: str = "/app/models"):
        self.model_path = Path(model_path)
        self.model_path.mkdir(exist_ok=True, parents=True)
        self.models = {}
        
    async def load_production_models(self) -> Dict[str, Any]:
        """
        Charge les modèles ML en production
        Priorité: MLflow -> Fichiers locaux
        """
        logger.info("📦 Chargement des modèles ML...")
        
        models = {}
        
        try:
            # Tentative de chargement depuis MLflow
            models.update(await self._load_from_mlflow())
            logger.info("✅ Modèles chargés depuis MLflow")
        except Exception as e:
            logger.warning(f"⚠️ MLflow indisponible: {e}")
            logger.info("📁 Chargement depuis les fichiers locaux...")
            models.update(await self._load_from_files())
        
        self.models = models
        logger.info(f"🎯 {len(models)} modèles chargés: {list(models.keys())}")
        return models
    
    async def _load_from_mlflow(self) -> Dict[str, Any]:
        """Charge les modèles depuis MLflow Registry"""
        models = {}
        
        model_configs = [
            {"name": "lightgbm_ma_score", "alias": "score_ma"},
            {"name": "xgboost_growth", "alias": "score_croissance"}, 
            {"name": "catboost_stability", "alias": "score_stabilite"}
        ]
        
        for config in model_configs:
            try:
                model_uri = f"models:/{config['name']}/Production"
                model = mlflow.pyfunc.load_model(model_uri)
                models[config['alias']] = model
                logger.info(f"✅ {config['alias']} chargé depuis MLflow")
            except Exception as e:
                logger.warning(f"❌ Erreur chargement {config['alias']}: {e}")
        
        return models
    
    async def _load_from_files(self) -> Dict[str, Any]:
        """Charge les modèles depuis les fichiers locaux"""
        models = {}
        
        model_files = {
            "score_ma": "lightgbm_ma_model.pkl",
            "score_croissance": "xgboost_growth_model.pkl",
            "score_stabilite": "catboost_stability_model.pkl"
        }
        
        for alias, filename in model_files.items():
            file_path = self.model_path / filename
            
            if file_path.exists():
                try:
                    with open(file_path, 'rb') as f:
                        model = pickle.load(f)
                    models[alias] = model
                    logger.info(f"✅ {alias} chargé depuis {filename}")
                except Exception as e:
                    logger.error(f"❌ Erreur chargement {alias}: {e}")
            else:
                logger.warning(f"⚠️ Fichier {filename} introuvable")
        
        # Si aucun modèle trouvé, créer des modèles par défaut
        if not models:
            logger.warning("🔄 Création de modèles par défaut...")
            models = await self._create_default_models()
        
        return models
    
    async def _create_default_models(self) -> Dict[str, Any]:
        """Crée des modèles par défaut pour le développement"""
        from lightgbm import LGBMRegressor
        from xgboost import XGBRegressor
        from catboost import CatBoostRegressor
        
        models = {
            "score_ma": LGBMRegressor(
                n_estimators=100,
                random_state=42,
                verbose=-1
            ),
            "score_croissance": XGBRegressor(
                n_estimators=100,
                random_state=42,
                verbosity=0
            ),
            "score_stabilite": CatBoostRegressor(
                iterations=100,
                random_state=42,
                verbose=False
            )
        }
        
        logger.info("🔧 Modèles par défaut créés (non entraînés)")
        return models
    
    async def save_model(self, model: Any, name: str, version: str = None):
        """Sauvegarde un modèle localement et dans MLflow"""
        
        # Sauvegarde locale
        filename = f"{name}_model.pkl"
        file_path = self.model_path / filename
        
        with open(file_path, 'wb') as f:
            pickle.dump(model, f)
        
        logger.info(f"💾 Modèle {name} sauvegardé: {file_path}")
        
        # Sauvegarde MLflow (optionnelle)
        try:
            with mlflow.start_run():
                mlflow.sklearn.log_model(model, name)
                if version:
                    mlflow.set_tag("version", version)
            logger.info(f"☁️ Modèle {name} sauvegardé dans MLflow")
        except Exception as e:
            logger.warning(f"⚠️ Échec sauvegarde MLflow: {e}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Retourne les informations sur les modèles chargés"""
        info = {}
        
        for name, model in self.models.items():
            info[name] = {
                "type": type(model).__name__,
                "loaded": True,
                "path": str(self.model_path)
            }
            
            # Informations spécifiques selon le type
            if hasattr(model, 'n_estimators'):
                info[name]["n_estimators"] = model.n_estimators
            if hasattr(model, 'feature_importances_'):
                info[name]["n_features"] = len(model.feature_importances_)
        
        return info