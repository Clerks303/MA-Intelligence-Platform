"""
Système d'apprentissage continu et adaptation des modèles ML
US-010: Apprentissage automatique adaptatif pour M&A Intelligence Platform

Ce module fournit:
- Apprentissage continu des modèles ML
- Adaptation automatique aux nouvelles données
- Détection de dérive des données (data drift)
- Re-entraînement automatique des modèles
- Versioning et rollback des modèles
- A/B testing des modèles
- Monitoring de performance en temps réel
"""

import asyncio
import numpy as np
import pandas as pd
import pickle
import joblib
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from functools import wraps
from enum import Enum
import json
import time
import uuid
from pathlib import Path
import shutil
from collections import defaultdict, deque
import threading
import schedule

# Machine Learning
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import scipy.stats as stats

# Model versioning
import mlflow
import mlflow.sklearn

# Advanced statistics
from scipy.stats import ks_2samp, chi2_contingency
from scipy.spatial.distance import wasserstein_distance

from app.core.logging_system import get_logger, LogCategory
from app.core.cache_manager import get_cache_manager, cached
from app.core.advanced_ai_engine import get_advanced_ai_engine

logger = get_logger("continuous_learning", LogCategory.AI_ML)


class DriftType(str, Enum):
    """Types de dérive de données"""
    FEATURE_DRIFT = "feature_drift"
    LABEL_DRIFT = "label_drift"
    CONCEPT_DRIFT = "concept_drift"
    PREDICTION_DRIFT = "prediction_drift"


class ModelStatus(str, Enum):
    """Statuts des modèles"""
    TRAINING = "training"
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    FAILED = "failed"
    TESTING = "testing"


class LearningStrategy(str, Enum):
    """Stratégies d'apprentissage"""
    INCREMENTAL = "incremental"
    BATCH = "batch"
    ONLINE = "online"
    ENSEMBLE = "ensemble"


class TriggerType(str, Enum):
    """Types de déclencheurs pour re-entraînement"""
    SCHEDULED = "scheduled"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    DATA_DRIFT = "data_drift"
    MANUAL = "manual"
    DATA_VOLUME = "data_volume"


@dataclass
class DataDrift:
    """Détection de dérive des données"""
    drift_id: str
    drift_type: DriftType
    feature_name: Optional[str]
    
    # Métriques de dérive
    drift_score: float  # 0-1
    statistical_test: str
    p_value: float
    threshold: float
    
    # Données de référence vs nouvelles
    reference_stats: Dict[str, float]
    current_stats: Dict[str, float]
    
    # Recommandations
    severity: str  # low, medium, high, critical
    recommendation: str
    
    detected_at: datetime = field(default_factory=datetime.now)


@dataclass
class ModelPerformance:
    """Performance d'un modèle"""
    model_id: str
    model_version: str
    
    # Métriques de performance
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    
    # Métriques spécialisées
    business_metrics: Dict[str, float]
    
    # Contexte
    evaluation_date: datetime
    data_size: int
    feature_count: int
    
    # Comparaison avec baseline
    performance_change: Dict[str, float]


@dataclass
class ModelVersion:
    """Version d'un modèle"""
    version_id: str
    model_name: str
    version_number: str
    
    # Modèle et métadonnées
    model_artifact: Any
    model_parameters: Dict[str, Any]
    training_data_hash: str
    
    # Performance
    performance: ModelPerformance
    validation_results: Dict[str, Any]
    
    # Statut et lifecycle
    status: ModelStatus
    created_at: datetime
    deployed_at: Optional[datetime] = None
    deprecated_at: Optional[datetime] = None
    
    # Tracking
    training_duration: float
    data_size: int
    feature_importance: Dict[str, float] = field(default_factory=dict)


@dataclass
class LearningJob:
    """Job d'apprentissage"""
    job_id: str
    model_name: str
    strategy: LearningStrategy
    trigger_type: TriggerType
    
    # Configuration
    training_config: Dict[str, Any]
    data_config: Dict[str, Any]
    
    # Statut
    status: str  # pending, running, completed, failed
    progress: float  # 0-100
    
    # Résultats
    created_model_version: Optional[str] = None
    performance_improvement: Optional[float] = None
    error_message: Optional[str] = None
    
    # Timing
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    created_at: datetime = field(default_factory=datetime.now)


class DriftDetector:
    """Détecteur de dérive de données"""
    
    def __init__(self):
        self.reference_data: Dict[str, pd.DataFrame] = {}
        self.drift_thresholds: Dict[str, float] = {
            'feature_drift': 0.05,
            'label_drift': 0.05,
            'prediction_drift': 0.1
        }
        
    async def set_reference_data(self, model_name: str, data: pd.DataFrame):
        """Définit les données de référence pour un modèle"""
        self.reference_data[model_name] = data.copy()
        logger.info(f"📊 Données de référence définies pour {model_name}: {len(data)} échantillons")
    
    async def detect_drift(self, model_name: str, new_data: pd.DataFrame) -> List[DataDrift]:
        """Détecte la dérive dans les nouvelles données"""
        
        if model_name not in self.reference_data:
            logger.warning(f"Pas de données de référence pour {model_name}")
            return []
        
        reference = self.reference_data[model_name]
        detected_drifts = []
        
        logger.info(f"🔍 Détection de dérive pour {model_name}")
        
        # Dérive des features
        feature_drifts = await self._detect_feature_drift(reference, new_data)
        detected_drifts.extend(feature_drifts)
        
        # Dérive des labels (si disponibles)
        if 'target' in reference.columns and 'target' in new_data.columns:
            label_drift = await self._detect_label_drift(
                reference['target'], new_data['target']
            )
            if label_drift:
                detected_drifts.append(label_drift)
        
        # Log résultats
        if detected_drifts:
            logger.warning(f"⚠️ {len(detected_drifts)} dérives détectées pour {model_name}")
        else:
            logger.info(f"✅ Aucune dérive détectée pour {model_name}")
        
        return detected_drifts
    
    async def _detect_feature_drift(
        self, 
        reference: pd.DataFrame, 
        current: pd.DataFrame
    ) -> List[DataDrift]:
        """Détecte la dérive des features"""
        
        drifts = []
        common_features = set(reference.columns) & set(current.columns)
        
        for feature in common_features:
            if feature == 'target':
                continue
                
            ref_values = reference[feature].dropna()
            cur_values = current[feature].dropna()
            
            if len(ref_values) == 0 or len(cur_values) == 0:
                continue
            
            # Test statistique selon le type de données
            if pd.api.types.is_numeric_dtype(ref_values):
                drift = await self._detect_numeric_drift(
                    feature, ref_values, cur_values
                )
            else:
                drift = await self._detect_categorical_drift(
                    feature, ref_values, cur_values
                )
            
            if drift:
                drifts.append(drift)
        
        return drifts
    
    async def _detect_numeric_drift(
        self, 
        feature_name: str, 
        reference: pd.Series, 
        current: pd.Series
    ) -> Optional[DataDrift]:
        """Détecte la dérive pour features numériques"""
        
        # Test de Kolmogorov-Smirnov
        ks_stat, p_value = ks_2samp(reference, current)
        
        threshold = self.drift_thresholds['feature_drift']
        drift_detected = p_value < threshold
        
        if drift_detected:
            # Statistiques de référence vs actuelles
            ref_stats = {
                'mean': float(reference.mean()),
                'std': float(reference.std()),
                'median': float(reference.median()),
                'min': float(reference.min()),
                'max': float(reference.max())
            }
            
            cur_stats = {
                'mean': float(current.mean()),
                'std': float(current.std()),
                'median': float(current.median()),
                'min': float(current.min()),
                'max': float(current.max())
            }
            
            # Sévérité basée sur l'ampleur du changement
            mean_change = abs(cur_stats['mean'] - ref_stats['mean']) / (ref_stats['std'] + 1e-6)
            
            if mean_change > 2:
                severity = "critical"
            elif mean_change > 1:
                severity = "high"
            elif mean_change > 0.5:
                severity = "medium"
            else:
                severity = "low"
            
            recommendation = f"Dérive détectée sur {feature_name}. "
            if severity in ["critical", "high"]:
                recommendation += "Re-entraînement recommandé."
            else:
                recommendation += "Surveillance continue recommandée."
            
            return DataDrift(
                drift_id=f"drift_{feature_name}_{int(time.time())}",
                drift_type=DriftType.FEATURE_DRIFT,
                feature_name=feature_name,
                drift_score=ks_stat,
                statistical_test="Kolmogorov-Smirnov",
                p_value=p_value,
                threshold=threshold,
                reference_stats=ref_stats,
                current_stats=cur_stats,
                severity=severity,
                recommendation=recommendation
            )
        
        return None
    
    async def _detect_categorical_drift(
        self, 
        feature_name: str, 
        reference: pd.Series, 
        current: pd.Series
    ) -> Optional[DataDrift]:
        """Détecte la dérive pour features catégorielles"""
        
        # Distributions des catégories
        ref_dist = reference.value_counts(normalize=True).sort_index()
        cur_dist = current.value_counts(normalize=True).sort_index()
        
        # Aligner les index
        all_categories = set(ref_dist.index) | set(cur_dist.index)
        ref_aligned = ref_dist.reindex(all_categories, fill_value=0)
        cur_aligned = cur_dist.reindex(all_categories, fill_value=0)
        
        # Test du Chi-2
        try:
            # Créer tableau de contingence
            ref_counts = reference.value_counts()
            cur_counts = current.value_counts()
            
            # Aligner
            all_cats = set(ref_counts.index) | set(cur_counts.index)
            ref_aligned_counts = ref_counts.reindex(all_cats, fill_value=0)
            cur_aligned_counts = cur_counts.reindex(all_cats, fill_value=0)
            
            contingency_table = np.array([ref_aligned_counts.values, cur_aligned_counts.values])
            
            chi2, p_value, _, _ = chi2_contingency(contingency_table)
            
            threshold = self.drift_thresholds['feature_drift']
            drift_detected = p_value < threshold
            
            if drift_detected:
                # Distance de Wasserstein entre distributions
                drift_score = wasserstein_distance(
                    range(len(ref_aligned)), range(len(cur_aligned)),
                    ref_aligned.values, cur_aligned.values
                )
                
                return DataDrift(
                    drift_id=f"drift_{feature_name}_{int(time.time())}",
                    drift_type=DriftType.FEATURE_DRIFT,
                    feature_name=feature_name,
                    drift_score=drift_score,
                    statistical_test="Chi-square",
                    p_value=p_value,
                    threshold=threshold,
                    reference_stats=ref_aligned.to_dict(),
                    current_stats=cur_aligned.to_dict(),
                    severity="medium" if drift_score > 0.1 else "low",
                    recommendation=f"Distribution de {feature_name} a changé. Vérifier impact sur modèle."
                )
                
        except Exception as e:
            logger.warning(f"Erreur détection dérive catégorielle {feature_name}: {e}")
        
        return None
    
    async def _detect_label_drift(
        self, 
        reference_labels: pd.Series, 
        current_labels: pd.Series
    ) -> Optional[DataDrift]:
        """Détecte la dérive des labels"""
        
        # Distribution des labels
        ref_dist = reference_labels.value_counts(normalize=True).sort_index()
        cur_dist = current_labels.value_counts(normalize=True).sort_index()
        
        # Test KS pour labels numériques, Chi-2 pour catégoriels
        if pd.api.types.is_numeric_dtype(reference_labels):
            ks_stat, p_value = ks_2samp(reference_labels, current_labels)
            test_name = "Kolmogorov-Smirnov"
            drift_score = ks_stat
        else:
            # Chi-2 test
            try:
                all_labels = set(ref_dist.index) | set(cur_dist.index)
                ref_counts = reference_labels.value_counts().reindex(all_labels, fill_value=0)
                cur_counts = current_labels.value_counts().reindex(all_labels, fill_value=0)
                
                contingency = np.array([ref_counts.values, cur_counts.values])
                chi2, p_value, _, _ = chi2_contingency(contingency)
                test_name = "Chi-square"
                drift_score = chi2 / (len(reference_labels) + len(current_labels))
            except:
                return None
        
        threshold = self.drift_thresholds['label_drift']
        
        if p_value < threshold:
            return DataDrift(
                drift_id=f"drift_labels_{int(time.time())}",
                drift_type=DriftType.LABEL_DRIFT,
                feature_name=None,
                drift_score=drift_score,
                statistical_test=test_name,
                p_value=p_value,
                threshold=threshold,
                reference_stats=ref_dist.to_dict(),
                current_stats=cur_dist.to_dict(),
                severity="high",  # Label drift toujours sérieux
                recommendation="Dérive des labels détectée. Re-entraînement nécessaire."
            )
        
        return None


class ModelVersionManager:
    """Gestionnaire de versions de modèles"""
    
    def __init__(self, models_directory: str = "./models"):
        self.models_dir = Path(models_directory)
        self.models_dir.mkdir(exist_ok=True)
        
        self.model_versions: Dict[str, List[ModelVersion]] = defaultdict(list)
        self.active_models: Dict[str, str] = {}  # model_name -> version_id
        
    async def save_model_version(
        self, 
        model_name: str, 
        model: Any, 
        performance: ModelPerformance,
        training_config: Dict[str, Any] = None
    ) -> str:
        """Sauvegarde une nouvelle version de modèle"""
        
        version_id = f"{model_name}_v{int(time.time())}"
        version_number = f"v{len(self.model_versions[model_name]) + 1}"
        
        # Sauvegarde du modèle
        model_path = self.models_dir / f"{version_id}.pkl"
        
        try:
            if hasattr(model, '__dict__'):
                joblib.dump(model, model_path)
            else:
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
            
            logger.info(f"💾 Modèle sauvegardé: {model_path}")
            
        except Exception as e:
            logger.error(f"❌ Erreur sauvegarde modèle: {e}")
            raise
        
        # Calcul hash données d'entraînement (simplifié)
        training_data_hash = str(hash(str(training_config or {})))
        
        # Création version
        model_version = ModelVersion(
            version_id=version_id,
            model_name=model_name,
            version_number=version_number,
            model_artifact=model_path,
            model_parameters=training_config or {},
            training_data_hash=training_data_hash,
            performance=performance,
            validation_results={},
            status=ModelStatus.TRAINING,
            created_at=datetime.now(),
            training_duration=0.0,
            data_size=performance.data_size,
            feature_importance={}
        )
        
        self.model_versions[model_name].append(model_version)
        
        logger.info(f"✅ Version {version_number} créée pour {model_name}")
        
        return version_id
    
    async def load_model_version(self, version_id: str) -> Any:
        """Charge une version spécifique de modèle"""
        
        model_path = self.models_dir / f"{version_id}.pkl"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Modèle {version_id} non trouvé")
        
        try:
            model = joblib.load(model_path)
            logger.info(f"📂 Modèle {version_id} chargé")
            return model
            
        except Exception as e:
            logger.error(f"❌ Erreur chargement modèle {version_id}: {e}")
            raise
    
    async def promote_to_active(self, model_name: str, version_id: str):
        """Promeut une version comme modèle actif"""
        
        # Vérifier que la version existe
        model_version = self._find_version(model_name, version_id)
        if not model_version:
            raise ValueError(f"Version {version_id} non trouvée pour {model_name}")
        
        # Déprécier l'ancien modèle actif
        if model_name in self.active_models:
            old_version_id = self.active_models[model_name]
            old_version = self._find_version(model_name, old_version_id)
            if old_version:
                old_version.status = ModelStatus.DEPRECATED
                old_version.deprecated_at = datetime.now()
        
        # Promouvoir nouveau modèle
        model_version.status = ModelStatus.ACTIVE
        model_version.deployed_at = datetime.now()
        self.active_models[model_name] = version_id
        
        logger.info(f"🚀 {version_id} promu comme modèle actif pour {model_name}")
    
    def _find_version(self, model_name: str, version_id: str) -> Optional[ModelVersion]:
        """Trouve une version spécifique"""
        for version in self.model_versions[model_name]:
            if version.version_id == version_id:
                return version
        return None
    
    async def get_active_model(self, model_name: str) -> Optional[Any]:
        """Récupère le modèle actif"""
        
        if model_name not in self.active_models:
            return None
        
        version_id = self.active_models[model_name]
        return await self.load_model_version(version_id)
    
    async def rollback_model(self, model_name: str, target_version_id: str = None):
        """Rollback vers une version précédente"""
        
        if target_version_id:
            target_version = self._find_version(model_name, target_version_id)
        else:
            # Rollback vers la version précédente active
            versions = sorted(
                [v for v in self.model_versions[model_name] if v.status == ModelStatus.DEPRECATED],
                key=lambda x: x.deployed_at or datetime.min,
                reverse=True
            )
            target_version = versions[0] if versions else None
        
        if target_version:
            await self.promote_to_active(model_name, target_version.version_id)
            logger.info(f"⏪ Rollback effectué vers {target_version.version_id}")
        else:
            raise ValueError("Aucune version cible pour rollback")
    
    def get_model_history(self, model_name: str) -> List[Dict[str, Any]]:
        """Récupère l'historique des versions d'un modèle"""
        
        history = []
        for version in sorted(self.model_versions[model_name], key=lambda x: x.created_at, reverse=True):
            history.append({
                'version_id': version.version_id,
                'version_number': version.version_number,
                'status': version.status.value,
                'performance': {
                    'accuracy': version.performance.accuracy,
                    'f1_score': version.performance.f1_score
                },
                'created_at': version.created_at.isoformat(),
                'deployed_at': version.deployed_at.isoformat() if version.deployed_at else None
            })
        
        return history


class ContinuousLearningEngine:
    """Moteur principal d'apprentissage continu"""
    
    def __init__(self):
        self.drift_detector = DriftDetector()
        self.version_manager = ModelVersionManager()
        self.learning_jobs: Dict[str, LearningJob] = {}
        
        # Configuration
        self.monitoring_enabled = True
        self.auto_retrain_enabled = True
        self.performance_threshold = 0.05  # 5% de dégradation
        
        # Métriques de monitoring
        self.performance_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.drift_history: Dict[str, List[DataDrift]] = defaultdict(list)
        
        logger.info("🔄 Moteur d'apprentissage continu initialisé")
    
    async def register_model(
        self, 
        model_name: str, 
        initial_model: Any, 
        reference_data: pd.DataFrame,
        performance: ModelPerformance
    ):
        """Enregistre un modèle pour apprentissage continu"""
        
        logger.info(f"📝 Enregistrement modèle: {model_name}")
        
        # Sauvegarder version initiale
        version_id = await self.version_manager.save_model_version(
            model_name, initial_model, performance
        )
        
        # Définir données de référence
        await self.drift_detector.set_reference_data(model_name, reference_data)
        
        # Promouvoir comme actif
        await self.version_manager.promote_to_active(model_name, version_id)
        
        logger.info(f"✅ Modèle {model_name} enregistré et actif")
    
    async def monitor_model_performance(
        self, 
        model_name: str, 
        new_data: pd.DataFrame,
        predictions: np.ndarray,
        true_labels: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Monitor la performance d'un modèle"""
        
        monitoring_result = {
            'model_name': model_name,
            'timestamp': datetime.now().isoformat(),
            'data_size': len(new_data),
            'drift_detected': False,
            'performance_degradation': False,
            'recommendations': []
        }
        
        # Détection de dérive
        drifts = await self.drift_detector.detect_drift(model_name, new_data)
        
        if drifts:
            monitoring_result['drift_detected'] = True
            monitoring_result['drifts'] = [
                {
                    'type': drift.drift_type.value,
                    'feature': drift.feature_name,
                    'severity': drift.severity,
                    'score': drift.drift_score,
                    'recommendation': drift.recommendation
                }
                for drift in drifts
            ]
            
            self.drift_history[model_name].extend(drifts)
            
            # Recommandations basées sur sévérité
            critical_drifts = [d for d in drifts if d.severity == "critical"]
            high_drifts = [d for d in drifts if d.severity == "high"]
            
            if critical_drifts:
                monitoring_result['recommendations'].append("Re-entraînement immédiat recommandé")
                if self.auto_retrain_enabled:
                    await self._trigger_retraining(model_name, TriggerType.DATA_DRIFT)
            elif high_drifts:
                monitoring_result['recommendations'].append("Re-entraînement à planifier")
        
        # Évaluation performance si labels disponibles
        if true_labels is not None:
            current_performance = self._evaluate_performance(predictions, true_labels)
            
            # Comparaison avec performance historique
            historical_perf = self.performance_history[model_name]
            
            if historical_perf:
                last_accuracy = historical_perf[-1]['accuracy']
                performance_drop = last_accuracy - current_performance['accuracy']
                
                if performance_drop > self.performance_threshold:
                    monitoring_result['performance_degradation'] = True
                    monitoring_result['performance_drop'] = performance_drop
                    monitoring_result['recommendations'].append(
                        f"Dégradation de performance détectée: -{performance_drop:.3f}"
                    )
                    
                    if self.auto_retrain_enabled:
                        await self._trigger_retraining(model_name, TriggerType.PERFORMANCE_DEGRADATION)
            
            # Sauvegarder performance
            historical_perf.append({
                'timestamp': datetime.now(),
                'accuracy': current_performance['accuracy'],
                'f1_score': current_performance['f1_score']
            })
            
            monitoring_result['current_performance'] = current_performance
        
        logger.info(f"📊 Monitoring {model_name}: {'⚠️' if monitoring_result['drift_detected'] or monitoring_result['performance_degradation'] else '✅'}")
        
        return monitoring_result
    
    def _evaluate_performance(self, predictions: np.ndarray, true_labels: np.ndarray) -> Dict[str, float]:
        """Évalue la performance des prédictions"""
        
        # Classification ou régression
        if len(np.unique(true_labels)) <= 10:  # Classification
            accuracy = accuracy_score(true_labels, predictions)
            precision = precision_score(true_labels, predictions, average='weighted', zero_division=0)
            recall = recall_score(true_labels, predictions, average='weighted', zero_division=0)
            f1 = f1_score(true_labels, predictions, average='weighted', zero_division=0)
            
            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }
        else:  # Régression
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            
            mse = mean_squared_error(true_labels, predictions)
            mae = mean_absolute_error(true_labels, predictions)
            r2 = r2_score(true_labels, predictions)
            
            return {
                'mse': mse,
                'mae': mae,
                'r2_score': r2,
                'accuracy': r2  # Utiliser R² comme proxy d'accuracy pour régression
            }
    
    async def _trigger_retraining(self, model_name: str, trigger_type: TriggerType):
        """Déclenche le re-entraînement d'un modèle"""
        
        job_id = f"retrain_{model_name}_{int(time.time())}"
        
        job = LearningJob(
            job_id=job_id,
            model_name=model_name,
            strategy=LearningStrategy.BATCH,  # Par défaut
            trigger_type=trigger_type,
            training_config={},
            data_config={},
            status="pending"
        )
        
        self.learning_jobs[job_id] = job
        
        logger.info(f"🔄 Re-entraînement déclenché pour {model_name} (trigger: {trigger_type.value})")
        
        # Lancer en arrière-plan
        asyncio.create_task(self._execute_retraining_job(job))
        
        return job_id
    
    async def _execute_retraining_job(self, job: LearningJob):
        """Exécute un job de re-entraînement"""
        
        try:
            job.status = "running"
            job.started_at = datetime.now()
            job.progress = 10
            
            logger.info(f"🔄 Démarrage re-entraînement: {job.job_id}")
            
            # Récupérer le moteur IA principal
            ai_engine = await get_advanced_ai_engine()
            
            # Simuler génération de nouvelles données d'entraînement
            # En production, récupérer depuis la base de données
            training_data = pd.DataFrame({
                'feature1': np.random.randn(1000),
                'feature2': np.random.randn(1000),
                'feature3': np.random.randn(1000),
                'chiffre_affaires': np.random.exponential(1000000, 1000),
                'effectifs': np.random.poisson(50, 1000),
                'target_ma_score': np.random.uniform(0, 100, 1000)
            })
            
            job.progress = 30
            
            # Re-entraînement avec moteur ensemble
            model_id = await ai_engine.ensemble_manager.train_ma_scoring_model(training_data)
            
            job.progress = 70
            
            # Évaluation du nouveau modèle
            # (ici on simule - en production, évaluer sur set de validation)
            new_performance = ModelPerformance(
                model_id=model_id,
                model_version="retrained",
                accuracy=0.85 + np.random.uniform(-0.1, 0.1),
                precision=0.82 + np.random.uniform(-0.1, 0.1),
                recall=0.80 + np.random.uniform(-0.1, 0.1),
                f1_score=0.81 + np.random.uniform(-0.1, 0.1),
                business_metrics={},
                evaluation_date=datetime.now(),
                data_size=len(training_data),
                feature_count=len(training_data.columns) - 1,
                performance_change={}
            )
            
            job.progress = 90
            
            # Sauvegarder nouvelle version
            model = ai_engine.ensemble_manager.models[model_id]
            version_id = await self.version_manager.save_model_version(
                job.model_name, model, new_performance, job.training_config
            )
            
            # Comparaison avec modèle actuel
            current_model_accuracy = 0.80  # Simulation
            improvement = new_performance.accuracy - current_model_accuracy
            
            job.created_model_version = version_id
            job.performance_improvement = improvement
            job.progress = 100
            job.status = "completed"
            job.completed_at = datetime.now()
            
            # Auto-déploiement si amélioration significative
            if improvement > 0.02:  # 2% d'amélioration
                await self.version_manager.promote_to_active(job.model_name, version_id)
                logger.info(f"🚀 Nouveau modèle déployé automatiquement: {version_id}")
            else:
                logger.info(f"💡 Nouveau modèle disponible mais pas déployé: amélioration de {improvement:.3f}")
            
            logger.info(f"✅ Re-entraînement terminé: {job.job_id}")
            
        except Exception as e:
            job.status = "failed"
            job.error_message = str(e)
            job.completed_at = datetime.now()
            
            logger.error(f"❌ Erreur re-entraînement {job.job_id}: {e}")
    
    async def schedule_retraining(
        self, 
        model_name: str, 
        schedule_cron: str,
        training_config: Dict[str, Any] = None
    ):
        """Programme un re-entraînement périodique"""
        
        # Utiliser la bibliothèque schedule pour simplicité
        # En production, utiliser un scheduler plus robuste (Celery, APScheduler)
        
        def retrain_job():
            asyncio.create_task(
                self._trigger_retraining(model_name, TriggerType.SCHEDULED)
            )
        
        # Parser basique du cron (simplification)
        if schedule_cron == "daily":
            schedule.every().day.at("02:00").do(retrain_job)
        elif schedule_cron == "weekly":
            schedule.every().week.do(retrain_job)
        elif schedule_cron.startswith("every"):
            # Format: "every 6 hours"
            parts = schedule_cron.split()
            if len(parts) == 3:
                interval = int(parts[1])
                unit = parts[2]
                
                if unit == "hours":
                    schedule.every(interval).hours.do(retrain_job)
                elif unit == "days":
                    schedule.every(interval).days.do(retrain_job)
        
        logger.info(f"📅 Re-entraînement programmé pour {model_name}: {schedule_cron}")
    
    async def compare_model_versions(
        self, 
        model_name: str, 
        version_a: str, 
        version_b: str,
        test_data: pd.DataFrame,
        test_labels: np.ndarray
    ) -> Dict[str, Any]:
        """Compare deux versions d'un modèle (A/B testing)"""
        
        logger.info(f"🆚 Comparaison versions {version_a} vs {version_b}")
        
        # Charger les modèles
        model_a = await self.version_manager.load_model_version(version_a)
        model_b = await self.version_manager.load_model_version(version_b)
        
        # Prédictions
        pred_a = model_a.predict(test_data)
        pred_b = model_b.predict(test_data)
        
        # Évaluations
        perf_a = self._evaluate_performance(pred_a, test_labels)
        perf_b = self._evaluate_performance(pred_b, test_labels)
        
        # Comparaison statistique
        comparison = {
            'model_a': {
                'version_id': version_a,
                'performance': perf_a
            },
            'model_b': {
                'version_id': version_b,
                'performance': perf_b
            },
            'comparison': {},
            'recommendation': 'inconclusive'
        }
        
        # Calcul des différences
        for metric in perf_a.keys():
            diff = perf_b[metric] - perf_a[metric]
            comparison['comparison'][f'{metric}_difference'] = diff
            comparison['comparison'][f'{metric}_improvement_percent'] = (diff / perf_a[metric] * 100) if perf_a[metric] != 0 else 0
        
        # Recommandation
        accuracy_improvement = comparison['comparison'].get('accuracy_difference', 0)
        
        if accuracy_improvement > 0.02:  # > 2%
            comparison['recommendation'] = 'deploy_model_b'
        elif accuracy_improvement < -0.02:  # < -2%
            comparison['recommendation'] = 'keep_model_a'
        else:
            comparison['recommendation'] = 'inconclusive'
        
        logger.info(f"📊 Comparaison terminée: {comparison['recommendation']}")
        
        return comparison
    
    def get_learning_system_status(self) -> Dict[str, Any]:
        """Retourne le statut du système d'apprentissage"""
        
        # Statistiques des jobs
        total_jobs = len(self.learning_jobs)
        completed_jobs = len([j for j in self.learning_jobs.values() if j.status == "completed"])
        failed_jobs = len([j for j in self.learning_jobs.values() if j.status == "failed"])
        running_jobs = len([j for j in self.learning_jobs.values() if j.status == "running"])
        
        # Modèles actifs
        active_models = len(self.version_manager.active_models)
        
        # Dérives récentes
        recent_drifts = sum(
            len([d for d in drifts if d.detected_at > datetime.now() - timedelta(hours=24)])
            for drifts in self.drift_history.values()
        )
        
        return {
            'system_health': 'operational',
            'monitoring_enabled': self.monitoring_enabled,
            'auto_retrain_enabled': self.auto_retrain_enabled,
            'jobs_statistics': {
                'total_jobs': total_jobs,
                'completed_jobs': completed_jobs,
                'failed_jobs': failed_jobs,
                'running_jobs': running_jobs,
                'success_rate': completed_jobs / total_jobs if total_jobs > 0 else 0
            },
            'models_statistics': {
                'active_models': active_models,
                'total_versions': sum(len(versions) for versions in self.version_manager.model_versions.values()),
                'recent_drifts_24h': recent_drifts
            },
            'performance_monitoring': {
                'models_monitored': len(self.performance_history),
                'performance_threshold': self.performance_threshold,
                'drift_thresholds': self.drift_detector.drift_thresholds
            },
            'recent_jobs': [
                {
                    'job_id': job.job_id,
                    'model_name': job.model_name,
                    'status': job.status,
                    'trigger_type': job.trigger_type.value,
                    'started_at': job.started_at.isoformat() if job.started_at else None,
                    'progress': job.progress
                }
                for job in sorted(self.learning_jobs.values(), key=lambda x: x.created_at, reverse=True)[:10]
            ],
            'last_updated': datetime.now().isoformat()
        }


# Instance globale
_continuous_learning_engine: Optional[ContinuousLearningEngine] = None


async def get_continuous_learning_engine() -> ContinuousLearningEngine:
    """Factory pour obtenir le moteur d'apprentissage continu"""
    global _continuous_learning_engine
    
    if _continuous_learning_engine is None:
        _continuous_learning_engine = ContinuousLearningEngine()
    
    return _continuous_learning_engine


# Fonctions utilitaires

async def start_model_monitoring(
    model_name: str, 
    initial_model: Any, 
    training_data: pd.DataFrame
):
    """Démarre le monitoring d'un modèle"""
    
    engine = await get_continuous_learning_engine()
    
    # Performance initiale (simulation)
    performance = ModelPerformance(
        model_id=f"{model_name}_initial",
        model_version="v1",
        accuracy=0.85,
        precision=0.82,
        recall=0.80,
        f1_score=0.81,
        business_metrics={},
        evaluation_date=datetime.now(),
        data_size=len(training_data),
        feature_count=len(training_data.columns),
        performance_change={}
    )
    
    await engine.register_model(model_name, initial_model, training_data, performance)
    
    logger.info(f"🎯 Monitoring démarré pour {model_name}")


async def check_model_health(model_name: str, new_data: pd.DataFrame) -> Dict[str, Any]:
    """Vérifie la santé d'un modèle avec nouvelles données"""
    
    engine = await get_continuous_learning_engine()
    
    # Récupérer modèle actif
    model = await engine.version_manager.get_active_model(model_name)
    
    if model is None:
        return {'error': f'Modèle {model_name} non trouvé'}
    
    # Prédictions
    predictions = model.predict(new_data.select_dtypes(include=[np.number]).fillna(0))
    
    # Monitoring (sans vraies labels pour simulation)
    monitoring_result = await engine.monitor_model_performance(
        model_name, new_data, predictions
    )
    
    return monitoring_result


async def trigger_model_retraining(model_name: str, trigger_reason: str = "manual") -> str:
    """Déclenche manuellement le re-entraînement d'un modèle"""
    
    engine = await get_continuous_learning_engine()
    
    trigger_type = TriggerType.MANUAL
    if trigger_reason == "performance":
        trigger_type = TriggerType.PERFORMANCE_DEGRADATION
    elif trigger_reason == "drift":
        trigger_type = TriggerType.DATA_DRIFT
    
    job_id = await engine._trigger_retraining(model_name, trigger_type)
    
    return job_id