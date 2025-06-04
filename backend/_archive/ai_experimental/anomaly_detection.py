"""
Système de détection d'anomalies et alertes automatiques
US-010: Détection intelligente d'anomalies pour M&A Intelligence Platform

Ce module fournit:
- Détection d'anomalies multi-dimensionnelle
- Alertes automatiques en temps réel
- Surveillance des patterns inhabituels
- Détection de fraudes et incohérences
- Analyse d'outliers intelligente
- Système d'escalade adaptatif
- Monitoring prédictif des risques
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from functools import wraps
from enum import Enum
import json
import time
import uuid
from collections import defaultdict, deque
import statistics

# Machine Learning pour détection d'anomalies
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.covariance import EllipticEnvelope
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
from scipy import stats
from scipy.spatial.distance import mahalanobis

# Time series anomaly detection
from scipy.signal import find_peaks
from scipy.stats import zscore
import pyod
from pyod.models.abod import ABOD
from pyod.models.auto_encoder import AutoEncoder
from pyod.models.copod import COPOD

# Alerting
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart

from app.core.logging_system import get_logger, LogCategory
from app.core.cache_manager import get_cache_manager, cached

logger = get_logger("anomaly_detection", LogCategory.AI_ML)


class AnomalyType(str, Enum):
    """Types d'anomalies"""
    STATISTICAL = "statistical"
    BUSINESS_RULE = "business_rule"
    PATTERN = "pattern"
    TEMPORAL = "temporal"
    FRAUD = "fraud"
    PERFORMANCE = "performance"
    DATA_QUALITY = "data_quality"


class AnomalySeverity(str, Enum):
    """Sévérité des anomalies"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class AlertChannel(str, Enum):
    """Canaux d'alerte"""
    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"
    DATABASE = "database"
    LOG = "log"


class DetectionMethod(str, Enum):
    """Méthodes de détection"""
    ISOLATION_FOREST = "isolation_forest"
    LOCAL_OUTLIER_FACTOR = "local_outlier_factor"
    ONE_CLASS_SVM = "one_class_svm"
    ELLIPTIC_ENVELOPE = "elliptic_envelope"
    DBSCAN = "dbscan"
    ZSCORE = "zscore"
    IQR = "iqr"
    AUTOENCODER = "autoencoder"
    ENSEMBLE = "ensemble"


@dataclass
class Anomaly:
    """Anomalie détectée"""
    anomaly_id: str
    anomaly_type: AnomalyType
    severity: AnomalySeverity
    
    # Détails de l'anomalie
    title: str
    description: str
    affected_entity: str  # SIREN, user_id, etc.
    
    # Données de l'anomalie
    anomaly_score: float  # 0-1
    confidence: float  # 0-1
    detection_method: DetectionMethod
    
    # Contexte
    feature_values: Dict[str, Any]
    expected_ranges: Dict[str, Tuple[float, float]]
    deviation_details: Dict[str, float]
    
    # Metadata
    business_impact: str
    recommended_actions: List[str]
    related_anomalies: List[str] = field(default_factory=list)
    
    # Timestamps
    detected_at: datetime = field(default_factory=datetime.now)
    first_seen: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    
    # Alerte
    alert_sent: bool = False
    alert_channels: List[AlertChannel] = field(default_factory=list)


@dataclass
class AlertRule:
    """Règle d'alerte"""
    rule_id: str
    name: str
    description: str
    
    # Conditions
    anomaly_types: List[AnomalyType]
    min_severity: AnomalySeverity
    min_confidence: float
    
    # Canaux et destinataires
    channels: List[AlertChannel]
    recipients: List[str]
    
    # Configuration
    enabled: bool = True
    cooldown_minutes: int = 60
    escalation_enabled: bool = False
    escalation_threshold_minutes: int = 120
    
    # Historique
    last_triggered: Optional[datetime] = None
    trigger_count: int = 0
    
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class DetectionConfig:
    """Configuration de détection"""
    method: DetectionMethod
    parameters: Dict[str, Any]
    
    # Seuils
    contamination: float = 0.1  # Proportion d'outliers attendue
    threshold: float = 0.5
    
    # Features
    feature_columns: List[str] = field(default_factory=list)
    scaling_method: str = "robust"  # standard, robust, minmax
    
    # Fenêtrage temporel
    window_size: Optional[int] = None
    overlap: float = 0.1
    
    enabled: bool = True


class StatisticalAnomalyDetector:
    """Détecteur d'anomalies statistiques"""
    
    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.scalers: Dict[str, Any] = {}
        self.baselines: Dict[str, Dict[str, float]] = {}
        
    async def detect_statistical_anomalies(
        self, 
        data: pd.DataFrame, 
        config: DetectionConfig
    ) -> List[Anomaly]:
        """Détecte les anomalies statistiques"""
        
        anomalies = []
        
        # Préparation des données
        if config.feature_columns:
            feature_data = data[config.feature_columns]
        else:
            feature_data = data.select_dtypes(include=[np.number])
        
        # Nettoyage
        feature_data = feature_data.fillna(feature_data.median())
        
        if len(feature_data) < 2:
            return []
        
        # Normalisation
        scaler_key = f"{config.method.value}_{hash(str(config.feature_columns))}"
        
        if scaler_key not in self.scalers:
            if config.scaling_method == "robust":
                self.scalers[scaler_key] = RobustScaler()
            else:
                self.scalers[scaler_key] = StandardScaler()
        
        scaled_data = self.scalers[scaler_key].fit_transform(feature_data)
        
        # Détection selon la méthode
        if config.method == DetectionMethod.ISOLATION_FOREST:
            anomaly_labels, scores = await self._isolation_forest_detection(
                scaled_data, config
            )
        elif config.method == DetectionMethod.LOCAL_OUTLIER_FACTOR:
            anomaly_labels, scores = await self._lof_detection(scaled_data, config)
        elif config.method == DetectionMethod.ONE_CLASS_SVM:
            anomaly_labels, scores = await self._ocsvm_detection(scaled_data, config)
        elif config.method == DetectionMethod.ELLIPTIC_ENVELOPE:
            anomaly_labels, scores = await self._elliptic_envelope_detection(
                scaled_data, config
            )
        elif config.method == DetectionMethod.ZSCORE:
            anomaly_labels, scores = await self._zscore_detection(
                feature_data, config
            )
        elif config.method == DetectionMethod.IQR:
            anomaly_labels, scores = await self._iqr_detection(feature_data, config)
        else:
            return []
        
        # Création des objets Anomaly
        for idx, (is_anomaly, score) in enumerate(zip(anomaly_labels, scores)):
            if is_anomaly:
                entity_id = data.iloc[idx].get('siren', f'entity_{idx}')
                
                # Analyse des déviations
                deviation_details = self._analyze_deviations(
                    feature_data.iloc[idx], feature_data, config.feature_columns
                )
                
                # Détermination de la sévérité
                severity = self._determine_severity(score, deviation_details)
                
                anomaly = Anomaly(
                    anomaly_id=f"stat_{uuid.uuid4().hex[:8]}",
                    anomaly_type=AnomalyType.STATISTICAL,
                    severity=severity,
                    title=f"Anomalie statistique détectée - {entity_id}",
                    description=f"Valeurs inhabituelles détectées avec {config.method.value}",
                    affected_entity=entity_id,
                    anomaly_score=abs(score),
                    confidence=min(abs(score), 1.0),
                    detection_method=config.method,
                    feature_values=feature_data.iloc[idx].to_dict(),
                    expected_ranges=self._calculate_expected_ranges(
                        feature_data, config.feature_columns
                    ),
                    deviation_details=deviation_details,
                    business_impact=self._assess_business_impact(deviation_details),
                    recommended_actions=self._generate_recommendations(deviation_details)
                )
                
                anomalies.append(anomaly)
        
        logger.info(f"📊 Détection statistique: {len(anomalies)} anomalies trouvées")
        
        return anomalies
    
    async def _isolation_forest_detection(
        self, 
        data: np.ndarray, 
        config: DetectionConfig
    ) -> Tuple[List[bool], List[float]]:
        """Détection avec Isolation Forest"""
        
        model = IsolationForest(
            contamination=config.contamination,
            random_state=42,
            **config.parameters
        )
        
        predictions = model.fit_predict(data)
        scores = model.score_samples(data)
        
        # -1 = anomalie, 1 = normal
        anomaly_labels = predictions == -1
        
        return anomaly_labels, scores
    
    async def _lof_detection(
        self, 
        data: np.ndarray, 
        config: DetectionConfig
    ) -> Tuple[List[bool], List[float]]:
        """Détection avec Local Outlier Factor"""
        
        model = LocalOutlierFactor(
            contamination=config.contamination,
            **config.parameters
        )
        
        predictions = model.fit_predict(data)
        scores = model.negative_outlier_factor_
        
        anomaly_labels = predictions == -1
        
        return anomaly_labels, scores
    
    async def _ocsvm_detection(
        self, 
        data: np.ndarray, 
        config: DetectionConfig
    ) -> Tuple[List[bool], List[float]]:
        """Détection avec One-Class SVM"""
        
        model = OneClassSVM(
            nu=config.contamination,
            **config.parameters
        )
        
        predictions = model.fit_predict(data)
        scores = model.score_samples(data)
        
        anomaly_labels = predictions == -1
        
        return anomaly_labels, scores
    
    async def _elliptic_envelope_detection(
        self, 
        data: np.ndarray, 
        config: DetectionConfig
    ) -> Tuple[List[bool], List[float]]:
        """Détection avec Elliptic Envelope"""
        
        model = EllipticEnvelope(
            contamination=config.contamination,
            **config.parameters
        )
        
        predictions = model.fit_predict(data)
        scores = model.score_samples(data)
        
        anomaly_labels = predictions == -1
        
        return anomaly_labels, scores
    
    async def _zscore_detection(
        self, 
        data: pd.DataFrame, 
        config: DetectionConfig
    ) -> Tuple[List[bool], List[float]]:
        """Détection avec Z-Score"""
        
        threshold = config.parameters.get('threshold', 3.0)
        
        z_scores = np.abs(stats.zscore(data, nan_policy='omit'))
        
        # Anomalie si au moins une feature a un z-score > threshold
        anomaly_labels = (z_scores > threshold).any(axis=1)
        scores = np.max(z_scores, axis=1)
        
        return anomaly_labels, scores
    
    async def _iqr_detection(
        self, 
        data: pd.DataFrame, 
        config: DetectionConfig
    ) -> Tuple[List[bool], List[float]]:
        """Détection avec méthode IQR"""
        
        multiplier = config.parameters.get('multiplier', 1.5)
        
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        
        # Vérifier pour chaque ligne si elle est hors limites
        anomaly_labels = []
        scores = []
        
        for idx, row in data.iterrows():
            outliers = (row < lower_bound) | (row > upper_bound)
            is_anomaly = outliers.any()
            
            # Score basé sur la distance aux limites
            if is_anomaly:
                distances = []
                for col in data.columns:
                    val = row[col]
                    if val < lower_bound[col]:
                        distances.append((lower_bound[col] - val) / (IQR[col] + 1e-6))
                    elif val > upper_bound[col]:
                        distances.append((val - upper_bound[col]) / (IQR[col] + 1e-6))
                
                score = max(distances) if distances else 0
            else:
                score = 0
            
            anomaly_labels.append(is_anomaly)
            scores.append(score)
        
        return anomaly_labels, scores
    
    def _analyze_deviations(
        self, 
        sample: pd.Series, 
        reference_data: pd.DataFrame, 
        feature_columns: List[str]
    ) -> Dict[str, float]:
        """Analyse les déviations d'un échantillon"""
        
        deviations = {}
        
        for col in feature_columns:
            if col not in sample or col not in reference_data.columns:
                continue
            
            value = sample[col]
            ref_values = reference_data[col].dropna()
            
            if len(ref_values) == 0:
                continue
            
            mean_val = ref_values.mean()
            std_val = ref_values.std()
            
            if std_val > 0:
                z_score = (value - mean_val) / std_val
                deviations[col] = abs(z_score)
            else:
                deviations[col] = 0.0
        
        return deviations
    
    def _determine_severity(
        self, 
        anomaly_score: float, 
        deviations: Dict[str, float]
    ) -> AnomalySeverity:
        """Détermine la sévérité d'une anomalie"""
        
        max_deviation = max(deviations.values()) if deviations else 0
        
        # Combinaison score et déviations
        severity_score = (abs(anomaly_score) + max_deviation) / 2
        
        if severity_score > 4.0:
            return AnomalySeverity.CRITICAL
        elif severity_score > 3.0:
            return AnomalySeverity.HIGH
        elif severity_score > 2.0:
            return AnomalySeverity.MEDIUM
        elif severity_score > 1.0:
            return AnomalySeverity.LOW
        else:
            return AnomalySeverity.INFO
    
    def _calculate_expected_ranges(
        self, 
        data: pd.DataFrame, 
        feature_columns: List[str]
    ) -> Dict[str, Tuple[float, float]]:
        """Calcule les plages attendues pour chaque feature"""
        
        ranges = {}
        
        for col in feature_columns:
            if col in data.columns:
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                
                ranges[col] = (lower, upper)
        
        return ranges
    
    def _assess_business_impact(self, deviations: Dict[str, float]) -> str:
        """Évalue l'impact business d'une anomalie"""
        
        if not deviations:
            return "Impact inconnu"
        
        max_deviation = max(deviations.values())
        
        # Analyse des features impactées
        financial_features = ['chiffre_affaires', 'benefices', 'ca_per_employee']
        operational_features = ['effectifs', 'productivity', 'growth_rate']
        
        financial_impact = any(
            feature in deviations and deviations[feature] > 2.0
            for feature in financial_features
        )
        
        operational_impact = any(
            feature in deviations and deviations[feature] > 2.0
            for feature in operational_features
        )
        
        if financial_impact and operational_impact:
            return "Impact financier et opérationnel significatif"
        elif financial_impact:
            return "Impact financier potentiel"
        elif operational_impact:
            return "Impact opérationnel possible"
        elif max_deviation > 3.0:
            return "Anomalie majeure nécessitant investigation"
        else:
            return "Anomalie mineure"
    
    def _generate_recommendations(self, deviations: Dict[str, float]) -> List[str]:
        """Génère des recommandations d'actions"""
        
        recommendations = []
        
        if not deviations:
            return ["Analyser les causes de l'anomalie"]
        
        # Recommandations spécifiques par feature
        for feature, deviation in deviations.items():
            if deviation > 3.0:
                if 'chiffre_affaires' in feature:
                    recommendations.append("Vérifier les données financières et la saisonnalité")
                elif 'effectifs' in feature:
                    recommendations.append("Contrôler les données RH et mouvements de personnel")
                elif 'score' in feature:
                    recommendations.append("Re-évaluer le modèle de scoring")
                else:
                    recommendations.append(f"Investiguer la feature {feature}")
        
        # Recommandations générales
        max_deviation = max(deviations.values())
        
        if max_deviation > 4.0:
            recommendations.append("Alerte prioritaire - Investigation immédiate recommandée")
        elif max_deviation > 2.0:
            recommendations.append("Analyser les tendances récentes")
        
        if len(deviations) > 3:
            recommendations.append("Anomalie multi-dimensionnelle - Analyse approfondie nécessaire")
        
        return recommendations[:5]  # Limiter à 5 recommandations


class BusinessRuleDetector:
    """Détecteur d'anomalies basé sur règles métier"""
    
    def __init__(self):
        self.business_rules: List[Dict[str, Any]] = []
        self._setup_default_rules()
    
    def _setup_default_rules(self):
        """Configure les règles métier par défaut"""
        
        self.business_rules = [
            {
                'rule_id': 'revenue_coherence',
                'name': 'Cohérence chiffre d\'affaires',
                'condition': lambda row: (
                    row.get('chiffre_affaires', 0) > 0 and 
                    row.get('effectifs', 0) > 0 and
                    (row['chiffre_affaires'] / row['effectifs']) < 10000
                ),
                'severity': AnomalySeverity.HIGH,
                'description': 'Chiffre d\'affaires par employé anormalement bas'
            },
            {
                'rule_id': 'negative_financials',
                'name': 'Indicateurs financiers négatifs',
                'condition': lambda row: row.get('chiffre_affaires', 0) < 0,
                'severity': AnomalySeverity.CRITICAL,
                'description': 'Chiffre d\'affaires négatif détecté'
            },
            {
                'rule_id': 'unrealistic_growth',
                'name': 'Croissance irréaliste',
                'condition': lambda row: abs(row.get('growth_rate', 0)) > 5.0,
                'severity': AnomalySeverity.MEDIUM,
                'description': 'Taux de croissance extrême détecté'
            },
            {
                'rule_id': 'missing_critical_data',
                'name': 'Données critiques manquantes',
                'condition': lambda row: (
                    pd.isna(row.get('siren')) or 
                    pd.isna(row.get('chiffre_affaires')) or
                    pd.isna(row.get('secteur_activite'))
                ),
                'severity': AnomalySeverity.HIGH,
                'description': 'Données critiques manquantes'
            },
            {
                'rule_id': 'ma_score_outlier',
                'name': 'Score M&A aberrant',
                'condition': lambda row: (
                    row.get('ma_score', 0) > 100 or row.get('ma_score', 0) < 0
                ),
                'severity': AnomalySeverity.HIGH,
                'description': 'Score M&A en dehors de la plage valide'
            }
        ]
    
    async def detect_business_rule_violations(self, data: pd.DataFrame) -> List[Anomaly]:
        """Détecte les violations de règles métier"""
        
        anomalies = []
        
        for idx, row in data.iterrows():
            for rule in self.business_rules:
                try:
                    # Évaluer la condition
                    violates_rule = rule['condition'](row)
                    
                    if violates_rule:
                        entity_id = row.get('siren', f'entity_{idx}')
                        
                        anomaly = Anomaly(
                            anomaly_id=f"rule_{rule['rule_id']}_{uuid.uuid4().hex[:8]}",
                            anomaly_type=AnomalyType.BUSINESS_RULE,
                            severity=rule['severity'],
                            title=f"Violation règle: {rule['name']}",
                            description=rule['description'],
                            affected_entity=entity_id,
                            anomaly_score=1.0,  # Violation binaire
                            confidence=0.95,  # Haute confiance pour règles métier
                            detection_method=DetectionMethod.ENSEMBLE,  # Rules-based
                            feature_values=row.to_dict(),
                            expected_ranges={},
                            deviation_details={'rule_violation': 1.0},
                            business_impact="Violation de règle métier critique",
                            recommended_actions=[
                                "Vérifier la qualité des données",
                                "Corriger les valeurs aberrantes",
                                "Valider avec les sources"
                            ]
                        )
                        
                        anomalies.append(anomaly)
                        
                except Exception as e:
                    logger.warning(f"Erreur évaluation règle {rule['rule_id']}: {e}")
        
        logger.info(f"📋 Règles métier: {len(anomalies)} violations détectées")
        
        return anomalies


class TemporalAnomalyDetector:
    """Détecteur d'anomalies temporelles"""
    
    def __init__(self):
        self.time_series_baselines: Dict[str, Dict[str, Any]] = {}
    
    async def detect_temporal_anomalies(
        self, 
        time_series_data: pd.DataFrame,
        datetime_column: str = 'timestamp',
        value_columns: List[str] = None
    ) -> List[Anomaly]:
        """Détecte les anomalies dans les séries temporelles"""
        
        if datetime_column not in time_series_data.columns:
            logger.warning(f"Colonne temporelle {datetime_column} non trouvée")
            return []
        
        anomalies = []
        
        # Préparation des données
        df = time_series_data.copy()
        df[datetime_column] = pd.to_datetime(df[datetime_column])
        df = df.sort_values(datetime_column)
        
        value_columns = value_columns or df.select_dtypes(include=[np.number]).columns.tolist()
        
        for column in value_columns:
            if column not in df.columns:
                continue
            
            series_anomalies = await self._detect_series_anomalies(
                df, datetime_column, column
            )
            anomalies.extend(series_anomalies)
        
        logger.info(f"⏰ Temporal: {len(anomalies)} anomalies temporelles détectées")
        
        return anomalies
    
    async def _detect_series_anomalies(
        self,
        df: pd.DataFrame,
        datetime_col: str,
        value_col: str
    ) -> List[Anomaly]:
        """Détecte les anomalies dans une série temporelle"""
        
        anomalies = []
        values = df[value_col].dropna()
        
        if len(values) < 10:  # Pas assez de données
            return []
        
        # Méthode 1: Z-Score sur fenêtre glissante
        window_size = min(30, len(values) // 3)
        z_scores = values.rolling(window=window_size).apply(
            lambda x: abs((x.iloc[-1] - x.mean()) / (x.std() + 1e-6))
        )
        
        # Méthode 2: Détection de pics
        peaks, _ = find_peaks(values, height=values.mean() + 2 * values.std())
        valleys, _ = find_peaks(-values, height=-values.mean() + 2 * values.std())
        
        # Méthode 3: Changements brusques
        diff = values.diff().abs()
        sudden_changes = diff > (diff.mean() + 3 * diff.std())
        
        # Combiner les détections
        anomaly_indices = set()
        
        # Z-Score anomalies
        z_anomalies = z_scores > 3.0
        anomaly_indices.update(z_anomalies[z_anomalies].index)
        
        # Pics et vallées
        anomaly_indices.update(peaks)
        anomaly_indices.update(valleys)
        
        # Changements brusques
        anomaly_indices.update(sudden_changes[sudden_changes].index)
        
        # Créer objets Anomaly
        for idx in anomaly_indices:
            if idx in df.index:
                row = df.loc[idx]
                
                # Calcul du score d'anomalie
                z_score = z_scores.get(idx, 0)
                change_score = diff.get(idx, 0) / (diff.std() + 1e-6) if idx in diff.index else 0
                
                anomaly_score = max(z_score, change_score) / 5.0  # Normalisation
                
                anomaly = Anomaly(
                    anomaly_id=f"temporal_{value_col}_{uuid.uuid4().hex[:8]}",
                    anomaly_type=AnomalyType.TEMPORAL,
                    severity=self._determine_temporal_severity(anomaly_score),
                    title=f"Anomalie temporelle - {value_col}",
                    description=f"Valeur inhabituelle dans la série temporelle {value_col}",
                    affected_entity=str(row.get('siren', f'time_{idx}')),
                    anomaly_score=min(anomaly_score, 1.0),
                    confidence=min(anomaly_score * 0.8, 1.0),
                    detection_method=DetectionMethod.ENSEMBLE,
                    feature_values={value_col: row[value_col]},
                    expected_ranges={value_col: (
                        values.mean() - 2 * values.std(),
                        values.mean() + 2 * values.std()
                    )},
                    deviation_details={
                        'z_score': z_score,
                        'change_score': change_score,
                        'is_peak': idx in peaks,
                        'is_valley': idx in valleys
                    },
                    business_impact=self._assess_temporal_impact(value_col, anomaly_score),
                    recommended_actions=self._generate_temporal_recommendations(value_col)
                )
                
                anomalies.append(anomaly)
        
        return anomalies
    
    def _determine_temporal_severity(self, anomaly_score: float) -> AnomalySeverity:
        """Détermine la sévérité d'une anomalie temporelle"""
        
        if anomaly_score > 0.8:
            return AnomalySeverity.CRITICAL
        elif anomaly_score > 0.6:
            return AnomalySeverity.HIGH
        elif anomaly_score > 0.4:
            return AnomalySeverity.MEDIUM
        elif anomaly_score > 0.2:
            return AnomalySeverity.LOW
        else:
            return AnomalySeverity.INFO
    
    def _assess_temporal_impact(self, column: str, score: float) -> str:
        """Évalue l'impact d'une anomalie temporelle"""
        
        if 'revenue' in column.lower() or 'chiffre' in column.lower():
            return "Impact financier potentiel"
        elif 'score' in column.lower():
            return "Impact sur évaluation M&A"
        elif score > 0.7:
            return "Anomalie majeure nécessitant attention"
        else:
            return "Variation inhabituelle"
    
    def _generate_temporal_recommendations(self, column: str) -> List[str]:
        """Génère des recommandations pour anomalies temporelles"""
        
        recommendations = ["Analyser le contexte temporel de l'anomalie"]
        
        if 'revenue' in column.lower():
            recommendations.extend([
                "Vérifier les facteurs saisonniers",
                "Analyser les événements business récents"
            ])
        elif 'score' in column.lower():
            recommendations.extend([
                "Réviser les paramètres du modèle",
                "Vérifier la qualité des données d'entrée"
            ])
        
        recommendations.append("Surveiller l'évolution dans les prochaines périodes")
        
        return recommendations


class AlertingSystem:
    """Système d'alertes automatiques"""
    
    def __init__(self):
        self.alert_rules: Dict[str, AlertRule] = {}
        self.alert_history: List[Dict[str, Any]] = []
        self.notification_queue: deque = deque()
        
        self._setup_default_rules()
    
    def _setup_default_rules(self):
        """Configure les règles d'alerte par défaut"""
        
        # Règle critique
        critical_rule = AlertRule(
            rule_id="critical_anomalies",
            name="Anomalies critiques",
            description="Alerte pour toutes les anomalies critiques",
            anomaly_types=[AnomalyType.STATISTICAL, AnomalyType.BUSINESS_RULE, AnomalyType.FRAUD],
            min_severity=AnomalySeverity.CRITICAL,
            min_confidence=0.8,
            channels=[AlertChannel.EMAIL, AlertChannel.LOG],
            recipients=["admin@company.com"],
            cooldown_minutes=30,
            escalation_enabled=True
        )
        
        # Règle haute priorité
        high_priority_rule = AlertRule(
            rule_id="high_priority",
            name="Anomalies haute priorité",
            description="Alerte pour anomalies importantes",
            anomaly_types=[AnomalyType.STATISTICAL, AnomalyType.BUSINESS_RULE],
            min_severity=AnomalySeverity.HIGH,
            min_confidence=0.7,
            channels=[AlertChannel.EMAIL],
            recipients=["monitoring@company.com"],
            cooldown_minutes=60
        )
        
        self.alert_rules[critical_rule.rule_id] = critical_rule
        self.alert_rules[high_priority_rule.rule_id] = high_priority_rule
    
    async def process_anomaly_alert(self, anomaly: Anomaly) -> bool:
        """Traite une anomalie pour déclenchement d'alertes"""
        
        alert_triggered = False
        
        for rule in self.alert_rules.values():
            if not rule.enabled:
                continue
            
            # Vérifier si l'anomalie correspond à la règle
            if self._matches_alert_rule(anomaly, rule):
                # Vérifier cooldown
                if self._is_in_cooldown(rule):
                    logger.debug(f"Règle {rule.rule_id} en cooldown")
                    continue
                
                # Déclencher alerte
                await self._trigger_alert(anomaly, rule)
                alert_triggered = True
                
                # Mettre à jour règle
                rule.last_triggered = datetime.now()
                rule.trigger_count += 1
        
        return alert_triggered
    
    def _matches_alert_rule(self, anomaly: Anomaly, rule: AlertRule) -> bool:
        """Vérifie si une anomalie correspond à une règle"""
        
        # Type d'anomalie
        if anomaly.anomaly_type not in rule.anomaly_types:
            return False
        
        # Sévérité minimale
        severity_order = {
            AnomalySeverity.INFO: 0,
            AnomalySeverity.LOW: 1,
            AnomalySeverity.MEDIUM: 2,
            AnomalySeverity.HIGH: 3,
            AnomalySeverity.CRITICAL: 4
        }
        
        if severity_order[anomaly.severity] < severity_order[rule.min_severity]:
            return False
        
        # Confiance minimale
        if anomaly.confidence < rule.min_confidence:
            return False
        
        return True
    
    def _is_in_cooldown(self, rule: AlertRule) -> bool:
        """Vérifie si une règle est en cooldown"""
        
        if rule.last_triggered is None:
            return False
        
        cooldown_expires = rule.last_triggered + timedelta(minutes=rule.cooldown_minutes)
        return datetime.now() < cooldown_expires
    
    async def _trigger_alert(self, anomaly: Anomaly, rule: AlertRule):
        """Déclenche une alerte"""
        
        logger.info(f"🚨 Déclenchement alerte: {rule.name} pour {anomaly.anomaly_id}")
        
        # Préparer message
        alert_message = self._format_alert_message(anomaly, rule)
        
        # Envoyer via tous les canaux
        for channel in rule.channels:
            try:
                if channel == AlertChannel.EMAIL:
                    await self._send_email_alert(alert_message, rule.recipients)
                elif channel == AlertChannel.LOG:
                    self._log_alert(alert_message)
                elif channel == AlertChannel.WEBHOOK:
                    await self._send_webhook_alert(alert_message)
                
                # Marquer comme envoyé
                if channel not in anomaly.alert_channels:
                    anomaly.alert_channels.append(channel)
                
            except Exception as e:
                logger.error(f"❌ Erreur envoi alerte via {channel.value}: {e}")
        
        anomaly.alert_sent = True
        
        # Sauvegarder dans l'historique
        self.alert_history.append({
            'anomaly_id': anomaly.anomaly_id,
            'rule_id': rule.rule_id,
            'timestamp': datetime.now().isoformat(),
            'severity': anomaly.severity.value,
            'message': alert_message
        })
    
    def _format_alert_message(self, anomaly: Anomaly, rule: AlertRule) -> str:
        """Formate le message d'alerte"""
        
        message = f"""
🚨 ALERTE: {rule.name}

Anomalie détectée:
- ID: {anomaly.anomaly_id}
- Type: {anomaly.anomaly_type.value}
- Sévérité: {anomaly.severity.value}
- Entité affectée: {anomaly.affected_entity}
- Score: {anomaly.anomaly_score:.3f}
- Confiance: {anomaly.confidence:.3f}

Description:
{anomaly.description}

Impact business:
{anomaly.business_impact}

Actions recommandées:
{' - '.join(anomaly.recommended_actions)}

Détecté le: {anomaly.detected_at.strftime('%Y-%m-%d %H:%M:%S')}
Méthode: {anomaly.detection_method.value}
        """.strip()
        
        return message
    
    async def _send_email_alert(self, message: str, recipients: List[str]):
        """Envoie une alerte par email"""
        
        # Configuration email (à personnaliser)
        smtp_server = "smtp.gmail.com"
        smtp_port = 587
        sender_email = "alerts@company.com"
        sender_password = "password"  # À récupérer depuis env variables
        
        try:
            # Créer message
            msg = MimeMultipart()
            msg['From'] = sender_email
            msg['Subject'] = "🚨 Alerte Anomalie M&A Intelligence"
            msg.attach(MimeText(message, 'plain'))
            
            # Envoyer à tous les destinataires
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                server.login(sender_email, sender_password)
                
                for recipient in recipients:
                    msg['To'] = recipient
                    server.sendmail(sender_email, recipient, msg.as_string())
            
            logger.info(f"📧 Email envoyé à {len(recipients)} destinataires")
            
        except Exception as e:
            logger.error(f"❌ Erreur envoi email: {e}")
    
    def _log_alert(self, message: str):
        """Log une alerte"""
        logger.warning(f"🚨 ALERTE:\n{message}")
    
    async def _send_webhook_alert(self, message: str):
        """Envoie une alerte via webhook"""
        
        # Exemple d'envoi vers Slack webhook
        import aiohttp
        
        webhook_url = "https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK"
        
        payload = {
            "text": f"🚨 Anomalie détectée",
            "attachments": [
                {
                    "color": "danger",
                    "text": message
                }
            ]
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(webhook_url, json=payload) as response:
                    if response.status == 200:
                        logger.info("📱 Webhook Slack envoyé")
                    else:
                        logger.warning(f"⚠️ Webhook réponse: {response.status}")
                        
        except Exception as e:
            logger.error(f"❌ Erreur webhook: {e}")


class AnomalyDetectionEngine:
    """Moteur principal de détection d'anomalies"""
    
    def __init__(self):
        self.statistical_detector = StatisticalAnomalyDetector()
        self.business_rule_detector = BusinessRuleDetector()
        self.temporal_detector = TemporalAnomalyDetector()
        self.alerting_system = AlertingSystem()
        
        # Configuration par défaut
        self.default_configs = {
            DetectionMethod.ISOLATION_FOREST: DetectionConfig(
                method=DetectionMethod.ISOLATION_FOREST,
                parameters={'n_estimators': 100},
                contamination=0.1
            ),
            DetectionMethod.LOCAL_OUTLIER_FACTOR: DetectionConfig(
                method=DetectionMethod.LOCAL_OUTLIER_FACTOR,
                parameters={'n_neighbors': 20},
                contamination=0.1
            )
        }
        
        # Historique des anomalies
        self.anomaly_history: List[Anomaly] = []
        self.detection_statistics: Dict[str, int] = defaultdict(int)
        
        logger.info("🔍 Moteur de détection d'anomalies initialisé")
    
    async def detect_anomalies(
        self, 
        data: pd.DataFrame,
        detection_methods: List[DetectionMethod] = None,
        include_temporal: bool = False,
        datetime_column: str = 'timestamp'
    ) -> List[Anomaly]:
        """Détection complète d'anomalies"""
        
        if detection_methods is None:
            detection_methods = [
                DetectionMethod.ISOLATION_FOREST,
                DetectionMethod.LOCAL_OUTLIER_FACTOR
            ]
        
        logger.info(f"🔍 Détection d'anomalies sur {len(data)} échantillons")
        
        all_anomalies = []
        
        # Détection statistique
        for method in detection_methods:
            if method in self.default_configs:
                config = self.default_configs[method]
                config.feature_columns = data.select_dtypes(include=[np.number]).columns.tolist()
                
                try:
                    anomalies = await self.statistical_detector.detect_statistical_anomalies(
                        data, config
                    )
                    all_anomalies.extend(anomalies)
                    self.detection_statistics[f'{method.value}_detections'] += len(anomalies)
                    
                except Exception as e:
                    logger.error(f"❌ Erreur détection {method.value}: {e}")
        
        # Détection règles métier
        try:
            business_anomalies = await self.business_rule_detector.detect_business_rule_violations(data)
            all_anomalies.extend(business_anomalies)
            self.detection_statistics['business_rule_violations'] += len(business_anomalies)
            
        except Exception as e:
            logger.error(f"❌ Erreur règles métier: {e}")
        
        # Détection temporelle
        if include_temporal and datetime_column in data.columns:
            try:
                temporal_anomalies = await self.temporal_detector.detect_temporal_anomalies(
                    data, datetime_column
                )
                all_anomalies.extend(temporal_anomalies)
                self.detection_statistics['temporal_anomalies'] += len(temporal_anomalies)
                
            except Exception as e:
                logger.error(f"❌ Erreur détection temporelle: {e}")
        
        # Déduplication (anomalies similaires)
        deduplicated_anomalies = self._deduplicate_anomalies(all_anomalies)
        
        # Sauvegarde
        self.anomaly_history.extend(deduplicated_anomalies)
        
        # Traitement des alertes
        for anomaly in deduplicated_anomalies:
            try:
                await self.alerting_system.process_anomaly_alert(anomaly)
            except Exception as e:
                logger.error(f"❌ Erreur traitement alerte {anomaly.anomaly_id}: {e}")
        
        self.detection_statistics['total_detections'] += len(deduplicated_anomalies)
        
        logger.info(f"✅ Détection terminée: {len(deduplicated_anomalies)} anomalies uniques")
        
        return deduplicated_anomalies
    
    def _deduplicate_anomalies(self, anomalies: List[Anomaly]) -> List[Anomaly]:
        """Déduplique les anomalies similaires"""
        
        if len(anomalies) <= 1:
            return anomalies
        
        # Grouper par entité affectée
        by_entity = defaultdict(list)
        for anomaly in anomalies:
            by_entity[anomaly.affected_entity].append(anomaly)
        
        deduplicated = []
        
        for entity, entity_anomalies in by_entity.items():
            if len(entity_anomalies) == 1:
                deduplicated.extend(entity_anomalies)
            else:
                # Garder la plus sévère ou celle avec le score le plus élevé
                best_anomaly = max(
                    entity_anomalies,
                    key=lambda a: (
                        {
                            AnomalySeverity.CRITICAL: 4,
                            AnomalySeverity.HIGH: 3,
                            AnomalySeverity.MEDIUM: 2,
                            AnomalySeverity.LOW: 1,
                            AnomalySeverity.INFO: 0
                        }[a.severity],
                        a.anomaly_score
                    )
                )
                
                # Combiner les méthodes de détection
                related_ids = [a.anomaly_id for a in entity_anomalies if a != best_anomaly]
                best_anomaly.related_anomalies = related_ids
                
                deduplicated.append(best_anomaly)
        
        return deduplicated
    
    def get_anomaly_statistics(self) -> Dict[str, Any]:
        """Retourne les statistiques de détection"""
        
        # Statistiques par sévérité
        severity_counts = Counter(a.severity.value for a in self.anomaly_history)
        
        # Statistiques par type
        type_counts = Counter(a.anomaly_type.value for a in self.anomaly_history)
        
        # Anomalies récentes (24h)
        recent_anomalies = [
            a for a in self.anomaly_history 
            if a.detected_at > datetime.now() - timedelta(hours=24)
        ]
        
        return {
            'total_anomalies_detected': len(self.anomaly_history),
            'anomalies_last_24h': len(recent_anomalies),
            'detection_statistics': dict(self.detection_statistics),
            'severity_distribution': dict(severity_counts),
            'type_distribution': dict(type_counts),
            'active_alert_rules': len([r for r in self.alerting_system.alert_rules.values() if r.enabled]),
            'alerts_sent_24h': len([
                h for h in self.alerting_system.alert_history
                if datetime.fromisoformat(h['timestamp']) > datetime.now() - timedelta(hours=24)
            ]),
            'system_health': 'operational',
            'last_detection': self.anomaly_history[-1].detected_at.isoformat() if self.anomaly_history else None
        }


# Instance globale
_anomaly_detection_engine: Optional[AnomalyDetectionEngine] = None


async def get_anomaly_detection_engine() -> AnomalyDetectionEngine:
    """Factory pour obtenir le moteur de détection d'anomalies"""
    global _anomaly_detection_engine
    
    if _anomaly_detection_engine is None:
        _anomaly_detection_engine = AnomalyDetectionEngine()
    
    return _anomaly_detection_engine


# Fonctions utilitaires

async def scan_for_anomalies(data: pd.DataFrame, include_temporal: bool = False) -> List[Dict[str, Any]]:
    """Interface simplifiée pour détection d'anomalies"""
    
    engine = await get_anomaly_detection_engine()
    
    anomalies = await engine.detect_anomalies(data, include_temporal=include_temporal)
    
    # Conversion en format API
    return [
        {
            'anomaly_id': a.anomaly_id,
            'type': a.anomaly_type.value,
            'severity': a.severity.value,
            'title': a.title,
            'description': a.description,
            'affected_entity': a.affected_entity,
            'anomaly_score': a.anomaly_score,
            'confidence': a.confidence,
            'business_impact': a.business_impact,
            'recommended_actions': a.recommended_actions,
            'detected_at': a.detected_at.isoformat(),
            'alert_sent': a.alert_sent
        }
        for a in anomalies
    ]


async def setup_custom_alert_rule(
    name: str,
    anomaly_types: List[str],
    min_severity: str,
    recipients: List[str]
) -> str:
    """Crée une règle d'alerte personnalisée"""
    
    engine = await get_anomaly_detection_engine()
    
    rule = AlertRule(
        rule_id=f"custom_{uuid.uuid4().hex[:8]}",
        name=name,
        description=f"Règle personnalisée: {name}",
        anomaly_types=[AnomalyType(t) for t in anomaly_types],
        min_severity=AnomalySeverity(min_severity),
        min_confidence=0.7,
        channels=[AlertChannel.EMAIL, AlertChannel.LOG],
        recipients=recipients
    )
    
    engine.alerting_system.alert_rules[rule.rule_id] = rule
    
    logger.info(f"📨 Règle d'alerte créée: {name}")
    
    return rule.rule_id