"""
Analyseur de tendances ML pour M&A Intelligence Platform
US-008: Modèles ML avancés pour analyse de tendances

Ce module implémente:
- Détection automatique de patterns dans les données M&A
- Classification de tendances sectorielles
- Modèles d'apprentissage pour cycles économiques
- Clustering d'entreprises par profil
- Analyse de corrélations complexes
- Modèles adaptatifs et auto-apprenants
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import warnings
from collections import defaultdict, deque

# Machine Learning
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA, FastICA
from sklearn.manifold import TSNE
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
import joblib

# Time Series Analysis
try:
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.stattools import adfuller, kpss
    from statsmodels.stats.diagnostic import acorr_ljungbox
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    warnings.warn("Statsmodels non disponible - analyse temporelle limitée")

# Advanced ML
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

# Network Analysis
try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

from app.core.logging_system import get_logger, LogCategory
from app.services.ai_scoring_engine import ScoringFeatures

logger = get_logger("ml_trend_analyzer", LogCategory.ML)


class TrendType(str, Enum):
    """Types de tendances détectables"""
    CYCLICAL = "cyclical"
    SEASONAL = "seasonal"
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    VOLATILE = "volatile"
    STABLE = "stable"
    ANOMALOUS = "anomalous"


class ClusteringMethod(str, Enum):
    """Méthodes de clustering disponibles"""
    KMEANS = "kmeans"
    DBSCAN = "dbscan"
    AGGLOMERATIVE = "agglomerative"
    ISOLATION_FOREST = "isolation_forest"


class AnalysisScope(str, Enum):
    """Portée de l'analyse"""
    GLOBAL = "global"
    SECTORIAL = "sectorial"
    REGIONAL = "regional"
    TEMPORAL = "temporal"
    COMPARATIVE = "comparative"


@dataclass
class TrendPattern:
    """Pattern de tendance identifié"""
    trend_type: TrendType
    strength: float  # 0-1
    period_months: Optional[int]
    confidence: float
    description: str
    statistical_significance: float
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CompanyCluster:
    """Cluster d'entreprises similaires"""
    cluster_id: int
    cluster_name: str
    companies: List[str]
    centroid_features: Dict[str, float]
    cluster_characteristics: List[str]
    size: int
    cohesion_score: float


@dataclass
class MarketCycle:
    """Cycle de marché détecté"""
    cycle_name: str
    duration_months: int
    phases: List[str]
    current_phase: str
    phase_duration_remaining: int
    confidence: float
    historical_patterns: Dict[str, Any]


@dataclass
class TrendAnalysisResult:
    """Résultat complet d'analyse de tendances"""
    analysis_scope: AnalysisScope
    timeframe: str
    
    # Patterns détectés
    trend_patterns: List[TrendPattern]
    company_clusters: List[CompanyCluster]
    market_cycles: List[MarketCycle]
    
    # Insights
    key_insights: List[str]
    anomalies_detected: List[Dict[str, Any]]
    correlations: Dict[str, float]
    
    # Métadonnées
    data_quality_score: float
    processing_time_ms: float
    model_performance: Dict[str, float]
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire pour JSON"""
        return {
            'analysis_scope': self.analysis_scope.value,
            'timeframe': self.timeframe,
            'trend_patterns': [
                {
                    'type': tp.trend_type.value,
                    'strength': round(tp.strength, 3),
                    'period_months': tp.period_months,
                    'confidence': round(tp.confidence, 3),
                    'description': tp.description,
                    'significance': round(tp.statistical_significance, 3)
                }
                for tp in self.trend_patterns
            ],
            'company_clusters': [
                {
                    'id': cc.cluster_id,
                    'name': cc.cluster_name,
                    'size': cc.size,
                    'characteristics': cc.cluster_characteristics,
                    'cohesion': round(cc.cohesion_score, 3)
                }
                for cc in self.company_clusters
            ],
            'market_cycles': [
                {
                    'name': mc.cycle_name,
                    'duration': mc.duration_months,
                    'current_phase': mc.current_phase,
                    'remaining_duration': mc.phase_duration_remaining,
                    'confidence': round(mc.confidence, 3)
                }
                for mc in self.market_cycles
            ],
            'key_insights': self.key_insights,
            'anomalies': self.anomalies_detected,
            'correlations': {k: round(v, 3) for k, v in self.correlations.items()},
            'data_quality': round(self.data_quality_score, 2),
            'processing_time_ms': round(self.processing_time_ms, 2),
            'model_performance': {k: round(v, 3) for k, v in self.model_performance.items()},
            'timestamp': self.timestamp.isoformat()
        }


class MLTrendAnalyzer:
    """Analyseur de tendances utilisant des modèles ML avancés"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.cached_analyses = {}
        
        # Configuration des modèles
        self.model_configs = {
            'clustering': {
                'kmeans': {'n_clusters': 5, 'random_state': 42, 'n_init': 10},
                'dbscan': {'eps': 0.5, 'min_samples': 5},
                'agglomerative': {'n_clusters': 5, 'linkage': 'ward'}
            },
            'anomaly_detection': {
                'isolation_forest': {'contamination': 0.1, 'random_state': 42}
            },
            'trend_detection': {
                'min_data_points': 12,  # Minimum 1 an de données
                'significance_threshold': 0.05,
                'trend_strength_threshold': 0.3
            }
        }
        
        # Données historiques simulées
        self.historical_data = None
        self.features_data = None
        
        logger.info("📊 Analyseur de tendances ML initialisé")
    
    async def initialize(self):
        """Initialise l'analyseur avec des données et modèles pré-entraînés"""
        try:
            # Générer données historiques pour entraînement
            await self._generate_historical_datasets()
            
            # Pré-entraîner modèles de base
            await self._pretrain_models()
            
            logger.info("✅ Analyseur de tendances ML initialisé avec succès")
            
        except Exception as e:
            logger.error(f"Erreur initialisation ML trend analyzer: {e}")
            raise
    
    async def analyze_market_trends(self, 
                                  data: pd.DataFrame,
                                  scope: AnalysisScope = AnalysisScope.GLOBAL,
                                  timeframe: str = "12_months") -> TrendAnalysisResult:
        """
        Analyse complète des tendances du marché
        
        Args:
            data: Données à analyser
            scope: Portée de l'analyse
            timeframe: Période d'analyse
            
        Returns:
            TrendAnalysisResult: Résultats d'analyse
        """
        start_time = datetime.now()
        
        try:
            logger.info(f"Analyse tendances {scope.value} sur {timeframe}")
            
            # Préparer et valider les données
            processed_data = self._prepare_data_for_analysis(data)
            data_quality = self._assess_data_quality(processed_data)
            
            # Détecter patterns de tendances
            trend_patterns = await self._detect_trend_patterns(processed_data)
            
            # Clustering d'entreprises/entités
            company_clusters = await self._perform_clustering_analysis(processed_data)
            
            # Détection de cycles de marché
            market_cycles = await self._detect_market_cycles(processed_data)
            
            # Analyse de corrélations
            correlations = self._analyze_correlations(processed_data)
            
            # Détection d'anomalies
            anomalies = await self._detect_anomalies(processed_data)
            
            # Générer insights
            key_insights = self._generate_insights(
                trend_patterns, company_clusters, market_cycles, correlations, anomalies
            )
            
            # Métriques de performance
            model_performance = self._calculate_model_performance(processed_data)
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            result = TrendAnalysisResult(
                analysis_scope=scope,
                timeframe=timeframe,
                trend_patterns=trend_patterns,
                company_clusters=company_clusters,
                market_cycles=market_cycles,
                key_insights=key_insights,
                anomalies_detected=anomalies,
                correlations=correlations,
                data_quality_score=data_quality,
                processing_time_ms=processing_time,
                model_performance=model_performance,
                timestamp=datetime.now()
            )
            
            # Mettre en cache pour optimisation future
            cache_key = f"{scope.value}_{timeframe}_{hash(str(data.shape))}"
            self.cached_analyses[cache_key] = result
            
            return result
            
        except Exception as e:
            logger.error(f"Erreur analyse tendances: {e}")
            raise
    
    async def cluster_companies_by_profile(self, 
                                         companies_data: List[Dict[str, Any]],
                                         method: ClusteringMethod = ClusteringMethod.KMEANS,
                                         n_clusters: Optional[int] = None) -> List[CompanyCluster]:
        """
        Groupe les entreprises par profil similaire
        
        Args:
            companies_data: Données des entreprises
            method: Méthode de clustering
            n_clusters: Nombre de clusters (optionnel)
            
        Returns:
            List[CompanyCluster]: Clusters d'entreprises
        """
        try:
            logger.info(f"Clustering {len(companies_data)} entreprises avec {method.value}")
            
            # Convertir en DataFrame et extraire features
            df = pd.DataFrame(companies_data)
            features_matrix = self._extract_clustering_features(df)
            
            # Normaliser les features
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features_matrix)
            
            # Appliquer clustering selon la méthode
            if method == ClusteringMethod.KMEANS:
                clusters = await self._kmeans_clustering(features_scaled, n_clusters)
            elif method == ClusteringMethod.DBSCAN:
                clusters = await self._dbscan_clustering(features_scaled)
            elif method == ClusteringMethod.AGGLOMERATIVE:
                clusters = await self._agglomerative_clustering(features_scaled, n_clusters)
            else:
                raise ValueError(f"Méthode de clustering non supportée: {method}")
            
            # Créer objets CompanyCluster
            company_clusters = self._create_company_clusters(
                df, clusters, features_matrix, features_scaled
            )
            
            # Évaluer qualité du clustering
            if len(set(clusters)) > 1:
                silhouette_avg = silhouette_score(features_scaled, clusters)
                calinski_score = calinski_harabasz_score(features_scaled, clusters)
                
                logger.info(f"Qualité clustering - Silhouette: {silhouette_avg:.3f}, Calinski-Harabasz: {calinski_score:.3f}")
            
            return company_clusters
            
        except Exception as e:
            logger.error(f"Erreur clustering entreprises: {e}")
            raise
    
    async def detect_market_anomalies(self, 
                                    time_series_data: pd.DataFrame,
                                    sensitivity: float = 0.1) -> List[Dict[str, Any]]:
        """
        Détecte les anomalies dans les données de marché
        
        Args:
            time_series_data: Données temporelles
            sensitivity: Sensibilité de détection (0-1)
            
        Returns:
            List[Dict]: Anomalies détectées
        """
        try:
            anomalies = []
            
            # Isolation Forest pour anomalies globales
            if 'isolation_forest' not in self.models:
                iso_forest = IsolationForest(
                    contamination=sensitivity,
                    random_state=42
                )
                self.models['isolation_forest'] = iso_forest
            else:
                iso_forest = self.models['isolation_forest']
            
            # Préparer features pour détection d'anomalies
            features = self._prepare_anomaly_features(time_series_data)
            
            # Détecter anomalies
            anomaly_scores = iso_forest.fit_predict(features)
            anomaly_probs = iso_forest.score_samples(features)
            
            # Identifier points anormaux
            for i, (score, prob) in enumerate(zip(anomaly_scores, anomaly_probs)):
                if score == -1:  # Anomalie détectée
                    anomaly_data = {
                        'index': i,
                        'timestamp': time_series_data.index[i].isoformat() if hasattr(time_series_data.index[i], 'isoformat') else str(time_series_data.index[i]),
                        'anomaly_score': float(prob),
                        'severity': self._classify_anomaly_severity(prob),
                        'affected_metrics': self._identify_anomaly_causes(time_series_data.iloc[i], features[i]),
                        'description': self._generate_anomaly_description(time_series_data.iloc[i])
                    }
                    anomalies.append(anomaly_data)
            
            # Détecter anomalies temporelles (changements brusques)
            temporal_anomalies = self._detect_temporal_anomalies(time_series_data)
            anomalies.extend(temporal_anomalies)
            
            logger.info(f"Détecté {len(anomalies)} anomalies dans les données")
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Erreur détection anomalies: {e}")
            return []
    
    async def analyze_sector_correlations(self, 
                                        sector_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Analyse les corrélations entre secteurs
        
        Args:
            sector_data: Données par secteur
            
        Returns:
            Dict: Analyse de corrélations
        """
        try:
            logger.info(f"Analyse corrélations entre {len(sector_data)} secteurs")
            
            # Aligner les données temporellement
            aligned_data = self._align_sector_data(sector_data)
            
            # Calculer matrice de corrélation
            correlation_matrix = aligned_data.corr()
            
            # Identifier corrélations fortes
            strong_correlations = self._identify_strong_correlations(correlation_matrix)
            
            # Analyse de clusters de secteurs
            sector_clusters = self._cluster_sectors_by_correlation(correlation_matrix)
            
            # Analyse de causalité (Granger causality si possible)
            causality_analysis = self._analyze_causality(aligned_data)
            
            # Analyse de volatilité croisée
            volatility_analysis = self._analyze_cross_volatility(aligned_data)
            
            return {
                'correlation_matrix': correlation_matrix.to_dict(),
                'strong_correlations': strong_correlations,
                'sector_clusters': sector_clusters,
                'causality_relationships': causality_analysis,
                'volatility_spillovers': volatility_analysis,
                'analysis_timestamp': datetime.now().isoformat(),
                'data_coverage': {
                    sector: f"{len(data)} points"
                    for sector, data in sector_data.items()
                }
            }
            
        except Exception as e:
            logger.error(f"Erreur analyse corrélations sectorielles: {e}")
            return {'error': str(e)}
    
    async def _generate_historical_datasets(self):
        """Génère des datasets historiques pour entraînement"""
        # Période de 5 ans avec données mensuelles
        dates = pd.date_range(start='2019-01-01', end='2024-01-01', freq='M')
        np.random.seed(42)
        
        # Dataset principal - activité M&A par secteur
        sectors = ['tech', 'healthcare', 'finance', 'manufacturing', 'energy', 'retail']
        
        sector_data = {}
        for sector in sectors:
            # Tendance de base avec spécificités sectorielles
            if sector == 'tech':
                base_trend = np.linspace(100, 200, len(dates))  # Forte croissance
                seasonality = 15 * np.sin(2 * np.pi * np.arange(len(dates)) / 12)
                volatility = 0.3
            elif sector == 'healthcare':
                base_trend = np.linspace(80, 120, len(dates))  # Croissance stable
                seasonality = 8 * np.sin(2 * np.pi * np.arange(len(dates)) / 12)
                volatility = 0.15
            elif sector == 'finance':
                base_trend = np.linspace(90, 110, len(dates))  # Croissance modérée
                seasonality = 20 * np.sin(2 * np.pi * np.arange(len(dates)) / 12)
                volatility = 0.4
            else:
                base_trend = np.linspace(70, 100, len(dates))
                seasonality = 10 * np.sin(2 * np.pi * np.arange(len(dates)) / 12)
                volatility = 0.2
            
            # Bruit et chocs
            noise = np.random.normal(0, volatility * 20, len(dates))
            covid_impact = np.where((dates.year == 2020) & (dates.month.isin([3, 4, 5])), -40, 0)
            
            sector_values = base_trend + seasonality + noise + covid_impact
            sector_values = np.maximum(sector_values, 10)  # Minimum réaliste
            
            sector_data[f'{sector}_activity'] = sector_values
        
        # Créer DataFrame principal
        self.historical_data = pd.DataFrame(sector_data, index=dates)
        
        # Dataset de features d'entreprises (pour clustering)
        n_companies = 1000
        
        company_features = {
            'chiffre_affaires': np.random.lognormal(14, 1.5, n_companies),
            'croissance_ca': np.random.normal(0.05, 0.15, n_companies),
            'marge_ebitda': np.random.normal(0.12, 0.08, n_companies),
            'endettement': np.random.beta(2, 3, n_companies),
            'age_entreprise': np.random.exponential(12, n_companies),
            'effectifs': np.random.lognormal(3, 1.2, n_companies),
            'secteur_id': np.random.randint(0, len(sectors), n_companies),
            'region_id': np.random.randint(0, 13, n_companies),
            'innovation_score': np.random.beta(2, 5, n_companies),
            'presence_digitale': np.random.uniform(0, 1, n_companies)
        }
        
        self.features_data = pd.DataFrame(company_features)
        
        logger.info(f"✅ Datasets générés: {len(dates)} points temporels, {n_companies} entreprises")
    
    async def _pretrain_models(self):
        """Pré-entraîne les modèles de base"""
        if self.historical_data is None or self.features_data is None:
            await self._generate_historical_datasets()
        
        # Entraîner modèle de clustering sur features d'entreprises
        features_for_clustering = self.features_data.select_dtypes(include=[np.number])
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features_for_clustering)
        
        # K-means
        kmeans = KMeans(**self.model_configs['clustering']['kmeans'])
        kmeans.fit(features_scaled)
        
        self.models['kmeans'] = kmeans
        self.scalers['clustering'] = scaler
        
        # Isolation Forest pour détection d'anomalies
        iso_forest = IsolationForest(**self.model_configs['anomaly_detection']['isolation_forest'])
        iso_forest.fit(features_scaled)
        
        self.models['isolation_forest'] = iso_forest
        
        logger.info("✅ Modèles ML pré-entraînés")
    
    def _prepare_data_for_analysis(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prépare et nettoie les données pour l'analyse"""
        processed = data.copy()
        
        # Gestion des valeurs manquantes
        numeric_columns = processed.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            # Remplir par interpolation ou médiane
            if processed[col].isnull().sum() > 0:
                if processed.index.dtype == 'datetime64[ns]' or hasattr(processed.index, 'to_datetime'):
                    # Interpolation temporelle
                    processed[col] = processed[col].interpolate(method='time')
                else:
                    # Remplissage par médiane
                    processed[col] = processed[col].fillna(processed[col].median())
        
        # Suppression des outliers extrêmes (>3 sigma)
        for col in numeric_columns:
            mean = processed[col].mean()
            std = processed[col].std()
            
            outlier_mask = np.abs(processed[col] - mean) > 3 * std
            if outlier_mask.sum() > 0:
                # Remplacer par valeur plafonnée
                lower_bound = mean - 3 * std
                upper_bound = mean + 3 * std
                processed[col] = processed[col].clip(lower_bound, upper_bound)
        
        return processed
    
    def _assess_data_quality(self, data: pd.DataFrame) -> float:
        """Évalue la qualité des données"""
        quality_metrics = []
        
        # Complétude
        completeness = (1 - data.isnull().sum().sum() / (data.shape[0] * data.shape[1]))
        quality_metrics.append(completeness)
        
        # Cohérence temporelle
        if hasattr(data.index, 'to_datetime') or data.index.dtype == 'datetime64[ns]':
            time_consistency = 1.0 - (data.index.duplicated().sum() / len(data.index))
            quality_metrics.append(time_consistency)
        
        # Variance des données (éviter données trop statiques)
        numeric_data = data.select_dtypes(include=[np.number])
        if not numeric_data.empty:
            variance_scores = []
            for col in numeric_data.columns:
                cv = numeric_data[col].std() / (numeric_data[col].mean() + 1e-8)  # Coefficient de variation
                variance_score = min(1.0, cv / 0.5)  # Normaliser
                variance_scores.append(variance_score)
            
            avg_variance = np.mean(variance_scores)
            quality_metrics.append(avg_variance)
        
        return np.mean(quality_metrics)
    
    async def _detect_trend_patterns(self, data: pd.DataFrame) -> List[TrendPattern]:
        """Détecte les patterns de tendances dans les données"""
        patterns = []
        
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        
        for column in numeric_columns:
            series = data[column].dropna()
            
            if len(series) < self.model_configs['trend_detection']['min_data_points']:
                continue
            
            # Décomposition temporelle (si données temporelles)
            if STATSMODELS_AVAILABLE and hasattr(data.index, 'freq') or len(series) >= 24:
                try:
                    decomposition = seasonal_decompose(series, model='additive', period=min(12, len(series)//2))
                    
                    # Analyser trend component
                    trend_component = decomposition.trend.dropna()
                    
                    if len(trend_component) > 0:
                        # Test de stationnarité
                        try:
                            adf_stat, adf_pvalue = adfuller(trend_component)[:2]
                            is_stationary = adf_pvalue < 0.05
                        except:
                            is_stationary = False
                        
                        # Classification de tendance
                        if not is_stationary:
                            # Tendance présente
                            slope = np.polyfit(range(len(trend_component)), trend_component.values, 1)[0]
                            
                            if abs(slope) > np.std(trend_component) * 0.1:
                                if slope > 0:
                                    trend_type = TrendType.LINEAR if slope < np.mean(trend_component) * 0.02 else TrendType.EXPONENTIAL
                                    description = f"Tendance haussière dans {column}"
                                else:
                                    trend_type = TrendType.LINEAR
                                    description = f"Tendance baissière dans {column}"
                                
                                strength = min(1.0, abs(slope) / np.std(trend_component))
                                
                                patterns.append(TrendPattern(
                                    trend_type=trend_type,
                                    strength=strength,
                                    period_months=None,
                                    confidence=1.0 - adf_pvalue,
                                    description=description,
                                    statistical_significance=1.0 - adf_pvalue,
                                    parameters={'slope': slope, 'adf_pvalue': adf_pvalue}
                                ))
                        
                        # Analyser saisonnalité
                        seasonal_component = decomposition.seasonal.dropna()
                        if len(seasonal_component) > 0 and np.std(seasonal_component) > np.std(series) * 0.1:
                            seasonal_strength = np.std(seasonal_component) / np.std(series)
                            
                            patterns.append(TrendPattern(
                                trend_type=TrendType.SEASONAL,
                                strength=seasonal_strength,
                                period_months=12,  # Assumé annuel
                                confidence=0.8,
                                description=f"Pattern saisonnier dans {column}",
                                statistical_significance=seasonal_strength,
                                parameters={'seasonal_amplitude': np.std(seasonal_component)}
                            ))
                        
                        # Analyser volatilité
                        residual_component = decomposition.resid.dropna()
                        if len(residual_component) > 0:
                            volatility = np.std(residual_component) / np.mean(np.abs(series))
                            
                            if volatility > 0.3:
                                patterns.append(TrendPattern(
                                    trend_type=TrendType.VOLATILE,
                                    strength=min(1.0, volatility),
                                    period_months=None,
                                    confidence=0.7,
                                    description=f"Forte volatilité dans {column}",
                                    statistical_significance=volatility,
                                    parameters={'volatility_ratio': volatility}
                                ))
                
                except Exception as e:
                    logger.warning(f"Erreur décomposition temporelle pour {column}: {e}")
            
            # Analyse de patterns simples
            simple_patterns = self._detect_simple_patterns(series, column)
            patterns.extend(simple_patterns)
        
        return patterns
    
    def _detect_simple_patterns(self, series: pd.Series, column_name: str) -> List[TrendPattern]:
        """Détecte des patterns simples dans une série"""
        patterns = []
        
        if len(series) < 3:
            return patterns
        
        # Stabilité
        cv = series.std() / (series.mean() + 1e-8)
        if cv < 0.1:
            patterns.append(TrendPattern(
                trend_type=TrendType.STABLE,
                strength=1.0 - cv,
                period_months=None,
                confidence=0.9,
                description=f"Série stable: {column_name}",
                statistical_significance=1.0 - cv,
                parameters={'coefficient_variation': cv}
            ))
        
        # Tendance linéaire simple
        x = np.arange(len(series))
        slope, intercept = np.polyfit(x, series.values, 1)
        r_squared = np.corrcoef(x, series.values)[0, 1] ** 2
        
        if r_squared > 0.7 and abs(slope) > series.std() * 0.01:
            patterns.append(TrendPattern(
                trend_type=TrendType.LINEAR,
                strength=r_squared,
                period_months=None,
                confidence=r_squared,
                description=f"Tendance linéaire {'croissante' if slope > 0 else 'décroissante'}: {column_name}",
                statistical_significance=r_squared,
                parameters={'slope': slope, 'r_squared': r_squared}
            ))
        
        return patterns
    
    async def _perform_clustering_analysis(self, data: pd.DataFrame) -> List[CompanyCluster]:
        """Effectue une analyse de clustering sur les données"""
        numeric_data = data.select_dtypes(include=[np.number])
        
        if numeric_data.empty or len(numeric_data) < 5:
            return []
        
        # Normaliser les données
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(numeric_data)
        
        # Déterminer nombre optimal de clusters
        optimal_k = self._find_optimal_clusters(data_scaled)
        
        # Appliquer K-means
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(data_scaled)
        
        # Créer objets CompanyCluster
        clusters = []
        for cluster_id in range(optimal_k):
            cluster_mask = cluster_labels == cluster_id
            cluster_indices = np.where(cluster_mask)[0]
            
            if len(cluster_indices) == 0:
                continue
            
            # Centroïde du cluster
            centroid = kmeans.cluster_centers_[cluster_id]
            centroid_original = scaler.inverse_transform(centroid.reshape(1, -1))[0]
            centroid_features = dict(zip(numeric_data.columns, centroid_original))
            
            # Caractéristiques du cluster
            cluster_data = numeric_data.iloc[cluster_indices]
            characteristics = self._describe_cluster_characteristics(cluster_data, centroid_features)
            
            # Cohésion du cluster
            cluster_points = data_scaled[cluster_mask]
            cohesion = 1.0 / (1.0 + np.mean([np.linalg.norm(point - centroid) for point in cluster_points]))
            
            cluster = CompanyCluster(
                cluster_id=cluster_id,
                cluster_name=f"Cluster {cluster_id + 1}",
                companies=[f"entity_{i}" for i in cluster_indices],  # IDs génériques
                centroid_features=centroid_features,
                cluster_characteristics=characteristics,
                size=len(cluster_indices),
                cohesion_score=cohesion
            )
            
            clusters.append(cluster)
        
        return clusters
    
    def _find_optimal_clusters(self, data: np.ndarray) -> int:
        """Trouve le nombre optimal de clusters"""
        max_k = min(10, len(data) // 2)
        
        if max_k < 2:
            return 2
        
        silhouette_scores = []
        k_range = range(2, max_k + 1)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(data)
            score = silhouette_score(data, labels)
            silhouette_scores.append(score)
        
        # Retourner k avec meilleur score silhouette
        optimal_k = k_range[np.argmax(silhouette_scores)]
        return optimal_k
    
    def _describe_cluster_characteristics(self, cluster_data: pd.DataFrame, centroid: Dict[str, float]) -> List[str]:
        """Décrit les caractéristiques d'un cluster"""
        characteristics = []
        
        # Analyser chaque feature
        for feature, centroid_value in centroid.items():
            if feature in cluster_data.columns:
                # Comparer à la moyenne globale
                global_mean = cluster_data[feature].mean()
                
                if centroid_value > global_mean * 1.2:
                    characteristics.append(f"Forte {feature}")
                elif centroid_value < global_mean * 0.8:
                    characteristics.append(f"Faible {feature}")
        
        # Taille du cluster
        if len(cluster_data) > len(cluster_data.index) * 0.3:
            characteristics.append("Cluster dominant")
        elif len(cluster_data) < len(cluster_data.index) * 0.1:
            characteristics.append("Cluster niche")
        
        return characteristics[:5]  # Limiter à 5 caractéristiques
    
    async def _detect_market_cycles(self, data: pd.DataFrame) -> List[MarketCycle]:
        """Détecte les cycles de marché"""
        cycles = []
        
        # Pour cette version, implémenter détection simple basée sur périodicité
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        
        for column in numeric_columns:
            series = data[column].dropna()
            
            if len(series) < 24:  # Minimum 2 ans
                continue
            
            # Recherche de périodicité via autocorrélation
            autocorr_scores = []
            periods = range(6, min(36, len(series) // 2))  # 6 mois à 3 ans
            
            for period in periods:
                if period < len(series):
                    # Autocorrélation à lag=period
                    autocorr = np.corrcoef(series[:-period], series[period:])[0, 1]
                    if not np.isnan(autocorr):
                        autocorr_scores.append((period, abs(autocorr)))
            
            if autocorr_scores:
                # Trouver période avec meilleure autocorrélation
                best_period, best_score = max(autocorr_scores, key=lambda x: x[1])
                
                if best_score > 0.3:  # Seuil de significativité
                    # Identifier phase actuelle
                    current_phase = self._identify_current_phase(series, best_period)
                    
                    cycle = MarketCycle(
                        cycle_name=f"Cycle {column}",
                        duration_months=best_period,
                        phases=["Expansion", "Pic", "Contraction", "Creux"],
                        current_phase=current_phase,
                        phase_duration_remaining=best_period // 4,  # Approximation
                        confidence=best_score,
                        historical_patterns={
                            'amplitude': float(series.std()),
                            'period': best_period,
                            'autocorr_score': best_score
                        }
                    )
                    
                    cycles.append(cycle)
        
        return cycles
    
    def _identify_current_phase(self, series: pd.Series, period: int) -> str:
        """Identifie la phase actuelle du cycle"""
        recent_values = series.tail(period // 4)
        recent_trend = np.polyfit(range(len(recent_values)), recent_values.values, 1)[0]
        
        current_value = series.iloc[-1]
        series_mean = series.mean()
        
        if recent_trend > 0 and current_value < series_mean:
            return "Expansion"
        elif recent_trend > 0 and current_value >= series_mean:
            return "Pic"
        elif recent_trend < 0 and current_value > series_mean:
            return "Contraction"
        else:
            return "Creux"
    
    def _analyze_correlations(self, data: pd.DataFrame) -> Dict[str, float]:
        """Analyse les corrélations dans les données"""
        numeric_data = data.select_dtypes(include=[np.number])
        
        if numeric_data.shape[1] < 2:
            return {}
        
        # Matrice de corrélation
        corr_matrix = numeric_data.corr()
        
        # Extraire corrélations significatives
        significant_correlations = {}
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                col1 = corr_matrix.columns[i]
                col2 = corr_matrix.columns[j]
                corr_value = corr_matrix.iloc[i, j]
                
                if not np.isnan(corr_value) and abs(corr_value) > 0.3:
                    pair_name = f"{col1}_vs_{col2}"
                    significant_correlations[pair_name] = corr_value
        
        return significant_correlations
    
    async def _detect_anomalies(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Détecte les anomalies dans les données"""
        anomalies = []
        
        numeric_data = data.select_dtypes(include=[np.number])
        
        if numeric_data.empty:
            return anomalies
        
        # Isolation Forest si pas déjà entraîné
        if 'isolation_forest' not in self.models:
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            self.models['isolation_forest'] = iso_forest
        else:
            iso_forest = self.models['isolation_forest']
        
        # Normaliser données
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(numeric_data)
        
        # Détecter anomalies
        try:
            anomaly_labels = iso_forest.fit_predict(data_scaled)
            anomaly_scores = iso_forest.score_samples(data_scaled)
            
            # Identifier points anormaux
            for i, (label, score) in enumerate(zip(anomaly_labels, anomaly_scores)):
                if label == -1:  # Anomalie
                    anomaly = {
                        'index': i,
                        'anomaly_score': float(score),
                        'severity': self._classify_anomaly_severity(score),
                        'affected_columns': self._identify_anomalous_features(data_scaled[i], scaler, numeric_data.columns),
                        'description': f"Anomalie détectée à l'index {i}"
                    }
                    anomalies.append(anomaly)
        
        except Exception as e:
            logger.warning(f"Erreur détection anomalies: {e}")
        
        return anomalies
    
    def _classify_anomaly_severity(self, score: float) -> str:
        """Classifie la sévérité d'une anomalie"""
        if score < -0.5:
            return "Critique"
        elif score < -0.3:
            return "Élevée"
        elif score < -0.1:
            return "Modérée"
        else:
            return "Faible"
    
    def _identify_anomalous_features(self, anomaly_point: np.ndarray, scaler: StandardScaler, column_names: List[str]) -> List[str]:
        """Identifie quelles features contribuent à l'anomalie"""
        # Transformer le point vers l'espace original
        original_point = scaler.inverse_transform(anomaly_point.reshape(1, -1))[0]
        
        # Identifier features les plus extrêmes
        z_scores = np.abs(anomaly_point)
        extreme_indices = np.where(z_scores > 2.0)[0]  # Plus de 2 écarts-types
        
        return [column_names[i] for i in extreme_indices]
    
    def _generate_insights(self, trend_patterns: List[TrendPattern], 
                         company_clusters: List[CompanyCluster],
                         market_cycles: List[MarketCycle],
                         correlations: Dict[str, float],
                         anomalies: List[Dict[str, Any]]) -> List[str]:
        """Génère des insights clés à partir de l'analyse"""
        insights = []
        
        # Insights sur tendances
        strong_trends = [tp for tp in trend_patterns if tp.strength > 0.7]
        if strong_trends:
            trends_desc = ", ".join([tp.trend_type.value for tp in strong_trends[:3]])
            insights.append(f"Tendances fortes détectées: {trends_desc}")
        
        # Insights sur clustering
        if company_clusters:
            largest_cluster = max(company_clusters, key=lambda c: c.size)
            insights.append(f"Cluster dominant: {largest_cluster.cluster_name} ({largest_cluster.size} entités)")
        
        # Insights sur cycles
        if market_cycles:
            current_phases = [mc.current_phase for mc in market_cycles]
            most_common_phase = max(set(current_phases), key=current_phases.count)
            insights.append(f"Phase de marché dominante: {most_common_phase}")
        
        # Insights sur corrélations
        strong_correlations = {k: v for k, v in correlations.items() if abs(v) > 0.7}
        if strong_correlations:
            insights.append(f"Corrélations fortes détectées entre {len(strong_correlations)} paires de variables")
        
        # Insights sur anomalies
        if anomalies:
            critical_anomalies = [a for a in anomalies if a['severity'] == 'Critique']
            if critical_anomalies:
                insights.append(f"{len(critical_anomalies)} anomalies critiques nécessitent une attention immédiate")
        
        # Insight général sur qualité des données
        if len(trend_patterns) > 5:
            insights.append("Données riches permettant une analyse approfondie des tendances")
        
        return insights[:5]  # Limiter à 5 insights principaux
    
    def _calculate_model_performance(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calcule les métriques de performance des modèles"""
        performance = {}
        
        # Performance clustering (si applicable)
        if 'kmeans' in self.models and not data.empty:
            try:
                numeric_data = data.select_dtypes(include=[np.number])
                if not numeric_data.empty and len(numeric_data) > 5:
                    scaler = StandardScaler()
                    data_scaled = scaler.fit_transform(numeric_data)
                    
                    labels = self.models['kmeans'].predict(data_scaled)
                    if len(set(labels)) > 1:
                        silhouette = silhouette_score(data_scaled, labels)
                        performance['clustering_silhouette'] = silhouette
            except:
                pass
        
        # Performance détection anomalies
        if 'isolation_forest' in self.models:
            performance['anomaly_detection_trained'] = 1.0
        
        # Complétude de l'analyse
        analysis_completeness = 0.0
        if hasattr(self, 'trend_patterns'):
            analysis_completeness += 0.25
        if hasattr(self, 'company_clusters'):
            analysis_completeness += 0.25
        if hasattr(self, 'market_cycles'):
            analysis_completeness += 0.25
        if hasattr(self, 'correlations'):
            analysis_completeness += 0.25
        
        performance['analysis_completeness'] = analysis_completeness
        
        return performance
    
    # Méthodes pour clustering spécialisé
    
    async def _kmeans_clustering(self, features: np.ndarray, n_clusters: Optional[int] = None) -> np.ndarray:
        """Clustering K-means"""
        if n_clusters is None:
            n_clusters = self._find_optimal_clusters(features)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        return kmeans.fit_predict(features)
    
    async def _dbscan_clustering(self, features: np.ndarray) -> np.ndarray:
        """Clustering DBSCAN"""
        dbscan = DBSCAN(**self.model_configs['clustering']['dbscan'])
        return dbscan.fit_predict(features)
    
    async def _agglomerative_clustering(self, features: np.ndarray, n_clusters: Optional[int] = None) -> np.ndarray:
        """Clustering hiérarchique agglomératif"""
        if n_clusters is None:
            n_clusters = self._find_optimal_clusters(features)
        
        agg_clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
        return agg_clustering.fit_predict(features)
    
    def _extract_clustering_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extrait features pour clustering"""
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_columns) == 0:
            # Si pas de colonnes numériques, créer features par défaut
            return np.random.randn(len(df), 5)
        
        return df[numeric_columns].fillna(df[numeric_columns].median()).values
    
    def _create_company_clusters(self, df: pd.DataFrame, clusters: np.ndarray, 
                               features: np.ndarray, features_scaled: np.ndarray) -> List[CompanyCluster]:
        """Crée les objets CompanyCluster"""
        company_clusters = []
        
        unique_clusters = np.unique(clusters)
        
        for cluster_id in unique_clusters:
            if cluster_id == -1:  # Bruit (DBSCAN)
                continue
            
            cluster_mask = clusters == cluster_id
            cluster_indices = np.where(cluster_mask)[0]
            
            if len(cluster_indices) == 0:
                continue
            
            # Centroïde
            cluster_features = features[cluster_mask]
            centroid = np.mean(cluster_features, axis=0)
            
            # Mapping vers noms de colonnes
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) == len(centroid):
                centroid_features = dict(zip(numeric_columns, centroid))
            else:
                centroid_features = {f'feature_{i}': val for i, val in enumerate(centroid)}
            
            # Caractéristiques
            characteristics = [f"Cluster de {len(cluster_indices)} éléments"]
            
            # Cohésion
            cluster_scaled = features_scaled[cluster_mask]
            centroid_scaled = np.mean(cluster_scaled, axis=0)
            distances = [np.linalg.norm(point - centroid_scaled) for point in cluster_scaled]
            cohesion = 1.0 / (1.0 + np.mean(distances))
            
            cluster = CompanyCluster(
                cluster_id=int(cluster_id),
                cluster_name=f"Groupe {cluster_id + 1}",
                companies=[f"entity_{i}" for i in cluster_indices],
                centroid_features=centroid_features,
                cluster_characteristics=characteristics,
                size=len(cluster_indices),
                cohesion_score=cohesion
            )
            
            company_clusters.append(cluster)
        
        return company_clusters
    
    # Méthodes utilitaires pour analyses avancées
    
    def _prepare_anomaly_features(self, data: pd.DataFrame) -> np.ndarray:
        """Prépare features pour détection d'anomalies"""
        numeric_data = data.select_dtypes(include=[np.number])
        
        if numeric_data.empty:
            return np.random.randn(len(data), 3)
        
        # Ajouter features dérivées
        features_list = [numeric_data.values]
        
        # Moyennes mobiles
        for col in numeric_data.columns:
            if len(numeric_data) > 3:
                rolling_mean = numeric_data[col].rolling(window=min(3, len(numeric_data))).mean()
                features_list.append(rolling_mean.fillna(numeric_data[col].mean()).values.reshape(-1, 1))
        
        return np.hstack(features_list)
    
    def _identify_anomaly_causes(self, data_point: pd.Series, feature_vector: np.ndarray) -> List[str]:
        """Identifie les causes potentielles d'une anomalie"""
        causes = []
        
        # Analyser les valeurs extrêmes
        for col, value in data_point.items():
            if pd.notnull(value) and isinstance(value, (int, float)):
                # Comparer à la distribution
                if abs(value) > 1000000:  # Valeur très élevée
                    causes.append(f"Valeur extrême: {col}")
        
        return causes[:3]  # Limiter à 3 causes
    
    def _generate_anomaly_description(self, data_point: pd.Series) -> str:
        """Génère une description d'anomalie"""
        return f"Point de données inhabituel détecté"
    
    def _detect_temporal_anomalies(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Détecte anomalies temporelles (changements brusques)"""
        anomalies = []
        
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            series = data[col].dropna()
            
            if len(series) < 3:
                continue
            
            # Détecter changements brusques (différences importantes)
            diffs = np.abs(np.diff(series.values))
            threshold = np.mean(diffs) + 2 * np.std(diffs)
            
            abrupt_changes = np.where(diffs > threshold)[0]
            
            for change_idx in abrupt_changes:
                anomaly = {
                    'type': 'temporal_anomaly',
                    'index': change_idx + 1,  # Index après le changement
                    'column': col,
                    'change_magnitude': float(diffs[change_idx]),
                    'description': f"Changement brusque dans {col}",
                    'severity': 'Modérée'
                }
                anomalies.append(anomaly)
        
        return anomalies
    
    def _align_sector_data(self, sector_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Aligne temporellement les données sectorielles"""
        # Prendre première série comme référence
        reference_index = None
        aligned_data = {}
        
        for sector, data in sector_data.items():
            if reference_index is None:
                reference_index = data.index
            
            # Réindexer sur l'index de référence
            if hasattr(data, 'reindex'):
                aligned_series = data.reindex(reference_index).interpolate()
            else:
                # Si c'est juste une série de valeurs
                aligned_series = pd.Series(data.values[:len(reference_index)], index=reference_index[:len(data)])
            
            # Prendre première colonne numérique ou valeurs
            if hasattr(aligned_series, 'select_dtypes'):
                numeric_cols = aligned_series.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    aligned_data[sector] = aligned_series[numeric_cols[0]]
                else:
                    aligned_data[sector] = aligned_series.iloc[:, 0] if len(aligned_series.columns) > 0 else aligned_series
            else:
                aligned_data[sector] = aligned_series
        
        return pd.DataFrame(aligned_data)
    
    def _identify_strong_correlations(self, correlation_matrix: pd.DataFrame) -> List[Dict[str, Any]]:
        """Identifie les corrélations fortes"""
        strong_correlations = []
        
        for i in range(len(correlation_matrix.columns)):
            for j in range(i + 1, len(correlation_matrix.columns)):
                corr_value = correlation_matrix.iloc[i, j]
                
                if not np.isnan(corr_value) and abs(corr_value) > 0.6:
                    strong_correlations.append({
                        'sector_1': correlation_matrix.columns[i],
                        'sector_2': correlation_matrix.columns[j],
                        'correlation': float(corr_value),
                        'strength': 'Forte' if abs(corr_value) > 0.8 else 'Modérée',
                        'direction': 'Positive' if corr_value > 0 else 'Négative'
                    })
        
        return sorted(strong_correlations, key=lambda x: abs(x['correlation']), reverse=True)
    
    def _cluster_sectors_by_correlation(self, correlation_matrix: pd.DataFrame) -> List[Dict[str, Any]]:
        """Groupe les secteurs par corrélation"""
        if len(correlation_matrix) < 2:
            return []
        
        # Utiliser clustering hiérarchique sur matrice de corrélation
        distance_matrix = 1 - np.abs(correlation_matrix.values)
        
        try:
            from scipy.cluster.hierarchy import linkage, fcluster
            from scipy.spatial.distance import squareform
            
            # Convertir en matrice de distance condensée
            condensed_distances = squareform(distance_matrix)
            
            # Clustering hiérarchique
            linkage_matrix = linkage(condensed_distances, method='ward')
            
            # Former clusters
            cluster_labels = fcluster(linkage_matrix, t=0.5, criterion='distance')
            
            # Regrouper secteurs par cluster
            sector_clusters = defaultdict(list)
            for sector, cluster_id in zip(correlation_matrix.columns, cluster_labels):
                sector_clusters[cluster_id].append(sector)
            
            return [
                {
                    'cluster_id': cluster_id,
                    'sectors': sectors,
                    'size': len(sectors)
                }
                for cluster_id, sectors in sector_clusters.items()
                if len(sectors) > 1  # Garder seulement clusters multi-secteurs
            ]
            
        except ImportError:
            # Fallback simple si scipy non disponible
            return [{'cluster_id': 1, 'sectors': list(correlation_matrix.columns), 'size': len(correlation_matrix.columns)}]
    
    def _analyze_causality(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Analyse de causalité entre séries (version simplifiée)"""
        causality_results = []
        
        # Test simple de lead-lag relationships
        columns = data.columns
        
        for i, col1 in enumerate(columns):
            for j, col2 in enumerate(columns):
                if i != j:
                    series1 = data[col1].dropna()
                    series2 = data[col2].dropna()
                    
                    if len(series1) > 5 and len(series2) > 5:
                        # Corr avec lag
                        min_len = min(len(series1), len(series2))
                        s1 = series1.iloc[:min_len-1].values
                        s2_lag = series2.iloc[1:min_len].values
                        
                        if len(s1) > 0 and len(s2_lag) > 0:
                            lag_corr = np.corrcoef(s1, s2_lag)[0, 1]
                            
                            if not np.isnan(lag_corr) and abs(lag_corr) > 0.4:
                                causality_results.append({
                                    'cause': col1,
                                    'effect': col2,
                                    'lag_correlation': float(lag_corr),
                                    'strength': 'Forte' if abs(lag_corr) > 0.6 else 'Modérée'
                                })
        
        return sorted(causality_results, key=lambda x: abs(x['lag_correlation']), reverse=True)[:5]
    
    def _analyze_cross_volatility(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyse de volatilité croisée"""
        # Calculer volatilités (écart-type mobile)
        volatilities = {}
        
        for col in data.columns:
            series = data[col].dropna()
            if len(series) > 5:
                rolling_std = series.rolling(window=min(5, len(series))).std()
                volatilities[col] = rolling_std.dropna()
        
        if len(volatilities) < 2:
            return {'volatility_spillovers': []}
        
        # Analyser corrélations entre volatilités
        vol_data = pd.DataFrame(volatilities)
        vol_corr = vol_data.corr()
        
        spillovers = []
        for i in range(len(vol_corr.columns)):
            for j in range(i + 1, len(vol_corr.columns)):
                corr_value = vol_corr.iloc[i, j]
                
                if not np.isnan(corr_value) and abs(corr_value) > 0.3:
                    spillovers.append({
                        'from_sector': vol_corr.columns[i],
                        'to_sector': vol_corr.columns[j],
                        'volatility_correlation': float(corr_value),
                        'interpretation': 'Contagion de volatilité' if corr_value > 0 else 'Volatilité inversée'
                    })
        
        return {
            'volatility_spillovers': spillovers,
            'average_volatility': {
                col: float(vol.mean()) for col, vol in volatilities.items()
            }
        }


# Instance globale
_ml_trend_analyzer: Optional[MLTrendAnalyzer] = None


async def get_ml_trend_analyzer() -> MLTrendAnalyzer:
    """Factory pour obtenir l'instance de l'analyseur de tendances ML"""
    global _ml_trend_analyzer
    
    if _ml_trend_analyzer is None:
        _ml_trend_analyzer = MLTrendAnalyzer()
        await _ml_trend_analyzer.initialize()
    
    return _ml_trend_analyzer


async def initialize_ml_trend_analyzer():
    """Initialise le système d'analyse de tendances ML au démarrage"""
    try:
        analyzer = await get_ml_trend_analyzer()
        logger.info("📊 Système d'analyse de tendances ML initialisé avec succès")
        return analyzer
    except Exception as e:
        logger.error(f"Erreur initialisation analyseur ML: {e}")
        raise