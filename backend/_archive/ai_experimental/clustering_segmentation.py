"""
Système de clustering automatique et segmentation intelligente
US-010: Segmentation automatique des entreprises et clients pour M&A Intelligence Platform

Ce module fournit:
- Clustering automatique multi-dimensionnel des entreprises
- Segmentation intelligente des clients/prospects
- Analyse de cohortes et comportements
- Profiling automatique des segments
- Recommandations de ciblage
- Clustering hiérarchique et dynamique
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from functools import wraps
from enum import Enum
import json
import time
from collections import defaultdict, Counter
import math

# Machine Learning pour clustering
from sklearn.cluster import (
    KMeans, DBSCAN, AgglomerativeClustering, 
    SpectralClustering, MeanShift
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA, TSNE, UMAP
from sklearn.manifold import TSNE as TSNEManifold
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.feature_selection import SelectKBest, f_classif

# Clustering spécialisé
try:
    import umap.umap_ as umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

# Visualisation
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from app.core.logging_system import get_logger, LogCategory
from app.core.cache_manager import get_cache_manager, cached

logger = get_logger("clustering_segmentation", LogCategory.AI_ML)


class ClusteringMethod(str, Enum):
    """Méthodes de clustering"""
    KMEANS = "kmeans"
    DBSCAN = "dbscan"
    HIERARCHICAL = "hierarchical"
    GAUSSIAN_MIXTURE = "gaussian_mixture"
    SPECTRAL = "spectral"
    MEAN_SHIFT = "mean_shift"
    ENSEMBLE = "ensemble"


class SegmentType(str, Enum):
    """Types de segments"""
    COMPANY_PROFILE = "company_profile"
    CLIENT_BEHAVIOR = "client_behavior"
    ACQUISITION_TARGET = "acquisition_target"
    INDUSTRY_SEGMENT = "industry_segment"
    GEOGRAPHIC_SEGMENT = "geographic_segment"
    SIZE_SEGMENT = "size_segment"


class SegmentPriority(str, Enum):
    """Priorité des segments"""
    VERY_HIGH = "very_high"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class ClusteringResult:
    """Résultat d'un clustering"""
    cluster_id: str
    method: ClusteringMethod
    n_clusters: int
    cluster_labels: List[int]
    cluster_centers: Optional[np.ndarray]
    
    # Métriques de qualité
    silhouette_score: float
    calinski_harabasz_score: float
    davies_bouldin_score: float
    inertia: Optional[float]
    
    # Métadonnées
    feature_names: List[str]
    n_samples: int
    training_time: float
    
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ClusterProfile:
    """Profil d'un cluster"""
    cluster_id: int
    cluster_label: str
    cluster_description: str
    
    # Statistiques
    size: int
    percentage: float
    
    # Caractéristiques moyennes
    avg_features: Dict[str, float]
    dominant_features: List[Tuple[str, float]]
    
    # Ranges des features
    feature_ranges: Dict[str, Tuple[float, float]]
    
    # Entreprises représentatives
    representative_companies: List[str]
    
    # Score d'attractivité M&A
    ma_attractiveness_score: float
    
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class Segment:
    """Segment d'entreprises ou clients"""
    segment_id: str
    segment_name: str
    segment_type: SegmentType
    priority: SegmentPriority
    
    # Composition
    entities: List[str]  # SIRENs ou user_ids
    size: int
    
    # Caractéristiques
    key_characteristics: Dict[str, Any]
    behavioral_patterns: Dict[str, Any]
    
    # Potentiel business
    revenue_potential: float
    acquisition_probability: float
    conversion_rate: float
    
    # Stratégies recommandées
    recommended_strategies: List[str]
    communication_preferences: Dict[str, Any]
    
    # Performance historique
    historical_performance: Dict[str, float]
    
    # Métadonnées
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)


class FeatureEngineering:
    """Ingénierie de features pour clustering"""
    
    def __init__(self):
        self.scalers: Dict[str, Any] = {}
        self.feature_selectors: Dict[str, Any] = {}
        self.dimensionality_reducers: Dict[str, Any] = {}
        
    def engineer_company_features(self, companies_df: pd.DataFrame) -> pd.DataFrame:
        """Ingénierie de features pour les entreprises"""
        
        logger.info("🔧 Ingénierie de features entreprises")
        
        enhanced_df = companies_df.copy()
        
        # Features financières
        if 'chiffre_affaires' in enhanced_df.columns:
            enhanced_df['ca_log'] = np.log1p(enhanced_df['chiffre_affaires'].fillna(0))
            enhanced_df['ca_category'] = pd.cut(
                enhanced_df['chiffre_affaires'].fillna(0),
                bins=[0, 100000, 1000000, 10000000, float('inf')],
                labels=['micro', 'petite', 'moyenne', 'grande']
            ).astype(str)
        
        if 'effectifs' in enhanced_df.columns:
            enhanced_df['effectifs_log'] = np.log1p(enhanced_df['effectifs'].fillna(0))
            enhanced_df['productivity'] = (
                enhanced_df['chiffre_affaires'].fillna(0) / 
                enhanced_df['effectifs'].fillna(1).replace(0, 1)
            )
        
        # Features temporelles
        if 'date_creation' in enhanced_df.columns:
            enhanced_df['date_creation'] = pd.to_datetime(enhanced_df['date_creation'], errors='coerce')
            enhanced_df['company_age'] = (
                datetime.now() - enhanced_df['date_creation']
            ).dt.days / 365.25
            enhanced_df['is_young_company'] = (enhanced_df['company_age'] < 5).astype(int)
        
        # Features géographiques
        if 'adresse' in enhanced_df.columns:
            enhanced_df['is_paris'] = enhanced_df['adresse'].str.contains('Paris', na=False).astype(int)
            enhanced_df['is_idf'] = enhanced_df['adresse'].str.contains(
                'Paris|Hauts-de-Seine|Seine-Saint-Denis|Val-de-Marne|Seine-et-Marne|Yvelines|Essonne|Val-d\'Oise', 
                na=False
            ).astype(int)
        
        # Features sectorielles
        if 'secteur_activite' in enhanced_df.columns:
            # Encoding secteur
            sector_dummies = pd.get_dummies(enhanced_df['secteur_activite'], prefix='secteur')
            enhanced_df = pd.concat([enhanced_df, sector_dummies], axis=1)
            
            # Secteurs stratégiques
            strategic_sectors = ['tech', 'digital', 'innovation', 'santé', 'environnement']
            enhanced_df['is_strategic_sector'] = enhanced_df['secteur_activite'].str.lower().str.contains(
                '|'.join(strategic_sectors), na=False
            ).astype(int)
        
        # Features de croissance
        numeric_cols = enhanced_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col.endswith('_historique') or 'evolution' in col.lower():
                enhanced_df[f'{col}_growth_rate'] = enhanced_df[col].pct_change().fillna(0)
        
        # Features de risque (basées sur des heuristiques)
        enhanced_df['financial_risk_score'] = 0
        if 'chiffre_affaires' in enhanced_df.columns:
            enhanced_df['financial_risk_score'] += (
                enhanced_df['chiffre_affaires'] < 100000
            ).astype(int) * 30
        
        if 'effectifs' in enhanced_df.columns:
            enhanced_df['financial_risk_score'] += (
                enhanced_df['effectifs'] < 5
            ).astype(int) * 20
        
        # Features d'attractivité M&A
        enhanced_df['ma_attractiveness'] = 0
        
        # Bonus taille optimale
        if 'chiffre_affaires' in enhanced_df.columns:
            ca = enhanced_df['chiffre_affaires'].fillna(0)
            enhanced_df['ma_attractiveness'] += np.where(
                (ca > 1000000) & (ca < 50000000), 30, 0
            )
        
        # Bonus secteur stratégique
        enhanced_df['ma_attractiveness'] += enhanced_df.get('is_strategic_sector', 0) * 25
        
        # Bonus localisation
        enhanced_df['ma_attractiveness'] += enhanced_df.get('is_idf', 0) * 15
        
        # Bonus croissance
        if 'productivity' in enhanced_df.columns:
            enhanced_df['ma_attractiveness'] += np.where(
                enhanced_df['productivity'] > enhanced_df['productivity'].quantile(0.75),
                20, 0
            )
        
        logger.info(f"✅ Features générées: {len(enhanced_df.columns)} colonnes")
        
        return enhanced_df
    
    def prepare_clustering_features(
        self, 
        df: pd.DataFrame, 
        feature_set: str = "default",
        n_components: Optional[int] = None
    ) -> Tuple[np.ndarray, List[str]]:
        """Prépare les features pour clustering"""
        
        # Sélection des features numériques
        numeric_features = df.select_dtypes(include=[np.number]).fillna(0)
        
        # Suppression des colonnes avec variance nulle
        numeric_features = numeric_features.loc[:, numeric_features.var() != 0]
        
        feature_names = numeric_features.columns.tolist()
        
        # Normalisation
        if feature_set not in self.scalers:
            self.scalers[feature_set] = RobustScaler()  # Résistant aux outliers
        
        scaled_features = self.scalers[feature_set].fit_transform(numeric_features)
        
        # Sélection de features si nécessaire
        if len(feature_names) > 20:  # Trop de features
            if feature_set not in self.feature_selectors:
                # Simuler des target labels pour la sélection (basé sur MA attractiveness)
                if 'ma_attractiveness' in df.columns:
                    target = df['ma_attractiveness'].fillna(0)
                else:
                    target = np.random.rand(len(df))  # Random fallback
                
                self.feature_selectors[feature_set] = SelectKBest(
                    score_func=f_classif, 
                    k=min(15, len(feature_names))
                )
                
                scaled_features = self.feature_selectors[feature_set].fit_transform(
                    scaled_features, target
                )
                
                # Mise à jour des noms de features
                selected_indices = self.feature_selectors[feature_set].get_support()
                feature_names = [name for i, name in enumerate(feature_names) if selected_indices[i]]
            else:
                scaled_features = self.feature_selectors[feature_set].transform(scaled_features)
        
        # Réduction de dimensionnalité si demandée
        if n_components and n_components < scaled_features.shape[1]:
            if feature_set not in self.dimensionality_reducers:
                self.dimensionality_reducers[feature_set] = PCA(n_components=n_components)
                reduced_features = self.dimensionality_reducers[feature_set].fit_transform(scaled_features)
            else:
                reduced_features = self.dimensionality_reducers[feature_set].transform(scaled_features)
            
            # Mise à jour des noms
            feature_names = [f"PC{i+1}" for i in range(n_components)]
            scaled_features = reduced_features
        
        logger.info(f"✅ Features préparées: {scaled_features.shape[1]} dimensions")
        
        return scaled_features, feature_names


class ClusteringEngine:
    """Moteur de clustering multi-méthodes"""
    
    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.results: Dict[str, ClusteringResult] = {}
        self.feature_engineering = FeatureEngineering()
        
    async def auto_cluster(
        self, 
        data: pd.DataFrame, 
        methods: List[ClusteringMethod] = None,
        max_clusters: int = 10
    ) -> Dict[str, ClusteringResult]:
        """Clustering automatique avec plusieurs méthodes"""
        
        if methods is None:
            methods = [
                ClusteringMethod.KMEANS,
                ClusteringMethod.DBSCAN,
                ClusteringMethod.GAUSSIAN_MIXTURE
            ]
        
        logger.info(f"🎯 Clustering automatique avec {len(methods)} méthodes")
        
        # Préparation des features
        enhanced_data = self.feature_engineering.engineer_company_features(data)
        X, feature_names = self.feature_engineering.prepare_clustering_features(enhanced_data)
        
        results = {}
        
        for method in methods:
            try:
                result = await self._cluster_with_method(
                    X, feature_names, method, max_clusters
                )
                results[method.value] = result
                
            except Exception as e:
                logger.error(f"❌ Erreur clustering {method.value}: {e}")
        
        # Sélection du meilleur clustering
        if results:
            best_method = max(results.keys(), key=lambda k: results[k].silhouette_score)
            logger.info(f"🏆 Meilleur clustering: {best_method} (silhouette: {results[best_method].silhouette_score:.3f})")
        
        self.results.update(results)
        return results
    
    async def _cluster_with_method(
        self,
        X: np.ndarray,
        feature_names: List[str],
        method: ClusteringMethod,
        max_clusters: int
    ) -> ClusteringResult:
        """Clustering avec une méthode spécifique"""
        
        start_time = time.time()
        cluster_id = f"{method.value}_{int(time.time())}"
        
        if method == ClusteringMethod.KMEANS:
            # Détermination automatique du nombre de clusters
            best_k = self._find_optimal_k(X, max_clusters)
            model = KMeans(n_clusters=best_k, random_state=42, n_init=10)
            
        elif method == ClusteringMethod.DBSCAN:
            # Estimation automatique d'epsilon
            eps = self._estimate_eps(X)
            model = DBSCAN(eps=eps, min_samples=5)
            
        elif method == ClusteringMethod.HIERARCHICAL:
            best_k = self._find_optimal_k(X, max_clusters)
            model = AgglomerativeClustering(n_clusters=best_k)
            
        elif method == ClusteringMethod.GAUSSIAN_MIXTURE:
            best_k = self._find_optimal_k(X, max_clusters, method='gmm')
            model = GaussianMixture(n_components=best_k, random_state=42)
            
        elif method == ClusteringMethod.SPECTRAL:
            best_k = self._find_optimal_k(X, max_clusters)
            model = SpectralClustering(n_clusters=best_k, random_state=42)
            
        elif method == ClusteringMethod.MEAN_SHIFT:
            model = MeanShift()
            
        else:
            raise ValueError(f"Méthode {method} non supportée")
        
        # Entraînement
        labels = model.fit_predict(X)
        
        # Gestion des outliers (label -1 pour DBSCAN)
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)
        if -1 in unique_labels:
            n_clusters -= 1  # Ne pas compter les outliers
        
        training_time = time.time() - start_time
        
        # Calcul des métriques
        if n_clusters > 1 and len(np.unique(labels)) > 1:
            silhouette = silhouette_score(X, labels)
            calinski_harabasz = calinski_harabasz_score(X, labels)
            davies_bouldin = davies_bouldin_score(X, labels)
        else:
            silhouette = 0.0
            calinski_harabasz = 0.0
            davies_bouldin = float('inf')
        
        # Centres des clusters
        cluster_centers = None
        inertia = None
        
        if hasattr(model, 'cluster_centers_'):
            cluster_centers = model.cluster_centers_
        if hasattr(model, 'inertia_'):
            inertia = model.inertia_
        
        result = ClusteringResult(
            cluster_id=cluster_id,
            method=method,
            n_clusters=n_clusters,
            cluster_labels=labels.tolist(),
            cluster_centers=cluster_centers,
            silhouette_score=silhouette,
            calinski_harabasz_score=calinski_harabasz,
            davies_bouldin_score=davies_bouldin,
            inertia=inertia,
            feature_names=feature_names,
            n_samples=len(X),
            training_time=training_time
        )
        
        self.models[cluster_id] = model
        
        logger.info(f"✅ {method.value}: {n_clusters} clusters, silhouette: {silhouette:.3f}")
        
        return result
    
    def _find_optimal_k(self, X: np.ndarray, max_k: int, method: str = 'kmeans') -> int:
        """Trouve le nombre optimal de clusters"""
        
        silhouette_scores = []
        k_range = range(2, min(max_k + 1, len(X)))
        
        for k in k_range:
            try:
                if method == 'gmm':
                    model = GaussianMixture(n_components=k, random_state=42)
                    labels = model.fit_predict(X)
                else:
                    model = KMeans(n_clusters=k, random_state=42, n_init=10)
                    labels = model.fit_predict(X)
                
                if len(np.unique(labels)) > 1:
                    score = silhouette_score(X, labels)
                    silhouette_scores.append(score)
                else:
                    silhouette_scores.append(0)
                    
            except Exception:
                silhouette_scores.append(0)
        
        if silhouette_scores:
            optimal_k = k_range[np.argmax(silhouette_scores)]
        else:
            optimal_k = 3  # Fallback
        
        return optimal_k
    
    def _estimate_eps(self, X: np.ndarray) -> float:
        """Estime le paramètre epsilon pour DBSCAN"""
        
        from sklearn.neighbors import NearestNeighbors
        
        # Méthode k-distance
        k = 4
        neighbors = NearestNeighbors(n_neighbors=k)
        neighbors.fit(X)
        distances, _ = neighbors.kneighbors(X)
        
        # Prendre la distance au k-ème voisin
        k_distances = distances[:, k-1]
        k_distances = np.sort(k_distances)
        
        # Chercher le "coude" dans la courbe
        # Approximation simple: prendre le 95e percentile
        eps = np.percentile(k_distances, 95)
        
        return eps


class SegmentAnalyzer:
    """Analyseur de segments pour profiling et insights"""
    
    def __init__(self):
        self.segment_profiles: Dict[str, List[ClusterProfile]] = {}
        
    async def create_cluster_profiles(
        self, 
        clustering_result: ClusteringResult,
        original_data: pd.DataFrame
    ) -> List[ClusterProfile]:
        """Crée les profils des clusters"""
        
        logger.info(f"📊 Création profils pour {clustering_result.n_clusters} clusters")
        
        profiles = []
        labels = np.array(clustering_result.cluster_labels)
        
        # Préparer les données avec features originales
        enhanced_data = self.clustering_engine.feature_engineering.engineer_company_features(original_data)
        numeric_data = enhanced_data.select_dtypes(include=[np.number]).fillna(0)
        
        unique_labels = np.unique(labels)
        
        for cluster_id in unique_labels:
            if cluster_id == -1:  # Outliers dans DBSCAN
                continue
                
            # Données du cluster
            cluster_mask = labels == cluster_id
            cluster_data = numeric_data[cluster_mask]
            cluster_original = original_data[cluster_mask]
            
            size = len(cluster_data)
            percentage = (size / len(original_data)) * 100
            
            # Statistiques moyennes
            avg_features = cluster_data.mean().to_dict()
            
            # Features dominantes (différence avec la moyenne globale)
            global_mean = numeric_data.mean()
            feature_importance = abs(cluster_data.mean() - global_mean) / (global_mean + 1e-6)
            
            dominant_features = [
                (feature, importance) 
                for feature, importance in feature_importance.items()
            ]
            dominant_features.sort(key=lambda x: x[1], reverse=True)
            dominant_features = dominant_features[:5]  # Top 5
            
            # Ranges des features
            feature_ranges = {
                feature: (cluster_data[feature].min(), cluster_data[feature].max())
                for feature in cluster_data.columns
            }
            
            # Entreprises représentatives (celles proches du centre)
            if clustering_result.cluster_centers is not None and cluster_id < len(clustering_result.cluster_centers):
                cluster_center = clustering_result.cluster_centers[cluster_id]
                # Calculer distances au centre (approximation)
                representative_companies = cluster_original.head(5)['siren'].tolist() if 'siren' in cluster_original.columns else []
            else:
                representative_companies = cluster_original.head(3)['siren'].tolist() if 'siren' in cluster_original.columns else []
            
            # Score d'attractivité M&A
            ma_score = avg_features.get('ma_attractiveness', 0)
            
            # Label et description du cluster
            cluster_label, cluster_description = self._generate_cluster_label(avg_features, dominant_features)
            
            profile = ClusterProfile(
                cluster_id=cluster_id,
                cluster_label=cluster_label,
                cluster_description=cluster_description,
                size=size,
                percentage=percentage,
                avg_features=avg_features,
                dominant_features=dominant_features,
                feature_ranges=feature_ranges,
                representative_companies=representative_companies,
                ma_attractiveness_score=ma_score
            )
            
            profiles.append(profile)
        
        # Trier par attractivité M&A
        profiles.sort(key=lambda p: p.ma_attractiveness_score, reverse=True)
        
        self.segment_profiles[clustering_result.cluster_id] = profiles
        
        logger.info(f"✅ {len(profiles)} profils créés")
        
        return profiles
    
    def _generate_cluster_label(
        self, 
        avg_features: Dict[str, float], 
        dominant_features: List[Tuple[str, float]]
    ) -> Tuple[str, str]:
        """Génère un label et description pour le cluster"""
        
        # Règles heuristiques pour labellisation
        ca_avg = avg_features.get('chiffre_affaires', 0)
        effectifs_avg = avg_features.get('effectifs', 0)
        age_avg = avg_features.get('company_age', 0)
        is_strategic = avg_features.get('is_strategic_sector', 0)
        is_paris = avg_features.get('is_paris', 0)
        
        # Détermination de la taille
        if ca_avg > 50000000:
            size_label = "Grandes entreprises"
        elif ca_avg > 10000000:
            size_label = "Entreprises moyennes"
        elif ca_avg > 1000000:
            size_label = "Petites entreprises"
        else:
            size_label = "Très petites entreprises"
        
        # Caractéristiques additionnelles
        characteristics = []
        
        if is_strategic > 0.5:
            characteristics.append("secteurs stratégiques")
        
        if is_paris > 0.5:
            characteristics.append("région parisienne")
        
        if age_avg < 5:
            characteristics.append("jeunes pousses")
        elif age_avg > 20:
            characteristics.append("établies")
        
        # Construction du label
        if characteristics:
            label = f"{size_label} - {', '.join(characteristics)}"
        else:
            label = size_label
        
        # Description détaillée
        description = f"{size_label.lower()} avec un chiffre d'affaires moyen de {ca_avg:,.0f}€"
        
        if effectifs_avg > 0:
            description += f" et {effectifs_avg:.0f} employés en moyenne"
        
        if characteristics:
            description += f". Principalement {', '.join(characteristics)}."
        
        # Ajouter features dominantes
        if dominant_features:
            top_feature = dominant_features[0][0]
            description += f" Caractérisé principalement par {top_feature}."
        
        return label, description


class SegmentationEngine:
    """Moteur principal de segmentation"""
    
    def __init__(self):
        self.clustering_engine = ClusteringEngine()
        self.segment_analyzer = SegmentAnalyzer()
        self.segments: Dict[str, Segment] = {}
        
        logger.info("🎯 Moteur de segmentation initialisé")
    
    @cached('company_segmentation', ttl_seconds=7200)  # Cache 2h
    async def segment_companies(
        self, 
        companies_data: pd.DataFrame,
        segment_type: SegmentType = SegmentType.COMPANY_PROFILE
    ) -> Dict[str, Any]:
        """Segmentation automatique des entreprises"""
        
        logger.info(f"🏢 Segmentation entreprises: {len(companies_data)} entreprises")
        
        try:
            # Clustering automatique
            clustering_results = await self.clustering_engine.auto_cluster(companies_data)
            
            if not clustering_results:
                raise Exception("Aucun clustering réussi")
            
            # Sélection du meilleur clustering
            best_method = max(
                clustering_results.keys(), 
                key=lambda k: clustering_results[k].silhouette_score
            )
            best_clustering = clustering_results[best_method]
            
            # Création des profils
            self.segment_analyzer.clustering_engine = self.clustering_engine
            cluster_profiles = await self.segment_analyzer.create_cluster_profiles(
                best_clustering, companies_data
            )
            
            # Conversion en segments
            segments = []
            labels = np.array(best_clustering.cluster_labels)
            
            for profile in cluster_profiles:
                cluster_mask = labels == profile.cluster_id
                company_sirens = companies_data[cluster_mask]['siren'].tolist() if 'siren' in companies_data.columns else []
                
                # Détermination de la priorité
                priority = self._determine_segment_priority(profile)
                
                # Patterns comportementaux (simulés pour démonstration)
                behavioral_patterns = self._extract_behavioral_patterns(
                    companies_data[cluster_mask], profile
                )
                
                # Stratégies recommandées
                strategies = self._recommend_strategies(profile)
                
                segment = Segment(
                    segment_id=f"segment_{segment_type.value}_{profile.cluster_id}",
                    segment_name=profile.cluster_label,
                    segment_type=segment_type,
                    priority=priority,
                    entities=company_sirens,
                    size=profile.size,
                    key_characteristics=profile.avg_features,
                    behavioral_patterns=behavioral_patterns,
                    revenue_potential=self._calculate_revenue_potential(profile),
                    acquisition_probability=profile.ma_attractiveness_score / 100,
                    conversion_rate=self._estimate_conversion_rate(profile),
                    recommended_strategies=strategies,
                    communication_preferences=self._determine_communication_preferences(profile),
                    historical_performance={'ma_score': profile.ma_attractiveness_score}
                )
                
                segments.append(segment)
                self.segments[segment.segment_id] = segment
            
            # Résumé de la segmentation
            segmentation_summary = {
                'method_used': best_method,
                'n_segments': len(segments),
                'quality_score': best_clustering.silhouette_score,
                'total_companies': len(companies_data),
                'segments': [
                    {
                        'segment_id': seg.segment_id,
                        'name': seg.segment_name,
                        'size': seg.size,
                        'percentage': (seg.size / len(companies_data)) * 100,
                        'priority': seg.priority.value,
                        'ma_attractiveness': seg.acquisition_probability * 100,
                        'revenue_potential': seg.revenue_potential,
                        'key_characteristics': list(seg.key_characteristics.keys())[:5]
                    }
                    for seg in segments
                ],
                'clustering_results': {
                    method: {
                        'n_clusters': result.n_clusters,
                        'silhouette_score': result.silhouette_score,
                        'training_time': result.training_time
                    }
                    for method, result in clustering_results.items()
                }
            }
            
            logger.info(f"✅ Segmentation terminée: {len(segments)} segments")
            
            return segmentation_summary
            
        except Exception as e:
            logger.error(f"❌ Erreur segmentation: {e}")
            raise
    
    def _determine_segment_priority(self, profile: ClusterProfile) -> SegmentPriority:
        """Détermine la priorité d'un segment"""
        
        # Basé sur attractivité M&A et taille
        ma_score = profile.ma_attractiveness_score
        size_percentage = profile.percentage
        
        if ma_score > 70 and size_percentage > 10:
            return SegmentPriority.VERY_HIGH
        elif ma_score > 50 and size_percentage > 5:
            return SegmentPriority.HIGH
        elif ma_score > 30 or size_percentage > 15:
            return SegmentPriority.MEDIUM
        else:
            return SegmentPriority.LOW
    
    def _extract_behavioral_patterns(
        self, 
        segment_data: pd.DataFrame, 
        profile: ClusterProfile
    ) -> Dict[str, Any]:
        """Extrait les patterns comportementaux d'un segment"""
        
        patterns = {}
        
        # Pattern de croissance
        if 'chiffre_affaires' in segment_data.columns:
            ca_values = segment_data['chiffre_affaires'].dropna()
            if len(ca_values) > 0:
                patterns['revenue_stability'] = 1 - (ca_values.std() / (ca_values.mean() + 1e-6))
                patterns['avg_revenue'] = ca_values.mean()
        
        # Pattern sectoriel
        if 'secteur_activite' in segment_data.columns:
            sector_counts = segment_data['secteur_activite'].value_counts()
            patterns['dominant_sector'] = sector_counts.index[0] if len(sector_counts) > 0 else 'Unknown'
            patterns['sector_concentration'] = sector_counts.iloc[0] / len(segment_data) if len(sector_counts) > 0 else 0
        
        # Pattern géographique
        if 'adresse' in segment_data.columns:
            paris_ratio = segment_data['adresse'].str.contains('Paris', na=False).mean()
            patterns['paris_concentration'] = paris_ratio
        
        # Pattern d'âge
        if 'company_age' in profile.avg_features:
            avg_age = profile.avg_features['company_age']
            if avg_age < 5:
                patterns['maturity_stage'] = 'startup'
            elif avg_age < 15:
                patterns['maturity_stage'] = 'growth'
            else:
                patterns['maturity_stage'] = 'mature'
        
        return patterns
    
    def _recommend_strategies(self, profile: ClusterProfile) -> List[str]:
        """Recommande des stratégies pour un segment"""
        
        strategies = []
        
        ma_score = profile.ma_attractiveness_score
        avg_ca = profile.avg_features.get('chiffre_affaires', 0)
        is_strategic = profile.avg_features.get('is_strategic_sector', 0)
        
        # Stratégies basées sur attractivité
        if ma_score > 70:
            strategies.append("Approche directe pour acquisition")
            strategies.append("Due diligence approfondie prioritaire")
        elif ma_score > 40:
            strategies.append("Évaluation approfondie du potentiel")
            strategies.append("Développement de relation long terme")
        else:
            strategies.append("Monitoring périodique")
            strategies.append("Évaluation sélective")
        
        # Stratégies basées sur taille
        if avg_ca > 10000000:
            strategies.append("Négociation structurée")
            strategies.append("Analyse concurrentielle")
        else:
            strategies.append("Approche flexible")
            strategies.append("Focus sur synergies")
        
        # Stratégies sectorielles
        if is_strategic > 0.5:
            strategies.append("Valorisation premium justifiée")
            strategies.append("Accélération du processus")
        
        return strategies
    
    def _calculate_revenue_potential(self, profile: ClusterProfile) -> float:
        """Calcule le potentiel de revenus d'un segment"""
        
        avg_ca = profile.avg_features.get('chiffre_affaires', 0)
        size = profile.size
        ma_score = profile.ma_attractiveness_score
        
        # Potentiel = CA moyen * taille * facteur attractivité
        potential = avg_ca * size * (ma_score / 100)
        
        return potential
    
    def _estimate_conversion_rate(self, profile: ClusterProfile) -> float:
        """Estime le taux de conversion d'un segment"""
        
        ma_score = profile.ma_attractiveness_score
        size_factor = min(1.0, profile.percentage / 20)  # Favorise segments moyens
        
        # Conversion basée sur attractivité et taille optimale
        base_rate = ma_score / 100 * 0.3  # Base 30% max
        conversion_rate = base_rate * (1 + size_factor)
        
        return min(conversion_rate, 0.5)  # Max 50%
    
    def _determine_communication_preferences(self, profile: ClusterProfile) -> Dict[str, Any]:
        """Détermine les préférences de communication"""
        
        preferences = {}
        
        avg_ca = profile.avg_features.get('chiffre_affaires', 0)
        avg_age = profile.avg_features.get('company_age', 0)
        is_tech = profile.avg_features.get('is_strategic_sector', 0)
        
        # Canal préféré
        if avg_ca > 50000000:
            preferences['preferred_channel'] = 'direct_meeting'
            preferences['decision_level'] = 'executive'
        elif avg_ca > 10000000:
            preferences['preferred_channel'] = 'email_phone'
            preferences['decision_level'] = 'management'
        else:
            preferences['preferred_channel'] = 'email'
            preferences['decision_level'] = 'owner'
        
        # Timing
        if avg_age < 5:
            preferences['optimal_timing'] = 'growth_phase'
        else:
            preferences['optimal_timing'] = 'strategic_moments'
        
        # Message type
        if is_tech > 0.5:
            preferences['message_focus'] = 'innovation_synergies'
        else:
            preferences['message_focus'] = 'operational_synergies'
        
        return preferences
    
    async def get_segment_insights(self, segment_id: str) -> Dict[str, Any]:
        """Génère des insights détaillés pour un segment"""
        
        if segment_id not in self.segments:
            raise ValueError(f"Segment {segment_id} non trouvé")
        
        segment = self.segments[segment_id]
        
        insights = {
            'segment_overview': {
                'name': segment.segment_name,
                'size': segment.size,
                'priority': segment.priority.value,
                'type': segment.segment_type.value
            },
            'business_potential': {
                'revenue_potential': segment.revenue_potential,
                'acquisition_probability': segment.acquisition_probability,
                'conversion_rate': segment.conversion_rate
            },
            'characteristics': segment.key_characteristics,
            'behavioral_patterns': segment.behavioral_patterns,
            'strategies': {
                'recommended_strategies': segment.recommended_strategies,
                'communication_preferences': segment.communication_preferences
            },
            'performance': segment.historical_performance,
            'recommendations': self._generate_segment_recommendations(segment)
        }
        
        return insights
    
    def _generate_segment_recommendations(self, segment: Segment) -> List[str]:
        """Génère des recommandations spécifiques pour un segment"""
        
        recommendations = []
        
        # Basé sur la priorité
        if segment.priority == SegmentPriority.VERY_HIGH:
            recommendations.append("Allouer des ressources dédiées à ce segment")
            recommendations.append("Développer une approche personnalisée")
        
        # Basé sur la taille
        if segment.size > 100:
            recommendations.append("Considérer une approche de marketing automation")
        else:
            recommendations.append("Approche individuelle recommandée")
        
        # Basé sur le potentiel
        if segment.revenue_potential > 1000000:
            recommendations.append("Investir dans le développement de relation")
        
        if segment.acquisition_probability > 0.6:
            recommendations.append("Préparer une pipeline d'acquisition active")
        
        return recommendations
    
    def get_segmentation_analytics(self) -> Dict[str, Any]:
        """Retourne les analytics de la segmentation"""
        
        if not self.segments:
            return {'error': 'Aucune segmentation disponible'}
        
        total_entities = sum(seg.size for seg in self.segments.values())
        
        # Distribution par priorité
        priority_distribution = Counter(seg.priority.value for seg in self.segments.values())
        
        # Distribution par type
        type_distribution = Counter(seg.segment_type.value for seg in self.segments.values())
        
        # Potentiel total
        total_revenue_potential = sum(seg.revenue_potential for seg in self.segments.values())
        avg_conversion_rate = np.mean([seg.conversion_rate for seg in self.segments.values()])
        
        # Top segments
        top_segments = sorted(
            self.segments.values(),
            key=lambda s: s.revenue_potential,
            reverse=True
        )[:5]
        
        return {
            'total_segments': len(self.segments),
            'total_entities': total_entities,
            'priority_distribution': dict(priority_distribution),
            'type_distribution': dict(type_distribution),
            'business_metrics': {
                'total_revenue_potential': total_revenue_potential,
                'average_conversion_rate': avg_conversion_rate,
                'high_priority_segments': len([s for s in self.segments.values() if s.priority in [SegmentPriority.VERY_HIGH, SegmentPriority.HIGH]])
            },
            'top_segments': [
                {
                    'segment_id': seg.segment_id,
                    'name': seg.segment_name,
                    'size': seg.size,
                    'revenue_potential': seg.revenue_potential,
                    'priority': seg.priority.value
                }
                for seg in top_segments
            ],
            'recommendations': [
                "Concentrer efforts sur segments haute priorité",
                "Développer approches spécialisées par segment",
                "Monitorer performance par segment"
            ],
            'last_updated': datetime.now().isoformat()
        }


# Instance globale
_segmentation_engine: Optional[SegmentationEngine] = None


async def get_segmentation_engine() -> SegmentationEngine:
    """Factory pour obtenir le moteur de segmentation"""
    global _segmentation_engine
    
    if _segmentation_engine is None:
        _segmentation_engine = SegmentationEngine()
    
    return _segmentation_engine


# Fonctions utilitaires

async def segment_companies_by_profile(companies_data: pd.DataFrame) -> Dict[str, Any]:
    """Interface simplifiée pour segmentation d'entreprises"""
    
    engine = await get_segmentation_engine()
    return await engine.segment_companies(companies_data, SegmentType.COMPANY_PROFILE)


async def get_acquisition_target_segments(companies_data: pd.DataFrame) -> Dict[str, Any]:
    """Segmentation spécialisée pour cibles d'acquisition"""
    
    engine = await get_segmentation_engine()
    return await engine.segment_companies(companies_data, SegmentType.ACQUISITION_TARGET)


async def analyze_segment_performance(segment_id: str) -> Dict[str, Any]:
    """Analyse la performance d'un segment spécifique"""
    
    engine = await get_segmentation_engine()
    return await engine.get_segment_insights(segment_id)