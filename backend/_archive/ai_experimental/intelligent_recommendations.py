"""
Système de recommandations intelligentes pour M&A Intelligence Platform
US-010: Moteur de recommandations basé sur IA pour optimiser les décisions M&A

Ce module fournit:
- Recommandations de cibles d'acquisition intelligentes
- Filtrage collaboratif et content-based filtering
- Recommandations de timing optimal
- Suggestions de stratégies d'approche
- Recommandations de prix et valorisation
- Système de scoring multi-dimensionnel
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from functools import wraps
from enum import Enum
import json
import math
from collections import defaultdict, Counter

# Machine Learning pour recommandations
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from scipy.sparse import csr_matrix
import networkx as nx

# Advanced ML
import xgboost as xgb
from lightgbm import LGBMRanker

from app.core.logging_system import get_logger, LogCategory
from app.core.cache_manager import get_cache_manager, cached
from app.core.advanced_ai_engine import get_advanced_ai_engine, PredictionConfidence

logger = get_logger("intelligent_recommendations", LogCategory.AI_ML)


class RecommendationType(str, Enum):
    """Types de recommandations"""
    TARGET_ACQUISITION = "target_acquisition"
    TIMING_OPTIMIZATION = "timing_optimization"
    PRICING_STRATEGY = "pricing_strategy"
    APPROACH_STRATEGY = "approach_strategy"
    PORTFOLIO_OPTIMIZATION = "portfolio_optimization"
    RISK_MITIGATION = "risk_mitigation"


class RecommendationPriority(str, Enum):
    """Priorité des recommandations"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class RecommendationMethod(str, Enum):
    """Méthodes de recommandation"""
    COLLABORATIVE_FILTERING = "collaborative_filtering"
    CONTENT_BASED = "content_based"
    HYBRID = "hybrid"
    KNOWLEDGE_BASED = "knowledge_based"
    DEEP_LEARNING = "deep_learning"


@dataclass
class RecommendationItem:
    """Item de recommandation"""
    item_id: str
    item_type: str
    title: str
    description: str
    score: float  # 0-100
    confidence: PredictionConfidence
    reasoning: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Recommendation:
    """Recommandation complète"""
    recommendation_id: str
    user_id: str
    recommendation_type: RecommendationType
    priority: RecommendationPriority
    method_used: RecommendationMethod
    
    title: str
    description: str
    items: List[RecommendationItem]
    
    # Scores et métriques
    relevance_score: float  # 0-100
    confidence_score: float  # 0-100
    potential_impact: float  # 0-100
    
    # Timing et validité
    recommended_timing: Optional[str] = None
    validity_period_days: int = 30
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    
    # Metadata
    supporting_data: Dict[str, Any] = field(default_factory=dict)
    alternative_options: List[Dict[str, Any]] = field(default_factory=list)
    
    def __post_init__(self):
        if self.expires_at is None:
            self.expires_at = self.created_at + timedelta(days=self.validity_period_days)


class CollaborativeFilteringEngine:
    """Moteur de filtrage collaboratif pour recommandations"""
    
    def __init__(self):
        self.user_item_matrix: Optional[csr_matrix] = None
        self.user_similarity_matrix: Optional[np.ndarray] = None
        self.item_similarity_matrix: Optional[np.ndarray] = None
        self.svd_model: Optional[TruncatedSVD] = None
        self.user_mappings: Dict[str, int] = {}
        self.item_mappings: Dict[str, int] = {}
        self.reverse_user_mappings: Dict[int, str] = {}
        self.reverse_item_mappings: Dict[int, str] = {}
        
    async def train_collaborative_model(self, interaction_data: pd.DataFrame):
        """Entraîne le modèle de filtrage collaboratif"""
        
        try:
            logger.info("🤝 Entraînement modèle collaboratif")
            
            # Création des mappings utilisateur-item
            users = interaction_data['user_id'].unique()
            items = interaction_data['item_id'].unique()
            
            self.user_mappings = {user: idx for idx, user in enumerate(users)}
            self.item_mappings = {item: idx for idx, item in enumerate(items)}
            self.reverse_user_mappings = {idx: user for user, idx in self.user_mappings.items()}
            self.reverse_item_mappings = {idx: item for item, idx in self.item_mappings.items()}
            
            # Création matrice utilisateur-item
            n_users = len(users)
            n_items = len(items)
            
            user_indices = [self.user_mappings[user] for user in interaction_data['user_id']]
            item_indices = [self.item_mappings[item] for item in interaction_data['item_id']]
            ratings = interaction_data['rating'].values
            
            self.user_item_matrix = csr_matrix(
                (ratings, (user_indices, item_indices)),
                shape=(n_users, n_items)
            )
            
            # Calcul similarité utilisateurs
            self.user_similarity_matrix = cosine_similarity(self.user_item_matrix)
            
            # Calcul similarité items
            self.item_similarity_matrix = cosine_similarity(self.user_item_matrix.T)
            
            # Modèle SVD pour réduction dimensionnelle
            self.svd_model = TruncatedSVD(n_components=min(50, min(n_users, n_items) - 1))
            self.svd_model.fit(self.user_item_matrix)
            
            logger.info(f"✅ Modèle collaboratif entraîné: {n_users} utilisateurs, {n_items} items")
            
        except Exception as e:
            logger.error(f"❌ Erreur entraînement collaboratif: {e}")
            raise
    
    async def get_user_recommendations(self, user_id: str, n_recommendations: int = 10) -> List[Tuple[str, float]]:
        """Recommandations pour un utilisateur basées sur filtrage collaboratif"""
        
        if user_id not in self.user_mappings:
            return []
        
        user_idx = self.user_mappings[user_id]
        
        # Utilisateurs similaires
        user_similarities = self.user_similarity_matrix[user_idx]
        similar_users = np.argsort(user_similarities)[::-1][1:11]  # Top 10 (excl. soi-même)
        
        # Items recommandés basés sur utilisateurs similaires
        recommendations_scores = defaultdict(float)
        
        for similar_user_idx in similar_users:
            similarity_score = user_similarities[similar_user_idx]
            
            # Items aimés par l'utilisateur similaire
            similar_user_items = self.user_item_matrix[similar_user_idx].nonzero()[1]
            
            for item_idx in similar_user_items:
                item_id = self.reverse_item_mappings[item_idx]
                rating = self.user_item_matrix[similar_user_idx, item_idx]
                
                # Vérifier que l'utilisateur n'a pas déjà interagi avec cet item
                if self.user_item_matrix[user_idx, item_idx] == 0:
                    recommendations_scores[item_id] += similarity_score * rating
        
        # Trier par score
        sorted_recommendations = sorted(
            recommendations_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return sorted_recommendations[:n_recommendations]
    
    async def get_item_recommendations(self, item_id: str, n_recommendations: int = 10) -> List[Tuple[str, float]]:
        """Recommandations d'items similaires"""
        
        if item_id not in self.item_mappings:
            return []
        
        item_idx = self.item_mappings[item_id]
        
        # Items similaires
        item_similarities = self.item_similarity_matrix[item_idx]
        similar_items_indices = np.argsort(item_similarities)[::-1][1:n_recommendations+1]
        
        similar_items = [
            (self.reverse_item_mappings[idx], item_similarities[idx])
            for idx in similar_items_indices
        ]
        
        return similar_items


class ContentBasedEngine:
    """Moteur de recommandations basé sur le contenu"""
    
    def __init__(self):
        self.tfidf_vectorizer: Optional[TfidfVectorizer] = None
        self.content_matrix: Optional[np.ndarray] = None
        self.feature_matrix: Optional[np.ndarray] = None
        self.item_features: Dict[str, Dict[str, Any]] = {}
        self.content_similarity_matrix: Optional[np.ndarray] = None
        
    async def train_content_model(self, companies_data: pd.DataFrame):
        """Entraîne le modèle basé sur le contenu des entreprises"""
        
        try:
            logger.info("📄 Entraînement modèle content-based")
            
            # Préparation des features textuelles
            text_features = []
            for _, company in companies_data.iterrows():
                text_content = " ".join([
                    str(company.get('secteur_activite', '')),
                    str(company.get('description', '')),
                    str(company.get('activite_principale', '')),
                    str(company.get('localisation', ''))
                ])
                text_features.append(text_content)
            
            # TF-IDF pour features textuelles
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',  # TODO: Ajouter stopwords français
                ngram_range=(1, 2)
            )
            
            self.content_matrix = self.tfidf_vectorizer.fit_transform(text_features)
            
            # Features numériques
            numeric_features = companies_data.select_dtypes(include=[np.number]).fillna(0)
            
            # Normalisation
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            normalized_numeric = scaler.fit_transform(numeric_features)
            
            # Combinaison features textuelles + numériques
            from scipy.sparse import hstack
            self.feature_matrix = hstack([self.content_matrix, csr_matrix(normalized_numeric)])
            
            # Calcul similarité contenu
            self.content_similarity_matrix = cosine_similarity(self.feature_matrix)
            
            # Stockage métadonnées des items
            for idx, (_, company) in enumerate(companies_data.iterrows()):
                siren = company.get('siren', f'company_{idx}')
                self.item_features[siren] = {
                    'secteur': company.get('secteur_activite', ''),
                    'chiffre_affaires': company.get('chiffre_affaires', 0),
                    'effectifs': company.get('effectifs', 0),
                    'localisation': company.get('localisation', ''),
                    'age': company.get('company_age', 0),
                    'index': idx
                }
            
            logger.info(f"✅ Modèle content-based entraîné: {len(companies_data)} entreprises")
            
        except Exception as e:
            logger.error(f"❌ Erreur entraînement content-based: {e}")
            raise
    
    async def get_similar_companies(self, siren: str, n_recommendations: int = 10) -> List[Tuple[str, float, str]]:
        """Trouve des entreprises similaires basées sur le contenu"""
        
        if siren not in self.item_features:
            return []
        
        company_idx = self.item_features[siren]['index']
        similarities = self.content_similarity_matrix[company_idx]
        
        # Trier par similarité (exclure l'entreprise elle-même)
        similar_indices = np.argsort(similarities)[::-1][1:n_recommendations+1]
        
        recommendations = []
        for idx in similar_indices:
            # Trouver le SIREN correspondant à cet index
            similar_siren = None
            for s, features in self.item_features.items():
                if features['index'] == idx:
                    similar_siren = s
                    break
            
            if similar_siren:
                similarity_score = similarities[idx]
                reason = self._get_similarity_reason(siren, similar_siren)
                recommendations.append((similar_siren, similarity_score, reason))
        
        return recommendations
    
    def _get_similarity_reason(self, siren1: str, siren2: str) -> str:
        """Génère une explication de la similarité entre deux entreprises"""
        
        features1 = self.item_features.get(siren1, {})
        features2 = self.item_features.get(siren2, {})
        
        reasons = []
        
        # Secteur similaire
        if features1.get('secteur') == features2.get('secteur'):
            reasons.append(f"Même secteur: {features1.get('secteur')}")
        
        # Taille similaire
        ca1 = features1.get('chiffre_affaires', 0)
        ca2 = features2.get('chiffre_affaires', 0)
        if abs(ca1 - ca2) / max(ca1, ca2, 1) < 0.3:  # Différence < 30%
            reasons.append("Taille similaire")
        
        # Localisation similaire
        if features1.get('localisation') == features2.get('localisation'):
            reasons.append(f"Même région: {features1.get('localisation')}")
        
        return "; ".join(reasons) if reasons else "Profil d'entreprise similaire"


class HybridRecommendationEngine:
    """Moteur de recommandations hybride combinant plusieurs approches"""
    
    def __init__(self):
        self.collaborative_engine = CollaborativeFilteringEngine()
        self.content_engine = ContentBasedEngine()
        self.knowledge_rules: List[Dict[str, Any]] = []
        self.weight_collaborative = 0.4
        self.weight_content = 0.4
        self.weight_knowledge = 0.2
        
    async def initialize_engines(self, interaction_data: pd.DataFrame, companies_data: pd.DataFrame):
        """Initialise tous les moteurs de recommandation"""
        
        try:
            logger.info("🔧 Initialisation moteurs de recommandation hybrides")
            
            # Entraînement parallèle des modèles
            await asyncio.gather(
                self.collaborative_engine.train_collaborative_model(interaction_data),
                self.content_engine.train_content_model(companies_data)
            )
            
            # Règles métier
            self._setup_knowledge_rules()
            
            logger.info("✅ Moteurs hybrides initialisés")
            
        except Exception as e:
            logger.error(f"❌ Erreur initialisation moteurs: {e}")
            raise
    
    def _setup_knowledge_rules(self):
        """Configure les règles métier pour recommandations"""
        
        self.knowledge_rules = [
            {
                'name': 'high_growth_tech',
                'condition': lambda company: (
                    company.get('secteur_activite', '').lower() in ['tech', 'technology', 'digital'] and
                    company.get('growth_rate', 0) > 0.2
                ),
                'boost': 20,
                'reason': 'Secteur tech à forte croissance'
            },
            {
                'name': 'profitable_mature',
                'condition': lambda company: (
                    company.get('chiffre_affaires', 0) > 10000000 and
                    company.get('company_age', 0) > 5
                ),
                'boost': 15,
                'reason': 'Entreprise mature et profitable'
            },
            {
                'name': 'strategic_location',
                'condition': lambda company: (
                    company.get('is_paris', False) or
                    'lyon' in company.get('localisation', '').lower()
                ),
                'boost': 10,
                'reason': 'Localisation stratégique'
            },
            {
                'name': 'family_business',
                'condition': lambda company: company.get('type_entreprise') == 'familiale',
                'boost': -5,
                'reason': 'Entreprise familiale (acquisition plus complexe)'
            }
        ]
    
    async def get_acquisition_recommendations(
        self, 
        user_id: str, 
        user_profile: Dict[str, Any],
        n_recommendations: int = 10
    ) -> List[Recommendation]:
        """Génère des recommandations d'acquisition hybrides"""
        
        try:
            logger.info(f"🎯 Génération recommandations acquisition pour {user_id}")
            
            # Recommandations collaboratives
            collab_recommendations = await self.collaborative_engine.get_user_recommendations(
                user_id, n_recommendations * 2
            )
            
            # Recommandations basées contenu (si l'utilisateur a des préférences)
            content_recommendations = []
            if 'preferred_sectors' in user_profile:
                for sector in user_profile['preferred_sectors']:
                    # Trouver une entreprise exemple du secteur pour similarity
                    example_company = self._find_sector_example(sector)
                    if example_company:
                        sector_recs = await self.content_engine.get_similar_companies(
                            example_company, n_recommendations
                        )
                        content_recommendations.extend(sector_recs)
            
            # Fusion et scoring hybride
            hybrid_scores = {}
            
            # Scores collaboratifs
            for item_id, score in collab_recommendations:
                hybrid_scores[item_id] = score * self.weight_collaborative
            
            # Scores contenu
            for item_id, score, reason in content_recommendations:
                if item_id in hybrid_scores:
                    hybrid_scores[item_id] += score * self.weight_content
                else:
                    hybrid_scores[item_id] = score * self.weight_content
            
            # Application règles métier
            for item_id in hybrid_scores.keys():
                company_data = self._get_company_data(item_id)
                knowledge_boost = self._apply_knowledge_rules(company_data)
                hybrid_scores[item_id] += knowledge_boost * self.weight_knowledge
            
            # Tri final
            sorted_recommendations = sorted(
                hybrid_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )[:n_recommendations]
            
            # Conversion en objets Recommendation
            recommendations = []
            for idx, (siren, score) in enumerate(sorted_recommendations):
                company_data = self._get_company_data(siren)
                
                # Génération reasoning
                reasoning = self._generate_recommendation_reasoning(siren, user_profile, company_data)
                
                # Confiance basée sur score et disponibilité des données
                confidence = self._calculate_confidence(score, company_data)
                
                recommendation_item = RecommendationItem(
                    item_id=siren,
                    item_type="acquisition_target",
                    title=f"Acquisition: {company_data.get('nom_entreprise', siren)}",
                    description=f"Entreprise du secteur {company_data.get('secteur_activite', 'N/A')}",
                    score=min(100, score * 10),  # Normalisation 0-100
                    confidence=confidence,
                    reasoning=reasoning,
                    metadata={
                        'chiffre_affaires': company_data.get('chiffre_affaires', 0),
                        'effectifs': company_data.get('effectifs', 0),
                        'secteur': company_data.get('secteur_activite', ''),
                        'localisation': company_data.get('localisation', '')
                    }
                )
                
                recommendation = Recommendation(
                    recommendation_id=f"acq_{user_id}_{siren}_{int(datetime.now().timestamp())}",
                    user_id=user_id,
                    recommendation_type=RecommendationType.TARGET_ACQUISITION,
                    priority=RecommendationPriority.HIGH if idx < 3 else RecommendationPriority.MEDIUM,
                    method_used=RecommendationMethod.HYBRID,
                    title=f"Cible d'acquisition recommandée #{idx+1}",
                    description=f"Entreprise {company_data.get('nom_entreprise', siren)} identifiée comme cible d'acquisition stratégique",
                    items=[recommendation_item],
                    relevance_score=min(100, score * 10),
                    confidence_score=self._confidence_to_score(confidence),
                    potential_impact=self._estimate_potential_impact(company_data),
                    supporting_data={
                        'hybrid_score': score,
                        'collaborative_contribution': hybrid_scores.get(siren, 0) * self.weight_collaborative / score if score > 0 else 0,
                        'content_contribution': self.weight_content,
                        'knowledge_contribution': self.weight_knowledge
                    }
                )
                
                recommendations.append(recommendation)
            
            logger.info(f"✅ {len(recommendations)} recommandations générées pour {user_id}")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"❌ Erreur génération recommandations: {e}")
            return []
    
    def _find_sector_example(self, sector: str) -> Optional[str]:
        """Trouve un exemple d'entreprise du secteur donné"""
        for siren, features in self.content_engine.item_features.items():
            if sector.lower() in features.get('secteur', '').lower():
                return siren
        return None
    
    def _get_company_data(self, siren: str) -> Dict[str, Any]:
        """Récupère les données d'une entreprise"""
        # En production, récupérer depuis la base de données
        return self.content_engine.item_features.get(siren, {})
    
    def _apply_knowledge_rules(self, company_data: Dict[str, Any]) -> float:
        """Applique les règles métier et retourne le boost de score"""
        total_boost = 0
        
        for rule in self.knowledge_rules:
            try:
                if rule['condition'](company_data):
                    total_boost += rule['boost']
            except Exception:
                continue  # Ignorer si données manquantes
        
        return total_boost
    
    def _generate_recommendation_reasoning(
        self, 
        siren: str, 
        user_profile: Dict[str, Any], 
        company_data: Dict[str, Any]
    ) -> List[str]:
        """Génère le raisonnement derrière la recommandation"""
        
        reasoning = []
        
        # Secteur match
        if 'preferred_sectors' in user_profile:
            company_sector = company_data.get('secteur', '')
            if any(sector in company_sector for sector in user_profile['preferred_sectors']):
                reasoning.append(f"Secteur d'intérêt: {company_sector}")
        
        # Taille appropriée
        ca = company_data.get('chiffre_affaires', 0)
        if 'min_revenue' in user_profile and ca >= user_profile['min_revenue']:
            reasoning.append(f"Chiffre d'affaires conforme: {ca:,.0f}€")
        
        # Localisation
        if 'preferred_regions' in user_profile:
            company_location = company_data.get('localisation', '')
            if any(region in company_location for region in user_profile['preferred_regions']):
                reasoning.append(f"Localisation stratégique: {company_location}")
        
        # Performance
        if company_data.get('growth_rate', 0) > 0.1:
            reasoning.append("Croissance positive détectée")
        
        # Règles métier appliquées
        for rule in self.knowledge_rules:
            try:
                if rule['condition'](company_data):
                    reasoning.append(rule['reason'])
            except Exception:
                continue
        
        return reasoning[:5]  # Limiter à 5 raisons principales
    
    def _calculate_confidence(self, score: float, company_data: Dict[str, Any]) -> PredictionConfidence:
        """Calcule le niveau de confiance de la recommandation"""
        
        # Facteurs de confiance
        data_completeness = len([v for v in company_data.values() if v]) / len(company_data) if company_data else 0
        score_strength = min(score / 10, 1.0)  # Normalisation
        
        confidence_score = (data_completeness * 0.4 + score_strength * 0.6) * 100
        
        if confidence_score >= 90:
            return PredictionConfidence.VERY_HIGH
        elif confidence_score >= 75:
            return PredictionConfidence.HIGH
        elif confidence_score >= 60:
            return PredictionConfidence.MEDIUM
        elif confidence_score >= 40:
            return PredictionConfidence.LOW
        else:
            return PredictionConfidence.VERY_LOW
    
    def _confidence_to_score(self, confidence: PredictionConfidence) -> float:
        """Convertit niveau de confiance en score numérique"""
        mapping = {
            PredictionConfidence.VERY_HIGH: 95,
            PredictionConfidence.HIGH: 80,
            PredictionConfidence.MEDIUM: 65,
            PredictionConfidence.LOW: 45,
            PredictionConfidence.VERY_LOW: 25
        }
        return mapping.get(confidence, 50)
    
    def _estimate_potential_impact(self, company_data: Dict[str, Any]) -> float:
        """Estime l'impact potentiel de l'acquisition"""
        
        impact_score = 50  # Base
        
        # Taille de l'entreprise
        ca = company_data.get('chiffre_affaires', 0)
        if ca > 50000000:  # > 50M€
            impact_score += 30
        elif ca > 10000000:  # > 10M€
            impact_score += 20
        elif ca > 1000000:  # > 1M€
            impact_score += 10
        
        # Secteur stratégique
        strategic_sectors = ['tech', 'digital', 'innovation', 'healthcare', 'fintech']
        sector = company_data.get('secteur', '').lower()
        if any(s in sector for s in strategic_sectors):
            impact_score += 15
        
        # Croissance
        growth = company_data.get('growth_rate', 0)
        if growth > 0.2:  # > 20%
            impact_score += 20
        elif growth > 0.1:  # > 10%
            impact_score += 10
        
        return min(100, impact_score)


class IntelligentRecommendationSystem:
    """Système principal de recommandations intelligentes"""
    
    def __init__(self):
        self.hybrid_engine = HybridRecommendationEngine()
        self.user_profiles: Dict[str, Dict[str, Any]] = {}
        self.recommendation_history: Dict[str, List[Recommendation]] = defaultdict(list)
        self.performance_metrics: Dict[str, float] = {}
        
        logger.info("🧠 Système de recommandations intelligentes initialisé")
    
    async def initialize_system(self, historical_data: Dict[str, pd.DataFrame]):
        """Initialise le système avec des données historiques"""
        
        try:
            logger.info("🚀 Initialisation système de recommandations")
            
            # Données d'interaction (utilisateur-entreprise)
            interaction_data = historical_data.get('interactions', pd.DataFrame())
            companies_data = historical_data.get('companies', pd.DataFrame())
            
            if interaction_data.empty:
                # Créer des données d'interaction simulées pour démonstration
                interaction_data = self._generate_sample_interactions(companies_data)
            
            # Initialisation des moteurs
            await self.hybrid_engine.initialize_engines(interaction_data, companies_data)
            
            logger.info("✅ Système de recommandations initialisé")
            
        except Exception as e:
            logger.error(f"❌ Erreur initialisation système: {e}")
            raise
    
    def _generate_sample_interactions(self, companies_data: pd.DataFrame) -> pd.DataFrame:
        """Génère des interactions simulées pour démonstration"""
        
        if companies_data.empty:
            return pd.DataFrame(columns=['user_id', 'item_id', 'rating', 'timestamp'])
        
        # Utilisateurs simulés
        users = ['user_1', 'user_2', 'user_3', 'user_4', 'user_5']
        
        interactions = []
        for user in users:
            # Chaque utilisateur interagit avec 10-20 entreprises
            n_interactions = np.random.randint(10, 21)
            selected_companies = companies_data.sample(n_interactions)
            
            for _, company in selected_companies.iterrows():
                siren = company.get('siren', f'company_{np.random.randint(1000, 9999)}')
                rating = np.random.uniform(1, 5)  # Rating 1-5
                
                interactions.append({
                    'user_id': user,
                    'item_id': siren,
                    'rating': rating,
                    'timestamp': datetime.now() - timedelta(days=np.random.randint(1, 365))
                })
        
        return pd.DataFrame(interactions)
    
    @cached('recommendations', ttl_seconds=1800)  # Cache 30 minutes
    async def get_personalized_recommendations(
        self, 
        user_id: str, 
        recommendation_types: List[RecommendationType] = None,
        n_recommendations: int = 10
    ) -> List[Recommendation]:
        """Génère des recommandations personnalisées pour un utilisateur"""
        
        if recommendation_types is None:
            recommendation_types = [RecommendationType.TARGET_ACQUISITION]
        
        user_profile = self.user_profiles.get(user_id, self._create_default_user_profile(user_id))
        
        all_recommendations = []
        
        for rec_type in recommendation_types:
            if rec_type == RecommendationType.TARGET_ACQUISITION:
                recommendations = await self.hybrid_engine.get_acquisition_recommendations(
                    user_id, user_profile, n_recommendations
                )
                all_recommendations.extend(recommendations)
            
            elif rec_type == RecommendationType.TIMING_OPTIMIZATION:
                timing_recs = await self._generate_timing_recommendations(user_id, user_profile)
                all_recommendations.extend(timing_recs)
            
            elif rec_type == RecommendationType.PRICING_STRATEGY:
                pricing_recs = await self._generate_pricing_recommendations(user_id, user_profile)
                all_recommendations.extend(pricing_recs)
        
        # Tri par pertinence globale
        all_recommendations.sort(key=lambda x: x.relevance_score, reverse=True)
        
        # Sauvegarde historique
        self.recommendation_history[user_id].extend(all_recommendations)
        
        return all_recommendations[:n_recommendations]
    
    def _create_default_user_profile(self, user_id: str) -> Dict[str, Any]:
        """Crée un profil utilisateur par défaut"""
        
        default_profile = {
            'user_id': user_id,
            'preferred_sectors': ['technology', 'services', 'manufacturing'],
            'preferred_regions': ['Île-de-France', 'Auvergne-Rhône-Alpes'],
            'min_revenue': 1000000,  # 1M€
            'max_revenue': 100000000,  # 100M€
            'risk_tolerance': 'medium',
            'investment_horizon': '2-3 years',
            'experience_level': 'intermediate',
            'created_at': datetime.now()
        }
        
        self.user_profiles[user_id] = default_profile
        return default_profile
    
    async def _generate_timing_recommendations(
        self, 
        user_id: str, 
        user_profile: Dict[str, Any]
    ) -> List[Recommendation]:
        """Génère des recommandations de timing optimal"""
        
        recommendations = []
        
        # Analyse des tendances saisonnières
        current_month = datetime.now().month
        
        if current_month in [9, 10, 11]:  # Q4
            timing_recommendation = Recommendation(
                recommendation_id=f"timing_{user_id}_{int(datetime.now().timestamp())}",
                user_id=user_id,
                recommendation_type=RecommendationType.TIMING_OPTIMIZATION,
                priority=RecommendationPriority.MEDIUM,
                method_used=RecommendationMethod.KNOWLEDGE_BASED,
                title="Timing optimal pour acquisitions Q4",
                description="Les acquisitions en Q4 bénéficient souvent de meilleures conditions",
                items=[],
                relevance_score=75,
                confidence_score=80,
                potential_impact=70,
                recommended_timing="Novembre-Décembre 2024",
                supporting_data={
                    'seasonal_factor': 'Q4_advantage',
                    'historical_success_rate': 0.85
                }
            )
            recommendations.append(timing_recommendation)
        
        return recommendations
    
    async def _generate_pricing_recommendations(
        self, 
        user_id: str, 
        user_profile: Dict[str, Any]
    ) -> List[Recommendation]:
        """Génère des recommandations de stratégie de prix"""
        
        recommendations = []
        
        # Recommandation basée sur l'expérience utilisateur
        experience = user_profile.get('experience_level', 'intermediate')
        
        if experience == 'beginner':
            pricing_rec = Recommendation(
                recommendation_id=f"pricing_{user_id}_{int(datetime.now().timestamp())}",
                user_id=user_id,
                recommendation_type=RecommendationType.PRICING_STRATEGY,
                priority=RecommendationPriority.HIGH,
                method_used=RecommendationMethod.KNOWLEDGE_BASED,
                title="Stratégie de prix conservative recommandée",
                description="Pour un premier acquéreur, privilégier une approche conservative",
                items=[],
                relevance_score=80,
                confidence_score=90,
                potential_impact=60,
                supporting_data={
                    'recommended_multiple_range': '3-5x EBITDA',
                    'risk_level': 'low'
                }
            )
            recommendations.append(pricing_rec)
        
        return recommendations
    
    async def track_recommendation_performance(
        self, 
        recommendation_id: str, 
        user_action: str, 
        outcome_score: float = None
    ):
        """Suit la performance des recommandations"""
        
        # Mise à jour métriques de performance
        if outcome_score is not None:
            if 'total_outcome_score' not in self.performance_metrics:
                self.performance_metrics['total_outcome_score'] = 0
                self.performance_metrics['total_recommendations'] = 0
            
            self.performance_metrics['total_outcome_score'] += outcome_score
            self.performance_metrics['total_recommendations'] += 1
            
            avg_performance = (
                self.performance_metrics['total_outcome_score'] /
                self.performance_metrics['total_recommendations']
            )
            
            self.performance_metrics['average_performance'] = avg_performance
        
        # Log de l'action
        logger.info(f"📊 Recommendation {recommendation_id}: action={user_action}, score={outcome_score}")
    
    def get_system_analytics(self) -> Dict[str, Any]:
        """Retourne les analytics du système de recommandations"""
        
        total_users = len(self.user_profiles)
        total_recommendations = sum(len(recs) for recs in self.recommendation_history.values())
        
        # Distribution par type de recommandation
        type_distribution = Counter()
        for user_recs in self.recommendation_history.values():
            for rec in user_recs:
                type_distribution[rec.recommendation_type.value] += 1
        
        # Métriques de confiance moyennes
        confidence_scores = []
        for user_recs in self.recommendation_history.values():
            confidence_scores.extend([rec.confidence_score for rec in user_recs])
        
        avg_confidence = np.mean(confidence_scores) if confidence_scores else 0
        
        return {
            'total_users': total_users,
            'total_recommendations_generated': total_recommendations,
            'average_confidence_score': avg_confidence,
            'performance_metrics': self.performance_metrics,
            'recommendation_type_distribution': dict(type_distribution),
            'system_health': 'operational',
            'last_updated': datetime.now().isoformat()
        }


# Instance globale
_recommendation_system: Optional[IntelligentRecommendationSystem] = None


async def get_recommendation_system() -> IntelligentRecommendationSystem:
    """Factory pour obtenir le système de recommandations"""
    global _recommendation_system
    
    if _recommendation_system is None:
        _recommendation_system = IntelligentRecommendationSystem()
    
    return _recommendation_system


# Fonctions utilitaires

async def get_user_recommendations(
    user_id: str, 
    recommendation_types: List[str] = None,
    n_recommendations: int = 10
) -> List[Dict[str, Any]]:
    """Interface simplifiée pour obtenir des recommandations utilisateur"""
    
    system = await get_recommendation_system()
    
    # Conversion des types string en enum
    if recommendation_types:
        rec_types = [RecommendationType(rt) for rt in recommendation_types]
    else:
        rec_types = [RecommendationType.TARGET_ACQUISITION]
    
    recommendations = await system.get_personalized_recommendations(
        user_id, rec_types, n_recommendations
    )
    
    # Conversion en dictionnaires pour API
    return [
        {
            'recommendation_id': rec.recommendation_id,
            'type': rec.recommendation_type.value,
            'priority': rec.priority.value,
            'title': rec.title,
            'description': rec.description,
            'relevance_score': rec.relevance_score,
            'confidence_score': rec.confidence_score,
            'potential_impact': rec.potential_impact,
            'items': [
                {
                    'item_id': item.item_id,
                    'title': item.title,
                    'score': item.score,
                    'confidence': item.confidence.value,
                    'reasoning': item.reasoning,
                    'metadata': item.metadata
                }
                for item in rec.items
            ],
            'created_at': rec.created_at.isoformat(),
            'expires_at': rec.expires_at.isoformat() if rec.expires_at else None
        }
        for rec in recommendations
    ]


async def initialize_recommendation_system_with_data(companies_data: pd.DataFrame):
    """Initialise le système de recommandations avec des données d'entreprises"""
    
    system = await get_recommendation_system()
    
    historical_data = {
        'companies': companies_data,
        'interactions': pd.DataFrame()  # Sera généré automatiquement
    }
    
    await system.initialize_system(historical_data)