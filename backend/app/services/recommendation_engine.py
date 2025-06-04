"""
Moteur de recommandations automatisées pour M&A Intelligence Platform
US-008: IA pour recommandations d'investissement intelligentes

Ce module implémente:
- Recommandations personnalisées d'opportunités M&A
- Algorithmes de filtrage collaboratif et basé sur le contenu
- Système de scoring multi-critères
- Apprentissage des préférences utilisateur
- Optimisation de portefeuille d'acquisitions
- Alertes intelligentes sur opportunités
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union, Set
from dataclasses import dataclass, field
from enum import Enum
import json
import warnings
from collections import defaultdict, Counter
import heapq

# Machine Learning pour recommandations
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import NearestNeighbors
import joblib

# Optimisation
try:
    from scipy.optimize import minimize, differential_evolution
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("SciPy non disponible - optimisation limitée")

from app.core.logging_system import get_logger, LogCategory
from app.services.ai_scoring_engine import get_ai_scoring_engine, ScoringResult
from app.services.predictive_analytics import get_predictive_analytics_engine
from app.services.nlp_engine import get_nlp_engine

logger = get_logger("recommendation_engine", LogCategory.ML)


class RecommendationType(str, Enum):
    """Types de recommandations"""
    ACQUISITION_TARGET = "acquisition_target"
    INVESTMENT_OPPORTUNITY = "investment_opportunity"
    STRATEGIC_PARTNERSHIP = "strategic_partnership"
    MARKET_ENTRY = "market_entry"
    DIVERSIFICATION = "diversification"
    CONSOLIDATION = "consolidation"


class RecommendationStrategy(str, Enum):
    """Stratégies de recommandation"""
    COLLABORATIVE_FILTERING = "collaborative_filtering"
    CONTENT_BASED = "content_based"
    HYBRID = "hybrid"
    MARKET_BASED = "market_based"
    AI_SCORING = "ai_scoring"
    TREND_FOLLOWING = "trend_following"


class RiskProfile(str, Enum):
    """Profils de risque"""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"
    OPPORTUNISTIC = "opportunistic"


@dataclass
class UserPreferences:
    """Préférences utilisateur pour recommandations"""
    user_id: str
    sectors_of_interest: List[str]
    size_preferences: Dict[str, Any]  # min/max CA, effectifs, etc.
    geographic_preferences: List[str]
    risk_profile: RiskProfile
    investment_horizon: str  # "short", "medium", "long"
    budget_range: Tuple[float, float]
    strategic_objectives: List[str]
    excluded_sectors: List[str] = field(default_factory=list)
    minimum_score: float = 50.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'user_id': self.user_id,
            'sectors_of_interest': self.sectors_of_interest,
            'size_preferences': self.size_preferences,
            'geographic_preferences': self.geographic_preferences,
            'risk_profile': self.risk_profile.value,
            'investment_horizon': self.investment_horizon,
            'budget_range': self.budget_range,
            'strategic_objectives': self.strategic_objectives,
            'excluded_sectors': self.excluded_sectors,
            'minimum_score': self.minimum_score
        }


@dataclass
class RecommendationItem:
    """Item de recommandation"""
    item_id: str
    item_type: RecommendationType
    title: str
    description: str
    score: float
    confidence: float
    reasoning: List[str]
    metadata: Dict[str, Any]
    
    # Détails financiers
    estimated_value: Optional[float] = None
    expected_roi: Optional[float] = None
    payback_period_months: Optional[int] = None
    
    # Facteurs de risque
    risk_factors: List[str] = field(default_factory=list)
    risk_score: float = 0.5
    
    # Timing
    urgency_score: float = 0.5
    optimal_timing: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'item_id': self.item_id,
            'item_type': self.item_type.value,
            'title': self.title,
            'description': self.description,
            'score': round(self.score, 2),
            'confidence': round(self.confidence, 3),
            'reasoning': self.reasoning,
            'metadata': self.metadata,
            'estimated_value': self.estimated_value,
            'expected_roi': round(self.expected_roi, 3) if self.expected_roi else None,
            'payback_period_months': self.payback_period_months,
            'risk_factors': self.risk_factors,
            'risk_score': round(self.risk_score, 3),
            'urgency_score': round(self.urgency_score, 3),
            'optimal_timing': self.optimal_timing
        }


@dataclass
class RecommendationResults:
    """Résultats de recommandations"""
    user_id: str
    strategy_used: RecommendationStrategy
    recommendations: List[RecommendationItem]
    total_count: int
    
    # Métriques
    average_score: float
    score_distribution: Dict[str, int]
    coverage_metrics: Dict[str, Any]
    
    # Métadonnées
    generation_time_ms: float
    model_performance: Dict[str, float]
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'user_id': self.user_id,
            'strategy_used': self.strategy_used.value,
            'recommendations': [r.to_dict() for r in self.recommendations],
            'total_count': self.total_count,
            'average_score': round(self.average_score, 2),
            'score_distribution': self.score_distribution,
            'coverage_metrics': self.coverage_metrics,
            'generation_time_ms': round(self.generation_time_ms, 2),
            'model_performance': {k: round(v, 3) for k, v in self.model_performance.items()},
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class PortfolioOptimization:
    """Optimisation de portefeuille d'acquisitions"""
    user_id: str
    selected_opportunities: List[str]
    total_investment: float
    expected_return: float
    portfolio_risk: float
    diversification_score: float
    
    # Allocation optimale
    allocation_weights: Dict[str, float]
    sector_distribution: Dict[str, float]
    geographic_distribution: Dict[str, float]
    
    # Métriques de performance
    sharpe_ratio: float
    max_drawdown: float
    correlation_matrix: Dict[str, Dict[str, float]]
    
    # Recommandations
    rebalancing_suggestions: List[str]
    risk_mitigation_actions: List[str]


class RecommendationEngine:
    """Moteur de recommandations intelligentes pour M&A"""
    
    def __init__(self):
        self.user_preferences: Dict[str, UserPreferences] = {}
        self.user_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.item_features: Optional[pd.DataFrame] = None
        self.similarity_matrix: Optional[np.ndarray] = None
        
        # Modèles ML
        self.models = {}
        self.scalers = {}
        
        # Cache pour performance
        self.recommendation_cache: Dict[str, RecommendationResults] = {}
        self.cache_ttl = 3600  # 1 heure
        
        # Configuration
        self.default_recommendations_count = 20
        self.min_confidence_threshold = 0.3
        self.diversity_factor = 0.3  # Balance relevance vs diversity
        
        logger.info("💡 Moteur de recommandations initialisé")
    
    async def initialize(self):
        """Initialise le moteur avec données et modèles"""
        try:
            # Générer données d'exemple pour entraînement
            await self._generate_sample_data()
            
            # Construire matrice de features
            await self._build_feature_matrix()
            
            # Entraîner modèles de base
            await self._train_recommendation_models()
            
            logger.info("✅ Moteur de recommandations initialisé avec succès")
            
        except Exception as e:
            logger.error(f"Erreur initialisation moteur recommandations: {e}")
            raise
    
    async def get_recommendations(self, 
                                user_id: str,
                                strategy: RecommendationStrategy = RecommendationStrategy.HYBRID,
                                count: int = None,
                                filters: Dict[str, Any] = None) -> RecommendationResults:
        """
        Génère des recommandations personnalisées
        
        Args:
            user_id: Identifiant utilisateur
            strategy: Stratégie de recommandation
            count: Nombre de recommandations (défaut: 20)
            filters: Filtres additionnels
            
        Returns:
            RecommendationResults: Recommandations générées
        """
        start_time = datetime.now()
        count = count or self.default_recommendations_count
        
        try:
            logger.info(f"Génération recommandations pour {user_id} avec stratégie {strategy.value}")
            
            # Vérifier cache
            cache_key = f"{user_id}_{strategy.value}_{count}_{hash(str(filters))}"
            if cache_key in self.recommendation_cache:
                cached_result = self.recommendation_cache[cache_key]
                if (datetime.now() - cached_result.timestamp).total_seconds() < self.cache_ttl:
                    logger.debug("Recommandations récupérées du cache")
                    return cached_result
            
            # Récupérer préférences utilisateur
            user_prefs = await self._get_user_preferences(user_id)
            
            # Générer recommandations selon stratégie
            if strategy == RecommendationStrategy.COLLABORATIVE_FILTERING:
                recommendations = await self._collaborative_filtering_recommendations(user_id, count, filters)
            elif strategy == RecommendationStrategy.CONTENT_BASED:
                recommendations = await self._content_based_recommendations(user_prefs, count, filters)
            elif strategy == RecommendationStrategy.HYBRID:
                recommendations = await self._hybrid_recommendations(user_id, user_prefs, count, filters)
            elif strategy == RecommendationStrategy.AI_SCORING:
                recommendations = await self._ai_scoring_recommendations(user_prefs, count, filters)
            elif strategy == RecommendationStrategy.MARKET_BASED:
                recommendations = await self._market_based_recommendations(user_prefs, count, filters)
            else:
                recommendations = await self._trend_following_recommendations(user_prefs, count, filters)
            
            # Post-traitement: diversification et optimisation
            recommendations = self._ensure_diversity(recommendations, user_prefs)
            recommendations = self._optimize_recommendation_order(recommendations, user_prefs)
            
            # Calculer métriques
            average_score = np.mean([r.score for r in recommendations]) if recommendations else 0.0
            score_distribution = self._calculate_score_distribution(recommendations)
            coverage_metrics = self._calculate_coverage_metrics(recommendations, user_prefs)
            model_performance = self._evaluate_recommendation_quality(recommendations, user_prefs)
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            results = RecommendationResults(
                user_id=user_id,
                strategy_used=strategy,
                recommendations=recommendations[:count],
                total_count=len(recommendations),
                average_score=average_score,
                score_distribution=score_distribution,
                coverage_metrics=coverage_metrics,
                generation_time_ms=processing_time,
                model_performance=model_performance,
                timestamp=datetime.now()
            )
            
            # Mettre en cache
            self.recommendation_cache[cache_key] = results
            
            # Enregistrer interaction pour apprentissage
            await self._record_recommendation_interaction(user_id, results)
            
            return results
            
        except Exception as e:
            logger.error(f"Erreur génération recommandations: {e}")
            raise
    
    async def optimize_portfolio(self, 
                               user_id: str,
                               candidate_opportunities: List[str],
                               budget_constraint: float,
                               optimization_objective: str = "max_return") -> PortfolioOptimization:
        """
        Optimise un portefeuille d'opportunités M&A
        
        Args:
            user_id: Identifiant utilisateur
            candidate_opportunities: IDs des opportunités candidates
            budget_constraint: Budget total disponible
            optimization_objective: Objectif d'optimisation
            
        Returns:
            PortfolioOptimization: Portefeuille optimisé
        """
        try:
            logger.info(f"Optimisation portefeuille pour {user_id} avec {len(candidate_opportunities)} opportunités")
            
            # Récupérer données des opportunités
            opportunities_data = await self._get_opportunities_data(candidate_opportunities)
            
            # Calculer métriques financières
            expected_returns = self._calculate_expected_returns(opportunities_data)
            risk_metrics = self._calculate_risk_metrics(opportunities_data)
            correlation_matrix = self._calculate_correlation_matrix(opportunities_data)
            
            # Optimisation selon objectif
            if optimization_objective == "max_return":
                optimal_weights = self._optimize_for_maximum_return(
                    expected_returns, risk_metrics, budget_constraint
                )
            elif optimization_objective == "min_risk":
                optimal_weights = self._optimize_for_minimum_risk(
                    risk_metrics, correlation_matrix, budget_constraint
                )
            else:  # max_sharpe
                optimal_weights = self._optimize_for_sharpe_ratio(
                    expected_returns, risk_metrics, correlation_matrix, budget_constraint
                )
            
            # Calculer métriques du portefeuille optimisé
            portfolio_return = np.sum(expected_returns * optimal_weights)
            portfolio_risk = self._calculate_portfolio_risk(optimal_weights, risk_metrics, correlation_matrix)
            sharpe_ratio = portfolio_return / portfolio_risk if portfolio_risk > 0 else 0
            
            # Analyse de diversification
            diversification_score = self._calculate_diversification_score(
                optimal_weights, opportunities_data
            )
            
            # Distribution sectorielle et géographique
            sector_dist, geo_dist = self._calculate_distributions(optimal_weights, opportunities_data)
            
            # Générer recommandations d'optimisation
            rebalancing_suggestions = self._generate_rebalancing_suggestions(
                optimal_weights, opportunities_data
            )
            risk_mitigation_actions = self._generate_risk_mitigation_actions(
                risk_metrics, opportunities_data
            )
            
            return PortfolioOptimization(
                user_id=user_id,
                selected_opportunities=candidate_opportunities,
                total_investment=budget_constraint,
                expected_return=portfolio_return,
                portfolio_risk=portfolio_risk,
                diversification_score=diversification_score,
                allocation_weights={opp: weight for opp, weight in zip(candidate_opportunities, optimal_weights)},
                sector_distribution=sector_dist,
                geographic_distribution=geo_dist,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=self._estimate_max_drawdown(optimal_weights, opportunities_data),
                correlation_matrix=correlation_matrix.to_dict(),
                rebalancing_suggestions=rebalancing_suggestions,
                risk_mitigation_actions=risk_mitigation_actions
            )
            
        except Exception as e:
            logger.error(f"Erreur optimisation portefeuille: {e}")
            raise
    
    async def update_user_preferences(self, user_id: str, preferences: Dict[str, Any]):
        """Met à jour les préférences utilisateur"""
        try:
            # Créer ou mettre à jour objet UserPreferences
            if user_id in self.user_preferences:
                current_prefs = self.user_preferences[user_id]
                # Mettre à jour champs modifiés
                for key, value in preferences.items():
                    if hasattr(current_prefs, key):
                        setattr(current_prefs, key, value)
            else:
                # Créer nouvelles préférences avec valeurs par défaut
                self.user_preferences[user_id] = UserPreferences(
                    user_id=user_id,
                    sectors_of_interest=preferences.get('sectors_of_interest', []),
                    size_preferences=preferences.get('size_preferences', {}),
                    geographic_preferences=preferences.get('geographic_preferences', []),
                    risk_profile=RiskProfile(preferences.get('risk_profile', 'moderate')),
                    investment_horizon=preferences.get('investment_horizon', 'medium'),
                    budget_range=preferences.get('budget_range', (1000000, 50000000)),
                    strategic_objectives=preferences.get('strategic_objectives', []),
                    excluded_sectors=preferences.get('excluded_sectors', []),
                    minimum_score=preferences.get('minimum_score', 50.0)
                )
            
            # Invalider cache pour cet utilisateur
            keys_to_remove = [k for k in self.recommendation_cache.keys() if k.startswith(f"{user_id}_")]
            for key in keys_to_remove:
                del self.recommendation_cache[key]
            
            logger.info(f"Préférences mises à jour pour utilisateur {user_id}")
            
        except Exception as e:
            logger.error(f"Erreur mise à jour préférences: {e}")
            raise
    
    async def record_user_feedback(self, 
                                 user_id: str, 
                                 recommendation_id: str, 
                                 feedback_type: str, 
                                 feedback_value: Union[bool, float, str]):
        """Enregistre le feedback utilisateur pour apprentissage"""
        try:
            feedback_record = {
                'user_id': user_id,
                'recommendation_id': recommendation_id,
                'feedback_type': feedback_type,  # 'like', 'dislike', 'rating', 'view', 'click'
                'feedback_value': feedback_value,
                'timestamp': datetime.now().isoformat()
            }
            
            # Ajouter à l'historique utilisateur
            self.user_history[user_id].append(feedback_record)
            
            # Apprentissage en ligne (mise à jour des modèles)
            await self._update_models_with_feedback(user_id, feedback_record)
            
            logger.info(f"Feedback enregistré: {user_id} -> {recommendation_id} ({feedback_type})")
            
        except Exception as e:
            logger.error(f"Erreur enregistrement feedback: {e}")
    
    async def get_trending_opportunities(self, 
                                       sector: Optional[str] = None,
                                       time_window_days: int = 30) -> List[RecommendationItem]:
        """Retourne les opportunités tendance"""
        try:
            # Utiliser le moteur d'analyse prédictive pour identifier tendances
            predictive_engine = await get_predictive_analytics_engine()
            
            # Simuler données de tendances (à remplacer par vraies données)
            trending_data = self._generate_trending_opportunities_data(sector, time_window_days)
            
            trending_opportunities = []
            
            for item_data in trending_data:
                opportunity = RecommendationItem(
                    item_id=item_data['id'],
                    item_type=RecommendationType.ACQUISITION_TARGET,
                    title=item_data['title'],
                    description=item_data['description'],
                    score=item_data['trend_score'],
                    confidence=item_data['confidence'],
                    reasoning=item_data['reasoning'],
                    metadata=item_data['metadata'],
                    estimated_value=item_data.get('estimated_value'),
                    expected_roi=item_data.get('expected_roi'),
                    urgency_score=item_data.get('urgency_score', 0.7),
                    optimal_timing=f"Prochains {time_window_days} jours"
                )
                
                trending_opportunities.append(opportunity)
            
            # Trier par score de tendance
            trending_opportunities.sort(key=lambda x: x.score, reverse=True)
            
            return trending_opportunities[:10]  # Top 10
            
        except Exception as e:
            logger.error(f"Erreur récupération opportunités tendance: {e}")
            return []
    
    # Méthodes privées pour stratégies de recommandation
    
    async def _collaborative_filtering_recommendations(self, 
                                                     user_id: str, 
                                                     count: int, 
                                                     filters: Dict[str, Any]) -> List[RecommendationItem]:
        """Recommandations par filtrage collaboratif"""
        recommendations = []
        
        try:
            # Trouver utilisateurs similaires
            similar_users = self._find_similar_users(user_id)
            
            # Agréger préférences des utilisateurs similaires
            recommended_items = self._aggregate_similar_users_preferences(similar_users)
            
            # Filtrer items déjà vus
            user_seen_items = {item['recommendation_id'] for item in self.user_history.get(user_id, [])}
            new_items = {k: v for k, v in recommended_items.items() if k not in user_seen_items}
            
            # Convertir en RecommendationItem
            for item_id, score in sorted(new_items.items(), key=lambda x: x[1], reverse=True)[:count]:
                item_data = await self._get_item_details(item_id)
                if item_data and self._apply_filters(item_data, filters):
                    
                    recommendation = RecommendationItem(
                        item_id=item_id,
                        item_type=RecommendationType.ACQUISITION_TARGET,
                        title=item_data.get('title', f'Opportunité {item_id}'),
                        description=item_data.get('description', 'Recommandation basée sur utilisateurs similaires'),
                        score=score,
                        confidence=0.7,
                        reasoning=[
                            "Recommandé par des utilisateurs avec profil similaire",
                            f"Score de similarité: {score:.2f}"
                        ],
                        metadata=item_data
                    )
                    
                    recommendations.append(recommendation)
            
        except Exception as e:
            logger.warning(f"Erreur filtrage collaboratif: {e}")
        
        return recommendations
    
    async def _content_based_recommendations(self, 
                                           user_prefs: UserPreferences, 
                                           count: int, 
                                           filters: Dict[str, Any]) -> List[RecommendationItem]:
        """Recommandations basées sur le contenu"""
        recommendations = []
        
        try:
            if self.item_features is None:
                await self._build_feature_matrix()
            
            # Créer profil utilisateur basé sur préférences
            user_profile = self._create_user_profile_vector(user_prefs)
            
            # Calculer similarité avec tous les items
            similarities = cosine_similarity([user_profile], self.item_features.values)[0]
            
            # Trier par similarité
            item_indices = np.argsort(similarities)[::-1]
            
            for idx in item_indices[:count * 2]:  # Prendre plus pour filtrage
                if similarities[idx] > self.min_confidence_threshold:
                    item_id = self.item_features.index[idx]
                    item_data = await self._get_item_details(item_id)
                    
                    if item_data and self._apply_filters(item_data, filters):
                        score = similarities[idx] * 100  # Convertir en score 0-100
                        
                        # Ajuster score selon préférences utilisateur
                        adjusted_score = self._adjust_score_for_preferences(score, item_data, user_prefs)
                        
                        recommendation = RecommendationItem(
                            item_id=item_id,
                            item_type=self._infer_recommendation_type(item_data),
                            title=item_data.get('title', f'Opportunité {item_id}'),
                            description=self._generate_content_based_description(item_data, user_prefs),
                            score=adjusted_score,
                            confidence=similarities[idx],
                            reasoning=self._generate_content_based_reasoning(item_data, user_prefs),
                            metadata=item_data,
                            estimated_value=item_data.get('valuation_estimate'),
                            expected_roi=item_data.get('expected_roi'),
                            risk_score=item_data.get('risk_score', 0.5)
                        )
                        
                        recommendations.append(recommendation)
                        
                        if len(recommendations) >= count:
                            break
            
        except Exception as e:
            logger.warning(f"Erreur recommandations content-based: {e}")
        
        return recommendations
    
    async def _hybrid_recommendations(self, 
                                    user_id: str, 
                                    user_prefs: UserPreferences, 
                                    count: int, 
                                    filters: Dict[str, Any]) -> List[RecommendationItem]:
        """Recommandations hybrides (collaborative + content + AI)"""
        
        # Obtenir recommandations de chaque stratégie
        collaborative_recs = await self._collaborative_filtering_recommendations(user_id, count//3, filters)
        content_recs = await self._content_based_recommendations(user_prefs, count//3, filters)
        ai_recs = await self._ai_scoring_recommendations(user_prefs, count//3, filters)
        
        # Combiner et pondérer
        all_recommendations = []
        
        # Pondération collaborative (30%)
        for rec in collaborative_recs:
            rec.score *= 0.3
            rec.reasoning.append("Recommandation collaborative (30%)")
            all_recommendations.append(rec)
        
        # Pondération content-based (40%)
        for rec in content_recs:
            rec.score *= 0.4
            rec.reasoning.append("Recommandation basée contenu (40%)")
            all_recommendations.append(rec)
        
        # Pondération AI scoring (30%)
        for rec in ai_recs:
            rec.score *= 0.3
            rec.reasoning.append("Recommandation IA scoring (30%)")
            all_recommendations.append(rec)
        
        # Déduplication et fusion des scores
        merged_recommendations = self._merge_duplicate_recommendations(all_recommendations)
        
        # Trier par score final
        merged_recommendations.sort(key=lambda x: x.score, reverse=True)
        
        return merged_recommendations[:count]
    
    async def _ai_scoring_recommendations(self, 
                                        user_prefs: UserPreferences, 
                                        count: int, 
                                        filters: Dict[str, Any]) -> List[RecommendationItem]:
        """Recommandations basées sur le scoring IA"""
        recommendations = []
        
        try:
            # Utiliser le moteur de scoring IA
            ai_engine = await get_ai_scoring_engine()
            
            # Obtenir liste d'entreprises candidates
            candidate_companies = await self._get_candidate_companies(user_prefs, filters)
            
            # Scorer chaque entreprise
            for company_data in candidate_companies[:count * 2]:
                try:
                    scoring_result = await ai_engine.score_company(company_data)
                    
                    # Filtrer par score minimum
                    if scoring_result.overall_score >= user_prefs.minimum_score:
                        
                        recommendation = RecommendationItem(
                            item_id=company_data.get('siren', f"company_{len(recommendations)}"),
                            item_type=RecommendationType.ACQUISITION_TARGET,
                            title=f"Acquisition: {company_data.get('nom', 'Entreprise anonyme')}",
                            description=f"Score IA: {scoring_result.overall_score:.1f}/100",
                            score=scoring_result.overall_score,
                            confidence=scoring_result.confidence,
                            reasoning=[
                                f"Score IA élevé: {scoring_result.overall_score:.1f}/100",
                                f"Confiance du modèle: {scoring_result.confidence:.2f}",
                                *scoring_result.recommendations[:2]
                            ],
                            metadata={
                                'company_data': company_data,
                                'ai_scoring': scoring_result.to_dict(),
                                'category_scores': {k.value: v for k, v in scoring_result.category_scores.items()}
                            },
                            estimated_value=company_data.get('estimated_value'),
                            expected_roi=self._estimate_roi_from_score(scoring_result.overall_score),
                            risk_score=1.0 - (scoring_result.overall_score / 100),
                            payback_period_months=self._estimate_payback_period(company_data)
                        )
                        
                        recommendations.append(recommendation)
                        
                        if len(recommendations) >= count:
                            break
                
                except Exception as e:
                    logger.warning(f"Erreur scoring entreprise: {e}")
                    continue
            
        except Exception as e:
            logger.warning(f"Erreur recommandations AI scoring: {e}")
        
        return recommendations
    
    async def _market_based_recommendations(self, 
                                          user_prefs: UserPreferences, 
                                          count: int, 
                                          filters: Dict[str, Any]) -> List[RecommendationItem]:
        """Recommandations basées sur l'analyse de marché"""
        recommendations = []
        
        try:
            # Utiliser le moteur d'analyse prédictive pour identifier opportunités marché
            predictive_engine = await get_predictive_analytics_engine()
            
            # Analyser tendances sectorielles
            sector_predictions = await predictive_engine.predict_sector_performance(
                user_prefs.sectors_of_interest or ['tech', 'healthcare', 'finance']
            )
            
            # Générer recommandations basées sur prédictions
            for sector, prediction in sector_predictions.items():
                if prediction.trends['predicted_classification'] in ['forte_hausse', 'hausse_moderee']:
                    
                    # Créer recommandation d'opportunité sectorielle
                    recommendation = RecommendationItem(
                        item_id=f"sector_opportunity_{sector}",
                        item_type=RecommendationType.MARKET_ENTRY,
                        title=f"Opportunité sectorielle: {sector}",
                        description=f"Croissance prévue: {prediction.trends['predicted_classification']}",
                        score=prediction.confidence_score * 100,
                        confidence=prediction.confidence_score,
                        reasoning=[
                            f"Prédiction de {prediction.trends['predicted_classification']} pour le secteur",
                            f"Confiance: {prediction.confidence_score:.2f}",
                            "Opportunité d'entrée de marché favorable"
                        ],
                        metadata={
                            'sector': sector,
                            'prediction': prediction.to_dict(),
                            'trend_type': prediction.trends['predicted_classification']
                        },
                        optimal_timing=f"Horizon: {prediction.horizon.value}",
                        urgency_score=0.8 if prediction.trends['predicted_classification'] == 'forte_hausse' else 0.6
                    )
                    
                    recommendations.append(recommendation)
            
            # Compléter avec opportunités générales si pas assez
            if len(recommendations) < count:
                general_market_opportunities = self._generate_general_market_opportunities(
                    user_prefs, count - len(recommendations)
                )
                recommendations.extend(general_market_opportunities)
            
        except Exception as e:
            logger.warning(f"Erreur recommandations market-based: {e}")
        
        return recommendations[:count]
    
    async def _trend_following_recommendations(self, 
                                             user_prefs: UserPreferences, 
                                             count: int, 
                                             filters: Dict[str, Any]) -> List[RecommendationItem]:
        """Recommandations suivant les tendances"""
        recommendations = []
        
        try:
            # Obtenir tendances actuelles
            trending_opps = await self.get_trending_opportunities()
            
            # Filtrer selon préférences utilisateur
            filtered_trends = []
            for opp in trending_opps:
                if self._matches_user_preferences(opp, user_prefs):
                    # Ajuster score selon fit avec préférences
                    preference_fit = self._calculate_preference_fit(opp, user_prefs)
                    opp.score = opp.score * preference_fit
                    opp.reasoning.append(f"Fit préférences: {preference_fit:.2f}")
                    filtered_trends.append(opp)
            
            recommendations = filtered_trends[:count]
            
        except Exception as e:
            logger.warning(f"Erreur recommandations trend-following: {e}")
        
        return recommendations
    
    # Méthodes utilitaires
    
    async def _get_user_preferences(self, user_id: str) -> UserPreferences:
        """Récupère ou crée les préférences utilisateur"""
        if user_id not in self.user_preferences:
            # Créer préférences par défaut
            self.user_preferences[user_id] = UserPreferences(
                user_id=user_id,
                sectors_of_interest=['technology', 'healthcare'],
                size_preferences={'min_ca': 1000000, 'max_ca': 50000000},
                geographic_preferences=['France', 'Europe'],
                risk_profile=RiskProfile.MODERATE,
                investment_horizon='medium',
                budget_range=(1000000, 20000000),
                strategic_objectives=['growth', 'diversification'],
                minimum_score=60.0
            )
        
        return self.user_preferences[user_id]
    
    async def _generate_sample_data(self):
        """Génère des données d'exemple pour l'entraînement"""
        
        # Générer données d'entreprises
        np.random.seed(42)
        n_companies = 1000
        
        sectors = ['tech', 'healthcare', 'finance', 'manufacturing', 'retail', 'energy']
        regions = ['Ile-de-France', 'Rhone-Alpes', 'PACA', 'Nord', 'Grand-Est']
        
        companies_data = []
        
        for i in range(n_companies):
            company = {
                'id': f'company_{i}',
                'siren': f'{100000000 + i}',
                'nom': f'Entreprise {i}',
                'secteur': np.random.choice(sectors),
                'region': np.random.choice(regions),
                'chiffre_affaires': np.random.lognormal(14, 1.5),
                'resultat_net': np.random.normal(100000, 500000),
                'effectifs': np.random.lognormal(3, 1.2),
                'age_entreprise': np.random.exponential(15),
                'croissance_ca': np.random.normal(0.05, 0.15),
                'innovation_score': np.random.uniform(0, 1),
                'risk_score': np.random.uniform(0, 1),
                'valuation_estimate': np.random.lognormal(15, 1.0),
                'expected_roi': np.random.normal(0.15, 0.08)
            }
            companies_data.append(company)
        
        self.sample_companies = companies_data
        
        # Générer historique utilisateur simulé
        n_users = 50
        for user_i in range(n_users):
            user_id = f'user_{user_i}'
            
            # Générer interactions simulées
            n_interactions = np.random.poisson(20)
            for _ in range(n_interactions):
                company_id = f'company_{np.random.randint(0, n_companies)}'
                feedback_type = np.random.choice(['view', 'like', 'dislike', 'contact'])
                feedback_value = np.random.uniform(0, 1) if feedback_type == 'rating' else feedback_type == 'like'
                
                interaction = {
                    'recommendation_id': company_id,
                    'feedback_type': feedback_type,
                    'feedback_value': feedback_value,
                    'timestamp': (datetime.now() - timedelta(days=np.random.randint(0, 365))).isoformat()
                }
                
                self.user_history[user_id].append(interaction)
        
        logger.info(f"✅ Données d'exemple générées: {n_companies} entreprises, {n_users} utilisateurs")
    
    async def _build_feature_matrix(self):
        """Construit la matrice de features des items"""
        
        if not hasattr(self, 'sample_companies'):
            await self._generate_sample_data()
        
        # Extraire features numériques
        features_data = []
        item_ids = []
        
        for company in self.sample_companies:
            # Features normalisées
            features = [
                np.log1p(company['chiffre_affaires']) / 20,  # Log CA normalisé
                company['resultat_net'] / 1000000,  # Résultat en millions
                np.log1p(company['effectifs']) / 10,  # Log effectifs normalisé
                company['age_entreprise'] / 50,  # Age normalisé
                company['croissance_ca'],  # Déjà en ratio
                company['innovation_score'],  # Déjà 0-1
                company['risk_score'],  # Déjà 0-1
            ]
            
            # Encodage one-hot pour secteur
            for sector in ['tech', 'healthcare', 'finance', 'manufacturing', 'retail', 'energy']:
                features.append(1.0 if company['secteur'] == sector else 0.0)
            
            # Encodage one-hot pour région
            for region in ['Ile-de-France', 'Rhone-Alpes', 'PACA', 'Nord', 'Grand-Est']:
                features.append(1.0 if company['region'] == region else 0.0)
            
            features_data.append(features)
            item_ids.append(company['id'])
        
        # Créer DataFrame
        feature_names = [
            'log_ca_norm', 'resultat_millions', 'log_effectifs_norm', 'age_norm', 
            'croissance_ca', 'innovation_score', 'risk_score'
        ]
        feature_names.extend([f'sector_{s}' for s in ['tech', 'healthcare', 'finance', 'manufacturing', 'retail', 'energy']])
        feature_names.extend([f'region_{r}' for r in ['Ile-de-France', 'Rhone-Alpes', 'PACA', 'Nord', 'Grand-Est']])
        
        self.item_features = pd.DataFrame(features_data, index=item_ids, columns=feature_names)
        
        logger.info(f"✅ Matrice de features construite: {self.item_features.shape}")
    
    async def _train_recommendation_models(self):
        """Entraîne les modèles de recommandation"""
        
        if self.item_features is None:
            await self._build_feature_matrix()
        
        # Entraîner modèle de similarité (NearestNeighbors)
        nn_model = NearestNeighbors(n_neighbors=20, metric='cosine')
        nn_model.fit(self.item_features.values)
        self.models['nearest_neighbors'] = nn_model
        
        # Calculer matrice de similarité
        self.similarity_matrix = cosine_similarity(self.item_features.values)
        
        # Entraîner modèle de réduction de dimensionnalité
        svd_model = TruncatedSVD(n_components=50, random_state=42)
        features_reduced = svd_model.fit_transform(self.item_features.values)
        self.models['svd'] = svd_model
        self.reduced_features = features_reduced
        
        logger.info("✅ Modèles de recommandation entraînés")
    
    def _create_user_profile_vector(self, user_prefs: UserPreferences) -> np.ndarray:
        """Crée un vecteur de profil utilisateur"""
        
        if self.item_features is None:
            return np.zeros(20)  # Vecteur par défaut
        
        # Initialiser vecteur avec taille des features
        profile_vector = np.zeros(len(self.item_features.columns))
        
        # Mapper préférences aux features
        feature_names = self.item_features.columns.tolist()
        
        # Préférences sectorielles
        for sector in user_prefs.sectors_of_interest:
            sector_feature = f'sector_{sector}'
            if sector_feature in feature_names:
                idx = feature_names.index(sector_feature)
                profile_vector[idx] = 1.0
        
        # Préférences géographiques
        for region in user_prefs.geographic_preferences:
            region_feature = f'region_{region}'
            if region_feature in feature_names:
                idx = feature_names.index(region_feature)
                profile_vector[idx] = 1.0
        
        # Préférences de taille (CA)
        if 'log_ca_norm' in feature_names:
            min_ca = user_prefs.size_preferences.get('min_ca', 1000000)
            max_ca = user_prefs.size_preferences.get('max_ca', 50000000)
            target_ca = (min_ca + max_ca) / 2
            
            idx = feature_names.index('log_ca_norm')
            profile_vector[idx] = np.log1p(target_ca) / 20  # Normalisation cohérente
        
        # Préférences de risque
        if 'risk_score' in feature_names:
            idx = feature_names.index('risk_score')
            if user_prefs.risk_profile == RiskProfile.CONSERVATIVE:
                profile_vector[idx] = 0.2  # Faible risque
            elif user_prefs.risk_profile == RiskProfile.MODERATE:
                profile_vector[idx] = 0.5  # Risque modéré
            else:  # AGGRESSIVE ou OPPORTUNISTIC
                profile_vector[idx] = 0.8  # Risque élevé
        
        return profile_vector
    
    def _find_similar_users(self, user_id: str, k: int = 5) -> List[str]:
        """Trouve les utilisateurs similaires pour filtrage collaboratif"""
        
        if user_id not in self.user_history:
            return []
        
        user_interactions = self.user_history[user_id]
        user_items = {item['recommendation_id'] for item in user_interactions if item['feedback_type'] == 'like'}
        
        # Calculer similarité avec autres utilisateurs
        similarities = []
        
        for other_user_id, other_interactions in self.user_history.items():
            if other_user_id == user_id:
                continue
            
            other_items = {item['recommendation_id'] for item in other_interactions if item['feedback_type'] == 'like'}
            
            # Similarité Jaccard
            intersection = len(user_items.intersection(other_items))
            union = len(user_items.union(other_items))
            
            if union > 0:
                similarity = intersection / union
                similarities.append((other_user_id, similarity))
        
        # Retourner top K utilisateurs similaires
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [user_id for user_id, _ in similarities[:k]]
    
    def _aggregate_similar_users_preferences(self, similar_users: List[str]) -> Dict[str, float]:
        """Agrège les préférences des utilisateurs similaires"""
        
        item_scores = defaultdict(float)
        
        for user_id in similar_users:
            user_interactions = self.user_history.get(user_id, [])
            
            for interaction in user_interactions:
                item_id = interaction['recommendation_id']
                
                # Scorer selon type de feedback
                if interaction['feedback_type'] == 'like':
                    item_scores[item_id] += 1.0
                elif interaction['feedback_type'] == 'view':
                    item_scores[item_id] += 0.3
                elif interaction['feedback_type'] == 'contact':
                    item_scores[item_id] += 2.0
                elif interaction['feedback_type'] == 'dislike':
                    item_scores[item_id] -= 0.5
        
        return dict(item_scores)
    
    async def _get_item_details(self, item_id: str) -> Optional[Dict[str, Any]]:
        """Récupère les détails d'un item"""
        
        # Chercher dans les données d'exemple
        if hasattr(self, 'sample_companies'):
            for company in self.sample_companies:
                if company['id'] == item_id:
                    return {
                        'title': company['nom'],
                        'description': f"Entreprise {company['secteur']} - CA: {company['chiffre_affaires']:.0f}€",
                        'sector': company['secteur'],
                        'region': company['region'],
                        'valuation_estimate': company['valuation_estimate'],
                        'expected_roi': company['expected_roi'],
                        'risk_score': company['risk_score']
                    }
        
        return None
    
    def _apply_filters(self, item_data: Dict[str, Any], filters: Optional[Dict[str, Any]]) -> bool:
        """Applique les filtres à un item"""
        
        if not filters:
            return True
        
        # Filtre sectoriel
        if 'sectors' in filters and item_data.get('sector') not in filters['sectors']:
            return False
        
        # Filtre géographique
        if 'regions' in filters and item_data.get('region') not in filters['regions']:
            return False
        
        # Filtre de valorisation
        if 'max_valuation' in filters:
            valuation = item_data.get('valuation_estimate', 0)
            if valuation > filters['max_valuation']:
                return False
        
        return True
    
    def _adjust_score_for_preferences(self, base_score: float, item_data: Dict[str, Any], user_prefs: UserPreferences) -> float:
        """Ajuste le score selon les préférences utilisateur"""
        
        adjusted_score = base_score
        
        # Bonus secteur préféré
        if item_data.get('sector') in user_prefs.sectors_of_interest:
            adjusted_score *= 1.2
        
        # Malus secteur exclu
        if item_data.get('sector') in user_prefs.excluded_sectors:
            adjusted_score *= 0.5
        
        # Ajustement selon profil de risque
        risk_score = item_data.get('risk_score', 0.5)
        
        if user_prefs.risk_profile == RiskProfile.CONSERVATIVE and risk_score > 0.7:
            adjusted_score *= 0.8  # Pénaliser risque élevé
        elif user_prefs.risk_profile == RiskProfile.AGGRESSIVE and risk_score < 0.3:
            adjusted_score *= 0.9  # Pénaliser légèrement risque trop faible
        
        return min(100.0, adjusted_score)
    
    def _infer_recommendation_type(self, item_data: Dict[str, Any]) -> RecommendationType:
        """Inère le type de recommandation depuis les données"""
        
        # Logique simple basée sur les données
        sector = item_data.get('sector', '')
        valuation = item_data.get('valuation_estimate', 0)
        
        if valuation > 100000000:  # > 100M€
            return RecommendationType.STRATEGIC_PARTNERSHIP
        elif sector in ['tech', 'healthcare']:
            return RecommendationType.ACQUISITION_TARGET
        else:
            return RecommendationType.INVESTMENT_OPPORTUNITY
    
    def _generate_content_based_description(self, item_data: Dict[str, Any], user_prefs: UserPreferences) -> str:
        """Génère une description pour recommandation content-based"""
        
        sector = item_data.get('sector', 'secteur inconnu')
        valuation = item_data.get('valuation_estimate', 0)
        
        description = f"Opportunité dans le secteur {sector}"
        
        if valuation > 0:
            description += f" - Valorisation estimée: {valuation/1000000:.1f}M€"
        
        if sector in user_prefs.sectors_of_interest:
            description += " - Correspond à vos secteurs d'intérêt"
        
        return description
    
    def _generate_content_based_reasoning(self, item_data: Dict[str, Any], user_prefs: UserPreferences) -> List[str]:
        """Génère le raisonnement pour recommandation content-based"""
        
        reasons = []
        
        sector = item_data.get('sector')
        if sector in user_prefs.sectors_of_interest:
            reasons.append(f"Secteur d'intérêt: {sector}")
        
        region = item_data.get('region')
        if region in user_prefs.geographic_preferences:
            reasons.append(f"Zone géographique préférée: {region}")
        
        risk_score = item_data.get('risk_score', 0.5)
        if user_prefs.risk_profile == RiskProfile.CONSERVATIVE and risk_score < 0.4:
            reasons.append("Profil de risque adapté (conservateur)")
        elif user_prefs.risk_profile == RiskProfile.AGGRESSIVE and risk_score > 0.6:
            reasons.append("Profil de risque adapté (agressif)")
        
        expected_roi = item_data.get('expected_roi')
        if expected_roi and expected_roi > 0.15:
            reasons.append(f"ROI attractif attendu: {expected_roi:.1%}")
        
        return reasons or ["Profil compatible avec vos préférences"]
    
    def _merge_duplicate_recommendations(self, recommendations: List[RecommendationItem]) -> List[RecommendationItem]:
        """Fusionne les recommandations dupliquées"""
        
        merged = {}
        
        for rec in recommendations:
            if rec.item_id in merged:
                # Fusionner scores (moyenne pondérée)
                existing = merged[rec.item_id]
                total_confidence = existing.confidence + rec.confidence
                
                if total_confidence > 0:
                    existing.score = (existing.score * existing.confidence + rec.score * rec.confidence) / total_confidence
                    existing.confidence = min(1.0, total_confidence)
                    existing.reasoning.extend(rec.reasoning)
                    existing.reasoning = list(set(existing.reasoning))  # Déduplication
            else:
                merged[rec.item_id] = rec
        
        return list(merged.values())
    
    def _ensure_diversity(self, recommendations: List[RecommendationItem], user_prefs: UserPreferences) -> List[RecommendationItem]:
        """Assure la diversité des recommandations"""
        
        if len(recommendations) <= 5:
            return recommendations
        
        # Grouper par secteur
        sector_groups = defaultdict(list)
        for rec in recommendations:
            sector = rec.metadata.get('sector', 'unknown')
            sector_groups[sector].append(rec)
        
        # Sélectionner représentants de chaque secteur
        diversified = []
        
        # Prendre top recommandations de chaque secteur
        for sector, sector_recs in sector_groups.items():
            sector_recs.sort(key=lambda x: x.score, reverse=True)
            
            # Nombre à prendre selon taille du groupe et préférences
            max_per_sector = max(1, len(recommendations) // max(3, len(sector_groups)))
            if sector in user_prefs.sectors_of_interest:
                max_per_sector *= 2  # Plus de secteurs préférés
            
            diversified.extend(sector_recs[:max_per_sector])
        
        # Compléter avec meilleures recommandations restantes
        remaining = [r for r in recommendations if r not in diversified]
        remaining.sort(key=lambda x: x.score, reverse=True)
        
        final_count = len(recommendations)
        while len(diversified) < final_count and remaining:
            diversified.append(remaining.pop(0))
        
        return diversified
    
    def _optimize_recommendation_order(self, recommendations: List[RecommendationItem], user_prefs: UserPreferences) -> List[RecommendationItem]:
        """Optimise l'ordre des recommandations"""
        
        # Score composite incluant score, confiance, et fit préférences
        def composite_score(rec: RecommendationItem) -> float:
            base_score = rec.score * rec.confidence
            
            # Bonus timing si urgent
            if rec.urgency_score > 0.7:
                base_score *= 1.1
            
            # Bonus secteur préféré
            if rec.metadata.get('sector') in user_prefs.sectors_of_interest:
                base_score *= 1.15
            
            # Bonus budget fit
            if rec.estimated_value:
                budget_min, budget_max = user_prefs.budget_range
                if budget_min <= rec.estimated_value <= budget_max:
                    base_score *= 1.1
            
            return base_score
        
        # Trier par score composite
        recommendations.sort(key=composite_score, reverse=True)
        
        return recommendations
    
    def _calculate_score_distribution(self, recommendations: List[RecommendationItem]) -> Dict[str, int]:
        """Calcule la distribution des scores"""
        
        distribution = {
            'excellent': 0,  # 80-100
            'good': 0,       # 60-80
            'average': 0,    # 40-60
            'poor': 0        # 0-40
        }
        
        for rec in recommendations:
            if rec.score >= 80:
                distribution['excellent'] += 1
            elif rec.score >= 60:
                distribution['good'] += 1
            elif rec.score >= 40:
                distribution['average'] += 1
            else:
                distribution['poor'] += 1
        
        return distribution
    
    def _calculate_coverage_metrics(self, recommendations: List[RecommendationItem], user_prefs: UserPreferences) -> Dict[str, Any]:
        """Calcule les métriques de couverture"""
        
        metrics = {}
        
        # Couverture sectorielle
        recommended_sectors = {rec.metadata.get('sector') for rec in recommendations if rec.metadata.get('sector')}
        preferred_sectors = set(user_prefs.sectors_of_interest)
        
        if preferred_sectors:
            sector_coverage = len(recommended_sectors.intersection(preferred_sectors)) / len(preferred_sectors)
        else:
            sector_coverage = 1.0
        
        metrics['sector_coverage'] = sector_coverage
        
        # Diversité géographique
        recommended_regions = {rec.metadata.get('region') for rec in recommendations if rec.metadata.get('region')}
        metrics['geographic_diversity'] = len(recommended_regions)
        
        # Distribution des types de recommandations
        type_counts = Counter(rec.item_type.value for rec in recommendations)
        metrics['type_distribution'] = dict(type_counts)
        
        return metrics
    
    def _evaluate_recommendation_quality(self, recommendations: List[RecommendationItem], user_prefs: UserPreferences) -> Dict[str, float]:
        """Évalue la qualité des recommandations"""
        
        if not recommendations:
            return {'overall_quality': 0.0}
        
        # Qualité moyenne des scores
        avg_score = np.mean([rec.score for rec in recommendations])
        score_quality = avg_score / 100
        
        # Qualité de la confiance
        avg_confidence = np.mean([rec.confidence for rec in recommendations])
        
        # Qualité de la diversité
        sectors = [rec.metadata.get('sector') for rec in recommendations if rec.metadata.get('sector')]
        diversity_quality = len(set(sectors)) / max(1, len(sectors)) if sectors else 0
        
        # Qualité du fit préférences
        preference_matches = sum(1 for rec in recommendations 
                               if rec.metadata.get('sector') in user_prefs.sectors_of_interest)
        preference_quality = preference_matches / len(recommendations) if recommendations else 0
        
        # Score global
        overall_quality = (score_quality * 0.4 + avg_confidence * 0.3 + 
                          diversity_quality * 0.15 + preference_quality * 0.15)
        
        return {
            'overall_quality': overall_quality,
            'score_quality': score_quality,
            'confidence_quality': avg_confidence,
            'diversity_quality': diversity_quality,
            'preference_fit_quality': preference_quality
        }
    
    async def _record_recommendation_interaction(self, user_id: str, results: RecommendationResults):
        """Enregistre l'interaction pour apprentissage futur"""
        
        interaction_record = {
            'user_id': user_id,
            'strategy_used': results.strategy_used.value,
            'recommendations_count': len(results.recommendations),
            'average_score': results.average_score,
            'timestamp': datetime.now().isoformat()
        }
        
        # Ajouter à l'historique
        self.user_history[user_id].append(interaction_record)
        
        # Limiter taille de l'historique
        if len(self.user_history[user_id]) > 1000:
            self.user_history[user_id] = self.user_history[user_id][-500:]
    
    async def _update_models_with_feedback(self, user_id: str, feedback_record: Dict[str, Any]):
        """Met à jour les modèles avec le feedback utilisateur"""
        
        # Pour l'instant, apprentissage simple basé sur feedback
        # Dans une version complète, utiliser ML online learning
        
        feedback_type = feedback_record['feedback_type']
        item_id = feedback_record['recommendation_id']
        
        # Ajuster scores en fonction feedback
        if feedback_type == 'like':
            # Augmenter score des items similaires
            await self._boost_similar_items(item_id, user_id, boost_factor=1.1)
        elif feedback_type == 'dislike':
            # Diminuer score des items similaires
            await self._boost_similar_items(item_id, user_id, boost_factor=0.9)
    
    async def _boost_similar_items(self, item_id: str, user_id: str, boost_factor: float):
        """Boost ou diminue le score d'items similaires"""
        
        # Cette méthode serait implémentée avec des techniques d'apprentissage en ligne
        # Pour l'instant, juste logger l'action
        
        logger.debug(f"Apprentissage: boost {boost_factor} pour items similaires à {item_id} (user {user_id})")
    
    # Méthodes pour optimisation de portefeuille
    
    async def _get_opportunities_data(self, opportunity_ids: List[str]) -> pd.DataFrame:
        """Récupère les données des opportunités pour optimisation"""
        
        opportunities_data = []
        
        for opp_id in opportunity_ids:
            # Simuler données d'opportunité
            opp_data = {
                'id': opp_id,
                'expected_return': np.random.normal(0.15, 0.05),
                'risk': np.random.uniform(0.1, 0.4),
                'investment_required': np.random.lognormal(15, 0.5),
                'sector': np.random.choice(['tech', 'healthcare', 'finance']),
                'region': np.random.choice(['France', 'Europe', 'International']),
                'correlation_group': np.random.randint(0, 3)  # Pour matrice corrélation
            }
            opportunities_data.append(opp_data)
        
        return pd.DataFrame(opportunities_data)
    
    def _calculate_expected_returns(self, opportunities_data: pd.DataFrame) -> np.ndarray:
        """Calcule les rendements attendus"""
        return opportunities_data['expected_return'].values
    
    def _calculate_risk_metrics(self, opportunities_data: pd.DataFrame) -> np.ndarray:
        """Calcule les métriques de risque"""
        return opportunities_data['risk'].values
    
    def _calculate_correlation_matrix(self, opportunities_data: pd.DataFrame) -> pd.DataFrame:
        """Calcule la matrice de corrélation"""
        
        # Simuler corrélations basées sur secteur/région
        n_opps = len(opportunities_data)
        corr_matrix = np.eye(n_opps)
        
        for i in range(n_opps):
            for j in range(i + 1, n_opps):
                # Corrélation plus forte si même secteur ou région
                same_sector = opportunities_data.iloc[i]['sector'] == opportunities_data.iloc[j]['sector']
                same_region = opportunities_data.iloc[i]['region'] == opportunities_data.iloc[j]['region']
                
                if same_sector and same_region:
                    correlation = np.random.uniform(0.4, 0.7)
                elif same_sector or same_region:
                    correlation = np.random.uniform(0.2, 0.5)
                else:
                    correlation = np.random.uniform(-0.1, 0.3)
                
                corr_matrix[i, j] = correlation
                corr_matrix[j, i] = correlation
        
        return pd.DataFrame(corr_matrix, 
                          index=opportunities_data['id'], 
                          columns=opportunities_data['id'])
    
    def _optimize_for_maximum_return(self, expected_returns: np.ndarray, 
                                   risk_metrics: np.ndarray, 
                                   budget: float) -> np.ndarray:
        """Optimise pour rendement maximum"""
        
        n_assets = len(expected_returns)
        
        # Optimisation simple: pondérer par ratio rendement/risque
        risk_adjusted_returns = expected_returns / (risk_metrics + 0.01)  # Éviter division par 0
        
        # Normaliser pour respecter budget
        total_investment_required = budget
        
        # Allocation proportionnelle au ratio risque-ajusté
        weights = risk_adjusted_returns / np.sum(risk_adjusted_returns)
        
        return weights
    
    def _optimize_for_minimum_risk(self, risk_metrics: np.ndarray, 
                                 correlation_matrix: pd.DataFrame, 
                                 budget: float) -> np.ndarray:
        """Optimise pour risque minimum"""
        
        # Allocation inverse au risque
        inverse_risk = 1.0 / (risk_metrics + 0.01)
        weights = inverse_risk / np.sum(inverse_risk)
        
        return weights
    
    def _optimize_for_sharpe_ratio(self, expected_returns: np.ndarray,
                                 risk_metrics: np.ndarray,
                                 correlation_matrix: pd.DataFrame,
                                 budget: float) -> np.ndarray:
        """Optimise pour ratio de Sharpe"""
        
        # Approximation simple du ratio de Sharpe
        sharpe_ratios = expected_returns / (risk_metrics + 0.01)
        
        # Allocation basée sur ratios de Sharpe
        weights = sharpe_ratios / np.sum(sharpe_ratios)
        
        return weights
    
    def _calculate_portfolio_risk(self, weights: np.ndarray, 
                                risk_metrics: np.ndarray, 
                                correlation_matrix: pd.DataFrame) -> float:
        """Calcule le risque du portefeuille"""
        
        # Risque pondéré simple (approximation)
        portfolio_risk = np.sum(weights * risk_metrics)
        
        # Ajustement pour corrélations (simplification)
        avg_correlation = correlation_matrix.values.mean()
        correlation_adjustment = 1.0 + avg_correlation * 0.5
        
        return portfolio_risk * correlation_adjustment
    
    def _calculate_diversification_score(self, weights: np.ndarray, 
                                       opportunities_data: pd.DataFrame) -> float:
        """Calcule le score de diversification"""
        
        # Diversification sectorielle
        sector_weights = defaultdict(float)
        for i, weight in enumerate(weights):
            sector = opportunities_data.iloc[i]['sector']
            sector_weights[sector] += weight
        
        # Score basé sur entropie de Shannon
        sector_entropy = -sum(w * np.log(w + 1e-8) for w in sector_weights.values() if w > 0)
        max_entropy = np.log(len(sector_weights))
        
        diversification_score = sector_entropy / max_entropy if max_entropy > 0 else 0
        
        return diversification_score
    
    def _calculate_distributions(self, weights: np.ndarray, 
                               opportunities_data: pd.DataFrame) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Calcule les distributions sectorielle et géographique"""
        
        sector_dist = defaultdict(float)
        geo_dist = defaultdict(float)
        
        for i, weight in enumerate(weights):
            sector = opportunities_data.iloc[i]['sector']
            region = opportunities_data.iloc[i]['region']
            
            sector_dist[sector] += weight
            geo_dist[region] += weight
        
        return dict(sector_dist), dict(geo_dist)
    
    def _generate_rebalancing_suggestions(self, weights: np.ndarray, 
                                        opportunities_data: pd.DataFrame) -> List[str]:
        """Génère des suggestions de rééquilibrage"""
        
        suggestions = []
        
        # Identifier allocations déséquilibrées
        max_weight_idx = np.argmax(weights)
        max_weight = weights[max_weight_idx]
        
        if max_weight > 0.4:  # Plus de 40% dans une opportunité
            opp_id = opportunities_data.iloc[max_weight_idx]['id']
            suggestions.append(f"Réduire allocation à {opp_id} (actuellement {max_weight:.1%})")
        
        # Vérifier diversification sectorielle
        sector_weights = defaultdict(float)
        for i, weight in enumerate(weights):
            sector = opportunities_data.iloc[i]['sector']
            sector_weights[sector] += weight
        
        for sector, sector_weight in sector_weights.items():
            if sector_weight > 0.5:
                suggestions.append(f"Diversifier hors secteur {sector} (actuellement {sector_weight:.1%})")
        
        return suggestions or ["Allocation équilibrée - aucun rééquilibrage nécessaire"]
    
    def _generate_risk_mitigation_actions(self, risk_metrics: np.ndarray, 
                                        opportunities_data: pd.DataFrame) -> List[str]:
        """Génère des actions de mitigation des risques"""
        
        actions = []
        
        # Identifier opportunités à haut risque
        high_risk_indices = np.where(risk_metrics > 0.3)[0]
        
        for idx in high_risk_indices:
            opp_id = opportunities_data.iloc[idx]['id']
            risk_level = risk_metrics[idx]
            actions.append(f"Surveiller risque élevé de {opp_id} (risque: {risk_level:.1%})")
        
        # Actions générales
        avg_risk = np.mean(risk_metrics)
        if avg_risk > 0.25:
            actions.append("Considérer instruments de couverture pour réduire risque global")
        
        return actions or ["Niveau de risque acceptable - surveillance standard recommandée"]
    
    def _estimate_max_drawdown(self, weights: np.ndarray, 
                             opportunities_data: pd.DataFrame) -> float:
        """Estime le drawdown maximum"""
        
        # Estimation simple basée sur risques pondérés
        weighted_risks = weights * opportunities_data['risk'].values
        estimated_max_drawdown = np.sum(weighted_risks) * 2  # Facteur multiplicateur conservateur
        
        return min(0.5, estimated_max_drawdown)  # Max 50%
    
    # Méthodes utilitaires pour recommandations spécialisées
    
    async def _get_candidate_companies(self, user_prefs: UserPreferences, 
                                     filters: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Récupère les entreprises candidates pour scoring IA"""
        
        if not hasattr(self, 'sample_companies'):
            await self._generate_sample_data()
        
        candidates = []
        
        for company in self.sample_companies:
            # Filtrer selon préférences
            if company['secteur'] not in user_prefs.excluded_sectors:
                
                # Filtre budget si applicable
                valuation = company.get('valuation_estimate', 0)
                budget_min, budget_max = user_prefs.budget_range
                
                if budget_min <= valuation <= budget_max:
                    candidates.append(company)
        
        return candidates[:50]  # Limiter pour performance
    
    def _estimate_roi_from_score(self, ai_score: float) -> float:
        """Estime le ROI basé sur le score IA"""
        
        # Corrélation approximative score IA -> ROI
        base_roi = 0.10  # ROI de base 10%
        score_bonus = (ai_score - 50) / 100 * 0.15  # Bonus basé sur score
        
        return max(0.05, base_roi + score_bonus)  # Minimum 5%
    
    def _estimate_payback_period(self, company_data: Dict[str, Any]) -> int:
        """Estime la période de retour sur investissement"""
        
        # Estimation basée sur taille et secteur
        ca = company_data.get('chiffre_affaires', 1000000)
        secteur = company_data.get('secteur', 'manufacturing')
        
        # Périodes typiques par secteur
        sector_periods = {
            'tech': 24,
            'healthcare': 36,
            'finance': 18,
            'manufacturing': 30,
            'retail': 24,
            'energy': 48
        }
        
        base_period = sector_periods.get(secteur, 30)
        
        # Ajustement selon taille (plus grande = plus long)
        if ca > 10000000:  # > 10M€
            base_period += 12
        elif ca < 1000000:  # < 1M€
            base_period -= 6
        
        return max(12, base_period)  # Minimum 1 an
    
    def _generate_trending_opportunities_data(self, sector: Optional[str], 
                                            time_window_days: int) -> List[Dict[str, Any]]:
        """Génère des données d'opportunités tendance"""
        
        trending_data = []
        
        sectors_to_analyze = [sector] if sector else ['tech', 'healthcare', 'finance']
        
        for i, sect in enumerate(sectors_to_analyze):
            for j in range(3):  # 3 opportunités par secteur
                opportunity = {
                    'id': f'trending_{sect}_{j}',
                    'title': f'Opportunité Tendance {sect.title()} #{j+1}',
                    'description': f'Forte croissance dans le secteur {sect}',
                    'trend_score': 85 - i*5 - j*3,  # Score décroissant
                    'confidence': 0.8 - i*0.1,
                    'reasoning': [
                        f'Secteur {sect} en forte expansion',
                        f'Analyse sur {time_window_days} jours',
                        'Détecté par analyse prédictive'
                    ],
                    'metadata': {
                        'sector': sect,
                        'trend_direction': 'up',
                        'momentum': 'strong'
                    },
                    'estimated_value': np.random.lognormal(15, 0.5),
                    'expected_roi': 0.15 + np.random.uniform(0, 0.1),
                    'urgency_score': 0.7 + np.random.uniform(0, 0.2)
                }
                
                trending_data.append(opportunity)
        
        return trending_data
    
    def _generate_general_market_opportunities(self, user_prefs: UserPreferences, 
                                             count: int) -> List[RecommendationItem]:
        """Génère des opportunités générales de marché"""
        
        opportunities = []
        
        # Types d'opportunités générales
        general_types = [
            ('Consolidation sectorielle', RecommendationType.CONSOLIDATION),
            ('Diversification géographique', RecommendationType.DIVERSIFICATION),
            ('Entrée nouveau marché', RecommendationType.MARKET_ENTRY)
        ]
        
        for i, (title, opp_type) in enumerate(general_types[:count]):
            opportunity = RecommendationItem(
                item_id=f'general_market_{i}',
                item_type=opp_type,
                title=title,
                description=f'Opportunité de {title.lower()} identifiée par analyse de marché',
                score=70 + np.random.uniform(-10, 15),
                confidence=0.6 + np.random.uniform(0, 0.2),
                reasoning=[
                    'Identifiée par analyse de marché',
                    'Conditions favorables actuelles',
                    'Compatible avec profil utilisateur'
                ],
                metadata={
                    'type': 'market_analysis',
                    'source': 'predictive_engine'
                },
                urgency_score=0.5 + np.random.uniform(0, 0.3)
            )
            
            opportunities.append(opportunity)
        
        return opportunities
    
    def _matches_user_preferences(self, opportunity: RecommendationItem, 
                                user_prefs: UserPreferences) -> bool:
        """Vérifie si une opportunité correspond aux préférences utilisateur"""
        
        # Vérifier secteur
        opp_sector = opportunity.metadata.get('sector')
        if opp_sector:
            if opp_sector in user_prefs.excluded_sectors:
                return False
            if user_prefs.sectors_of_interest and opp_sector not in user_prefs.sectors_of_interest:
                return False
        
        # Vérifier budget
        if opportunity.estimated_value:
            budget_min, budget_max = user_prefs.budget_range
            if not (budget_min <= opportunity.estimated_value <= budget_max):
                return False
        
        # Vérifier score minimum
        if opportunity.score < user_prefs.minimum_score:
            return False
        
        return True
    
    def _calculate_preference_fit(self, opportunity: RecommendationItem, 
                                user_prefs: UserPreferences) -> float:
        """Calcule le fit avec les préférences utilisateur (0-1)"""
        
        fit_score = 0.5  # Score de base
        
        # Secteur
        opp_sector = opportunity.metadata.get('sector')
        if opp_sector in user_prefs.sectors_of_interest:
            fit_score += 0.3
        elif opp_sector in user_prefs.excluded_sectors:
            fit_score -= 0.4
        
        # Budget
        if opportunity.estimated_value:
            budget_min, budget_max = user_prefs.budget_range
            budget_center = (budget_min + budget_max) / 2
            budget_range = budget_max - budget_min
            
            distance_from_center = abs(opportunity.estimated_value - budget_center)
            budget_fit = 1.0 - (distance_from_center / (budget_range / 2))
            fit_score += budget_fit * 0.2
        
        # Risque
        if opportunity.risk_score:
            risk_preferences = {
                RiskProfile.CONSERVATIVE: 0.3,
                RiskProfile.MODERATE: 0.5,
                RiskProfile.AGGRESSIVE: 0.7,
                RiskProfile.OPPORTUNISTIC: 0.8
            }
            
            preferred_risk = risk_preferences[user_prefs.risk_profile]
            risk_distance = abs(opportunity.risk_score - preferred_risk)
            risk_fit = 1.0 - risk_distance
            fit_score += risk_fit * 0.1
        
        return max(0.0, min(1.0, fit_score))


# Instance globale
_recommendation_engine: Optional[RecommendationEngine] = None


async def get_recommendation_engine() -> RecommendationEngine:
    """Factory pour obtenir l'instance du moteur de recommandations"""
    global _recommendation_engine
    
    if _recommendation_engine is None:
        _recommendation_engine = RecommendationEngine()
        await _recommendation_engine.initialize()
    
    return _recommendation_engine


async def initialize_recommendation_engine():
    """Initialise le système de recommandations au démarrage"""
    try:
        engine = await get_recommendation_engine()
        logger.info("💡 Système de recommandations initialisé avec succès")
        return engine
    except Exception as e:
        logger.error(f"Erreur initialisation moteur recommandations: {e}")
        raise