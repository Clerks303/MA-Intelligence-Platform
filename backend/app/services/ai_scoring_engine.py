"""
Système de scoring IA avancé pour M&A Intelligence Platform
US-008: Intelligence Artificielle et Machine Learning

Ce module implémente un système de scoring sophistiqué utilisant:
- Machine Learning pour l'évaluation du potentiel M&A
- Analyse prédictive des opportunités
- Score multi-critères avec pondération intelligente
- Apprentissage continu basé sur les résultats
"""

import asyncio
import json
import pickle
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import diskcache

from app.config import settings
from app.core.logging_system import get_logger, LogCategory
from app.models.company import Company
from app.models.schemas import CompanyCreate

logger = get_logger("ai_scoring_engine", LogCategory.ML)


class ScoringModel(str, Enum):
    """Types de modèles de scoring disponibles"""
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting" 
    LINEAR_REGRESSION = "linear_regression"
    RIDGE_REGRESSION = "ridge_regression"
    ENSEMBLE = "ensemble"


class ScoreCategory(str, Enum):
    """Catégories de scores"""
    FINANCIAL_HEALTH = "financial_health"
    GROWTH_POTENTIAL = "growth_potential"
    MARKET_POSITION = "market_position"
    OPERATIONAL_EFFICIENCY = "operational_efficiency"
    STRATEGIC_FIT = "strategic_fit"
    RISK_ASSESSMENT = "risk_assessment"


@dataclass
class ScoringFeatures:
    """Features utilisées pour le scoring ML"""
    
    # Features financières
    chiffre_affaires: float = 0.0
    resultat_net: float = 0.0
    capitaux_propres: float = 0.0
    total_bilan: float = 0.0
    endettement: float = 0.0
    
    # Features de croissance
    croissance_ca_1an: float = 0.0
    croissance_ca_3ans: float = 0.0
    evolution_effectifs: float = 0.0
    
    # Features sectorielles
    secteur_code: int = 0
    secteur_performance: float = 0.0
    concurrence_locale: float = 0.0
    
    # Features géographiques
    region_code: int = 0
    densite_population: float = 0.0
    pib_regional: float = 0.0
    
    # Features temporelles
    age_entreprise: int = 0
    stabilite_dirigeants: float = 0.0
    
    # Features qualitatives (encodées)
    qualite_site_web: float = 0.0
    presence_digitale: float = 0.0
    innovation_score: float = 0.0
    
    def to_array(self) -> np.ndarray:
        """Convertit les features en array numpy"""
        return np.array([
            self.chiffre_affaires, self.resultat_net, self.capitaux_propres,
            self.total_bilan, self.endettement, self.croissance_ca_1an,
            self.croissance_ca_3ans, self.evolution_effectifs, self.secteur_code,
            self.secteur_performance, self.concurrence_locale, self.region_code,
            self.densite_population, self.pib_regional, self.age_entreprise,
            self.stabilite_dirigeants, self.qualite_site_web, self.presence_digitale,
            self.innovation_score
        ])
    
    @classmethod
    def get_feature_names(cls) -> List[str]:
        """Retourne les noms des features"""
        return [
            'chiffre_affaires', 'resultat_net', 'capitaux_propres',
            'total_bilan', 'endettement', 'croissance_ca_1an',
            'croissance_ca_3ans', 'evolution_effectifs', 'secteur_code',
            'secteur_performance', 'concurrence_locale', 'region_code',
            'densite_population', 'pib_regional', 'age_entreprise',
            'stabilite_dirigeants', 'qualite_site_web', 'presence_digitale',
            'innovation_score'
        ]


@dataclass
class ScoringResult:
    """Résultat détaillé du scoring IA"""
    
    company_id: str
    overall_score: float
    confidence: float
    category_scores: Dict[ScoreCategory, float]
    feature_importance: Dict[str, float]
    model_used: ScoringModel
    processing_time_ms: float
    timestamp: datetime
    explanation: Dict[str, Any]
    recommendations: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire pour JSON"""
        return {
            'company_id': self.company_id,
            'overall_score': round(self.overall_score, 2),
            'confidence': round(self.confidence, 3),
            'category_scores': {k.value: round(v, 2) for k, v in self.category_scores.items()},
            'feature_importance': {k: round(v, 3) for k, v in self.feature_importance.items()},
            'model_used': self.model_used.value,
            'processing_time_ms': round(self.processing_time_ms, 2),
            'timestamp': self.timestamp.isoformat(),
            'explanation': self.explanation,
            'recommendations': self.recommendations
        }


class ModelMetrics:
    """Métriques de performance des modèles"""
    
    def __init__(self):
        self.accuracy_metrics = {
            'mse': [],
            'mae': [], 
            'r2': [],
            'cross_val_scores': []
        }
        self.prediction_history = []
        self.feature_importance_history = []
        
    def add_prediction(self, actual: float, predicted: float, confidence: float):
        """Ajoute une prédiction pour calculer les métriques"""
        self.prediction_history.append({
            'actual': actual,
            'predicted': predicted,
            'confidence': confidence,
            'error': abs(actual - predicted),
            'timestamp': datetime.now()
        })
    
    def calculate_metrics(self) -> Dict[str, float]:
        """Calcule les métriques de performance"""
        if not self.prediction_history:
            return {}
        
        actual_values = [p['actual'] for p in self.prediction_history]
        predicted_values = [p['predicted'] for p in self.prediction_history]
        
        return {
            'mse': mean_squared_error(actual_values, predicted_values),
            'mae': mean_absolute_error(actual_values, predicted_values),
            'r2': r2_score(actual_values, predicted_values),
            'prediction_count': len(self.prediction_history),
            'average_error': np.mean([p['error'] for p in self.prediction_history]),
            'confidence_correlation': np.corrcoef(
                [p['confidence'] for p in self.prediction_history],
                [1/p['error'] if p['error'] > 0 else 1 for p in self.prediction_history]
            )[0, 1] if len(self.prediction_history) > 1 else 0
        }


class AICompanyScoringEngine:
    """Moteur de scoring IA pour évaluer le potentiel M&A des entreprises"""
    
    def __init__(self, cache_ttl: int = 3600):
        """
        Initialise le moteur de scoring IA
        
        Args:
            cache_ttl: Durée de vie du cache en secondes (défaut: 1 heure)
        """
        self.cache = diskcache.Cache('/tmp/ma_intelligence_ml_cache')
        self.cache_ttl = cache_ttl
        
        # Modèles ML
        self.models: Dict[ScoringModel, Any] = {}
        self.scalers: Dict[ScoringModel, StandardScaler] = {}
        self.label_encoders: Dict[str, LabelEncoder] = {}
        
        # Métriques et historique
        self.model_metrics: Dict[ScoringModel, ModelMetrics] = {
            model: ModelMetrics() for model in ScoringModel
        }
        
        # Configuration des poids par catégorie
        self.category_weights = {
            ScoreCategory.FINANCIAL_HEALTH: 0.25,
            ScoreCategory.GROWTH_POTENTIAL: 0.20,
            ScoreCategory.MARKET_POSITION: 0.15,
            ScoreCategory.OPERATIONAL_EFFICIENCY: 0.15,
            ScoreCategory.STRATEGIC_FIT: 0.15,
            ScoreCategory.RISK_ASSESSMENT: 0.10
        }
        
        # Dataset d'entraînement synthétique
        self.training_data: Optional[pd.DataFrame] = None
        self.is_trained = False
        
        logger.info("🤖 Moteur de scoring IA initialisé")
    
    async def initialize_models(self):
        """Initialise et entraîne les modèles ML"""
        try:
            # Générer ou charger données d'entraînement
            await self._load_or_generate_training_data()
            
            # Entraîner les modèles
            await self._train_models()
            
            self.is_trained = True
            logger.info("✅ Modèles IA entraînés avec succès")
            
        except Exception as e:
            logger.error(f"Erreur initialisation modèles IA: {e}")
            raise
    
    async def score_company(self, 
                           company_data: Dict[str, Any],
                           model: ScoringModel = ScoringModel.ENSEMBLE,
                           use_cache: bool = True) -> ScoringResult:
        """
        Score une entreprise en utilisant l'IA
        
        Args:
            company_data: Données de l'entreprise
            model: Modèle à utiliser
            use_cache: Utiliser le cache si disponible
            
        Returns:
            ScoringResult: Résultat détaillé du scoring
        """
        start_time = datetime.now()
        
        try:
            # Vérifier cache
            cache_key = self._generate_cache_key(company_data, model)
            
            if use_cache:
                cached_result = self.cache.get(cache_key)
                if cached_result:
                    logger.debug(f"Score récupéré du cache pour {company_data.get('siren', 'unknown')}")
                    return ScoringResult(**cached_result)
            
            # Extraire et préparer features
            features = self._extract_features(company_data)
            
            # Prédiction selon le modèle
            if model == ScoringModel.ENSEMBLE:
                score, confidence, feature_importance = await self._ensemble_prediction(features)
            else:
                score, confidence, feature_importance = await self._single_model_prediction(features, model)
            
            # Calculer scores par catégorie
            category_scores = self._calculate_category_scores(features, feature_importance)
            
            # Générer explications et recommandations
            explanation = self._generate_explanation(features, feature_importance, category_scores)
            recommendations = self._generate_recommendations(category_scores, company_data)
            
            # Créer résultat
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            result = ScoringResult(
                company_id=company_data.get('siren', 'unknown'),
                overall_score=score,
                confidence=confidence,
                category_scores=category_scores,
                feature_importance=feature_importance,
                model_used=model,
                processing_time_ms=processing_time,
                timestamp=datetime.now(),
                explanation=explanation,
                recommendations=recommendations
            )
            
            # Mettre en cache
            if use_cache:
                self.cache.set(cache_key, result.to_dict(), expire=self.cache_ttl)
            
            logger.info(f"Score calculé: {score:.2f} (confiance: {confidence:.3f}) pour {company_data.get('siren', 'unknown')}")
            
            return result
            
        except Exception as e:
            logger.error(f"Erreur scoring entreprise: {e}")
            # Retourner score par défaut en cas d'erreur
            return await self._fallback_scoring(company_data)
    
    async def batch_score_companies(self, 
                                  companies_data: List[Dict[str, Any]],
                                  model: ScoringModel = ScoringModel.ENSEMBLE) -> List[ScoringResult]:
        """
        Score un lot d'entreprises de manière optimisée
        
        Args:
            companies_data: Liste des données d'entreprises
            model: Modèle à utiliser
            
        Returns:
            List[ScoringResult]: Résultats de scoring
        """
        logger.info(f"Scoring par lot de {len(companies_data)} entreprises")
        
        # Traitement parallèle avec semaphore pour limiter la charge
        semaphore = asyncio.Semaphore(10)  # Max 10 scoring simultanés
        
        async def score_with_semaphore(company_data):
            async with semaphore:
                return await self.score_company(company_data, model)
        
        # Exécuter en parallèle
        results = await asyncio.gather(*[
            score_with_semaphore(company) for company in companies_data
        ], return_exceptions=True)
        
        # Filtrer les erreurs
        valid_results = [r for r in results if isinstance(r, ScoringResult)]
        error_count = len(results) - len(valid_results)
        
        if error_count > 0:
            logger.warning(f"{error_count} erreurs dans le scoring par lot")
        
        return valid_results
    
    async def retrain_models(self, feedback_data: List[Dict[str, Any]]):
        """
        Ré-entraîne les modèles avec nouvelles données de feedback
        
        Args:
            feedback_data: Données de feedback (score réel vs prédit)
        """
        logger.info(f"Ré-entraînement avec {len(feedback_data)} nouvelles données")
        
        try:
            # Ajouter feedback au dataset d'entraînement
            feedback_df = pd.DataFrame(feedback_data)
            
            if self.training_data is not None:
                self.training_data = pd.concat([self.training_data, feedback_df], ignore_index=True)
            else:
                self.training_data = feedback_df
            
            # Ré-entraîner modèles
            await self._train_models()
            
            # Sauvegarder modèles
            await self._save_models()
            
            logger.info("✅ Modèles ré-entraînés avec succès")
            
        except Exception as e:
            logger.error(f"Erreur ré-entraînement: {e}")
            raise
    
    def get_model_performance(self) -> Dict[str, Any]:
        """Retourne les métriques de performance des modèles"""
        performance = {}
        
        for model_type, metrics in self.model_metrics.items():
            performance[model_type.value] = metrics.calculate_metrics()
        
        return performance
    
    async def _load_or_generate_training_data(self):
        """Charge ou génère des données d'entraînement"""
        
        # Essayer de charger depuis fichier
        try:
            self.training_data = pd.read_pickle('/tmp/ma_intelligence_training_data.pkl')
            logger.info(f"Données d'entraînement chargées: {len(self.training_data)} échantillons")
            return
        except FileNotFoundError:
            pass
        
        # Générer données synthétiques si pas de fichier
        logger.info("Génération de données d'entraînement synthétiques")
        
        np.random.seed(42)  # Pour reproductibilité
        n_samples = 10000
        
        # Générer features aléatoires mais réalistes
        data = {
            'chiffre_affaires': np.random.lognormal(14, 1.5, n_samples),  # ~1M€ médiane
            'resultat_net': np.random.normal(50000, 200000, n_samples),
            'capitaux_propres': np.random.lognormal(12, 1.2, n_samples),
            'total_bilan': np.random.lognormal(13.5, 1.3, n_samples),
            'endettement': np.random.uniform(0, 0.8, n_samples),
            'croissance_ca_1an': np.random.normal(0.05, 0.15, n_samples),
            'croissance_ca_3ans': np.random.normal(0.08, 0.20, n_samples),
            'evolution_effectifs': np.random.normal(0.02, 0.10, n_samples),
            'secteur_code': np.random.randint(1, 20, n_samples),
            'secteur_performance': np.random.normal(0.5, 0.2, n_samples),
            'concurrence_locale': np.random.uniform(0.1, 0.9, n_samples),
            'region_code': np.random.randint(1, 13, n_samples),
            'densite_population': np.random.lognormal(6, 1, n_samples),
            'pib_regional': np.random.normal(30000, 10000, n_samples),
            'age_entreprise': np.random.exponential(15, n_samples),
            'stabilite_dirigeants': np.random.uniform(0, 1, n_samples),
            'qualite_site_web': np.random.uniform(0, 1, n_samples),
            'presence_digitale': np.random.uniform(0, 1, n_samples),
            'innovation_score': np.random.uniform(0, 1, n_samples)
        }
        
        df = pd.DataFrame(data)
        
        # Générer score cible basé sur une combinaison logique des features
        df['target_score'] = (
            0.15 * np.log1p(df['chiffre_affaires']) / 20 +
            0.10 * (df['resultat_net'] > 0).astype(float) +
            0.15 * np.clip(df['croissance_ca_3ans'], -0.5, 0.5) + 0.5 +
            0.10 * (1 - df['endettement']) +
            0.10 * df['secteur_performance'] +
            0.10 * (1 - df['concurrence_locale']) +
            0.10 * df['stabilite_dirigeants'] +
            0.10 * df['presence_digitale'] +
            0.10 * df['innovation_score']
        ) * 100
        
        # Ajouter du bruit réaliste
        df['target_score'] += np.random.normal(0, 5, n_samples)
        df['target_score'] = np.clip(df['target_score'], 0, 100)
        
        self.training_data = df
        
        # Sauvegarder pour usage futur
        df.to_pickle('/tmp/ma_intelligence_training_data.pkl')
        
        logger.info(f"✅ {n_samples} échantillons d'entraînement générés")
    
    async def _train_models(self):
        """Entraîne tous les modèles ML"""
        
        if self.training_data is None:
            raise ValueError("Pas de données d'entraînement disponibles")
        
        # Préparer données
        feature_names = ScoringFeatures.get_feature_names()
        X = self.training_data[feature_names].values
        y = self.training_data['target_score'].values
        
        # Normaliser features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        # Définir modèles
        models_config = {
            ScoringModel.RANDOM_FOREST: RandomForestRegressor(
                n_estimators=100, random_state=42, n_jobs=-1
            ),
            ScoringModel.GRADIENT_BOOSTING: GradientBoostingRegressor(
                n_estimators=100, random_state=42
            ),
            ScoringModel.LINEAR_REGRESSION: LinearRegression(),
            ScoringModel.RIDGE_REGRESSION: Ridge(alpha=1.0)
        }
        
        # Entraîner chaque modèle
        for model_type, model in models_config.items():
            logger.info(f"Entraînement {model_type.value}...")
            
            # Entraîner
            model.fit(X_train, y_train)
            
            # Évaluer
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='r2')
            
            # Sauvegarder modèle et scaler
            self.models[model_type] = model
            self.scalers[model_type] = scaler
            
            # Mettre à jour métriques
            metrics = self.model_metrics[model_type]
            metrics.accuracy_metrics['mse'].append(mse)
            metrics.accuracy_metrics['r2'].append(r2)
            metrics.accuracy_metrics['cross_val_scores'].append(cv_scores.mean())
            
            logger.info(f"✅ {model_type.value}: R² = {r2:.3f}, CV = {cv_scores.mean():.3f}")
    
    async def _ensemble_prediction(self, features: ScoringFeatures) -> Tuple[float, float, Dict[str, float]]:
        """Prédiction ensemble combinant plusieurs modèles"""
        
        feature_array = features.to_array().reshape(1, -1)
        predictions = []
        importances = []
        
        # Prédictions de chaque modèle
        for model_type in [ScoringModel.RANDOM_FOREST, ScoringModel.GRADIENT_BOOSTING, ScoringModel.RIDGE_REGRESSION]:
            if model_type in self.models:
                scaler = self.scalers[model_type]
                model = self.models[model_type]
                
                scaled_features = scaler.transform(feature_array)
                pred = model.predict(scaled_features)[0]
                predictions.append(pred)
                
                # Feature importance (si disponible)
                if hasattr(model, 'feature_importances_'):
                    importances.append(model.feature_importances_)
        
        if not predictions:
            raise ValueError("Aucun modèle entraîné disponible")
        
        # Moyenne pondérée (plus de poids aux modèles performants)
        weights = [0.4, 0.4, 0.2]  # RF, GB, Ridge
        ensemble_score = np.average(predictions[:len(weights)], weights=weights[:len(predictions)])
        
        # Confiance basée sur la variance des prédictions
        confidence = 1.0 / (1.0 + np.std(predictions))
        
        # Feature importance moyenne
        avg_importance = {}
        if importances:
            avg_importances = np.mean(importances, axis=0)
            feature_names = ScoringFeatures.get_feature_names()
            avg_importance = dict(zip(feature_names, avg_importances))
        
        return float(ensemble_score), float(confidence), avg_importance
    
    async def _single_model_prediction(self, features: ScoringFeatures, model_type: ScoringModel) -> Tuple[float, float, Dict[str, float]]:
        """Prédiction avec un seul modèle"""
        
        if model_type not in self.models:
            raise ValueError(f"Modèle {model_type.value} non entraîné")
        
        model = self.models[model_type]
        scaler = self.scalers[model_type]
        
        feature_array = features.to_array().reshape(1, -1)
        scaled_features = scaler.transform(feature_array)
        
        prediction = model.predict(scaled_features)[0]
        
        # Confiance basée sur les métriques du modèle
        metrics = self.model_metrics[model_type].calculate_metrics()
        confidence = metrics.get('r2', 0.5)
        
        # Feature importance
        feature_importance = {}
        if hasattr(model, 'feature_importances_'):
            feature_names = ScoringFeatures.get_feature_names()
            feature_importance = dict(zip(feature_names, model.feature_importances_))
        
        return float(prediction), float(confidence), feature_importance
    
    def _extract_features(self, company_data: Dict[str, Any]) -> ScoringFeatures:
        """Extrait les features ML depuis les données d'entreprise"""
        
        # Mapping des données vers features
        features = ScoringFeatures()
        
        # Features financières
        features.chiffre_affaires = float(company_data.get('chiffre_affaires', 0))
        features.resultat_net = float(company_data.get('resultat_net', 0))
        features.capitaux_propres = float(company_data.get('capitaux_propres', 0))
        features.total_bilan = float(company_data.get('total_bilan', 0))
        
        # Calculer ratio d'endettement
        if features.total_bilan > 0:
            dettes = features.total_bilan - features.capitaux_propres
            features.endettement = max(0, dettes / features.total_bilan)
        
        # Features de croissance (simulées si pas disponibles)
        features.croissance_ca_1an = float(company_data.get('croissance_ca_1an', np.random.normal(0.05, 0.1)))
        features.croissance_ca_3ans = float(company_data.get('croissance_ca_3ans', np.random.normal(0.08, 0.15)))
        features.evolution_effectifs = float(company_data.get('evolution_effectifs', np.random.normal(0.02, 0.08)))
        
        # Features sectorielles
        secteur = company_data.get('code_ape', '')
        features.secteur_code = hash(secteur[:2]) % 20 if secteur else 10
        features.secteur_performance = np.random.uniform(0.3, 0.7)  # À enrichir avec données réelles
        features.concurrence_locale = np.random.uniform(0.2, 0.8)
        
        # Features géographiques
        region = company_data.get('region', '')
        features.region_code = hash(region) % 13 if region else 6
        features.densite_population = float(company_data.get('densite_population', np.random.lognormal(6, 1)))
        features.pib_regional = float(company_data.get('pib_regional', 30000))
        
        # Features temporelles
        date_creation = company_data.get('date_creation')
        if date_creation:
            if isinstance(date_creation, str):
                try:
                    date_creation = datetime.fromisoformat(date_creation.replace('Z', '+00:00'))
                except:
                    date_creation = datetime.now()
            features.age_entreprise = (datetime.now() - date_creation).days // 365
        else:
            features.age_entreprise = np.random.exponential(15)
        
        # Features qualitatives (à enrichir avec scraping et NLP)
        features.stabilite_dirigeants = np.random.uniform(0.4, 0.9)
        features.qualite_site_web = np.random.uniform(0.2, 0.8)
        features.presence_digitale = np.random.uniform(0.1, 0.7)
        features.innovation_score = np.random.uniform(0.1, 0.6)
        
        return features
    
    def _calculate_category_scores(self, features: ScoringFeatures, feature_importance: Dict[str, float]) -> Dict[ScoreCategory, float]:
        """Calcule les scores par catégorie"""
        
        # Mapping features -> catégories
        category_features = {
            ScoreCategory.FINANCIAL_HEALTH: [
                'chiffre_affaires', 'resultat_net', 'capitaux_propres', 'total_bilan', 'endettement'
            ],
            ScoreCategory.GROWTH_POTENTIAL: [
                'croissance_ca_1an', 'croissance_ca_3ans', 'evolution_effectifs'
            ],
            ScoreCategory.MARKET_POSITION: [
                'secteur_performance', 'concurrence_locale', 'presence_digitale'
            ],
            ScoreCategory.OPERATIONAL_EFFICIENCY: [
                'age_entreprise', 'stabilite_dirigeants', 'innovation_score'
            ],
            ScoreCategory.STRATEGIC_FIT: [
                'qualite_site_web', 'secteur_code', 'region_code'
            ],
            ScoreCategory.RISK_ASSESSMENT: [
                'endettement', 'stabilite_dirigeants', 'secteur_performance'
            ]
        }
        
        category_scores = {}
        feature_array = features.to_array()
        feature_names = ScoringFeatures.get_feature_names()
        
        for category, feature_list in category_features.items():
            category_score = 0.0
            weight_sum = 0.0
            
            for feature_name in feature_list:
                if feature_name in feature_names:
                    idx = feature_names.index(feature_name)
                    feature_value = feature_array[idx]
                    importance = feature_importance.get(feature_name, 0.1)
                    
                    # Normaliser valeur feature (0-1)
                    normalized_value = self._normalize_feature_value(feature_name, feature_value)
                    
                    category_score += normalized_value * importance
                    weight_sum += importance
            
            # Score final de la catégorie (0-100)
            if weight_sum > 0:
                category_scores[category] = (category_score / weight_sum) * 100
            else:
                category_scores[category] = 50.0  # Score neutre
        
        return category_scores
    
    def _normalize_feature_value(self, feature_name: str, value: float) -> float:
        """Normalise une valeur de feature entre 0 et 1"""
        
        # Règles de normalisation par type de feature
        if feature_name in ['chiffre_affaires', 'total_bilan', 'capitaux_propres']:
            # Pour les montants financiers, utiliser log scale
            return min(1.0, np.log1p(max(0, value)) / 20)
        
        elif feature_name in ['croissance_ca_1an', 'croissance_ca_3ans', 'evolution_effectifs']:
            # Pour les taux de croissance, centrer autour de 0
            return max(0, min(1.0, (value + 0.5) / 1.0))
        
        elif feature_name == 'endettement':
            # Pour l'endettement, inverser (moins = mieux)
            return max(0, 1.0 - value)
        
        elif feature_name == 'age_entreprise':
            # Optimum vers 10-15 ans
            optimal_age = 12
            normalized = 1.0 - abs(value - optimal_age) / 50
            return max(0, min(1.0, normalized))
        
        else:
            # Par défaut, valeurs déjà entre 0 et 1
            return max(0, min(1.0, value))
    
    def _generate_explanation(self, features: ScoringFeatures, 
                            feature_importance: Dict[str, float],
                            category_scores: Dict[ScoreCategory, float]) -> Dict[str, Any]:
        """Génère une explication du score"""
        
        # Top 5 features les plus importantes
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Points forts et faibles
        strong_categories = [cat.value for cat, score in category_scores.items() if score >= 75]
        weak_categories = [cat.value for cat, score in category_scores.items() if score <= 40]
        
        return {
            'top_influential_factors': [
                {'feature': name, 'importance': round(imp, 3)} 
                for name, imp in top_features
            ],
            'strong_points': strong_categories,
            'improvement_areas': weak_categories,
            'overall_assessment': self._get_score_interpretation(
                sum(category_scores.values()) / len(category_scores)
            ),
            'data_completeness': self._assess_data_completeness(features)
        }
    
    def _generate_recommendations(self, category_scores: Dict[ScoreCategory, float], 
                                company_data: Dict[str, Any]) -> List[str]:
        """Génère des recommandations basées sur le scoring"""
        
        recommendations = []
        overall_score = sum(category_scores.values()) / len(category_scores)
        
        # Recommandations générales
        if overall_score >= 80:
            recommendations.append("Entreprise avec excellent potentiel M&A - priorité haute pour approche")
        elif overall_score >= 60:
            recommendations.append("Potentiel M&A solide - évaluation approfondie recommandée")
        elif overall_score >= 40:
            recommendations.append("Potentiel M&A modéré - analyse des risques nécessaire")
        else:
            recommendations.append("Potentiel M&A faible - considérer autres opportunités")
        
        # Recommandations par catégorie
        for category, score in category_scores.items():
            if score <= 30:
                if category == ScoreCategory.FINANCIAL_HEALTH:
                    recommendations.append("Analyser en détail la santé financière avant négociation")
                elif category == ScoreCategory.GROWTH_POTENTIAL:
                    recommendations.append("Évaluer les perspectives de croissance et plan de développement")
                elif category == ScoreCategory.MARKET_POSITION:
                    recommendations.append("Étudier la position concurrentielle et part de marché")
        
        # Recommandations sectorielles
        if company_data.get('chiffre_affaires', 0) > 10000000:  # > 10M€
            recommendations.append("Grande entreprise - prévoir due diligence approfondie")
        
        return recommendations[:5]  # Limiter à 5 recommandations
    
    def _get_score_interpretation(self, score: float) -> str:
        """Interprète le score global"""
        
        if score >= 85:
            return "Cible M&A exceptionnelle"
        elif score >= 70:
            return "Très bon potentiel M&A"
        elif score >= 55:
            return "Potentiel M&A intéressant"
        elif score >= 40:
            return "Potentiel M&A modéré"
        else:
            return "Potentiel M&A limité"
    
    def _assess_data_completeness(self, features: ScoringFeatures) -> float:
        """Évalue la complétude des données (0-1)"""
        
        feature_array = features.to_array()
        non_zero_count = np.count_nonzero(feature_array)
        completeness = non_zero_count / len(feature_array)
        
        return round(completeness, 2)
    
    def _generate_cache_key(self, company_data: Dict[str, Any], model: ScoringModel) -> str:
        """Génère une clé de cache unique"""
        
        # Utiliser SIREN + hash des données importantes + modèle
        key_data = {
            'siren': company_data.get('siren', ''),
            'chiffre_affaires': company_data.get('chiffre_affaires', 0),
            'resultat_net': company_data.get('resultat_net', 0),
            'model': model.value
        }
        
        key_string = json.dumps(key_data, sort_keys=True)
        return f"ai_score_{hashlib.md5(key_string.encode()).hexdigest()}"
    
    async def _fallback_scoring(self, company_data: Dict[str, Any]) -> ScoringResult:
        """Score de secours en cas d'erreur"""
        
        # Score basique basé sur le CA
        ca = float(company_data.get('chiffre_affaires', 0))
        basic_score = min(100, max(0, np.log1p(ca) * 5))
        
        return ScoringResult(
            company_id=company_data.get('siren', 'unknown'),
            overall_score=basic_score,
            confidence=0.3,
            category_scores={cat: basic_score for cat in ScoreCategory},
            feature_importance={},
            model_used=ScoringModel.LINEAR_REGRESSION,
            processing_time_ms=1.0,
            timestamp=datetime.now(),
            explanation={'error': 'Score de secours utilisé'},
            recommendations=['Données insuffisantes pour scoring IA complet']
        )
    
    async def _save_models(self):
        """Sauvegarde les modèles entraînés"""
        
        try:
            model_data = {
                'models': self.models,
                'scalers': self.scalers,
                'metrics': self.model_metrics,
                'timestamp': datetime.now()
            }
            
            joblib.dump(model_data, '/tmp/ma_intelligence_ai_models.pkl')
            logger.info("✅ Modèles IA sauvegardés")
            
        except Exception as e:
            logger.error(f"Erreur sauvegarde modèles: {e}")


# Instance globale
_ai_scoring_engine: Optional[AICompanyScoringEngine] = None


async def get_ai_scoring_engine() -> AICompanyScoringEngine:
    """Factory pour obtenir l'instance du moteur de scoring IA"""
    global _ai_scoring_engine
    
    if _ai_scoring_engine is None:
        _ai_scoring_engine = AICompanyScoringEngine()
        await _ai_scoring_engine.initialize_models()
    
    return _ai_scoring_engine


async def initialize_ai_scoring():
    """Initialise le système de scoring IA au démarrage"""
    try:
        engine = await get_ai_scoring_engine()
        logger.info("🤖 Système de scoring IA initialisé avec succès")
        return engine
    except Exception as e:
        logger.error(f"Erreur initialisation scoring IA: {e}")
        raise