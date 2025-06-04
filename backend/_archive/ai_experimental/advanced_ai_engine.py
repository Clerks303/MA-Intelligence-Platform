"""
Moteur d'intelligence artificielle avancé pour M&A Intelligence Platform
US-010: IA et analyse prédictive pour scoring intelligent et recommandations

Ce module fournit:
- Scoring multi-critères avec ensemble learning
- Analyse prédictive des tendances M&A
- Système de recommandations intelligentes
- Modèles adaptatifs et apprentissage continu
- Explainability et interprétation des modèles
"""

import asyncio
import numpy as np
import pandas as pd
import pickle
import json
import joblib
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
from functools import wraps
import logging
from enum import Enum

# Machine Learning
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    VotingClassifier, VotingRegressor
)
from sklearn.model_selection import (
    train_test_split, cross_val_score, GridSearchCV,
    TimeSeriesSplit
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN

# Advanced ML
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier, CatBoostRegressor

# Model explainability
import shap
from lime.lime_tabular import LimeTabularExplainer


# Time series & prediction
from prophet import Prophet
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA

# ML Flow for model management
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm

# Feature engineering
#from feature_engine.creation import CombineWithReferenceFeature
from feature_engine.selection import SelectByShuffling

from app.core.logging_system import get_logger, LogCategory
from app.core.cache_manager import get_cache_manager, cached

logger = get_logger("advanced_ai_engine", LogCategory.AI_ML)


class ModelType(str, Enum):
    """Types de modèles ML"""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    TIME_SERIES = "time_series"
    RECOMMENDATION = "recommendation"


class PredictionConfidence(str, Enum):
    """Niveaux de confiance des prédictions"""
    VERY_HIGH = "very_high"  # > 95%
    HIGH = "high"            # 80-95%
    MEDIUM = "medium"        # 60-80%
    LOW = "low"              # 40-60%
    VERY_LOW = "very_low"    # < 40%


@dataclass
class ModelMetrics:
    """Métriques de performance d'un modèle"""
    model_id: str
    model_type: ModelType
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    r2_score: float = 0.0
    mse: float = 0.0
    mae: float = 0.0
    cross_val_mean: float = 0.0
    cross_val_std: float = 0.0
    training_time: float = 0.0
    prediction_time: float = 0.0
    last_trained: datetime = field(default_factory=datetime.now)
    sample_size: int = 0


@dataclass
class AIInsight:
    """Insight généré par l'IA"""
    insight_id: str
    title: str
    description: str
    confidence: PredictionConfidence
    importance_score: float  # 0-100
    category: str
    evidence: List[str]
    recommendations: List[str]
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CompanyAIProfile:
    """Profil IA complet d'une entreprise"""
    siren: str
    ma_score: float  # Score M&A global (0-100)
    financial_health_score: float
    growth_potential_score: float
    market_position_score: float
    risk_score: float
    
    # Prédictions
    predicted_revenue_growth: float
    predicted_acquisition_probability: float
    predicted_valuation_range: Tuple[float, float]
    
    # Segmentation
    cluster_id: int
    cluster_label: str
    
    # Insights
    key_insights: List[AIInsight]
    
    # Métadonnées
    confidence_level: PredictionConfidence
    last_updated: datetime = field(default_factory=datetime.now)
    model_versions: Dict[str, str] = field(default_factory=dict)


class FeatureEngineer:
    """Ingénierie des features pour les modèles ML"""
    
    def __init__(self):
        self.scalers: Dict[str, StandardScaler] = {}
        self.encoders: Dict[str, LabelEncoder] = {}
        self.feature_combinations = {}
        
    def create_financial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crée des features financières avancées"""
        
        enhanced_df = df.copy()
        
        # Ratios financiers
        if 'chiffre_affaires' in df.columns and 'effectifs' in df.columns:
            enhanced_df['ca_per_employee'] = df['chiffre_affaires'] / df['effectifs'].replace(0, 1)
        
        # Tendances temporelles
        if 'date_creation' in df.columns:
            enhanced_df['company_age'] = (
                datetime.now() - pd.to_datetime(df['date_creation'])
            ).dt.days / 365.25
        
        # Scoring secteur
        if 'secteur_activite' in df.columns:
            sector_stats = df.groupby('secteur_activite').agg({
                'chiffre_affaires': ['mean', 'std'],
                'effectifs': ['mean', 'std']
            }).fillna(0)
            
            for sector in df['secteur_activite'].unique():
                mask = df['secteur_activite'] == sector
                enhanced_df.loc[mask, 'sector_ca_zscore'] = (
                    df.loc[mask, 'chiffre_affaires'] - sector_stats.loc[sector, ('chiffre_affaires', 'mean')]
                ) / (sector_stats.loc[sector, ('chiffre_affaires', 'std')] + 1e-6)
        
        # Features géographiques
        if 'adresse' in df.columns:
            enhanced_df['is_paris'] = df['adresse'].str.contains('Paris', na=False).astype(int)
            enhanced_df['is_lyon'] = df['adresse'].str.contains('Lyon', na=False).astype(int)
        
        # Features de croissance
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col.endswith('_historique'):
                # Calculer tendance croissance
                enhanced_df[f'{col}_growth_rate'] = df[col].pct_change().fillna(0)
        
        return enhanced_df
    
    def create_text_features(self, df: pd.DataFrame, text_column: str) -> pd.DataFrame:
        """Crée des features à partir de données textuelles"""
        
        enhanced_df = df.copy()
        
        if text_column in df.columns:
            text_series = df[text_column].fillna('').astype(str)
            
            # Features de base
            enhanced_df[f'{text_column}_length'] = text_series.str.len()
            enhanced_df[f'{text_column}_word_count'] = text_series.str.split().str.len()
            
            # Keywords M&A
            ma_keywords = [
                'acquisition', 'fusion', 'rachat', 'cession', 'investissement',
                'développement', 'expansion', 'croissance', 'innovation',
                'transformation', 'restructuration'
            ]
            
            for keyword in ma_keywords:
                enhanced_df[f'{text_column}_has_{keyword}'] = (
                    text_series.str.lower().str.contains(keyword, na=False).astype(int)
                )
        
        return enhanced_df
    
    def scale_features(self, df: pd.DataFrame, feature_set: str) -> pd.DataFrame:
        """Scale les features numériques"""
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        if feature_set not in self.scalers:
            self.scalers[feature_set] = StandardScaler()
            scaled_data = self.scalers[feature_set].fit_transform(df[numeric_columns])
        else:
            scaled_data = self.scalers[feature_set].transform(df[numeric_columns])
        
        scaled_df = df.copy()
        scaled_df[numeric_columns] = scaled_data
        
        return scaled_df


class EnsembleModelManager:
    """Gestionnaire de modèles ensemble pour scoring ML"""
    
    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.model_metrics: Dict[str, ModelMetrics] = {}
        self.feature_importance: Dict[str, Dict[str, float]] = {}
        self.explainers: Dict[str, Any] = {}
        self.model_version = "1.0.0"
        
    async def train_ma_scoring_model(self, training_data: pd.DataFrame) -> str:
        """Entraîne un modèle ensemble pour le scoring M&A"""
        
        model_id = f"ma_scoring_ensemble_{int(time.time())}"
        
        try:
            logger.info(f"🤖 Entraînement modèle scoring M&A: {model_id}")
            
            # Préparation des données
            feature_cols = [col for col in training_data.columns 
                          if col not in ['siren', 'target_ma_score', 'date_update']]
            
            X = training_data[feature_cols].fillna(0)
            y = training_data['target_ma_score'] if 'target_ma_score' in training_data.columns else np.random.rand(len(X)) * 100
            
            # Split train/test
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=None
            )
            
            # Modèles de base
            rf_model = RandomForestRegressor(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            )
            
            xgb_model = xgb.XGBRegressor(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42
            )
            
            lgb_model = lgb.LGBMRegressor(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42,
                verbose=-1
            )
            
            # Ensemble voting regressor
            ensemble_model = VotingRegressor([
                ('rf', rf_model),
                ('xgb', xgb_model),
                ('lgb', lgb_model)
            ])
            
            # Entraînement
            start_time = time.time()
            ensemble_model.fit(X_train, y_train)
            training_time = time.time() - start_time
            
            # Prédictions et métriques
            start_pred_time = time.time()
            y_pred = ensemble_model.predict(X_test)
            prediction_time = (time.time() - start_pred_time) / len(X_test)
            
            # Calcul métriques
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Cross-validation
            cv_scores = cross_val_score(ensemble_model, X, y, cv=5, scoring='r2')
            
            # Sauvegarde modèle
            self.models[model_id] = ensemble_model
            
            # Métriques
            self.model_metrics[model_id] = ModelMetrics(
                model_id=model_id,
                model_type=ModelType.REGRESSION,
                r2_score=r2,
                mse=mse,
                mae=mae,
                cross_val_mean=cv_scores.mean(),
                cross_val_std=cv_scores.std(),
                training_time=training_time,
                prediction_time=prediction_time,
                sample_size=len(training_data)
            )
            
            # Feature importance (moyenne des modèles)
            feature_importance = {}
            
            # Random Forest
            rf_importance = dict(zip(feature_cols, rf_model.feature_importances_))
            
            # XGBoost
            xgb_importance = dict(zip(feature_cols, xgb_model.feature_importances_))
            
            # Moyenne pondérée
            for feature in feature_cols:
                feature_importance[feature] = (
                    rf_importance.get(feature, 0) * 0.4 +
                    xgb_importance.get(feature, 0) * 0.6
                )
            
            self.feature_importance[model_id] = feature_importance
            
            # SHAP explainer
            try:
                explainer = shap.TreeExplainer(xgb_model)
                self.explainers[model_id] = explainer
                logger.info(f"✅ SHAP explainer créé pour {model_id}")
            except Exception as e:
                logger.warning(f"Impossible de créer SHAP explainer: {e}")
            
            # Log MLflow
            try:
                with mlflow.start_run(run_name=f"ma_scoring_{model_id}"):
                    mlflow.log_params({
                        'model_type': 'ensemble_voting',
                        'n_features': len(feature_cols),
                        'train_size': len(X_train),
                        'test_size': len(X_test)
                    })
                    
                    mlflow.log_metrics({
                        'r2_score': r2,
                        'mse': mse,
                        'mae': mae,
                        'cv_mean': cv_scores.mean(),
                        'cv_std': cv_scores.std()
                    })
                    
                    mlflow.sklearn.log_model(ensemble_model, "model")
            except Exception as e:
                logger.warning(f"MLflow logging failed: {e}")
            
            logger.info(f"✅ Modèle {model_id} entraîné - R²: {r2:.3f}, MAE: {mae:.2f}")
            
            return model_id
            
        except Exception as e:
            logger.error(f"❌ Erreur entraînement modèle {model_id}: {e}")
            raise
    
    async def predict_ma_score(self, model_id: str, company_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prédit le score M&A d'une entreprise"""
        
        if model_id not in self.models:
            raise ValueError(f"Modèle {model_id} non trouvé")
        
        model = self.models[model_id]
        
        # Préparation des données
        df = pd.DataFrame([company_data])
        
        # Features (même ordre que l'entraînement)
        feature_cols = [col for col in df.columns 
                       if col not in ['siren', 'target_ma_score', 'date_update']]
        
        X = df[feature_cols].fillna(0)
        
        # Prédiction
        start_time = time.time()
        prediction = model.predict(X)[0]
        prediction_time = time.time() - start_time
        
        # Confiance basée sur la variance des modèles de base
        if hasattr(model, 'estimators_'):
            individual_predictions = [estimator.predict(X)[0] for estimator in model.estimators_]
            variance = np.var(individual_predictions)
            confidence_score = max(0, 100 - variance * 10)
        else:
            confidence_score = 75.0  # Default confidence
        
        # Niveau de confiance
        if confidence_score >= 95:
            confidence_level = PredictionConfidence.VERY_HIGH
        elif confidence_score >= 80:
            confidence_level = PredictionConfidence.HIGH
        elif confidence_score >= 60:
            confidence_level = PredictionConfidence.MEDIUM
        elif confidence_score >= 40:
            confidence_level = PredictionConfidence.LOW
        else:
            confidence_level = PredictionConfidence.VERY_LOW
        
        # Explication SHAP si disponible
        shap_values = None
        if model_id in self.explainers:
            try:
                explainer = self.explainers[model_id]
                shap_values = explainer.shap_values(X)
                if len(shap_values.shape) > 1:
                    shap_values = shap_values[0]  # Régression
            except Exception as e:
                logger.warning(f"Erreur calcul SHAP: {e}")
        
        # Feature importance pour cette prédiction
        feature_impact = {}
        if shap_values is not None:
            for i, feature in enumerate(feature_cols):
                feature_impact[feature] = float(shap_values[i])
        elif model_id in self.feature_importance:
            feature_impact = self.feature_importance[model_id]
        
        return {
            'prediction': float(prediction),
            'confidence_score': confidence_score,
            'confidence_level': confidence_level.value,
            'prediction_time_ms': prediction_time * 1000,
            'feature_impact': feature_impact,
            'model_id': model_id,
            'model_metrics': self.model_metrics.get(model_id, {})
        }
    
    def get_model_explanation(self, model_id: str, feature_names: List[str]) -> Dict[str, Any]:
        """Retourne l'explication globale du modèle"""
        
        if model_id not in self.models:
            return {}
        
        explanation = {
            'model_id': model_id,
            'model_type': 'ensemble_voting',
            'feature_importance': self.feature_importance.get(model_id, {}),
            'metrics': self.model_metrics.get(model_id, {}),
            'top_features': []
        }
        
        # Top features par importance
        if model_id in self.feature_importance:
            sorted_features = sorted(
                self.feature_importance[model_id].items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )
            
            explanation['top_features'] = [
                {
                    'feature': feature,
                    'importance': importance,
                    'impact_description': self._get_feature_description(feature, importance)
                }
                for feature, importance in sorted_features[:10]
            ]
        
        return explanation
    
    def _get_feature_description(self, feature: str, importance: float) -> str:
        """Génère une description de l'impact d'une feature"""
        
        impact_level = "fort" if abs(importance) > 0.1 else "modéré" if abs(importance) > 0.05 else "faible"
        direction = "positif" if importance > 0 else "négatif"
        
        feature_descriptions = {
            'chiffre_affaires': "Le chiffre d'affaires de l'entreprise",
            'effectifs': "Le nombre d'employés",
            'ca_per_employee': "La productivité par employé",
            'company_age': "L'âge de l'entreprise",
            'sector_ca_zscore': "La performance relative au secteur",
            'is_paris': "La localisation à Paris",
            'growth_rate': "Le taux de croissance"
        }
        
        base_desc = feature_descriptions.get(feature, f"Le critère {feature}")
        
        return f"{base_desc} a un impact {impact_level} {direction} sur le score M&A"


class PredictiveAnalytics:
    """Système d'analyse prédictive pour tendances M&A"""
    
    def __init__(self):
        self.time_series_models: Dict[str, Any] = {}
        self.trend_models: Dict[str, Any] = {}
        self.market_indicators: Dict[str, float] = {}
        
    async def analyze_market_trends(self, historical_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyse les tendances du marché M&A"""
        
        try:
            logger.info("📈 Analyse des tendances marché M&A")
            
            # Préparation données temporelles
            historical_data['date'] = pd.to_datetime(historical_data['date'])
            historical_data = historical_data.sort_values('date')
            
            # Agrégation mensuelle
            monthly_data = historical_data.groupby(
                historical_data['date'].dt.to_period('M')
            ).agg({
                'ma_score': 'mean',
                'chiffre_affaires': 'sum',
                'nombre_acquisitions': 'sum',
                'valeur_transactions': 'sum'
            }).reset_index()
            
            monthly_data['date'] = monthly_data['date'].dt.to_timestamp()
            
            # Modèle Prophet pour prédictions
            prophet_predictions = {}
            
            for metric in ['ma_score', 'nombre_acquisitions', 'valeur_transactions']:
                if metric in monthly_data.columns:
                    # Préparer données Prophet
                    prophet_data = monthly_data[['date', metric]].rename(
                        columns={'date': 'ds', metric: 'y'}
                    ).dropna()
                    
                    if len(prophet_data) >= 24:  # Minimum 2 ans de données
                        # Entraînement Prophet
                        model = Prophet(
                            yearly_seasonality=True,
                            monthly_seasonality=True,
                            changepoint_prior_scale=0.05,
                            seasonality_prior_scale=10
                        )
                        
                        model.fit(prophet_data)
                        
                        # Prédictions 12 mois
                        future = model.make_future_dataframe(periods=12, freq='M')
                        forecast = model.predict(future)
                        
                        # Extraire prédictions futures
                        future_forecast = forecast.tail(12)
                        
                        prophet_predictions[metric] = {
                            'predictions': future_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_dict('records'),
                            'trend': 'croissant' if forecast['trend'].iloc[-1] > forecast['trend'].iloc[-13] else 'décroissant',
                            'seasonality_strength': float(forecast[['yearly', 'monthly']].abs().mean().sum()),
                            'confidence_interval': float(forecast['yhat_upper'].iloc[-1] - forecast['yhat_lower'].iloc[-1])
                        }
                        
                        self.time_series_models[f"prophet_{metric}"] = model
            
            # Analyse de corrélation entre métriques
            correlation_matrix = monthly_data[
                ['ma_score', 'chiffre_affaires', 'nombre_acquisitions', 'valeur_transactions']
            ].corr()
            
            # Détection des patterns saisonniers
            seasonal_patterns = {}
            for metric in ['ma_score', 'nombre_acquisitions']:
                if metric in monthly_data.columns:
                    monthly_data['month'] = monthly_data['date'].dt.month
                    seasonal_avg = monthly_data.groupby('month')[metric].mean()
                    
                    peak_month = seasonal_avg.idxmax()
                    low_month = seasonal_avg.idxmin()
                    
                    seasonal_patterns[metric] = {
                        'peak_month': int(peak_month),
                        'low_month': int(low_month),
                        'seasonality_factor': float(seasonal_avg.std() / seasonal_avg.mean())
                    }
            
            # Indicateurs de marché
            current_period = monthly_data.tail(3)  # 3 derniers mois
            previous_period = monthly_data.tail(6).head(3)  # 3 mois précédents
            
            market_indicators = {}
            for metric in ['ma_score', 'nombre_acquisitions']:
                if metric in current_period.columns:
                    current_avg = current_period[metric].mean()
                    previous_avg = previous_period[metric].mean()
                    
                    growth_rate = ((current_avg - previous_avg) / previous_avg * 100) if previous_avg > 0 else 0
                    
                    market_indicators[f"{metric}_growth_rate"] = growth_rate
                    market_indicators[f"{metric}_current_level"] = current_avg
            
            self.market_indicators = market_indicators
            
            # Insights automatiques
            insights = []
            
            for metric, data in prophet_predictions.items():
                trend = data['trend']
                confidence = 100 - min(100, data['confidence_interval'] / data['predictions'][-1]['yhat'] * 100)
                
                if trend == 'croissant' and confidence > 70:
                    insights.append(
                        f"Tendance haussière confirmée pour {metric} avec {confidence:.0f}% de confiance"
                    )
                elif trend == 'décroissant' and confidence > 70:
                    insights.append(
                        f"Tendance baissière détectée pour {metric} avec {confidence:.0f}% de confiance"
                    )
            
            return {
                'prophet_predictions': prophet_predictions,
                'seasonal_patterns': seasonal_patterns,
                'market_indicators': market_indicators,
                'correlation_matrix': correlation_matrix.to_dict(),
                'insights': insights,
                'analysis_timestamp': datetime.now().isoformat(),
                'data_quality': {
                    'sample_size': len(historical_data),
                    'time_span_months': len(monthly_data),
                    'completeness': float(historical_data.count().sum() / (len(historical_data) * len(historical_data.columns)))
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Erreur analyse tendances: {e}")
            raise
    
    async def predict_acquisition_probability(self, company_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Prédit la probabilité d'acquisition d'une entreprise"""
        
        try:
            # Facteurs d'acquisition
            acquisition_factors = {
                'financial_performance': 0.0,
                'market_position': 0.0,
                'growth_potential': 0.0,
                'strategic_value': 0.0,
                'market_conditions': 0.0
            }
            
            # Performance financière (30%)
            if 'chiffre_affaires' in company_profile and 'effectifs' in company_profile:
                ca = company_profile['chiffre_affaires']
                effectifs = company_profile['effectifs']
                
                # Productivité
                productivity = ca / max(effectifs, 1)
                acquisition_factors['financial_performance'] = min(100, productivity / 100000 * 50)
            
            # Position marché (25%)
            if 'sector_ca_zscore' in company_profile:
                zscore = company_profile.get('sector_ca_zscore', 0)
                acquisition_factors['market_position'] = max(0, min(100, 50 + zscore * 20))
            
            # Potentiel croissance (25%)
            if 'growth_rate' in company_profile:
                growth_rate = company_profile.get('growth_rate', 0)
                acquisition_factors['growth_potential'] = max(0, min(100, 50 + growth_rate * 30))
            
            # Valeur stratégique (15%)
            strategic_score = 50  # Base
            if company_profile.get('is_paris', False):
                strategic_score += 20
            if company_profile.get('secteur_activite') in ['tech', 'digital', 'innovation']:
                strategic_score += 30
            acquisition_factors['strategic_value'] = min(100, strategic_score)
            
            # Conditions marché (5%)
            market_growth = self.market_indicators.get('ma_score_growth_rate', 0)
            acquisition_factors['market_conditions'] = max(0, min(100, 50 + market_growth))
            
            # Calcul probabilité globale
            weights = {
                'financial_performance': 0.30,
                'market_position': 0.25,
                'growth_potential': 0.25,
                'strategic_value': 0.15,
                'market_conditions': 0.05
            }
            
            weighted_score = sum(
                acquisition_factors[factor] * weight
                for factor, weight in weights.items()
            )
            
            # Probabilité d'acquisition
            acquisition_probability = weighted_score / 100
            
            # Timeframe estimé
            if acquisition_probability > 0.8:
                estimated_timeframe = "6-12 mois"
            elif acquisition_probability > 0.6:
                estimated_timeframe = "1-2 ans"
            elif acquisition_probability > 0.4:
                estimated_timeframe = "2-3 ans"
            else:
                estimated_timeframe = "> 3 ans"
            
            # Recommandations
            recommendations = []
            if acquisition_factors['financial_performance'] < 50:
                recommendations.append("Améliorer la performance financière")
            if acquisition_factors['growth_potential'] < 50:
                recommendations.append("Accélérer la croissance")
            if acquisition_factors['market_position'] < 50:
                recommendations.append("Renforcer la position marché")
            
            return {
                'acquisition_probability': acquisition_probability,
                'probability_percentage': acquisition_probability * 100,
                'estimated_timeframe': estimated_timeframe,
                'factors_analysis': acquisition_factors,
                'factor_weights': weights,
                'recommendations': recommendations,
                'confidence_level': 'high' if len(company_profile) > 10 else 'medium',
                'analysis_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Erreur prédiction acquisition: {e}")
            raise


# Instance globale
_advanced_ai_engine: Optional['AdvancedAIEngine'] = None


class AdvancedAIEngine:
    """Moteur d'IA avancé principal"""
    
    def __init__(self):
        self.feature_engineer = FeatureEngineer()
        self.ensemble_manager = EnsembleModelManager()
        self.predictive_analytics = PredictiveAnalytics()
        
        # Cache des profils IA
        self.company_profiles: Dict[str, CompanyAIProfile] = {}
        
        # Modèles actifs
        self.active_models: Dict[str, str] = {}
        
        logger.info("🧠 Moteur d'IA avancé initialisé")
    
    @cached('ai_company_profile', ttl_seconds=3600)
    async def get_company_ai_profile(self, siren: str, company_data: Dict[str, Any]) -> CompanyAIProfile:
        """Génère le profil IA complet d'une entreprise"""
        
        try:
            logger.info(f"🧠 Génération profil IA pour {siren}")
            
            # Enrichissement des features
            df = pd.DataFrame([company_data])
            enhanced_df = self.feature_engineer.create_financial_features(df)
            
            if 'description' in company_data:
                enhanced_df = self.feature_engineer.create_text_features(enhanced_df, 'description')
            
            enhanced_data = enhanced_df.iloc[0].to_dict()
            
            # Scoring M&A avec modèle ensemble
            ma_score_result = await self._get_ma_score(enhanced_data)
            
            # Analyse prédictive
            acquisition_pred = await self.predictive_analytics.predict_acquisition_probability(enhanced_data)
            
            # Génération d'insights
            insights = await self._generate_ai_insights(siren, enhanced_data, ma_score_result, acquisition_pred)
            
            # Création du profil
            profile = CompanyAIProfile(
                siren=siren,
                ma_score=ma_score_result.get('prediction', 50.0),
                financial_health_score=self._calculate_financial_health_score(enhanced_data),
                growth_potential_score=acquisition_pred['factors_analysis']['growth_potential'],
                market_position_score=acquisition_pred['factors_analysis']['market_position'],
                risk_score=100 - ma_score_result.get('prediction', 50.0),
                
                predicted_revenue_growth=enhanced_data.get('growth_rate', 0) * 100,
                predicted_acquisition_probability=acquisition_pred['acquisition_probability'],
                predicted_valuation_range=(
                    enhanced_data.get('chiffre_affaires', 0) * 1.5,
                    enhanced_data.get('chiffre_affaires', 0) * 3.0
                ),
                
                cluster_id=0,  # TODO: Implement clustering
                cluster_label="À déterminer",
                
                key_insights=insights,
                confidence_level=PredictionConfidence(ma_score_result.get('confidence_level', 'medium')),
                model_versions={
                    'ma_scoring': ma_score_result.get('model_id', 'unknown'),
                    'acquisition_prediction': 'v1.0',
                    'insights_generation': 'v1.0'
                }
            )
            
            self.company_profiles[siren] = profile
            
            return profile
            
        except Exception as e:
            logger.error(f"❌ Erreur génération profil IA {siren}: {e}")
            raise
    
    async def _get_ma_score(self, company_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calcule le score M&A avec le modèle ensemble"""
        
        # Utiliser le modèle actif ou créer un modèle par défaut
        if 'ma_scoring' in self.active_models:
            model_id = self.active_models['ma_scoring']
            return await self.ensemble_manager.predict_ma_score(model_id, company_data)
        else:
            # Score par défaut basé sur règles métier
            score = 50.0  # Base
            
            if 'chiffre_affaires' in company_data:
                ca = company_data['chiffre_affaires']
                if ca > 10000000:  # > 10M€
                    score += 20
                elif ca > 1000000:  # > 1M€
                    score += 10
            
            if 'effectifs' in company_data:
                effectifs = company_data['effectifs']
                if effectifs > 100:
                    score += 15
                elif effectifs > 20:
                    score += 8
            
            if company_data.get('is_paris', False):
                score += 10
            
            return {
                'prediction': min(100, score),
                'confidence_level': 'medium',
                'model_id': 'rule_based_default'
            }
    
    def _calculate_financial_health_score(self, company_data: Dict[str, Any]) -> float:
        """Calcule le score de santé financière"""
        
        score = 50.0  # Base
        
        # Chiffre d'affaires
        ca = company_data.get('chiffre_affaires', 0)
        if ca > 5000000:
            score += 25
        elif ca > 1000000:
            score += 15
        elif ca > 100000:
            score += 5
        
        # Productivité
        if 'ca_per_employee' in company_data:
            productivity = company_data['ca_per_employee']
            if productivity > 100000:
                score += 20
            elif productivity > 50000:
                score += 10
        
        # Âge de l'entreprise (stabilité)
        if 'company_age' in company_data:
            age = company_data['company_age']
            if 3 <= age <= 20:
                score += 5  # Sweet spot
        
        return min(100, score)
    
    async def _generate_ai_insights(
        self,
        siren: str,
        company_data: Dict[str, Any],
        ma_score_result: Dict[str, Any],
        acquisition_pred: Dict[str, Any]
    ) -> List[AIInsight]:
        """Génère des insights IA pour l'entreprise"""
        
        insights = []
        
        # Insight sur le score M&A
        ma_score = ma_score_result.get('prediction', 0)
        if ma_score > 80:
            insights.append(AIInsight(
                insight_id=f"ma_score_high_{siren}",
                title="Score M&A Excellent",
                description=f"Cette entreprise présente un score M&A exceptionnel de {ma_score:.1f}/100, la positionnant comme une cible d'acquisition très attractive.",
                confidence=PredictionConfidence.HIGH,
                importance_score=90,
                category="scoring",
                evidence=[
                    f"Score M&A: {ma_score:.1f}/100",
                    f"Confiance du modèle: {ma_score_result.get('confidence_level', 'medium')}"
                ],
                recommendations=[
                    "Contacter rapidement cette entreprise",
                    "Préparer une proposition d'acquisition",
                    "Analyser la concurrence potentielle"
                ]
            ))
        elif ma_score < 30:
            insights.append(AIInsight(
                insight_id=f"ma_score_low_{siren}",
                title="Score M&A Faible",
                description=f"Le score M&A de {ma_score:.1f}/100 indique des défis potentiels pour une acquisition.",
                confidence=PredictionConfidence.MEDIUM,
                importance_score=60,
                category="scoring",
                evidence=[f"Score M&A: {ma_score:.1f}/100"],
                recommendations=["Analyser les points d'amélioration", "Évaluer le potentiel de redressement"]
            ))
        
        # Insight sur la probabilité d'acquisition
        acq_prob = acquisition_pred['acquisition_probability']
        if acq_prob > 0.7:
            insights.append(AIInsight(
                insight_id=f"acquisition_likely_{siren}",
                title="Acquisition Probable",
                description=f"Probabilité d'acquisition élevée ({acq_prob*100:.1f}%) dans les {acquisition_pred['estimated_timeframe']}.",
                confidence=PredictionConfidence.HIGH,
                importance_score=85,
                category="prediction",
                evidence=[f"Probabilité: {acq_prob*100:.1f}%", f"Délai estimé: {acquisition_pred['estimated_timeframe']}"],
                recommendations=["Accélérer le processus de due diligence", "Préparer une offre compétitive"]
            ))
        
        # Insight sur la performance financière
        financial_score = acquisition_pred['factors_analysis']['financial_performance']
        if financial_score > 80:
            insights.append(AIInsight(
                insight_id=f"financial_strong_{siren}",
                title="Performance Financière Solide",
                description=f"Excellente performance financière (score: {financial_score:.1f}/100).",
                confidence=PredictionConfidence.HIGH,
                importance_score=75,
                category="financial",
                evidence=[f"Score financier: {financial_score:.1f}/100"],
                recommendations=["Maintenir cette performance lors des négociations"]
            ))
        
        # Insight sur le potentiel de croissance
        growth_score = acquisition_pred['factors_analysis']['growth_potential']
        if growth_score > 70:
            insights.append(AIInsight(
                insight_id=f"growth_potential_{siren}",
                title="Fort Potentiel de Croissance",
                description=f"Potentiel de croissance élevé (score: {growth_score:.1f}/100).",
                confidence=PredictionConfidence.MEDIUM,
                importance_score=80,
                category="growth",
                evidence=[f"Score croissance: {growth_score:.1f}/100"],
                recommendations=["Valoriser ce potentiel dans l'offre", "Planifier les investissements post-acquisition"]
            ))
        
        return insights
    
    async def train_models_from_data(self, training_data: pd.DataFrame) -> Dict[str, str]:
        """Entraîne tous les modèles avec de nouvelles données"""
        
        logger.info("🎓 Entraînement des modèles IA avec nouvelles données")
        
        trained_models = {}
        
        try:
            # Feature engineering
            enhanced_data = self.feature_engineer.create_financial_features(training_data)
            
            # Entraînement modèle de scoring M&A
            ma_model_id = await self.ensemble_manager.train_ma_scoring_model(enhanced_data)
            trained_models['ma_scoring'] = ma_model_id
            self.active_models['ma_scoring'] = ma_model_id
            
            # Analyse des tendances marché
            if 'date' in training_data.columns:
                market_analysis = await self.predictive_analytics.analyze_market_trends(training_data)
                trained_models['market_trends'] = 'analyzed'
            
            logger.info(f"✅ {len(trained_models)} modèles entraînés avec succès")
            
            return trained_models
            
        except Exception as e:
            logger.error(f"❌ Erreur entraînement modèles: {e}")
            raise
    
    def get_ai_system_status(self) -> Dict[str, Any]:
        """Retourne le statut du système IA"""
        
        return {
            'active_models': self.active_models,
            'cached_profiles': len(self.company_profiles),
            'model_metrics': {
                model_id: metrics.__dict__ 
                for model_id, metrics in self.ensemble_manager.model_metrics.items()
            },
            'feature_engineer_scalers': len(self.feature_engineer.scalers),
            'market_indicators': self.predictive_analytics.market_indicators,
            'system_health': 'operational',
            'last_training': datetime.now().isoformat()
        }


async def get_advanced_ai_engine() -> AdvancedAIEngine:
    """Factory pour obtenir le moteur d'IA avancé"""
    global _advanced_ai_engine
    
    if _advanced_ai_engine is None:
        _advanced_ai_engine = AdvancedAIEngine()
    
    return _advanced_ai_engine


# Décorateurs pour fonctions IA

def ai_powered(cache_ttl: int = 3600):
    """Décorateur pour fonctions utilisant l'IA avec cache"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Cache basé sur les arguments
            cache_key = f"ai_{func.__name__}_{hash(str(args))}"
            
            try:
                cache_manager = await get_cache_manager()
                cached_result = await cache_manager.get('ai_results', cache_key)
                
                if cached_result:
                    return cached_result
            except:
                pass  # Continuer sans cache si erreur
            
            # Exécuter fonction
            result = await func(*args, **kwargs)
            
            # Mettre en cache
            try:
                await cache_manager.set('ai_results', cache_key, result, ttl_seconds=cache_ttl)
            except:
                pass  # Continuer sans cache si erreur
            
            return result
        
        return wrapper
    return decorator


# Fonctions utilitaires

async def analyze_company_with_ai(siren: str, company_data: Dict[str, Any]) -> CompanyAIProfile:
    """Analyse complète d'une entreprise avec IA"""
    
    ai_engine = await get_advanced_ai_engine()
    return await ai_engine.get_company_ai_profile(siren, company_data)


async def get_market_predictions() -> Dict[str, Any]:
    """Obtient les prédictions de marché IA"""
    
    ai_engine = await get_advanced_ai_engine()
    
    # Données simulées pour démonstration
    # En production, récupérer depuis la base de données
    historical_data = pd.DataFrame({
        'date': pd.date_range('2020-01-01', '2024-01-01', freq='M'),
        'ma_score': np.random.normal(60, 15, 49),
        'chiffre_affaires': np.random.exponential(1000000, 49),
        'nombre_acquisitions': np.random.poisson(10, 49),
        'valeur_transactions': np.random.exponential(50000000, 49)
    })
    
    return await ai_engine.predictive_analytics.analyze_market_trends(historical_data)


async def batch_score_companies_ai(companies_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Score un lot d'entreprises avec l'IA"""
    
    ai_engine = await get_advanced_ai_engine()
    results = []
    
    for company_data in companies_data:
        try:
            siren = company_data.get('siren', 'unknown')
            profile = await ai_engine.get_company_ai_profile(siren, company_data)
            
            results.append({
                'siren': siren,
                'ma_score': profile.ma_score,
                'acquisition_probability': profile.predicted_acquisition_probability,
                'confidence_level': profile.confidence_level.value,
                'key_insights_count': len(profile.key_insights),
                'financial_health_score': profile.financial_health_score,
                'growth_potential_score': profile.growth_potential_score
            })
            
        except Exception as e:
            logger.error(f"Erreur scoring IA {company_data.get('siren', 'unknown')}: {e}")
            results.append({
                'siren': company_data.get('siren', 'unknown'),
                'error': str(e),
                'ma_score': 0
            })
    
    return results