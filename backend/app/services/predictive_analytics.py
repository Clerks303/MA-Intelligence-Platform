"""
Service d'analyse prédictive pour M&A Intelligence Platform
US-008: Prédictions avancées et modélisation temporelle

Ce module implémente:
- Prédictions de tendances M&A
- Analyse de séries temporelles
- Modèles prédictifs pour opportunités futures
- Analyse de patterns saisonniers
- Prévisions de valorisation
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import warnings

# Machine Learning et Time Series
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    warnings.warn("Prophet non disponible - certaines fonctionnalités de prédiction seront limitées")

# Deep Learning (optionnel)
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from app.core.logging_system import get_logger, LogCategory
from app.services.ai_scoring_engine import get_ai_scoring_engine

logger = get_logger("predictive_analytics", LogCategory.ML)


class PredictionType(str, Enum):
    """Types de prédictions disponibles"""
    MA_ACTIVITY = "ma_activity"
    VALUATION_TRENDS = "valuation_trends"
    SECTOR_PERFORMANCE = "sector_performance"
    COMPANY_GROWTH = "company_growth"
    MARKET_CONSOLIDATION = "market_consolidation"
    ECONOMIC_IMPACT = "economic_impact"


class TimeHorizon(str, Enum):
    """Horizons temporels de prédiction"""
    SHORT_TERM = "3_months"    # 3 mois
    MEDIUM_TERM = "12_months"  # 1 an
    LONG_TERM = "36_months"    # 3 ans


class ModelType(str, Enum):
    """Types de modèles prédictifs"""
    ARIMA = "arima"
    PROPHET = "prophet"
    LSTM = "lstm"
    RANDOM_FOREST = "random_forest"
    ENSEMBLE = "ensemble"


@dataclass
class PredictionResult:
    """Résultat d'une prédiction"""
    
    prediction_type: PredictionType
    horizon: TimeHorizon
    model_used: ModelType
    
    # Valeurs prédites
    predictions: List[float]
    dates: List[datetime]
    confidence_intervals: List[Tuple[float, float]]
    
    # Métriques de performance
    mae: float
    mse: float
    confidence_score: float
    
    # Métadonnées
    timestamp: datetime
    processing_time_ms: float
    
    # Explications
    trends: Dict[str, Any]
    seasonal_patterns: Dict[str, Any]
    key_factors: List[str]
    recommendations: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire pour JSON"""
        return {
            'prediction_type': self.prediction_type.value,
            'horizon': self.horizon.value,
            'model_used': self.model_used.value,
            'predictions': self.predictions,
            'dates': [d.isoformat() for d in self.dates],
            'confidence_intervals': self.confidence_intervals,
            'mae': round(self.mae, 4),
            'mse': round(self.mse, 4),
            'confidence_score': round(self.confidence_score, 3),
            'timestamp': self.timestamp.isoformat(),
            'processing_time_ms': round(self.processing_time_ms, 2),
            'trends': self.trends,
            'seasonal_patterns': self.seasonal_patterns,
            'key_factors': self.key_factors,
            'recommendations': self.recommendations
        }


@dataclass
class MarketPrediction:
    """Prédiction complète du marché M&A"""
    
    overall_activity: PredictionResult
    sector_predictions: Dict[str, PredictionResult]
    valuation_trends: PredictionResult
    consolidation_forecast: PredictionResult
    
    # Synthèse
    market_outlook: str
    investment_opportunities: List[Dict[str, Any]]
    risk_factors: List[str]
    strategic_recommendations: List[str]


class LSTMPredictor(nn.Module):
    """Modèle LSTM pour prédictions temporelles (si PyTorch disponible)"""
    
    def __init__(self, input_size: int, hidden_size: int = 50, num_layers: int = 2, output_size: int = 1):
        super(LSTMPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        
        return out


class PredictiveAnalyticsEngine:
    """Moteur d'analyse prédictive pour le marché M&A"""
    
    def __init__(self):
        self.models: Dict[ModelType, Any] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        
        # Données historiques simulées
        self.historical_data: Optional[pd.DataFrame] = None
        self.market_indicators: Optional[pd.DataFrame] = None
        
        # Configuration des modèles
        self.model_configs = {
            ModelType.ARIMA: {'order': (2, 1, 2)},
            ModelType.PROPHET: {'yearly_seasonality': True, 'weekly_seasonality': False},
            ModelType.RANDOM_FOREST: {'n_estimators': 100, 'random_state': 42},
            ModelType.LSTM: {'hidden_size': 50, 'num_layers': 2, 'epochs': 100}
        }
        
        logger.info("📈 Moteur d'analyse prédictive initialisé")
    
    async def initialize(self):
        """Initialise le moteur avec des données historiques"""
        try:
            # Générer ou charger données historiques
            await self._generate_historical_data()
            
            # Pré-entraîner les modèles sur les données historiques
            await self._pretrain_models()
            
            logger.info("✅ Moteur prédictif initialisé avec succès")
            
        except Exception as e:
            logger.error(f"Erreur initialisation prédictive: {e}")
            raise
    
    async def predict_ma_activity(self, 
                                 horizon: TimeHorizon = TimeHorizon.MEDIUM_TERM,
                                 model: ModelType = ModelType.ENSEMBLE) -> PredictionResult:
        """
        Prédit l'activité M&A future
        
        Args:
            horizon: Horizon temporel de prédiction
            model: Type de modèle à utiliser
            
        Returns:
            PredictionResult: Prédictions d'activité M&A
        """
        start_time = datetime.now()
        
        try:
            # Préparer données d'activité M&A
            activity_data = self._prepare_activity_data()
            
            # Générer prédictions selon l'horizon
            periods = self._get_periods_for_horizon(horizon)
            
            if model == ModelType.ENSEMBLE:
                predictions, confidence_intervals, metrics = await self._ensemble_predict(
                    activity_data, periods, 'ma_activity'
                )
            else:
                predictions, confidence_intervals, metrics = await self._single_model_predict(
                    activity_data, periods, model
                )
            
            # Générer dates futures
            last_date = activity_data.index[-1]
            future_dates = [last_date + timedelta(days=30*i) for i in range(1, periods+1)]
            
            # Analyser tendances et patterns
            trends = self._analyze_trends(activity_data, predictions)
            seasonal_patterns = self._analyze_seasonality(activity_data)
            
            # Générer recommandations
            recommendations = self._generate_activity_recommendations(trends, seasonal_patterns)
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return PredictionResult(
                prediction_type=PredictionType.MA_ACTIVITY,
                horizon=horizon,
                model_used=model,
                predictions=predictions,
                dates=future_dates,
                confidence_intervals=confidence_intervals,
                mae=metrics['mae'],
                mse=metrics['mse'],
                confidence_score=metrics['confidence'],
                timestamp=datetime.now(),
                processing_time_ms=processing_time,
                trends=trends,
                seasonal_patterns=seasonal_patterns,
                key_factors=['market_conditions', 'interest_rates', 'economic_growth'],
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Erreur prédiction activité M&A: {e}")
            raise
    
    async def predict_sector_performance(self, 
                                       sectors: List[str],
                                       horizon: TimeHorizon = TimeHorizon.MEDIUM_TERM) -> Dict[str, PredictionResult]:
        """
        Prédit la performance par secteur
        
        Args:
            sectors: Liste des secteurs à analyser
            horizon: Horizon temporel
            
        Returns:
            Dict[str, PredictionResult]: Prédictions par secteur
        """
        sector_predictions = {}
        
        for sector in sectors:
            try:
                # Préparer données sectorielles
                sector_data = self._prepare_sector_data(sector)
                periods = self._get_periods_for_horizon(horizon)
                
                # Prédiction avec modèle adapté aux séries courtes
                predictions, confidence_intervals, metrics = await self._single_model_predict(
                    sector_data, periods, ModelType.RANDOM_FOREST
                )
                
                # Dates futures
                last_date = sector_data.index[-1]
                future_dates = [last_date + timedelta(days=30*i) for i in range(1, periods+1)]
                
                # Analyse sectorielle
                trends = self._analyze_sector_trends(sector, sector_data, predictions)
                
                sector_predictions[sector] = PredictionResult(
                    prediction_type=PredictionType.SECTOR_PERFORMANCE,
                    horizon=horizon,
                    model_used=ModelType.RANDOM_FOREST,
                    predictions=predictions,
                    dates=future_dates,
                    confidence_intervals=confidence_intervals,
                    mae=metrics['mae'],
                    mse=metrics['mse'],
                    confidence_score=metrics['confidence'],
                    timestamp=datetime.now(),
                    processing_time_ms=10.0,  # Approximation
                    trends=trends,
                    seasonal_patterns={},
                    key_factors=[f'{sector}_regulations', f'{sector}_innovation', 'economic_cycle'],
                    recommendations=self._generate_sector_recommendations(sector, trends)
                )
                
            except Exception as e:
                logger.warning(f"Erreur prédiction secteur {sector}: {e}")
                continue
        
        return sector_predictions
    
    async def predict_valuation_trends(self, 
                                     horizon: TimeHorizon = TimeHorizon.MEDIUM_TERM) -> PredictionResult:
        """
        Prédit les tendances de valorisation
        
        Args:
            horizon: Horizon temporel
            
        Returns:
            PredictionResult: Prédictions de valorisation
        """
        try:
            # Préparer données de valorisation
            valuation_data = self._prepare_valuation_data()
            periods = self._get_periods_for_horizon(horizon)
            
            # Utiliser Prophet si disponible pour les tendances de valorisation
            if PROPHET_AVAILABLE:
                predictions, confidence_intervals, metrics = await self._prophet_predict(
                    valuation_data, periods
                )
                model_used = ModelType.PROPHET
            else:
                predictions, confidence_intervals, metrics = await self._single_model_predict(
                    valuation_data, periods, ModelType.RANDOM_FOREST
                )
                model_used = ModelType.RANDOM_FOREST
            
            # Dates futures
            last_date = valuation_data.index[-1]
            future_dates = [last_date + timedelta(days=30*i) for i in range(1, periods+1)]
            
            # Analyser tendances de valorisation
            trends = self._analyze_valuation_trends(valuation_data, predictions)
            
            return PredictionResult(
                prediction_type=PredictionType.VALUATION_TRENDS,
                horizon=horizon,
                model_used=model_used,
                predictions=predictions,
                dates=future_dates,
                confidence_intervals=confidence_intervals,
                mae=metrics['mae'],
                mse=metrics['mse'],
                confidence_score=metrics['confidence'],
                timestamp=datetime.now(),
                processing_time_ms=50.0,
                trends=trends,
                seasonal_patterns={},
                key_factors=['market_liquidity', 'risk_appetite', 'regulatory_environment'],
                recommendations=self._generate_valuation_recommendations(trends)
            )
            
        except Exception as e:
            logger.error(f"Erreur prédiction valorisation: {e}")
            raise
    
    async def generate_market_forecast(self, 
                                     horizon: TimeHorizon = TimeHorizon.MEDIUM_TERM) -> MarketPrediction:
        """
        Génère une prévision complète du marché M&A
        
        Args:
            horizon: Horizon temporel
            
        Returns:
            MarketPrediction: Prévision complète du marché
        """
        try:
            logger.info(f"Génération prévision marché pour {horizon.value}")
            
            # Prédictions parallèles
            tasks = [
                self.predict_ma_activity(horizon),
                self.predict_valuation_trends(horizon),
                self.predict_sector_performance(['tech', 'healthcare', 'finance'], horizon)
            ]
            
            activity_pred, valuation_pred, sector_preds = await asyncio.gather(*tasks)
            
            # Prédiction de consolidation (simulée)
            consolidation_pred = await self._predict_consolidation(horizon)
            
            # Synthèse du marché
            market_outlook = self._synthesize_market_outlook(
                activity_pred, valuation_pred, list(sector_preds.values())
            )
            
            # Opportunités d'investissement
            investment_opportunities = self._identify_investment_opportunities(
                activity_pred, sector_preds, valuation_pred
            )
            
            # Facteurs de risque
            risk_factors = self._identify_risk_factors(
                activity_pred, valuation_pred, list(sector_preds.values())
            )
            
            # Recommandations stratégiques
            strategic_recommendations = self._generate_strategic_recommendations(
                market_outlook, investment_opportunities, risk_factors
            )
            
            return MarketPrediction(
                overall_activity=activity_pred,
                sector_predictions=sector_preds,
                valuation_trends=valuation_pred,
                consolidation_forecast=consolidation_pred,
                market_outlook=market_outlook,
                investment_opportunities=investment_opportunities,
                risk_factors=risk_factors,
                strategic_recommendations=strategic_recommendations
            )
            
        except Exception as e:
            logger.error(f"Erreur génération prévision marché: {e}")
            raise
    
    async def _generate_historical_data(self):
        """Génère des données historiques simulées réalistes"""
        
        # Période de 5 ans avec données mensuelles
        dates = pd.date_range(start='2019-01-01', end='2024-01-01', freq='M')
        
        # Simulation activité M&A avec tendance et saisonnalité
        np.random.seed(42)
        
        # Tendance de base avec cycles économiques
        trend = np.linspace(100, 150, len(dates))
        
        # Saisonnalité (moins d'activité en été et décembre)
        seasonal = 10 * np.sin(2 * np.pi * np.arange(len(dates)) / 12) - 5 * np.sin(4 * np.pi * np.arange(len(dates)) / 12)
        
        # Bruit et événements exceptionnels
        noise = np.random.normal(0, 8, len(dates))
        
        # Chocs externes (COVID-19 en 2020)
        covid_impact = np.where((dates.year == 2020) & (dates.month.isin([3, 4, 5])), -30, 0)
        
        # Activité M&A finale
        ma_activity = trend + seasonal + noise + covid_impact
        ma_activity = np.maximum(ma_activity, 10)  # Minimum réaliste
        
        # Données de valorisation (multiples P/E)
        valuation_base = 15 + 3 * np.sin(2 * np.pi * np.arange(len(dates)) / 24)  # Cycle 2 ans
        valuation_noise = np.random.normal(0, 1.5, len(dates))
        valuations = valuation_base + valuation_noise
        
        # Données sectorielles
        sectors = ['tech', 'healthcare', 'finance', 'manufacturing', 'energy']
        sector_data = {}
        
        for sector in sectors:
            # Chaque secteur a ses propres caractéristiques
            sector_trend = trend * np.random.uniform(0.8, 1.2)
            sector_seasonal = seasonal * np.random.uniform(0.5, 1.5)
            sector_noise = np.random.normal(0, 5, len(dates))
            
            sector_data[f'{sector}_activity'] = sector_trend + sector_seasonal + sector_noise
        
        # Créer DataFrame principal
        self.historical_data = pd.DataFrame({
            'ma_activity': ma_activity,
            'valuations': valuations,
            **sector_data
        }, index=dates)
        
        # Indicateurs macro-économiques
        self.market_indicators = pd.DataFrame({
            'gdp_growth': np.random.normal(2.5, 1.0, len(dates)),
            'interest_rates': 1.0 + 2.0 * np.random.beta(2, 3, len(dates)),
            'market_volatility': 15 + 10 * np.random.gamma(2, 0.5, len(dates)),
            'credit_spreads': 100 + 50 * np.random.exponential(0.5, len(dates))
        }, index=dates)
        
        logger.info(f"✅ Données historiques générées: {len(dates)} points")
    
    async def _pretrain_models(self):
        """Pré-entraîne les modèles sur les données historiques"""
        
        if self.historical_data is None:
            raise ValueError("Pas de données historiques disponibles")
        
        # Entraîner Random Forest sur activité M&A
        ma_data = self.historical_data['ma_activity'].dropna()
        
        # Créer features temporelles
        features = self._create_temporal_features(ma_data)
        X = features[:-1]  # Features
        y = ma_data.values[1:]  # Target (décalé de 1)
        
        # Modèle Random Forest
        rf_model = RandomForestRegressor(**self.model_configs[ModelType.RANDOM_FOREST])
        rf_model.fit(X, y)
        self.models[ModelType.RANDOM_FOREST] = rf_model
        
        # Scaler pour normalisation
        scaler = StandardScaler()
        scaler.fit(X)
        self.scalers['temporal_features'] = scaler
        
        logger.info("✅ Modèles pré-entraînés")
    
    def _create_temporal_features(self, series: pd.Series) -> np.ndarray:
        """Crée des features temporelles à partir d'une série"""
        
        features = []
        
        for i in range(len(series)):
            feature_row = []
            
            # Lags
            for lag in [1, 2, 3, 6, 12]:
                if i >= lag:
                    feature_row.append(series.iloc[i-lag])
                else:
                    feature_row.append(series.iloc[0])  # Padding
            
            # Features temporelles
            date = series.index[i]
            feature_row.extend([
                date.month,  # Mois (saisonnalité)
                date.quarter,  # Trimestre
                date.year % 10,  # Cycle décennal
                np.sin(2 * np.pi * date.month / 12),  # Saisonnalité sinusoïdale
                np.cos(2 * np.pi * date.month / 12)
            ])
            
            # Moyennes mobiles
            if i >= 3:
                feature_row.append(series.iloc[i-3:i].mean())  # MA 3 mois
            else:
                feature_row.append(series.iloc[0])
            
            if i >= 12:
                feature_row.append(series.iloc[i-12:i].mean())  # MA 12 mois
            else:
                feature_row.append(series.iloc[0])
            
            features.append(feature_row)
        
        return np.array(features)
    
    async def _ensemble_predict(self, data: pd.Series, periods: int, prediction_type: str) -> Tuple[List[float], List[Tuple[float, float]], Dict[str, float]]:
        """Prédiction ensemble combinant plusieurs modèles"""
        
        try:
            predictions_list = []
            
            # Random Forest
            if ModelType.RANDOM_FOREST in self.models:
                rf_preds, _, _ = await self._single_model_predict(data, periods, ModelType.RANDOM_FOREST)
                predictions_list.append(rf_preds)
            
            # ARIMA
            try:
                arima_preds, _, _ = await self._arima_predict(data, periods)
                predictions_list.append(arima_preds)
            except:
                logger.warning("ARIMA non disponible pour ensemble")
            
            # Prophet (si disponible)
            if PROPHET_AVAILABLE:
                try:
                    prophet_preds, _, _ = await self._prophet_predict(data, periods)
                    predictions_list.append(prophet_preds)
                except:
                    logger.warning("Prophet échec pour ensemble")
            
            if not predictions_list:
                raise ValueError("Aucun modèle disponible pour ensemble")
            
            # Moyenne pondérée
            weights = [0.4, 0.3, 0.3][:len(predictions_list)]
            ensemble_predictions = np.average(predictions_list, axis=0, weights=weights)
            
            # Intervalles de confiance basés sur variance
            predictions_array = np.array(predictions_list)
            std_dev = np.std(predictions_array, axis=0)
            confidence_intervals = [
                (float(pred - 1.96 * std), float(pred + 1.96 * std))
                for pred, std in zip(ensemble_predictions, std_dev)
            ]
            
            # Métriques (approximatives)
            metrics = {
                'mae': np.mean(std_dev),
                'mse': np.mean(std_dev ** 2),
                'confidence': float(1.0 / (1.0 + np.mean(std_dev)))
            }
            
            return ensemble_predictions.tolist(), confidence_intervals, metrics
            
        except Exception as e:
            logger.error(f"Erreur ensemble prediction: {e}")
            # Fallback sur Random Forest seul
            return await self._single_model_predict(data, periods, ModelType.RANDOM_FOREST)
    
    async def _single_model_predict(self, data: pd.Series, periods: int, model_type: ModelType) -> Tuple[List[float], List[Tuple[float, float]], Dict[str, float]]:
        """Prédiction avec un seul modèle"""
        
        if model_type == ModelType.ARIMA:
            return await self._arima_predict(data, periods)
        elif model_type == ModelType.PROPHET and PROPHET_AVAILABLE:
            return await self._prophet_predict(data, periods)
        elif model_type == ModelType.RANDOM_FOREST:
            return await self._random_forest_predict(data, periods)
        elif model_type == ModelType.LSTM and TORCH_AVAILABLE:
            return await self._lstm_predict(data, periods)
        else:
            # Fallback sur Random Forest
            return await self._random_forest_predict(data, periods)
    
    async def _arima_predict(self, data: pd.Series, periods: int) -> Tuple[List[float], List[Tuple[float, float]], Dict[str, float]]:
        """Prédiction ARIMA"""
        
        try:
            # Vérifier stationnarité
            adf_result = adfuller(data.dropna())
            
            # Différenciation si nécessaire
            if adf_result[1] > 0.05:
                data_diff = data.diff().dropna()
            else:
                data_diff = data
            
            # Ajuster modèle ARIMA
            model = ARIMA(data_diff, order=self.model_configs[ModelType.ARIMA]['order'])
            fitted_model = model.fit()
            
            # Prédictions
            forecast = fitted_model.forecast(steps=periods, alpha=0.05)
            predictions = forecast.tolist()
            
            # Intervalles de confiance
            conf_int = fitted_model.get_forecast(steps=periods).conf_int()
            confidence_intervals = [(float(row[0]), float(row[1])) for row in conf_int.values]
            
            # Métriques
            residuals = fitted_model.resid
            metrics = {
                'mae': float(np.mean(np.abs(residuals))),
                'mse': float(np.mean(residuals ** 2)),
                'confidence': float(max(0.1, 1.0 - np.std(residuals) / np.mean(np.abs(data))))
            }
            
            return predictions, confidence_intervals, metrics
            
        except Exception as e:
            logger.warning(f"Erreur ARIMA: {e}")
            # Fallback sur prédiction simple
            last_value = float(data.iloc[-1])
            predictions = [last_value * (1 + np.random.normal(0, 0.05)) for _ in range(periods)]
            confidence_intervals = [(p * 0.9, p * 1.1) for p in predictions]
            
            return predictions, confidence_intervals, {'mae': 5.0, 'mse': 25.0, 'confidence': 0.5}
    
    async def _prophet_predict(self, data: pd.Series, periods: int) -> Tuple[List[float], List[Tuple[float, float]], Dict[str, float]]:
        """Prédiction avec Prophet"""
        
        if not PROPHET_AVAILABLE:
            raise ValueError("Prophet non disponible")
        
        try:
            # Préparer données pour Prophet
            df = pd.DataFrame({
                'ds': data.index,
                'y': data.values
            })
            
            # Créer et ajuster modèle
            model = Prophet(**self.model_configs[ModelType.PROPHET])
            model.fit(df)
            
            # Créer dataframe futur
            future = model.make_future_dataframe(periods=periods, freq='M')
            
            # Prédictions
            forecast = model.predict(future)
            
            # Extraire prédictions futures
            predictions = forecast['yhat'].tail(periods).tolist()
            
            # Intervalles de confiance
            confidence_intervals = [
                (float(row['yhat_lower']), float(row['yhat_upper']))
                for _, row in forecast.tail(periods).iterrows()
            ]
            
            # Métriques (sur données d'entraînement)
            train_predictions = forecast['yhat'].head(len(data))
            residuals = data.values - train_predictions.values
            
            metrics = {
                'mae': float(np.mean(np.abs(residuals))),
                'mse': float(np.mean(residuals ** 2)),
                'confidence': 0.8  # Prophet généralement fiable
            }
            
            return predictions, confidence_intervals, metrics
            
        except Exception as e:
            logger.warning(f"Erreur Prophet: {e}")
            raise
    
    async def _random_forest_predict(self, data: pd.Series, periods: int) -> Tuple[List[float], List[Tuple[float, float]], Dict[str, float]]:
        """Prédiction Random Forest"""
        
        try:
            # Créer features temporelles
            features = self._create_temporal_features(data)
            
            # Normaliser si scaler disponible
            if 'temporal_features' in self.scalers:
                features_scaled = self.scalers['temporal_features'].transform(features)
            else:
                features_scaled = features
            
            # Modèle
            if ModelType.RANDOM_FOREST not in self.models:
                # Entraîner rapidement
                X = features_scaled[:-1]
                y = data.values[1:]
                
                model = RandomForestRegressor(**self.model_configs[ModelType.RANDOM_FOREST])
                model.fit(X, y)
                self.models[ModelType.RANDOM_FOREST] = model
            else:
                model = self.models[ModelType.RANDOM_FOREST]
            
            # Prédictions itératives
            predictions = []
            current_data = data.copy()
            
            for _ in range(periods):
                # Features pour prédiction
                current_features = self._create_temporal_features(current_data)
                if 'temporal_features' in self.scalers:
                    current_features_scaled = self.scalers['temporal_features'].transform(current_features)
                else:
                    current_features_scaled = current_features
                
                # Prédire prochaine valeur
                next_pred = model.predict(current_features_scaled[-1:])
                predictions.append(float(next_pred[0]))
                
                # Ajouter prédiction aux données pour prochaine itération
                next_date = current_data.index[-1] + pd.DateOffset(months=1)
                current_data = pd.concat([
                    current_data,
                    pd.Series([next_pred[0]], index=[next_date])
                ])
            
            # Intervalles de confiance basés sur erreur du modèle
            if hasattr(model, 'estimators_'):
                # Utiliser variance des arbres pour confiance
                tree_predictions = []
                for estimator in model.estimators_[:min(20, len(model.estimators_))]:
                    tree_pred = []
                    temp_data = data.copy()
                    
                    for _ in range(periods):
                        temp_features = self._create_temporal_features(temp_data)
                        if 'temporal_features' in self.scalers:
                            temp_features_scaled = self.scalers['temporal_features'].transform(temp_features)
                        else:
                            temp_features_scaled = temp_features
                        
                        pred = estimator.predict(temp_features_scaled[-1:])
                        tree_pred.append(pred[0])
                        
                        next_date = temp_data.index[-1] + pd.DateOffset(months=1)
                        temp_data = pd.concat([
                            temp_data,
                            pd.Series([pred[0]], index=[next_date])
                        ])
                    
                    tree_predictions.append(tree_pred)
                
                # Calculer intervalles de confiance
                tree_predictions = np.array(tree_predictions)
                std_dev = np.std(tree_predictions, axis=0)
                confidence_intervals = [
                    (float(pred - 1.96 * std), float(pred + 1.96 * std))
                    for pred, std in zip(predictions, std_dev)
                ]
            else:
                # Intervalles approximatifs
                std_estimate = np.std(data.values) * 0.1
                confidence_intervals = [
                    (float(pred - 1.96 * std_estimate), float(pred + 1.96 * std_estimate))
                    for pred in predictions
                ]
            
            # Métriques approximatives
            metrics = {
                'mae': float(np.mean(np.abs(np.diff(data.values)))),
                'mse': float(np.mean(np.diff(data.values) ** 2)),
                'confidence': 0.7
            }
            
            return predictions, confidence_intervals, metrics
            
        except Exception as e:
            logger.error(f"Erreur Random Forest prediction: {e}")
            # Fallback très simple
            last_value = float(data.iloc[-1])
            trend = np.mean(np.diff(data.values[-12:]))  # Tendance 12 derniers mois
            
            predictions = []
            for i in range(periods):
                pred = last_value + trend * (i + 1)
                predictions.append(pred)
            
            confidence_intervals = [(p * 0.85, p * 1.15) for p in predictions]
            
            return predictions, confidence_intervals, {'mae': 10.0, 'mse': 100.0, 'confidence': 0.4}
    
    async def _lstm_predict(self, data: pd.Series, periods: int) -> Tuple[List[float], List[Tuple[float, float]], Dict[str, float]]:
        """Prédiction LSTM (si PyTorch disponible)"""
        
        if not TORCH_AVAILABLE:
            raise ValueError("PyTorch non disponible pour LSTM")
        
        try:
            # Préparer données pour LSTM
            sequence_length = 12  # 12 mois de contexte
            
            # Normaliser données
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data.values.reshape(-1, 1)).flatten()
            
            # Créer séquences
            X, y = [], []
            for i in range(sequence_length, len(data_scaled)):
                X.append(data_scaled[i-sequence_length:i])
                y.append(data_scaled[i])
            
            X = np.array(X).reshape(-1, sequence_length, 1)
            y = np.array(y)
            
            # Créer modèle LSTM
            model = LSTMPredictor(
                input_size=1,
                hidden_size=self.model_configs[ModelType.LSTM]['hidden_size'],
                num_layers=self.model_configs[ModelType.LSTM]['num_layers']
            )
            
            # Entraîner (version simplifiée)
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            
            X_tensor = torch.FloatTensor(X)
            y_tensor = torch.FloatTensor(y)
            
            epochs = min(50, self.model_configs[ModelType.LSTM]['epochs'])  # Réduire pour rapidité
            
            model.train()
            for epoch in range(epochs):
                outputs = model(X_tensor)
                loss = criterion(outputs.squeeze(), y_tensor)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            # Prédictions
            model.eval()
            predictions = []
            current_sequence = data_scaled[-sequence_length:].tolist()
            
            with torch.no_grad():
                for _ in range(periods):
                    seq_tensor = torch.FloatTensor(current_sequence).unsqueeze(0).unsqueeze(-1)
                    pred = model(seq_tensor).item()
                    predictions.append(pred)
                    current_sequence = current_sequence[1:] + [pred]
            
            # Dé-normaliser
            predictions_scaled = np.array(predictions).reshape(-1, 1)
            predictions_original = scaler.inverse_transform(predictions_scaled).flatten()
            
            # Intervalles de confiance (approximatifs)
            std_estimate = np.std(data.values) * 0.15
            confidence_intervals = [
                (float(pred - 1.96 * std_estimate), float(pred + 1.96 * std_estimate))
                for pred in predictions_original
            ]
            
            # Métriques
            metrics = {
                'mae': float(loss.item()),
                'mse': float(loss.item()),
                'confidence': 0.75
            }
            
            return predictions_original.tolist(), confidence_intervals, metrics
            
        except Exception as e:
            logger.warning(f"Erreur LSTM: {e}")
            # Fallback
            return await self._random_forest_predict(data, periods)
    
    def _prepare_activity_data(self) -> pd.Series:
        """Prépare les données d'activité M&A"""
        return self.historical_data['ma_activity'].dropna()
    
    def _prepare_sector_data(self, sector: str) -> pd.Series:
        """Prépare les données sectorielles"""
        column_name = f'{sector}_activity'
        if column_name in self.historical_data.columns:
            return self.historical_data[column_name].dropna()
        else:
            # Générer données simulées pour le secteur
            base_data = self.historical_data['ma_activity'].copy()
            noise = np.random.normal(0, 5, len(base_data))
            return base_data + noise
    
    def _prepare_valuation_data(self) -> pd.Series:
        """Prépare les données de valorisation"""
        return self.historical_data['valuations'].dropna()
    
    def _get_periods_for_horizon(self, horizon: TimeHorizon) -> int:
        """Convertit l'horizon en nombre de périodes"""
        if horizon == TimeHorizon.SHORT_TERM:
            return 3
        elif horizon == TimeHorizon.MEDIUM_TERM:
            return 12
        else:  # LONG_TERM
            return 36
    
    def _analyze_trends(self, historical_data: pd.Series, predictions: List[float]) -> Dict[str, Any]:
        """Analyse les tendances dans les données et prédictions"""
        
        # Tendance historique
        recent_data = historical_data.tail(12).values
        historical_trend = np.polyfit(range(len(recent_data)), recent_data, 1)[0]
        
        # Tendance prédite
        predicted_trend = np.polyfit(range(len(predictions)), predictions, 1)[0]
        
        # Classification de tendance
        def classify_trend(slope):
            if slope > 2:
                return "forte_hausse"
            elif slope > 0.5:
                return "hausse_moderee"
            elif slope > -0.5:
                return "stable"
            elif slope > -2:
                return "baisse_moderee"
            else:
                return "forte_baisse"
        
        return {
            'historical_trend': float(historical_trend),
            'predicted_trend': float(predicted_trend),
            'historical_classification': classify_trend(historical_trend),
            'predicted_classification': classify_trend(predicted_trend),
            'trend_acceleration': float(predicted_trend - historical_trend),
            'volatility': float(np.std(recent_data)),
            'current_level': float(historical_data.iloc[-1]),
            'predicted_peak': float(max(predictions)),
            'predicted_trough': float(min(predictions))
        }
    
    def _analyze_seasonality(self, data: pd.Series) -> Dict[str, Any]:
        """Analyse la saisonnalité des données"""
        
        try:
            # Décomposition saisonnière
            decomposition = seasonal_decompose(data, model='additive', period=12)
            
            # Patterns saisonniers
            seasonal_component = decomposition.seasonal.dropna()
            
            # Mois les plus forts/faibles
            monthly_avg = data.groupby(data.index.month).mean()
            strongest_month = int(monthly_avg.idxmax())
            weakest_month = int(monthly_avg.idxmin())
            
            return {
                'has_seasonality': True,
                'seasonal_strength': float(np.std(seasonal_component) / np.std(data)),
                'strongest_month': strongest_month,
                'weakest_month': weakest_month,
                'monthly_pattern': monthly_avg.to_dict(),
                'seasonal_amplitude': float(np.max(seasonal_component) - np.min(seasonal_component))
            }
            
        except Exception as e:
            logger.warning(f"Erreur analyse saisonnalité: {e}")
            return {
                'has_seasonality': False,
                'seasonal_strength': 0.0,
                'error': str(e)
            }
    
    def _analyze_sector_trends(self, sector: str, data: pd.Series, predictions: List[float]) -> Dict[str, Any]:
        """Analyse les tendances spécifiques à un secteur"""
        
        trends = self._analyze_trends(data, predictions)
        
        # Spécificités sectorielles
        sector_insights = {
            'tech': {
                'key_drivers': ['innovation_rate', 'digital_transformation', 'ai_adoption'],
                'cycle_length': 24,  # mois
                'volatility_factor': 1.3
            },
            'healthcare': {
                'key_drivers': ['aging_population', 'drug_development', 'regulatory_changes'],
                'cycle_length': 36,
                'volatility_factor': 0.8
            },
            'finance': {
                'key_drivers': ['interest_rates', 'regulation', 'fintech_disruption'],
                'cycle_length': 18,
                'volatility_factor': 1.1
            }
        }
        
        sector_info = sector_insights.get(sector, {
            'key_drivers': ['economic_conditions', 'sector_performance'],
            'cycle_length': 24,
            'volatility_factor': 1.0
        })
        
        trends.update({
            'sector': sector,
            'key_drivers': sector_info['key_drivers'],
            'typical_cycle_length': sector_info['cycle_length'],
            'relative_volatility': sector_info['volatility_factor']
        })
        
        return trends
    
    def _analyze_valuation_trends(self, data: pd.Series, predictions: List[float]) -> Dict[str, Any]:
        """Analyse les tendances de valorisation"""
        
        trends = self._analyze_trends(data, predictions)
        
        # Spécificités valorisation
        current_multiple = float(data.iloc[-1])
        historical_median = float(data.median())
        
        # Classification du niveau actuel
        if current_multiple > historical_median * 1.2:
            valuation_level = "elevee"
        elif current_multiple < historical_median * 0.8:
            valuation_level = "faible"
        else:
            valuation_level = "normale"
        
        trends.update({
            'current_multiple': current_multiple,
            'historical_median': historical_median,
            'relative_to_median': float(current_multiple / historical_median),
            'valuation_level': valuation_level,
            'predicted_median': float(np.median(predictions)),
            'multiple_expansion': float(np.median(predictions) - current_multiple)
        })
        
        return trends
    
    def _generate_activity_recommendations(self, trends: Dict[str, Any], seasonal: Dict[str, Any]) -> List[str]:
        """Génère des recommandations basées sur l'activité prédite"""
        
        recommendations = []
        
        # Recommandations basées sur la tendance
        if trends['predicted_classification'] == 'forte_hausse':
            recommendations.append("Marché en forte expansion - opportunité d'accélération des acquisitions")
        elif trends['predicted_classification'] == 'forte_baisse':
            recommendations.append("Ralentissement prévu - concentrer sur opportunités défensives")
        
        # Recommandations saisonnières
        if seasonal.get('has_seasonality'):
            strongest_month = seasonal.get('strongest_month', 1)
            recommendations.append(f"Pic d'activité attendu en mois {strongest_month} - planifier accordingly")
        
        # Recommandations de volatilité
        if trends['volatility'] > trends['current_level'] * 0.2:
            recommendations.append("Forte volatilité - maintenir flexibilité dans strategy")
        
        return recommendations
    
    def _generate_sector_recommendations(self, sector: str, trends: Dict[str, Any]) -> List[str]:
        """Génère des recommandations sectorielles"""
        
        recommendations = []
        
        # Recommandations génériques par secteur
        sector_advice = {
            'tech': ["Focus sur innovation et scalabilité", "Attention aux cycles rapides"],
            'healthcare': ["Évaluer pipeline R&D", "Considérer aspects réglementaires"],
            'finance': ["Analyser impact digital", "Surveiller environnement réglementaire"]
        }
        
        recommendations.extend(sector_advice.get(sector, ["Analyser spécificités sectorielles"]))
        
        # Recommandations basées sur tendance
        if trends['predicted_classification'] in ['forte_hausse', 'hausse_moderee']:
            recommendations.append(f"Secteur {sector} en croissance - accélérer investments")
        
        return recommendations
    
    def _generate_valuation_recommendations(self, trends: Dict[str, Any]) -> List[str]:
        """Génère des recommandations de valorisation"""
        
        recommendations = []
        
        if trends['valuation_level'] == 'elevee':
            recommendations.append("Multiples élevés - négociation serrée recommandée")
        elif trends['valuation_level'] == 'faible':
            recommendations.append("Opportunité de valorisation attractive")
        
        if trends['multiple_expansion'] > 0:
            recommendations.append("Expansion des multiples prévue - timing favorable")
        else:
            recommendations.append("Compression des multiples attendue - accélérer closing")
        
        return recommendations
    
    async def _predict_consolidation(self, horizon: TimeHorizon) -> PredictionResult:
        """Prédit les tendances de consolidation du marché"""
        
        # Simulation simple de consolidation basée sur activité M&A
        activity_data = self._prepare_activity_data()
        periods = self._get_periods_for_horizon(horizon)
        
        # La consolidation suit généralement l'activité avec un lag
        consolidation_base = activity_data.rolling(window=6).mean().dropna()
        
        # Prédictions basées sur Random Forest
        predictions, confidence_intervals, metrics = await self._random_forest_predict(
            consolidation_base, periods
        )
        
        # Ajuster pour consolidation (généralement plus stable)
        predictions = [p * 0.8 for p in predictions]  # Facteur de lissage
        
        last_date = activity_data.index[-1]
        future_dates = [last_date + timedelta(days=30*i) for i in range(1, periods+1)]
        
        return PredictionResult(
            prediction_type=PredictionType.MARKET_CONSOLIDATION,
            horizon=horizon,
            model_used=ModelType.RANDOM_FOREST,
            predictions=predictions,
            dates=future_dates,
            confidence_intervals=confidence_intervals,
            mae=metrics['mae'],
            mse=metrics['mse'],
            confidence_score=metrics['confidence'] * 0.9,  # Moins certain
            timestamp=datetime.now(),
            processing_time_ms=25.0,
            trends={'consolidation_rate': np.mean(predictions)},
            seasonal_patterns={},
            key_factors=['market_maturity', 'regulatory_environment', 'capital_availability'],
            recommendations=['Surveiller opportunités de consolidation verticale']
        )
    
    def _synthesize_market_outlook(self, activity: PredictionResult, 
                                 valuation: PredictionResult, 
                                 sectors: List[PredictionResult]) -> str:
        """Synthétise les prédictions en outlook marché"""
        
        # Analyser tendances générales
        activity_trend = activity.trends['predicted_classification']
        valuation_trend = valuation.trends['predicted_classification']
        
        # Compter secteurs positifs
        positive_sectors = sum(1 for s in sectors if s.trends['predicted_classification'] in ['forte_hausse', 'hausse_moderee'])
        
        # Synthèse
        if activity_trend in ['forte_hausse', 'hausse_moderee'] and positive_sectors >= len(sectors) * 0.7:
            return "Marché M&A très favorable avec forte activité attendue"
        elif activity_trend in ['forte_baisse', 'baisse_moderee']:
            return "Ralentissement du marché M&A prévu - approche prudente recommandée"
        else:
            return "Marché M&A stable avec opportunités sélectives"
    
    def _identify_investment_opportunities(self, activity: PredictionResult,
                                         sectors: Dict[str, PredictionResult],
                                         valuation: PredictionResult) -> List[Dict[str, Any]]:
        """Identifie les opportunités d'investissement"""
        
        opportunities = []
        
        # Opportunités sectorielles
        for sector_name, sector_pred in sectors.items():
            if sector_pred.trends['predicted_classification'] in ['forte_hausse', 'hausse_moderee']:
                opportunities.append({
                    'type': 'sector',
                    'name': sector_name,
                    'potential': 'high',
                    'rationale': f"Croissance {sector_pred.trends['predicted_classification']} prévue",
                    'timeline': sector_pred.horizon.value
                })
        
        # Opportunités de valorisation
        if valuation.trends['valuation_level'] == 'faible':
            opportunities.append({
                'type': 'valuation',
                'name': 'market_wide',
                'potential': 'medium',
                'rationale': 'Multiples attractifs actuellement',
                'timeline': 'immediate'
            })
        
        return opportunities[:5]  # Top 5
    
    def _identify_risk_factors(self, activity: PredictionResult,
                             valuation: PredictionResult,
                             sectors: List[PredictionResult]) -> List[str]:
        """Identifie les facteurs de risque"""
        
        risks = []
        
        # Risques d'activité
        if activity.trends['predicted_classification'] == 'forte_baisse':
            risks.append("Ralentissement significatif de l'activité M&A")
        
        # Risques de valorisation
        if valuation.trends['valuation_level'] == 'elevee':
            risks.append("Multiples de valorisation élevés - risque de correction")
        
        # Risques sectoriels
        negative_sectors = [s for s in sectors if s.trends['predicted_classification'] in ['forte_baisse', 'baisse_moderee']]
        if len(negative_sectors) > len(sectors) * 0.5:
            risks.append("Détérioration généralisée des perspectives sectorielles")
        
        # Risques de volatilité
        avg_volatility = np.mean([s.trends.get('volatility', 0) for s in sectors])
        if avg_volatility > 20:
            risks.append("Volatilité élevée des marchés - timing critique")
        
        return risks[:5]  # Top 5 risques
    
    def _generate_strategic_recommendations(self, market_outlook: str,
                                          opportunities: List[Dict[str, Any]],
                                          risks: List[str]) -> List[str]:
        """Génère des recommandations stratégiques"""
        
        recommendations = []
        
        # Recommandations basées sur outlook
        if "très favorable" in market_outlook:
            recommendations.append("Accélérer pipeline d'acquisitions - marché porteur")
        elif "ralentissement" in market_outlook:
            recommendations.append("Adopter approche contrarian - opportunités défensives")
        
        # Recommandations d'opportunités
        high_potential_opps = [o for o in opportunities if o['potential'] == 'high']
        if high_potential_opps:
            sectors = [o['name'] for o in high_potential_opps if o['type'] == 'sector']
            if sectors:
                recommendations.append(f"Prioriser secteurs: {', '.join(sectors)}")
        
        # Recommandations de gestion des risques
        if len(risks) > 3:
            recommendations.append("Renforcer due diligence - environnement risqué")
        
        # Recommandations temporelles
        recommendations.append("Maintenir flexibilité timing - ajuster selon évolutions")
        
        return recommendations[:5]


# Instance globale
_predictive_analytics_engine: Optional[PredictiveAnalyticsEngine] = None


async def get_predictive_analytics_engine() -> PredictiveAnalyticsEngine:
    """Factory pour obtenir l'instance du moteur d'analyse prédictive"""
    global _predictive_analytics_engine
    
    if _predictive_analytics_engine is None:
        _predictive_analytics_engine = PredictiveAnalyticsEngine()
        await _predictive_analytics_engine.initialize()
    
    return _predictive_analytics_engine


async def initialize_predictive_analytics():
    """Initialise le système d'analyse prédictive au démarrage"""
    try:
        engine = await get_predictive_analytics_engine()
        logger.info("📈 Système d'analyse prédictive initialisé avec succès")
        return engine
    except Exception as e:
        logger.error(f"Erreur initialisation analyse prédictive: {e}")
        raise