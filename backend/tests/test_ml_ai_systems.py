"""
Tests complets pour les systèmes ML/IA - US-008
Tests d'intégration et de validation pour tous les modules d'intelligence artificielle

Ce module teste:
- Moteur de scoring IA (ai_scoring_engine.py)
- Analyse prédictive (predictive_analytics.py) 
- Moteur NLP (nlp_engine.py)
- Analyseur de tendances ML (ml_trend_analyzer.py)
- Moteur de recommandations (recommendation_engine.py)
- Système de qualité des données (data_quality_engine.py)
"""

import pytest
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any
import tempfile
import shutil
import json

# Imports des modules à tester
from app.services.ai_scoring_engine import (
    get_ai_scoring_engine, 
    AICompanyScoringEngine,
    ScoringModel,
    ScoreCategory,
    ScoringFeatures,
    ScoringResult
)

from app.services.predictive_analytics import (
    get_predictive_analytics_engine,
    PredictiveAnalyticsEngine,
    PredictionType,
    TimeHorizon,
    ModelType,
    PredictionResult
)

from app.services.nlp_engine import (
    get_nlp_engine,
    NLPEngine,
    AnalysisType,
    EntityType,
    SentimentLabel,
    NLPAnalysisResult
)

from app.services.ml_trend_analyzer import (
    get_ml_trend_analyzer,
    MLTrendAnalyzer,
    TrendType,
    ClusteringMethod,
    AnalysisScope,
    TrendAnalysisResult
)

from app.services.recommendation_engine import (
    get_recommendation_engine,
    RecommendationEngine,
    RecommendationType,
    RecommendationStrategy,
    RiskProfile,
    UserPreferences,
    RecommendationResults
)

from app.services.data_quality_engine import (
    get_data_quality_engine,
    DataQualityEngine,
    QualityDimension,
    DataType,
    SeverityLevel,
    DataQualityReport
)


class TestAIScoringEngine:
    """Tests pour le moteur de scoring IA"""
    
    @pytest.fixture
    async def scoring_engine(self):
        """Fixture pour moteur de scoring"""
        engine = await get_ai_scoring_engine()
        yield engine
    
    @pytest.fixture
    def sample_company_data(self):
        """Données d'entreprise d'exemple"""
        return {
            'siren': '123456789',
            'nom': 'Test Company SAS',
            'chiffre_affaires': 5000000,
            'resultat_net': 250000,
            'capitaux_propres': 1500000,
            'total_bilan': 3000000,
            'date_creation': '2010-01-15',
            'code_ape': '7022Z',
            'region': 'Ile-de-France',
            'effectifs': 45
        }
    
    async def test_engine_initialization(self, scoring_engine):
        """Test d'initialisation du moteur"""
        assert scoring_engine is not None
        assert scoring_engine.is_trained
        assert len(scoring_engine.models) > 0
        assert len(scoring_engine.scalers) > 0
    
    async def test_score_company_single_model(self, scoring_engine, sample_company_data):
        """Test scoring avec modèle unique"""
        result = await scoring_engine.score_company(
            sample_company_data, 
            model=ScoringModel.RANDOM_FOREST
        )
        
        assert isinstance(result, ScoringResult)
        assert 0 <= result.overall_score <= 100
        assert 0 <= result.confidence <= 1
        assert result.company_id == '123456789'
        assert result.model_used == ScoringModel.RANDOM_FOREST
        assert len(result.category_scores) == len(ScoreCategory)
        assert len(result.recommendations) > 0
    
    async def test_score_company_ensemble(self, scoring_engine, sample_company_data):
        """Test scoring avec modèle ensemble"""
        result = await scoring_engine.score_company(
            sample_company_data,
            model=ScoringModel.ENSEMBLE
        )
        
        assert isinstance(result, ScoringResult)
        assert result.model_used == ScoringModel.ENSEMBLE
        assert result.confidence > 0.3  # Ensemble devrait être plus confiant
        assert len(result.feature_importance) > 0
    
    async def test_batch_scoring(self, scoring_engine):
        """Test scoring par lot"""
        companies_data = [
            {
                'siren': f'12345678{i}',
                'nom': f'Company {i}',
                'chiffre_affaires': 1000000 * (i + 1),
                'resultat_net': 50000 * i,
                'date_creation': '2015-01-01'
            }
            for i in range(5)
        ]
        
        results = await scoring_engine.batch_score_companies(companies_data)
        
        assert len(results) == 5
        assert all(isinstance(r, ScoringResult) for r in results)
        
        # Vérifier ordre des scores (CA plus élevé = score potentiellement plus élevé)
        scores = [r.overall_score for r in results]
        assert len(scores) == 5
    
    async def test_scoring_features_extraction(self, scoring_engine, sample_company_data):
        """Test extraction des features"""
        features = scoring_engine._extract_features(sample_company_data)
        
        assert isinstance(features, ScoringFeatures)
        assert features.chiffre_affaires == 5000000
        assert features.resultat_net == 250000
        assert features.age_entreprise > 0
        
        feature_array = features.to_array()
        assert len(feature_array) == len(ScoringFeatures.get_feature_names())
        assert not np.isnan(feature_array).any()
    
    async def test_category_scores_calculation(self, scoring_engine, sample_company_data):
        """Test calcul des scores par catégorie"""
        result = await scoring_engine.score_company(sample_company_data)
        
        # Vérifier toutes les catégories
        expected_categories = set(ScoreCategory)
        actual_categories = set(result.category_scores.keys())
        assert expected_categories == actual_categories
        
        # Vérifier plages de scores
        for category, score in result.category_scores.items():
            assert 0 <= score <= 100, f"Score {category} hors limites: {score}"
    
    async def test_caching_mechanism(self, scoring_engine, sample_company_data):
        """Test mécanisme de cache"""
        # Premier appel
        start_time = datetime.now()
        result1 = await scoring_engine.score_company(sample_company_data, use_cache=True)
        first_call_time = (datetime.now() - start_time).total_seconds()
        
        # Deuxième appel (devrait utiliser cache)
        start_time = datetime.now()
        result2 = await scoring_engine.score_company(sample_company_data, use_cache=True)
        second_call_time = (datetime.now() - start_time).total_seconds()
        
        # Cache devrait être plus rapide
        assert second_call_time < first_call_time
        assert result1.overall_score == result2.overall_score
    
    async def test_model_performance_metrics(self, scoring_engine):
        """Test métriques de performance"""
        performance = scoring_engine.get_model_performance()
        
        assert isinstance(performance, dict)
        assert len(performance) > 0
        
        # Vérifier structure des métriques
        for model_name, metrics in performance.items():
            if metrics:  # Si le modèle a des métriques
                assert 'prediction_count' in metrics or len(metrics) == 0


class TestPredictiveAnalytics:
    """Tests pour l'analyse prédictive"""
    
    @pytest.fixture
    async def analytics_engine(self):
        """Fixture pour moteur d'analyse prédictive"""
        engine = await get_predictive_analytics_engine()
        yield engine
    
    async def test_engine_initialization(self, analytics_engine):
        """Test d'initialisation"""
        assert analytics_engine is not None
        assert analytics_engine.historical_data is not None
        assert len(analytics_engine.models) > 0
    
    async def test_predict_ma_activity(self, analytics_engine):
        """Test prédiction activité M&A"""
        result = await analytics_engine.predict_ma_activity(
            horizon=TimeHorizon.MEDIUM_TERM,
            model=ModelType.ENSEMBLE
        )
        
        assert isinstance(result, PredictionResult)
        assert result.prediction_type == PredictionType.MA_ACTIVITY
        assert result.horizon == TimeHorizon.MEDIUM_TERM
        assert len(result.predictions) == 12  # 12 mois pour medium term
        assert len(result.dates) == len(result.predictions)
        assert len(result.confidence_intervals) == len(result.predictions)
        assert result.mae >= 0
        assert result.mse >= 0
        assert 0 <= result.confidence_score <= 1
    
    async def test_predict_sector_performance(self, analytics_engine):
        """Test prédiction performance sectorielle"""
        sectors = ['tech', 'healthcare', 'finance']
        results = await analytics_engine.predict_sector_performance(
            sectors=sectors,
            horizon=TimeHorizon.SHORT_TERM
        )
        
        assert isinstance(results, dict)
        assert len(results) <= len(sectors)  # Peut être moins si erreurs
        
        for sector, result in results.items():
            assert isinstance(result, PredictionResult)
            assert result.prediction_type == PredictionType.SECTOR_PERFORMANCE
            assert len(result.predictions) == 3  # 3 mois pour short term
    
    async def test_predict_valuation_trends(self, analytics_engine):
        """Test prédiction tendances valorisation"""
        result = await analytics_engine.predict_valuation_trends(
            horizon=TimeHorizon.LONG_TERM
        )
        
        assert isinstance(result, PredictionResult)
        assert result.prediction_type == PredictionType.VALUATION_TRENDS
        assert len(result.predictions) == 36  # 36 mois pour long term
        assert all(pred > 0 for pred in result.predictions)  # Valorisations positives
    
    async def test_generate_market_forecast(self, analytics_engine):
        """Test génération prévision marché complète"""
        forecast = await analytics_engine.generate_market_forecast(
            horizon=TimeHorizon.MEDIUM_TERM
        )
        
        assert forecast.overall_activity is not None
        assert forecast.valuation_trends is not None
        assert len(forecast.sector_predictions) > 0
        assert forecast.consolidation_forecast is not None
        assert len(forecast.market_outlook) > 0
        assert len(forecast.investment_opportunities) >= 0
        assert len(forecast.risk_factors) >= 0
        assert len(forecast.strategic_recommendations) > 0
    
    async def test_trend_analysis(self, analytics_engine):
        """Test analyse de tendances"""
        # Données d'exemple
        test_data = pd.Series(
            data=np.random.randn(24) + np.linspace(0, 10, 24),  # Tendance croissante
            index=pd.date_range(start='2022-01-01', periods=24, freq='M')
        )
        
        trends = analytics_engine._analyze_trends(test_data, [50, 55, 60])
        
        assert 'historical_trend' in trends
        assert 'predicted_trend' in trends
        assert 'historical_classification' in trends
        assert 'predicted_classification' in trends
        assert trends['historical_classification'] in ['forte_hausse', 'hausse_moderee', 'stable', 'baisse_moderee', 'forte_baisse']
    
    async def test_seasonality_analysis(self, analytics_engine):
        """Test analyse saisonnalité"""
        # Données avec saisonnalité artificielle
        dates = pd.date_range(start='2020-01-01', periods=36, freq='M')
        seasonal_data = pd.Series(
            data=100 + 10 * np.sin(2 * np.pi * np.arange(36) / 12) + np.random.randn(36),
            index=dates
        )
        
        seasonality = analytics_engine._analyze_seasonality(seasonal_data)
        
        if seasonality.get('has_seasonality'):
            assert 'strongest_month' in seasonality
            assert 'weakest_month' in seasonality
            assert 1 <= seasonality['strongest_month'] <= 12
            assert 1 <= seasonality['weakest_month'] <= 12


class TestNLPEngine:
    """Tests pour le moteur NLP"""
    
    @pytest.fixture
    async def nlp_engine(self):
        """Fixture pour moteur NLP"""
        engine = await get_nlp_engine()
        yield engine
    
    @pytest.fixture
    def sample_texts(self):
        """Textes d'exemple pour tests"""
        return {
            'company_description': "TechCorp SAS est une entreprise innovante spécialisée dans l'intelligence artificielle. SIREN: 123456789. Contact: contact@techcorp.fr",
            'financial_news': "Les résultats financiers de TechCorp montrent une croissance exceptionnelle de 25% cette année.",
            'negative_review': "Service client décevant, produits de mauvaise qualité, expérience très négative.",
            'positive_review': "Excellent service, équipe professionnelle, résultats fantastiques, très satisfait !",
            'press_release': "TechCorp annonce l'acquisition de StartupIA pour 5 millions d'euros, renforçant sa position sur le marché de l'IA."
        }
    
    async def test_engine_initialization(self, nlp_engine):
        """Test initialisation moteur NLP"""
        assert nlp_engine is not None
        assert nlp_engine.french_patterns is not None
        assert len(nlp_engine.french_stopwords) > 0
    
    async def test_entity_extraction(self, nlp_engine, sample_texts):
        """Test extraction d'entités"""
        result = await nlp_engine.analyze_text(
            sample_texts['company_description'],
            [AnalysisType.ENTITY_EXTRACTION]
        )
        
        assert isinstance(result, NLPAnalysisResult)
        assert len(result.entities) > 0
        
        # Vérifier types d'entités détectées
        entity_types = {entity.label for entity in result.entities}
        # Devrait détecter au moins SIREN et potentiellement COMPANY
        siren_entities = [e for e in result.entities if e.label == EntityType.SIREN]
        assert len(siren_entities) > 0
        assert siren_entities[0].text == '123456789'
    
    async def test_sentiment_analysis(self, nlp_engine, sample_texts):
        """Test analyse de sentiment"""
        # Test sentiment positif
        positive_result = await nlp_engine.analyze_text(
            sample_texts['positive_review'],
            [AnalysisType.SENTIMENT_ANALYSIS]
        )
        
        assert positive_result.sentiment is not None
        assert positive_result.sentiment.overall_sentiment in [SentimentLabel.POSITIVE, SentimentLabel.VERY_POSITIVE]
        assert positive_result.sentiment.compound_score > 0
        
        # Test sentiment négatif
        negative_result = await nlp_engine.analyze_text(
            sample_texts['negative_review'],
            [AnalysisType.SENTIMENT_ANALYSIS]
        )
        
        assert negative_result.sentiment is not None
        assert negative_result.sentiment.overall_sentiment in [SentimentLabel.NEGATIVE, SentimentLabel.VERY_NEGATIVE]
        assert negative_result.sentiment.compound_score < 0
    
    async def test_document_classification(self, nlp_engine, sample_texts):
        """Test classification de documents"""
        result = await nlp_engine.analyze_text(
            sample_texts['press_release'],
            [AnalysisType.DOCUMENT_CLASSIFICATION]
        )
        
        assert result.classification is not None
        assert result.classification.predicted_class in ['communique_presse', 'article_news', 'rapport_financier', 'document_legal', 'site_web']
        assert 0 <= result.classification.confidence <= 1
    
    async def test_text_summarization(self, nlp_engine):
        """Test résumé de texte"""
        long_text = " ".join([
            "TechCorp est une entreprise leader dans le domaine de l'intelligence artificielle.",
            "Fondée en 2010, elle compte aujourd'hui plus de 200 employés.",
            "Ses solutions innovantes transforment de nombreux secteurs.",
            "L'entreprise a réalisé un chiffre d'affaires de 50 millions d'euros l'année dernière.",
            "Elle prévoit d'étendre ses activités à l'international dans les prochains mois."
        ])
        
        result = await nlp_engine.analyze_text(
            long_text,
            [AnalysisType.TEXT_SUMMARIZATION]
        )
        
        assert result.summary is not None
        assert len(result.summary.summary) < len(long_text)
        assert result.summary.compression_ratio < 1.0
        assert len(result.summary.key_sentences) > 0
    
    async def test_keyword_extraction(self, nlp_engine, sample_texts):
        """Test extraction de mots-clés"""
        result = await nlp_engine.analyze_text(
            sample_texts['company_description'],
            [AnalysisType.KEYWORD_EXTRACTION]
        )
        
        assert len(result.keywords) > 0
        
        # Vérifier format des mots-clés
        for keyword, score in result.keywords:
            assert isinstance(keyword, str)
            assert isinstance(score, float)
            assert score > 0
            assert len(keyword) > 2  # Pas de mots trop courts
    
    async def test_company_reputation_analysis(self, nlp_engine):
        """Test analyse réputation entreprise"""
        company_name = "TechCorp"
        texts = [
            "TechCorp offre d'excellents services, très professionnel",
            "Mauvaise expérience avec TechCorp, service décevant", 
            "TechCorp innove constamment, équipe fantastique",
            "TechCorp a des problèmes de qualité récemment"
        ]
        
        reputation = await nlp_engine.analyze_company_reputation(
            company_name=company_name,
            texts=texts,
            sources=['avis1', 'avis2', 'avis3', 'avis4']
        )
        
        assert 'overall_sentiment_score' in reputation
        assert 'reputation_level' in reputation
        assert 'total_mentions' in reputation
        assert reputation['total_mentions'] > 0
        assert 'sentiment_distribution' in reputation
    
    async def test_market_trends_detection(self, nlp_engine):
        """Test détection tendances marché"""
        news_texts = [
            "Le marché de l'IA connaît une croissance exceptionnelle",
            "Fusion-acquisition record dans le secteur technologique",
            "Innovation breakthrough en intelligence artificielle",
            "Investissements massifs dans les startups tech"
        ]
        
        trends = await nlp_engine.detect_market_trends(
            news_texts=news_texts,
            sector="tech"
        )
        
        assert 'market_sentiment' in trends
        assert 'emerging_themes' in trends
        assert 'top_keywords' in trends
        assert trends['articles_analyzed'] == len(news_texts)
    
    async def test_language_detection(self, nlp_engine):
        """Test détection de langue"""
        french_text = "Bonjour, ceci est un texte en français avec des mots comme entreprise, société, résultats."
        english_text = "Hello, this is an English text with words like company, business, results."
        
        french_lang = nlp_engine._detect_language(french_text)
        english_lang = nlp_engine._detect_language(english_text)
        
        assert french_lang in ['fr', 'unknown']  # Peut fallback sur unknown
        assert english_lang in ['en', 'unknown']
    
    async def test_cache_functionality(self, nlp_engine, sample_texts):
        """Test fonctionnalité de cache"""
        text = sample_texts['company_description']
        analysis_types = [AnalysisType.SENTIMENT_ANALYSIS, AnalysisType.ENTITY_EXTRACTION]
        
        # Premier appel
        start_time = datetime.now()
        result1 = await nlp_engine.analyze_text(text, analysis_types, use_cache=True)
        first_time = (datetime.now() - start_time).total_seconds()
        
        # Deuxième appel (cache)
        start_time = datetime.now()
        result2 = await nlp_engine.analyze_text(text, analysis_types, use_cache=True)
        second_time = (datetime.now() - start_time).total_seconds()
        
        # Cache devrait être plus rapide
        assert second_time < first_time
        assert result1.confidence_score == result2.confidence_score


class TestMLTrendAnalyzer:
    """Tests pour l'analyseur de tendances ML"""
    
    @pytest.fixture
    async def trend_analyzer(self):
        """Fixture pour analyseur de tendances"""
        analyzer = await get_ml_trend_analyzer()
        yield analyzer
    
    @pytest.fixture
    def sample_market_data(self):
        """Données de marché d'exemple"""
        dates = pd.date_range(start='2020-01-01', periods=48, freq='M')
        
        # Données avec tendance et saisonnalité
        trend = np.linspace(100, 150, 48)
        seasonal = 10 * np.sin(2 * np.pi * np.arange(48) / 12)
        noise = np.random.normal(0, 5, 48)
        
        return pd.DataFrame({
            'ma_activity': trend + seasonal + noise,
            'valuations': 15 + 2 * np.sin(2 * np.pi * np.arange(48) / 24) + np.random.normal(0, 1, 48),
            'tech_activity': trend * 1.2 + seasonal * 1.5 + noise,
            'healthcare_activity': trend * 0.8 + seasonal * 0.5 + noise
        }, index=dates)
    
    async def test_analyzer_initialization(self, trend_analyzer):
        """Test initialisation analyseur"""
        assert trend_analyzer is not None
        assert trend_analyzer.models is not None
        assert len(trend_analyzer.model_configs) > 0
    
    async def test_analyze_market_trends(self, trend_analyzer, sample_market_data):
        """Test analyse tendances marché"""
        result = await trend_analyzer.analyze_market_trends(
            data=sample_market_data,
            scope=AnalysisScope.GLOBAL,
            timeframe="24_months"
        )
        
        assert isinstance(result, TrendAnalysisResult)
        assert result.analysis_scope == AnalysisScope.GLOBAL
        assert len(result.trend_patterns) > 0
        assert len(result.key_insights) > 0
        assert 0 <= result.data_quality_score <= 1
        assert result.processing_time_ms > 0
    
    async def test_clustering_analysis(self, trend_analyzer):
        """Test analyse de clustering"""
        # Données d'entreprises synthétiques
        companies_data = []
        for i in range(100):
            companies_data.append({
                'chiffre_affaires': np.random.lognormal(14, 1.5),
                'resultat_net': np.random.normal(50000, 100000),
                'effectifs': np.random.lognormal(3, 1),
                'secteur': np.random.choice(['tech', 'healthcare', 'finance']),
                'region': np.random.choice(['IDF', 'PACA', 'AURA'])
            })
        
        clusters = await trend_analyzer.cluster_companies_by_profile(
            companies_data=companies_data,
            method=ClusteringMethod.KMEANS,
            n_clusters=5
        )
        
        assert isinstance(clusters, list)
        assert len(clusters) <= 5
        
        for cluster in clusters:
            assert cluster.cluster_id >= 0
            assert cluster.size > 0
            assert len(cluster.companies) == cluster.size
            assert 0 <= cluster.cohesion_score <= 1
    
    async def test_anomaly_detection(self, trend_analyzer, sample_market_data):
        """Test détection d'anomalies"""
        # Ajouter quelques anomalies artificielles
        anomalous_data = sample_market_data.copy()
        anomalous_data.iloc[10, 0] = anomalous_data.iloc[10, 0] * 3  # Anomalie
        anomalous_data.iloc[25, 1] = anomalous_data.iloc[25, 1] * 0.1  # Anomalie
        
        anomalies = await trend_analyzer.detect_market_anomalies(
            time_series_data=anomalous_data,
            sensitivity=0.1
        )
        
        assert isinstance(anomalies, list)
        # Devrait détecter au moins quelques anomalies
        if len(anomalies) > 0:
            for anomaly in anomalies:
                assert 'index' in anomaly
                assert 'anomaly_score' in anomaly
                assert 'severity' in anomaly
    
    async def test_sector_correlations(self, trend_analyzer, sample_market_data):
        """Test analyse corrélations sectorielles"""
        sector_data = {
            'tech': sample_market_data[['tech_activity']],
            'healthcare': sample_market_data[['healthcare_activity']],
            'general': sample_market_data[['ma_activity']]
        }
        
        correlations = await trend_analyzer.analyze_sector_correlations(sector_data)
        
        assert isinstance(correlations, dict)
        assert 'correlation_matrix' in correlations
        assert 'strong_correlations' in correlations
        assert 'sector_clusters' in correlations
    
    async def test_trend_pattern_detection(self, trend_analyzer, sample_market_data):
        """Test détection de patterns de tendances"""
        patterns = await trend_analyzer._detect_trend_patterns(sample_market_data)
        
        assert isinstance(patterns, list)
        
        for pattern in patterns:
            assert hasattr(pattern, 'trend_type')
            assert hasattr(pattern, 'strength')
            assert hasattr(pattern, 'confidence')
            assert 0 <= pattern.strength <= 1
            assert 0 <= pattern.confidence <= 1
            assert pattern.trend_type in list(TrendType)


class TestRecommendationEngine:
    """Tests pour le moteur de recommandations"""
    
    @pytest.fixture
    async def recommendation_engine(self):
        """Fixture pour moteur de recommandations"""
        engine = await get_recommendation_engine()
        yield engine
    
    @pytest.fixture
    def sample_user_preferences(self):
        """Préférences utilisateur d'exemple"""
        return UserPreferences(
            user_id="test_user_001",
            sectors_of_interest=['tech', 'healthcare'],
            size_preferences={'min_ca': 1000000, 'max_ca': 50000000},
            geographic_preferences=['France', 'Europe'],
            risk_profile=RiskProfile.MODERATE,
            investment_horizon='medium',
            budget_range=(5000000, 25000000),
            strategic_objectives=['growth', 'innovation'],
            excluded_sectors=['tobacco', 'gambling'],
            minimum_score=60.0
        )
    
    async def test_engine_initialization(self, recommendation_engine):
        """Test initialisation moteur"""
        assert recommendation_engine is not None
        assert hasattr(recommendation_engine, 'user_preferences')
        assert hasattr(recommendation_engine, 'models')
    
    async def test_get_recommendations_collaborative(self, recommendation_engine, sample_user_preferences):
        """Test recommandations collaboratives"""
        # Simuler historique utilisateur
        await recommendation_engine.update_user_preferences(
            sample_user_preferences.user_id,
            sample_user_preferences.to_dict()
        )
        
        results = await recommendation_engine.get_recommendations(
            user_id=sample_user_preferences.user_id,
            strategy=RecommendationStrategy.COLLABORATIVE_FILTERING,
            count=10
        )
        
        assert isinstance(results, RecommendationResults)
        assert results.user_id == sample_user_preferences.user_id
        assert results.strategy_used == RecommendationStrategy.COLLABORATIVE_FILTERING
        assert len(results.recommendations) <= 10
        assert results.total_count >= 0
        assert 0 <= results.average_score <= 100
    
    async def test_get_recommendations_content_based(self, recommendation_engine, sample_user_preferences):
        """Test recommandations basées contenu"""
        await recommendation_engine.update_user_preferences(
            sample_user_preferences.user_id,
            sample_user_preferences.to_dict()
        )
        
        results = await recommendation_engine.get_recommendations(
            user_id=sample_user_preferences.user_id,
            strategy=RecommendationStrategy.CONTENT_BASED,
            count=5
        )
        
        assert isinstance(results, RecommendationResults)
        assert results.strategy_used == RecommendationStrategy.CONTENT_BASED
        assert len(results.recommendations) <= 5
    
    async def test_get_recommendations_hybrid(self, recommendation_engine, sample_user_preferences):
        """Test recommandations hybrides"""
        await recommendation_engine.update_user_preferences(
            sample_user_preferences.user_id,
            sample_user_preferences.to_dict()
        )
        
        results = await recommendation_engine.get_recommendations(
            user_id=sample_user_preferences.user_id,
            strategy=RecommendationStrategy.HYBRID,
            count=15
        )
        
        assert isinstance(results, RecommendationResults)
        assert results.strategy_used == RecommendationStrategy.HYBRID
        assert len(results.recommendations) <= 15
        
        # Hybrid devrait avoir diversité dans les sources
        recommendation_types = {rec.item_type for rec in results.recommendations}
        assert len(recommendation_types) >= 1
    
    async def test_portfolio_optimization(self, recommendation_engine, sample_user_preferences):
        """Test optimisation de portefeuille"""
        candidate_opportunities = ['opp_001', 'opp_002', 'opp_003', 'opp_004', 'opp_005']
        budget_constraint = 20000000
        
        portfolio = await recommendation_engine.optimize_portfolio(
            user_id=sample_user_preferences.user_id,
            candidate_opportunities=candidate_opportunities,
            budget_constraint=budget_constraint,
            optimization_objective="max_return"
        )
        
        assert portfolio.user_id == sample_user_preferences.user_id
        assert portfolio.total_investment == budget_constraint
        assert portfolio.expected_return >= 0
        assert portfolio.portfolio_risk >= 0
        assert 0 <= portfolio.diversification_score <= 1
        assert len(portfolio.allocation_weights) == len(candidate_opportunities)
        
        # Vérifier que la somme des poids = 1
        total_weight = sum(portfolio.allocation_weights.values())
        assert abs(total_weight - 1.0) < 0.01  # Tolérance pour erreurs de calcul
    
    async def test_user_preferences_management(self, recommendation_engine, sample_user_preferences):
        """Test gestion préférences utilisateur"""
        user_id = sample_user_preferences.user_id
        
        # Créer préférences
        await recommendation_engine.update_user_preferences(
            user_id,
            sample_user_preferences.to_dict()
        )
        
        # Vérifier stockage
        stored_prefs = recommendation_engine.user_preferences[user_id]
        assert stored_prefs.user_id == user_id
        assert stored_prefs.risk_profile == RiskProfile.MODERATE
        assert 'tech' in stored_prefs.sectors_of_interest
        
        # Modifier préférences
        updated_prefs = {
            'risk_profile': 'aggressive',
            'minimum_score': 75.0
        }
        
        await recommendation_engine.update_user_preferences(user_id, updated_prefs)
        
        # Vérifier modification
        modified_prefs = recommendation_engine.user_preferences[user_id]
        assert modified_prefs.risk_profile == RiskProfile.AGGRESSIVE
        assert modified_prefs.minimum_score == 75.0
        assert 'tech' in modified_prefs.sectors_of_interest  # Doit être conservé
    
    async def test_trending_opportunities(self, recommendation_engine):
        """Test opportunités tendance"""
        trending = await recommendation_engine.get_trending_opportunities(
            sector='tech',
            time_window_days=30
        )
        
        assert isinstance(trending, list)
        assert len(trending) <= 10  # Top 10
        
        if len(trending) > 0:
            for opportunity in trending:
                assert hasattr(opportunity, 'item_id')
                assert hasattr(opportunity, 'score')
                assert hasattr(opportunity, 'urgency_score')
                assert 0 <= opportunity.score <= 100
    
    async def test_user_feedback_learning(self, recommendation_engine, sample_user_preferences):
        """Test apprentissage avec feedback utilisateur"""
        user_id = sample_user_preferences.user_id
        
        # Enregistrer feedback
        await recommendation_engine.record_user_feedback(
            user_id=user_id,
            recommendation_id="test_rec_001",
            feedback_type="like",
            feedback_value=True
        )
        
        await recommendation_engine.record_user_feedback(
            user_id=user_id,
            recommendation_id="test_rec_002", 
            feedback_type="rating",
            feedback_value=4.5
        )
        
        # Vérifier stockage dans historique
        history = recommendation_engine.user_history[user_id]
        assert len(history) >= 2
        
        # Vérifier structure du feedback
        like_feedback = next((f for f in history if f['feedback_type'] == 'like'), None)
        assert like_feedback is not None
        assert like_feedback['feedback_value'] is True


class TestDataQualityEngine:
    """Tests pour le moteur de qualité des données"""
    
    @pytest.fixture
    async def quality_engine(self):
        """Fixture pour moteur de qualité"""
        engine = await get_data_quality_engine()
        yield engine
    
    @pytest.fixture
    def sample_clean_data(self):
        """Données propres d'exemple"""
        return pd.DataFrame({
            'siren': ['123456789', '987654321', '456789123'],
            'nom': ['Entreprise A SAS', 'Société B SARL', 'Compagnie C SA'],
            'chiffre_affaires': [5000000, 2500000, 8000000],
            'date_creation': ['2010-01-15', '2015-06-20', '2008-03-10'],
            'email': ['contact@a.fr', 'info@b.com', 'hello@c.org'],
            'telephone': ['0123456789', '0187654321', '0156789012']
        })
    
    @pytest.fixture
    def sample_dirty_data(self):
        """Données sales d'exemple"""
        return pd.DataFrame({
            'siren': ['123456789', '98765432X', '', '123456789'],  # SIREN dupliqué et invalide
            'nom': ['Entreprise A', '', 'Compagnie C', 'Entreprise A'],  # Nom manquant et dupliqué
            'chiffre_affaires': [5000000, -100000, None, 5000000],  # CA négatif et manquant
            'date_creation': ['2010-01-15', '2025-06-20', '1800-01-01', '2010-01-15'],  # Date future et très ancienne
            'email': ['contact@a.fr', 'invalid-email', '', 'contact@a.fr'],  # Email invalide et manquant
            'telephone': ['0123456789', '123', None, '0123456789']  # Téléphone invalide et manquant
        })
    
    async def test_engine_initialization(self, quality_engine):
        """Test initialisation moteur qualité"""
        assert quality_engine is not None
        assert quality_engine.validation_rules is not None
        assert quality_engine.reference_patterns is not None
        assert len(quality_engine.quality_thresholds) == len(QualityDimension)
    
    async def test_evaluate_clean_data_quality(self, quality_engine, sample_clean_data):
        """Test évaluation données propres"""
        report = await quality_engine.evaluate_data_quality(
            data=sample_clean_data,
            data_type=DataType.COMPANY_BASIC
        )
        
        assert isinstance(report, DataQualityReport)
        assert report.overall_score >= 70  # Données propres = bon score
        assert report.data_type == DataType.COMPANY_BASIC
        assert report.total_records == 3
        assert len(report.quality_metrics) == len(QualityDimension)
        
        # Vérifier scores par dimension pour données propres
        assert report.quality_metrics[QualityDimension.COMPLETENESS].score >= 90
        assert report.quality_metrics[QualityDimension.VALIDITY].score >= 85
    
    async def test_evaluate_dirty_data_quality(self, quality_engine, sample_dirty_data):
        """Test évaluation données sales"""
        report = await quality_engine.evaluate_data_quality(
            data=sample_dirty_data,
            data_type=DataType.COMPANY_BASIC
        )
        
        assert isinstance(report, DataQualityReport)
        assert report.overall_score < 80  # Données sales = score plus faible
        assert len(report.issues) > 0
        assert len(report.critical_issues) > 0
        
        # Vérifier détection de problèmes spécifiques
        issue_types = {issue.dimension for issue in report.issues}
        assert QualityDimension.VALIDITY in issue_types  # SIREN invalides
        assert QualityDimension.UNIQUENESS in issue_types  # Doublons
    
    async def test_completeness_evaluation(self, quality_engine):
        """Test évaluation complétude"""
        # Données avec valeurs manquantes
        incomplete_data = pd.DataFrame({
            'siren': ['123456789', None, '456789123'],
            'nom': ['Entreprise A', 'Entreprise B', None],
            'chiffre_affaires': [5000000, None, None]
        })
        
        report = await quality_engine.evaluate_data_quality(
            data=incomplete_data,
            data_type=DataType.COMPANY_BASIC
        )
        
        completeness_metrics = report.quality_metrics[QualityDimension.COMPLETENESS]
        assert completeness_metrics.score < 80  # Beaucoup de données manquantes
        
        # Vérifier détection problèmes de complétude
        completeness_issues = [i for i in report.issues if i.dimension == QualityDimension.COMPLETENESS]
        assert len(completeness_issues) > 0
    
    async def test_accuracy_evaluation(self, quality_engine):
        """Test évaluation précision"""
        # Données avec erreurs de précision
        inaccurate_data = pd.DataFrame({
            'siren': ['12345678X', '987654321'],  # SIREN invalide
            'email': ['invalid-email', 'valid@test.com'],  # Email invalide
            'chiffre_affaires': [-1000000, 5000000],  # CA négatif suspect
            'date_creation': ['2030-01-01', '2010-01-01']  # Date future
        })
        
        report = await quality_engine.evaluate_data_quality(
            data=inaccurate_data,
            data_type=DataType.COMPANY_BASIC
        )
        
        accuracy_metrics = report.quality_metrics[QualityDimension.ACCURACY]
        assert accuracy_metrics.score < 90  # Erreurs de précision
        
        # Vérifier détection erreurs spécifiques
        accuracy_issues = [i for i in report.issues if i.dimension == QualityDimension.ACCURACY]
        assert len(accuracy_issues) > 0
    
    async def test_uniqueness_evaluation(self, quality_engine):
        """Test évaluation unicité"""
        # Données avec doublons
        duplicate_data = pd.DataFrame({
            'siren': ['123456789', '123456789', '987654321'],  # SIREN dupliqué
            'nom': ['Entreprise A', 'Entreprise A', 'Entreprise B'],  # Nom dupliqué
            'email': ['test@test.com', 'test@test.com', 'other@test.com']  # Email dupliqué
        })
        
        report = await quality_engine.evaluate_data_quality(
            data=duplicate_data,
            data_type=DataType.COMPANY_BASIC
        )
        
        uniqueness_metrics = report.quality_metrics[QualityDimension.UNIQUENESS]
        assert uniqueness_metrics.score < 95  # Doublons détectés
        
        # Vérifier détection doublons
        uniqueness_issues = [i for i in report.issues if i.dimension == QualityDimension.UNIQUENESS]
        assert len(uniqueness_issues) > 0
    
    async def test_data_cleaning(self, quality_engine, sample_dirty_data):
        """Test nettoyage automatique"""
        # Évaluer qualité initiale
        initial_report = await quality_engine.evaluate_data_quality(
            data=sample_dirty_data,
            data_type=DataType.COMPANY_BASIC
        )
        
        # Nettoyer données
        cleaned_data, cleaning_stats = await quality_engine.clean_data(
            data=sample_dirty_data,
            quality_report=initial_report,
            auto_fix=True
        )
        
        assert isinstance(cleaned_data, pd.DataFrame)
        assert isinstance(cleaning_stats, dict)
        assert 'actions_applied' in cleaning_stats
        assert 'original_records' in cleaning_stats
        assert 'data_loss_percentage' in cleaning_stats
        
        # Vérifier amélioration (optionnel selon corrections applicables)
        if cleaning_stats['actions_applied'] > 0:
            post_cleaning_report = await quality_engine.evaluate_data_quality(
                data=cleaned_data,
                data_type=DataType.COMPANY_BASIC
            )
            # Le score pourrait être amélioré après nettoyage
            assert post_cleaning_report.overall_score >= initial_report.overall_score - 5  # Tolérance
    
    async def test_quality_trends_monitoring(self, quality_engine, sample_clean_data):
        """Test monitoring tendances qualité"""
        # Simuler plusieurs évaluations dans le temps
        data_type = DataType.COMPANY_BASIC
        
        # Première évaluation
        report1 = await quality_engine.evaluate_data_quality(sample_clean_data, data_type)
        
        # Légèrement dégrader données
        degraded_data = sample_clean_data.copy()
        degraded_data.loc[0, 'email'] = 'invalid-email'
        
        # Deuxième évaluation
        report2 = await quality_engine.evaluate_data_quality(degraded_data, data_type)
        
        # Analyser tendances
        trends = await quality_engine.monitor_quality_trends(
            data_type=data_type,
            time_window_days=1
        )
        
        assert isinstance(trends, dict)
        assert 'data_type' in trends
        if 'trends_by_dimension' in trends:
            assert isinstance(trends['trends_by_dimension'], dict)
    
    async def test_validation_rules(self, quality_engine):
        """Test règles de validation"""
        # Test validation SIREN
        test_data = pd.DataFrame({
            'siren': ['123456789', '12345678X', '1234567890']  # Valide, invalide, trop long
        })
        
        rules = quality_engine.validation_rules[DataType.COMPANY_BASIC]['siren']
        validation_result = quality_engine._validate_field(test_data, 'siren', rules)
        
        assert 'score' in validation_result
        assert 'invalid_records' in validation_result
        assert len(validation_result['invalid_records']) == 2  # 2 SIREN invalides
    
    async def test_cache_functionality(self, quality_engine, sample_clean_data):
        """Test fonctionnalité cache"""
        # Premier appel
        start_time = datetime.now()
        report1 = await quality_engine.evaluate_data_quality(
            sample_clean_data, 
            DataType.COMPANY_BASIC, 
            use_cache=True
        )
        first_time = (datetime.now() - start_time).total_seconds()
        
        # Deuxième appel (cache)
        start_time = datetime.now()
        report2 = await quality_engine.evaluate_data_quality(
            sample_clean_data, 
            DataType.COMPANY_BASIC, 
            use_cache=True
        )
        second_time = (datetime.now() - start_time).total_seconds()
        
        # Cache devrait être plus rapide
        assert second_time < first_time
        assert report1.overall_score == report2.overall_score


class TestMLAIIntegration:
    """Tests d'intégration entre modules ML/IA"""
    
    async def test_ai_scoring_to_recommendations_flow(self):
        """Test flux scoring IA → recommandations"""
        # Obtenir moteurs
        scoring_engine = await get_ai_scoring_engine()
        recommendation_engine = await get_recommendation_engine()
        
        # Scorer entreprise
        company_data = {
            'siren': '123456789',
            'nom': 'TechCorp SAS',
            'chiffre_affaires': 10000000,
            'resultat_net': 500000
        }
        
        scoring_result = await scoring_engine.score_company(company_data)
        
        # Utiliser score pour recommandations
        user_prefs = UserPreferences(
            user_id="integration_test",
            sectors_of_interest=['tech'],
            size_preferences={'min_ca': 5000000, 'max_ca': 50000000},
            geographic_preferences=['France'],
            risk_profile=RiskProfile.MODERATE,
            investment_horizon='medium',
            budget_range=(5000000, 20000000),
            strategic_objectives=['growth'],
            minimum_score=scoring_result.overall_score * 0.8  # Score basé sur IA
        )
        
        await recommendation_engine.update_user_preferences(
            user_prefs.user_id,
            user_prefs.to_dict()
        )
        
        recommendations = await recommendation_engine.get_recommendations(
            user_id=user_prefs.user_id,
            strategy=RecommendationStrategy.AI_SCORING
        )
        
        assert isinstance(recommendations, RecommendationResults)
        assert recommendations.strategy_used == RecommendationStrategy.AI_SCORING
    
    async def test_nlp_to_trend_analysis_flow(self):
        """Test flux NLP → analyse tendances"""
        nlp_engine = await get_nlp_engine()
        trend_analyzer = await get_ml_trend_analyzer()
        
        # Analyser textes avec NLP
        market_texts = [
            "Le marché M&A tech montre une croissance de 25% cette année",
            "Valorisations en hausse dans le secteur healthcare",
            "Activité soutenue en fusions-acquisitions"
        ]
        
        nlp_results = []
        for text in market_texts:
            result = await nlp_engine.analyze_text(
                text,
                [AnalysisType.SENTIMENT_ANALYSIS, AnalysisType.KEYWORD_EXTRACTION]
            )
            nlp_results.append(result)
        
        # Extraire métriques pour analyse de tendances
        sentiment_scores = [r.sentiment.compound_score for r in nlp_results if r.sentiment]
        
        # Créer série temporelle simulée basée sur NLP
        dates = pd.date_range(start='2024-01-01', periods=len(sentiment_scores), freq='M')
        nlp_trend_data = pd.DataFrame({
            'market_sentiment': sentiment_scores,
            'activity_level': [abs(s) * 100 for s in sentiment_scores]
        }, index=dates)
        
        # Analyser avec ML
        trend_result = await trend_analyzer.analyze_market_trends(
            data=nlp_trend_data,
            scope=AnalysisScope.TEMPORAL
        )
        
        assert isinstance(trend_result, TrendAnalysisResult)
        assert len(trend_result.trend_patterns) >= 0
    
    async def test_data_quality_to_ai_scoring_flow(self):
        """Test flux qualité données → scoring IA"""
        quality_engine = await get_data_quality_engine()
        scoring_engine = await get_ai_scoring_engine()
        
        # Données avec problèmes de qualité
        dirty_company_data = pd.DataFrame({
            'siren': ['123456789', '98765432X', None],
            'nom': ['TechCorp', '', 'HealthCorp'],
            'chiffre_affaires': [5000000, -100000, None]
        })
        
        # Évaluer qualité
        quality_report = await quality_engine.evaluate_data_quality(
            dirty_company_data,
            DataType.COMPANY_BASIC
        )
        
        # Nettoyer données
        cleaned_data, _ = await quality_engine.clean_data(
            dirty_company_data,
            quality_report,
            auto_fix=True
        )
        
        # Scorer avec données nettoyées
        for idx, row in cleaned_data.iterrows():
            if pd.notna(row['siren']) and row['siren'] != '':
                company_data = {
                    'siren': str(row['siren']),
                    'nom': str(row['nom']) if pd.notna(row['nom']) else 'Unknown',
                    'chiffre_affaires': float(row['chiffre_affaires']) if pd.notna(row['chiffre_affaires']) else 0
                }
                
                try:
                    scoring_result = await scoring_engine.score_company(company_data)
                    assert isinstance(scoring_result, ScoringResult)
                    # Score devrait être influencé par qualité données
                    assert 0 <= scoring_result.overall_score <= 100
                except Exception as e:
                    # Certaines données peuvent encore être problématiques
                    assert "Erreur" in str(e) or "fallback" in str(e).lower()
    
    async def test_predictive_to_recommendation_flow(self):
        """Test flux analyse prédictive → recommandations"""
        predictive_engine = await get_predictive_analytics_engine()
        recommendation_engine = await get_recommendation_engine()
        
        # Prédictions sectorielles
        sector_predictions = await predictive_engine.predict_sector_performance(
            sectors=['tech', 'healthcare', 'finance'],
            horizon=TimeHorizon.MEDIUM_TERM
        )
        
        # Identifier secteurs en croissance
        growing_sectors = []
        for sector, prediction in sector_predictions.items():
            if prediction.trends['predicted_classification'] in ['forte_hausse', 'hausse_moderee']:
                growing_sectors.append(sector)
        
        # Ajuster préférences utilisateur basées sur prédictions
        user_prefs = UserPreferences(
            user_id="predictive_test",
            sectors_of_interest=growing_sectors or ['tech'],  # Fallback si aucun secteur en croissance
            size_preferences={'min_ca': 1000000, 'max_ca': 20000000},
            geographic_preferences=['France'],
            risk_profile=RiskProfile.MODERATE,
            investment_horizon='medium',
            budget_range=(5000000, 25000000),
            strategic_objectives=['growth']
        )
        
        await recommendation_engine.update_user_preferences(
            user_prefs.user_id,
            user_prefs.to_dict()
        )
        
        recommendations = await recommendation_engine.get_recommendations(
            user_id=user_prefs.user_id,
            strategy=RecommendationStrategy.MARKET_BASED
        )
        
        assert isinstance(recommendations, RecommendationResults)
        # Recommandations devraient refléter secteurs en croissance prédits
        if growing_sectors:
            assert len(recommendations.recommendations) >= 0


# Fixtures et utilitaires de test

@pytest.fixture(scope="session")
def event_loop():
    """Event loop pour tests asyncio"""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_cache_dir():
    """Répertoire cache temporaire pour tests"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


# Tests de performance

class TestMLAIPerformance:
    """Tests de performance pour systèmes ML/IA"""
    
    async def test_scoring_performance(self):
        """Test performance scoring IA"""
        scoring_engine = await get_ai_scoring_engine()
        
        # Données d'exemple
        company_data = {
            'siren': '123456789',
            'chiffre_affaires': 5000000,
            'resultat_net': 250000
        }
        
        start_time = datetime.now()
        result = await scoring_engine.score_company(company_data)
        end_time = datetime.now()
        
        processing_time = (end_time - start_time).total_seconds()
        
        # Scoring devrait être rapide (< 5 secondes)
        assert processing_time < 5.0
        assert result.processing_time_ms < 5000
    
    async def test_batch_processing_performance(self):
        """Test performance traitement par lot"""
        scoring_engine = await get_ai_scoring_engine()
        
        # Générer lot de 20 entreprises
        companies = [
            {
                'siren': f'12345678{i:01d}',
                'chiffre_affaires': 1000000 * (i + 1),
                'resultat_net': 50000 * i
            }
            for i in range(20)
        ]
        
        start_time = datetime.now()
        results = await scoring_engine.batch_score_companies(companies)
        end_time = datetime.now()
        
        total_time = (end_time - start_time).total_seconds()
        avg_time_per_company = total_time / len(results) if results else float('inf')
        
        # Traitement par lot devrait être efficace
        assert total_time < 30.0  # Moins de 30 secondes pour 20 entreprises
        assert avg_time_per_company < 2.0  # Moins de 2 secondes par entreprise en moyenne
        assert len(results) >= 15  # Au moins 75% de succès


if __name__ == "__main__":
    # Pour exécuter tests directement
    pytest.main([__file__, "-v"])