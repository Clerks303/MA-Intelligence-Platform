"""
Moteur de traitement du langage naturel pour M&A Intelligence Platform
US-008: NLP avancé pour extraction et analyse de données textuelles

Ce module implémente:
- Extraction d'entités à partir de textes non-structurés
- Analyse de sentiment sur contenus web
- Classification automatique de documents
- Résumé automatique de contenus longs
- Détection de tendances dans les actualités
- Analyse de la réputation d'entreprise
"""

import asyncio
import re
import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import warnings

# NLP Libraries
try:
    import spacy
    from spacy import displacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    warnings.warn("spaCy non disponible - certaines fonctionnalités NLP seront limitées")

try:
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.chunk import ne_chunk
    from nltk.tag import pos_tag
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    warnings.warn("NLTK non disponible - analyse de sentiment limitée")

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    from sentence_transformers import SentenceTransformer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    warnings.warn("Transformers non disponible - modèles avancés indisponibles")

try:
    from langdetect import detect
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False

from wordcloud import WordCloud
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
import diskcache

from app.core.logging_system import get_logger, LogCategory

logger = get_logger("nlp_engine", LogCategory.ML)


class AnalysisType(str, Enum):
    """Types d'analyse NLP disponibles"""
    ENTITY_EXTRACTION = "entity_extraction"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    DOCUMENT_CLASSIFICATION = "document_classification"
    TEXT_SUMMARIZATION = "text_summarization"
    KEYWORD_EXTRACTION = "keyword_extraction"
    REPUTATION_ANALYSIS = "reputation_analysis"
    TREND_DETECTION = "trend_detection"


class EntityType(str, Enum):
    """Types d'entités à extraire"""
    COMPANY = "company"
    PERSON = "person"
    LOCATION = "location"
    MONEY = "money"
    DATE = "date"
    PRODUCT = "product"
    SECTOR = "sector"
    SIREN = "siren"
    SIRET = "siret"


class SentimentLabel(str, Enum):
    """Labels de sentiment"""
    VERY_POSITIVE = "very_positive"
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    VERY_NEGATIVE = "very_negative"


@dataclass
class ExtractedEntity:
    """Entité extraite du texte"""
    text: str
    label: EntityType
    confidence: float
    start_char: int
    end_char: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SentimentResult:
    """Résultat d'analyse de sentiment"""
    overall_sentiment: SentimentLabel
    confidence: float
    positive_score: float
    neutral_score: float
    negative_score: float
    compound_score: float
    sentiment_by_sentence: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class DocumentClassification:
    """Classification de document"""
    predicted_class: str
    confidence: float
    all_scores: Dict[str, float]
    reasoning: List[str]


@dataclass
class TextSummary:
    """Résumé de texte"""
    summary: str
    original_length: int
    summary_length: int
    compression_ratio: float
    key_sentences: List[str]
    importance_scores: List[float]


@dataclass
class NLPAnalysisResult:
    """Résultat complet d'analyse NLP"""
    
    analysis_type: AnalysisType
    source_text: str
    language: str
    
    # Résultats spécifiques
    entities: List[ExtractedEntity] = field(default_factory=list)
    sentiment: Optional[SentimentResult] = None
    classification: Optional[DocumentClassification] = None
    summary: Optional[TextSummary] = None
    keywords: List[Tuple[str, float]] = field(default_factory=list)
    
    # Métadonnées
    processing_time_ms: float = 0.0
    confidence_score: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire pour JSON"""
        result = {
            'analysis_type': self.analysis_type.value,
            'language': self.language,
            'processing_time_ms': round(self.processing_time_ms, 2),
            'confidence_score': round(self.confidence_score, 3),
            'timestamp': self.timestamp.isoformat()
        }
        
        if self.entities:
            result['entities'] = [
                {
                    'text': e.text,
                    'label': e.label.value,
                    'confidence': round(e.confidence, 3),
                    'start_char': e.start_char,
                    'end_char': e.end_char,
                    'metadata': e.metadata
                }
                for e in self.entities
            ]
        
        if self.sentiment:
            result['sentiment'] = {
                'overall': self.sentiment.overall_sentiment.value,
                'confidence': round(self.sentiment.confidence, 3),
                'scores': {
                    'positive': round(self.sentiment.positive_score, 3),
                    'neutral': round(self.sentiment.neutral_score, 3),
                    'negative': round(self.sentiment.negative_score, 3),
                    'compound': round(self.sentiment.compound_score, 3)
                }
            }
        
        if self.classification:
            result['classification'] = {
                'predicted_class': self.classification.predicted_class,
                'confidence': round(self.classification.confidence, 3),
                'all_scores': {k: round(v, 3) for k, v in self.classification.all_scores.items()}
            }
        
        if self.summary:
            result['summary'] = {
                'text': self.summary.summary,
                'compression_ratio': round(self.summary.compression_ratio, 2),
                'key_sentences': self.summary.key_sentences[:3]  # Top 3
            }
        
        if self.keywords:
            result['keywords'] = [
                {'word': word, 'score': round(score, 3)}
                for word, score in self.keywords[:10]  # Top 10
            ]
        
        return result


class NLPEngine:
    """Moteur de traitement du langage naturel"""
    
    def __init__(self, cache_ttl: int = 7200):  # 2 heures de cache
        """
        Initialise le moteur NLP
        
        Args:
            cache_ttl: Durée de vie du cache en secondes
        """
        self.cache = diskcache.Cache('/tmp/ma_intelligence_nlp_cache')
        self.cache_ttl = cache_ttl
        
        # Modèles NLP
        self.spacy_model = None
        self.sentiment_analyzer = None
        self.transformers_pipeline = None
        self.sentence_transformer = None
        
        # Patterns regex pour entités françaises
        self.french_patterns = {
            EntityType.SIREN: re.compile(r'\b\d{9}\b'),
            EntityType.SIRET: re.compile(r'\b\d{14}\b'),
            EntityType.MONEY: re.compile(r'(?:\d{1,3}(?:\s\d{3})*|\d+)(?:,\d+)?\s*(?:€|euros?|EUR|millions?|milliards?)', re.IGNORECASE),
            EntityType.COMPANY: re.compile(r'\b[A-Z][a-z]*(?:\s[A-Z][a-z]*)*\s(?:SA|SAS|SARL|SNC|SCI|EURL|SASU)\b'),
        }
        
        # Stopwords français
        self.french_stopwords = {
            'le', 'de', 'et', 'à', 'un', 'il', 'être', 'et', 'en', 'avoir', 'que', 'pour',
            'dans', 'ce', 'son', 'une', 'sur', 'avec', 'ne', 'se', 'pas', 'tout', 'plus',
            'par', 'grand', 'en', 'une', 'être', 'et', 'en', 'avoir', 'que', 'pour'
        }
        
        # Classifications de documents
        self.document_classes = {
            'communique_presse': ['communiqué', 'presse', 'annonce', 'information'],
            'rapport_financier': ['résultats', 'chiffre', 'bilan', 'financier', 'comptable'],
            'article_news': ['actualité', 'news', 'article', 'information'],
            'document_legal': ['contrat', 'juridique', 'tribunal', 'décision', 'loi'],
            'site_web': ['accueil', 'services', 'contact', 'entreprise', 'société']
        }
        
        logger.info("🔤 Moteur NLP initialisé")
    
    async def initialize(self):
        """Initialise les modèles NLP"""
        try:
            # Charger modèle spaCy français
            if SPACY_AVAILABLE:
                try:
                    self.spacy_model = spacy.load("fr_core_news_sm")
                    logger.info("✅ Modèle spaCy français chargé")
                except OSError:
                    logger.warning("Modèle spaCy français non trouvé - fonctionnalités limitées")
            
            # Initialiser analyseur de sentiment NLTK
            if NLTK_AVAILABLE:
                try:
                    # Télécharger ressources NLTK nécessaires
                    import ssl
                    try:
                        _create_unverified_https_context = ssl._create_unverified_context
                    except AttributeError:
                        pass
                    else:
                        ssl._create_default_https_context = _create_unverified_https_context
                    
                    nltk.download('vader_lexicon', quiet=True)
                    nltk.download('punkt', quiet=True)
                    nltk.download('stopwords', quiet=True)
                    nltk.download('averaged_perceptron_tagger', quiet=True)
                    nltk.download('maxent_ne_chunker', quiet=True)
                    nltk.download('words', quiet=True)
                    
                    self.sentiment_analyzer = SentimentIntensityAnalyzer()
                    logger.info("✅ Analyseur de sentiment NLTK initialisé")
                except Exception as e:
                    logger.warning(f"Erreur initialisation NLTK: {e}")
            
            # Initialiser pipeline Transformers
            if TRANSFORMERS_AVAILABLE:
                try:
                    # Modèle de sentiment français
                    self.transformers_pipeline = pipeline(
                        "sentiment-analysis",
                        model="nlptown/bert-base-multilingual-uncased-sentiment",
                        device=-1  # CPU
                    )
                    
                    # Modèle d'embeddings de phrases
                    self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
                    
                    logger.info("✅ Pipelines Transformers initialisés")
                except Exception as e:
                    logger.warning(f"Erreur initialisation Transformers: {e}")
            
            logger.info("🔤 Moteur NLP complètement initialisé")
            
        except Exception as e:
            logger.error(f"Erreur initialisation NLP: {e}")
            raise
    
    async def analyze_text(self, 
                          text: str,
                          analysis_types: List[AnalysisType],
                          use_cache: bool = True) -> NLPAnalysisResult:
        """
        Analyse complète d'un texte
        
        Args:
            text: Texte à analyser
            analysis_types: Types d'analyses à effectuer
            use_cache: Utiliser le cache si disponible
            
        Returns:
            NLPAnalysisResult: Résultats d'analyse
        """
        start_time = datetime.now()
        
        try:
            # Vérifier cache
            cache_key = self._generate_cache_key(text, analysis_types)
            
            if use_cache:
                cached_result = self.cache.get(cache_key)
                if cached_result:
                    logger.debug("Résultat NLP récupéré du cache")
                    return NLPAnalysisResult(**cached_result)
            
            # Détecter langue
            language = self._detect_language(text)
            
            # Initialiser résultat
            result = NLPAnalysisResult(
                analysis_type=analysis_types[0] if analysis_types else AnalysisType.ENTITY_EXTRACTION,
                source_text=text[:500] + "..." if len(text) > 500 else text,  # Limiter pour stockage
                language=language
            )
            
            # Effectuer analyses demandées
            for analysis_type in analysis_types:
                if analysis_type == AnalysisType.ENTITY_EXTRACTION:
                    result.entities = await self._extract_entities(text)
                
                elif analysis_type == AnalysisType.SENTIMENT_ANALYSIS:
                    result.sentiment = await self._analyze_sentiment(text)
                
                elif analysis_type == AnalysisType.DOCUMENT_CLASSIFICATION:
                    result.classification = await self._classify_document(text)
                
                elif analysis_type == AnalysisType.TEXT_SUMMARIZATION:
                    result.summary = await self._summarize_text(text)
                
                elif analysis_type == AnalysisType.KEYWORD_EXTRACTION:
                    result.keywords = await self._extract_keywords(text)
            
            # Calculer métriques
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            result.processing_time_ms = processing_time
            result.confidence_score = self._calculate_overall_confidence(result)
            
            # Mettre en cache
            if use_cache:
                self.cache.set(cache_key, result.to_dict(), expire=self.cache_ttl)
            
            logger.info(f"Analyse NLP terminée en {processing_time:.1f}ms")
            
            return result
            
        except Exception as e:
            logger.error(f"Erreur analyse NLP: {e}")
            raise
    
    async def analyze_company_reputation(self, 
                                       company_name: str,
                                       texts: List[str],
                                       sources: List[str] = None) -> Dict[str, Any]:
        """
        Analyse la réputation d'une entreprise à partir de textes
        
        Args:
            company_name: Nom de l'entreprise
            texts: Liste de textes mentionnant l'entreprise
            sources: Sources des textes (optionnel)
            
        Returns:
            Dict: Analyse de réputation complète
        """
        try:
            logger.info(f"Analyse réputation pour {company_name}")
            
            if not texts:
                return {'error': 'Aucun texte fourni'}
            
            # Analyser sentiment de chaque texte
            sentiment_results = []
            entity_mentions = []
            keywords_global = Counter()
            
            for i, text in enumerate(texts):
                # Vérifier que le texte mentionne bien l'entreprise
                if company_name.lower() not in text.lower():
                    continue
                
                # Analyse NLP complète
                analysis = await self.analyze_text(
                    text,
                    [AnalysisType.SENTIMENT_ANALYSIS, AnalysisType.ENTITY_EXTRACTION, AnalysisType.KEYWORD_EXTRACTION]
                )
                
                if analysis.sentiment:
                    sentiment_result = {
                        'source': sources[i] if sources and i < len(sources) else f'source_{i}',
                        'sentiment': analysis.sentiment.overall_sentiment.value,
                        'confidence': analysis.sentiment.confidence,
                        'compound_score': analysis.sentiment.compound_score,
                        'text_preview': text[:200] + "..." if len(text) > 200 else text
                    }
                    sentiment_results.append(sentiment_result)
                
                # Extraire entités liées à l'entreprise
                company_entities = [
                    e for e in analysis.entities 
                    if e.label in [EntityType.COMPANY, EntityType.PERSON, EntityType.MONEY]
                ]
                entity_mentions.extend(company_entities)
                
                # Agréger mots-clés
                for keyword, score in analysis.keywords:
                    keywords_global[keyword] += score
            
            if not sentiment_results:
                return {'error': 'Aucune mention de l\'entreprise trouvée'}
            
            # Calculer métriques de réputation
            sentiment_scores = [r['compound_score'] for r in sentiment_results]
            overall_sentiment = np.mean(sentiment_scores)
            sentiment_volatility = np.std(sentiment_scores)
            
            # Classification de réputation
            if overall_sentiment >= 0.3:
                reputation_level = "Excellente"
            elif overall_sentiment >= 0.1:
                reputation_level = "Bonne"
            elif overall_sentiment >= -0.1:
                reputation_level = "Neutre"
            elif overall_sentiment >= -0.3:
                reputation_level = "Préoccupante"
            else:
                reputation_level = "Négative"
            
            # Analyser tendances temporelles (si sources datées)
            temporal_trends = self._analyze_temporal_sentiment_trends(sentiment_results)
            
            # Extraire thèmes principaux
            top_keywords = keywords_global.most_common(10)
            themes = self._extract_reputation_themes(sentiment_results, top_keywords)
            
            # Recommandations
            recommendations = self._generate_reputation_recommendations(
                overall_sentiment, sentiment_volatility, reputation_level, themes
            )
            
            return {
                'company_name': company_name,
                'overall_sentiment_score': round(overall_sentiment, 3),
                'reputation_level': reputation_level,
                'sentiment_volatility': round(sentiment_volatility, 3),
                'total_mentions': len(sentiment_results),
                'sentiment_distribution': {
                    'positive': len([r for r in sentiment_results if r['compound_score'] > 0.1]),
                    'neutral': len([r for r in sentiment_results if -0.1 <= r['compound_score'] <= 0.1]),
                    'negative': len([r for r in sentiment_results if r['compound_score'] < -0.1])
                },
                'detailed_sentiments': sentiment_results,
                'key_themes': themes,
                'top_keywords': [{'keyword': k, 'frequency': v} for k, v in top_keywords],
                'entity_mentions': [
                    {'text': e.text, 'type': e.label.value, 'confidence': e.confidence}
                    for e in entity_mentions
                ],
                'temporal_trends': temporal_trends,
                'recommendations': recommendations,
                'analysis_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Erreur analyse réputation: {e}")
            return {'error': f'Erreur analyse: {str(e)}'}
    
    async def detect_market_trends(self, 
                                 news_texts: List[str],
                                 sector: str = None,
                                 time_window_days: int = 30) -> Dict[str, Any]:
        """
        Détecte les tendances du marché à partir d'actualités
        
        Args:
            news_texts: Textes d'actualités
            sector: Secteur spécifique (optionnel)
            time_window_days: Fenêtre temporelle d'analyse
            
        Returns:
            Dict: Tendances détectées
        """
        try:
            logger.info(f"Détection tendances marché sur {len(news_texts)} articles")
            
            # Analyser sentiment général des actualités
            sentiment_scores = []
            all_keywords = Counter()
            entities_detected = []
            
            for text in news_texts:
                analysis = await self.analyze_text(
                    text,
                    [AnalysisType.SENTIMENT_ANALYSIS, AnalysisType.KEYWORD_EXTRACTION, AnalysisType.ENTITY_EXTRACTION]
                )
                
                if analysis.sentiment:
                    sentiment_scores.append(analysis.sentiment.compound_score)
                
                # Agréger mots-clés
                for keyword, score in analysis.keywords:
                    all_keywords[keyword] += score
                
                # Entités importantes
                entities_detected.extend([
                    e for e in analysis.entities 
                    if e.label in [EntityType.COMPANY, EntityType.MONEY, EntityType.SECTOR]
                ])
            
            # Calculer tendance de sentiment du marché
            market_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0.0
            sentiment_trend = self._classify_market_sentiment(market_sentiment)
            
            # Identifier thèmes émergents
            emerging_themes = self._identify_emerging_themes(all_keywords.most_common(20))
            
            # Analyser mentions d'entreprises
            company_mentions = self._analyze_company_mentions(entities_detected)
            
            # Détecter signaux faibles
            weak_signals = self._detect_weak_signals(all_keywords, entities_detected)
            
            # Prédictions de tendances
            trend_predictions = self._predict_trend_evolution(
                sentiment_scores, all_keywords, emerging_themes
            )
            
            return {
                'analysis_period': f'{time_window_days} jours',
                'articles_analyzed': len(news_texts),
                'market_sentiment': {
                    'score': round(market_sentiment, 3),
                    'trend': sentiment_trend,
                    'volatility': round(np.std(sentiment_scores), 3) if sentiment_scores else 0.0
                },
                'emerging_themes': emerging_themes,
                'top_keywords': [
                    {'keyword': k, 'frequency': v, 'relevance': round(v / len(news_texts), 2)}
                    for k, v in all_keywords.most_common(15)
                ],
                'company_activity': company_mentions,
                'weak_signals': weak_signals,
                'trend_predictions': trend_predictions,
                'sector_focus': sector,
                'analysis_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Erreur détection tendances: {e}")
            return {'error': f'Erreur analyse: {str(e)}'}
    
    async def classify_business_content(self, 
                                      content: str,
                                      custom_classes: List[str] = None) -> DocumentClassification:
        """
        Classifie du contenu business
        
        Args:
            content: Contenu à classifier
            custom_classes: Classes personnalisées (optionnel)
            
        Returns:
            DocumentClassification: Classification du document
        """
        try:
            # Utiliser classes par défaut ou personnalisées
            classes = custom_classes or list(self.document_classes.keys())
            
            # Préparer le texte
            content_lower = content.lower()
            
            # Calculer scores pour chaque classe
            class_scores = {}
            
            for class_name in classes:
                if class_name in self.document_classes:
                    # Utiliser mots-clés prédéfinis
                    keywords = self.document_classes[class_name]
                else:
                    # Classes personnalisées - score générique
                    keywords = [class_name.replace('_', ' ')]
                
                # Calculer score basé sur présence de mots-clés
                score = 0.0
                for keyword in keywords:
                    count = content_lower.count(keyword.lower())
                    score += count * (1.0 / len(keywords))
                
                # Normaliser par longueur du texte
                normalized_score = score / (len(content.split()) / 100)
                class_scores[class_name] = min(1.0, normalized_score)
            
            # Classe prédite et confiance
            if class_scores:
                predicted_class = max(class_scores.items(), key=lambda x: x[1])
                predicted_class_name = predicted_class[0]
                confidence = predicted_class[1]
            else:
                predicted_class_name = "unknown"
                confidence = 0.0
            
            # Raisonnement
            reasoning = []
            if confidence > 0.3:
                keywords_found = []
                if predicted_class_name in self.document_classes:
                    for keyword in self.document_classes[predicted_class_name]:
                        if keyword.lower() in content_lower:
                            keywords_found.append(keyword)
                
                if keywords_found:
                    reasoning.append(f"Mots-clés détectés: {', '.join(keywords_found)}")
            else:
                reasoning.append("Classification incertaine - contenu ambigu")
            
            return DocumentClassification(
                predicted_class=predicted_class_name,
                confidence=confidence,
                all_scores=class_scores,
                reasoning=reasoning
            )
            
        except Exception as e:
            logger.error(f"Erreur classification: {e}")
            return DocumentClassification(
                predicted_class="error",
                confidence=0.0,
                all_scores={},
                reasoning=[f"Erreur: {str(e)}"]
            )
    
    async def _extract_entities(self, text: str) -> List[ExtractedEntity]:
        """Extrait les entités nommées du texte"""
        entities = []
        
        try:
            # Utiliser spaCy si disponible
            if self.spacy_model:
                doc = self.spacy_model(text)
                
                for ent in doc.ents:
                    # Mapper les labels spaCy vers nos types
                    entity_type = self._map_spacy_label(ent.label_)
                    if entity_type:
                        entities.append(ExtractedEntity(
                            text=ent.text,
                            label=entity_type,
                            confidence=0.8,  # spaCy ne fournit pas de score de confiance
                            start_char=ent.start_char,
                            end_char=ent.end_char,
                            metadata={'spacy_label': ent.label_}
                        ))
            
            # Extraction avec patterns regex
            regex_entities = self._extract_entities_with_regex(text)
            entities.extend(regex_entities)
            
            # Déduplication basée sur la position
            entities = self._deduplicate_entities(entities)
            
        except Exception as e:
            logger.error(f"Erreur extraction entités: {e}")
        
        return entities
    
    def _extract_entities_with_regex(self, text: str) -> List[ExtractedEntity]:
        """Extrait entités avec patterns regex français"""
        entities = []
        
        for entity_type, pattern in self.french_patterns.items():
            for match in pattern.finditer(text):
                entities.append(ExtractedEntity(
                    text=match.group(),
                    label=entity_type,
                    confidence=0.9,  # Haute confiance pour patterns spécifiques
                    start_char=match.start(),
                    end_char=match.end(),
                    metadata={'method': 'regex'}
                ))
        
        return entities
    
    def _map_spacy_label(self, spacy_label: str) -> Optional[EntityType]:
        """Mappe les labels spaCy vers nos types d'entités"""
        mapping = {
            'ORG': EntityType.COMPANY,
            'PERSON': EntityType.PERSON,
            'PER': EntityType.PERSON,
            'LOC': EntityType.LOCATION,
            'GPE': EntityType.LOCATION,
            'MONEY': EntityType.MONEY,
            'DATE': EntityType.DATE,
            'TIME': EntityType.DATE
        }
        return mapping.get(spacy_label)
    
    def _deduplicate_entities(self, entities: List[ExtractedEntity]) -> List[ExtractedEntity]:
        """Supprime les entités dupliquées basées sur la position"""
        deduplicated = []
        seen_positions = set()
        
        for entity in entities:
            position_key = (entity.start_char, entity.end_char)
            if position_key not in seen_positions:
                deduplicated.append(entity)
                seen_positions.add(position_key)
        
        return deduplicated
    
    async def _analyze_sentiment(self, text: str) -> SentimentResult:
        """Analyse le sentiment du texte"""
        try:
            # Initialiser scores
            positive_score = 0.0
            neutral_score = 0.0
            negative_score = 0.0
            compound_score = 0.0
            
            # Utiliser NLTK VADER si disponible
            if self.sentiment_analyzer:
                scores = self.sentiment_analyzer.polarity_scores(text)
                positive_score = scores['pos']
                neutral_score = scores['neu']
                negative_score = scores['neg']
                compound_score = scores['compound']
            
            # Utiliser Transformers comme alternative/complément
            elif self.transformers_pipeline:
                try:
                    result = self.transformers_pipeline(text[:512])  # Limiter longueur
                    
                    # Convertir score Transformers
                    if result[0]['label'] == 'POSITIVE':
                        positive_score = result[0]['score']
                        compound_score = result[0]['score']
                    else:
                        negative_score = result[0]['score']
                        compound_score = -result[0]['score']
                    
                    neutral_score = 1.0 - positive_score - negative_score
                    
                except Exception as e:
                    logger.warning(f"Erreur Transformers sentiment: {e}")
            
            # Fallback avec TextBlob
            elif TEXTBLOB_AVAILABLE:
                blob = TextBlob(text)
                polarity = blob.sentiment.polarity
                
                if polarity > 0:
                    positive_score = polarity
                elif polarity < 0:
                    negative_score = abs(polarity)
                else:
                    neutral_score = 1.0
                
                compound_score = polarity
            
            # Déterminer sentiment global
            if compound_score >= 0.5:
                overall_sentiment = SentimentLabel.VERY_POSITIVE
            elif compound_score >= 0.1:
                overall_sentiment = SentimentLabel.POSITIVE
            elif compound_score >= -0.1:
                overall_sentiment = SentimentLabel.NEUTRAL
            elif compound_score >= -0.5:
                overall_sentiment = SentimentLabel.NEGATIVE
            else:
                overall_sentiment = SentimentLabel.VERY_NEGATIVE
            
            # Confiance basée sur l'écart par rapport à neutral
            confidence = min(1.0, abs(compound_score) * 2)
            
            # Analyse par phrase
            sentence_sentiments = []
            if NLTK_AVAILABLE:
                try:
                    sentences = sent_tokenize(text)
                    for sentence in sentences[:5]:  # Limiter à 5 phrases
                        if self.sentiment_analyzer:
                            sent_scores = self.sentiment_analyzer.polarity_scores(sentence)
                            sentence_sentiments.append({
                                'sentence': sentence,
                                'compound': sent_scores['compound'],
                                'sentiment': self._classify_sentiment_score(sent_scores['compound']).value
                            })
                except:
                    pass
            
            return SentimentResult(
                overall_sentiment=overall_sentiment,
                confidence=confidence,
                positive_score=positive_score,
                neutral_score=neutral_score,
                negative_score=negative_score,
                compound_score=compound_score,
                sentiment_by_sentence=sentence_sentiments
            )
            
        except Exception as e:
            logger.error(f"Erreur analyse sentiment: {e}")
            # Retourner sentiment neutre en cas d'erreur
            return SentimentResult(
                overall_sentiment=SentimentLabel.NEUTRAL,
                confidence=0.0,
                positive_score=0.33,
                neutral_score=0.34,
                negative_score=0.33,
                compound_score=0.0
            )
    
    def _classify_sentiment_score(self, score: float) -> SentimentLabel:
        """Classifie un score de sentiment"""
        if score >= 0.5:
            return SentimentLabel.VERY_POSITIVE
        elif score >= 0.1:
            return SentimentLabel.POSITIVE
        elif score >= -0.1:
            return SentimentLabel.NEUTRAL
        elif score >= -0.5:
            return SentimentLabel.NEGATIVE
        else:
            return SentimentLabel.VERY_NEGATIVE
    
    async def _classify_document(self, text: str) -> DocumentClassification:
        """Classifie le type de document"""
        return await self.classify_business_content(text)
    
    async def _summarize_text(self, text: str, max_sentences: int = 3) -> TextSummary:
        """Résume le texte (méthode extractive simple)"""
        try:
            if NLTK_AVAILABLE:
                sentences = sent_tokenize(text)
            else:
                # Fallback simple
                sentences = [s.strip() for s in text.split('.') if s.strip()]
            
            if len(sentences) <= max_sentences:
                return TextSummary(
                    summary=text,
                    original_length=len(text),
                    summary_length=len(text),
                    compression_ratio=1.0,
                    key_sentences=sentences,
                    importance_scores=[1.0] * len(sentences)
                )
            
            # Calculer scores d'importance (fréquence TF-IDF simplifiée)
            word_freq = Counter()
            
            # Tokenisation simple
            words = re.findall(r'\b\w+\b', text.lower())
            for word in words:
                if word not in self.french_stopwords and len(word) > 3:
                    word_freq[word] += 1
            
            # Score de chaque phrase
            sentence_scores = []
            for sentence in sentences:
                sentence_words = re.findall(r'\b\w+\b', sentence.lower())
                score = sum(word_freq.get(word, 0) for word in sentence_words)
                sentence_scores.append(score)
            
            # Sélectionner top phrases
            indexed_sentences = list(enumerate(sentence_scores))
            top_sentences = sorted(indexed_sentences, key=lambda x: x[1], reverse=True)[:max_sentences]
            
            # Réordonner selon ordre original
            selected_indices = sorted([idx for idx, score in top_sentences])
            key_sentences = [sentences[i] for i in selected_indices]
            importance_scores = [sentence_scores[i] for i in selected_indices]
            
            summary = '. '.join(key_sentences)
            
            return TextSummary(
                summary=summary,
                original_length=len(text),
                summary_length=len(summary),
                compression_ratio=len(summary) / len(text),
                key_sentences=key_sentences,
                importance_scores=importance_scores
            )
            
        except Exception as e:
            logger.error(f"Erreur résumé: {e}")
            # Retourner début du texte en cas d'erreur
            preview = text[:300] + "..." if len(text) > 300 else text
            return TextSummary(
                summary=preview,
                original_length=len(text),
                summary_length=len(preview),
                compression_ratio=len(preview) / len(text),
                key_sentences=[preview],
                importance_scores=[1.0]
            )
    
    async def _extract_keywords(self, text: str, max_keywords: int = 20) -> List[Tuple[str, float]]:
        """Extrait les mots-clés importants"""
        try:
            # Nettoyer et tokeniser
            words = re.findall(r'\b\w+\b', text.lower())
            
            # Filtrer stopwords et mots courts
            filtered_words = [
                word for word in words 
                if word not in self.french_stopwords and len(word) > 3
            ]
            
            # Calculer fréquences
            word_freq = Counter(filtered_words)
            
            # Calculer TF-IDF simplifié (sans corpus de référence)
            total_words = len(filtered_words)
            
            # Score basé sur fréquence et longueur
            keyword_scores = []
            for word, freq in word_freq.most_common(max_keywords * 2):
                # Score combinant fréquence et rareté relative
                tf_score = freq / total_words
                length_bonus = min(1.2, len(word) / 10)  # Bonus pour mots plus longs
                
                # Pénalité pour mots très fréquents
                frequency_penalty = 1.0 if freq < 5 else 0.8
                
                final_score = tf_score * length_bonus * frequency_penalty
                keyword_scores.append((word, final_score))
            
            # Trier et retourner top mots-clés
            keyword_scores.sort(key=lambda x: x[1], reverse=True)
            
            return keyword_scores[:max_keywords]
            
        except Exception as e:
            logger.error(f"Erreur extraction mots-clés: {e}")
            return []
    
    def _detect_language(self, text: str) -> str:
        """Détecte la langue du texte"""
        try:
            if LANGDETECT_AVAILABLE:
                lang = detect(text)
                return lang
            else:
                # Détection simple basée sur mots français courants
                french_words = ['le', 'de', 'et', 'à', 'un', 'être', 'avoir', 'que', 'pour', 'dans']
                text_lower = text.lower()
                french_count = sum(1 for word in french_words if word in text_lower)
                
                return 'fr' if french_count >= 3 else 'en'
                
        except Exception:
            return 'unknown'
    
    def _calculate_overall_confidence(self, result: NLPAnalysisResult) -> float:
        """Calcule la confiance globale de l'analyse"""
        confidences = []
        
        if result.sentiment:
            confidences.append(result.sentiment.confidence)
        
        if result.classification:
            confidences.append(result.classification.confidence)
        
        if result.entities:
            entity_confidences = [e.confidence for e in result.entities]
            if entity_confidences:
                confidences.append(np.mean(entity_confidences))
        
        return np.mean(confidences) if confidences else 0.5
    
    def _generate_cache_key(self, text: str, analysis_types: List[AnalysisType]) -> str:
        """Génère une clé de cache unique"""
        key_data = {
            'text_hash': hashlib.md5(text.encode()).hexdigest(),
            'analysis_types': [at.value for at in analysis_types]
        }
        key_string = json.dumps(key_data, sort_keys=True)
        return f"nlp_{hashlib.md5(key_string.encode()).hexdigest()}"
    
    def _analyze_temporal_sentiment_trends(self, sentiment_results: List[Dict]) -> Dict[str, Any]:
        """Analyse les tendances temporelles de sentiment"""
        # Pour cette version, retourner analyse basique
        # Dans une version complète, analyser les dates des sources
        
        scores = [r['compound_score'] for r in sentiment_results]
        
        if len(scores) < 2:
            return {'trend': 'stable', 'data_insufficient': True}
        
        # Tendance simple basée sur première/dernière moitié
        mid_point = len(scores) // 2
        first_half_avg = np.mean(scores[:mid_point])
        second_half_avg = np.mean(scores[mid_point:])
        
        if second_half_avg > first_half_avg + 0.1:
            trend = 'amelioration'
        elif second_half_avg < first_half_avg - 0.1:
            trend = 'degradation'
        else:
            trend = 'stable'
        
        return {
            'trend': trend,
            'first_period_avg': round(first_half_avg, 3),
            'second_period_avg': round(second_half_avg, 3),
            'change_magnitude': round(abs(second_half_avg - first_half_avg), 3)
        }
    
    def _extract_reputation_themes(self, sentiment_results: List[Dict], top_keywords: List[Tuple]) -> List[Dict]:
        """Extrait les thèmes principaux de réputation"""
        themes = []
        
        # Analyser sentiment par mots-clés principaux
        for keyword, frequency in top_keywords[:5]:
            related_sentiments = []
            
            for result in sentiment_results:
                if keyword.lower() in result['text_preview'].lower():
                    related_sentiments.append(result['compound_score'])
            
            if related_sentiments:
                avg_sentiment = np.mean(related_sentiments)
                theme = {
                    'theme': keyword,
                    'frequency': frequency,
                    'avg_sentiment': round(avg_sentiment, 3),
                    'sentiment_label': self._classify_sentiment_score(avg_sentiment).value,
                    'mention_count': len(related_sentiments)
                }
                themes.append(theme)
        
        return themes
    
    def _generate_reputation_recommendations(self, overall_sentiment: float, 
                                           volatility: float, 
                                           reputation_level: str, 
                                           themes: List[Dict]) -> List[str]:
        """Génère des recommandations de gestion de réputation"""
        recommendations = []
        
        # Recommandations basées sur niveau général
        if reputation_level == "Négative":
            recommendations.append("Priorité absolue : plan de communication de crise")
            recommendations.append("Analyser causes profondes des perceptions négatives")
        elif reputation_level == "Préoccupante":
            recommendations.append("Surveillance renforcée et actions préventives recommandées")
        elif reputation_level == "Excellente":
            recommendations.append("Maintenir efforts actuels et capitaliser sur réputation positive")
        
        # Recommandations basées sur volatilité
        if volatility > 0.3:
            recommendations.append("Forte volatilité détectée - stabiliser message corporate")
        
        # Recommandations basées sur thèmes
        negative_themes = [t for t in themes if t['avg_sentiment'] < -0.1]
        if negative_themes:
            main_issue = negative_themes[0]['theme']
            recommendations.append(f"Adresser spécifiquement le thème '{main_issue}'")
        
        return recommendations[:5]
    
    def _classify_market_sentiment(self, sentiment_score: float) -> str:
        """Classifie le sentiment du marché"""
        if sentiment_score >= 0.3:
            return "Très optimiste"
        elif sentiment_score >= 0.1:
            return "Optimiste"
        elif sentiment_score >= -0.1:
            return "Neutre"
        elif sentiment_score >= -0.3:
            return "Pessimiste"
        else:
            return "Très pessimiste"
    
    def _identify_emerging_themes(self, top_keywords: List[Tuple]) -> List[Dict]:
        """Identifie les thèmes émergents"""
        themes = []
        
        # Regrouper mots-clés par thématiques business
        business_themes = {
            'digital_transformation': ['digital', 'numérique', 'technologie', 'ia', 'automatisation'],
            'sustainability': ['durable', 'environnement', 'vert', 'carbone', 'énergie'],
            'ma_activity': ['acquisition', 'fusion', 'rachat', 'investissement', 'capital'],
            'innovation': ['innovation', 'recherche', 'développement', 'startup', 'innovation'],
            'regulation': ['régulation', 'loi', 'réglementation', 'compliance', 'juridique']
        }
        
        for theme_name, theme_keywords in business_themes.items():
            theme_score = 0
            theme_mentions = []
            
            for keyword, frequency in top_keywords:
                if any(tk in keyword.lower() for tk in theme_keywords):
                    theme_score += frequency
                    theme_mentions.append(keyword)
            
            if theme_score > 0:
                themes.append({
                    'theme': theme_name,
                    'total_score': theme_score,
                    'keywords': theme_mentions[:3],  # Top 3 mots-clés
                    'relevance': round(theme_score / sum(freq for _, freq in top_keywords), 3)
                })
        
        # Trier par pertinence
        themes.sort(key=lambda x: x['total_score'], reverse=True)
        return themes[:5]
    
    def _analyze_company_mentions(self, entities: List[ExtractedEntity]) -> Dict[str, Any]:
        """Analyse les mentions d'entreprises"""
        company_entities = [e for e in entities if e.label == EntityType.COMPANY]
        
        # Compter mentions par entreprise
        company_counts = Counter(e.text for e in company_entities)
        
        # Analyser activité M&A mentionnée
        ma_indicators = Counter()
        for entity in entities:
            if entity.label == EntityType.MONEY:
                ma_indicators['financial_transactions'] += 1
        
        return {
            'total_company_mentions': len(company_entities),
            'unique_companies': len(company_counts),
            'top_mentioned_companies': [
                {'company': name, 'mentions': count}
                for name, count in company_counts.most_common(10)
            ],
            'ma_activity_indicators': dict(ma_indicators)
        }
    
    def _detect_weak_signals(self, keywords: Counter, entities: List[ExtractedEntity]) -> List[Dict]:
        """Détecte les signaux faibles du marché"""
        weak_signals = []
        
        # Mots-clés émergents (faible fréquence mais potentiellement importants)
        emerging_keywords = [
            (word, freq) for word, freq in keywords.items()
            if 2 <= freq <= 5 and len(word) > 5  # Fréquence modérée, mots spécifiques
        ]
        
        # Analyser secteurs émergents
        sector_keywords = ['fintech', 'biotech', 'cleantech', 'edtech', 'proptech']
        for keyword, freq in emerging_keywords:
            if any(sector in keyword.lower() for sector in sector_keywords):
                weak_signals.append({
                    'type': 'emerging_sector',
                    'signal': keyword,
                    'frequency': freq,
                    'description': f"Activité émergente dans {keyword}"
                })
        
        # Nouvelles réglementations
        regulatory_keywords = ['nouvelle', 'réglementation', 'directive', 'compliance']
        regulatory_signals = [
            word for word, freq in emerging_keywords
            if any(reg in word.lower() for reg in regulatory_keywords)
        ]
        
        for signal in regulatory_signals:
            weak_signals.append({
                'type': 'regulatory_change',
                'signal': signal,
                'description': f"Changement réglementaire potentiel: {signal}"
            })
        
        return weak_signals[:5]
    
    def _predict_trend_evolution(self, sentiment_scores: List[float], 
                               keywords: Counter, 
                               themes: List[Dict]) -> Dict[str, Any]:
        """Prédit l'évolution des tendances"""
        
        # Analyse de momentum basée sur sentiment
        if len(sentiment_scores) >= 5:
            recent_trend = np.mean(sentiment_scores[-3:]) - np.mean(sentiment_scores[:3])
            
            if recent_trend > 0.1:
                momentum = "acceleration_positive"
            elif recent_trend < -0.1:
                momentum = "acceleration_negative"
            else:
                momentum = "stable"
        else:
            momentum = "insufficient_data"
        
        # Prédiction basée sur thèmes dominants
        if themes:
            dominant_theme = themes[0]['theme']
            theme_prediction = f"Tendance dominante: {dominant_theme}"
        else:
            theme_prediction = "Aucune tendance claire identifiée"
        
        # Niveau de confiance
        confidence = min(1.0, len(sentiment_scores) / 20)  # Plus de données = plus de confiance
        
        return {
            'momentum': momentum,
            'dominant_theme_prediction': theme_prediction,
            'confidence_level': round(confidence, 2),
            'prediction_horizon': "1-3 mois",
            'key_indicators_to_watch': [
                theme['theme'] for theme in themes[:3]
            ]
        }


# Instance globale
_nlp_engine: Optional[NLPEngine] = None


async def get_nlp_engine() -> NLPEngine:
    """Factory pour obtenir l'instance du moteur NLP"""
    global _nlp_engine
    
    if _nlp_engine is None:
        _nlp_engine = NLPEngine()
        await _nlp_engine.initialize()
    
    return _nlp_engine


async def initialize_nlp_engine():
    """Initialise le système NLP au démarrage"""
    try:
        engine = await get_nlp_engine()
        logger.info("🔤 Système NLP initialisé avec succès")
        return engine
    except Exception as e:
        logger.error(f"Erreur initialisation NLP: {e}")
        raise