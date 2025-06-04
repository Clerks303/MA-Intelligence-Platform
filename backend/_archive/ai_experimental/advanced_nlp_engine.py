"""
Moteur NLP avancé pour analyse de sentiment et traitement textuel
US-010: Analyse intelligente des données textuelles pour M&A Intelligence Platform

Ce module fournit:
- Analyse de sentiment multi-langue (français/anglais)
- Extraction d'entités nommées (NER)
- Classification automatique de textes
- Détection de topics et clustering sémantique
- Analyse de réputation et risques
- Résumé automatique de documents
- Similarité sémantique et matching
"""

import asyncio
import re
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from functools import wraps, lru_cache
from enum import Enum
import unicodedata

# NLP Libraries
import spacy
import nltk
from textblob import TextBlob
from langdetect import detect, LangDetectError

# Transformers pour modèles avancés
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    AutoModelForTokenClassification, AutoModel,
    pipeline, BertTokenizer, BertModel
)
from sentence_transformers import SentenceTransformer

# Sklearn pour clustering et vectorisation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

# Autres
import numpy as np
import pandas as pd
from wordcloud import WordCloud
import networkx as nx
from collections import Counter, defaultdict

from app.core.logging_system import get_logger, LogCategory
from app.core.cache_manager import get_cache_manager, cached

logger = get_logger("advanced_nlp_engine", LogCategory.AI_ML)


class SentimentScore(str, Enum):
    """Scores de sentiment"""
    VERY_POSITIVE = "very_positive"    # > 0.6
    POSITIVE = "positive"              # 0.2 to 0.6
    NEUTRAL = "neutral"                # -0.2 to 0.2
    NEGATIVE = "negative"              # -0.6 to -0.2
    VERY_NEGATIVE = "very_negative"    # < -0.6


class TextCategory(str, Enum):
    """Catégories de texte pour classification"""
    FINANCIAL_NEWS = "financial_news"
    COMPANY_DESCRIPTION = "company_description"
    PRESS_RELEASE = "press_release"
    ANALYST_REPORT = "analyst_report"
    SOCIAL_MEDIA = "social_media"
    REGULATORY_FILING = "regulatory_filing"
    INDUSTRY_REPORT = "industry_report"
    OTHER = "other"


class RiskLevel(str, Enum):
    """Niveaux de risque détectés"""
    VERY_HIGH = "very_high"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    VERY_LOW = "very_low"


@dataclass
class SentimentAnalysis:
    """Résultat d'analyse de sentiment"""
    text_id: str
    language: str
    sentiment_score: float  # -1 to 1
    sentiment_label: SentimentScore
    confidence: float  # 0 to 1
    
    # Détails par aspect
    emotional_indicators: Dict[str, float]  # joy, fear, anger, etc.
    subjectivity: float  # 0 (objective) to 1 (subjective)
    
    # Mots-clés sentiment
    positive_keywords: List[str]
    negative_keywords: List[str]
    
    # Métadonnées
    analysis_timestamp: datetime = field(default_factory=datetime.now)
    model_used: str = "ensemble"


@dataclass
class EntityExtraction:
    """Résultat d'extraction d'entités"""
    text_id: str
    entities: List[Dict[str, Any]]  # {'text': str, 'label': str, 'confidence': float, 'start': int, 'end': int}
    
    # Entités par catégorie
    persons: List[str]
    organizations: List[str]
    locations: List[str]
    financial_amounts: List[str]
    dates: List[str]
    
    # Relations détectées
    relationships: List[Dict[str, Any]]
    
    analysis_timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class TextClassification:
    """Résultat de classification de texte"""
    text_id: str
    predicted_category: TextCategory
    confidence: float
    
    # Probabilités pour toutes les catégories
    category_probabilities: Dict[TextCategory, float]
    
    # Features importantes pour la classification
    key_features: List[Tuple[str, float]]
    
    analysis_timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class TopicAnalysis:
    """Analyse de topics/thèmes"""
    text_id: str
    topics: List[Dict[str, Any]]  # {'topic_id': int, 'keywords': List[str], 'weight': float}
    dominant_topic: int
    
    # Clustering sémantique
    semantic_cluster: int
    cluster_description: str
    
    analysis_timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class RiskAssessment:
    """Évaluation de risques basée sur NLP"""
    text_id: str
    overall_risk_level: RiskLevel
    risk_score: float  # 0 to 100
    
    # Risques par catégorie
    financial_risk: float
    operational_risk: float
    regulatory_risk: float
    reputation_risk: float
    
    # Indicateurs de risque détectés
    risk_indicators: List[Dict[str, Any]]
    
    # Recommandations
    risk_mitigation_suggestions: List[str]
    
    analysis_timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ComprehensiveTextAnalysis:
    """Analyse complète d'un texte"""
    text_id: str
    original_text: str
    cleaned_text: str
    language: str
    
    # Analyses spécialisées
    sentiment: SentimentAnalysis
    entities: EntityExtraction
    classification: TextClassification
    topics: TopicAnalysis
    risk_assessment: RiskAssessment
    
    # Métrics textuelles
    readability_score: float
    complexity_score: float
    word_count: int
    sentence_count: int
    
    # Résumé automatique
    summary: str
    key_phrases: List[str]
    
    analysis_timestamp: datetime = field(default_factory=datetime.now)


class TextPreprocessor:
    """Préprocesseur de texte avancé"""
    
    def __init__(self):
        # Patterns de nettoyage
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        self.phone_pattern = re.compile(r'(\+33|0)[1-9](\d{8})')
        self.financial_pattern = re.compile(r'(\d+(?:\.\d+)?)\s*([M€|M$|K€|K$|€|$|EUR|USD])')
        
        # Mots vides français/anglais
        try:
            nltk.download('stopwords', quiet=True)
            from nltk.corpus import stopwords
            self.french_stopwords = set(stopwords.words('french'))
            self.english_stopwords = set(stopwords.words('english'))
        except:
            self.french_stopwords = set()
            self.english_stopwords = set()
    
    def clean_text(self, text: str, preserve_entities: bool = True) -> str:
        """Nettoie le texte en préservant les entités importantes"""
        
        if not text or not isinstance(text, str):
            return ""
        
        # Normalisation Unicode
        text = unicodedata.normalize('NFKD', text)
        
        # Remplacement des entités si preservation demandée
        if preserve_entities:
            # Conserver mais marquer les montants financiers
            text = self.financial_pattern.sub(r'MONTANT_FINANCIER_\1_\2', text)
            
            # Conserver mais marquer emails/téléphones
            text = self.email_pattern.sub('EMAIL_ADDRESS', text)
            text = self.phone_pattern.sub('PHONE_NUMBER', text)
        else:
            # Supprimer complètement
            text = self.financial_pattern.sub('', text)
            text = self.email_pattern.sub('', text)
            text = self.phone_pattern.sub('', text)
        
        # Supprimer URLs
        text = self.url_pattern.sub('', text)
        
        # Nettoyage des caractères spéciaux mais préservation de la ponctuation importante
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\'\"€$%]', ' ', text)
        
        # Normalisation des espaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def extract_financial_amounts(self, text: str) -> List[Dict[str, Any]]:
        """Extrait les montants financiers du texte"""
        
        amounts = []
        matches = self.financial_pattern.finditer(text)
        
        for match in matches:
            amount_str = match.group(1)
            currency = match.group(2)
            
            try:
                amount = float(amount_str)
                
                # Conversion en euros si nécessaire
                if currency in ['K€', 'K$']:
                    amount *= 1000
                elif currency in ['M€', 'M$']:
                    amount *= 1000000
                
                amounts.append({
                    'original_text': match.group(0),
                    'amount': amount,
                    'currency': currency,
                    'position': (match.start(), match.end())
                })
            except ValueError:
                continue
        
        return amounts
    
    def detect_language(self, text: str) -> str:
        """Détecte la langue du texte"""
        
        try:
            return detect(text)
        except LangDetectError:
            # Fallback: heuristique simple
            french_indicators = ['le', 'la', 'les', 'et', 'est', 'une', 'des', 'pour', 'avec']
            english_indicators = ['the', 'and', 'is', 'a', 'of', 'to', 'in', 'that', 'for']
            
            text_lower = text.lower()
            french_count = sum(1 for word in french_indicators if word in text_lower)
            english_count = sum(1 for word in english_indicators if word in text_lower)
            
            return 'fr' if french_count > english_count else 'en'


class MultilingualSentimentAnalyzer:
    """Analyseur de sentiment multilingue"""
    
    def __init__(self):
        self.models = {}
        self.tokenizers = {}
        self._load_models()
        
    def _load_models(self):
        """Charge les modèles de sentiment"""
        
        try:
            # Modèle français
            self.models['fr'] = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                tokenizer="cardiffnlp/twitter-roberta-base-sentiment-latest"
            )
            
            # Modèle anglais
            self.models['en'] = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest"
            )
            
            logger.info("✅ Modèles de sentiment chargés")
            
        except Exception as e:
            logger.warning(f"Erreur chargement modèles sentiment: {e}")
            # Fallback vers TextBlob
            self.models = {}
    
    async def analyze_sentiment(self, text: str, language: str = 'auto') -> SentimentAnalysis:
        """Analyse le sentiment d'un texte"""
        
        text_id = f"text_{hash(text)}_{int(time.time())}"
        
        if language == 'auto':
            preprocessor = TextPreprocessor()
            language = preprocessor.detect_language(text)
        
        # Nettoyage basique
        cleaned_text = re.sub(r'[^\w\s\.\,\!\?]', ' ', text)
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        
        # Analyse avec modèle approprié
        if language in self.models:
            try:
                result = self.models[language](cleaned_text)[0]
                
                # Mapping des labels
                label_mapping = {
                    'POSITIVE': 1,
                    'NEGATIVE': -1,
                    'NEUTRAL': 0,
                    'LABEL_0': -1,  # Négatif
                    'LABEL_1': 0,   # Neutre  
                    'LABEL_2': 1    # Positif
                }
                
                sentiment_score = label_mapping.get(result['label'], 0) * result['score']
                confidence = result['score']
                
            except Exception as e:
                logger.warning(f"Erreur modèle sentiment: {e}")
                # Fallback TextBlob
                blob = TextBlob(cleaned_text)
                sentiment_score = blob.sentiment.polarity
                confidence = abs(sentiment_score)
        else:
            # Fallback TextBlob
            blob = TextBlob(cleaned_text)
            sentiment_score = blob.sentiment.polarity
            confidence = abs(sentiment_score)
        
        # Classification du sentiment
        if sentiment_score > 0.6:
            sentiment_label = SentimentScore.VERY_POSITIVE
        elif sentiment_score > 0.2:
            sentiment_label = SentimentScore.POSITIVE
        elif sentiment_score > -0.2:
            sentiment_label = SentimentScore.NEUTRAL
        elif sentiment_score > -0.6:
            sentiment_label = SentimentScore.NEGATIVE
        else:
            sentiment_label = SentimentScore.VERY_NEGATIVE
        
        # Analyse des indicateurs émotionnels
        emotional_indicators = self._analyze_emotional_indicators(cleaned_text, language)
        
        # Extraction mots-clés sentiment
        positive_keywords, negative_keywords = self._extract_sentiment_keywords(cleaned_text, language)
        
        # Subjectivité (TextBlob)
        try:
            subjectivity = TextBlob(cleaned_text).sentiment.subjectivity
        except:
            subjectivity = 0.5
        
        return SentimentAnalysis(
            text_id=text_id,
            language=language,
            sentiment_score=sentiment_score,
            sentiment_label=sentiment_label,
            confidence=confidence,
            emotional_indicators=emotional_indicators,
            subjectivity=subjectivity,
            positive_keywords=positive_keywords,
            negative_keywords=negative_keywords
        )
    
    def _analyze_emotional_indicators(self, text: str, language: str) -> Dict[str, float]:
        """Analyse les indicateurs émotionnels"""
        
        # Dictionnaires d'émotions
        emotion_words = {
            'joy': {
                'fr': ['joie', 'heureux', 'content', 'ravi', 'satisfait', 'enthousiaste', 'optimiste'],
                'en': ['joy', 'happy', 'pleased', 'delighted', 'satisfied', 'enthusiastic', 'optimistic']
            },
            'fear': {
                'fr': ['peur', 'inquiet', 'anxieux', 'crainte', 'risque', 'danger', 'menace'],
                'en': ['fear', 'worried', 'anxious', 'concern', 'risk', 'danger', 'threat']
            },
            'anger': {
                'fr': ['colère', 'irrité', 'frustré', 'mécontent', 'outragé'],
                'en': ['anger', 'irritated', 'frustrated', 'upset', 'outraged']
            },
            'trust': {
                'fr': ['confiance', 'fiable', 'sûr', 'crédible', 'solide'],
                'en': ['trust', 'reliable', 'secure', 'credible', 'solid']
            }
        }
        
        text_lower = text.lower()
        indicators = {}
        
        for emotion, words_dict in emotion_words.items():
            words = words_dict.get(language, [])
            count = sum(1 for word in words if word in text_lower)
            
            # Normalisation par longueur du texte
            word_count = len(text_lower.split())
            indicators[emotion] = count / max(word_count, 1) * 100
        
        return indicators
    
    def _extract_sentiment_keywords(self, text: str, language: str) -> Tuple[List[str], List[str]]:
        """Extrait les mots-clés de sentiment positif et négatif"""
        
        positive_words = {
            'fr': ['excellent', 'bon', 'bien', 'succès', 'croissance', 'profit', 'amélioration', 'opportunité'],
            'en': ['excellent', 'good', 'great', 'success', 'growth', 'profit', 'improvement', 'opportunity']
        }
        
        negative_words = {
            'fr': ['mauvais', 'problème', 'échec', 'perte', 'baisse', 'risque', 'difficulté', 'crise'],
            'en': ['bad', 'problem', 'failure', 'loss', 'decline', 'risk', 'difficulty', 'crisis']
        }
        
        text_words = set(text.lower().split())
        
        found_positive = [word for word in positive_words.get(language, []) if word in text_words]
        found_negative = [word for word in negative_words.get(language, []) if word in text_words]
        
        return found_positive, found_negative


class NamedEntityRecognizer:
    """Reconnaissance d'entités nommées"""
    
    def __init__(self):
        self.nlp_models = {}
        self._load_spacy_models()
        
    def _load_spacy_models(self):
        """Charge les modèles spaCy"""
        
        try:
            # Modèle français
            self.nlp_models['fr'] = spacy.load('fr_core_news_sm')
            logger.info("✅ Modèle spaCy français chargé")
        except OSError:
            logger.warning("Modèle spaCy français non disponible")
        
        try:
            # Modèle anglais
            self.nlp_models['en'] = spacy.load('en_core_web_sm')
            logger.info("✅ Modèle spaCy anglais chargé")
        except OSError:
            logger.warning("Modèle spaCy anglais non disponible")
    
    async def extract_entities(self, text: str, language: str = 'fr') -> EntityExtraction:
        """Extrait les entités nommées du texte"""
        
        text_id = f"entities_{hash(text)}_{int(time.time())}"
        
        entities = []
        persons = []
        organizations = []
        locations = []
        financial_amounts = []
        dates = []
        relationships = []
        
        # Utilisation de spaCy si disponible
        if language in self.nlp_models:
            nlp = self.nlp_models[language]
            doc = nlp(text)
            
            for ent in doc.ents:
                entity_info = {
                    'text': ent.text,
                    'label': ent.label_,
                    'confidence': 0.9,  # spaCy ne fournit pas de score de confiance direct
                    'start': ent.start_char,
                    'end': ent.end_char,
                    'description': spacy.explain(ent.label_)
                }
                
                entities.append(entity_info)
                
                # Catégorisation
                if ent.label_ in ['PERSON', 'PER']:
                    persons.append(ent.text)
                elif ent.label_ in ['ORG', 'ORGANIZATION']:
                    organizations.append(ent.text)
                elif ent.label_ in ['GPE', 'LOC', 'LOCATION']:
                    locations.append(ent.text)
                elif ent.label_ in ['MONEY', 'MONETARY']:
                    financial_amounts.append(ent.text)
                elif ent.label_ in ['DATE', 'TIME']:
                    dates.append(ent.text)
            
            # Extraction de relations simples
            relationships = self._extract_relationships(doc)
        
        else:
            # Fallback: extraction avec expressions régulières
            entities, persons, organizations, locations, financial_amounts, dates = self._regex_entity_extraction(text)
        
        # Extraction montants financiers avec préprocesseur
        preprocessor = TextPreprocessor()
        extracted_amounts = preprocessor.extract_financial_amounts(text)
        financial_amounts.extend([amount['original_text'] for amount in extracted_amounts])
        
        return EntityExtraction(
            text_id=text_id,
            entities=entities,
            persons=list(set(persons)),
            organizations=list(set(organizations)),
            locations=list(set(locations)),
            financial_amounts=list(set(financial_amounts)),
            dates=list(set(dates)),
            relationships=relationships
        )
    
    def _extract_relationships(self, doc) -> List[Dict[str, Any]]:
        """Extrait des relations simples entre entités"""
        
        relationships = []
        
        for sent in doc.sents:
            entities_in_sent = [(ent.text, ent.label_) for ent in sent.ents]
            
            # Relations simples: Personne-Organisation
            persons_in_sent = [ent[0] for ent in entities_in_sent if ent[1] in ['PERSON', 'PER']]
            orgs_in_sent = [ent[0] for ent in entities_in_sent if ent[1] in ['ORG', 'ORGANIZATION']]
            
            for person in persons_in_sent:
                for org in orgs_in_sent:
                    relationships.append({
                        'subject': person,
                        'relation': 'ASSOCIATED_WITH',
                        'object': org,
                        'confidence': 0.7,
                        'sentence': sent.text
                    })
        
        return relationships
    
    def _regex_entity_extraction(self, text: str) -> Tuple[List[Dict], List[str], List[str], List[str], List[str], List[str]]:
        """Extraction d'entités avec expressions régulières (fallback)"""
        
        entities = []
        
        # Patterns simples
        person_pattern = r'\b[A-Z][a-z]+ [A-Z][a-z]+\b'
        org_pattern = r'\b[A-Z][A-Za-z\s&\.]+ (SA|SAS|SARL|SNC|EURL|SCI)\b'
        location_pattern = r'\b(Paris|Lyon|Marseille|Toulouse|Nice|Nantes|Strasbourg|Montpellier|Bordeaux|Lille)\b'
        date_pattern = r'\b\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}\b'
        
        persons = re.findall(person_pattern, text)
        organizations = re.findall(org_pattern, text)
        locations = re.findall(location_pattern, text)
        dates = re.findall(date_pattern, text)
        
        # Création des entités
        for person in persons:
            entities.append({
                'text': person,
                'label': 'PERSON',
                'confidence': 0.6,
                'start': text.find(person),
                'end': text.find(person) + len(person)
            })
        
        return entities, persons, organizations, locations, [], dates


class TextClassifier:
    """Classificateur de texte pour catégorisation automatique"""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.classifier = LogisticRegression(max_iter=1000)
        self.is_trained = False
        self.label_encoder = {}
        
    async def train_classifier(self, training_data: List[Tuple[str, TextCategory]]):
        """Entraîne le classificateur"""
        
        if len(training_data) < 10:
            logger.warning("Pas assez de données d'entraînement pour le classificateur")
            return
        
        try:
            texts, labels = zip(*training_data)
            
            # Vectorisation
            X = self.vectorizer.fit_transform(texts)
            
            # Entraînement
            self.classifier.fit(X, labels)
            self.is_trained = True
            
            logger.info(f"✅ Classificateur entraîné avec {len(training_data)} exemples")
            
        except Exception as e:
            logger.error(f"❌ Erreur entraînement classificateur: {e}")
    
    async def classify_text(self, text: str) -> TextClassification:
        """Classifie un texte"""
        
        text_id = f"classification_{hash(text)}_{int(time.time())}"
        
        if not self.is_trained:
            # Classification basée sur mots-clés (fallback)
            return self._keyword_based_classification(text_id, text)
        
        try:
            # Vectorisation
            X = self.vectorizer.transform([text])
            
            # Prédiction
            predicted_proba = self.classifier.predict_proba(X)[0]
            predicted_class = self.classifier.predict(X)[0]
            confidence = max(predicted_proba)
            
            # Probabilités par catégorie
            classes = self.classifier.classes_
            category_probabilities = {
                TextCategory(cls): prob 
                for cls, prob in zip(classes, predicted_proba)
            }
            
            # Features importantes
            feature_names = self.vectorizer.get_feature_names_out()
            feature_importance = abs(self.classifier.coef_[0])
            top_features_idx = feature_importance.argsort()[-10:][::-1]
            
            key_features = [
                (feature_names[idx], feature_importance[idx])
                for idx in top_features_idx
            ]
            
            return TextClassification(
                text_id=text_id,
                predicted_category=TextCategory(predicted_class),
                confidence=confidence,
                category_probabilities=category_probabilities,
                key_features=key_features
            )
            
        except Exception as e:
            logger.error(f"❌ Erreur classification: {e}")
            return self._keyword_based_classification(text_id, text)
    
    def _keyword_based_classification(self, text_id: str, text: str) -> TextClassification:
        """Classification basée sur mots-clés (fallback)"""
        
        text_lower = text.lower()
        
        # Mots-clés par catégorie
        category_keywords = {
            TextCategory.FINANCIAL_NEWS: ['bourse', 'action', 'dividende', 'résultats', 'chiffre affaires'],
            TextCategory.COMPANY_DESCRIPTION: ['entreprise', 'activité', 'secteur', 'fondée', 'spécialisée'],
            TextCategory.PRESS_RELEASE: ['annonce', 'communiqué', 'informe', 'déclare'],
            TextCategory.ANALYST_REPORT: ['analyse', 'recommandation', 'objectif', 'valorisation'],
            TextCategory.REGULATORY_FILING: ['amf', 'déclaration', 'franchissement', 'seuil'],
        }
        
        scores = {}
        for category, keywords in category_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            scores[category] = score / len(keywords)
        
        predicted_category = max(scores, key=scores.get) if scores else TextCategory.OTHER
        confidence = scores.get(predicted_category, 0.1)
        
        category_probabilities = {cat: score for cat, score in scores.items()}
        
        return TextClassification(
            text_id=text_id,
            predicted_category=predicted_category,
            confidence=confidence,
            category_probabilities=category_probabilities,
            key_features=[]
        )


class TopicModeling:
    """Modélisation de topics et clustering sémantique"""
    
    def __init__(self, n_topics: int = 10):
        self.n_topics = n_topics
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.lda_model = LatentDirichletAllocation(n_components=n_topics, random_state=42)
        self.nmf_model = NMF(n_components=n_topics, random_state=42)
        self.is_trained = False
        
    async def train_topic_models(self, texts: List[str]):
        """Entraîne les modèles de topics"""
        
        try:
            # Vectorisation
            X = self.vectorizer.fit_transform(texts)
            
            # Entraînement LDA
            self.lda_model.fit(X)
            
            # Entraînement NMF
            self.nmf_model.fit(X)
            
            self.is_trained = True
            
            logger.info(f"✅ Modèles de topics entraînés avec {len(texts)} documents")
            
        except Exception as e:
            logger.error(f"❌ Erreur entraînement topics: {e}")
    
    async def analyze_topics(self, text: str) -> TopicAnalysis:
        """Analyse les topics d'un texte"""
        
        text_id = f"topics_{hash(text)}_{int(time.time())}"
        
        if not self.is_trained:
            return TopicAnalysis(
                text_id=text_id,
                topics=[],
                dominant_topic=0,
                semantic_cluster=0,
                cluster_description="Non entraîné"
            )
        
        try:
            # Vectorisation
            X = self.vectorizer.transform([text])
            
            # Analyse LDA
            lda_topic_probs = self.lda_model.transform(X)[0]
            dominant_topic = lda_topic_probs.argmax()
            
            # Extraction des topics avec leurs mots-clés
            feature_names = self.vectorizer.get_feature_names_out()
            topics = []
            
            for topic_idx, topic in enumerate(self.lda_model.components_):
                top_words_idx = topic.argsort()[-10:][::-1]
                keywords = [feature_names[i] for i in top_words_idx]
                weight = lda_topic_probs[topic_idx]
                
                topics.append({
                    'topic_id': topic_idx,
                    'keywords': keywords,
                    'weight': float(weight)
                })
            
            # Clustering sémantique simple
            semantic_cluster = dominant_topic % 5  # Groupement en 5 clusters
            cluster_descriptions = [
                "Affaires et Finance",
                "Technologie et Innovation", 
                "Industrie et Manufacturing",
                "Services et Commerce",
                "Gouvernance et Régulation"
            ]
            cluster_description = cluster_descriptions[semantic_cluster]
            
            return TopicAnalysis(
                text_id=text_id,
                topics=topics,
                dominant_topic=dominant_topic,
                semantic_cluster=semantic_cluster,
                cluster_description=cluster_description
            )
            
        except Exception as e:
            logger.error(f"❌ Erreur analyse topics: {e}")
            return TopicAnalysis(
                text_id=text_id,
                topics=[],
                dominant_topic=0,
                semantic_cluster=0,
                cluster_description="Erreur d'analyse"
            )


class RiskAnalyzer:
    """Analyseur de risques basé sur NLP"""
    
    def __init__(self):
        self.risk_keywords = {
            'financial': {
                'high': ['faillite', 'liquidation', 'défaut', 'impayé', 'pertes importantes'],
                'medium': ['endettement', 'cash flow négatif', 'restructuration', 'difficultés'],
                'low': ['bénéfices en baisse', 'marges réduites', 'concurrence']
            },
            'operational': {
                'high': ['arrêt production', 'grève', 'accident grave', 'défaillance système'],
                'medium': ['retards livraison', 'qualité dégradée', 'turnover élevé'],
                'low': ['maintenance', 'formation nécessaire', 'amélioration process']
            },
            'regulatory': {
                'high': ['enquête', 'sanctions', 'non-conformité', 'violation'],
                'medium': ['mise en demeure', 'audit réglementaire', 'changement loi'],
                'low': ['veille réglementaire', 'conformité à vérifier']
            },
            'reputation': {
                'high': ['scandale', 'boycott', 'polémique majeure', 'crise image'],
                'medium': ['critiques presse', 'avis clients négatifs', 'controverse'],
                'low': ['communication à améliorer', 'image à rafraîchir']
            }
        }
    
    async def assess_risk(self, text: str) -> RiskAssessment:
        """Évalue les risques dans un texte"""
        
        text_id = f"risk_{hash(text)}_{int(time.time())}"
        text_lower = text.lower()
        
        # Scores de risque par catégorie
        risk_scores = {
            'financial_risk': 0,
            'operational_risk': 0,
            'regulatory_risk': 0,
            'reputation_risk': 0
        }
        
        risk_indicators = []
        
        # Analyse par catégorie de risque
        for risk_category, levels in self.risk_keywords.items():
            category_score = 0
            
            for level, keywords in levels.items():
                level_weight = {'high': 3, 'medium': 2, 'low': 1}[level]
                
                for keyword in keywords:
                    if keyword in text_lower:
                        category_score += level_weight
                        risk_indicators.append({
                            'keyword': keyword,
                            'category': risk_category,
                            'level': level,
                            'weight': level_weight
                        })
            
            # Normalisation (max 30 pour éviter explosion)
            risk_scores[f'{risk_category}_risk'] = min(category_score * 10, 100)
        
        # Score global de risque
        overall_risk_score = np.mean(list(risk_scores.values()))
        
        # Niveau de risque global
        if overall_risk_score > 80:
            overall_risk_level = RiskLevel.VERY_HIGH
        elif overall_risk_score > 60:
            overall_risk_level = RiskLevel.HIGH
        elif overall_risk_score > 40:
            overall_risk_level = RiskLevel.MEDIUM
        elif overall_risk_score > 20:
            overall_risk_level = RiskLevel.LOW
        else:
            overall_risk_level = RiskLevel.VERY_LOW
        
        # Suggestions de mitigation
        mitigation_suggestions = self._generate_mitigation_suggestions(risk_indicators)
        
        return RiskAssessment(
            text_id=text_id,
            overall_risk_level=overall_risk_level,
            risk_score=overall_risk_score,
            financial_risk=risk_scores['financial_risk'],
            operational_risk=risk_scores['operational_risk'],
            regulatory_risk=risk_scores['regulatory_risk'],
            reputation_risk=risk_scores['reputation_risk'],
            risk_indicators=risk_indicators,
            risk_mitigation_suggestions=mitigation_suggestions
        )
    
    def _generate_mitigation_suggestions(self, risk_indicators: List[Dict[str, Any]]) -> List[str]:
        """Génère des suggestions de mitigation des risques"""
        
        suggestions = []
        
        # Grouper par catégorie
        by_category = defaultdict(list)
        for indicator in risk_indicators:
            by_category[indicator['category']].append(indicator)
        
        for category, indicators in by_category.items():
            if category == 'financial':
                suggestions.append("Effectuer un audit financier approfondi")
                suggestions.append("Analyser la structure de financement")
            elif category == 'operational':
                suggestions.append("Évaluer les processus opérationnels")
                suggestions.append("Vérifier la continuité d'activité")
            elif category == 'regulatory':
                suggestions.append("Audit de conformité réglementaire")
                suggestions.append("Mise à jour des procédures")
            elif category == 'reputation':
                suggestions.append("Analyse de l'image de marque")
                suggestions.append("Plan de communication de crise")
        
        return list(set(suggestions))  # Dédupliquer


class AdvancedNLPEngine:
    """Moteur NLP principal intégrant tous les composants"""
    
    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.sentiment_analyzer = MultilingualSentimentAnalyzer()
        self.entity_recognizer = NamedEntityRecognizer()
        self.text_classifier = TextClassifier()
        self.topic_modeler = TopicModeling()
        self.risk_analyzer = RiskAnalyzer()
        
        # Cache des analyses
        self.analysis_cache: Dict[str, ComprehensiveTextAnalysis] = {}
        
        logger.info("🔤 Moteur NLP avancé initialisé")
    
    @cached('nlp_analysis', ttl_seconds=3600)
    async def analyze_text_comprehensive(self, text: str, text_id: Optional[str] = None) -> ComprehensiveTextAnalysis:
        """Analyse complète d'un texte"""
        
        if not text_id:
            text_id = f"text_{hash(text)}_{int(time.time())}"
        
        try:
            logger.info(f"🔍 Analyse NLP complète: {text_id}")
            
            # Nettoyage et préparation
            cleaned_text = self.preprocessor.clean_text(text, preserve_entities=True)
            language = self.preprocessor.detect_language(text)
            
            # Analyses en parallèle
            sentiment_task = self.sentiment_analyzer.analyze_sentiment(cleaned_text, language)
            entities_task = self.entity_recognizer.extract_entities(text, language)
            classification_task = self.text_classifier.classify_text(cleaned_text)
            topics_task = self.topic_modeler.analyze_topics(cleaned_text)
            risk_task = self.risk_analyzer.assess_risk(text)
            
            # Attendre tous les résultats
            sentiment, entities, classification, topics, risk_assessment = await asyncio.gather(
                sentiment_task, entities_task, classification_task, topics_task, risk_task
            )
            
            # Métriques textuelles
            word_count = len(cleaned_text.split())
            sentence_count = len(re.split(r'[.!?]+', cleaned_text))
            
            # Scores de lisibilité et complexité (simplifiés)
            avg_word_length = np.mean([len(word) for word in cleaned_text.split()])
            avg_sentence_length = word_count / max(sentence_count, 1)
            
            readability_score = max(0, 100 - (avg_word_length * 5 + avg_sentence_length))
            complexity_score = min(100, avg_word_length * 10 + avg_sentence_length * 2)
            
            # Résumé automatique (simple)
            summary = self._generate_summary(cleaned_text, 2)
            
            # Phrases clés
            key_phrases = self._extract_key_phrases(cleaned_text, language)
            
            # Création de l'analyse complète
            comprehensive_analysis = ComprehensiveTextAnalysis(
                text_id=text_id,
                original_text=text,
                cleaned_text=cleaned_text,
                language=language,
                sentiment=sentiment,
                entities=entities,
                classification=classification,
                topics=topics,
                risk_assessment=risk_assessment,
                readability_score=readability_score,
                complexity_score=complexity_score,
                word_count=word_count,
                sentence_count=sentence_count,
                summary=summary,
                key_phrases=key_phrases
            )
            
            # Cache
            self.analysis_cache[text_id] = comprehensive_analysis
            
            logger.info(f"✅ Analyse NLP terminée: {text_id}")
            
            return comprehensive_analysis
            
        except Exception as e:
            logger.error(f"❌ Erreur analyse NLP {text_id}: {e}")
            raise
    
    def _generate_summary(self, text: str, num_sentences: int = 2) -> str:
        """Génère un résumé automatique simple"""
        
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        
        if len(sentences) <= num_sentences:
            return '. '.join(sentences)
        
        # Scoring simple basé sur longueur et position
        sentence_scores = []
        for i, sentence in enumerate(sentences):
            score = len(sentence.split())  # Score basé sur longueur
            if i < len(sentences) * 0.3:  # Bonus pour début de texte
                score *= 1.2
            sentence_scores.append((score, sentence))
        
        # Sélection des meilleures phrases
        sentence_scores.sort(reverse=True, key=lambda x: x[0])
        best_sentences = [s[1] for s in sentence_scores[:num_sentences]]
        
        return '. '.join(best_sentences)
    
    def _extract_key_phrases(self, text: str, language: str) -> List[str]:
        """Extrait les phrases clés du texte"""
        
        # Utilisation simple de TF-IDF pour extraire les termes importants
        try:
            vectorizer = TfidfVectorizer(
                ngram_range=(1, 3),
                max_features=20,
                stop_words='english' if language == 'en' else None
            )
            
            tfidf_matrix = vectorizer.fit_transform([text])
            feature_names = vectorizer.get_feature_names_out()
            tfidf_scores = tfidf_matrix.toarray()[0]
            
            # Trier par score TF-IDF
            phrase_scores = list(zip(feature_names, tfidf_scores))
            phrase_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Retourner top phrases
            return [phrase for phrase, score in phrase_scores[:10] if score > 0]
            
        except Exception as e:
            logger.warning(f"Erreur extraction phrases clés: {e}")
            return []
    
    async def batch_analyze_texts(self, texts: List[Tuple[str, str]]) -> List[ComprehensiveTextAnalysis]:
        """Analyse un lot de textes en parallèle"""
        
        tasks = [
            self.analyze_text_comprehensive(text, text_id)
            for text_id, text in texts
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filtrer les erreurs
        valid_results = [
            result for result in results 
            if isinstance(result, ComprehensiveTextAnalysis)
        ]
        
        logger.info(f"✅ Analyse batch terminée: {len(valid_results)}/{len(texts)} réussies")
        
        return valid_results
    
    def get_nlp_system_status(self) -> Dict[str, Any]:
        """Retourne le statut du système NLP"""
        
        return {
            'cached_analyses': len(self.analysis_cache),
            'models_loaded': {
                'sentiment': len(self.sentiment_analyzer.models) > 0,
                'spacy_fr': 'fr' in self.entity_recognizer.nlp_models,
                'spacy_en': 'en' in self.entity_recognizer.nlp_models,
                'classifier_trained': self.text_classifier.is_trained,
                'topic_model_trained': self.topic_modeler.is_trained
            },
            'supported_languages': ['fr', 'en'],
            'analysis_capabilities': [
                'sentiment_analysis',
                'entity_recognition',
                'text_classification',
                'topic_modeling',
                'risk_assessment',
                'summarization'
            ],
            'system_health': 'operational',
            'last_updated': datetime.now().isoformat()
        }


# Instance globale
_nlp_engine: Optional[AdvancedNLPEngine] = None


async def get_nlp_engine() -> AdvancedNLPEngine:
    """Factory pour obtenir le moteur NLP"""
    global _nlp_engine
    
    if _nlp_engine is None:
        _nlp_engine = AdvancedNLPEngine()
    
    return _nlp_engine


# Fonctions utilitaires

async def analyze_company_text(company_siren: str, text_content: str) -> Dict[str, Any]:
    """Analyse le contenu textuel d'une entreprise"""
    
    nlp_engine = await get_nlp_engine()
    text_id = f"company_{company_siren}"
    
    analysis = await nlp_engine.analyze_text_comprehensive(text_content, text_id)
    
    return {
        'siren': company_siren,
        'text_analysis': {
            'sentiment': {
                'score': analysis.sentiment.sentiment_score,
                'label': analysis.sentiment.sentiment_label.value,
                'confidence': analysis.sentiment.confidence
            },
            'risk_assessment': {
                'overall_level': analysis.risk_assessment.overall_risk_level.value,
                'score': analysis.risk_assessment.risk_score,
                'financial_risk': analysis.risk_assessment.financial_risk,
                'operational_risk': analysis.risk_assessment.operational_risk
            },
            'classification': {
                'category': analysis.classification.predicted_category.value,
                'confidence': analysis.classification.confidence
            },
            'entities': {
                'organizations': analysis.entities.organizations,
                'persons': analysis.entities.persons,
                'locations': analysis.entities.locations,
                'financial_amounts': analysis.entities.financial_amounts
            },
            'summary': analysis.summary,
            'key_phrases': analysis.key_phrases,
            'language': analysis.language,
            'readability_score': analysis.readability_score
        },
        'analysis_timestamp': analysis.analysis_timestamp.isoformat()
    }


async def batch_analyze_companies_text(companies_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Analyse les textes de plusieurs entreprises"""
    
    nlp_engine = await get_nlp_engine()
    
    # Préparation des textes
    texts_to_analyze = []
    for company in companies_data:
        siren = company.get('siren', 'unknown')
        
        # Concaténation des champs textuels
        text_content = ' '.join([
            str(company.get('description', '')),
            str(company.get('activite_principale', '')),
            str(company.get('secteur_activite', '')),
            str(company.get('adresse', ''))
        ])
        
        texts_to_analyze.append((f"company_{siren}", text_content))
    
    # Analyse en batch
    analyses = await nlp_engine.batch_analyze_texts(texts_to_analyze)
    
    # Conversion en format API
    results = []
    for analysis in analyses:
        company_siren = analysis.text_id.replace('company_', '')
        
        results.append({
            'siren': company_siren,
            'sentiment_score': analysis.sentiment.sentiment_score,
            'sentiment_label': analysis.sentiment.sentiment_label.value,
            'risk_score': analysis.risk_assessment.risk_score,
            'risk_level': analysis.risk_assessment.overall_risk_level.value,
            'text_category': analysis.classification.predicted_category.value,
            'dominant_topic': analysis.topics.dominant_topic,
            'readability_score': analysis.readability_score,
            'summary': analysis.summary,
            'analysis_timestamp': analysis.analysis_timestamp.isoformat()
        })
    
    return results