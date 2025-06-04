"""
Moteur de recherche sémantique avancé
US-012: Recherche intelligente et contextuelle pour documents M&A

Ce module fournit:
- Recherche sémantique multi-modale (texte, métadonnées, contexte)
- Compréhension du langage naturel avec NLP avancé
- Filtrage intelligent et suggestions automatiques
- Recherche par similarité et clustering de documents
- Analytics de recherche et optimisation continue
"""

import asyncio
import re
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, Counter
import math

import numpy as np
from sentence_transformers import SentenceTransformer, util
import spacy
from spacy.matcher import Matcher
import torch
from transformers import pipeline, AutoTokenizer, AutoModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import faiss

from app.core.document_storage import DocumentMetadata, DocumentType, SearchResult, get_document_storage
from app.core.logging_system import get_logger, LogCategory
from app.core.cache_manager import get_cache_manager

logger = get_logger("semantic_search", LogCategory.DOCUMENT)


class QueryType(str, Enum):
    """Types de requêtes de recherche"""
    SIMPLE = "simple"
    SEMANTIC = "semantic"
    BOOLEAN = "boolean"
    FUZZY = "fuzzy"
    CONTEXTUAL = "contextual"
    MULTIMODAL = "multimodal"


class SearchMode(str, Enum):
    """Modes de recherche"""
    RELEVANCE = "relevance"
    RECENT = "recent"
    POPULAR = "popular"
    COMPREHENSIVE = "comprehensive"


@dataclass
class QueryIntent:
    """Intention détectée dans une requête"""
    intent_type: str
    confidence: float
    entities: List[Dict[str, Any]] = field(default_factory=list)
    temporal_context: Optional[str] = None
    document_types: List[DocumentType] = field(default_factory=list)
    suggested_filters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchQuery:
    """Requête de recherche enrichie"""
    original_query: str
    processed_query: str
    query_type: QueryType
    intent: Optional[QueryIntent] = None
    filters: Dict[str, Any] = field(default_factory=dict)
    boost_factors: Dict[str, float] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchAnalytics:
    """Analytics de recherche"""
    query: str
    query_type: QueryType
    results_count: int
    execution_time: float
    user_id: Optional[str] = None
    clicked_results: List[str] = field(default_factory=list)
    session_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


class NLPProcessor:
    """Processeur de langage naturel pour l'analyse de requêtes"""
    
    def __init__(self):
        # Charger modèle spaCy français
        try:
            self.nlp = spacy.load("fr_core_news_sm")
        except OSError:
            logger.warning("Modèle spaCy français non trouvé, utilisation de 'en_core_web_sm'")
            self.nlp = spacy.load("en_core_web_sm")
        
        # Matcher pour les patterns M&A
        self.matcher = Matcher(self.nlp.vocab)
        self._setup_patterns()
        
        # Pipeline pour classification d'intention
        self.intent_classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli"
        )
        
        # Labels d'intention pour M&A
        self.intent_labels = [
            "recherche financière",
            "analyse juridique",
            "due diligence",
            "évaluation entreprise",
            "contrats et accords",
            "recherche générale"
        ]
    
    def _setup_patterns(self):
        """Configure les patterns de reconnaissance d'entités M&A"""
        
        # Patterns pour entités financières
        financial_patterns = [
            [{"LOWER": {"IN": ["chiffre", "ca", "revenus", "bénéfice", "ebitda"]}],
            [{"LOWER": "chiffre"}, {"LOWER": "d'affaires"}],
            [{"LOWER": {"IN": ["bilan", "compte", "résultat"]}],
            [{"TEXT": {"REGEX": r"\d+[kKmMgG]?€"}}],  # Montants
        ]
        
        # Patterns pour entités juridiques
        legal_patterns = [
            [{"LOWER": {"IN": ["contrat", "accord", "statuts", "nda", "loi"]}],
            [{"LOWER": "due"}, {"LOWER": "diligence"}],
            [{"LOWER": {"IN": ["sarl", "sas", "sa", "eurl", "sci"]}, {"IS_ALPHA": True}],
        ]
        
        # Patterns temporels
        temporal_patterns = [
            [{"LOWER": {"IN": ["dernière", "derniers", "récent", "récents"]}],
            [{"TEXT": {"REGEX": r"20\d{2}"}}],  # Années
            [{"LOWER": {"IN": ["janvier", "février", "mars", "avril", "mai", "juin",
                              "juillet", "août", "septembre", "octobre", "novembre", "décembre"]}],
        ]
        
        # Ajouter patterns au matcher
        self.matcher.add("FINANCIAL", financial_patterns)
        self.matcher.add("LEGAL", legal_patterns)
        self.matcher.add("TEMPORAL", temporal_patterns)
    
    async def analyze_query(self, query: str) -> QueryIntent:
        """Analyse une requête pour détecter l'intention et les entités"""
        
        try:
            doc = self.nlp(query)
            
            # Classification d'intention
            intent_result = self.intent_classifier(query, self.intent_labels)
            intent_type = intent_result['labels'][0]
            confidence = intent_result['scores'][0]
            
            # Extraction d'entités
            entities = []
            
            # Entités nommées standard
            for ent in doc.ents:
                entities.append({
                    "text": ent.text,
                    "label": ent.label_,
                    "start": ent.start_char,
                    "end": ent.end_char,
                    "type": "named_entity"
                })
            
            # Entités patterns M&A
            matches = self.matcher(doc)
            for match_id, start, end in matches:
                span = doc[start:end]
                entities.append({
                    "text": span.text,
                    "label": self.nlp.vocab.strings[match_id],
                    "start": span.start_char,
                    "end": span.end_char,
                    "type": "pattern_match"
                })
            
            # Détection contexte temporel
            temporal_context = None
            temporal_entities = [e for e in entities if e["label"] == "TEMPORAL"]
            if temporal_entities:
                temporal_context = "recent"
            
            # Suggestion de types de documents
            suggested_types = []
            if "financière" in intent_type or any("FINANCIAL" in e["label"] for e in entities):
                suggested_types.append(DocumentType.FINANCIAL)
            if "juridique" in intent_type or any("LEGAL" in e["label"] for e in entities):
                suggested_types.append(DocumentType.LEGAL)
            if "due diligence" in intent_type:
                suggested_types.append(DocumentType.DUE_DILIGENCE)
            
            return QueryIntent(
                intent_type=intent_type,
                confidence=confidence,
                entities=entities,
                temporal_context=temporal_context,
                document_types=suggested_types
            )
            
        except Exception as e:
            logger.error(f"❌ Erreur analyse NLP: {e}")
            return QueryIntent(
                intent_type="recherche générale",
                confidence=0.5
            )


class SemanticSearchEngine:
    """Moteur de recherche sémantique avancé"""
    
    def __init__(self):
        # Modèles de similarité sémantique
        self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
        self.multilingual_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        
        # Processeur NLP
        self.nlp_processor = NLPProcessor()
        
        # Vectoriseur TF-IDF pour recherche de mots-clés
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=10000,
            stop_words='english',  # Ajouter stop words français si disponible
            ngram_range=(1, 3)
        )
        
        # Index FAISS pour recherche vectorielle rapide
        self.faiss_index = None
        self.document_vectors = []
        self.document_ids = []
        
        # Cache des embeddings
        self.cache = get_cache_manager()
        
        # Analytics
        self.search_analytics: List[SearchAnalytics] = []
        self.query_suggestions = defaultdict(int)
        
        # État d'initialisation
        self.is_initialized = False
    
    async def initialize(self):
        """Initialise le moteur de recherche"""
        try:
            logger.info("🚀 Initialisation du moteur de recherche sémantique...")
            
            # Obtenir le storage de documents
            self.document_storage = await get_document_storage()
            
            # Construire index initial
            await self._build_search_index()
            
            self.is_initialized = True
            logger.info("✅ Moteur de recherche sémantique initialisé")
            
        except Exception as e:
            logger.error(f"❌ Erreur initialisation recherche: {e}")
            raise
    
    async def _build_search_index(self):
        """Construit l'index de recherche à partir des documents existants"""
        try:
            documents = self.document_storage.list_documents(limit=10000)
            
            if not documents:
                logger.info("📄 Aucun document à indexer")
                return
            
            # Extraire textes et métadonnées
            texts = []
            self.document_ids = []
            
            for doc in documents:
                # Texte combiné : titre + description + contenu extrait
                combined_text = " ".join(filter(None, [
                    doc.title or "",
                    doc.description or "",
                    doc.extracted_text or "",
                    " ".join(doc.tags)
                ])).strip()
                
                if combined_text:
                    texts.append(combined_text)
                    self.document_ids.append(doc.document_id)
            
            if not texts:
                logger.info("📄 Aucun texte à indexer")
                return
            
            # Créer embeddings
            logger.info(f"🔍 Génération embeddings pour {len(texts)} documents...")
            embeddings = self.sentence_transformer.encode(texts, convert_to_tensor=False)
            self.document_vectors = embeddings
            
            # Construire index FAISS
            dimension = embeddings.shape[1]
            self.faiss_index = faiss.IndexFlatIP(dimension)  # Inner Product pour cosinus
            
            # Normaliser pour similarité cosinus
            faiss.normalize_L2(embeddings)
            self.faiss_index.add(embeddings)
            
            # Construire index TF-IDF
            self.tfidf_vectorizer.fit(texts)
            
            logger.info(f"✅ Index construit: {len(texts)} documents indexés")
            
        except Exception as e:
            logger.error(f"❌ Erreur construction index: {e}")
            raise
    
    async def search(
        self,
        query: str,
        user_id: str = None,
        mode: SearchMode = SearchMode.RELEVANCE,
        limit: int = 20,
        filters: Dict[str, Any] = None
    ) -> List[SearchResult]:
        """Recherche sémantique avancée"""
        
        start_time = datetime.now()
        
        try:
            if not self.is_initialized:
                await self.initialize()
            
            # Analyser la requête
            query_intent = await self.nlp_processor.analyze_query(query)
            
            # Créer objet requête enrichi
            search_query = SearchQuery(
                original_query=query,
                processed_query=self._preprocess_query(query),
                query_type=self._detect_query_type(query),
                intent=query_intent,
                filters=filters or {}
            )
            
            # Appliquer filtres suggérés par l'intention
            if query_intent.document_types:
                search_query.filters["document_types"] = [dt.value for dt in query_intent.document_types]
            
            # Recherche selon le mode
            if mode == SearchMode.SEMANTIC:
                results = await self._semantic_search(search_query, limit)
            elif mode == SearchMode.BOOLEAN:
                results = await self._boolean_search(search_query, limit)
            elif mode == SearchMode.COMPREHENSIVE:
                results = await self._comprehensive_search(search_query, limit)
            else:
                results = await self._hybrid_search(search_query, limit)
            
            # Filtrer selon permissions utilisateur
            if user_id:
                results = await self._apply_user_permissions(results, user_id)
            
            # Appliquer filtres additionnels
            results = self._apply_filters(results, search_query.filters)
            
            # Réordonner selon le mode
            results = self._rerank_results(results, mode, search_query)
            
            # Enregistrer analytics
            execution_time = (datetime.now() - start_time).total_seconds()
            await self._record_search_analytics(search_query, results, execution_time, user_id)
            
            logger.info(f"🔍 Recherche: '{query}' -> {len(results)} résultats ({execution_time:.3f}s)")
            
            return results[:limit]
            
        except Exception as e:
            logger.error(f"❌ Erreur recherche: {e}")
            return []
    
    def _preprocess_query(self, query: str) -> str:
        """Préprocesse une requête de recherche"""
        
        # Nettoyer et normaliser
        query = query.strip().lower()
        
        # Remplacer synonymes M&A
        synonyms = {
            "ca": "chiffre d'affaires",
            "m&a": "fusion acquisition",
            "dd": "due diligence",
            "ceo": "directeur général",
            "cfo": "directeur financier"
        }
        
        for synonym, replacement in synonyms.items():
            query = query.replace(synonym, replacement)
        
        return query
    
    def _detect_query_type(self, query: str) -> QueryType:
        """Détecte le type de requête"""
        
        # Opérateurs booléens
        if any(op in query.upper() for op in ["AND", "OR", "NOT", "+", "-"]):
            return QueryType.BOOLEAN
        
        # Recherche par phrases exactes
        if '"' in query:
            return QueryType.SIMPLE
        
        # Wildcards
        if any(char in query for char in ["*", "?"]):
            return QueryType.FUZZY
        
        # Par défaut, recherche sémantique
        return QueryType.SEMANTIC
    
    async def _semantic_search(self, search_query: SearchQuery, limit: int) -> List[SearchResult]:
        """Recherche sémantique pure"""
        
        if not self.faiss_index or not self.document_ids:
            return []
        
        try:
            # Encoder la requête
            query_embedding = self.sentence_transformer.encode([search_query.processed_query])
            faiss.normalize_L2(query_embedding)
            
            # Rechercher dans l'index
            scores, indices = self.faiss_index.search(query_embedding, min(limit * 2, len(self.document_ids)))
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.document_ids):
                    doc_id = self.document_ids[idx]
                    metadata = self.document_storage.get_document_metadata(doc_id)
                    
                    if metadata:
                        results.append(SearchResult(
                            document_id=doc_id,
                            metadata=metadata,
                            relevance_score=float(score)
                        ))
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Erreur recherche sémantique: {e}")
            return []
    
    async def _boolean_search(self, search_query: SearchQuery, limit: int) -> List[SearchResult]:
        """Recherche booléenne avec opérateurs"""
        
        # Implémentation simplifiée pour démo
        # Dans un vrai système, utiliser Elasticsearch ou Solr
        
        query = search_query.processed_query.upper()
        terms = re.split(r'\s+(?:AND|OR|NOT)\s+|\s+', query)
        terms = [t.strip().lower() for t in terms if t.strip()]
        
        results = []
        documents = self.document_storage.list_documents(limit=1000)
        
        for doc in documents:
            text = " ".join(filter(None, [
                doc.title or "",
                doc.description or "",
                doc.extracted_text or ""
            ])).lower()
            
            # Score basé sur présence des termes
            score = sum(1 for term in terms if term in text) / len(terms) if terms else 0
            
            if score > 0:
                results.append(SearchResult(
                    document_id=doc.document_id,
                    metadata=doc,
                    relevance_score=score
                ))
        
        return sorted(results, key=lambda x: x.relevance_score, reverse=True)[:limit]
    
    async def _comprehensive_search(self, search_query: SearchQuery, limit: int) -> List[SearchResult]:
        """Recherche comprehensive combinant plusieurs méthodes"""
        
        # Combiner recherche sémantique et mots-clés
        semantic_results = await self._semantic_search(search_query, limit)
        boolean_results = await self._boolean_search(search_query, limit)
        
        # Fusionner et déduplicater
        combined_results = {}
        
        # Ajouter résultats sémantiques avec poids élevé
        for result in semantic_results:
            combined_results[result.document_id] = SearchResult(
                document_id=result.document_id,
                metadata=result.metadata,
                relevance_score=result.relevance_score * 0.7
            )
        
        # Ajouter résultats booléens avec poids moyen
        for result in boolean_results:
            if result.document_id in combined_results:
                # Combiner scores
                existing = combined_results[result.document_id]
                combined_results[result.document_id] = SearchResult(
                    document_id=result.document_id,
                    metadata=result.metadata,
                    relevance_score=existing.relevance_score + (result.relevance_score * 0.3)
                )
            else:
                combined_results[result.document_id] = SearchResult(
                    document_id=result.document_id,
                    metadata=result.metadata,
                    relevance_score=result.relevance_score * 0.3
                )
        
        # Retourner triés par score
        results = list(combined_results.values())
        return sorted(results, key=lambda x: x.relevance_score, reverse=True)[:limit]
    
    async def _hybrid_search(self, search_query: SearchQuery, limit: int) -> List[SearchResult]:
        """Recherche hybride optimisée"""
        
        # Pour l'instant, utilise la recherche comprehensive
        # À terme, pourrait inclure d'autres méthodes (graph search, etc.)
        return await self._comprehensive_search(search_query, limit)
    
    async def _apply_user_permissions(self, results: List[SearchResult], user_id: str) -> List[SearchResult]:
        """Applique les permissions utilisateur"""
        
        filtered_results = []
        
        for result in results:
            metadata = result.metadata
            
            # Vérifier permissions basiques
            if (metadata.access_level.value == "public" or
                metadata.owner_id == user_id or
                user_id in metadata.allowed_users):
                filtered_results.append(result)
        
        return filtered_results
    
    def _apply_filters(self, results: List[SearchResult], filters: Dict[str, Any]) -> List[SearchResult]:
        """Applique les filtres de recherche"""
        
        if not filters:
            return results
        
        filtered_results = []
        
        for result in results:
            metadata = result.metadata
            include = True
            
            # Filtre par type de document
            if "document_types" in filters:
                if metadata.document_type.value not in filters["document_types"]:
                    include = False
            
            # Filtre par date
            if "date_from" in filters:
                date_from = datetime.fromisoformat(filters["date_from"])
                if metadata.created_at < date_from:
                    include = False
            
            if "date_to" in filters:
                date_to = datetime.fromisoformat(filters["date_to"])
                if metadata.created_at > date_to:
                    include = False
            
            # Filtre par taille
            if "min_size" in filters:
                if metadata.file_size < filters["min_size"]:
                    include = False
            
            if "max_size" in filters:
                if metadata.file_size > filters["max_size"]:
                    include = False
            
            # Filtre par tags
            if "tags" in filters:
                required_tags = set(filters["tags"])
                document_tags = set(metadata.tags)
                if not required_tags.intersection(document_tags):
                    include = False
            
            if include:
                filtered_results.append(result)
        
        return filtered_results
    
    def _rerank_results(
        self,
        results: List[SearchResult],
        mode: SearchMode,
        search_query: SearchQuery
    ) -> List[SearchResult]:
        """Réordonne les résultats selon le mode"""
        
        if mode == SearchMode.RECENT:
            # Privilégier documents récents
            for result in results:
                recency_boost = 1.0
                days_old = (datetime.now() - result.metadata.created_at).days
                if days_old < 7:
                    recency_boost = 1.5
                elif days_old < 30:
                    recency_boost = 1.2
                elif days_old < 90:
                    recency_boost = 1.1
                
                result.relevance_score *= recency_boost
        
        elif mode == SearchMode.POPULAR:
            # Privilégier documents populaires
            for result in results:
                popularity_boost = 1.0 + (result.metadata.view_count / 100)
                result.relevance_score *= popularity_boost
        
        # Boost basé sur l'intention détectée
        if search_query.intent:
            for result in results:
                if search_query.intent.document_types:
                    if result.metadata.document_type in search_query.intent.document_types:
                        result.relevance_score *= 1.3
        
        return sorted(results, key=lambda x: x.relevance_score, reverse=True)
    
    async def _record_search_analytics(
        self,
        search_query: SearchQuery,
        results: List[SearchResult],
        execution_time: float,
        user_id: str = None
    ):
        """Enregistre les analytics de recherche"""
        
        analytics = SearchAnalytics(
            query=search_query.original_query,
            query_type=search_query.query_type,
            results_count=len(results),
            execution_time=execution_time,
            user_id=user_id
        )
        
        self.search_analytics.append(analytics)
        
        # Garder seulement les 1000 dernières recherches
        if len(self.search_analytics) > 1000:
            self.search_analytics = self.search_analytics[-1000:]
        
        # Mettre à jour suggestions
        self.query_suggestions[search_query.processed_query] += 1
    
    async def get_search_suggestions(self, partial_query: str, limit: int = 5) -> List[str]:
        """Génère des suggestions de recherche"""
        
        partial_lower = partial_query.lower()
        
        # Suggestions basées sur l'historique
        suggestions = []
        for query, count in self.query_suggestions.most_common():
            if partial_lower in query.lower() and query not in suggestions:
                suggestions.append(query)
                if len(suggestions) >= limit:
                    break
        
        # Compléter avec suggestions génériques M&A si nécessaire
        if len(suggestions) < limit:
            generic_suggestions = [
                "bilans financiers",
                "contrats de vente",
                "due diligence",
                "évaluation entreprise",
                "analyse concurrentielle",
                "comptes de résultat",
                "accords de confidentialité"
            ]
            
            for suggestion in generic_suggestions:
                if partial_lower in suggestion and suggestion not in suggestions:
                    suggestions.append(suggestion)
                    if len(suggestions) >= limit:
                        break
        
        return suggestions[:limit]
    
    async def get_related_documents(self, document_id: str, limit: int = 5) -> List[SearchResult]:
        """Trouve des documents similaires"""
        
        try:
            metadata = self.document_storage.get_document_metadata(document_id)
            if not metadata:
                return []
            
            # Utiliser le texte du document pour trouver des similaires
            query_text = " ".join(filter(None, [
                metadata.title or "",
                metadata.description or "",
                " ".join(metadata.tags)
            ]))
            
            if not query_text.strip():
                return []
            
            # Recherche sémantique
            results = await self._semantic_search(
                SearchQuery(
                    original_query=query_text,
                    processed_query=query_text,
                    query_type=QueryType.SEMANTIC
                ),
                limit + 1  # +1 pour exclure le document original
            )
            
            # Exclure le document original
            related = [r for r in results if r.document_id != document_id]
            
            return related[:limit]
            
        except Exception as e:
            logger.error(f"❌ Erreur recherche documents similaires: {e}")
            return []
    
    def get_search_analytics(self) -> Dict[str, Any]:
        """Retourne les analytics de recherche"""
        
        if not self.search_analytics:
            return {
                "total_searches": 0,
                "average_execution_time": 0,
                "most_popular_queries": [],
                "search_trends": {}
            }
        
        total_searches = len(self.search_analytics)
        avg_execution_time = sum(a.execution_time for a in self.search_analytics) / total_searches
        
        # Requêtes populaires
        query_counts = Counter(a.query for a in self.search_analytics)
        popular_queries = [{"query": q, "count": c} for q, c in query_counts.most_common(10)]
        
        # Tendances par jour
        daily_counts = defaultdict(int)
        for analytics in self.search_analytics:
            day = analytics.timestamp.date().isoformat()
            daily_counts[day] += 1
        
        return {
            "total_searches": total_searches,
            "average_execution_time": round(avg_execution_time, 3),
            "most_popular_queries": popular_queries,
            "search_trends": dict(daily_counts)
        }


# Instance globale
_semantic_search_engine: Optional[SemanticSearchEngine] = None


async def get_semantic_search_engine() -> SemanticSearchEngine:
    """Factory pour obtenir le moteur de recherche sémantique"""
    global _semantic_search_engine
    
    if _semantic_search_engine is None:
        _semantic_search_engine = SemanticSearchEngine()
        await _semantic_search_engine.initialize()
    
    return _semantic_search_engine