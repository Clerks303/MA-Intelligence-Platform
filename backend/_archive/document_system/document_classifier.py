"""
Système de classification automatique de documents
US-012: Classification intelligente et tagging automatique pour documents M&A

Ce module fournit:
- Classification automatique par type de document M&A
- Extraction et tagging automatique de métadonnées
- Détection de sensibilité et niveau de confidentialité
- Classification par contenu et structure
- Apprentissage continu des patterns de classification
"""

import asyncio
import re
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, Counter
import string

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import spacy
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import joblib

from app.core.document_storage import DocumentMetadata, DocumentType, AccessLevel
from app.core.logging_system import get_logger, LogCategory

logger = get_logger("document_classifier", LogCategory.DOCUMENT)


class ConfidentialityLevel(str, Enum):
    """Niveaux de confidentialité détectés"""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    HIGHLY_CONFIDENTIAL = "highly_confidential"
    TOP_SECRET = "top_secret"


class SensitivityType(str, Enum):
    """Types de sensibilité des données"""
    FINANCIAL = "financial"
    PERSONAL = "personal"
    LEGAL = "legal"
    STRATEGIC = "strategic"
    TECHNICAL = "technical"
    NONE = "none"


@dataclass
class ClassificationResult:
    """Résultat de classification d'un document"""
    
    # Type de document prédit
    predicted_type: DocumentType
    confidence: float
    
    # Confidentialité et sensibilité
    confidentiality_level: ConfidentialityLevel
    sensitivity_types: List[SensitivityType]
    
    # Tags automatiques
    extracted_tags: List[str]
    
    # Métadonnées extraites
    extracted_title: Optional[str] = None
    extracted_entities: List[Dict[str, Any]] = field(default_factory=list)
    
    # Indicateurs de qualité
    ocr_quality_score: Optional[float] = None
    text_clarity_score: Optional[float] = None
    
    # Détails de classification
    classification_features: Dict[str, float] = field(default_factory=dict)
    alternative_types: List[Tuple[DocumentType, float]] = field(default_factory=list)


class DocumentFeatureExtractor:
    """Extracteur de features pour classification"""
    
    def __init__(self):
        # Patterns spécialisés M&A
        self.ma_patterns = {
            DocumentType.FINANCIAL: [
                r'bilan|compte[s]?\s+de\s+résultat|chiffre[s]?\s+d[\'']affaires',
                r'ebitda|ebit|roi|roe|cash[- ]flow',
                r'actif|passif|capitaux propres|dettes',
                r'revenus|charges|bénéfice|perte',
                r'ratio[s]?|liquidité|solvabilité',
                r'budget|prévision[s]?|forecast'
            ],
            DocumentType.LEGAL: [
                r'contrat|accord|convention|avenant',
                r'statuts|procès[- ]verbal|assemblée générale',
                r'clause|article|conditions générales',
                r'nda|confidentialité|non[- ]disclosure',
                r'droit[s]?|loi|réglementation|juridique',
                r'responsabilité|garantie|assurance'
            ],
            DocumentType.DUE_DILIGENCE: [
                r'due\s+diligence|audit|évaluation',
                r'analyse\s+de\s+risque[s]?|risk\s+assessment',
                r'compliance|conformité|réglementation',
                r'recommandation[s]?|rapport\s+d[\'']expertise',
                r'vérification|contrôle|inspection',
                r'benchmark|comparaison\s+concurrentielle'
            ],
            DocumentType.COMMUNICATION: [
                r'lettre|courrier|correspondance|email',
                r'mémo|note\s+interne|communication',
                r'présentation|rapport\s+de\s+réunion',
                r'proposition|offre|devis',
                r'négociation|discussion|échange',
                r'accord\s+de\s+principe|intention'
            ],
            DocumentType.TECHNICAL: [
                r'spécification[s]?|cahier\s+des\s+charges',
                r'architecture|infrastructure|système[s]?',
                r'procédure[s]?|processus|méthode[s]?',
                r'manuel|guide|documentation\s+technique',
                r'api|interface|intégration',
                r'sécurité\s+informatique|cybersécurité'
            ],
            DocumentType.HR: [
                r'ressources\s+humaines|rh|personnel',
                r'organigramme|effectif[s]?|employé[s]?',
                r'salaire[s]?|rémunération|avantages',
                r'formation|compétence[s]?|qualification[s]?',
                r'recrutement|embauche|licenciement',
                r'convention\s+collective|accord\s+d[\'']entreprise'
            ],
            DocumentType.COMMERCIAL: [
                r'commercial|vente[s]?|chiffre[s]?\s+d[\'']affaires',
                r'client[s]?|prospect[s]?|marché[s]?',
                r'stratégie\s+commerciale|plan\s+commercial',
                r'tarif[s]?|prix|catalogue|offre[s]?',
                r'distribution|canal|partenaire[s]?',
                r'marketing|communication|publicité'
            ]
        }
        
        # Patterns de confidentialité
        self.confidentiality_patterns = {
            ConfidentialityLevel.PUBLIC: [
                r'public|diffusion\s+libre|communication\s+externe'
            ],
            ConfidentialityLevel.INTERNAL: [
                r'interne|usage\s+interne|diffusion\s+restreinte'
            ],
            ConfidentialityLevel.CONFIDENTIAL: [
                r'confidentiel|strictement\s+confidentiel',
                r'ne\s+pas\s+diffuser|diffusion\s+interdite',
                r'secret\s+professionnel|confidentiel\s+défense'
            ],
            ConfidentialityLevel.HIGHLY_CONFIDENTIAL: [
                r'très\s+confidentiel|hautement\s+confidentiel',
                r'top\s+secret|ultra\s+confidentiel',
                r'secret\s+défense|classification\s+spéciale'
            ]
        }
        
        # Mots-clés de sensibilité
        self.sensitivity_keywords = {
            SensitivityType.FINANCIAL: [
                'prix', 'coût', 'budget', 'financier', 'euro', 'dollar',
                'millions', 'milliards', 'investissement', 'profit'
            ],
            SensitivityType.PERSONAL: [
                'nom', 'prénom', 'adresse', 'téléphone', 'email', 'personnel',
                'privé', 'individuel', 'identité', 'données personnelles'
            ],
            SensitivityType.LEGAL: [
                'contrat', 'juridique', 'loi', 'réglementation', 'tribunal',
                'avocat', 'légal', 'droit', 'clause', 'responsabilité'
            ],
            SensitivityType.STRATEGIC: [
                'stratégie', 'concurrentiel', 'avantage', 'position',
                'marché', 'opportunité', 'menace', 'innovation'
            ],
            SensitivityType.TECHNICAL: [
                'technique', 'technologie', 'système', 'méthode',
                'processus', 'savoir-faire', 'expertise', 'propriété intellectuelle'
            ]
        }
        
        # Charger modèle spaCy
        try:
            self.nlp = spacy.load("fr_core_news_sm")
        except OSError:
            logger.warning("Modèle spaCy français non trouvé, utilisation de l'anglais")
            self.nlp = spacy.load("en_core_web_sm")
    
    def extract_features(self, text: str, filename: str = "") -> Dict[str, Any]:
        """Extrait les features d'un document"""
        
        try:
            # Préprocessing
            text_lower = text.lower()
            text_clean = self._clean_text(text)
            
            # Features basiques
            features = {
                'char_count': len(text),
                'word_count': len(text.split()),
                'sentence_count': len(text.split('.')),
                'paragraph_count': len(text.split('\n\n')),
                'avg_word_length': np.mean([len(word) for word in text.split()]) if text.split() else 0,
                'punctuation_ratio': len([c for c in text if c in string.punctuation]) / max(len(text), 1)
            }
            
            # Features du nom de fichier
            filename_lower = filename.lower()
            features.update({
                'filename_has_financial': any(word in filename_lower for word in ['bilan', 'finance', 'budget', 'ca']),
                'filename_has_legal': any(word in filename_lower for word in ['contrat', 'accord', 'legal', 'juridique']),
                'filename_has_dd': any(word in filename_lower for word in ['dd', 'due_diligence', 'audit']),
                'filename_has_date': bool(re.search(r'\d{2,4}[-/]\d{1,2}[-/]\d{2,4}', filename)),
                'filename_has_version': bool(re.search(r'v\d+|version', filename_lower))
            })
            
            # Features de contenu par type
            for doc_type, patterns in self.ma_patterns.items():
                feature_name = f'content_{doc_type.value}_score'
                score = 0
                for pattern in patterns:
                    matches = len(re.findall(pattern, text_lower))
                    score += matches
                features[feature_name] = score / max(len(text.split()), 1)  # Normaliser par longueur
            
            # Features de confidentialité
            for conf_level, patterns in self.confidentiality_patterns.items():
                feature_name = f'confidentiality_{conf_level.value}_score'
                score = 0
                for pattern in patterns:
                    matches = len(re.findall(pattern, text_lower))
                    score += matches
                features[feature_name] = score
            
            # Features de sensibilité
            for sens_type, keywords in self.sensitivity_keywords.items():
                feature_name = f'sensitivity_{sens_type.value}_score'
                score = sum(1 for keyword in keywords if keyword in text_lower)
                features[feature_name] = score / max(len(keywords), 1)
            
            # Features linguistiques avec spaCy
            if len(text) < 100000:  # Éviter les textes trop longs
                doc = self.nlp(text[:50000])  # Limiter pour performance
                
                features.update({
                    'entity_count': len(doc.ents),
                    'person_count': len([ent for ent in doc.ents if ent.label_ == "PERSON"]),
                    'org_count': len([ent for ent in doc.ents if ent.label_ == "ORG"]),
                    'money_count': len([ent for ent in doc.ents if ent.label_ == "MONEY"]),
                    'date_count': len([ent for ent in doc.ents if ent.label_ == "DATE"]),
                    'noun_ratio': len([token for token in doc if token.pos_ == "NOUN"]) / max(len(doc), 1),
                    'verb_ratio': len([token for token in doc if token.pos_ == "VERB"]) / max(len(doc), 1)
                })
            
            return features
            
        except Exception as e:
            logger.error(f"❌ Erreur extraction features: {e}")
            return {}
    
    def _clean_text(self, text: str) -> str:
        """Nettoie le texte pour l'analyse"""
        # Supprimer caractères de contrôle
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x84\x86-\x9f]', ' ', text)
        # Normaliser espaces
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extrait les entités nommées"""
        
        try:
            if len(text) > 100000:
                text = text[:100000]  # Limiter pour performance
            
            doc = self.nlp(text)
            entities = []
            
            for ent in doc.ents:
                entities.append({
                    'text': ent.text,
                    'label': ent.label_,
                    'start': ent.start_char,
                    'end': ent.end_char,
                    'confidence': float(ent._.get('confidence', 0.8))  # Confidence par défaut
                })
            
            return entities
            
        except Exception as e:
            logger.error(f"❌ Erreur extraction entités: {e}")
            return []
    
    def extract_title(self, text: str) -> Optional[str]:
        """Extrait un titre potentiel du document"""
        
        lines = text.split('\n')
        
        # Chercher première ligne non vide et relativement courte
        for line in lines[:10]:  # Chercher dans les 10 premières lignes
            line = line.strip()
            if line and len(line) < 200 and len(line) > 10:
                # Vérifier si ça ressemble à un titre
                if not line.endswith('.') and not line.startswith('http'):
                    return line
        
        return None


class MLDocumentClassifier:
    """Classificateur ML pour documents"""
    
    def __init__(self):
        self.feature_extractor = DocumentFeatureExtractor()
        
        # Modèles ML
        self.type_classifier = None
        self.confidentiality_classifier = None
        
        # Vectoriseurs
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 3),
            stop_words='english'  # À remplacer par stop words français
        )
        
        # Données d'entraînement
        self.training_data = []
        self.is_trained = False
        
        # Pipeline de classification de transformers pour backup
        self.transformer_classifier = None
        self._setup_transformer_classifier()
    
    def _setup_transformer_classifier(self):
        """Configure le classificateur transformer de backup"""
        try:
            self.transformer_classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli"
            )
        except Exception as e:
            logger.warning(f"⚠️ Erreur configuration transformer: {e}")
    
    async def train_models(self, training_documents: List[Tuple[str, str, DocumentMetadata]]):
        """Entraîne les modèles de classification"""
        
        try:
            logger.info(f"🏋️ Entraînement des modèles avec {len(training_documents)} documents...")
            
            if len(training_documents) < 10:
                logger.warning("⚠️ Pas assez de données d'entraînement")
                return
            
            # Extraction des features
            X_features = []
            X_text = []
            y_type = []
            y_confidentiality = []
            
            for text, filename, metadata in training_documents:
                features = self.feature_extractor.extract_features(text, filename)
                
                if features:
                    X_features.append(list(features.values()))
                    X_text.append(text)
                    y_type.append(metadata.document_type.value)
                    y_confidentiality.append(metadata.access_level.value)
            
            if not X_features:
                logger.error("❌ Aucune feature extraite pour l'entraînement")
                return
            
            X_features = np.array(X_features)
            
            # Entraîner vectoriseur TF-IDF
            self.tfidf_vectorizer.fit(X_text)
            X_tfidf = self.tfidf_vectorizer.transform(X_text)
            
            # Combiner features
            X_combined = np.hstack([X_features, X_tfidf.toarray()])
            
            # Entraîner classificateur de type
            self.type_classifier = RandomForestClassifier(
                n_estimators=100,
                random_state=42
            )
            self.type_classifier.fit(X_combined, y_type)
            
            # Entraîner classificateur de confidentialité
            self.confidentiality_classifier = LogisticRegression(
                random_state=42,
                max_iter=1000
            )
            self.confidentiality_classifier.fit(X_combined, y_confidentiality)
            
            self.is_trained = True
            
            # Évaluation basique
            if len(set(y_type)) > 1:
                X_train, X_test, y_train, y_test = train_test_split(
                    X_combined, y_type, test_size=0.2, random_state=42
                )
                
                test_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
                test_classifier.fit(X_train, y_train)
                
                predictions = test_classifier.predict(X_test)
                accuracy = sum(p == t for p, t in zip(predictions, y_test)) / len(y_test)
                
                logger.info(f"✅ Modèles entraînés - Précision type: {accuracy:.2%}")
            
        except Exception as e:
            logger.error(f"❌ Erreur entraînement modèles: {e}")
    
    async def classify_document(
        self,
        text: str,
        filename: str = "",
        use_transformer_backup: bool = True
    ) -> ClassificationResult:
        """Classifie un document"""
        
        try:
            # Extraction features
            features = self.feature_extractor.extract_features(text, filename)
            entities = self.feature_extractor.extract_entities(text)
            title = self.feature_extractor.extract_title(text)
            
            # Classification du type de document
            if self.is_trained and self.type_classifier:
                predicted_type, confidence = await self._ml_classify_type(text, features)
            elif use_transformer_backup and self.transformer_classifier:
                predicted_type, confidence = await self._transformer_classify_type(text)
            else:
                predicted_type, confidence = self._rule_based_classify_type(text, features)
            
            # Classification de la confidentialité
            confidentiality_level = self._classify_confidentiality(text, features)
            
            # Détection de sensibilité
            sensitivity_types = self._detect_sensitivity(text, features)
            
            # Génération de tags automatiques
            tags = self._generate_tags(text, entities, features)
            
            # Scores de qualité
            text_clarity_score = self._calculate_text_clarity(text)
            
            result = ClassificationResult(
                predicted_type=predicted_type,
                confidence=confidence,
                confidentiality_level=confidentiality_level,
                sensitivity_types=sensitivity_types,
                extracted_tags=tags,
                extracted_title=title,
                extracted_entities=entities,
                text_clarity_score=text_clarity_score,
                classification_features=features
            )
            
            logger.info(
                f"📊 Document classifié: {predicted_type.value} "
                f"(conf: {confidence:.2f}, sensibilité: {len(sensitivity_types)})"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Erreur classification: {e}")
            return ClassificationResult(
                predicted_type=DocumentType.OTHER,
                confidence=0.0,
                confidentiality_level=ConfidentialityLevel.INTERNAL,
                sensitivity_types=[],
                extracted_tags=[]
            )
    
    async def _ml_classify_type(self, text: str, features: Dict[str, Any]) -> Tuple[DocumentType, float]:
        """Classification ML du type de document"""
        
        try:
            X_features = np.array([list(features.values())])
            X_tfidf = self.tfidf_vectorizer.transform([text])
            X_combined = np.hstack([X_features, X_tfidf.toarray()])
            
            # Prédiction avec probabilités
            probabilities = self.type_classifier.predict_proba(X_combined)[0]
            classes = self.type_classifier.classes_
            
            # Meilleure prédiction
            best_idx = np.argmax(probabilities)
            predicted_type = DocumentType(classes[best_idx])
            confidence = probabilities[best_idx]
            
            return predicted_type, confidence
            
        except Exception as e:
            logger.error(f"❌ Erreur classification ML: {e}")
            return DocumentType.OTHER, 0.0
    
    async def _transformer_classify_type(self, text: str) -> Tuple[DocumentType, float]:
        """Classification avec transformer (backup)"""
        
        try:
            # Labels pour classification zero-shot
            type_labels = [
                "document financier",
                "document juridique",
                "due diligence",
                "communication",
                "document technique",
                "ressources humaines",
                "document commercial",
                "autre document"
            ]
            
            # Classification
            result = self.transformer_classifier(text[:1000], type_labels)  # Limiter pour performance
            
            # Mapper vers DocumentType
            label_mapping = {
                "document financier": DocumentType.FINANCIAL,
                "document juridique": DocumentType.LEGAL,
                "due diligence": DocumentType.DUE_DILIGENCE,
                "communication": DocumentType.COMMUNICATION,
                "document technique": DocumentType.TECHNICAL,
                "ressources humaines": DocumentType.HR,
                "document commercial": DocumentType.COMMERCIAL,
                "autre document": DocumentType.OTHER
            }
            
            predicted_label = result['labels'][0]
            confidence = result['scores'][0]
            predicted_type = label_mapping.get(predicted_label, DocumentType.OTHER)
            
            return predicted_type, confidence
            
        except Exception as e:
            logger.error(f"❌ Erreur classification transformer: {e}")
            return DocumentType.OTHER, 0.0
    
    def _rule_based_classify_type(self, text: str, features: Dict[str, Any]) -> Tuple[DocumentType, float]:
        """Classification basée sur des règles (fallback)"""
        
        text_lower = text.lower()
        scores = {}
        
        # Calculer scores pour chaque type
        for doc_type, patterns in self.feature_extractor.ma_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, text_lower))
                score += matches
            scores[doc_type] = score
        
        # Normaliser par longueur du texte
        text_length = len(text.split())
        for doc_type in scores:
            scores[doc_type] = scores[doc_type] / max(text_length, 1)
        
        # Meilleur score
        if scores:
            best_type = max(scores, key=scores.get)
            confidence = min(scores[best_type] * 10, 1.0)  # Normaliser à [0, 1]
            
            if confidence > 0.1:
                return best_type, confidence
        
        return DocumentType.OTHER, 0.1
    
    def _classify_confidentiality(self, text: str, features: Dict[str, Any]) -> ConfidentialityLevel:
        """Classifie le niveau de confidentialité"""
        
        text_lower = text.lower()
        
        # Vérifier patterns de confidentialité
        for level, patterns in self.feature_extractor.confidentiality_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    return level
        
        # Heuristiques basées sur le contenu
        sensitive_count = sum(features.get(f'sensitivity_{sens_type.value}_score', 0) 
                            for sens_type in SensitivityType if sens_type != SensitivityType.NONE)
        
        if sensitive_count > 0.5:
            return ConfidentialityLevel.CONFIDENTIAL
        elif sensitive_count > 0.2:
            return ConfidentialityLevel.INTERNAL
        else:
            return ConfidentialityLevel.INTERNAL  # Par défaut
    
    def _detect_sensitivity(self, text: str, features: Dict[str, Any]) -> List[SensitivityType]:
        """Détecte les types de sensibilité"""
        
        sensitivity_types = []
        
        for sens_type, keywords in self.feature_extractor.sensitivity_keywords.items():
            if sens_type == SensitivityType.NONE:
                continue
            
            score = features.get(f'sensitivity_{sens_type.value}_score', 0)
            if score > 0.1:  # Seuil de détection
                sensitivity_types.append(sens_type)
        
        return sensitivity_types if sensitivity_types else [SensitivityType.NONE]
    
    def _generate_tags(self, text: str, entities: List[Dict], features: Dict[str, Any]) -> List[str]:
        """Génère des tags automatiques"""
        
        tags = set()
        
        # Tags basés sur les entités
        for entity in entities:
            if entity['label'] in ['ORG', 'PERSON', 'MONEY', 'DATE']:
                # Nettoyer et ajouter
                tag = entity['text'].strip().lower()
                if len(tag) > 2 and len(tag) < 30:
                    tags.add(tag)
        
        # Tags basés sur le type de document détecté
        text_lower = text.lower()
        
        # Tags financiers
        if features.get('content_financial_score', 0) > 0.1:
            financial_tags = ['finance', 'budget', 'comptabilité']
            for tag in financial_tags:
                if tag in text_lower:
                    tags.add(tag)
        
        # Tags juridiques
        if features.get('content_legal_score', 0) > 0.1:
            legal_tags = ['contrat', 'juridique', 'accord', 'loi']
            for tag in legal_tags:
                if tag in text_lower:
                    tags.add(tag)
        
        # Tags temporels
        current_year = datetime.now().year
        for year in range(current_year - 5, current_year + 2):
            if str(year) in text:
                tags.add(str(year))
        
        # Limiter le nombre de tags
        return sorted(list(tags))[:10]
    
    def _calculate_text_clarity(self, text: str) -> float:
        """Calcule un score de clarté du texte"""
        
        try:
            if not text.strip():
                return 0.0
            
            # Facteurs de qualité
            factors = []
            
            # Longueur appropriée
            char_count = len(text)
            if 100 <= char_count <= 100000:
                factors.append(1.0)
            elif char_count < 100:
                factors.append(char_count / 100)
            else:
                factors.append(1.0 - (char_count - 100000) / 100000)
            
            # Ratio caractères/mots raisonnable
            words = text.split()
            if words:
                avg_word_length = sum(len(word) for word in words) / len(words)
                if 3 <= avg_word_length <= 8:
                    factors.append(1.0)
                else:
                    factors.append(0.5)
            
            # Présence de ponctuation
            punct_ratio = len([c for c in text if c in string.punctuation]) / len(text)
            if 0.01 <= punct_ratio <= 0.15:
                factors.append(1.0)
            else:
                factors.append(0.7)
            
            # Pas trop de caractères de contrôle
            control_chars = len(re.findall(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x84\x86-\x9f]', text))
            control_ratio = control_chars / max(len(text), 1)
            if control_ratio < 0.01:
                factors.append(1.0)
            else:
                factors.append(max(0.2, 1.0 - control_ratio * 10))
            
            return min(1.0, sum(factors) / len(factors))
            
        except Exception as e:
            logger.error(f"❌ Erreur calcul clarté: {e}")
            return 0.5
    
    def save_model(self, filepath: str):
        """Sauvegarde les modèles entraînés"""
        try:
            model_data = {
                'type_classifier': self.type_classifier,
                'confidentiality_classifier': self.confidentiality_classifier,
                'tfidf_vectorizer': self.tfidf_vectorizer,
                'is_trained': self.is_trained
            }
            
            joblib.dump(model_data, filepath)
            logger.info(f"💾 Modèles sauvegardés: {filepath}")
            
        except Exception as e:
            logger.error(f"❌ Erreur sauvegarde modèles: {e}")
    
    def load_model(self, filepath: str):
        """Charge les modèles sauvegardés"""
        try:
            model_data = joblib.load(filepath)
            
            self.type_classifier = model_data.get('type_classifier')
            self.confidentiality_classifier = model_data.get('confidentiality_classifier')
            self.tfidf_vectorizer = model_data.get('tfidf_vectorizer')
            self.is_trained = model_data.get('is_trained', False)
            
            logger.info(f"📂 Modèles chargés: {filepath}")
            
        except Exception as e:
            logger.error(f"❌ Erreur chargement modèles: {e}")


# Instance globale
_document_classifier: Optional[MLDocumentClassifier] = None


async def get_document_classifier() -> MLDocumentClassifier:
    """Factory pour obtenir le classificateur de documents"""
    global _document_classifier
    
    if _document_classifier is None:
        _document_classifier = MLDocumentClassifier()
        
        # Charger modèle pré-entraîné s'il existe
        model_path = "document_classifier_models.joblib"
        if os.path.exists(model_path):
            _document_classifier.load_model(model_path)
    
    return _document_classifier