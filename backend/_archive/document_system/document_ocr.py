"""
Système OCR et extraction de données automatisée
US-012: Extraction intelligente de texte et données à partir de documents scannés/images

Ce module fournit:
- OCR multi-moteurs (Tesseract, cloud APIs, deep learning)
- Extraction structurée de données financières et juridiques
- Détection automatique de la mise en page et structure
- Correction et amélioration de la qualité OCR
- Extraction d'entités spécialisées M&A
"""

import asyncio
import os
import io
import tempfile
import re
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import base64

import numpy as np
import cv2
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract
import pdf2image
import fitz  # PyMuPDF
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch

from app.core.logging_system import get_logger, LogCategory

logger = get_logger("document_ocr", LogCategory.DOCUMENT)


class OCREngine(str, Enum):
    """Moteurs OCR disponibles"""
    TESSERACT = "tesseract"
    TROCR = "trocr"
    CLOUD_VISION = "cloud_vision"
    AZURE_VISION = "azure_vision"
    AWS_TEXTRACT = "aws_textract"


class DocumentLayout(str, Enum):
    """Types de mise en page détectés"""
    SINGLE_COLUMN = "single_column"
    MULTI_COLUMN = "multi_column"
    TABLE = "table"
    FORM = "form"
    MIXED = "mixed"
    HANDWRITTEN = "handwritten"


@dataclass
class BoundingBox:
    """Rectangle de délimitation"""
    x: int
    y: int
    width: int
    height: int
    confidence: float = 1.0


@dataclass
class OCRWord:
    """Mot détecté par OCR"""
    text: str
    bbox: BoundingBox
    confidence: float
    font_size: Optional[float] = None


@dataclass
class OCRLine:
    """Ligne de texte détectée"""
    text: str
    words: List[OCRWord]
    bbox: BoundingBox
    confidence: float
    
    @property
    def word_count(self) -> int:
        return len(self.words)


@dataclass
class OCRBlock:
    """Bloc de texte détecté"""
    text: str
    lines: List[OCRLine]
    bbox: BoundingBox
    block_type: str  # paragraph, heading, table, etc.
    confidence: float


@dataclass
class OCRResult:
    """Résultat complet de l'OCR"""
    
    # Texte extrait
    full_text: str
    blocks: List[OCRBlock]
    
    # Métadonnées de qualité
    confidence_score: float
    total_words: int
    low_confidence_words: int
    
    # Analyse de layout
    detected_layout: DocumentLayout
    page_count: int
    
    # Données structurées extraites
    extracted_data: Dict[str, Any] = field(default_factory=dict)
    
    # Métriques de performance
    processing_time: float = 0.0
    engine_used: OCREngine = OCREngine.TESSERACT


@dataclass
class ExtractedEntity:
    """Entité extraite du document"""
    entity_type: str
    value: str
    confidence: float
    bbox: Optional[BoundingBox] = None
    context: Optional[str] = None


class ImagePreprocessor:
    """Préprocesseur d'images pour améliorer l'OCR"""
    
    @staticmethod
    def enhance_image_for_ocr(image: Image.Image) -> Image.Image:
        """Améliore une image pour l'OCR"""
        
        try:
            # Convertir en niveaux de gris si nécessaire
            if image.mode != 'L':
                image = image.convert('L')
            
            # Redimensionner si trop petite
            width, height = image.size
            if width < 1000 or height < 1000:
                scale_factor = max(1000 / width, 1000 / height)
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Améliorer le contraste
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.5)
            
            # Améliorer la netteté
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(1.2)
            
            # Appliquer un filtre de débruitage léger
            image = image.filter(ImageFilter.MedianFilter(size=3))
            
            return image
            
        except Exception as e:
            logger.error(f"❌ Erreur amélioration image: {e}")
            return image
    
    @staticmethod
    def deskew_image(image: Image.Image) -> Image.Image:
        """Corrige l'inclinaison d'une image"""
        
        try:
            # Convertir en OpenCV
            img_array = np.array(image)
            
            # Détection des contours
            edges = cv2.Canny(img_array, 50, 150, apertureSize=3)
            
            # Détection des lignes avec transformation de Hough
            lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
            
            if lines is not None:
                # Calculer l'angle moyen
                angles = []
                for rho, theta in lines[:10]:  # Prendre les 10 premières lignes
                    angle = theta * 180 / np.pi - 90
                    angles.append(angle)
                
                if angles:
                    median_angle = np.median(angles)
                    
                    # Appliquer la rotation si l'angle est significatif
                    if abs(median_angle) > 0.5:
                        return image.rotate(-median_angle, expand=True, fillcolor='white')
            
            return image
            
        except Exception as e:
            logger.error(f"❌ Erreur correction inclinaison: {e}")
            return image
    
    @staticmethod
    def remove_noise(image: Image.Image) -> Image.Image:
        """Supprime le bruit d'une image"""
        
        try:
            # Convertir en array numpy
            img_array = np.array(image)
            
            # Appliquer un filtre médian pour réduire le bruit
            denoised = cv2.medianBlur(img_array, 3)
            
            # Appliquer un filtre gaussien léger
            denoised = cv2.GaussianBlur(denoised, (3, 3), 0)
            
            return Image.fromarray(denoised)
            
        except Exception as e:
            logger.error(f"❌ Erreur suppression bruit: {e}")
            return image


class TesseractOCR:
    """Moteur OCR Tesseract"""
    
    def __init__(self):
        # Configuration Tesseract optimisée
        self.config = (
            '--oem 3 --psm 6 '  # OCR Engine Mode 3, Page Segmentation Mode 6
            '-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz '
            '-c preserve_interword_spaces=1'
        )
    
    async def extract_text(self, image: Image.Image) -> OCRResult:
        """Extrait le texte avec Tesseract"""
        
        start_time = datetime.now()
        
        try:
            # Améliorer l'image
            preprocessor = ImagePreprocessor()
            enhanced_image = preprocessor.enhance_image_for_ocr(image)
            enhanced_image = preprocessor.deskew_image(enhanced_image)
            
            # Extraction avec données détaillées
            data = pytesseract.image_to_data(
                enhanced_image,
                config=self.config,
                output_type=pytesseract.Output.DICT,
                lang='fra+eng'  # Français + Anglais
            )
            
            # Traitement des résultats
            blocks = []
            current_block_words = []
            current_block_bbox = None
            
            full_text_parts = []
            total_words = 0
            low_confidence_words = 0
            
            for i in range(len(data['text'])):
                text = data['text'][i].strip()
                if not text:
                    continue
                
                confidence = float(data['conf'][i])
                if confidence < 0:
                    continue
                
                total_words += 1
                if confidence < 50:
                    low_confidence_words += 1
                
                # Créer mot
                word = OCRWord(
                    text=text,
                    bbox=BoundingBox(
                        x=data['left'][i],
                        y=data['top'][i],
                        width=data['width'][i],
                        height=data['height'][i],
                        confidence=confidence / 100
                    ),
                    confidence=confidence / 100
                )
                
                current_block_words.append(word)
                full_text_parts.append(text)
            
            # Créer un bloc unique (simplification)
            if current_block_words:
                full_text = ' '.join(full_text_parts)
                
                # Calculer bbox global
                all_x = [w.bbox.x for w in current_block_words]
                all_y = [w.bbox.y for w in current_block_words]
                all_x2 = [w.bbox.x + w.bbox.width for w in current_block_words]
                all_y2 = [w.bbox.y + w.bbox.height for w in current_block_words]
                
                global_bbox = BoundingBox(
                    x=min(all_x),
                    y=min(all_y),
                    width=max(all_x2) - min(all_x),
                    height=max(all_y2) - min(all_y)
                )
                
                # Créer lignes (simplification: une ligne par phrase)
                sentences = full_text.split('. ')
                lines = []
                
                for sentence in sentences:
                    if sentence.strip():
                        lines.append(OCRLine(
                            text=sentence.strip(),
                            words=[w for w in current_block_words if w.text in sentence],
                            bbox=global_bbox,  # Simplification
                            confidence=sum(w.confidence for w in current_block_words) / len(current_block_words)
                        ))
                
                block = OCRBlock(
                    text=full_text,
                    lines=lines,
                    bbox=global_bbox,
                    block_type="paragraph",
                    confidence=sum(w.confidence for w in current_block_words) / len(current_block_words)
                )
                
                blocks.append(block)
            
            # Calculer métriques
            processing_time = (datetime.now() - start_time).total_seconds()
            confidence_score = 1.0 - (low_confidence_words / max(total_words, 1))
            
            result = OCRResult(
                full_text=' '.join(full_text_parts),
                blocks=blocks,
                confidence_score=confidence_score,
                total_words=total_words,
                low_confidence_words=low_confidence_words,
                detected_layout=DocumentLayout.SINGLE_COLUMN,  # Simplification
                page_count=1,
                processing_time=processing_time,
                engine_used=OCREngine.TESSERACT
            )
            
            logger.info(f"📄 OCR Tesseract: {total_words} mots, confiance {confidence_score:.2f}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Erreur OCR Tesseract: {e}")
            return OCRResult(
                full_text="",
                blocks=[],
                confidence_score=0.0,
                total_words=0,
                low_confidence_words=0,
                detected_layout=DocumentLayout.SINGLE_COLUMN,
                page_count=1,
                processing_time=(datetime.now() - start_time).total_seconds(),
                engine_used=OCREngine.TESSERACT
            )


class TrOCREngine:
    """Moteur OCR basé sur TrOCR (Transformer-based OCR)"""
    
    def __init__(self):
        self.processor = None
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    async def initialize(self):
        """Initialise le modèle TrOCR"""
        try:
            logger.info("🤖 Initialisation TrOCR...")
            
            # Modèle pré-entraîné Microsoft
            self.processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-printed')
            self.model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-printed')
            self.model.to(self.device)
            
            logger.info("✅ TrOCR initialisé")
            
        except Exception as e:
            logger.error(f"❌ Erreur initialisation TrOCR: {e}")
    
    async def extract_text(self, image: Image.Image) -> OCRResult:
        """Extrait le texte avec TrOCR"""
        
        start_time = datetime.now()
        
        try:
            if not self.model:
                await self.initialize()
            
            # Préprocesser l'image
            preprocessor = ImagePreprocessor()
            enhanced_image = preprocessor.enhance_image_for_ocr(image)
            
            # Redimensionner pour TrOCR
            enhanced_image = enhanced_image.resize((384, 384))
            
            # Traitement avec TrOCR
            pixel_values = self.processor(enhanced_image, return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(self.device)
            
            # Génération
            with torch.no_grad():
                generated_ids = self.model.generate(pixel_values)
            
            # Décodage
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            # Créer résultat simplifié
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Estimer la confiance (TrOCR ne fournit pas de score de confiance direct)
            confidence_score = 0.8  # Score par défaut pour TrOCR
            
            # Créer un bloc simple
            words = generated_text.split()
            bbox = BoundingBox(x=0, y=0, width=image.width, height=image.height)
            
            ocr_words = [
                OCRWord(
                    text=word,
                    bbox=bbox,  # Simplification
                    confidence=confidence_score
                ) for word in words
            ]
            
            line = OCRLine(
                text=generated_text,
                words=ocr_words,
                bbox=bbox,
                confidence=confidence_score
            )
            
            block = OCRBlock(
                text=generated_text,
                lines=[line],
                bbox=bbox,
                block_type="paragraph",
                confidence=confidence_score
            )
            
            result = OCRResult(
                full_text=generated_text,
                blocks=[block],
                confidence_score=confidence_score,
                total_words=len(words),
                low_confidence_words=0,
                detected_layout=DocumentLayout.SINGLE_COLUMN,
                page_count=1,
                processing_time=processing_time,
                engine_used=OCREngine.TROCR
            )
            
            logger.info(f"🤖 OCR TrOCR: {len(words)} mots extraits")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Erreur OCR TrOCR: {e}")
            return OCRResult(
                full_text="",
                blocks=[],
                confidence_score=0.0,
                total_words=0,
                low_confidence_words=0,
                detected_layout=DocumentLayout.SINGLE_COLUMN,
                page_count=1,
                processing_time=(datetime.now() - start_time).total_seconds(),
                engine_used=OCREngine.TROCR
            )


class DataExtractor:
    """Extracteur de données structurées spécialisé M&A"""
    
    def __init__(self):
        # Patterns pour extraction d'entités financières
        self.financial_patterns = {
            'montant_euro': [
                r'(\d{1,3}(?:\s?\d{3})*(?:[,\.]\d{2})?)\s*(?:€|EUR|euros?)',
                r'€\s*(\d{1,3}(?:\s?\d{3})*(?:[,\.]\d{2})?)',
                r'(\d{1,3}(?:\s?\d{3})*(?:[,\.]\d{2})?)\s*euros?'
            ],
            'pourcentage': [
                r'(\d{1,3}(?:[,\.]\d{1,2})?)\s*%',
                r'(\d{1,3}(?:[,\.]\d{1,2})?)\s*pour\s*cent'
            ],
            'date': [
                r'(\d{1,2})[/\-.](\d{1,2})[/\-.](20\d{2})',
                r'(\d{1,2})\s+(janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre)\s+(20\d{2})',
                r'(20\d{2})[/\-.](\d{1,2})[/\-.](\d{1,2})'
            ],
            'numero_siret': [
                r'\b(\d{14})\b',
                r'SIRET\s*:?\s*(\d{14})'
            ],
            'numero_siren': [
                r'\b(\d{9})\b(?!\d)',
                r'SIREN\s*:?\s*(\d{9})'
            ]
        }
        
        # Patterns pour entités juridiques
        self.legal_patterns = {
            'forme_juridique': [
                r'\b(SARL|SAS|SA|EURL|SCI|SASU|SNC|SELARL)\b',
                r'\b(Société Anonyme|Société par Actions Simplifiée|Société à Responsabilité Limitée)\b'
            ],
            'capital_social': [
                r'capital\s*(?:social)?\s*:?\s*(\d{1,3}(?:\s?\d{3})*(?:[,\.]\d{2})?)\s*(?:€|EUR)',
                r'au\s*capital\s*de\s*(\d{1,3}(?:\s?\d{3})*(?:[,\.]\d{2})?)\s*(?:€|EUR)'
            ]
        }
        
        # Patterns pour ratios financiers
        self.ratio_patterns = {
            'ratio_endettement': [
                r'ratio\s*(?:d[\'\"])?endettement\s*:?\s*(\d{1,3}(?:[,\.]\d{1,2})?)\s*%?',
                r'endettement\s*:?\s*(\d{1,3}(?:[,\.]\d{1,2})?)\s*%'
            ],
            'marge_beneficiaire': [
                r'marge\s*(?:bénéficiaire)?\s*:?\s*(\d{1,3}(?:[,\.]\d{1,2})?)\s*%',
                r'rentabilité\s*:?\s*(\d{1,3}(?:[,\.]\d{1,2})?)\s*%'
            ]
        }
    
    def extract_entities(self, text: str) -> List[ExtractedEntity]:
        """Extrait les entités structurées du texte"""
        
        entities = []
        
        try:
            # Extraction entités financières
            for entity_type, patterns in self.financial_patterns.items():
                for pattern in patterns:
                    matches = re.finditer(pattern, text, re.IGNORECASE)
                    for match in matches:
                        entities.append(ExtractedEntity(
                            entity_type=entity_type,
                            value=match.group(1) if match.groups() else match.group(0),
                            confidence=0.9,  # Confiance élevée pour regex
                            context=self._get_context(text, match.start(), match.end())
                        ))
            
            # Extraction entités juridiques
            for entity_type, patterns in self.legal_patterns.items():
                for pattern in patterns:
                    matches = re.finditer(pattern, text, re.IGNORECASE)
                    for match in matches:
                        entities.append(ExtractedEntity(
                            entity_type=entity_type,
                            value=match.group(1) if match.groups() else match.group(0),
                            confidence=0.9,
                            context=self._get_context(text, match.start(), match.end())
                        ))
            
            # Extraction ratios
            for entity_type, patterns in self.ratio_patterns.items():
                for pattern in patterns:
                    matches = re.finditer(pattern, text, re.IGNORECASE)
                    for match in matches:
                        entities.append(ExtractedEntity(
                            entity_type=entity_type,
                            value=match.group(1) if match.groups() else match.group(0),
                            confidence=0.85,
                            context=self._get_context(text, match.start(), match.end())
                        ))
            
            # Déduplication basique
            entities = self._deduplicate_entities(entities)
            
            logger.info(f"📊 {len(entities)} entités extraites")
            
            return entities
            
        except Exception as e:
            logger.error(f"❌ Erreur extraction entités: {e}")
            return []
    
    def _get_context(self, text: str, start: int, end: int, context_size: int = 50) -> str:
        """Récupère le contexte autour d'une entité"""
        
        context_start = max(0, start - context_size)
        context_end = min(len(text), end + context_size)
        
        return text[context_start:context_end].strip()
    
    def _deduplicate_entities(self, entities: List[ExtractedEntity]) -> List[ExtractedEntity]:
        """Supprime les entités dupliquées"""
        
        seen = set()
        deduplicated = []
        
        for entity in entities:
            key = (entity.entity_type, entity.value.lower())
            if key not in seen:
                seen.add(key)
                deduplicated.append(entity)
        
        return deduplicated
    
    def extract_financial_summary(self, text: str) -> Dict[str, Any]:
        """Extrait un résumé financier du document"""
        
        summary = {
            'chiffre_affaires': None,
            'benefice': None,
            'capital_social': None,
            'effectifs': None,
            'ratios': {},
            'dates_cles': []
        }
        
        try:
            # Extraction CA
            ca_patterns = [
                r'chiffre\s*d[\'\"]affaires\s*:?\s*(\d{1,3}(?:\s?\d{3})*(?:[,\.]\d{2})?)\s*(?:€|EUR|K€|M€)',
                r'CA\s*:?\s*(\d{1,3}(?:\s?\d{3})*(?:[,\.]\d{2})?)\s*(?:€|EUR|K€|M€)',
                r'revenus?\s*:?\s*(\d{1,3}(?:\s?\d{3})*(?:[,\.]\d{2})?)\s*(?:€|EUR|K€|M€)'
            ]
            
            for pattern in ca_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    summary['chiffre_affaires'] = match.group(1)
                    break
            
            # Extraction bénéfice
            profit_patterns = [
                r'bénéfice\s*(?:net)?\s*:?\s*(\d{1,3}(?:\s?\d{3})*(?:[,\.]\d{2})?)\s*(?:€|EUR|K€|M€)',
                r'résultat\s*(?:net)?\s*:?\s*(\d{1,3}(?:\s?\d{3})*(?:[,\.]\d{2})?)\s*(?:€|EUR|K€|M€)'
            ]
            
            for pattern in profit_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    summary['benefice'] = match.group(1)
                    break
            
            # Extraction effectifs
            effectif_patterns = [
                r'effectifs?\s*:?\s*(\d{1,4})\s*(?:personnes?|salariés?|employés?)',
                r'(\d{1,4})\s*(?:personnes?|salariés?|employés?)'
            ]
            
            for pattern in effectif_patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    summary['effectifs'] = int(match.group(1))
                    break
            
            return summary
            
        except Exception as e:
            logger.error(f"❌ Erreur extraction résumé financier: {e}")
            return summary


class OCRDocumentProcessor:
    """Processeur principal pour l'OCR et extraction de documents"""
    
    def __init__(self):
        self.tesseract_engine = TesseractOCR()
        self.trocr_engine = TrOCREngine()
        self.data_extractor = DataExtractor()
        
        # Configuration des moteurs par type de document
        self.engine_preferences = {
            'application/pdf': [OCREngine.TESSERACT, OCREngine.TROCR],
            'image/jpeg': [OCREngine.TROCR, OCREngine.TESSERACT],
            'image/png': [OCREngine.TROCR, OCREngine.TESSERACT],
            'image/tiff': [OCREngine.TESSERACT, OCREngine.TROCR]
        }
    
    async def process_document(
        self,
        file_data: bytes,
        mime_type: str,
        filename: str = "",
        extract_entities: bool = True
    ) -> OCRResult:
        """Traite un document complet avec OCR et extraction"""
        
        start_time = datetime.now()
        
        try:
            logger.info(f"📄 Traitement OCR document: {filename} ({mime_type})")
            
            # Convertir en images
            images = await self._convert_to_images(file_data, mime_type)
            
            if not images:
                logger.error("❌ Impossible de convertir le document en images")
                return OCRResult(
                    full_text="",
                    blocks=[],
                    confidence_score=0.0,
                    total_words=0,
                    low_confidence_words=0,
                    detected_layout=DocumentLayout.SINGLE_COLUMN,
                    page_count=0
                )
            
            # Choisir moteur OCR optimal
            preferred_engines = self.engine_preferences.get(mime_type, [OCREngine.TESSERACT])
            
            # Traiter chaque page
            all_results = []
            
            for i, image in enumerate(images):
                logger.info(f"📄 Traitement page {i+1}/{len(images)}")
                
                # Essayer les moteurs dans l'ordre de préférence
                best_result = None
                best_confidence = 0.0
                
                for engine in preferred_engines:
                    try:
                        if engine == OCREngine.TESSERACT:
                            result = await self.tesseract_engine.extract_text(image)
                        elif engine == OCREngine.TROCR:
                            result = await self.trocr_engine.extract_text(image)
                        else:
                            continue  # Moteurs cloud non implémentés dans cette démo
                        
                        if result.confidence_score > best_confidence:
                            best_result = result
                            best_confidence = result.confidence_score
                            
                        # Si confiance > 80%, utiliser ce résultat
                        if result.confidence_score > 0.8:
                            break
                            
                    except Exception as e:
                        logger.warning(f"⚠️ Erreur moteur {engine}: {e}")
                        continue
                
                if best_result:
                    all_results.append(best_result)
            
            # Fusionner les résultats de toutes les pages
            combined_result = self._combine_page_results(all_results)
            
            # Extraction des entités si demandée
            if extract_entities and combined_result.full_text:
                entities = self.data_extractor.extract_entities(combined_result.full_text)
                financial_summary = self.data_extractor.extract_financial_summary(combined_result.full_text)
                
                combined_result.extracted_data = {
                    'entities': [
                        {
                            'type': e.entity_type,
                            'value': e.value,
                            'confidence': e.confidence,
                            'context': e.context
                        } for e in entities
                    ],
                    'financial_summary': financial_summary
                }
            
            # Calculer temps total
            combined_result.processing_time = (datetime.now() - start_time).total_seconds()
            
            logger.info(
                f"✅ OCR terminé: {combined_result.total_words} mots, "
                f"confiance {combined_result.confidence_score:.2f}, "
                f"{len(combined_result.extracted_data.get('entities', []))} entités extraites"
            )
            
            return combined_result
            
        except Exception as e:
            logger.error(f"❌ Erreur traitement document OCR: {e}")
            return OCRResult(
                full_text="",
                blocks=[],
                confidence_score=0.0,
                total_words=0,
                low_confidence_words=0,
                detected_layout=DocumentLayout.SINGLE_COLUMN,
                page_count=0,
                processing_time=(datetime.now() - start_time).total_seconds()
            )
    
    async def _convert_to_images(self, file_data: bytes, mime_type: str) -> List[Image.Image]:
        """Convertit un document en images"""
        
        try:
            if mime_type == 'application/pdf':
                return await self._pdf_to_images(file_data)
            elif mime_type.startswith('image/'):
                image = Image.open(io.BytesIO(file_data))
                return [image]
            else:
                logger.error(f"❌ Type de fichier non supporté: {mime_type}")
                return []
                
        except Exception as e:
            logger.error(f"❌ Erreur conversion en images: {e}")
            return []
    
    async def _pdf_to_images(self, pdf_data: bytes) -> List[Image.Image]:
        """Convertit un PDF en images"""
        
        try:
            # Méthode 1: pdf2image
            try:
                images = pdf2image.convert_from_bytes(
                    pdf_data,
                    dpi=300,  # Haute résolution pour OCR
                    fmt='RGB'
                )
                return images
            except Exception as e:
                logger.warning(f"⚠️ pdf2image échoué: {e}")
            
            # Méthode 2: PyMuPDF (fallback)
            doc = fitz.open(stream=pdf_data, filetype="pdf")
            images = []
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                
                # Render en haute résolution
                mat = fitz.Matrix(2.0, 2.0)  # Facteur de zoom 2x
                pix = page.get_pixmap(matrix=mat)
                
                # Convertir en PIL Image
                img_data = pix.tobytes("ppm")
                image = Image.open(io.BytesIO(img_data))
                images.append(image)
            
            doc.close()
            return images
            
        except Exception as e:
            logger.error(f"❌ Erreur conversion PDF: {e}")
            return []
    
    def _combine_page_results(self, page_results: List[OCRResult]) -> OCRResult:
        """Combine les résultats de plusieurs pages"""
        
        if not page_results:
            return OCRResult(
                full_text="",
                blocks=[],
                confidence_score=0.0,
                total_words=0,
                low_confidence_words=0,
                detected_layout=DocumentLayout.SINGLE_COLUMN,
                page_count=0
            )
        
        # Combiner textes
        combined_text = '\n\n--- PAGE BREAK ---\n\n'.join(
            result.full_text for result in page_results if result.full_text
        )
        
        # Combiner blocs
        combined_blocks = []
        for result in page_results:
            combined_blocks.extend(result.blocks)
        
        # Calculer métriques globales
        total_words = sum(result.total_words for result in page_results)
        total_low_confidence = sum(result.low_confidence_words for result in page_results)
        
        avg_confidence = sum(
            result.confidence_score for result in page_results
        ) / len(page_results)
        
        return OCRResult(
            full_text=combined_text,
            blocks=combined_blocks,
            confidence_score=avg_confidence,
            total_words=total_words,
            low_confidence_words=total_low_confidence,
            detected_layout=DocumentLayout.MIXED,  # Plusieurs pages = mixed
            page_count=len(page_results),
            engine_used=page_results[0].engine_used if page_results else OCREngine.TESSERACT
        )


# Instance globale
_ocr_processor: Optional[OCRDocumentProcessor] = None


async def get_ocr_processor() -> OCRDocumentProcessor:
    """Factory pour obtenir le processeur OCR"""
    global _ocr_processor
    
    if _ocr_processor is None:
        _ocr_processor = OCRDocumentProcessor()
        
        # Initialiser TrOCR en arrière-plan
        try:
            await _ocr_processor.trocr_engine.initialize()
        except Exception as e:
            logger.warning(f"⚠️ TrOCR non disponible: {e}")
    
    return _ocr_processor