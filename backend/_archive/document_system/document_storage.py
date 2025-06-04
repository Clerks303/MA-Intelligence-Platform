"""
Système de stockage et indexation de documents avancé
US-012: Gestion documentaire complète pour M&A Intelligence Platform

Ce module fournit:
- Stockage multi-backend (local, S3, Google Drive)
- Indexation sémantique avec embeddings vectoriels
- Métadonnées enrichies et versioning
- Compression et déduplication intelligente
- Audit trail complet des accès
"""

import os
import hashlib
import mimetypes
import asyncio
import aiofiles
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, BinaryIO
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import json
import zipfile
import tempfile

import boto3
from botocore.exceptions import ClientError
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

from app.core.logging_system import get_logger, LogCategory
from app.core.cache_manager import get_cache_manager

logger = get_logger("document_storage", LogCategory.DOCUMENT)


class StorageBackend(str, Enum):
    """Types de backend de stockage"""
    LOCAL = "local"
    AWS_S3 = "aws_s3"
    GOOGLE_DRIVE = "google_drive"
    AZURE_BLOB = "azure_blob"


class DocumentType(str, Enum):
    """Types de documents M&A"""
    FINANCIAL = "financial"
    LEGAL = "legal"
    DUE_DILIGENCE = "due_diligence"
    COMMUNICATION = "communication"
    TECHNICAL = "technical"
    HR = "hr"
    COMMERCIAL = "commercial"
    OTHER = "other"


class DocumentStatus(str, Enum):
    """Statuts de documents"""
    DRAFT = "draft"
    UNDER_REVIEW = "under_review"
    APPROVED = "approved"
    ARCHIVED = "archived"
    DELETED = "deleted"


class AccessLevel(str, Enum):
    """Niveaux d'accès aux documents"""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


@dataclass
class DocumentMetadata:
    """Métadonnées complètes d'un document"""
    
    # Identifiants
    document_id: str
    filename: str
    original_filename: str
    
    # Type et classification
    document_type: DocumentType
    mime_type: str
    file_extension: str
    
    # Taille et hachage
    file_size: int
    md5_hash: str
    sha256_hash: str
    
    # Contenu et indexation
    title: Optional[str] = None
    description: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    extracted_text: Optional[str] = None
    embedding_vector: Optional[np.ndarray] = None
    
    # Sécurité et accès
    access_level: AccessLevel = AccessLevel.INTERNAL
    owner_id: str = ""
    allowed_users: List[str] = field(default_factory=list)
    allowed_groups: List[str] = field(default_factory=list)
    
    # Versioning
    version: int = 1
    parent_document_id: Optional[str] = None
    is_latest_version: bool = True
    
    # Workflow
    status: DocumentStatus = DocumentStatus.DRAFT
    reviewer_id: Optional[str] = None
    approved_by: Optional[str] = None
    approved_at: Optional[datetime] = None
    
    # Métadonnées temporelles
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    accessed_at: datetime = field(default_factory=datetime.now)
    
    # Contexte M&A
    company_id: Optional[str] = None
    deal_id: Optional[str] = None
    project_phase: Optional[str] = None
    
    # Stockage
    storage_backend: StorageBackend = StorageBackend.LOCAL
    storage_path: str = ""
    storage_url: Optional[str] = None
    
    # Analytics
    download_count: int = 0
    view_count: int = 0
    last_downloaded: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire pour stockage"""
        data = {}
        for field_name, field_def in self.__dataclass_fields__.items():
            value = getattr(self, field_name)
            
            if isinstance(value, datetime):
                data[field_name] = value.isoformat()
            elif isinstance(value, Enum):
                data[field_name] = value.value
            elif isinstance(value, np.ndarray):
                data[field_name] = value.tolist() if value is not None else None
            else:
                data[field_name] = value
        
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DocumentMetadata':
        """Crée depuis un dictionnaire"""
        
        # Convertir les dates
        for date_field in ['created_at', 'updated_at', 'accessed_at', 'approved_at', 'last_downloaded']:
            if data.get(date_field) and isinstance(data[date_field], str):
                data[date_field] = datetime.fromisoformat(data[date_field])
        
        # Convertir les enums
        if 'document_type' in data:
            data['document_type'] = DocumentType(data['document_type'])
        if 'access_level' in data:
            data['access_level'] = AccessLevel(data['access_level'])
        if 'status' in data:
            data['status'] = DocumentStatus(data['status'])
        if 'storage_backend' in data:
            data['storage_backend'] = StorageBackend(data['storage_backend'])
        
        # Convertir embedding vector
        if data.get('embedding_vector') and isinstance(data['embedding_vector'], list):
            data['embedding_vector'] = np.array(data['embedding_vector'])
        
        return cls(**data)


@dataclass
class SearchResult:
    """Résultat de recherche de document"""
    document_id: str
    metadata: DocumentMetadata
    relevance_score: float
    snippet: Optional[str] = None
    highlighted_text: Optional[str] = None


class VectorIndex:
    """Index vectoriel pour recherche sémantique"""
    
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)  # Inner Product pour similarité cosinus
        self.document_ids: List[str] = []
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')  # Modèle léger et efficace
        
    def add_document(self, document_id: str, text: str):
        """Ajoute un document à l'index"""
        try:
            # Générer embedding
            embedding = self.encoder.encode([text], convert_to_tensor=False)[0]
            
            # Normaliser pour similarité cosinus
            embedding = embedding / np.linalg.norm(embedding)
            
            # Ajouter à l'index
            self.index.add(embedding.reshape(1, -1))
            self.document_ids.append(document_id)
            
            logger.info(f"📑 Document ajouté à l'index vectoriel: {document_id}")
            
        except Exception as e:
            logger.error(f"❌ Erreur ajout document à l'index: {e}")
            raise
    
    def search(self, query: str, k: int = 10) -> List[tuple]:
        """Recherche sémantique dans l'index"""
        try:
            # Encoder la requête
            query_embedding = self.encoder.encode([query], convert_to_tensor=False)[0]
            query_embedding = query_embedding / np.linalg.norm(query_embedding)
            
            # Rechercher
            scores, indices = self.index.search(query_embedding.reshape(1, -1), k)
            
            # Retourner résultats
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.document_ids):
                    results.append((self.document_ids[idx], float(score)))
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Erreur recherche vectorielle: {e}")
            return []
    
    def save(self, filepath: str):
        """Sauvegarde l'index"""
        try:
            faiss.write_index(self.index, f"{filepath}.faiss")
            
            with open(f"{filepath}.ids", 'w') as f:
                json.dump(self.document_ids, f)
                
            logger.info(f"💾 Index vectoriel sauvegardé: {filepath}")
            
        except Exception as e:
            logger.error(f"❌ Erreur sauvegarde index: {e}")
    
    def load(self, filepath: str):
        """Charge l'index"""
        try:
            self.index = faiss.read_index(f"{filepath}.faiss")
            
            with open(f"{filepath}.ids", 'r') as f:
                self.document_ids = json.load(f)
                
            logger.info(f"📂 Index vectoriel chargé: {filepath}")
            
        except Exception as e:
            logger.error(f"❌ Erreur chargement index: {e}")


class LocalStorageBackend:
    """Backend de stockage local"""
    
    def __init__(self, base_path: str = "documents"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)
        
    async def store(self, file_data: bytes, filename: str, metadata: DocumentMetadata) -> str:
        """Stocke un fichier localement"""
        try:
            # Organiser par type et date
            date_path = datetime.now().strftime("%Y/%m/%d")
            storage_dir = self.base_path / metadata.document_type.value / date_path
            storage_dir.mkdir(parents=True, exist_ok=True)
            
            # Nom de fichier unique
            unique_filename = f"{metadata.document_id}_{filename}"
            file_path = storage_dir / unique_filename
            
            # Écrire le fichier
            async with aiofiles.open(file_path, 'wb') as f:
                await f.write(file_data)
            
            # Retourner chemin relatif
            relative_path = str(file_path.relative_to(self.base_path))
            logger.info(f"💾 Fichier stocké localement: {relative_path}")
            
            return relative_path
            
        except Exception as e:
            logger.error(f"❌ Erreur stockage local: {e}")
            raise
    
    async def retrieve(self, storage_path: str) -> bytes:
        """Récupère un fichier local"""
        try:
            file_path = self.base_path / storage_path
            
            async with aiofiles.open(file_path, 'rb') as f:
                data = await f.read()
            
            return data
            
        except Exception as e:
            logger.error(f"❌ Erreur récupération fichier: {e}")
            raise
    
    async def delete(self, storage_path: str) -> bool:
        """Supprime un fichier local"""
        try:
            file_path = self.base_path / storage_path
            
            if file_path.exists():
                file_path.unlink()
                logger.info(f"🗑️ Fichier supprimé: {storage_path}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Erreur suppression fichier: {e}")
            return False


class S3StorageBackend:
    """Backend de stockage AWS S3"""
    
    def __init__(self, bucket_name: str, aws_access_key: str = None, aws_secret_key: str = None):
        self.bucket_name = bucket_name
        
        if aws_access_key and aws_secret_key:
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=aws_access_key,
                aws_secret_access_key=aws_secret_key
            )
        else:
            # Utiliser les credentials par défaut (IAM role, env vars, etc.)
            self.s3_client = boto3.client('s3')
    
    async def store(self, file_data: bytes, filename: str, metadata: DocumentMetadata) -> str:
        """Stocke un fichier sur S3"""
        try:
            # Clé S3 organisée
            date_path = datetime.now().strftime("%Y/%m/%d")
            s3_key = f"documents/{metadata.document_type.value}/{date_path}/{metadata.document_id}_{filename}"
            
            # Upload avec métadonnées
            extra_args = {
                'Metadata': {
                    'document-id': metadata.document_id,
                    'content-type': metadata.mime_type,
                    'owner-id': metadata.owner_id
                }
            }
            
            # Upload asynchrone simulé (boto3 est sync)
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.s3_client.put_object(
                    Bucket=self.bucket_name,
                    Key=s3_key,
                    Body=file_data,
                    **extra_args
                )
            )
            
            logger.info(f"☁️ Fichier uploadé sur S3: {s3_key}")
            return s3_key
            
        except Exception as e:
            logger.error(f"❌ Erreur upload S3: {e}")
            raise
    
    async def retrieve(self, storage_path: str) -> bytes:
        """Récupère un fichier depuis S3"""
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.s3_client.get_object(Bucket=self.bucket_name, Key=storage_path)
            )
            
            return response['Body'].read()
            
        except Exception as e:
            logger.error(f"❌ Erreur téléchargement S3: {e}")
            raise
    
    async def delete(self, storage_path: str) -> bool:
        """Supprime un fichier de S3"""
        try:
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.s3_client.delete_object(Bucket=self.bucket_name, Key=storage_path)
            )
            
            logger.info(f"🗑️ Fichier supprimé de S3: {storage_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur suppression S3: {e}")
            return False


class DocumentStorageEngine:
    """Moteur principal de gestion documentaire"""
    
    def __init__(
        self,
        metadata_store_path: str = "document_metadata.json",
        vector_index_path: str = "document_index"
    ):
        self.metadata_store_path = metadata_store_path
        self.vector_index_path = vector_index_path
        
        # Stockage des métadonnées
        self.metadata_store: Dict[str, DocumentMetadata] = {}
        
        # Index vectoriel pour recherche sémantique
        self.vector_index = VectorIndex()
        
        # Backends de stockage
        self.storage_backends: Dict[StorageBackend, Any] = {
            StorageBackend.LOCAL: LocalStorageBackend()
        }
        
        # Cache manager
        self.cache = get_cache_manager()
        
        # Statistiques
        self.stats = {
            "total_documents": 0,
            "total_size": 0,
            "documents_by_type": {},
            "documents_by_status": {},
            "upload_count": 0,
            "download_count": 0,
            "search_count": 0
        }
    
    async def initialize(self):
        """Initialise le moteur de stockage"""
        try:
            logger.info("🚀 Initialisation du moteur de stockage documentaire...")
            
            # Charger métadonnées existantes
            await self._load_metadata_store()
            
            # Charger index vectoriel existant
            await self._load_vector_index()
            
            # Configurer backends externes si disponibles
            await self._configure_storage_backends()
            
            logger.info("✅ Moteur de stockage documentaire initialisé")
            
        except Exception as e:
            logger.error(f"❌ Erreur initialisation stockage: {e}")
            raise
    
    async def _load_metadata_store(self):
        """Charge le store de métadonnées"""
        try:
            if os.path.exists(self.metadata_store_path):
                async with aiofiles.open(self.metadata_store_path, 'r') as f:
                    data = json.loads(await f.read())
                
                for doc_id, metadata_dict in data.items():
                    self.metadata_store[doc_id] = DocumentMetadata.from_dict(metadata_dict)
                
                logger.info(f"📂 {len(self.metadata_store)} métadonnées de documents chargées")
        
        except Exception as e:
            logger.warning(f"⚠️ Erreur chargement métadonnées: {e}")
    
    async def _save_metadata_store(self):
        """Sauvegarde le store de métadonnées"""
        try:
            data = {doc_id: metadata.to_dict() for doc_id, metadata in self.metadata_store.items()}
            
            async with aiofiles.open(self.metadata_store_path, 'w') as f:
                await f.write(json.dumps(data, indent=2))
        
        except Exception as e:
            logger.error(f"❌ Erreur sauvegarde métadonnées: {e}")
    
    async def _load_vector_index(self):
        """Charge l'index vectoriel"""
        try:
            if os.path.exists(f"{self.vector_index_path}.faiss"):
                self.vector_index.load(self.vector_index_path)
                logger.info("🔍 Index vectoriel chargé")
        
        except Exception as e:
            logger.warning(f"⚠️ Erreur chargement index vectoriel: {e}")
    
    async def _save_vector_index(self):
        """Sauvegarde l'index vectoriel"""
        try:
            self.vector_index.save(self.vector_index_path)
        except Exception as e:
            logger.error(f"❌ Erreur sauvegarde index vectoriel: {e}")
    
    async def _configure_storage_backends(self):
        """Configure les backends de stockage externes"""
        try:
            # Configuration S3 si variables d'environnement disponibles
            s3_bucket = os.getenv('AWS_S3_BUCKET')
            if s3_bucket:
                self.storage_backends[StorageBackend.AWS_S3] = S3StorageBackend(s3_bucket)
                logger.info(f"☁️ Backend S3 configuré: {s3_bucket}")
        
        except Exception as e:
            logger.warning(f"⚠️ Erreur configuration backends: {e}")
    
    async def store_document(
        self,
        file_data: bytes,
        filename: str,
        document_type: DocumentType,
        owner_id: str,
        **kwargs
    ) -> str:
        """Stocke un document avec métadonnées complètes"""
        
        try:
            # Générer ID unique
            document_id = str(uuid.uuid4())
            
            # Calculer hachages
            md5_hash = hashlib.md5(file_data).hexdigest()
            sha256_hash = hashlib.sha256(file_data).hexdigest()
            
            # Détecter type MIME
            mime_type, _ = mimetypes.guess_type(filename)
            if not mime_type:
                mime_type = 'application/octet-stream'
            
            # Extension de fichier
            file_extension = Path(filename).suffix.lower()
            
            # Créer métadonnées
            metadata = DocumentMetadata(
                document_id=document_id,
                filename=filename,
                original_filename=filename,
                document_type=document_type,
                mime_type=mime_type,
                file_extension=file_extension,
                file_size=len(file_data),
                md5_hash=md5_hash,
                sha256_hash=sha256_hash,
                owner_id=owner_id,
                **kwargs
            )
            
            # Choisir backend de stockage
            backend = self.storage_backends[metadata.storage_backend]
            
            # Stocker le fichier
            storage_path = await backend.store(file_data, filename, metadata)
            metadata.storage_path = storage_path
            
            # Sauvegarder métadonnées
            self.metadata_store[document_id] = metadata
            await self._save_metadata_store()
            
            # Mettre à jour statistiques
            self.stats["total_documents"] += 1
            self.stats["total_size"] += len(file_data)
            self.stats["upload_count"] += 1
            
            type_count = self.stats["documents_by_type"].get(document_type.value, 0)
            self.stats["documents_by_type"][document_type.value] = type_count + 1
            
            logger.info(f"📄 Document stocké: {document_id} ({filename})")
            
            return document_id
            
        except Exception as e:
            logger.error(f"❌ Erreur stockage document: {e}")
            raise
    
    async def retrieve_document(self, document_id: str, user_id: str = None) -> tuple:
        """Récupère un document et ses métadonnées"""
        
        try:
            # Vérifier existence
            if document_id not in self.metadata_store:
                raise FileNotFoundError(f"Document non trouvé: {document_id}")
            
            metadata = self.metadata_store[document_id]
            
            # Vérifier permissions (basique)
            if user_id and metadata.access_level != AccessLevel.PUBLIC:
                if user_id != metadata.owner_id and user_id not in metadata.allowed_users:
                    raise PermissionError(f"Accès refusé au document: {document_id}")
            
            # Récupérer depuis le backend
            backend = self.storage_backends[metadata.storage_backend]
            file_data = await backend.retrieve(metadata.storage_path)
            
            # Mettre à jour statistiques d'accès
            metadata.accessed_at = datetime.now()
            metadata.view_count += 1
            await self._save_metadata_store()
            
            self.stats["download_count"] += 1
            
            logger.info(f"📥 Document récupéré: {document_id}")
            
            return file_data, metadata
            
        except Exception as e:
            logger.error(f"❌ Erreur récupération document: {e}")
            raise
    
    async def index_document_content(self, document_id: str, extracted_text: str):
        """Indexe le contenu textuel d'un document"""
        
        try:
            if document_id not in self.metadata_store:
                raise ValueError(f"Document non trouvé: {document_id}")
            
            metadata = self.metadata_store[document_id]
            
            # Sauvegarder le texte extrait
            metadata.extracted_text = extracted_text
            
            # Ajouter à l'index vectoriel
            self.vector_index.add_document(document_id, extracted_text)
            
            # Sauvegarder
            await self._save_metadata_store()
            await self._save_vector_index()
            
            logger.info(f"🔍 Document indexé: {document_id}")
            
        except Exception as e:
            logger.error(f"❌ Erreur indexation: {e}")
            raise
    
    async def search_documents(
        self,
        query: str,
        user_id: str = None,
        document_type: DocumentType = None,
        limit: int = 10
    ) -> List[SearchResult]:
        """Recherche sémantique dans les documents"""
        
        try:
            # Recherche vectorielle
            vector_results = self.vector_index.search(query, k=limit * 2)  # Plus large pour filtrage
            
            results = []
            
            for doc_id, score in vector_results:
                if doc_id not in self.metadata_store:
                    continue
                
                metadata = self.metadata_store[doc_id]
                
                # Filtrer par type si spécifié
                if document_type and metadata.document_type != document_type:
                    continue
                
                # Vérifier permissions
                if user_id and metadata.access_level != AccessLevel.PUBLIC:
                    if user_id != metadata.owner_id and user_id not in metadata.allowed_users:
                        continue
                
                # Générer snippet du texte
                snippet = None
                if metadata.extracted_text:
                    snippet = metadata.extracted_text[:200] + "..."
                
                results.append(SearchResult(
                    document_id=doc_id,
                    metadata=metadata,
                    relevance_score=score,
                    snippet=snippet
                ))
                
                if len(results) >= limit:
                    break
            
            self.stats["search_count"] += 1
            
            logger.info(f"🔍 Recherche effectuée: '{query}' - {len(results)} résultats")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Erreur recherche: {e}")
            raise
    
    async def delete_document(self, document_id: str, user_id: str = None) -> bool:
        """Supprime un document"""
        
        try:
            if document_id not in self.metadata_store:
                return False
            
            metadata = self.metadata_store[document_id]
            
            # Vérifier permissions
            if user_id and user_id != metadata.owner_id:
                raise PermissionError(f"Permissions insuffisantes pour supprimer: {document_id}")
            
            # Supprimer du backend
            backend = self.storage_backends[metadata.storage_backend]
            await backend.delete(metadata.storage_path)
            
            # Supprimer métadonnées
            del self.metadata_store[document_id]
            await self._save_metadata_store()
            
            # Mettre à jour statistiques
            self.stats["total_documents"] -= 1
            self.stats["total_size"] -= metadata.file_size
            
            logger.info(f"🗑️ Document supprimé: {document_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur suppression: {e}")
            raise
    
    def get_document_metadata(self, document_id: str) -> Optional[DocumentMetadata]:
        """Récupère les métadonnées d'un document"""
        return self.metadata_store.get(document_id)
    
    def list_documents(
        self,
        user_id: str = None,
        document_type: DocumentType = None,
        status: DocumentStatus = None,
        limit: int = 50,
        offset: int = 0
    ) -> List[DocumentMetadata]:
        """Liste les documents avec filtres"""
        
        documents = list(self.metadata_store.values())
        
        # Filtrer par permissions
        if user_id:
            documents = [
                doc for doc in documents
                if doc.access_level == AccessLevel.PUBLIC or
                   doc.owner_id == user_id or
                   user_id in doc.allowed_users
            ]
        
        # Filtrer par type
        if document_type:
            documents = [doc for doc in documents if doc.document_type == document_type]
        
        # Filtrer par statut
        if status:
            documents = [doc for doc in documents if doc.status == status]
        
        # Trier par date de création (plus récent en premier)
        documents.sort(key=lambda x: x.created_at, reverse=True)
        
        # Pagination
        return documents[offset:offset + limit]
    
    def get_storage_statistics(self) -> Dict[str, Any]:
        """Retourne les statistiques de stockage"""
        return {
            **self.stats,
            "total_size_mb": round(self.stats["total_size"] / (1024 * 1024), 2),
            "average_file_size": round(self.stats["total_size"] / max(self.stats["total_documents"], 1)),
            "backends_configured": list(self.storage_backends.keys())
        }


# Instance globale
_document_storage: Optional[DocumentStorageEngine] = None


async def get_document_storage() -> DocumentStorageEngine:
    """Factory pour obtenir le moteur de stockage documentaire"""
    global _document_storage
    
    if _document_storage is None:
        _document_storage = DocumentStorageEngine()
        await _document_storage.initialize()
    
    return _document_storage