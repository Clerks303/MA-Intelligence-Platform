"""    
Intégration Y.js pour édition collaborative temps réel
Sprint 5 - Y.js + WebSocket pour synchronisation robuste de documents

Ce module fournit:
- Intégration Y.js (Ypy) pour synchronisation de documents 
- WebSocket provider compatible Y.js
- Gestion de la persistance des documents Y.js
- Synchronisation avec le système de collaboration existant
- Fallback polling pour compatibilité
"""

import asyncio
import json
import uuid
import base64
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import weakref
from collections import defaultdict

try:
    import y_py as Y
    YJSPY_AVAILABLE = True
except ImportError:
    YJSPY_AVAILABLE = False
    # Fallback implementation
    class Y:
        class YDoc:
            def __init__(self): pass
            def get_text(self, name): return YText()
            def encode_state_as_update(self): return b''
            def apply_update(self, update): pass
        
        class YText:
            def __init__(self): self._content = ""
            def insert(self, index, content): pass
            def delete(self, index, length): pass
            def __str__(self): return self._content

import websockets
from websockets.server import WebSocketServerProtocol
from websockets.exceptions import ConnectionClosed

from app.core.document_collaboration import (
    DocumentCollaborationManager, 
    CollaborationSession,
    WebSocketManager,
    get_document_collaboration_manager
)
from app.core.logging_system import get_logger, LogCategory
from app.core.cache_manager import get_cache_manager

logger = get_logger("yjs_integration", LogCategory.DOCUMENT)


class YjsMessageType(str, Enum):
    """Types de messages Y.js WebSocket"""
    SYNC_STEP_1 = "sync_step_1"
    SYNC_STEP_2 = "sync_step_2"
    UPDATE = "update" 
    AWARENESS = "awareness"
    QUERY_AWARENESS = "query_awareness"
    AUTH = "auth"
    PERMISSION_DENIED = "permission_denied"
    

@dataclass
class YjsDocument:
    """Document Y.js avec métadonnées"""
    
    document_id: str
    ydoc: Y.YDoc
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    
    # Statistiques
    total_updates: int = 0
    active_clients: Set[str] = field(default_factory=set)
    
    # Configuration
    auto_save_interval: int = 30  # secondes
    max_history_size: int = 1000
    
    def get_text_content(self, text_name: str = "default") -> str:
        """Obtient le contenu texte d'un YText"""
        try:
            ytext = self.ydoc.get_text(text_name)
            return str(ytext)
        except Exception as e:
            logger.error(f"❌ Erreur lecture YText {text_name}: {e}")
            return ""
    
    def get_state_vector(self) -> bytes:
        """Obtient le vecteur d'état du document"""
        try:
            return self.ydoc.encode_state_vector()
        except Exception as e:
            logger.error(f"❌ Erreur encodage state vector: {e}")
            return b''
    
    def get_update_since(self, state_vector: bytes) -> bytes:
        """Obtient les mises à jour depuis un état donné"""
        try:
            return self.ydoc.encode_state_as_update(state_vector)
        except Exception as e:
            logger.error(f"❌ Erreur encodage update: {e}")
            return b''
    
    def apply_update(self, update: bytes) -> bool:
        """Applique une mise à jour au document"""
        try:
            self.ydoc.apply_update(update)
            self.last_updated = datetime.now()
            self.total_updates += 1
            return True
        except Exception as e:
            logger.error(f"❌ Erreur application update: {e}")
            return False


@dataclass
class YjsClient:
    """Client Y.js connecté"""
    
    client_id: str
    user_id: str
    document_id: str
    websocket: WebSocketServerProtocol
    connected_at: datetime = field(default_factory=datetime.now)
    last_ping: datetime = field(default_factory=datetime.now)
    
    # Awareness (présence utilisateur)
    awareness_state: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire"""
        return {
            "client_id": self.client_id,
            "user_id": self.user_id,
            "document_id": self.document_id,
            "connected_at": self.connected_at.isoformat(),
            "last_ping": self.last_ping.isoformat(),
            "awareness_state": self.awareness_state
        }


class YjsProvider:
    """Provider Y.js WebSocket pour synchronisation temps réel"""
    
    def __init__(self, collaboration_manager: DocumentCollaborationManager):
        self.collaboration_manager = collaboration_manager
        self.cache = get_cache_manager()
        
        # Documents Y.js actifs
        self.documents: Dict[str, YjsDocument] = {}
        
        # Clients connectés
        self.clients: Dict[str, YjsClient] = {}
        self.document_clients: Dict[str, Set[str]] = defaultdict(set)
        
        # Configuration
        self.sync_timeout = 30.0  # secondes
        self.awareness_timeout = 60.0  # secondes
        self.auto_save_interval = 30  # secondes
        
        # Tâches de maintenance
        self.maintenance_task = None
        
    async def initialize(self):
        """Initialise le provider Y.js"""
        try:
            logger.info("🚀 Initialisation Y.js Provider...")
            
            if not YJSPY_AVAILABLE:
                logger.warning("⚠️ Y.js (ypy) non disponible, utilisation du fallback")
            
            # Démarrer tâche de maintenance
            self.maintenance_task = asyncio.create_task(self._maintenance_loop())
            
            logger.info("✅ Y.js Provider initialisé")
            
        except Exception as e:
            logger.error(f"❌ Erreur initialisation Y.js Provider: {e}")
            raise
    
    async def shutdown(self):
        """Arrête le provider"""
        if self.maintenance_task:
            self.maintenance_task.cancel()
        
        # Sauvegarder tous les documents
        for doc in self.documents.values():
            await self._save_document(doc)
    
    async def handle_client_connection(
        self, 
        document_id: str, 
        user_id: str, 
        websocket: WebSocketServerProtocol
    ) -> str:
        """Gère la connexion d'un client Y.js"""
        
        try:
            client_id = str(uuid.uuid4())
            
            # Créer client
            client = YjsClient(
                client_id=client_id,
                user_id=user_id,
                document_id=document_id,
                websocket=websocket
            )
            
            # Ajouter aux registres
            self.clients[client_id] = client
            self.document_clients[document_id].add(client_id)
            
            # Obtenir ou créer document Y.js
            ydoc = await self._get_or_create_document(document_id)
            ydoc.active_clients.add(client_id)
            
            # Envoyer synchronisation initiale
            await self._send_sync_step_1(client, ydoc)
            
            logger.info(f"🔌 Client Y.js connecté: {client_id} (user: {user_id}, doc: {document_id})")
            
            return client_id
            
        except Exception as e:
            logger.error(f"❌ Erreur connexion client Y.js: {e}")
            raise
    
    async def handle_client_disconnection(self, client_id: str):
        """Gère la déconnexion d'un client"""
        
        try:
            client = self.clients.get(client_id)
            if not client:
                return
            
            # Supprimer des registres
            document_id = client.document_id
            
            if document_id in self.document_clients:
                self.document_clients[document_id].discard(client_id)
                
                # Supprimer document si plus de clients
                if not self.document_clients[document_id]:
                    if document_id in self.documents:
                        await self._save_document(self.documents[document_id])
                        del self.documents[document_id]
                    del self.document_clients[document_id]
            
            if document_id in self.documents:
                self.documents[document_id].active_clients.discard(client_id)
            
            del self.clients[client_id]
            
            # Notifier autres clients de la déconnexion
            await self._broadcast_awareness_update(document_id, client_id, None)
            
            logger.info(f"🔌 Client Y.js déconnecté: {client_id}")
            
        except Exception as e:
            logger.error(f"❌ Erreur déconnexion client Y.js: {e}")
    
    async def handle_message(self, client_id: str, message: bytes):
        """Gère un message WebSocket Y.js"""
        
        try:
            client = self.clients.get(client_id)
            if not client:
                logger.warning(f"⚠️ Message reçu pour client inexistant: {client_id}")
                return
            
            # Décoder message Y.js (format binaire)
            if len(message) < 1:
                return
            
            message_type = message[0]
            content = message[1:] if len(message) > 1 else b''
            
            await self._process_yjs_message(client, message_type, content)
            
        except Exception as e:
            logger.error(f"❌ Erreur traitement message Y.js: {e}")
    
    async def _process_yjs_message(self, client: YjsClient, message_type: int, content: bytes):
        """Traite un message Y.js spécifique"""
        
        ydoc = self.documents.get(client.document_id)
        if not ydoc:
            logger.warning(f"⚠️ Document Y.js non trouvé: {client.document_id}")
            return
        
        try:
            if message_type == 0:  # SYNC_STEP_1
                await self._handle_sync_step_1(client, ydoc, content)
            
            elif message_type == 1:  # SYNC_STEP_2  
                await self._handle_sync_step_2(client, ydoc, content)
            
            elif message_type == 2:  # UPDATE
                await self._handle_update(client, ydoc, content)
            
            elif message_type == 3:  # AWARENESS
                await self._handle_awareness(client, content)
            
            elif message_type == 4:  # QUERY_AWARENESS
                await self._handle_query_awareness(client)
            
            else:
                logger.warning(f"⚠️ Type de message Y.js inconnu: {message_type}")
                
        except Exception as e:
            logger.error(f"❌ Erreur traitement message Y.js type {message_type}: {e}")
    
    async def _handle_sync_step_1(self, client: YjsClient, ydoc: YjsDocument, state_vector: bytes):
        """Gère SYNC_STEP_1 - client envoie son state vector"""
        
        try:
            # Calculer diff depuis le state vector du client
            update = ydoc.get_update_since(state_vector)
            
            if update:
                # Envoyer SYNC_STEP_2 avec les mises à jour
                message = bytes([1]) + update  # Type 1 = SYNC_STEP_2
                await client.websocket.send(message)
                
                logger.debug(f"📤 SYNC_STEP_2 envoyé à {client.client_id}: {len(update)} bytes")
        
        except Exception as e:
            logger.error(f"❌ Erreur SYNC_STEP_1: {e}")
    
    async def _handle_sync_step_2(self, client: YjsClient, ydoc: YjsDocument, update: bytes):
        """Gère SYNC_STEP_2 - client envoie ses mises à jour"""
        
        try:
            if update:
                # Appliquer les mises à jour au document
                if ydoc.apply_update(update):
                    # Diffuser aux autres clients
                    await self._broadcast_update(client.document_id, update, exclude_client=client.client_id)
                    
                    logger.debug(f"📥 SYNC_STEP_2 reçu de {client.client_id}: {len(update)} bytes")
        
        except Exception as e:
            logger.error(f"❌ Erreur SYNC_STEP_2: {e}")
    
    async def _handle_update(self, client: YjsClient, ydoc: YjsDocument, update: bytes):
        """Gère UPDATE - mise à jour temps réel"""
        
        try:
            if update:
                # Appliquer mise à jour
                if ydoc.apply_update(update):
                    # Diffuser aux autres clients  
                    await self._broadcast_update(client.document_id, update, exclude_client=client.client_id)
                    
                    # Synchroniser avec le système de collaboration existant
                    await self._sync_with_collaboration_system(client, ydoc)
                    
                    logger.debug(f"📥 UPDATE reçu de {client.client_id}: {len(update)} bytes")
        
        except Exception as e:
            logger.error(f"❌ Erreur UPDATE: {e}")
    
    async def _handle_awareness(self, client: YjsClient, awareness_update: bytes):
        """Gère AWARENESS - état de présence utilisateur"""
        
        try:
            # Décoder l'awareness (simplifié pour cette démo)
            # Dans un vrai système, décoder le format Y.js awareness
            
            client.awareness_state = {
                "timestamp": datetime.now().isoformat(),
                "update_size": len(awareness_update)
            }
            client.last_ping = datetime.now()
            
            # Diffuser awareness aux autres clients
            await self._broadcast_awareness_update(
                client.document_id, 
                client.client_id, 
                awareness_update
            )
            
            logger.debug(f"👁️ Awareness reçu de {client.client_id}")
        
        except Exception as e:
            logger.error(f"❌ Erreur AWARENESS: {e}")
    
    async def _handle_query_awareness(self, client: YjsClient):
        """Gère QUERY_AWARENESS - demande d'état awareness"""
        
        try:
            # Envoyer awareness de tous les autres clients
            for other_client_id in self.document_clients.get(client.document_id, set()):
                if other_client_id != client.client_id:
                    other_client = self.clients.get(other_client_id)
                    if other_client and other_client.awareness_state:
                        # Envoyer awareness de l'autre client
                        awareness_message = bytes([3]) + b'awareness_placeholder'
                        await client.websocket.send(awareness_message)
            
            logger.debug(f"👁️ Query awareness traité pour {client.client_id}")
        
        except Exception as e:
            logger.error(f"❌ Erreur QUERY_AWARENESS: {e}")
    
    async def _send_sync_step_1(self, client: YjsClient, ydoc: YjsDocument):
        """Envoie SYNC_STEP_1 au client"""
        
        try:
            # Envoyer state vector pour initier la synchronisation
            state_vector = ydoc.get_state_vector()
            message = bytes([0]) + state_vector  # Type 0 = SYNC_STEP_1
            
            await client.websocket.send(message)
            
            logger.debug(f"📤 SYNC_STEP_1 envoyé à {client.client_id}")
        
        except Exception as e:
            logger.error(f"❌ Erreur envoi SYNC_STEP_1: {e}")
    
    async def _broadcast_update(self, document_id: str, update: bytes, exclude_client: str = None):
        """Diffuse une mise à jour à tous les clients d'un document"""
        
        try:
            client_ids = self.document_clients.get(document_id, set())
            
            message = bytes([2]) + update  # Type 2 = UPDATE
            
            for client_id in client_ids:
                if client_id != exclude_client:
                    client = self.clients.get(client_id)
                    if client:
                        try:
                            await client.websocket.send(message)
                        except ConnectionClosed:
                            # Client déconnecté
                            await self.handle_client_disconnection(client_id)
                        except Exception as e:
                            logger.error(f"❌ Erreur diffusion update à {client_id}: {e}")
        
        except Exception as e:
            logger.error(f"❌ Erreur broadcast update: {e}")
    
    async def _broadcast_awareness_update(
        self, 
        document_id: str, 
        from_client_id: str, 
        awareness_update: Optional[bytes]
    ):
        """Diffuse une mise à jour awareness"""
        
        try:
            client_ids = self.document_clients.get(document_id, set())
            
            if awareness_update is not None:
                message = bytes([3]) + awareness_update  # Type 3 = AWARENESS
            else:
                # Client déconnecté, envoyer awareness vide
                message = bytes([3]) + b''
            
            for client_id in client_ids:
                if client_id != from_client_id:
                    client = self.clients.get(client_id)
                    if client:
                        try:
                            await client.websocket.send(message)
                        except ConnectionClosed:
                            await self.handle_client_disconnection(client_id)
                        except Exception as e:
                            logger.error(f"❌ Erreur diffusion awareness à {client_id}: {e}")
        
        except Exception as e:
            logger.error(f"❌ Erreur broadcast awareness: {e}")
    
    async def _get_or_create_document(self, document_id: str) -> YjsDocument:
        """Obtient ou crée un document Y.js"""
        
        if document_id in self.documents:
            return self.documents[document_id]
        
        try:
            # Créer nouveau document Y.js
            ydoc_instance = Y.YDoc()
            
            # Charger contenu existant depuis le système de collaboration
            existing_content = await self._load_existing_content(document_id)
            if existing_content:
                # Initialiser le document avec le contenu existant
                ytext = ydoc_instance.get_text("default")
                ytext.insert(0, existing_content)
            
            ydoc = YjsDocument(
                document_id=document_id,
                ydoc=ydoc_instance
            )
            
            self.documents[document_id] = ydoc
            
            logger.info(f"📄 Document Y.js créé: {document_id}")
            
            return ydoc
            
        except Exception as e:
            logger.error(f"❌ Erreur création document Y.js: {e}")
            raise
    
    async def _load_existing_content(self, document_id: str) -> str:
        """Charge le contenu existant d'un document"""
        
        try:
            # Vérifier dans les sessions de collaboration actives
            for session in self.collaboration_manager.active_sessions.values():
                if session.document_id == document_id:
                    return session.document_content
            
            # Sinon, charger depuis le cache ou la base de données
            cache_key = f"document_content:{document_id}"
            cached_content = await self.cache.get(cache_key)
            
            if cached_content:
                return json.loads(cached_content).get("content", "")
            
            return ""
            
        except Exception as e:
            logger.error(f"❌ Erreur chargement contenu existant: {e}")
            return ""
    
    async def _save_document(self, ydoc: YjsDocument):
        """Sauvegarde un document Y.js"""
        
        try:
            # Extraire contenu texte
            content = ydoc.get_text_content()
            
            # Sauvegarder dans le cache
            cache_key = f"document_content:{ydoc.document_id}"
            document_data = {
                "content": content,
                "last_updated": ydoc.last_updated.isoformat(),
                "total_updates": ydoc.total_updates
            }
            
            await self.cache.set(cache_key, json.dumps(document_data), expire=86400)
            
            # Synchroniser avec le système de collaboration
            await self._sync_document_with_collaboration(ydoc, content)
            
            logger.debug(f"💾 Document Y.js sauvegardé: {ydoc.document_id}")
            
        except Exception as e:
            logger.error(f"❌ Erreur sauvegarde document Y.js: {e}")
    
    async def _sync_with_collaboration_system(self, client: YjsClient, ydoc: YjsDocument):
        """Synchronise avec le système de collaboration existant"""
        
        try:
            # Trouver session de collaboration correspondante
            collaboration_session = None
            for session in self.collaboration_manager.active_sessions.values():
                if session.document_id == client.document_id:
                    collaboration_session = session
                    break
            
            if collaboration_session:
                # Mettre à jour le contenu dans la session de collaboration
                new_content = ydoc.get_text_content()
                collaboration_session.document_content = new_content
                collaboration_session.document_version += 1
                collaboration_session.last_activity = datetime.now()
        
        except Exception as e:
            logger.error(f"❌ Erreur sync collaboration system: {e}")
    
    async def _sync_document_with_collaboration(self, ydoc: YjsDocument, content: str):
        """Synchronise un document avec le système de collaboration"""
        
        try:
            # Mettre à jour dans toutes les sessions actives
            for session in self.collaboration_manager.active_sessions.values():
                if session.document_id == ydoc.document_id:
                    session.document_content = content
                    session.document_version += 1
                    session.last_activity = datetime.now()
        
        except Exception as e:
            logger.error(f"❌ Erreur sync document collaboration: {e}")
    
    async def _maintenance_loop(self):
        """Boucle de maintenance pour nettoyage et sauvegarde"""
        
        while True:
            try:
                await asyncio.sleep(self.auto_save_interval)
                
                # Sauvegarder documents modifiés
                for ydoc in self.documents.values():
                    if ydoc.last_updated > datetime.now() - timedelta(seconds=self.auto_save_interval):
                        await self._save_document(ydoc)
                
                # Nettoyer clients inactifs
                now = datetime.now()
                inactive_clients = []
                
                for client_id, client in self.clients.items():
                    if now - client.last_ping > timedelta(seconds=self.awareness_timeout):
                        inactive_clients.append(client_id)
                
                for client_id in inactive_clients:
                    await self.handle_client_disconnection(client_id)
                
                if inactive_clients:
                    logger.info(f"🧹 {len(inactive_clients)} clients Y.js inactifs supprimés")
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ Erreur maintenance Y.js: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Retourne les statistiques Y.js"""
        
        total_clients = len(self.clients)
        total_documents = len(self.documents)
        total_updates = sum(doc.total_updates for doc in self.documents.values())
        
        documents_info = []
        for doc in self.documents.values():
            documents_info.append({
                "document_id": doc.document_id,
                "active_clients": len(doc.active_clients),
                "total_updates": doc.total_updates,
                "last_updated": doc.last_updated.isoformat(),
                "content_length": len(doc.get_text_content())
            })
        
        return {
            "yjs_available": YJSPY_AVAILABLE,
            "total_clients": total_clients,
            "total_documents": total_documents,
            "total_updates": total_updates,
            "documents": documents_info,
            "clients_by_document": {
                doc_id: len(client_ids) 
                for doc_id, client_ids in self.document_clients.items()
            }
        }


# Instance globale
_yjs_provider: Optional[YjsProvider] = None


async def get_yjs_provider() -> YjsProvider:
    """Factory pour obtenir le provider Y.js"""
    global _yjs_provider
    
    if _yjs_provider is None:
        collaboration_manager = await get_document_collaboration_manager()
        _yjs_provider = YjsProvider(collaboration_manager)
        await _yjs_provider.initialize()
    
    return _yjs_provider


async def initialize_yjs_system():
    """Initialise le système Y.js complet"""
    try:
        logger.info("🚀 Initialisation système Y.js...")
        
        # Initialiser le provider
        provider = await get_yjs_provider()
        
        logger.info("✅ Système Y.js initialisé avec succès")
        
        return provider
        
    except Exception as e:
        logger.error(f"❌ Erreur initialisation système Y.js: {e}")
        raise