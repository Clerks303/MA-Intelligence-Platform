"""
Système de collaboration temps réel sur documents
US-012: Collaboration multi-utilisateurs en temps réel pour documents M&A

Ce module fournit:
- Édition collaborative temps réel (WebSocket)
- Gestion des conflits et synchronisation
- Commentaires et annotations en temps réel
- Suivi des modifications par utilisateur
- Notifications push instantanées
- Sessions de travail collaboratif
"""

import asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
import weakref
from collections import defaultdict

import websockets
from websockets.server import WebSocketServerProtocol
from websockets.exceptions import ConnectionClosed

from app.core.document_storage import DocumentMetadata, get_document_storage
from app.core.logging_system import get_logger, LogCategory
from app.core.cache_manager import get_cache_manager

logger = get_logger("document_collaboration", LogCategory.DOCUMENT)


class OperationType(str, Enum):
    """Types d'opérations collaboratives"""
    INSERT = "insert"
    DELETE = "delete"
    RETAIN = "retain"
    FORMAT = "format"
    COMMENT = "comment"
    ANNOTATION = "annotation"
    CURSOR_MOVE = "cursor_move"
    SELECTION = "selection"


class EventType(str, Enum):
    """Types d'événements de collaboration"""
    USER_JOIN = "user_join"
    USER_LEAVE = "user_leave"
    DOCUMENT_OPEN = "document_open"
    DOCUMENT_CLOSE = "document_close"
    OPERATION = "operation"
    CONFLICT_RESOLUTION = "conflict_resolution"
    COMMENT_ADD = "comment_add"
    COMMENT_EDIT = "comment_edit"
    COMMENT_DELETE = "comment_delete"
    NOTIFICATION = "notification"
    CURSOR_UPDATE = "cursor_update"
    SELECTION_UPDATE = "selection_update"


class ConflictResolutionStrategy(str, Enum):
    """Stratégies de résolution de conflits"""
    LAST_WRITER_WINS = "last_writer_wins"
    OPERATIONAL_TRANSFORM = "operational_transform"
    MERGE_CHANGES = "merge_changes"
    MANUAL_RESOLUTION = "manual_resolution"


@dataclass
class Operation:
    """Opération dans le document"""
    
    operation_id: str
    operation_type: OperationType
    position: int
    content: str = ""
    length: int = 0
    attributes: Dict[str, Any] = field(default_factory=dict)
    
    # Métadonnées de l'opération
    user_id: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    document_version: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire"""
        return {
            "operation_id": self.operation_id,
            "operation_type": self.operation_type.value,
            "position": self.position,
            "content": self.content,
            "length": self.length,
            "attributes": self.attributes,
            "user_id": self.user_id,
            "timestamp": self.timestamp.isoformat(),
            "document_version": self.document_version
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Operation':
        """Crée depuis un dictionnaire"""
        if 'timestamp' in data and isinstance(data['timestamp'], str):
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        if 'operation_type' in data:
            data['operation_type'] = OperationType(data['operation_type'])
        return cls(**data)


@dataclass
class Comment:
    """Commentaire sur le document"""
    
    comment_id: str
    document_id: str
    position: int
    content: str
    author_id: str
    created_at: datetime = field(default_factory=datetime.now)
    
    # Métadonnées du commentaire
    is_resolved: bool = False
    resolved_by: Optional[str] = None
    resolved_at: Optional[datetime] = None
    
    # Réponses
    replies: List['Comment'] = field(default_factory=list)
    parent_comment_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire"""
        return {
            "comment_id": self.comment_id,
            "document_id": self.document_id,
            "position": self.position,
            "content": self.content,
            "author_id": self.author_id,
            "created_at": self.created_at.isoformat(),
            "is_resolved": self.is_resolved,
            "resolved_by": self.resolved_by,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "replies": [reply.to_dict() for reply in self.replies],
            "parent_comment_id": self.parent_comment_id
        }


@dataclass
class CollaboratorCursor:
    """Curseur d'un collaborateur"""
    
    user_id: str
    position: int
    selection_start: Optional[int] = None
    selection_end: Optional[int] = None
    last_update: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire"""
        return {
            "user_id": self.user_id,
            "position": self.position,
            "selection_start": self.selection_start,
            "selection_end": self.selection_end,
            "last_update": self.last_update.isoformat()
        }


@dataclass
class CollaborationSession:
    """Session de collaboration sur un document"""
    
    session_id: str
    document_id: str
    created_at: datetime = field(default_factory=datetime.now)
    
    # Participants
    active_users: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    user_cursors: Dict[str, CollaboratorCursor] = field(default_factory=dict)
    
    # État du document
    document_content: str = ""
    document_version: int = 0
    operations_history: List[Operation] = field(default_factory=list)
    
    # Commentaires et annotations
    comments: Dict[str, Comment] = field(default_factory=dict)
    
    # Connexions WebSocket
    connections: Dict[str, WebSocketServerProtocol] = field(default_factory=dict)
    
    # Statistiques
    total_operations: int = 0
    total_comments: int = 0
    last_activity: datetime = field(default_factory=datetime.now)


class OperationalTransform:
    """Transformateur opérationnel pour résolution de conflits"""
    
    @staticmethod
    def transform_operation(op1: Operation, op2: Operation) -> tuple[Operation, Operation]:
        """Transforme deux opérations concurrentes"""
        
        # Implémentation simplifiée de l'Operational Transform
        # Dans un vrai système, implémenter l'algorithme complet
        
        if op1.operation_type == OperationType.INSERT and op2.operation_type == OperationType.INSERT:
            # Deux insertions
            if op1.position <= op2.position:
                # op2 doit être décalée
                op2_transformed = Operation(
                    operation_id=op2.operation_id,
                    operation_type=op2.operation_type,
                    position=op2.position + len(op1.content),
                    content=op2.content,
                    user_id=op2.user_id,
                    timestamp=op2.timestamp,
                    document_version=op2.document_version
                )
                return op1, op2_transformed
            else:
                # op1 doit être décalée
                op1_transformed = Operation(
                    operation_id=op1.operation_id,
                    operation_type=op1.operation_type,
                    position=op1.position + len(op2.content),
                    content=op1.content,
                    user_id=op1.user_id,
                    timestamp=op1.timestamp,
                    document_version=op1.document_version
                )
                return op1_transformed, op2
        
        elif op1.operation_type == OperationType.DELETE and op2.operation_type == OperationType.DELETE:
            # Deux suppressions
            if op1.position <= op2.position:
                if op1.position + op1.length <= op2.position:
                    # Suppressions non overlapping
                    op2_transformed = Operation(
                        operation_id=op2.operation_id,
                        operation_type=op2.operation_type,
                        position=op2.position - op1.length,
                        length=op2.length,
                        user_id=op2.user_id,
                        timestamp=op2.timestamp,
                        document_version=op2.document_version
                    )
                    return op1, op2_transformed
                else:
                    # Suppressions overlapping - résolution complexe
                    # Pour simplifier, on garde la première opération
                    return op1, Operation(
                        operation_id=op2.operation_id,
                        operation_type=OperationType.RETAIN,
                        position=0,
                        user_id=op2.user_id,
                        timestamp=op2.timestamp,
                        document_version=op2.document_version
                    )
        
        elif op1.operation_type == OperationType.INSERT and op2.operation_type == OperationType.DELETE:
            # Insertion + suppression
            if op1.position <= op2.position:
                op2_transformed = Operation(
                    operation_id=op2.operation_id,
                    operation_type=op2.operation_type,
                    position=op2.position + len(op1.content),
                    length=op2.length,
                    user_id=op2.user_id,
                    timestamp=op2.timestamp,
                    document_version=op2.document_version
                )
                return op1, op2_transformed
            else:
                return op1, op2
        
        elif op1.operation_type == OperationType.DELETE and op2.operation_type == OperationType.INSERT:
            # Suppression + insertion
            if op2.position <= op1.position:
                op1_transformed = Operation(
                    operation_id=op1.operation_id,
                    operation_type=op1.operation_type,
                    position=op1.position + len(op2.content),
                    length=op1.length,
                    user_id=op1.user_id,
                    timestamp=op1.timestamp,
                    document_version=op1.document_version
                )
                return op1_transformed, op2
            else:
                return op1, op2
        
        # Par défaut, retourner les opérations inchangées
        return op1, op2
    
    @staticmethod
    def apply_operation(content: str, operation: Operation) -> str:
        """Applique une opération au contenu"""
        
        if operation.operation_type == OperationType.INSERT:
            return content[:operation.position] + operation.content + content[operation.position:]
        
        elif operation.operation_type == OperationType.DELETE:
            return content[:operation.position] + content[operation.position + operation.length:]
        
        elif operation.operation_type == OperationType.RETAIN:
            return content
        
        else:
            return content


class WebSocketManager:
    """Gestionnaire des connexions WebSocket"""
    
    def __init__(self):
        self.connections: Dict[str, WebSocketServerProtocol] = {}
        self.user_sessions: Dict[str, Set[str]] = defaultdict(set)  # user_id -> session_ids
        
    async def add_connection(self, user_id: str, session_id: str, websocket: WebSocketServerProtocol):
        """Ajoute une connexion WebSocket"""
        connection_id = f"{user_id}:{session_id}"
        self.connections[connection_id] = websocket
        self.user_sessions[user_id].add(session_id)
        
        logger.info(f"🔌 Connexion WebSocket ajoutée: {connection_id}")
    
    async def remove_connection(self, user_id: str, session_id: str):
        """Supprime une connexion WebSocket"""
        connection_id = f"{user_id}:{session_id}"
        
        if connection_id in self.connections:
            del self.connections[connection_id]
        
        if user_id in self.user_sessions:
            self.user_sessions[user_id].discard(session_id)
            if not self.user_sessions[user_id]:
                del self.user_sessions[user_id]
        
        logger.info(f"🔌 Connexion WebSocket supprimée: {connection_id}")
    
    async def broadcast_to_session(self, session_id: str, message: Dict[str, Any], exclude_user: str = None):
        """Diffuse un message à tous les utilisateurs d'une session"""
        
        message_json = json.dumps(message)
        disconnected = []
        
        for connection_id, websocket in self.connections.items():
            user_id, conn_session_id = connection_id.split(":", 1)
            
            if conn_session_id == session_id and user_id != exclude_user:
                try:
                    await websocket.send(message_json)
                except ConnectionClosed:
                    disconnected.append(connection_id)
                except Exception as e:
                    logger.error(f"❌ Erreur envoi WebSocket {connection_id}: {e}")
                    disconnected.append(connection_id)
        
        # Nettoyer connexions fermées
        for connection_id in disconnected:
            user_id, conn_session_id = connection_id.split(":", 1)
            await self.remove_connection(user_id, conn_session_id)
    
    async def send_to_user(self, user_id: str, session_id: str, message: Dict[str, Any]):
        """Envoie un message à un utilisateur spécifique"""
        
        connection_id = f"{user_id}:{session_id}"
        
        if connection_id in self.connections:
            try:
                await self.connections[connection_id].send(json.dumps(message))
            except ConnectionClosed:
                await self.remove_connection(user_id, session_id)
            except Exception as e:
                logger.error(f"❌ Erreur envoi WebSocket {connection_id}: {e}")


class DocumentCollaborationManager:
    """Gestionnaire principal de collaboration documentaire"""
    
    def __init__(self):
        self.active_sessions: Dict[str, CollaborationSession] = {}
        self.websocket_manager = WebSocketManager()
        self.operational_transform = OperationalTransform()
        self.cache = get_cache_manager()
        
        # Configuration
        self.max_operations_history = 1000
        self.session_timeout = timedelta(hours=24)
        self.cursor_update_interval = timedelta(seconds=1)
        
        # Tâches de nettoyage
        self.cleanup_task = None
    
    async def initialize(self):
        """Initialise le gestionnaire de collaboration"""
        try:
            logger.info("🚀 Initialisation du gestionnaire de collaboration...")
            
            # Démarrer tâche de nettoyage
            self.cleanup_task = asyncio.create_task(self._cleanup_sessions())
            
            logger.info("✅ Gestionnaire de collaboration initialisé")
            
        except Exception as e:
            logger.error(f"❌ Erreur initialisation collaboration: {e}")
            raise
    
    async def shutdown(self):
        """Arrête le gestionnaire"""
        if self.cleanup_task:
            self.cleanup_task.cancel()
    
    async def join_session(
        self, 
        document_id: str, 
        user_id: str, 
        websocket: WebSocketServerProtocol,
        user_info: Dict[str, Any] = None
    ) -> str:
        """Fait rejoindre un utilisateur à une session de collaboration"""
        
        try:
            # Créer ou récupérer session
            session = await self._get_or_create_session(document_id)
            
            # Ajouter utilisateur à la session
            session.active_users[user_id] = {
                "user_id": user_id,
                "joined_at": datetime.now(),
                "user_info": user_info or {},
                "is_active": True
            }
            
            # Initialiser curseur
            session.user_cursors[user_id] = CollaboratorCursor(
                user_id=user_id,
                position=0
            )
            
            # Ajouter connexion WebSocket
            await self.websocket_manager.add_connection(user_id, session.session_id, websocket)
            
            # Envoyer état initial à l'utilisateur
            await self._send_initial_state(session, user_id)
            
            # Notifier les autres utilisateurs
            await self._broadcast_user_event(session, user_id, EventType.USER_JOIN)
            
            session.last_activity = datetime.now()
            
            logger.info(f"👥 Utilisateur {user_id} a rejoint session {session.session_id}")
            
            return session.session_id
            
        except Exception as e:
            logger.error(f"❌ Erreur rejoindre session: {e}")
            raise
    
    async def leave_session(self, session_id: str, user_id: str):
        """Fait quitter un utilisateur de la session"""
        
        try:
            session = self.active_sessions.get(session_id)
            if not session:
                return
            
            # Supprimer utilisateur
            if user_id in session.active_users:
                del session.active_users[user_id]
            
            if user_id in session.user_cursors:
                del session.user_cursors[user_id]
            
            # Supprimer connexion WebSocket
            await self.websocket_manager.remove_connection(user_id, session_id)
            
            # Notifier les autres utilisateurs
            await self._broadcast_user_event(session, user_id, EventType.USER_LEAVE)
            
            # Supprimer session si plus d'utilisateurs
            if not session.active_users:
                await self._close_session(session_id)
            
            logger.info(f"👥 Utilisateur {user_id} a quitté session {session_id}")
            
        except Exception as e:
            logger.error(f"❌ Erreur quitter session: {e}")
    
    async def apply_operation(self, session_id: str, user_id: str, operation_data: Dict[str, Any]) -> bool:
        """Applique une opération de modification"""
        
        try:
            session = self.active_sessions.get(session_id)
            if not session:
                raise ValueError("Session non trouvée")
            
            if user_id not in session.active_users:
                raise ValueError("Utilisateur non dans la session")
            
            # Créer opération
            operation = Operation(
                operation_id=str(uuid.uuid4()),
                operation_type=OperationType(operation_data["operation_type"]),
                position=operation_data["position"],
                content=operation_data.get("content", ""),
                length=operation_data.get("length", 0),
                attributes=operation_data.get("attributes", {}),
                user_id=user_id,
                document_version=session.document_version
            )
            
            # Transformer avec opérations concurrentes
            transformed_operation = await self._transform_operation(session, operation)
            
            # Appliquer l'opération
            session.document_content = self.operational_transform.apply_operation(
                session.document_content, 
                transformed_operation
            )
            
            # Mettre à jour version
            session.document_version += 1
            transformed_operation.document_version = session.document_version
            
            # Ajouter à l'historique
            session.operations_history.append(transformed_operation)
            
            # Limiter historique
            if len(session.operations_history) > self.max_operations_history:
                session.operations_history = session.operations_history[-self.max_operations_history:]
            
            # Diffuser aux autres utilisateurs
            await self._broadcast_operation(session, transformed_operation, exclude_user=user_id)
            
            session.total_operations += 1
            session.last_activity = datetime.now()
            
            logger.debug(f"✏️ Opération appliquée: {operation.operation_type.value} par {user_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur application opération: {e}")
            return False
    
    async def add_comment(self, session_id: str, user_id: str, comment_data: Dict[str, Any]) -> str:
        """Ajoute un commentaire"""
        
        try:
            session = self.active_sessions.get(session_id)
            if not session:
                raise ValueError("Session non trouvée")
            
            # Créer commentaire
            comment = Comment(
                comment_id=str(uuid.uuid4()),
                document_id=session.document_id,
                position=comment_data["position"],
                content=comment_data["content"],
                author_id=user_id,
                parent_comment_id=comment_data.get("parent_comment_id")
            )
            
            # Ajouter à la session
            session.comments[comment.comment_id] = comment
            
            # Si c'est une réponse, l'ajouter au commentaire parent
            if comment.parent_comment_id and comment.parent_comment_id in session.comments:
                session.comments[comment.parent_comment_id].replies.append(comment)
            
            # Diffuser le commentaire
            await self._broadcast_comment_event(session, comment, EventType.COMMENT_ADD)
            
            session.total_comments += 1
            session.last_activity = datetime.now()
            
            logger.info(f"💬 Commentaire ajouté par {user_id} dans session {session_id}")
            
            return comment.comment_id
            
        except Exception as e:
            logger.error(f"❌ Erreur ajout commentaire: {e}")
            raise
    
    async def update_cursor(self, session_id: str, user_id: str, cursor_data: Dict[str, Any]):
        """Met à jour la position du curseur d'un utilisateur"""
        
        try:
            session = self.active_sessions.get(session_id)
            if not session or user_id not in session.user_cursors:
                return
            
            # Mettre à jour curseur
            cursor = session.user_cursors[user_id]
            cursor.position = cursor_data["position"]
            cursor.selection_start = cursor_data.get("selection_start")
            cursor.selection_end = cursor_data.get("selection_end")
            cursor.last_update = datetime.now()
            
            # Diffuser mise à jour (throttling)
            last_broadcast = getattr(cursor, '_last_broadcast', datetime.min)
            if datetime.now() - last_broadcast > self.cursor_update_interval:
                await self._broadcast_cursor_update(session, cursor, exclude_user=user_id)
                cursor._last_broadcast = datetime.now()
            
        except Exception as e:
            logger.error(f"❌ Erreur mise à jour curseur: {e}")
    
    async def _get_or_create_session(self, document_id: str) -> CollaborationSession:
        """Récupère ou crée une session de collaboration"""
        
        # Chercher session existante
        for session in self.active_sessions.values():
            if session.document_id == document_id:
                return session
        
        # Créer nouvelle session
        session_id = str(uuid.uuid4())
        
        # Récupérer contenu du document
        document_storage = await get_document_storage()
        metadata = document_storage.get_document_metadata(document_id)
        
        if not metadata:
            raise ValueError(f"Document non trouvé: {document_id}")
        
        # Charger contenu (simulation pour cette démo)
        # Dans un vrai système, récupérer le contenu réel
        document_content = metadata.extracted_text or ""
        
        session = CollaborationSession(
            session_id=session_id,
            document_id=document_id,
            document_content=document_content
        )
        
        self.active_sessions[session_id] = session
        
        logger.info(f"📄 Nouvelle session créée: {session_id} pour document {document_id}")
        
        return session
    
    async def _close_session(self, session_id: str):
        """Ferme une session de collaboration"""
        
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            
            # Sauvegarder état final du document
            await self._save_session_state(session)
            
            # Supprimer session
            del self.active_sessions[session_id]
            
            logger.info(f"📄 Session fermée: {session_id}")
    
    async def _send_initial_state(self, session: CollaborationSession, user_id: str):
        """Envoie l'état initial à un utilisateur"""
        
        message = {
            "type": "initial_state",
            "session_id": session.session_id,
            "document_id": session.document_id,
            "document_content": session.document_content,
            "document_version": session.document_version,
            "active_users": [
                {
                    "user_id": uid,
                    "user_info": info.get("user_info", {}),
                    "joined_at": info["joined_at"].isoformat()
                }
                for uid, info in session.active_users.items()
                if uid != user_id  # Exclure l'utilisateur lui-même
            ],
            "cursors": [
                cursor.to_dict() for cursor in session.user_cursors.values()
                if cursor.user_id != user_id
            ],
            "comments": [
                comment.to_dict() for comment in session.comments.values()
                if not comment.parent_comment_id  # Seulement commentaires racine
            ]
        }
        
        await self.websocket_manager.send_to_user(user_id, session.session_id, message)
    
    async def _broadcast_user_event(self, session: CollaborationSession, user_id: str, event_type: EventType):
        """Diffuse un événement utilisateur"""
        
        user_info = session.active_users.get(user_id, {})
        
        message = {
            "type": event_type.value,
            "user_id": user_id,
            "user_info": user_info.get("user_info", {}),
            "timestamp": datetime.now().isoformat()
        }
        
        await self.websocket_manager.broadcast_to_session(
            session.session_id, 
            message, 
            exclude_user=user_id
        )
    
    async def _broadcast_operation(self, session: CollaborationSession, operation: Operation, exclude_user: str):
        """Diffuse une opération aux autres utilisateurs"""
        
        message = {
            "type": "operation",
            "operation": operation.to_dict()
        }
        
        await self.websocket_manager.broadcast_to_session(
            session.session_id, 
            message, 
            exclude_user=exclude_user
        )
    
    async def _broadcast_comment_event(self, session: CollaborationSession, comment: Comment, event_type: EventType):
        """Diffuse un événement de commentaire"""
        
        message = {
            "type": event_type.value,
            "comment": comment.to_dict(),
            "timestamp": datetime.now().isoformat()
        }
        
        await self.websocket_manager.broadcast_to_session(session.session_id, message)
    
    async def _broadcast_cursor_update(self, session: CollaborationSession, cursor: CollaboratorCursor, exclude_user: str):
        """Diffuse une mise à jour de curseur"""
        
        message = {
            "type": "cursor_update",
            "cursor": cursor.to_dict()
        }
        
        await self.websocket_manager.broadcast_to_session(
            session.session_id, 
            message, 
            exclude_user=exclude_user
        )
    
    async def _transform_operation(self, session: CollaborationSession, operation: Operation) -> Operation:
        """Transforme une opération avec les opérations concurrentes"""
        
        # Récupérer opérations concurrentes depuis la version de base
        concurrent_operations = [
            op for op in session.operations_history
            if op.document_version > operation.document_version
        ]
        
        transformed_operation = operation
        
        # Transformer avec chaque opération concurrente
        for concurrent_op in concurrent_operations:
            transformed_operation, _ = self.operational_transform.transform_operation(
                transformed_operation, 
                concurrent_op
            )
        
        return transformed_operation
    
    async def _save_session_state(self, session: CollaborationSession):
        """Sauvegarde l'état d'une session"""
        
        try:
            # Dans un vrai système, sauvegarder dans la base de données
            # Ici, utiliser le cache pour la démo
            
            session_data = {
                "document_id": session.document_id,
                "final_content": session.document_content,
                "final_version": session.document_version,
                "total_operations": session.total_operations,
                "total_comments": session.total_comments,
                "duration": (datetime.now() - session.created_at).total_seconds(),
                "participants": list(session.active_users.keys())
            }
            
            cache_key = f"session_history:{session.session_id}"
            await self.cache.set(cache_key, json.dumps(session_data), expire=86400)  # 24h
            
            logger.info(f"💾 État session sauvegardé: {session.session_id}")
            
        except Exception as e:
            logger.error(f"❌ Erreur sauvegarde session: {e}")
    
    async def _cleanup_sessions(self):
        """Tâche de nettoyage des sessions inactives"""
        
        while True:
            try:
                await asyncio.sleep(300)  # Vérifier toutes les 5 minutes
                
                now = datetime.now()
                inactive_sessions = []
                
                for session_id, session in self.active_sessions.items():
                    # Session inactive si aucun utilisateur actif ou timeout
                    if (not session.active_users or 
                        now - session.last_activity > self.session_timeout):
                        inactive_sessions.append(session_id)
                
                # Fermer sessions inactives
                for session_id in inactive_sessions:
                    await self._close_session(session_id)
                
                if inactive_sessions:
                    logger.info(f"🧹 {len(inactive_sessions)} sessions inactives fermées")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ Erreur nettoyage sessions: {e}")
    
    def get_active_sessions(self) -> List[Dict[str, Any]]:
        """Retourne la liste des sessions actives"""
        
        sessions = []
        
        for session in self.active_sessions.values():
            sessions.append({
                "session_id": session.session_id,
                "document_id": session.document_id,
                "active_users_count": len(session.active_users),
                "active_users": list(session.active_users.keys()),
                "created_at": session.created_at.isoformat(),
                "last_activity": session.last_activity.isoformat(),
                "total_operations": session.total_operations,
                "total_comments": session.total_comments
            })
        
        return sessions
    
    def get_collaboration_statistics(self) -> Dict[str, Any]:
        """Retourne les statistiques de collaboration"""
        
        total_sessions = len(self.active_sessions)
        total_users = sum(len(session.active_users) for session in self.active_sessions.values())
        total_operations = sum(session.total_operations for session in self.active_sessions.values())
        total_comments = sum(session.total_comments for session in self.active_sessions.values())
        
        return {
            "active_sessions": total_sessions,
            "total_active_users": total_users,
            "total_operations": total_operations,
            "total_comments": total_comments,
            "websocket_connections": len(self.websocket_manager.connections),
            "average_users_per_session": total_users / max(total_sessions, 1)
        }


# Instance globale
_document_collaboration_manager: Optional[DocumentCollaborationManager] = None


async def get_document_collaboration_manager() -> DocumentCollaborationManager:
    """Factory pour obtenir le gestionnaire de collaboration"""
    global _document_collaboration_manager
    
    if _document_collaboration_manager is None:
        _document_collaboration_manager = DocumentCollaborationManager()
        await _document_collaboration_manager.initialize()
    
    return _document_collaboration_manager