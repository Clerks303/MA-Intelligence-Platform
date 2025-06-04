"""    
API Routes pour édition collaborative temps réel
Sprint 5 - WebSocket endpoints pour Y.js et collaboration

Endpoints:
- WebSocket Y.js pour synchronisation documents
- API REST pour gestion collaboration
- Fallback polling pour compatibilité
"""

import json
import asyncio
from datetime import datetime
from typing import Dict, Any, List, Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.core.yjs_integration import get_yjs_provider, YjsProvider
from app.core.document_collaboration import get_document_collaboration_manager
from app.core.dependencies import get_current_active_user
from app.core.logging_system import get_logger, LogCategory
from app.models.user import User

logger = get_logger("collaborative_editing", LogCategory.API)

router = APIRouter(prefix="/collaborative", tags=["collaborative-editing"])


# === MODELS ===

class CollaborationSessionInfo(BaseModel):
    """Informations session de collaboration"""
    session_id: str
    document_id: str
    active_users_count: int
    active_users: List[str]
    created_at: str
    last_activity: str
    total_operations: int
    total_comments: int


class YjsStatistics(BaseModel):
    """Statistiques Y.js"""
    yjs_available: bool
    total_clients: int
    total_documents: int
    total_updates: int
    documents: List[Dict[str, Any]]
    clients_by_document: Dict[str, int]


class DocumentContentRequest(BaseModel):
    """Requête de contenu document"""
    document_id: str


class DocumentContentResponse(BaseModel):
    """Réponse contenu document"""
    document_id: str
    content: str
    version: int
    last_updated: str
    active_collaborators: int


class PresenceInfo(BaseModel):
    """Informations de présence utilisateur"""
    user_id: str
    user_info: Dict[str, Any]
    document_id: str
    connected_at: str
    last_activity: str
    cursor_position: Optional[int] = None
    selection_start: Optional[int] = None
    selection_end: Optional[int] = None


# === WEBSOCKET ENDPOINTS ===

@router.websocket("/ws/yjs/{document_id}")
async def websocket_yjs_collaboration(
    websocket: WebSocket,
    document_id: str,
    user_id: str = Query(..., description="ID de l'utilisateur"),
    token: str = Query(..., description="Token d'authentification")
):
    """
    WebSocket endpoint pour collaboration Y.js
    Compatible avec le protocole Y.js standard
    """
    
    client_id = None
    
    try:
        # Valider authentification (simulation - dans un vrai système, valider le token)
        if not token or len(token) < 10:
            await websocket.close(code=4001, reason="Token invalide")
            return
        
        # Accepter connexion WebSocket
        await websocket.accept()
        
        logger.info(f"🔌 Connexion Y.js WebSocket: user={user_id}, doc={document_id}")
        
        # Obtenir provider Y.js
        yjs_provider = await get_yjs_provider()
        
        # Connecter client Y.js
        client_id = await yjs_provider.handle_client_connection(
            document_id=document_id,
            user_id=user_id,
            websocket=websocket
        )
        
        # Boucle de réception messages
        while True:
            try:
                # Recevoir message (binaire pour Y.js)
                message = await websocket.receive_bytes()
                
                # Traiter message Y.js
                await yjs_provider.handle_message(client_id, message)
                
            except WebSocketDisconnect:
                logger.info(f"🔌 Client Y.js déconnecté: {client_id}")
                break
                
            except Exception as e:
                logger.error(f"❌ Erreur traitement message Y.js: {e}")
                break
    
    except Exception as e:
        logger.error(f"❌ Erreur WebSocket Y.js: {e}")
        try:
            await websocket.close(code=4000, reason=f"Erreur serveur: {str(e)}")
        except:
            pass
    
    finally:
        # Nettoyer connexion
        if client_id:
            try:
                yjs_provider = await get_yjs_provider()
                await yjs_provider.handle_client_disconnection(client_id)
            except Exception as e:
                logger.error(f"❌ Erreur nettoyage client Y.js: {e}")


@router.websocket("/ws/collaboration/{document_id}")
async def websocket_collaboration_fallback(
    websocket: WebSocket,
    document_id: str,
    user_id: str = Query(..., description="ID de l'utilisateur"),
    user_info: str = Query("{}", description="Informations utilisateur (JSON)")
):
    """
    WebSocket endpoint de fallback pour collaboration
    Utilise le système de collaboration existant (non-Y.js)
    """
    
    session_id = None
    
    try:
        # Parser informations utilisateur
        try:
            user_info_dict = json.loads(user_info)
        except:
            user_info_dict = {"username": user_id}
        
        # Accepter connexion
        await websocket.accept()
        
        logger.info(f"🔌 Connexion collaboration fallback: user={user_id}, doc={document_id}")
        
        # Obtenir gestionnaire de collaboration
        collaboration_manager = await get_document_collaboration_manager()
        
        # Rejoindre session
        session_id = await collaboration_manager.join_session(
            document_id=document_id,
            user_id=user_id,
            websocket=websocket,
            user_info=user_info_dict
        )
        
        # Boucle de réception messages
        while True:
            try:
                # Recevoir message JSON
                data = await websocket.receive_json()
                
                # Traiter différents types de messages
                message_type = data.get("type")
                
                if message_type == "operation":
                    await collaboration_manager.apply_operation(
                        session_id=session_id,
                        user_id=user_id,
                        operation_data=data["operation"]
                    )
                
                elif message_type == "cursor_update":
                    await collaboration_manager.update_cursor(
                        session_id=session_id,
                        user_id=user_id,
                        cursor_data=data["cursor"]
                    )
                
                elif message_type == "comment":
                    await collaboration_manager.add_comment(
                        session_id=session_id,
                        user_id=user_id,
                        comment_data=data["comment"]
                    )
                
                else:
                    logger.warning(f"⚠️ Type de message collaboration inconnu: {message_type}")
                
            except WebSocketDisconnect:
                logger.info(f"🔌 Client collaboration déconnecté: {user_id}")
                break
                
            except Exception as e:
                logger.error(f"❌ Erreur traitement message collaboration: {e}")
                break
    
    except Exception as e:
        logger.error(f"❌ Erreur WebSocket collaboration: {e}")
        try:
            await websocket.close(code=4000, reason=f"Erreur serveur: {str(e)}")
        except:
            pass
    
    finally:
        # Nettoyer session
        if session_id:
            try:
                collaboration_manager = await get_document_collaboration_manager()
                await collaboration_manager.leave_session(session_id, user_id)
            except Exception as e:
                logger.error(f"❌ Erreur nettoyage session collaboration: {e}")


# === REST API ENDPOINTS ===

@router.get("/sessions", response_model=List[CollaborationSessionInfo])
async def get_active_sessions(
    current_user: User = Depends(get_current_active_user)
) -> List[CollaborationSessionInfo]:
    """
    Récupère la liste des sessions de collaboration actives
    """
    
    try:
        collaboration_manager = await get_document_collaboration_manager()
        sessions_data = collaboration_manager.get_active_sessions()
        
        sessions = []
        for session_data in sessions_data:
            sessions.append(CollaborationSessionInfo(**session_data))
        
        return sessions
        
    except Exception as e:
        logger.error(f"❌ Erreur récupération sessions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/statistics", response_model=Dict[str, Any])
async def get_collaboration_statistics(
    current_user: User = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """
    Récupère les statistiques de collaboration complètes
    """
    
    try:
        # Statistiques collaboration classique
        collaboration_manager = await get_document_collaboration_manager()
        collaboration_stats = collaboration_manager.get_collaboration_statistics()
        
        # Statistiques Y.js
        yjs_provider = await get_yjs_provider()
        yjs_stats = yjs_provider.get_statistics()
        
        return {
            "collaboration": collaboration_stats,
            "yjs": yjs_stats,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Erreur récupération statistiques: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/document/content", response_model=DocumentContentResponse)
async def get_document_content(
    request: DocumentContentRequest,
    current_user: User = Depends(get_current_active_user)
) -> DocumentContentResponse:
    """
    Récupère le contenu actuel d'un document
    (Fallback API pour clients sans WebSocket)
    """
    
    try:
        collaboration_manager = await get_document_collaboration_manager()
        
        # Chercher dans les sessions actives
        document_content = ""
        document_version = 0
        last_updated = datetime.now()
        active_collaborators = 0
        
        for session in collaboration_manager.active_sessions.values():
            if session.document_id == request.document_id:
                document_content = session.document_content
                document_version = session.document_version
                last_updated = session.last_activity
                active_collaborators = len(session.active_users)
                break
        
        return DocumentContentResponse(
            document_id=request.document_id,
            content=document_content,
            version=document_version,
            last_updated=last_updated.isoformat(),
            active_collaborators=active_collaborators
        )
        
    except Exception as e:
        logger.error(f"❌ Erreur récupération contenu document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/document/{document_id}/presence", response_model=List[PresenceInfo])
async def get_document_presence(
    document_id: str,
    current_user: User = Depends(get_current_active_user)
) -> List[PresenceInfo]:
    """
    Récupère les informations de présence pour un document
    """
    
    try:
        presence_info = []
        
        # Présence depuis collaboration manager
        collaboration_manager = await get_document_collaboration_manager()
        
        for session in collaboration_manager.active_sessions.values():
            if session.document_id == document_id:
                for user_id, user_data in session.active_users.items():
                    cursor = session.user_cursors.get(user_id)
                    
                    presence_info.append(PresenceInfo(
                        user_id=user_id,
                        user_info=user_data.get("user_info", {}),
                        document_id=document_id,
                        connected_at=user_data["joined_at"].isoformat(),
                        last_activity=session.last_activity.isoformat(),
                        cursor_position=cursor.position if cursor else None,
                        selection_start=cursor.selection_start if cursor else None,
                        selection_end=cursor.selection_end if cursor else None
                    ))
        
        # Présence depuis Y.js provider  
        yjs_provider = await get_yjs_provider()
        
        for client in yjs_provider.clients.values():
            if client.document_id == document_id:
                # Éviter les doublons
                if not any(p.user_id == client.user_id for p in presence_info):
                    presence_info.append(PresenceInfo(
                        user_id=client.user_id,
                        user_info=client.awareness_state,
                        document_id=document_id,
                        connected_at=client.connected_at.isoformat(),
                        last_activity=client.last_ping.isoformat()
                    ))
        
        return presence_info
        
    except Exception as e:
        logger.error(f"❌ Erreur récupération présence: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/document/{document_id}/polling")
async def polling_fallback(
    document_id: str,
    last_version: int = Query(0, description="Dernière version connue"),
    current_user: User = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """
    Endpoint de polling pour clients sans WebSocket
    Retourne les mises à jour depuis une version donnée
    """
    
    try:
        collaboration_manager = await get_document_collaboration_manager()
        
        # Trouver session pour ce document
        target_session = None
        for session in collaboration_manager.active_sessions.values():
            if session.document_id == document_id:
                target_session = session
                break
        
        if not target_session:
            return {
                "has_updates": False,
                "current_version": last_version,
                "content": "",
                "operations": [],
                "comments": [],
                "presence": []
            }
        
        # Vérifier s'il y a des mises à jour
        has_updates = target_session.document_version > last_version
        
        # Récupérer opérations depuis la version demandée
        new_operations = [
            op.to_dict() for op in target_session.operations_history
            if op.document_version > last_version
        ]
        
        # Informations de présence
        presence = []
        for user_id, user_data in target_session.active_users.items():
            cursor = target_session.user_cursors.get(user_id)
            presence.append({
                "user_id": user_id,
                "user_info": user_data.get("user_info", {}),
                "cursor_position": cursor.position if cursor else None,
                "selection_start": cursor.selection_start if cursor else None,
                "selection_end": cursor.selection_end if cursor else None
            })
        
        return {
            "has_updates": has_updates,
            "current_version": target_session.document_version,
            "content": target_session.document_content,
            "operations": new_operations,
            "comments": [comment.to_dict() for comment in target_session.comments.values()],
            "presence": presence,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Erreur polling fallback: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def collaboration_health_check() -> Dict[str, Any]:
    """
    Vérification santé du système de collaboration
    """
    
    try:
        # Test collaboration manager
        collaboration_manager = await get_document_collaboration_manager()
        collaboration_stats = collaboration_manager.get_collaboration_statistics()
        
        # Test Y.js provider
        yjs_provider = await get_yjs_provider()
        yjs_stats = yjs_provider.get_statistics()
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "collaboration_system": {
                "active": True,
                "sessions": collaboration_stats["active_sessions"],
                "users": collaboration_stats["total_active_users"]
            },
            "yjs_system": {
                "active": True,
                "available": yjs_stats["yjs_available"],
                "clients": yjs_stats["total_clients"],
                "documents": yjs_stats["total_documents"]
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Erreur health check collaboration: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }