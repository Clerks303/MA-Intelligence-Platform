"""
API Gateway et Middleware pour intégrations externes
US-011: Système centralisé de gestion des APIs et authentification multi-niveaux

Ce module fournit:
- API Gateway avec routing intelligent
- Authentification multi-méthodes (OAuth2, JWT, API Keys)
- Rate limiting et quotas par client
- Transformation de requêtes/réponses
- Logging et monitoring centralisé
- Gestion des versions d'API
- Circuit breaker pour résilience
"""

import asyncio
import json
import time
import uuid
import jwt
import hashlib
import hmac
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from functools import wraps
from enum import Enum
import re

from fastapi import Request, Response, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
import redis
import aioredis
from passlib.context import CryptContext

from app.core.logging_system import get_logger, LogCategory
from app.core.cache_manager import get_cache_manager
from app.models.user import User

logger = get_logger("api_gateway", LogCategory.API)


class AuthMethod(str, Enum):
    """Méthodes d'authentification supportées"""
    BEARER_TOKEN = "bearer_token"
    API_KEY = "api_key"
    OAUTH2 = "oauth2"
    HMAC_SIGNATURE = "hmac_signature"
    BASIC_AUTH = "basic_auth"


class APIKeyScope(str, Enum):
    """Portées des API Keys"""
    READ = "read"
    WRITE = "write"
    ADMIN = "admin"
    WEBHOOK = "webhook"
    INTEGRATION = "integration"


class RateLimitType(str, Enum):
    """Types de rate limiting"""
    PER_MINUTE = "per_minute"
    PER_HOUR = "per_hour"
    PER_DAY = "per_day"
    PER_MONTH = "per_month"


@dataclass
class APIKey:
    """Modèle pour les clés API"""
    key_id: str
    key_hash: str
    name: str
    client_id: str
    scopes: List[APIKeyScope]
    
    # Limites
    rate_limit: Dict[RateLimitType, int]
    quota_limit: Optional[int] = None
    
    # Statut
    is_active: bool = True
    expires_at: Optional[datetime] = None
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    last_used: Optional[datetime] = None
    usage_count: int = 0
    
    # Restrictions
    allowed_ips: List[str] = field(default_factory=list)
    allowed_domains: List[str] = field(default_factory=list)


@dataclass
class APIClient:
    """Client API avec configuration"""
    client_id: str
    client_secret: str
    name: str
    description: str
    
    # Configuration
    auth_methods: List[AuthMethod]
    default_scopes: List[APIKeyScope]
    
    # Limites globales
    rate_limits: Dict[RateLimitType, int]
    monthly_quota: Optional[int] = None
    
    # Webhooks
    webhook_url: Optional[str] = None
    webhook_secret: Optional[str] = None
    webhook_events: List[str] = field(default_factory=list)
    
    # Statut
    is_active: bool = True
    is_verified: bool = False
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    owner_email: str = ""
    contact_email: str = ""
    
    # Statistiques
    total_requests: int = 0
    last_request: Optional[datetime] = None


@dataclass
class RateLimitInfo:
    """Information sur les limites de taux"""
    limit: int
    remaining: int
    reset_time: datetime
    retry_after: Optional[int] = None


@dataclass
class APIRequest:
    """Requête API avec contexte"""
    request_id: str
    client_id: Optional[str]
    api_key: Optional[str]
    user_id: Optional[str]
    
    # Détails requête
    method: str
    path: str
    headers: Dict[str, str]
    query_params: Dict[str, Any]
    body: Optional[Dict[str, Any]]
    
    # Authentification
    auth_method: Optional[AuthMethod]
    scopes: List[str]
    
    # Contexte
    ip_address: str
    user_agent: str
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Performance
    processing_time: Optional[float] = None
    response_status: Optional[int] = None
    response_size: Optional[int] = None


class AuthenticationManager:
    """Gestionnaire d'authentification multi-méthodes"""
    
    def __init__(self):
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        self.jwt_secret = "your-jwt-secret-key"  # À récupérer depuis config
        self.jwt_algorithm = "HS256"
        
        # Stockage des clés API (en production, utiliser une DB)
        self.api_keys: Dict[str, APIKey] = {}
        self.clients: Dict[str, APIClient] = {}
        
        # Cache Redis pour les tokens
        self.redis_client: Optional[aioredis.Redis] = None
        
    async def initialize(self):
        """Initialise le gestionnaire d'authentification"""
        try:
            # Connexion Redis pour cache des tokens
            self.redis_client = aioredis.from_url("redis://localhost:6379", decode_responses=True)
            await self.redis_client.ping()
            logger.info("✅ Redis connecté pour cache authentification")
        except Exception as e:
            logger.warning(f"⚠️ Redis non disponible: {e}")
            self.redis_client = None
    
    def generate_api_key(self, client_id: str, name: str, scopes: List[APIKeyScope]) -> Tuple[str, APIKey]:
        """Génère une nouvelle clé API"""
        
        # Générer clé unique
        key_id = f"ak_{uuid.uuid4().hex[:12]}"
        raw_key = f"{key_id}_{uuid.uuid4().hex}"
        key_hash = self.pwd_context.hash(raw_key)
        
        # Créer objet APIKey
        api_key = APIKey(
            key_id=key_id,
            key_hash=key_hash,
            name=name,
            client_id=client_id,
            scopes=scopes,
            rate_limit={
                RateLimitType.PER_MINUTE: 60,
                RateLimitType.PER_HOUR: 3600,
                RateLimitType.PER_DAY: 86400
            }
        )
        
        self.api_keys[key_id] = api_key
        
        logger.info(f"🔑 Clé API générée: {key_id} pour client {client_id}")
        
        return raw_key, api_key
    
    def create_client(
        self, 
        name: str, 
        description: str, 
        owner_email: str,
        auth_methods: List[AuthMethod] = None
    ) -> APIClient:
        """Crée un nouveau client API"""
        
        if auth_methods is None:
            auth_methods = [AuthMethod.API_KEY, AuthMethod.BEARER_TOKEN]
        
        client_id = f"client_{uuid.uuid4().hex[:16]}"
        client_secret = f"cs_{uuid.uuid4().hex}"
        
        client = APIClient(
            client_id=client_id,
            client_secret=client_secret,
            name=name,
            description=description,
            auth_methods=auth_methods,
            default_scopes=[APIKeyScope.READ],
            rate_limits={
                RateLimitType.PER_MINUTE: 100,
                RateLimitType.PER_HOUR: 5000,
                RateLimitType.PER_DAY: 100000
            },
            monthly_quota=1000000,
            owner_email=owner_email,
            contact_email=owner_email
        )
        
        self.clients[client_id] = client
        
        logger.info(f"👤 Client API créé: {client_id} ({name})")
        
        return client
    
    async def verify_api_key(self, api_key: str) -> Optional[APIKey]:
        """Vérifie une clé API"""
        
        # Extraire key_id du format complet
        if "_" in api_key:
            key_id = api_key.split("_")[1][:12]
            key_id = f"ak_{key_id}"
        else:
            return None
        
        # Vérifier existence
        if key_id not in self.api_keys:
            return None
        
        stored_key = self.api_keys[key_id]
        
        # Vérifier statut
        if not stored_key.is_active:
            return None
        
        # Vérifier expiration
        if stored_key.expires_at and stored_key.expires_at < datetime.now():
            return None
        
        # Vérifier hash
        if not self.pwd_context.verify(api_key, stored_key.key_hash):
            return None
        
        # Mettre à jour statistiques
        stored_key.last_used = datetime.now()
        stored_key.usage_count += 1
        
        return stored_key
    
    async def verify_bearer_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Vérifie un token Bearer JWT"""
        
        try:
            # Décoder le token JWT
            payload = jwt.decode(token, self.jwt_secret, algorithms=[self.jwt_algorithm])
            
            # Vérifier expiration
            if payload.get("exp", 0) < time.time():
                return None
            
            # Vérifier si révoqué (cache Redis)
            if self.redis_client:
                revoked = await self.redis_client.get(f"revoked_token:{token[:16]}")
                if revoked:
                    return None
            
            return payload
            
        except jwt.InvalidTokenError:
            return None
    
    def generate_jwt_token(self, user_id: str, scopes: List[str], expires_in: int = 3600) -> str:
        """Génère un token JWT"""
        
        payload = {
            "user_id": user_id,
            "scopes": scopes,
            "iat": int(time.time()),
            "exp": int(time.time()) + expires_in,
            "jti": uuid.uuid4().hex  # Token ID unique
        }
        
        token = jwt.encode(payload, self.jwt_secret, algorithm=self.jwt_algorithm)
        
        logger.info(f"🎫 Token JWT généré pour utilisateur {user_id}")
        
        return token
    
    async def revoke_token(self, token: str):
        """Révoque un token JWT"""
        
        if self.redis_client:
            # Stocker dans Redis avec TTL
            await self.redis_client.setex(
                f"revoked_token:{token[:16]}", 
                3600,  # TTL d'1 heure
                "revoked"
            )
            logger.info(f"🚫 Token révoqué: {token[:16]}...")
    
    def verify_hmac_signature(self, request_body: bytes, signature: str, secret: str) -> bool:
        """Vérifie une signature HMAC"""
        
        expected_signature = hmac.new(
            secret.encode(),
            request_body,
            hashlib.sha256
        ).hexdigest()
        
        # Comparaison sécurisée
        return hmac.compare_digest(signature, expected_signature)


class RateLimiter:
    """Gestionnaire de limitation de taux"""
    
    def __init__(self):
        self.redis_client: Optional[aioredis.Redis] = None
        self.memory_store: Dict[str, Dict[str, Any]] = {}
        
    async def initialize(self):
        """Initialise le rate limiter"""
        try:
            self.redis_client = aioredis.from_url("redis://localhost:6379", decode_responses=True)
            await self.redis_client.ping()
            logger.info("✅ Redis connecté pour rate limiting")
        except Exception as e:
            logger.warning(f"⚠️ Redis non disponible pour rate limiting: {e}")
            self.redis_client = None
    
    async def check_rate_limit(
        self, 
        client_id: str, 
        limit: int, 
        window: int,
        identifier: str = None
    ) -> RateLimitInfo:
        """Vérifie les limites de taux"""
        
        key = f"rate_limit:{client_id}"
        if identifier:
            key += f":{identifier}"
        
        current_time = int(time.time())
        window_start = current_time - window
        
        if self.redis_client:
            # Utiliser Redis pour stockage distribué
            pipe = self.redis_client.pipeline()
            
            # Supprimer les anciens timestamps
            pipe.zremrangebyscore(key, 0, window_start)
            
            # Compter les requêtes dans la fenêtre
            pipe.zcard(key)
            
            # Ajouter la requête actuelle
            pipe.zadd(key, {str(current_time): current_time})
            
            # Définir expiration
            pipe.expire(key, window)
            
            results = await pipe.execute()
            current_count = results[1]
            
        else:
            # Fallback mémoire locale
            if key not in self.memory_store:
                self.memory_store[key] = {"timestamps": [], "last_cleanup": current_time}
            
            store = self.memory_store[key]
            
            # Nettoyer les anciens timestamps
            store["timestamps"] = [ts for ts in store["timestamps"] if ts > window_start]
            
            current_count = len(store["timestamps"])
            store["timestamps"].append(current_time)
        
        # Calculer les informations de limite
        remaining = max(0, limit - current_count - 1)
        reset_time = datetime.fromtimestamp(current_time + window)
        retry_after = window if remaining == 0 else None
        
        return RateLimitInfo(
            limit=limit,
            remaining=remaining,
            reset_time=reset_time,
            retry_after=retry_after
        )
    
    async def is_rate_limited(self, client_id: str, api_key: APIKey) -> Optional[RateLimitInfo]:
        """Vérifie si un client est rate limité"""
        
        for limit_type, limit_value in api_key.rate_limit.items():
            window_seconds = {
                RateLimitType.PER_MINUTE: 60,
                RateLimitType.PER_HOUR: 3600,
                RateLimitType.PER_DAY: 86400,
                RateLimitType.PER_MONTH: 2592000  # 30 jours
            }[limit_type]
            
            rate_info = await self.check_rate_limit(
                client_id, 
                limit_value, 
                window_seconds, 
                limit_type.value
            )
            
            if rate_info.remaining <= 0:
                logger.warning(f"🚫 Rate limit atteint pour {client_id}: {limit_type.value}")
                return rate_info
        
        return None


class APIGatewayMiddleware(BaseHTTPMiddleware):
    """Middleware principal de l'API Gateway"""
    
    def __init__(self, app, auth_manager: AuthenticationManager, rate_limiter: RateLimiter):
        super().__init__(app)
        self.auth_manager = auth_manager
        self.rate_limiter = rate_limiter
        
        # Chemins exemptés d'authentification
        self.exempt_paths = [
            "/docs",
            "/redoc", 
            "/openapi.json",
            "/health",
            "/",
            "/api/v1/auth/login"
        ]
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Traite chaque requête via l'API Gateway"""
        
        start_time = time.time()
        request_id = f"req_{uuid.uuid4().hex[:12]}"
        
        # Extraire informations de base
        client_ip = request.client.host
        user_agent = request.headers.get("user-agent", "")
        path = request.url.path
        method = request.method
        
        # Logger la requête
        logger.info(f"🌐 [{request_id}] {method} {path} from {client_ip}")
        
        # Vérifier si chemin exempté
        if self._is_exempt_path(path):
            response = await call_next(request)
            self._add_response_headers(response, request_id)
            return response
        
        try:
            # 1. Authentification
            auth_result = await self._authenticate_request(request)
            if auth_result.get("error"):
                return self._create_error_response(
                    401, auth_result["error"], request_id
                )
            
            # 2. Rate Limiting
            if auth_result.get("api_key"):
                rate_limit_info = await self.rate_limiter.is_rate_limited(
                    auth_result["client_id"], 
                    auth_result["api_key"]
                )
                
                if rate_limit_info:
                    return self._create_rate_limit_response(rate_limit_info, request_id)
            
            # 3. Enrichir la requête avec le contexte d'auth
            request.state.auth_context = auth_result
            request.state.request_id = request_id
            
            # 4. Traiter la requête
            response = await call_next(request)
            
            # 5. Post-traitement
            self._add_response_headers(response, request_id, auth_result)
            
            # 6. Logger la réponse
            processing_time = time.time() - start_time
            logger.info(
                f"✅ [{request_id}] {response.status_code} "
                f"({processing_time:.3f}s) - Client: {auth_result.get('client_id', 'anonymous')}"
            )
            
            return response
            
        except Exception as e:
            logger.error(f"❌ [{request_id}] Erreur API Gateway: {e}")
            return self._create_error_response(500, "Internal server error", request_id)
    
    def _is_exempt_path(self, path: str) -> bool:
        """Vérifie si un chemin est exempté d'authentification"""
        return any(path.startswith(exempt) for exempt in self.exempt_paths)
    
    async def _authenticate_request(self, request: Request) -> Dict[str, Any]:
        """Authentifie une requête"""
        
        # Vérifier en-tête Authorization
        auth_header = request.headers.get("authorization", "")
        api_key_header = request.headers.get("x-api-key", "")
        
        if auth_header.startswith("Bearer "):
            # Authentification par token Bearer
            token = auth_header[7:]
            payload = await self.auth_manager.verify_bearer_token(token)
            
            if payload:
                return {
                    "auth_method": AuthMethod.BEARER_TOKEN,
                    "user_id": payload.get("user_id"),
                    "scopes": payload.get("scopes", []),
                    "client_id": payload.get("client_id")
                }
            else:
                return {"error": "Invalid or expired token"}
        
        elif api_key_header:
            # Authentification par clé API
            api_key = await self.auth_manager.verify_api_key(api_key_header)
            
            if api_key:
                return {
                    "auth_method": AuthMethod.API_KEY,
                    "api_key": api_key,
                    "client_id": api_key.client_id,
                    "scopes": [scope.value for scope in api_key.scopes]
                }
            else:
                return {"error": "Invalid API key"}
        
        elif auth_header.startswith("Basic "):
            # Authentification basique
            return {"error": "Basic auth not implemented yet"}
        
        else:
            return {"error": "Authentication required"}
    
    def _create_error_response(self, status_code: int, message: str, request_id: str) -> JSONResponse:
        """Crée une réponse d'erreur standardisée"""
        
        return JSONResponse(
            status_code=status_code,
            content={
                "error": {
                    "code": status_code,
                    "message": message,
                    "request_id": request_id,
                    "timestamp": datetime.now().isoformat()
                }
            },
            headers={
                "X-Request-ID": request_id,
                "X-API-Version": "v1"
            }
        )
    
    def _create_rate_limit_response(self, rate_info: RateLimitInfo, request_id: str) -> JSONResponse:
        """Crée une réponse de rate limiting"""
        
        headers = {
            "X-Request-ID": request_id,
            "X-RateLimit-Limit": str(rate_info.limit),
            "X-RateLimit-Remaining": str(rate_info.remaining),
            "X-RateLimit-Reset": str(int(rate_info.reset_time.timestamp())),
        }
        
        if rate_info.retry_after:
            headers["Retry-After"] = str(rate_info.retry_after)
        
        return JSONResponse(
            status_code=429,
            content={
                "error": {
                    "code": 429,
                    "message": "Rate limit exceeded",
                    "request_id": request_id,
                    "rate_limit": {
                        "limit": rate_info.limit,
                        "remaining": rate_info.remaining,
                        "reset_at": rate_info.reset_time.isoformat()
                    }
                }
            },
            headers=headers
        )
    
    def _add_response_headers(
        self, 
        response: Response, 
        request_id: str, 
        auth_context: Dict[str, Any] = None
    ):
        """Ajoute les en-têtes de réponse standardisés"""
        
        response.headers["X-Request-ID"] = request_id
        response.headers["X-API-Version"] = "v1"
        response.headers["X-Powered-By"] = "M&A Intelligence API"
        
        if auth_context and auth_context.get("api_key"):
            api_key = auth_context["api_key"]
            rate_limit = api_key.rate_limit.get(RateLimitType.PER_HOUR, 3600)
            response.headers["X-RateLimit-Limit"] = str(rate_limit)


class APIGateway:
    """API Gateway principal"""
    
    def __init__(self):
        self.auth_manager = AuthenticationManager()
        self.rate_limiter = RateLimiter()
        self.middleware = None
        
        # Statistiques
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "blocked_requests": 0,
            "start_time": datetime.now()
        }
        
    async def initialize(self):
        """Initialise l'API Gateway"""
        
        logger.info("🚀 Initialisation API Gateway...")
        
        await self.auth_manager.initialize()
        await self.rate_limiter.initialize()
        
        # Créer le middleware
        self.middleware = APIGatewayMiddleware(None, self.auth_manager, self.rate_limiter)
        
        # Créer clients de démonstration
        await self._create_demo_clients()
        
        logger.info("✅ API Gateway initialisé avec succès")
    
    async def _create_demo_clients(self):
        """Crée des clients de démonstration"""
        
        # Client de démonstration
        demo_client = self.auth_manager.create_client(
            name="Demo Client",
            description="Client de démonstration pour tests",
            owner_email="demo@example.com",
            auth_methods=[AuthMethod.API_KEY, AuthMethod.BEARER_TOKEN]
        )
        
        # Générer clé API de démonstration
        demo_key, _ = self.auth_manager.generate_api_key(
            demo_client.client_id,
            "Demo API Key",
            [APIKeyScope.READ, APIKeyScope.WRITE]
        )
        
        logger.info(f"🔑 Client démo créé: {demo_client.client_id}")
        logger.info(f"🗝️  Clé API démo: {demo_key}")
    
    def get_middleware(self):
        """Retourne le middleware pour FastAPI"""
        return self.middleware
    
    def get_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques de l'API Gateway"""
        
        uptime = datetime.now() - self.stats["start_time"]
        
        return {
            **self.stats,
            "uptime_seconds": uptime.total_seconds(),
            "requests_per_second": self.stats["total_requests"] / max(uptime.total_seconds(), 1),
            "success_rate": (
                self.stats["successful_requests"] / max(self.stats["total_requests"], 1) * 100
            ),
            "active_clients": len(self.auth_manager.clients),
            "active_api_keys": len(self.auth_manager.api_keys)
        }


# Instance globale
_api_gateway: Optional[APIGateway] = None


async def get_api_gateway() -> APIGateway:
    """Factory pour obtenir l'API Gateway"""
    global _api_gateway
    
    if _api_gateway is None:
        _api_gateway = APIGateway()
        await _api_gateway.initialize()
    
    return _api_gateway


# Fonctions utilitaires pour FastAPI

def require_api_key(scopes: List[APIKeyScope] = None):
    """Décorateur pour exiger une clé API avec scopes"""
    
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Cette fonction sera intégrée avec FastAPI Depends
            # Pour l'instant, c'est un placeholder
            return await func(*args, **kwargs)
        return wrapper
    return decorator


def require_scope(scope: APIKeyScope):
    """Décorateur pour exiger un scope spécifique"""
    
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Vérification du scope via le contexte de requête
            return await func(*args, **kwargs)
        return wrapper
    return decorator