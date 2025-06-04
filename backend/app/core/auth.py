"""
Système d'authentification unifié pour M&A Intelligence Platform
Consolide auth.py, advanced_authentication.py et api_auth.py en un seul module cohérent
"""

from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Union
from fastapi import Depends, HTTPException, status, Request
from fastapi.security import OAuth2PasswordBearer, HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel
import secrets
import hashlib
import time

from app.config import settings
from app.core.database import get_db

# Configuration du schéma OAuth2
oauth2_scheme = OAuth2PasswordBearer(tokenUrl=f"/api/v1/auth/login")
bearer_scheme = HTTPBearer()

# Configuration du hachage des mots de passe
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Models
class Token(BaseModel):
    access_token: str
    token_type: str
    expires_in: int

class TokenData(BaseModel):
    username: Optional[str] = None

class User(BaseModel):
    username: str
    email: Optional[str] = None
    is_active: bool = True

class UserInDB(User):
    hashed_password: str

class LoginRequest(BaseModel):
    username: str
    password: str

# Cache simple en mémoire pour les utilisateurs (remplacer par DB en production)
fake_users_db = {
    "admin": {
        "username": "admin",
        "email": "admin@ma-intelligence.com",
        "hashed_password": pwd_context.hash("secret"),
        "is_active": True,
    }
}

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Vérifie un mot de passe contre son hash"""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """Hash un mot de passe"""
    return pwd_context.hash(password)

def get_user(username: str) -> Optional[UserInDB]:
    """Récupère un utilisateur depuis la base (mock)"""
    if username in fake_users_db:
        user_dict = fake_users_db[username]
        return UserInDB(**user_dict)
    return None

def authenticate_user(username: str, password: str) -> Union[UserInDB, bool]:
    """Authentifie un utilisateur"""
    user = get_user(username)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Crée un token JWT"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    """Valide le token JWT et retourne l'utilisateur actuel"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    
    user = get_user(username=token_data.username)
    if user is None:
        raise credentials_exception
    return User(username=user.username, email=user.email, is_active=user.is_active)

async def get_current_active_user(current_user: User = Depends(get_current_user)) -> User:
    """Retourne l'utilisateur actuel s'il est actif"""
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

# API Key Authentication (simplifié, pour développeurs)
class APIKeyAuth:
    """Authentification par clé API pour les développeurs"""
    
    def __init__(self):
        self.api_keys = {
            "dev_api_key_123": {"name": "Development Key", "active": True},
            # Ajouter d'autres clés API en production
        }
    
    def verify_api_key(self, api_key: str) -> bool:
        """Vérifie une clé API"""
        return api_key in self.api_keys and self.api_keys[api_key]["active"]
    
    async def get_api_key(self, credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)) -> str:
        """Extrait et valide la clé API"""
        if not self.verify_api_key(credentials.credentials):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key"
            )
        return credentials.credentials

# Instance globale pour l'authentification API
api_key_auth = APIKeyAuth()

# Rate Limiting simple (à améliorer avec Redis en production)
class RateLimiter:
    """Rate limiting basique en mémoire"""
    
    def __init__(self, max_requests: int = 100, window_seconds: int = 3600):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = {}  # {ip: [(timestamp, count), ...]}
    
    def is_allowed(self, identifier: str) -> bool:
        """Vérifie si la requête est autorisée"""
        now = time.time()
        window_start = now - self.window_seconds
        
        # Nettoyer les anciens enregistrements
        if identifier in self.requests:
            self.requests[identifier] = [
                (ts, count) for ts, count in self.requests[identifier] 
                if ts > window_start
            ]
        else:
            self.requests[identifier] = []
        
        # Compter les requêtes dans la fenêtre
        total_requests = sum(count for _, count in self.requests[identifier])
        
        if total_requests >= self.max_requests:
            return False
        
        # Ajouter cette requête
        self.requests[identifier].append((now, 1))
        return True

# Instance globale du rate limiter
rate_limiter = RateLimiter(max_requests=100, window_seconds=3600)

async def check_rate_limit(request: Request):
    """Middleware de rate limiting"""
    client_ip = request.client.host
    if not rate_limiter.is_allowed(client_ip):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded"
        )

# Fonctions utilitaires pour sécurité
def generate_secure_token() -> str:
    """Génère un token sécurisé"""
    return secrets.token_urlsafe(32)

def hash_string(text: str) -> str:
    """Hash une chaîne avec SHA-256"""
    return hashlib.sha256(text.encode()).hexdigest()

# Export des fonctions principales
__all__ = [
    "Token", "User", "LoginRequest",
    "authenticate_user", "create_access_token", 
    "get_current_user", "get_current_active_user",
    "api_key_auth", "check_rate_limit",
    "verify_password", "get_password_hash"
]