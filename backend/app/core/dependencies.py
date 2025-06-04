from typing import Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from jose import JWTError, jwt
import logging

from app.core.database import get_db
from app.config import settings
from app.models.user import User

logger = logging.getLogger(__name__)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl=f"{settings.API_V1_STR}/auth/token")

async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
) -> User:
    """Récupérer l'utilisateur courant depuis le token JWT"""
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
    except JWTError as e:
        logger.warning(f"JWT validation failed: {e}")
        raise credentials_exception
    
    # Validate user exists and is active in database
    user = db.query(User).filter(
        User.username == username, 
        User.is_active == True
    ).first()
    if user is None:
        logger.warning(f"User not found or inactive: {username}")
        raise credentials_exception
    
    return user

async def get_current_active_user(
    current_user: User = Depends(get_current_user)
) -> User:
    """Vérifier que l'utilisateur est actif"""
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

async def get_current_superuser(
    current_user: User = Depends(get_current_user)
) -> User:
    """Vérifier que l'utilisateur est superuser"""
    if not current_user.is_superuser:
        raise HTTPException(
            status_code=403, 
            detail="The user doesn't have enough privileges"
        )
    return current_user