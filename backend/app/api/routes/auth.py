from datetime import timedelta
from typing import Optional
from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from app.config import settings
from app.core.database import get_db
from app.core.security import create_access_token, verify_password, get_password_hash
from app.core.dependencies import get_current_active_user
from app.models.user import User
from app.models.schemas import Token, UserLogin

router = APIRouter()

@router.get("/test")
async def test_endpoint():
    """Endpoint de test simple pour vérifier la connectivité"""
    return {"message": "Backend accessible", "status": "OK"}

def authenticate_user(db: Session, username: str, password: str) -> Optional[User]:
    """Authentifier un utilisateur"""
    user = db.query(User).filter(User.username == username).first()
    if not user:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user

@router.post("/token", response_model=Token)
async def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    """Connexion et génération de token d'accès"""
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@router.post("/login", response_model=Token) 
async def login(
    user_credentials: UserLogin,
    db: Session = Depends(get_db)
):
    """Endpoint de connexion alternatif avec JSON"""
    user = authenticate_user(db, user_credentials.username, user_credentials.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password"
        )
    
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@router.post("/signup", response_model=Token)
async def signup(
    user_credentials: UserLogin,
    db: Session = Depends(get_db)
):
    """Inscription d'un nouvel utilisateur (pour le développement)"""
    # Vérifier si l'utilisateur existe déjà
    existing_user = db.query(User).filter(User.username == user_credentials.username).first()
    if existing_user:
        raise HTTPException(
            status_code=400,
            detail="Username already registered"
        )
    
    # Créer le nouvel utilisateur
    new_user = User(
        username=user_credentials.username,
        email=f"{user_credentials.username}@example.com",
        hashed_password=get_password_hash(user_credentials.password),
        is_superuser=False,
        is_active=True
    )
    
    try:
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        
        # Générer le token
        access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": user_credentials.username}, expires_delta=access_token_expires
        )
        return {"access_token": access_token, "token_type": "bearer"}
        
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error creating user: {str(e)}")

@router.get("/me")
async def read_users_me(current_user: User = Depends(get_current_active_user)):
    """Obtenir les informations de l'utilisateur connecté"""
    return {
        "id": current_user.id,
        "username": current_user.username,
        "email": current_user.email,
        "is_active": current_user.is_active,
        "is_superuser": current_user.is_superuser
    }