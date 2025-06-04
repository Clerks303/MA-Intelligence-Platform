"""
FastAPI App - Version simplifiée et optimisée
Suppression des dépendances non-essentielles et middleware optionnels
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

# Imports CORE uniquement
from app.config import settings
from app.api.routes import companies, scraping, stats, auth
from app.core.database import init_db

# Configure logging - simple et efficace
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown simple et fiable"""
    # Startup
    logger.info("Starting M&A Intelligence Platform...")
    
    # Database uniquement - essentiel
    try:
        await init_db()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        # FAIL FAST - database is required
        raise e
    
    yield
    
    # Shutdown
    logger.info("Shutting down M&A Intelligence Platform...")

# App FastAPI - configuration minimale
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    lifespan=lifespan,
    docs_url="/docs" if not settings.ENVIRONMENT == "production" else None,
    redoc_url="/redoc" if not settings.ENVIRONMENT == "production" else None
)

# CORS - secure mais simple
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins_list,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["Authorization", "Content-Type", "Accept"],
)

# Routes CORE uniquement
app.include_router(auth.router, prefix=f"{settings.API_V1_STR}/auth", tags=["auth"])
app.include_router(companies.router, prefix=f"{settings.API_V1_STR}/companies", tags=["companies"])
app.include_router(scraping.router, prefix=f"{settings.API_V1_STR}/scraping", tags=["scraping"])
app.include_router(stats.router, prefix=f"{settings.API_V1_STR}/stats", tags=["stats"])

@app.get("/")
async def root():
    """Public root endpoint"""
    return {
        "service": settings.PROJECT_NAME,
        "status": "running",
        "version": settings.VERSION
    }

@app.get("/health")
async def health_check():
    """Health check simplifié"""
    try:
        from app.core.database import SessionLocal
        db = SessionLocal()
        try:
            db.execute("SELECT 1")
            db_status = "connected"
        except Exception:
            db_status = "disconnected"
        finally:
            db.close()
    except Exception:
        db_status = "error"
    
    return {
        "status": "healthy" if db_status == "connected" else "unhealthy",
        "service": settings.PROJECT_NAME,
        "database": db_status
    }

# Optional features import - loaded only if needed
def enable_advanced_features():
    """Load advanced features optionally"""
    try:
        from app.core.rate_limiting import RateLimitMiddleware
        from app.core.security_middleware import SecurityMiddleware
        from app.api.routes import ai_dashboard
        
        # Add advanced middleware
        app.add_middleware(SecurityMiddleware)
        app.add_middleware(RateLimitMiddleware, calls_per_minute=100)
        
        # Add advanced routes
        app.include_router(ai_dashboard.router, prefix=f"{settings.API_V1_STR}/ai-dashboard", tags=["ai-dashboard"])
        
        logger.info("Advanced features enabled")
        return True
    except ImportError as e:
        logger.warning(f"Advanced features not available: {e}")
        return False

# Auto-enable advanced features if available
if settings.ENABLE_ADVANCED_FEATURES:
    enable_advanced_features()