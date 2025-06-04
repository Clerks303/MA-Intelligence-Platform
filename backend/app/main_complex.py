from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
logging.basicConfig(level=logging.DEBUG)

from app.config import settings
from app.api.routes import companies, scraping, stats, auth, ai_dashboard, api_auth, external_api, collaborative_editing
from app.core.database import init_db
from app.core.rate_limiting import RateLimitMiddleware
from app.core.security_middleware import SecurityMiddleware
from app.core.api_gateway import get_api_gateway
from app.core.yjs_integration import initialize_yjs_system

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting M&A Intelligence Platform...")
    try:
        await init_db()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        # Continue without database for debugging
    
    # Initialize API Gateway
    try:
        api_gateway = await get_api_gateway()
        logger.info("API Gateway initialized successfully")
        
        # Add API Gateway middleware to the app
        app.add_middleware(lambda app: api_gateway.get_middleware())
        logger.info("API Gateway middleware added")
    except Exception as e:
        logger.error(f"API Gateway initialization failed: {e}")
    
    # Initialize Y.js collaborative editing system
    try:
        await initialize_yjs_system()
        logger.info("Y.js collaborative editing system initialized successfully")
    except Exception as e:
        logger.error(f"Y.js system initialization failed: {e}")
        # Continue without Y.js - fallback will be used
    
    yield
    # Shutdown
    logger.info("Shutting down M&A Intelligence Platform...")

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    lifespan=lifespan,
    docs_url="/docs" if settings.DOMAIN is None else None,  # Disable docs in production
    redoc_url="/redoc" if settings.DOMAIN is None else None
)

# Security Middleware (first layer) - TEMPORARILY DISABLED FOR DEBUG
# app.add_middleware(SecurityMiddleware)

# Rate Limiting Middleware - TEMPORARILY DISABLED FOR DEBUG
# app.add_middleware(
#     RateLimitMiddleware,
#     calls_per_minute=100,
#     calls_per_hour=1000
# )

# CORS - Secure configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins_list,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
    allow_headers=["Authorization", "Content-Type", "Accept", "Origin", "X-Requested-With"],
)

# Include routers
app.include_router(auth.router, prefix=f"{settings.API_V1_STR}/auth", tags=["auth"])
app.include_router(companies.router, prefix=f"{settings.API_V1_STR}/companies", tags=["companies"])
app.include_router(scraping.router, prefix=f"{settings.API_V1_STR}/scraping", tags=["scraping"])
app.include_router(stats.router, prefix=f"{settings.API_V1_STR}/stats", tags=["stats"])
app.include_router(ai_dashboard.router, prefix=f"{settings.API_V1_STR}/ai-dashboard", tags=["ai-dashboard"])

# API Gateway routes for external integrations
app.include_router(api_auth.router, prefix=f"{settings.API_V1_STR}", tags=["API Authentication"])
app.include_router(external_api.router, prefix=f"{settings.API_V1_STR}", tags=["External API"])

# Collaborative editing routes
app.include_router(collaborative_editing.router, prefix=f"{settings.API_V1_STR}", tags=["collaborative-editing"])

@app.get("/")
async def root():
    """Public root endpoint - minimal information"""
    return {
        "service": settings.PROJECT_NAME,
        "status": "running",
        "database": "postgresql"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    try:
        # Test database connection
        from app.core.database import SessionLocal
        db = SessionLocal()
        try:
            db.execute("SELECT 1")
            db_status = "connected"
        except Exception as e:
            db_status = f"error: {str(e)}"
        finally:
            db.close()
    except Exception as e:
        db_status = f"error: {str(e)}"
    
    return {
        "status": "healthy",
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "database": db_status
    }