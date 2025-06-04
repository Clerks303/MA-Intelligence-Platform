from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI."""
    logger.info("🚀 Starting M&A Intelligence Backend")
    yield
    logger.info("⚡ Shutting down M&A Intelligence Backend")

# Create FastAPI application
app = FastAPI(
    title="M&A Intelligence API",
    description="Plateforme d'intelligence M&A pour l'analyse des cabinets comptables",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Basic routes for testing
@app.get("/")
async def root():
    return {
        "message": "✅ M&A Intelligence Backend API",
        "version": "2.0.0",
        "status": "active",
        "docs": "/docs"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "ma-intelligence-backend"
    }

# Try to import core routes safely
try:
    from app.api.routes import auth, companies, scraping, stats
    app.include_router(auth.router, prefix="/api/v1/auth", tags=["Auth"])
    app.include_router(companies.router, prefix="/api/v1/companies", tags=["Companies"]) 
    app.include_router(scraping.router, prefix="/api/v1/scraping", tags=["Scraping"])
    app.include_router(stats.router, prefix="/api/v1/stats", tags=["Stats"])
    logger.info("✅ Core API routes loaded successfully")
except ImportError as e:
    logger.warning(f"⚠️ Some advanced routes not available: {e}")
    pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)