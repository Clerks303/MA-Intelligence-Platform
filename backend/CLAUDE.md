# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 🚨 AUDIT COMPLET EFFECTUÉ - CORRECTIONS DISPONIBLES

### ⚡ CORRECTIONS CRITIQUES DÉJÀ IMPLEMENTÉES

1. **✅ Sécurité authentification renforcée** (`app/core/dependencies.py`)
2. **✅ Configuration unifiée et validation** (`app/core/config.py`) 
3. **✅ CORS sécurisé configurable** (`app/main.py`)
4. **✅ Gestion d'erreurs structurée** (`app/core/exceptions.py`)
5. **✅ Validation d'entrées** (`app/core/validators.py`)
6. **✅ Constantes centralisées** (`app/core/constants.py`)

### 🔥 ACTIONS RESTANTES PRIORITAIRES

**IMMÉDIAT (Aujourd'hui)**
```bash
# 1. Protéger TOUS les endpoints
# Ajouter à chaque route dans companies.py, scraping.py, stats.py :
current_user: User = Depends(get_current_active_user)

# 2. Rate limiting
pip install slowapi
# Implémenter dans main.py

# 3. Middleware de sécurité
pip install python-multipart  # Si pas déjà installé
# Ajouter security headers middleware
```

**CETTE SEMAINE**
- Migration base de données async (performance x10)
- Tests automatisés complets
- Monitoring et logging structuré
- Optimisation queries database

## Commands

### Development Server
```bash
# Backend (from /backend directory)
uvicorn app.main:app --reload

# Alternative using Python module
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Database Operations
```bash
# Run database migrations
alembic upgrade head

# Create new migration
alembic revision --autogenerate -m "migration_description"

# Reset database (create all tables)
python -c "from app.core.database import init_db; import asyncio; asyncio.run(init_db())"
```

### Testing
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_api.py

# Run tests with coverage
pytest --cov=app tests/

# Run specific test function
pytest tests/test_api.py::test_login -v
```

### Code Quality
```bash
# Format code with Black
black app/ tests/

# Check formatting without changes
black --check app/ tests/

# Security audit
pip install bandit
bandit -r app/

# Type checking
pip install mypy
mypy app/
```

### Dependencies
```bash
# Install dependencies
pip install -r requirements.txt

# Update requirements (after adding new packages)
pip freeze > requirements.txt
```

## Architecture Overview

### Project Structure
This is a **FastAPI-based M&A Intelligence Platform** for scraping and analyzing French accounting firms data. The backend serves as an API for a React frontend and handles web scraping, data processing, and business intelligence.

### 🔒 Security Architecture (POST-AUDIT)

**Authentication System**
- JWT-based authentication with improved validation (`app/core/dependencies.py`)
- Database user validation on each request
- Active user status checking
- Comprehensive error logging
- Protected routes with `get_current_active_user` dependency

**Input Validation**
- SIREN/SIRET format validation (`app/core/validators.py`)
- Password strength requirements
- Search term sanitization (SQL injection prevention)
- File upload restrictions and validation

**Error Handling**
- Custom exception classes (`app/core/exceptions.py`)
- Generic error responses (no information leakage)
- Structured logging with security events

### Core Components

**Database Layer**
- Uses **Supabase** (PostgreSQL) as primary database via `app/db/supabase.py`
- **SQLAlchemy** ORM with models in `app/models/` for local development/testing
- **Alembic** for database migrations
- Database initialization happens in `app/core/database.py:init_db()`

**API Structure**
- FastAPI app configured in `app/main.py` with secure CORS
- Route modules in `app/api/routes/`: `auth.py`, `companies.py`, `scraping.py`, `stats.py`
- All routes prefixed with `/api/v1`
- JWT authentication with improved validation

**Data Models**
- **Company model** (`app/models/company.py`): Core entity with SIREN/SIRET identifiers
- **User model** (`app/models/user.py`): Authentication and authorization
- **Pydantic schemas** in `app/models/schemas.py` for API serialization

**Web Scraping Engine**
- **Pappers API client** (`app/scrapers/pappers.py`): Asynchronous company data scraping
- **Société.com scraper** (`app/scrapers/societe.py`): Web scraping with Selenium
- **Infogreffe scraper** (`app/scrapers/infogreffe.py`): Official registry data
- Configuration centralized in `app/core/constants.py`

**Business Logic Services**
- **Data processing** (`app/services/data_processing.py`): CSV import/export
- **Enrichment** (`app/services/enrichment.py`): Data quality improvement
- **Scoring** (`app/services/scoring.py`): AI-powered M&A scoring

### Configuration

**Environment Variables** (`.env` file required)
```env
# Security (REQUIRED - validated)
SECRET_KEY=your-32-character-minimum-secret-key
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Database
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-anon-key

# Initial user (validated)
FIRST_SUPERUSER=admin
FIRST_SUPERUSER_PASSWORD=minimum-8-chars

# CORS (configurable)
ALLOWED_ORIGINS=http://localhost:3000,https://yourdomain.com

# External APIs (optional)
OPENAI_API_KEY=sk-...
PAPPERS_API_KEY=your-pappers-key

# Scraping
HEADLESS=true
```

**Settings Management**
- Unified configuration in `app/config.py` with validation
- Duplicate fields removed (SECRET_KEY/secret_key → SECRET_KEY)
- Password and secret key strength validation
- CORS origins as configurable list

### 🚀 Performance Optimizations Available

**Database**
- Connection pooling configuration ready
- Async database sessions pattern
- Query optimization recommendations
- Composite indexes for common filters

**Caching Strategy**
- Stats caching (5min TTL) - `app/core/constants.py`
- Company data caching (1hr TTL)
- Search results caching (10min TTL)

**Scraping Performance**
- Async batch processing patterns
- Rate limiting with semaphores
- Memory management for large datasets
- Progress tracking optimizations

### 🧪 Testing Strategy

**Test Structure**
- Tests in `/tests/` directory using pytest
- Authentication fixtures with secure token generation
- API endpoint testing with proper auth headers
- Async test support

**Security Tests**
```python
# Add to test suite
def test_unauthorized_access():
    """Test that protected endpoints require authentication"""
    
def test_invalid_token():
    """Test invalid JWT token handling"""
    
def test_sql_injection_protection():
    """Test search parameter sanitization"""
```

### 📊 Monitoring & Observability

**Logging Configuration**
- Structured logging with security events
- Audit trail for sensitive operations
- Error tracking with context
- Performance metrics collection

**Health Checks**
```python
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "database": "connected",
        "external_apis": "available"
    }
```

### 🔮 Long-term Roadmap

**Phase 1 (Next Sprint)**
- Complete endpoint protection implementation
- Rate limiting middleware
- Comprehensive test suite
- Database query optimization

**Phase 2 (Next Month)**  
- Redis caching layer
- Celery background tasks
- API versioning strategy
- Performance monitoring dashboard

**Phase 3 (Next Quarter)**
- Microservices architecture consideration
- Real-time websocket features
- Advanced AI scoring models
- Multi-tenant support

### ⚠️ Known Issues & Debt

1. **Mixed database approaches** (Supabase + SQLAlchemy) - needs consolidation
2. **Global state in scraping** - refactor to proper state management
3. **Synchronous operations in async context** - migrate to full async
4. **Missing comprehensive test coverage** - prioritize critical paths
5. **Hard-coded business logic** - extract to configuration

### 🛡️ Security Checklist

- ✅ JWT validation with database user check
- ✅ Input validation and sanitization  
- ✅ CORS configuration secured
- ✅ Password strength requirements
- ⏳ Rate limiting (implementation ready)
- ⏳ SQL injection protection (validators ready)
- ⏳ File upload security (validation ready)
- ⏳ API key management improvements
- ⏳ Security headers middleware
- ⏳ Request/response logging

### Development Notes

**Adding New Endpoints**
1. Always add `current_user: User = Depends(get_current_active_user)` 
2. Use custom exceptions from `app/core/exceptions.py`
3. Validate inputs with `app/core/validators.py`
4. Log security events appropriately
5. Add comprehensive tests

**Database Changes**
1. Create Alembic migration
2. Update both SQLAlchemy and Supabase schemas
3. Consider performance impact of new queries
4. Add appropriate indexes

**External API Integration**
1. Use constants from `app/core/constants.py`
2. Implement proper timeout and retry logic
3. Add rate limiting and error handling
4. Consider caching strategy