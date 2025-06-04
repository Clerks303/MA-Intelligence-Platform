# Guide Développeur - M&A Intelligence Platform

## 🚀 Quick Start

### Prérequis
- Python 3.11+
- PostgreSQL (via Supabase)
- Redis (optionnel pour cache)
- Git

### Installation rapide
```bash
# Clone et setup
git clone <repo>
cd backend/

# Environment virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou venv\Scripts\activate  # Windows

# Dépendances
pip install -r requirements.txt

# Configuration
cp .env.example .env
# Éditer .env avec vos clés API

# Database
alembic upgrade head

# Démarrage
uvicorn app.main:app --reload
```

### URLs développement
- **API** : http://localhost:8000
- **Docs** : http://localhost:8000/docs
- **Health** : http://localhost:8000/health

## 📋 Tâches courantes

### Développement API

#### Ajouter un endpoint
```python
# 1. Dans app/api/routes/companies.py
@router.get("/new-endpoint")
async def new_endpoint(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    return {"message": "Success"}

# 2. Tests dans tests/test_api.py
def test_new_endpoint(client, auth_headers):
    response = client.get("/api/v1/companies/new-endpoint", headers=auth_headers)
    assert response.status_code == 200
```

#### Validation des données
```python
# 1. Schema dans app/models/schemas.py
class NewRequestSchema(BaseModel):
    siren: str = Field(..., min_length=9, max_length=9)
    name: str = Field(..., min_length=1, max_length=255)

# 2. Utilisation
async def endpoint(data: NewRequestSchema):
    # Données automatiquement validées
    pass
```

### Base de données

#### Nouvelle migration
```bash
# Modifier app/models/company.py
# Puis générer migration
alembic revision --autogenerate -m "Add new field"

# Appliquer
alembic upgrade head
```

#### Nouveau modèle
```python
# 1. Dans app/models/company.py
class NewModel(Base):
    __tablename__ = "new_table"
    
    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

# 2. Ajouter dans app/models/__init__.py
from .company import Company, NewModel
__all__ = ["Company", "NewModel"]

# 3. Migration
alembic revision --autogenerate -m "Add NewModel"
```

### Scraping

#### Nouveau scraper
```python
# 1. Dans app/scrapers/new_source.py
class NewScraper:
    def __init__(self):
        self.session = aiohttp.ClientSession()
    
    async def scrape_company(self, siren: str) -> Dict:
        try:
            # Logique scraping
            data = await self._fetch_data(siren)
            return self._parse_data(data)
        except Exception as e:
            logger.error(f"Scraping error {siren}: {e}")
            raise

# 2. Intégrer dans app/core/background_jobs.py
elif source == "new_source":
    from app.scrapers.new_source import NewScraper
    scraper = NewScraper()
```

#### Rate limiting
```python
# Utiliser semaphore pour contrôler débit
semaphore = asyncio.Semaphore(5)  # 5 requêtes max en parallèle

async def scrape_with_limit(siren: str):
    async with semaphore:
        await scraper.scrape_company(siren)
        await asyncio.sleep(0.1)  # Délai entre requêtes
```

### Tests

#### Tests d'endpoint
```python
def test_companies_list(client, auth_headers):
    response = client.get("/api/v1/companies", headers=auth_headers)
    assert response.status_code == 200
    assert "companies" in response.json()

def test_create_company(client, auth_headers):
    data = {"siren": "123456789", "nom_entreprise": "Test Company"}
    response = client.post("/api/v1/companies", json=data, headers=auth_headers)
    assert response.status_code == 201
```

#### Tests de scraping
```python
@pytest.mark.asyncio
async def test_pappers_scraper():
    scraper = PappersScraper()
    result = await scraper.scrape_company("123456789")
    assert "nom_entreprise" in result
    assert "siren" in result
```

#### Mocking APIs externes
```python
@pytest.fixture
def mock_pappers_api():
    with aioresponses() as m:
        m.get(
            "https://api.pappers.fr/v2/entreprise",
            payload={"nom": "Test Company"}
        )
        yield m

def test_with_mock(mock_pappers_api):
    # Test utilise le mock
    pass
```

## 🔧 Architecture & Patterns

### Structure recommandée

#### Endpoint pattern
```python
@router.post("/endpoint")
async def endpoint_name(
    # 1. Données d'entrée validées
    data: RequestSchema,
    # 2. Authentification OBLIGATOIRE
    current_user: User = Depends(get_current_active_user),
    # 3. Session DB
    db: Session = Depends(get_db)
):
    try:
        # 4. Logique métier dans services/
        result = await service_function(data, db)
        
        # 5. Log audit si nécessaire
        logger.info(f"Action performed by {current_user.email}")
        
        # 6. Retour standardisé
        return {"success": True, "data": result}
        
    except ValidationError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        logger.error(f"Endpoint error: {e}")
        raise HTTPException(500, "Internal error")
```

#### Service pattern
```python
# app/services/company_service.py
class CompanyService:
    def __init__(self, db: Session):
        self.db = db
    
    async def create_company(self, data: Dict) -> Company:
        # 1. Validation métier
        await self._validate_siren(data["siren"])
        
        # 2. Création
        company = Company(**data)
        self.db.add(company)
        self.db.commit()
        
        # 3. Actions post-création
        await self._schedule_enrichment(company.id)
        
        return company
```

### Gestion d'erreurs

#### Exceptions custom
```python
# app/core/exceptions.py
class CompanyNotFoundError(HTTPException):
    def __init__(self, siren: str):
        super().__init__(404, f"Company {siren} not found")

class ScrapingError(Exception):
    def __init__(self, source: str, message: str):
        self.source = source
        self.message = message
        super().__init__(f"Scraping {source}: {message}")
```

#### Error handling global
```python
# Dans main.py
@app.exception_handler(ScrapingError)
async def scraping_error_handler(request, exc):
    logger.error(f"Scraping error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Scraping temporarily unavailable"}
    )
```

### Performance & Optimisation

#### Queries efficaces
```python
# ✅ BON : Pagination + filtres
def get_companies(db: Session, skip: int = 0, limit: int = 100, filters: Dict = None):
    query = db.query(Company)
    
    if filters.get("statut"):
        query = query.filter(Company.statut == filters["statut"])
    
    return query.offset(skip).limit(min(limit, 100)).all()

# ❌ MAUVAIS : Pas de limite
def get_all_companies(db: Session):
    return db.query(Company).all()  # Peut renvoyer 100k+ résultats
```

#### Cache intelligent
```python
from functools import wraps
from app.core.cache import cache_manager

def cache_result(ttl: int = 300):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            cache_key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"
            
            # Vérifier cache
            cached = await cache_manager.get(cache_key)
            if cached:
                return cached
            
            # Calculer et mettre en cache
            result = await func(*args, **kwargs)
            await cache_manager.set(cache_key, result, ttl)
            return result
        return wrapper
    return decorator

@cache_result(ttl=3600)  # Cache 1h
async def expensive_computation(data):
    # Calcul coûteux
    pass
```

## 🔍 Debugging & Monitoring

### Logging

#### Configuration
```python
import logging

# Setup dans main.py
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/app.log"),
        logging.StreamHandler()
    ]
)

# Dans modules
logger = logging.getLogger(__name__)
```

#### Usage recommandé
```python
# ✅ Logs utiles
logger.info(f"User {user.email} created company {company.siren}")
logger.warning(f"Scraping rate limit hit for {source}")
logger.error(f"Database connection failed: {e}", exc_info=True)

# ❌ Logs inutiles
logger.info("Function called")  # Trop générique
logger.debug(f"Variable x = {x}")  # TMI en production
```

### Performance monitoring

#### Middleware timing
```python
@app.middleware("http")
async def timing_middleware(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start
    
    if duration > 1.0:  # Log requêtes lentes
        logger.warning(f"Slow request {request.url}: {duration:.2f}s")
    
    response.headers["X-Process-Time"] = str(duration)
    return response
```

#### Database monitoring
```python
# Dans app/core/database.py
from sqlalchemy import event
from sqlalchemy.engine import Engine

@event.listens_for(Engine, "before_cursor_execute")
def receive_before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
    conn.info.setdefault('query_start_time', []).append(time.time())

@event.listens_for(Engine, "after_cursor_execute")
def receive_after_cursor_execute(conn, cursor, statement, parameters, context, executemany):
    total = time.time() - conn.info['query_start_time'].pop(-1)
    if total > 0.5:  # Log queries lentes
        logger.warning(f"Slow query ({total:.2f}s): {statement[:100]}...")
```

## 🧪 Tests & Quality

### Structure tests
```
tests/
├── conftest.py              # Fixtures communes
├── test_api.py              # Tests endpoints
├── test_auth.py             # Tests authentification
├── test_models.py           # Tests modèles
├── test_scraping.py         # Tests scrapers
├── test_services.py         # Tests logique métier
└── test_integration.py      # Tests E2E
```

### Fixtures utiles
```python
# conftest.py
@pytest.fixture
def test_db():
    """Base de données test isolée"""
    engine = create_engine("sqlite:///test.db")
    Base.metadata.create_all(engine)
    session = SessionLocal(bind=engine)
    yield session
    session.close()
    os.remove("test.db")

@pytest.fixture
def test_user(test_db):
    """Utilisateur test"""
    user = User(email="test@example.com", hashed_password="hash")
    test_db.add(user)
    test_db.commit()
    return user

@pytest.fixture
def auth_headers(test_user):
    """Headers authentification"""
    token = create_access_token({"sub": test_user.email})
    return {"Authorization": f"Bearer {token}"}
```

### Tests patterns
```python
class TestCompanyAPI:
    def test_list_companies_success(self, client, auth_headers):
        response = client.get("/api/v1/companies", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert "companies" in data
        assert isinstance(data["companies"], list)
    
    def test_list_companies_unauthorized(self, client):
        response = client.get("/api/v1/companies")
        assert response.status_code == 401
    
    @pytest.mark.parametrize("invalid_siren", ["12345", "abcdefghi", ""])
    def test_create_company_invalid_siren(self, client, auth_headers, invalid_siren):
        data = {"siren": invalid_siren, "nom_entreprise": "Test"}
        response = client.post("/api/v1/companies", json=data, headers=auth_headers)
        assert response.status_code == 422
```

## 🚀 Déploiement

### Environnements

#### Development
```bash
export ENVIRONMENT=development
export DATABASE_URL=sqlite:///dev.db
export ENABLE_ADVANCED_FEATURES=true
uvicorn app.main:app --reload --port 8000
```

#### Production
```bash
export ENVIRONMENT=production
export DATABASE_URL=postgresql://user:pass@host/db
export ENABLE_ADVANCED_FEATURES=false
gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker
```

### Docker
```dockerfile
# Dockerfile.prod
FROM python:3.11-slim

WORKDIR /app

# Dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App
COPY app/ app/
COPY alembic/ alembic/
COPY alembic.ini .

# Run
CMD ["gunicorn", "app.main:app", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000"]
```

### CI/CD
```yaml
# .github/workflows/backend.yml
name: Backend CI
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: 3.11
      
      - name: Install dependencies
        run: |
          cd backend
          pip install -r requirements.txt
      
      - name: Run tests
        run: |
          cd backend
          pytest --cov=app tests/
      
      - name: Lint
        run: |
          cd backend
          black --check app/ tests/
          flake8 app/ tests/
```

## 📚 Ressources

### Documentation
- **FastAPI** : https://fastapi.tiangolo.com/
- **SQLAlchemy** : https://docs.sqlalchemy.org/
- **Alembic** : https://alembic.sqlalchemy.org/
- **Pytest** : https://docs.pytest.org/

### Outils développement
- **uvicorn** : Serveur ASGI développement
- **black** : Formatage code Python
- **flake8** : Linting Python
- **pytest** : Framework tests
- **alembic** : Migrations database

### APIs externes
- **Pappers** : https://www.pappers.fr/api
- **Supabase** : https://supabase.com/docs

---

**Pour questions techniques** : Voir ARCHITECTURE.md  
**Pour bugs** : Créer issue GitHub  
**Pour features** : Discussion équipe technique