# M&A Intelligence Platform v2.0

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)
![React](https://img.shields.io/badge/React-18+-lightblue.svg)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-14+-blue.svg)
![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

**Plateforme intelligente d'analyse M&A pour cabinets comptables français**

*Enrichissement automatisé • Scoring IA • Identification de cibles*

</div>

## 📋 Table des matières

- [Vue d'ensemble](#-vue-densemble)
- [Architecture technique](#-architecture-technique)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Usage](#-usage)
- [Tests](#-tests)
- [Déploiement](#-déploiement)
- [Modules principaux](#-modules-principaux)
- [API Documentation](#-api-documentation)
- [Roadmap](#-roadmap)
- [Contribution](#-contribution)
- [Support](#-support)

## 🎯 Vue d'ensemble

La **M&A Intelligence Platform** est une solution complète d'analyse et d'identification d'opportunités M&A dans l'écosystème des cabinets d'experts comptables français. Elle automatise l'enrichissement de données d'entreprises, calcule des scores de potentiel M&A et facilite l'export vers différents formats.

### Fonctionnalités principales

- **🔍 Enrichissement multi-sources** : Pappers API, Infogreffe, Société.com, Kaspr
- **🎯 Scoring M&A intelligent** : Algorithme configurable avec 8 composants
- **👥 Enrichissement contacts** : Dirigeants avec emails, téléphones, LinkedIn
- **📊 Exports flexibles** : CSV, Airtable, SQL avec formatage métier
- **⚡ Pipeline asynchrone** : Traitement haute performance par lots
- **🔒 Sécurité robuste** : JWT, rate limiting, validation des entrées
- **📈 Dashboard analytique** : Visualisation des KPIs et métriques M&A

### Cas d'usage cibles

- **Cabinets de conseil M&A** : Identification de cibles d'acquisition
- **Investisseurs** : Sourcing et qualification de prospects  
- **Cabinets comptables** : Analyse de portefeuilles clients
- **Analystes financiers** : Research et due diligence automatisés

## 🏗️ Architecture technique

### Stack technologique

```
Frontend                Backend                 Data Sources
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ React 18        │    │ FastAPI         │    │ Pappers API     │
│ Tailwind CSS    │    │ Python 3.8+     │    │ Infogreffe      │
│ ShadCN/UI       │◄──►│ SQLAlchemy      │◄──►│ Société.com     │
│ React Query     │    │ Alembic         │    │ Kaspr API       │
│ Recharts        │    │ PostgreSQL      │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │ Infrastructure  │
                    │ Docker          │
                    │ Nginx           │
                    │ Redis (cache)   │
                    └─────────────────┘
```

### Architecture des modules backend

```
app/
├── api/                     # Couche API REST
│   ├── routes/             # Endpoints par domaine
│   │   ├── auth.py         # Authentification JWT
│   │   ├── companies.py    # CRUD entreprises
│   │   ├── scraping.py     # Orchestration enrichissement
│   │   └── stats.py        # Analytics et KPIs
│   └── deps.py             # Dépendances FastAPI
│
├── core/                    # Configuration et utilitaires
│   ├── database.py         # Connexion PostgreSQL
│   ├── security.py         # JWT, hashing, validation
│   ├── exceptions.py       # Gestion d'erreurs custom
│   └── constants.py        # Configuration métier
│
├── models/                  # Modèles de données
│   ├── company.py          # Entreprise + contacts
│   ├── user.py             # Utilisateurs
│   └── schemas.py          # Schémas Pydantic
│
├── services/                # Logique métier
│   ├── scraping_orchestrator.py    # Pipeline enrichissement
│   ├── ma_scoring.py              # Scoring M&A
│   ├── export_manager.py          # Exports multi-formats
│   └── data_processing.py         # Traitement données
│
└── scrapers/               # Clients API externes
    ├── pappers.py          # API Pappers (données légales)
    ├── infogreffe.py       # Registre du commerce
    ├── societe.py          # Scraping Société.com
    └── kaspr.py            # Enrichissement contacts
```

### Flux de données

```
┌─────────────┐    ┌─────────────────┐    ┌──────────────┐    ┌─────────────┐
│   Import    │    │  Enrichissement │    │   Scoring    │    │   Export    │
│   SIREN     │───►│  Multi-sources  │───►│     M&A      │───►│   Données   │
│             │    │                 │    │              │    │             │
└─────────────┘    └─────────────────┘    └──────────────┘    └─────────────┘
       │                      │                     │                  │
       │            ┌─────────▼─────────┐          │         ┌────────▼────────┐
       │            │ • Pappers API     │          │         │ • CSV (Excel)   │
       │            │ • Infogreffe      │          │         │ • Airtable      │
       │            │ • Société.com     │          │         │ • SQL Database  │
       │            │ • Kaspr Contacts  │          │         │ • API JSON      │
       │            └───────────────────┘          │         └─────────────────┘
       │                                          │
       │            ┌─────────────────────────────▼─────────────────────────────┐
       │            │           Score M&A (0-100)                              │
       └───────────►│ • Performance financière  • Trajectoire croissance      │
                    │ • Rentabilité             • Risque endettement          │
                    │ • Taille critique         • Qualité dirigeants          │
                    │ • Position marché         • Innovation/digitalisation   │
                    └───────────────────────────────────────────────────────────┘
```

## 🚀 Installation

### Prérequis

- **Python 3.8+** avec pip
- **Node.js 16+** avec npm
- **PostgreSQL 14+** ou Docker
- **Git** pour le versioning

### Installation locale

#### 1. Clonage du projet

```bash
git clone https://github.com/your-org/ma-intelligence-platform.git
cd ma-intelligence-platform
```

#### 2. Configuration Backend

```bash
# Création environnement virtuel
cd backend
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows

# Installation dépendances
pip install -r requirements.txt

# Configuration base de données
cp .env.example .env
# Éditer .env avec vos paramètres
```

#### 3. Configuration Frontend

```bash
cd ../frontend
npm install
cp .env.example .env
# Configurer l'URL de l'API backend
```

#### 4. Base de données

```bash
# Avec PostgreSQL local
createdb ma_intelligence

# Ou avec Docker
docker run -d \
  --name ma-postgres \
  -e POSTGRES_DB=ma_intelligence \
  -e POSTGRES_USER=ma_user \
  -e POSTGRES_PASSWORD=your_password \
  -p 5432:5432 \
  postgres:14

# Migrations
cd backend
alembic upgrade head

# Données initiales (optionnel)
python create_test_users.py
```

### Installation avec Docker

```bash
# Lancement complet
docker-compose up -d

# Build et lancement
docker-compose build
docker-compose up

# Logs
docker-compose logs -f backend
```

## ⚙️ Configuration

### Variables d'environnement Backend

Créer `backend/.env` :

```env
# Base de données
DATABASE_URL=postgresql://user:password@localhost:5432/ma_intelligence

# Sécurité
SECRET_KEY=your-super-secret-key-min-32-chars
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# APIs externes (optionnelles)
PAPPERS_API_KEY=your-pappers-api-key
KASPR_API_KEY=your-kaspr-api-key
AIRTABLE_API_KEY=your-airtable-api-key

# Configuration scoring
DEFAULT_SCORING_WEIGHTS=balanced

# Environnement
ENVIRONMENT=development
DEBUG=true
ALLOWED_ORIGINS=http://localhost:3000,http://127.0.0.1:3000
```

### Variables d'environnement Frontend

Créer `frontend/.env` :

```env
# API Backend
REACT_APP_API_URL=http://localhost:8000/api/v1

# Configuration UI
REACT_APP_COMPANY_NAME=M&A Intelligence Platform
REACT_APP_VERSION=2.0.0

# Analytics (optionnel)
REACT_APP_ANALYTICS_ID=your-analytics-id
```

### Configuration avancée

#### Scoring M&A personnalisé

```python
# backend/app/core/constants.py
CUSTOM_SCORING_WEIGHTS = {
    'financial_performance': 0.25,    # Performance financière
    'growth_trajectory': 0.20,        # Trajectoire de croissance
    'profitability': 0.15,           # Rentabilité
    'debt_risk': 0.10,               # Risque d'endettement
    'critical_size': 0.10,           # Taille critique
    'management_quality': 0.10,      # Qualité management
    'market_position': 0.05,         # Position marché
    'innovation_digital': 0.05       # Innovation/digitalisation
}
```

#### Rate limiting

```python
# backend/app/core/rate_limiting.py
RATE_LIMITS = {
    'pappers': {'calls': 1000, 'period': 'day'},
    'kaspr': {'calls': 500, 'period': 'day'},
    'exports': {'calls': 100, 'period': 'hour'}
}
```

## 💻 Usage

### Démarrage des services

#### Développement local

```bash
# Terminal 1 - Backend
cd backend
source venv/bin/activate
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Terminal 2 - Frontend  
cd frontend
npm start

# Accès
# Frontend: http://localhost:3000
# API: http://localhost:8000
# Docs API: http://localhost:8000/docs
```

#### Production

```bash
# Avec Docker
docker-compose --profile production up -d

# Ou manuel
cd backend
gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker

cd frontend
npm run build
serve -s build
```

### Exemples d'usage API

#### 1. Authentification

```bash
# Login
curl -X POST "http://localhost:8000/api/v1/auth/login" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=admin&password=secret"

# Réponse
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "token_type": "bearer"
}
```

#### 2. Enrichissement d'entreprise

```bash
# Enrichissement simple
curl -X POST "http://localhost:8000/api/v1/scraping/enrich" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "siren": "123456789",
    "sources": ["pappers", "infogreffe", "societe", "kaspr"],
    "force_refresh": true
  }'

# Réponse
{
  "siren": "123456789",
  "status": "success",
  "data": {
    "nom_entreprise": "Cabinet Exemple SAS",
    "chiffre_affaires": 15000000,
    "effectif": 85,
    "ma_score": 78.5,
    "priorite_contact": "MEDIUM",
    "kaspr_contacts": [
      {
        "nom_complet": "Jean Martin",
        "poste": "PDG",
        "email_professionnel": "j.martin@cabinet-exemple.fr",
        "telephone_direct": "01 23 45 67 89"
      }
    ]
  }
}
```

#### 3. Enrichissement par lot

```bash
# Upload CSV + enrichissement
curl -X POST "http://localhost:8000/api/v1/scraping/batch" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "file=@companies.csv" \
  -F "max_concurrent=3" \
  -F "sources=pappers,kaspr"
```

#### 4. Export de données

```bash
# Export CSV
curl -X POST "http://localhost:8000/api/v1/companies/export/csv" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "filters": {
      "score_min": 70,
      "ca_min": 5000000,
      "priorite": ["HIGH", "MEDIUM"]
    },
    "format": "ma_analysis",
    "include_contacts": true
  }'

# Export Airtable
curl -X POST "http://localhost:8000/api/v1/companies/export/airtable" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "base_id": "appXXXXXXXXXXXXXX",
    "table_name": "Prospects_MA",
    "filters": {"score_min": 60}
  }'
```

### Exemples d'usage Python

#### Enrichissement programmatique

```python
import asyncio
from app.services.scraping_orchestrator import ScrapingOrchestrator
from app.core.database import get_db

async def enrich_company_example():
    async with get_db() as db:
        async with ScrapingOrchestrator(db) as orchestrator:
            result = await orchestrator.enrich_company_full(
                siren="123456789",
                force_refresh=True
            )
            
            print(f"Score M&A: {result['data']['ma_score']}")
            print(f"Contacts: {len(result['data']['kaspr_contacts'])}")

# Exécution
asyncio.run(enrich_company_example())
```

#### Scoring personnalisé

```python
from app.services.ma_scoring import MAScoring, ScoringWeights

# Configuration scoring agressive
aggressive_weights = ScoringWeights(
    financial_performance=0.30,
    growth_trajectory=0.25,
    profitability=0.20,
    debt_risk=0.15,
    critical_size=0.10
)

scoring = MAScoring(weights=aggressive_weights)
score_result = await scoring.calculate_ma_score(company_data)

print(f"Score: {score_result.final_score}/100")
print(f"Détail: {score_result.component_scores}")
```

#### Export avancé

```python
from app.services.export_manager import ExportManager

async def export_prospects():
    export_manager = ExportManager()
    
    # Filtrage des prospects prioritaires
    high_value_companies = await get_companies_by_criteria({
        'ma_score__gte': 75,
        'chiffre_affaires__gte': 10_000_000,
        'priorite_contact': 'HIGH'
    })
    
    # Export multi-format
    csv_result = await export_manager.export_to_csv(
        companies=high_value_companies,
        export_format="ma_analysis",
        include_contacts=True
    )
    
    airtable_result = await export_manager.sync_to_airtable(
        companies=high_value_companies,
        base_id="appXXXXXXXXXXXXXX",
        table_name="Prospects_Premium"
    )
    
    return {
        'csv_file': csv_result.file_path,
        'airtable_synced': airtable_result.records_exported
    }
```

## 🧪 Tests

### Tests unitaires

```bash
# Backend - Tests complets
cd backend
pytest --cov=app tests/

# Tests spécifiques
pytest tests/test_api.py                    # Tests API
pytest tests/test_scoring.py               # Tests scoring M&A
pytest tests/test_scrapers.py              # Tests scrapers
pytest tests/test_export.py                # Tests exports

# Avec couverture détaillée
pytest --cov=app --cov-report=html tests/
# Rapport: htmlcov/index.html
```

### Tests d'intégration

```bash
# Tests base de données
pytest tests/test_database.py

# Tests API end-to-end
pytest tests/test_api_integration.py

# Tests avec vraies APIs (nécessite clés)
PAPPERS_API_KEY=real-key pytest tests/test_real_apis.py
```

### Tests End-to-End

```bash
# Suite complète E2E (avec mocks)
cd backend
./run_e2e_tests.sh

# Ou manuel
python test_e2e_ma_pipeline.py

# Tests spécifiques
python test_kaspr.py                       # Test enrichissement contacts
python demo_export_manager.py             # Test exports
```

### Tests Frontend

```bash
cd frontend

# Tests unitaires
npm test

# Tests avec couverture
npm run test:coverage

# Tests E2E avec Cypress
npm run cypress:open
npm run cypress:run
```

### Tests de performance

```bash
# Load testing avec locust
cd backend
pip install locust
locust -f tests/load_test.py

# Benchmarks enrichissement
python tests/benchmark_enrichment.py

# Profiling
python -m cProfile -o profile.stats tests/test_performance.py
python -c "import pstats; pstats.Stats('profile.stats').sort_stats('cumulative').print_stats(20)"
```

## 🚢 Déploiement

### Déploiement Docker (recommandé)

#### 1. Build des images

```bash
# Images complètes
docker-compose build

# Images production optimisées
docker-compose -f docker-compose.prod.yml build
```

#### 2. Configuration production

Créer `docker-compose.prod.yml` :

```yaml
version: '3.8'

services:
  backend:
    build:
      context: ./backend
      dockerfile: Dockerfile.prod
    environment:
      - DATABASE_URL=postgresql://user:pass@db:5432/ma_intelligence
      - SECRET_KEY=${SECRET_KEY}
      - ENVIRONMENT=production
      - DEBUG=false
    depends_on:
      - db
      - redis

  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile.prod
    environment:
      - REACT_APP_API_URL=https://api.ma-intelligence.com

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/ssl
    depends_on:
      - backend
      - frontend

  db:
    image: postgres:14
    environment:
      - POSTGRES_DB=ma_intelligence
      - POSTGRES_USER=${DB_USER}
      - POSTGRES_PASSWORD=${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:alpine
    volumes:
      - redis_data:/data

volumes:
  postgres_data:
  redis_data:
```

#### 3. Lancement production

```bash
# Variables d'environnement
export SECRET_KEY="your-super-secret-production-key"
export DB_USER="ma_user"
export DB_PASSWORD="secure-password"

# Déploiement
docker-compose -f docker-compose.prod.yml up -d

# Vérification
docker-compose ps
docker-compose logs -f backend
```

### Déploiement Cloud

#### AWS (ECS + RDS + ElastiCache)

```bash
# 1. Infrastructure avec Terraform
cd infrastructure/aws
terraform init
terraform plan
terraform apply

# 2. Build et push images
aws ecr get-login-password --region eu-west-1 | docker login --username AWS --password-stdin 123456789.dkr.ecr.eu-west-1.amazonaws.com

docker build -t ma-intelligence-backend ./backend
docker tag ma-intelligence-backend:latest 123456789.dkr.ecr.eu-west-1.amazonaws.com/ma-intelligence-backend:latest
docker push 123456789.dkr.ecr.eu-west-1.amazonaws.com/ma-intelligence-backend:latest

# 3. Déploiement ECS
aws ecs update-service --cluster ma-intelligence --service backend --force-new-deployment
```

#### Google Cloud Platform (Cloud Run + Cloud SQL)

```bash
# 1. Configuration
gcloud config set project ma-intelligence-platform

# 2. Build et déploiement
gcloud builds submit --tag gcr.io/ma-intelligence-platform/backend ./backend
gcloud run deploy backend \
  --image gcr.io/ma-intelligence-platform/backend \
  --platform managed \
  --region europe-west1 \
  --add-cloudsql-instances ma-intelligence:europe-west1:ma-db

# 3. Configuration base de données
gcloud sql databases create ma_intelligence --instance=ma-db
```

### Configuration SSL/TLS

#### Nginx avec Let's Encrypt

```nginx
# nginx/nginx.conf
server {
    listen 80;
    server_name ma-intelligence.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name ma-intelligence.com;

    ssl_certificate /etc/ssl/certs/ma-intelligence.crt;
    ssl_certificate_key /etc/ssl/private/ma-intelligence.key;

    location / {
        proxy_pass http://frontend:3000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    location /api {
        proxy_pass http://backend:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### Checklist déploiement production

- [ ] **Sécurité**
  - [ ] Clés secrètes générées et sécurisées
  - [ ] SSL/TLS configuré
  - [ ] Rate limiting activé
  - [ ] CORS configuré restrictif
  - [ ] Validation d'entrée stricte

- [ ] **Base de données**
  - [ ] Migrations appliquées
  - [ ] Utilisateur de service créé
  - [ ] Backups automatiques configurés
  - [ ] Index optimisés

- [ ] **Infrastructure**
  - [ ] Load balancer configuré
  - [ ] Auto-scaling activé
  - [ ] Health checks opérationnels
  - [ ] Monitoring déployé

- [ ] **Tests**
  - [ ] Tests E2E passent en environnement staging
  - [ ] Load tests validés
  - [ ] Rollback testé

## 📚 Modules principaux

### 🔄 Scraping Orchestrator

**Fichier** : `backend/app/services/scraping_orchestrator.py`

**Responsabilité** : Coordination de l'enrichissement multi-sources avec gestion d'erreurs robuste.

**Fonctionnalités** :
- Pipeline asynchrone configurable (Pappers → Infogreffe → Société.com → Kaspr)
- Retry automatique avec backoff exponentiel
- Validation des données selon critères M&A
- Hooks extensibles pour scoring et export
- Statistiques détaillées par source

**Usage** :
```python
async with ScrapingOrchestrator(db_client) as orchestrator:
    result = await orchestrator.enrich_company_full("123456789")
    batch_results = await orchestrator.enrich_companies_batch(siren_list)
```

### 🎯 MA Scoring Engine

**Fichier** : `backend/app/services/ma_scoring.py`

**Responsabilité** : Calcul intelligent du potentiel M&A avec pondérations configurables.

**Composants de scoring** :
- Performance financière (25%) : CA, croissance, rentabilité
- Trajectoire de croissance (20%) : Évolution 3 ans, tendances
- Rentabilité (15%) : Marges, efficacité opérationnelle
- Risque d'endettement (10%) : Ratios financiers, stabilité
- Taille critique (10%) : Effectif, market fit
- Qualité management (10%) : Profils dirigeants, gouvernance
- Position marché (5%) : Concurrence, différenciation
- Innovation/digitalisation (5%) : Modernité, tech stack

**Configurations prédéfinies** :
- `balanced` : Répartition équilibrée
- `growth_focused` : Priorité croissance
- `value_focused` : Focus rentabilité
- `risk_averse` : Minimisation risques

### 👥 Kaspr Contact Enrichment

**Fichier** : `backend/app/scrapers/kaspr.py`

**Responsabilité** : Enrichissement contacts dirigeants avec système de mock intelligent.

**Fonctionnalités** :
- API Kaspr réelle ou mock automatique selon disponibilité clé
- Ciblage dirigeants (CEO, PDG, DG, CFO, Associés)
- Scoring de priorité des contacts (0-100)
- Validation emails et téléphones
- Formatage pour modèle CompanyContact

**Données enrichies** :
- Informations personnelles (nom, prénom, poste)
- Coordonnées (email pro, tél direct/mobile, LinkedIn)
- Métadonnées (confiance, ancienneté, département)

### 📤 Export Manager

**Fichier** : `backend/app/services/export_manager.py`

**Responsabilité** : Exports multi-formats avec formatage métier adapté.

**Formats supportés** :
- **CSV** : MA Analysis (complet), Excel (français), Standard (brut)
- **Airtable** : Sync bidirectionnelle avec mapping automatique
- **SQL** : PostgreSQL, MySQL, SQLite, SQL Server, Oracle

**Fonctionnalités avancées** :
- Filtrage et tri des données
- Formatage français (devises, dates, pourcentages)
- Métadonnées d'export intégrées
- Gestion par lots avec rate limiting
- Statistiques d'export détaillées

### 🔌 External Scrapers

#### Pappers API Client
**Fichier** : `backend/app/scrapers/pappers.py`
- Données légales officielles (SIREN, forme juridique, dirigeants)
- Informations financières de base
- Historique et événements entreprise

#### Infogreffe API Client  
**Fichier** : `backend/app/scrapers/infogreffe.py`
- Données du registre du commerce
- Actes et publications officielles
- Validation SIREN/SIRET

#### Société.com Scraper
**Fichier** : `backend/app/scrapers/societe.py`
- Scraping web avec Playwright
- Données financières détaillées
- Informations concurrence et marché

### 🗄️ Data Models

**Fichiers** : `backend/app/models/`

#### Company Model
- Informations de base (SIREN, nom, forme juridique)
- Données financières (CA, résultat, effectif)
- Scores M&A et métadonnées d'enrichissement
- Relations avec contacts et historique

#### CompanyContact Model
- Contacts dirigeants multi-sources
- Coordonnées vérifiées
- Scoring de qualité et priorité
- Tracking des interactions

### 🔐 Security & Core

**Fichiers** : `backend/app/core/`

#### Security
- Authentification JWT avec refresh tokens
- Hachage bcrypt pour mots de passe
- Validation Pydantic stricte
- Rate limiting par utilisateur/endpoint

#### Database
- SQLAlchemy avec sessions async
- Pool de connexions optimisé
- Migrations Alembic automatisées
- Cache Redis pour performances

## 📖 API Documentation

### Accès documentation

- **Swagger UI** : `http://localhost:8000/docs`
- **ReDoc** : `http://localhost:8000/redoc`
- **OpenAPI JSON** : `http://localhost:8000/openapi.json`

### Endpoints principaux

| Endpoint | Méthode | Description |
|----------|---------|-------------|
| `/api/v1/auth/login` | POST | Authentification utilisateur |
| `/api/v1/companies` | GET | Liste des entreprises |
| `/api/v1/companies/{id}` | GET | Détail entreprise |
| `/api/v1/scraping/enrich` | POST | Enrichissement simple |
| `/api/v1/scraping/batch` | POST | Enrichissement par lot |
| `/api/v1/companies/export/csv` | POST | Export CSV |
| `/api/v1/companies/export/airtable` | POST | Export Airtable |
| `/api/v1/stats/dashboard` | GET | KPIs dashboard |

### Schémas de données

```python
# Enrichissement request
{
  "siren": "123456789",
  "sources": ["pappers", "infogreffe", "societe", "kaspr"],
  "force_refresh": true
}

# Company response
{
  "siren": "123456789",
  "nom_entreprise": "Cabinet Exemple SAS",
  "chiffre_affaires": 15000000,
  "ma_score": 78.5,
  "priorite_contact": "MEDIUM",
  "kaspr_contacts": [...],
  "enrichment_status": {...}
}
```

## 🛣️ Roadmap

### Version 2.1

#### 🔍 Enrichissement avancé
- [ ] **Scraper LinkedIn Sales Navigator** : Contacts dirigeants premium
- [ ] **API Infonet** : Données financières temps réel
- [ ] **Scraper Décideurs Magazine** : Actualités dirigeants
- [ ] **Enrichissement géographique** : Zones de chalandise, concurrence locale

#### 🎯 Scoring IA amélioré
- [ ] **Machine Learning** : Modèle prédictif basé sur historique deals
- [ ] **NLP pour actualités** : Analyse sentiment et événements marquants
- [ ] **Scoring concurrence** : Analyse comparative sectorielle
- [ ] **Prédiction churn** : Risque de perte de clients

#### 📊 Analytics avancés
- [ ] **Dashboard prédictif** : Tendances marché et opportunités
- [ ] **Alertes intelligentes** : Notifications événements M&A
- [ ] **Reporting automatisé** : Rapports périodiques personnalisés
- [ ] **Benchmarking sectoriel** : Comparaisons par secteur d'activité

### Version 2.2 

#### 🔗 Intégrations CRM
- [ ] **Salesforce connector** : Sync bidirectionnelle prospects
- [ ] **HubSpot integration** : Pipeline M&A automatisé
- [ ] **Microsoft Dynamics** : Workflow deal management
- [ ] **Pipedrive sync** : Suivi commercial enrichi

#### 🤖 Automatisation workflows
- [ ] **Séquences email** : Outreach personnalisé dirigeants
- [ ] **Qualification automatique** : Scoring en temps réel
- [ ] **Orchestration deals** : Workflow M&A de bout en bout
- [ ] **AI-powered insights** : Recommandations IA personnalisées

#### 🌍 Expansion géographique
- [ ] **Base UE** : Enrichissement entreprises européennes
- [ ] **API Companies House** : Données UK
- [ ] **Registres belges/suisses** : Couverture DACH
- [ ] **Compliance RGPD** : Respect réglementations européennes

### Version 3.0

#### 🚀 Plateforme complète M&A
- [ ] **Deal room virtuel** : Collaboration sécurisée M&A
- [ ] **Due diligence automatisée** : Checklist et validation IA
- [ ] **Modélisation financière** : Templates et simulation deals
- [ ] **Legal document analysis** : IA pour analyse contrats

#### 📱 Applications mobiles
- [ ] **App iOS/Android** : CRM mobile dirigeants
- [ ] **Notifications push** : Alertes opportunités temps réel
- [ ] **Mode offline** : Consultation données sans réseau
- [ ] **Scan cartes visite** : Enrichissement contact instantané

#### 🔒 Sécurité enterprise
- [ ] **SSO enterprise** : SAML, OAuth2, LDAP
- [ ] **Audit logs** : Traçabilité complète actions
- [ ] **Encryption E2E** : Chiffrement données sensibles
- [ ] **Compliance SOC2** : Certification sécurité enterprise

### Innovations R&D

#### 🧠 Intelligence artificielle
- [ ] **GPT-4 integration** : Analyse qualitative dirigeants
- [ ] **Computer vision** : Analyse documents financiers
- [ ] **Predictive modeling** : ML prédiction succès deals
- [ ] **Natural Language Query** : Recherche en langage naturel

#### 🔮 Technologies émergentes
- [ ] **Quantum computing** : Optimisation portefeuilles M&A
- [ ] **IoT data integration** : Données objets connectés entreprises
- [ ] **AR/VR due diligence** : Visite virtuelle actifs
- [ ] **Edge computing** : Traitement local données sensibles

### Métriques de succès

| KPI | Objectif 2024 | Objectif 2025 |
|-----|---------------|---------------|
| Entreprises enrichies/mois | 100K | 500K |
| Précision scoring M&A | 85% | 92% |
| Temps enrichissement moyen | <30s | <10s |
| Taux conversion prospects | 15% | 25% |
| NPS utilisateurs | +50 | +70 |

## 🤝 Contribution

### Guide de contribution

1. **Fork** le projet
2. **Créer une branche** : `git checkout -b feature/nouvelle-fonctionnalite`
3. **Commit changes** : `git commit -m 'Add: nouvelle fonctionnalité'`
4. **Push branch** : `git push origin feature/nouvelle-fonctionnalite`
5. **Ouvrir Pull Request**

### Standards de code

- **Python** : PEP 8, type hints, docstrings
- **JavaScript** : ESLint, Prettier, JSDoc
- **Tests** : Couverture >80%, tests unitaires + intégration
- **Documentation** : README, API docs, inline comments

### Architecture Decision Records (ADR)

Voir `docs/adr/` pour les décisions architecturales importantes.

**M&A Intelligence Platform v2.0**

*Développé avec ❤️ pour révolutionner l'analyse M&A*

[![GitHub stars](https://img.shields.io/github/stars/your-org/ma-intelligence-platform.svg?style=social&label=Star)](https://github.com/your-org/ma-intelligence-platform)
[![GitHub forks](https://img.shields.io/github/forks/your-org/ma-intelligence-platform.svg?style=social&label=Fork)](https://github.com/your-org/ma-intelligence-platform/fork)

[🌟 Star le projet](https://github.com/your-org/ma-intelligence-platform) • [🐛 Reporter un bug](https://github.com/your-org/ma-intelligence-platform/issues) • [💡 Proposer une amélioration](https://github.com/your-org/ma-intelligence-platform/issues/new)

</div>