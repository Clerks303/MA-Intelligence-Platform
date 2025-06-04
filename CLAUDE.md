# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**M&A Intelligence Platform v2.0** - Full-stack application for scraping and analyzing French accounting firms data to identify M&A opportunities. Built with FastAPI backend, React frontend, and Docker containerization.

## Commands

### Development

```bash
# Full stack with Docker
docker-compose up -d

# Backend only (recommended)
./start_backend.sh
./start_backend_simple.sh  # Simplified version for quick testing

# Backend manual start
cd backend && uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Frontend
cd frontend && npm start          # Uses CRACO configuration
cd frontend && npm run dev        # Alternative dev command

# Production deployment
docker-compose --profile production up -d
```

### Database Operations

```bash
# Run migrations (from backend directory)
cd backend && alembic upgrade head

# Create new migration
cd backend && alembic revision --autogenerate -m "description"

# Initialize database
cd backend && python -c "from app.core.database import init_db; import asyncio; asyncio.run(init_db())"
```

### Testing & Code Quality

```bash
# Backend tests
cd backend && pytest
cd backend && pytest --cov=app tests/
cd backend && pytest tests/test_api.py                    # API tests
cd backend && pytest tests/test_security_us007.py        # Security tests
cd backend && pytest tests/test_performance.py           # Performance tests

# End-to-end testing
cd backend && ./run_e2e_tests.sh
cd backend && python test_e2e_ma_pipeline.py

# User Story validation
cd backend && python validate_us005.py --benchmark
cd backend && python validate_us006.py --detailed
cd backend && python validate_us007.py
cd backend && python validate_us008.py
cd backend && python validate_us009.py
cd backend && python app/scripts/validate_us010.py
cd backend && python app/scripts/validate_us011.py
cd backend && python app/scripts/validate_us012.py

# Security auditing
cd backend && bandit -r app/                              # Security linting
cd backend && safety check                                # Dependency vulnerabilities  
cd backend && semgrep --config=auto app/                  # Static analysis

# Frontend tests  
cd frontend && npm test
cd frontend && npm run test:coverage                      # With coverage

# Frontend type checking
cd frontend && npm run type-check

# Frontend build (production)
cd frontend && npm run build

# Code formatting
cd backend && black app/ tests/

# Performance monitoring
cd backend && python scripts/monitor_slow_queries.py
cd backend && python scripts/test_database_performance.py
```

### Dependencies

```bash
# Backend dependencies
cd backend && pip install -r requirements.txt

# Frontend dependencies
cd frontend && npm install

# Update after adding packages
cd backend && pip freeze > requirements.txt
cd frontend && npm install --save package-name
```

## Architecture

### Full-Stack Structure

**Backend (FastAPI)**
- **API Routes**: `/api/v1/` prefix with auth, companies, scraping, stats endpoints
- **Authentication**: JWT-based with passlib bcrypt hashing and database user validation
- **Database**: Supabase (PostgreSQL) with SQLAlchemy 2.0+ and Alembic migrations
- **Scraping Engine**: Async scrapers for Pappers API, Société.com, and Infogreffe
- **Security**: CORS, rate limiting, input validation, custom middleware
- **Data Validation**: Pydantic 2.0+ with email validation

**Frontend (React)**
- **UI Framework**: Material-UI v7.1.1 + ShadCN/UI + Tailwind CSS v3.4.16 + Radix UI primitives
- **State Management**: TanStack Query v5.17.19 for server state, Zustand v5.0.5 for client state
- **Routing**: React Router v6 with protected routes
- **Forms**: React Hook Form v7.51.0 + Zod v3.22.4, Formik v2.4.6 + Yup v1.6.1
- **Data Visualization**: Recharts v2.15.3 + D3.js v7.9.0 for analytics dashboard
- **Icons**: Lucide React + MUI Icons
- **Type Safety**: TypeScript v4.9.5 with strict configuration
- **Build Tool**: CRACO v7.1.0 (Create React App Configuration Override)
- **Tables**: TanStack React Virtual v3.13.9 + React Window v1.8.11 for virtualization
- **Animations**: Framer Motion v12.16.0
- **File Handling**: React Dropzone v14.3.8 + React PDF v9.2.1

**Infrastructure**
- **Containerization**: Docker with multi-stage builds
- **Reverse Proxy**: Nginx for production with SSL/TLS
- **Caching**: Redis for session management and performance optimization
- **Task Queue**: Celery with Redis broker for async processing
- **Environment**: Separate .env files for backend and frontend
- **Monitoring**: Prometheus/Grafana integration with health checks

### Key Business Logic

**Company Data Model**
- SIREN/SIRET as primary identifiers (French business registry numbers)
- Enrichment pipeline: Basic data → Scraping → AI scoring → Export
- Status tracking: "à contacter", "contacté", "qualifié", etc.

**Scraping Pipeline**
1. **Data Input**: CSV import or manual search
2. **Enrichment**: Pappers API for company details
3. **Deep Scraping**: Société.com for financials and contacts
4. **Scoring**: AI-powered M&A potential assessment
5. **Export**: Filtered CSV downloads

**Authentication Flow**
- JWT tokens with 30-minute expiration
- Database user validation on each request
- Protected routes using `get_current_active_user` dependency

### Configuration

**Required Environment Variables**
```env
# Backend (.env in backend/ directory)
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-anon-key
SECRET_KEY=your-32-character-minimum-secret-key
FIRST_SUPERUSER=admin
FIRST_SUPERUSER_PASSWORD=minimum-8-chars

# Redis Configuration
REDIS_URL=redis://localhost:6379
REDIS_DATABASE_CACHE=0
REDIS_DATABASE_SESSIONS=1
REDIS_DATABASE_CELERY=2

# Celery Configuration
CELERY_BROKER_URL=redis://localhost:6379/2
CELERY_RESULT_BACKEND=redis://localhost:6379/2

# Performance & Features
FEATURE_FLAGS_ENABLED=true
RATE_LIMITING_ENABLED=true
PERFORMANCE_MONITORING=true

# Optional API keys
PAPPERS_API_KEY=your-pappers-key
KASPR_API_KEY=your-kaspr-key
OPENAI_API_KEY=sk-your-openai-key
AIRTABLE_API_KEY=your-airtable-key

# Frontend (.env in frontend/ directory)  
REACT_APP_API_URL=http://localhost:8000/api/v1
```

**Default Test Account**
- Username: `admin`
- Password: `secret`

**Development URLs**
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs (Swagger UI)
- API Documentation: http://localhost:8000/redoc (ReDoc)

### Frontend Architecture Details

**Modern React Architecture**
```
frontend/src/
├── components/          # Reusable UI components
│   ├── ui/             # ShadCN/UI base components
│   ├── auth/           # Authentication components
│   ├── companies/      # Company-specific components
│   ├── dashboard/      # Dashboard widgets
│   ├── scraping/       # Scraping interface components
│   └── monitoring/     # System monitoring components
├── features/           # Feature-based modules
│   ├── dashboard/      # Dashboard feature
│   │   ├── components/ # Feature components
│   │   ├── hooks/      # Feature hooks
│   │   ├── services/   # Feature services
│   │   └── types/      # Feature types
│   └── documents/      # Document management
├── contexts/           # React contexts (auth, theme, monitoring)
├── hooks/              # Shared custom hooks
├── stores/             # Zustand stores
├── types/              # TypeScript type definitions
└── utils/              # Utility functions
```

**TypeScript Integration**
- Strict TypeScript configuration with comprehensive type checking
- Path aliases configured (`@/*`, `@/components/*`, etc.)
- Zod schemas for runtime validation and TypeScript inference
- Type-safe API calls with proper error handling
- Component prop types with proper generics

**Tailwind Configuration**
- Custom M&A brand colors (`ma-blue`, `ma-slate`, etc.)
- Extended spacing and typography scales
- Custom animations and keyframes
- Responsive design utilities and breakpoints

### Data Flow

**Import Process**
1. CSV upload via drag-drop interface (`frontend/src/pages/Scraping.js`)
2. File processing in `backend/app/services/data_processing.py`
3. SIREN validation using `backend/app/core/validators.py`
4. Database insertion via Supabase client

**Scraping Process**
1. Batch selection in DataGrid interface
2. Async scraping orchestration (`backend/app/api/routes/scraping.py`)
3. Rate-limited API calls to external services
4. Real-time progress updates via polling

**Analytics Dashboard**
1. KPI calculations in `backend/app/api/routes/stats.py`
2. Caching layer (5-minute TTL) for performance
3. Interactive charts in `frontend/src/pages/Dashboard.js`

### Development Patterns

**Backend Patterns**
- Async/await throughout for performance
- Pydantic models for request/response validation
- Custom exceptions with structured error responses
- Dependency injection for database and auth
- Celery for async task processing and background jobs
- Redis caching with multi-database separation by use case

**Frontend Patterns**
- Custom hooks for API calls and state management
- ShadCN/UI design system with Tailwind CSS utilities and Radix UI primitives
- TanStack Query for server state caching and synchronization
- Modular component structure: `ui/`, `auth/`, `companies/`, `dashboard/`, `scraping/`, `monitoring/`
- TypeScript with strict type checking and Zod schemas
- Feature-based architecture in `features/` directory
- Secure token storage with automatic expiration handling
- Dark/light theme support with CSS custom properties

**Security Patterns**
- All API endpoints require authentication (except `/auth/login`)
- Input sanitization for search parameters (SQL injection prevention)
- CORS restricted to specific origins
- Rate limiting: 100 requests/minute, 1000/hour

### Key Project Structure

**Backend Structure**
- `backend/app/main.py` - FastAPI application entry point
- `backend/app/main_simple.py` - Simplified version for testing
- `backend/start_backend.sh` - Quick start script
- `backend/start_backend_simple.sh` - Simplified start script
- `backend/scripts/` - Validation and monitoring scripts
- `backend/logs/` - Structured logging with rotation
- `backend/sql/` - Database optimization scripts
- `backend/scheduler/` - Celery task scheduler configuration
- `backend/sdks/` - Multi-language SDKs (Python, JavaScript, PHP)
- `backend/shared/` - Shared utilities and configurations

**Frontend Structure**
- `frontend/craco.config.js` - CRACO configuration for custom webpack
- `frontend/tailwind.config.js` - Tailwind CSS configuration
- `frontend/tsconfig.json` - TypeScript configuration

**Advanced Components Not in Standard Structure**
- ML service in `backend/ml-service/` for machine learning operations
- Advanced AI engines in `backend/app/core/advanced_*` modules
- Monitoring stack with Prometheus/Grafana integration
- Feature flags system for gradual rollouts
- Multi-format export manager with business intelligence

### Common Development Tasks

**Adding New API Endpoint**
1. Define route in appropriate `backend/app/api/routes/*.py` file
2. Add `current_user: User = Depends(get_current_active_user)` dependency
3. Use custom exceptions from `app/core/exceptions.py`
4. Add Pydantic schemas in `app/models/schemas.py`
5. Write tests in `tests/test_api.py`
6. Update monitoring configuration if needed

**Adding New Scraper**
1. Create module in `backend/app/scrapers/`
2. Follow async pattern from existing scrapers
3. Add configuration constants to `app/core/constants.py`
4. Integrate in scraping orchestration logic
5. Add error handling and rate limiting
6. Add Celery task for background processing if needed

**Frontend Component Development**
1. Use ShadCN/UI components with Tailwind CSS for styling and Radix UI for primitives
2. Follow TanStack Query patterns for API calls (`useQuery`/`useMutation`)
3. Use TypeScript with proper type definitions and Zod schemas for validation
4. Place components in appropriate subdirectory (`ui/`, `auth/`, `features/`)
5. Ensure responsive design with Tailwind breakpoint utilities
6. Add loading states and error handling with TanStack Query
7. Use secure token storage from `utils/tokenStorage.js`
8. Follow feature-based architecture for complex components
9. Leverage path aliases for clean imports

### Performance Considerations

**Database**
- Use composite indexes for common filter combinations
- Pagination for large datasets (companies listing)
- Connection pooling configured in Supabase client

**Frontend**
- TanStack Query caching (5min stale, 10min cache) reduces API calls
- React virtualization for large data tables with react-window
- Tailwind CSS optimized bundle with utilities
- TypeScript compile-time optimizations
- Dark/light theme with CSS custom properties for performance
- Secure token storage with automatic cleanup
- Feature-based code splitting for better performance

**Scraping**
- Async batch processing with semaphores
- Configurable delays between requests
- Memory management for large CSV imports
- Progress tracking without blocking UI

### Troubleshooting

**Common Issues**
- **CORS errors**: Check `ALLOWED_ORIGINS` in backend config
- **Authentication failures**: Verify JWT secret and user exists in database
- **Scraping timeouts**: Adjust rate limits in `app/core/constants.py`
- **CSV import fails**: Check SIREN format validation and encoding (UTF-8/Latin1)

**Debug Commands**
```bash
# Check backend logs
docker logs ma-intelligence-backend

# Test API directly
curl -H "Authorization: Bearer $TOKEN" http://localhost:8000/api/v1/companies

# Frontend build issues
cd frontend && npm run build

# Database connection test
cd backend && python test_db_connection.py

# Validate specific user stories
cd backend && python validate_us005.py
cd backend && python validate_us007.py

# Check environment setup
cd backend && python -c "import app.core.database; print('Database connection OK')"
```

### Deployment Notes

**Docker Production**
- Multi-stage builds for optimized images
- Nginx handles SSL termination and static files
- Environment variables injected at runtime
- Health checks configured for container orchestration
- Redis service with persistent storage
- PostgreSQL with performance optimizations
- Production profiles activated with `--profile production`

**Database Setup**
- Supabase project with RLS (Row Level Security) disabled for service role
- Tables: `cabinets_comptables`, `activity_logs`
- Indexes on `siren`, `statut`, `score_prospection`, `chiffre_affaires`
- Performance monitoring with slow query analysis
- Database partitioning for large datasets

**SDK Integration**
- Python SDK in `backend/sdks/python/` for programmatic access
- JavaScript SDK for frontend and Node.js integration
- PHP SDK for legacy system integration
- Consistent error handling and authentication across all SDKs

The project follows a modular architecture with clear separation between frontend presentation, backend API, external service integrations, and advanced features like ML/AI, monitoring, and multi-language SDKs. Security and performance are prioritized throughout the stack with enterprise-grade features.