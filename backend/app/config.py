from pydantic_settings import BaseSettings
from pydantic import validator
from typing import Optional, List

class Settings(BaseSettings):
    PROJECT_NAME: str = "M&A Intelligence Platform"
    VERSION: str = "2.0.0"
    API_V1_STR: str = "/api/v1"

    # Security
    SECRET_KEY: str
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    # PostgreSQL Database Configuration
    DB_HOST: str = "localhost"
    DB_PORT: int = 5432
    DB_USER: str = "romainsultan"
    DB_PASSWORD: str = "password123"
    DB_NAME: str = "scrapperdb"
    DB_ECHO: bool = False  # Log SQL queries if True
    
    # Legacy database URLs (compatibility)
    DATABASE_URL: Optional[str] = None
    SQLALCHEMY_DATABASE_URL: Optional[str] = None
    
    # User Management
    FIRST_SUPERUSER: str = "admin"
    FIRST_SUPERUSER_PASSWORD: str
    
    # Supabase (backup/migration)
    SUPABASE_URL: Optional[str] = None
    SUPABASE_KEY: Optional[str] = None

    # External APIs
    OPENAI_API_KEY: Optional[str] = None
    PAPPERS_API_KEY: Optional[str] = None
    INFOGREFFE_API_KEY: Optional[str] = None
    KASPR_API_KEY: Optional[str] = None

    # CORS
    ALLOWED_ORIGINS: str = "http://localhost:3000,http://127.0.0.1:3000"

    # Scraping Configuration
    HEADLESS: bool = True
    SCRAPING_CONCURRENT_LIMIT: int = 5
    SCRAPING_DELAY_MIN: float = 1.0
    SCRAPING_DELAY_MAX: float = 3.0

    # Production
    DOMAIN: Optional[str] = None
    SSL_EMAIL: Optional[str] = None
    ENVIRONMENT: str = "development"

    # Redis Cache Configuration (US-002)
    REDIS_URL: str = "redis://localhost:6379"
    REDIS_CACHE_DB: int = 0
    CACHE_TTL_ENRICHMENT: int = 86400  # 24h for Pappers data
    CACHE_TTL_SCORING: int = 3600      # 1h for scoring results
    CACHE_DEFAULT_TTL: int = 7200      # 2h default

    # Monitoring
    SENTRY_DSN: Optional[str] = None
    LOGSTASH_HOST: Optional[str] = None
    LOGSTASH_PORT: Optional[str] = None

    @validator('SECRET_KEY')
    def validate_secret_key(cls, v):
        if len(v) < 32:
            raise ValueError('SECRET_KEY must be at least 32 characters for security')
        return v

    @validator('FIRST_SUPERUSER_PASSWORD')
    def validate_password(cls, v):
        if len(v) < 8:
            raise ValueError('FIRST_SUPERUSER_PASSWORD must be at least 8 characters')
        return v

    @property
    def allowed_origins_list(self) -> List[str]:
        return [origin.strip() for origin in self.ALLOWED_ORIGINS.split(',')]

    class Config:
        env_file = ".env"

settings = Settings()
