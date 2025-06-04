from typing import List

class ScrapingConfig:
    """Configuration constants for web scraping operations"""
    
    # Target business sectors (Accounting firms)
    CODES_NAF: List[str] = ['6920Z']
    
    # Île-de-France departments
    DEPARTEMENTS_IDF: List[str] = ['75', '77', '78', '91', '92', '93', '94', '95']
    
    # Financial criteria for filtering companies
    CA_MIN: int = 3_000_000  # Minimum revenue: 3M€
    CA_MAX: int = 50_000_000  # Maximum revenue: 50M€
    
    # API rate limiting
    API_DELAY_SECONDS: float = 0.5
    MAX_CONCURRENT_REQUESTS: int = 5
    
    # Batch processing
    BATCH_SIZE: int = 500
    CSV_BATCH_SIZE: int = 50

class DatabaseConfig:
    """Database configuration constants"""
    
    # Connection pool settings
    POOL_SIZE: int = 20
    MAX_OVERFLOW: int = 30
    POOL_TIMEOUT: int = 30
    POOL_RECYCLE: int = 3600
    
    # Query limits
    DEFAULT_PAGE_SIZE: int = 100
    MAX_PAGE_SIZE: int = 1000

class CacheConfig:
    """Cache configuration constants"""
    
    # TTL values in seconds
    STATS_CACHE_TTL: int = 300  # 5 minutes
    COMPANY_CACHE_TTL: int = 3600  # 1 hour
    SEARCH_CACHE_TTL: int = 600  # 10 minutes
    
    # Cache sizes
    STATS_CACHE_SIZE: int = 100
    COMPANY_CACHE_SIZE: int = 1000

# US-002: Redis cache TTL constants
CACHE_TTL = {
    'enrichment_pappers': 86400,  # 24h - legal data is stable
    'enrichment_kaspr': 21600,    # 6h - contact data can change
    'scoring_ma': 3600,           # 1h - scoring results
    'export_csv': 1800,           # 30min - exports
    'api_external': 7200,         # 2h - external API calls
    'default': 3600               # 1h default
}

class SecurityConfig:
    """Security configuration constants"""
    
    # Rate limiting (requests per time window)
    LOGIN_RATE_LIMIT: str = "5/minute"
    API_RATE_LIMIT: str = "100/minute"
    UPLOAD_RATE_LIMIT: str = "10/hour"
    
    # Token settings
    TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 30
    
    # Password requirements
    MIN_PASSWORD_LENGTH: int = 8
    REQUIRE_UPPERCASE: bool = True
    REQUIRE_LOWERCASE: bool = True
    REQUIRE_NUMBERS: bool = True
    REQUIRE_SPECIAL_CHARS: bool = False

class FileConfig:
    """File handling configuration"""
    
    # Upload limits
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS: List[str] = ['.csv']
    
    # Processing limits
    MAX_CSV_ROWS: int = 10000
    ENCODING_OPTIONS: List[str] = ['utf-8', 'latin1', 'cp1252']

class LoggingConfig:
    """Logging configuration constants"""
    
    # Log levels
    DEFAULT_LOG_LEVEL: str = "INFO"
    SQL_LOG_LEVEL: str = "WARNING"
    
    # Log formats
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Audit events
    AUDIT_EVENTS = {
        'USER_LOGIN': 'user_login',
        'USER_LOGOUT': 'user_logout', 
        'COMPANY_VIEW': 'company_view',
        'COMPANY_UPDATE': 'company_update',
        'DATA_EXPORT': 'data_export',
        'SCRAPING_START': 'scraping_start',
        'SCRAPING_COMPLETE': 'scraping_complete'
    }

class APIConfig:
    """API configuration constants"""
    
    # Timeouts (in seconds)
    EXTERNAL_API_TIMEOUT: int = 30
    PAPPERS_API_TIMEOUT: int = 60
    OPENAI_API_TIMEOUT: int = 120
    
    # Retry configuration
    MAX_RETRIES: int = 3
    RETRY_BACKOFF_FACTOR: float = 1.5