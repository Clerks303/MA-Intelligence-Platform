"""
Système de logging structuré et audit trail pour M&A Intelligence Platform
US-003: Logging centralisé avec contexte business et sécurité

Features:
- Logging structuré JSON avec contexte métier
- Audit trail pour actions sensibles (authentification, modifications données)
- Niveaux de log configurables par module
- Rotation automatique et rétention configurable
- Intégration avec systèmes externes (ELK, Sentry)
- Corrélation des logs avec request ID unique
"""

import json
import logging
import logging.handlers
import traceback
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from contextvars import ContextVar
from dataclasses import dataclass, asdict
from enum import Enum
import asyncio
from pathlib import Path

from app.config import settings

# Context variables pour tracking des requests
request_id_var: ContextVar[str] = ContextVar('request_id', default='')
user_id_var: ContextVar[str] = ContextVar('user_id', default='')
session_id_var: ContextVar[str] = ContextVar('session_id', default='')


class LogLevel(str, Enum):
    """Niveaux de log étendus"""
    TRACE = "TRACE"
    DEBUG = "DEBUG" 
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"
    AUDIT = "AUDIT"  # Niveau spécial pour audit trail


class LogCategory(str, Enum):
    """Catégories de logs pour classification"""
    SYSTEM = "system"
    BUSINESS = "business"
    SECURITY = "security"
    PERFORMANCE = "performance"
    API = "api"
    DATABASE = "database"
    CACHE = "cache"
    SCRAPING = "scraping"
    AUTHENTICATION = "authentication"
    AUDIT = "audit"
    AI_ML = "ai_ml"


@dataclass
class LogContext:
    """Contexte enrichi pour les logs"""
    request_id: str = ""
    user_id: str = ""
    session_id: str = ""
    ip_address: str = ""
    user_agent: str = ""
    endpoint: str = ""
    method: str = ""
    duration_ms: Optional[float] = None
    business_context: Dict[str, Any] = None

    def __post_init__(self):
        if self.business_context is None:
            self.business_context = {}


@dataclass  
class AuditLogEntry:
    """Entrée d'audit trail structurée"""
    action: str
    resource_type: str
    resource_id: str
    user_id: str
    timestamp: datetime
    details: Dict[str, Any]
    ip_address: str = ""
    user_agent: str = ""
    success: bool = True
    risk_score: int = 0  # 0=low, 1=medium, 2=high, 3=critical


class StructuredLogger:
    """
    Logger structuré avec contexte métier et audit trail
    
    Fonctionnalités:
    - Formatage JSON structuré
    - Enrichissement automatique du contexte
    - Corrélation des logs via request ID
    - Audit trail pour actions sensibles
    - Performance tracking
    """
    
    def __init__(self, name: str, category: LogCategory = LogCategory.SYSTEM):
        self.name = name
        self.category = category
        self.logger = logging.getLogger(name)
        self._setup_handlers()
        
        # Métriques logging
        self.log_counts = {level.value: 0 for level in LogLevel}
        self.error_patterns = {}
        
    def _setup_handlers(self):
        """Configuration des handlers de logging"""
        
        if self.logger.handlers:
            return  # Déjà configuré
            
        # Handler console avec formatage JSON
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(StructuredFormatter())
        console_handler.setLevel(getattr(logging, settings.ENVIRONMENT == "development" and "DEBUG" or "INFO"))
        
        # Handler fichier avec rotation
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_dir / f"{self.name}.log",
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setFormatter(StructuredFormatter())
        file_handler.setLevel(logging.DEBUG)
        
        # Handler erreurs séparé
        error_handler = logging.handlers.RotatingFileHandler(
            log_dir / f"{self.name}_errors.log", 
            maxBytes=5*1024*1024,  # 5MB
            backupCount=10,
            encoding='utf-8'
        )
        error_handler.setFormatter(StructuredFormatter())
        error_handler.setLevel(logging.ERROR)
        
        # Handler audit trail
        if self.category in [LogCategory.SECURITY, LogCategory.AUDIT, LogCategory.AUTHENTICATION]:
            audit_handler = logging.handlers.RotatingFileHandler(
                log_dir / "audit_trail.log",
                maxBytes=50*1024*1024,  # 50MB - audit logs are critical
                backupCount=50,
                encoding='utf-8'
            )
            audit_handler.setFormatter(AuditFormatter())
            audit_handler.setLevel(logging.INFO)
            self.logger.addHandler(audit_handler)
        
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler) 
        self.logger.addHandler(error_handler)
        self.logger.setLevel(logging.DEBUG)
    
    def _enrich_context(self, extra: Dict[str, Any] = None) -> Dict[str, Any]:
        """Enrichissement automatique du contexte"""
        context = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'logger_name': self.name,
            'category': self.category.value,
            'request_id': request_id_var.get(''),
            'user_id': user_id_var.get(''),
            'session_id': session_id_var.get(''),
            'environment': settings.ENVIRONMENT
        }
        
        if extra:
            context.update(extra)
            
        return context
    
    def trace(self, message: str, **kwargs):
        """Log niveau TRACE (debugging très détaillé)"""
        self._log(LogLevel.TRACE, message, **kwargs)
    
    def debug(self, message: str, **kwargs):
        """Log niveau DEBUG"""
        self._log(LogLevel.DEBUG, message, **kwargs)
        
    def info(self, message: str, **kwargs):
        """Log niveau INFO"""
        self._log(LogLevel.INFO, message, **kwargs)
        
    def warning(self, message: str, **kwargs):
        """Log niveau WARNING"""
        self._log(LogLevel.WARNING, message, **kwargs)
        
    def error(self, message: str, exception: Exception = None, **kwargs):
        """Log niveau ERROR avec stack trace optionnelle"""
        if exception:
            kwargs['exception'] = {
                'type': type(exception).__name__,
                'message': str(exception),
                'traceback': traceback.format_exc()
            }
        self._log(LogLevel.ERROR, message, **kwargs)
        
    def critical(self, message: str, exception: Exception = None, **kwargs):
        """Log niveau CRITICAL"""
        if exception:
            kwargs['exception'] = {
                'type': type(exception).__name__,
                'message': str(exception),
                'traceback': traceback.format_exc()
            }
        self._log(LogLevel.CRITICAL, message, **kwargs)
    
    def audit(self, action: str, resource_type: str, resource_id: str, 
             success: bool = True, details: Dict[str, Any] = None, **kwargs):
        """Log d'audit trail pour actions sensibles"""
        
        audit_entry = AuditLogEntry(
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            user_id=user_id_var.get('anonymous'),
            timestamp=datetime.now(timezone.utc),
            details=details or {},
            success=success,
            **kwargs
        )
        
        log_data = asdict(audit_entry)
        log_data['audit_trail'] = True
        
        self._log(LogLevel.AUDIT, f"AUDIT: {action} on {resource_type}:{resource_id}", 
                 extra=log_data)
    
    def performance(self, operation: str, duration_ms: float, 
                   success: bool = True, **kwargs):
        """Log de performance pour tracking des opérations"""
        perf_data = {
            'operation': operation,
            'duration_ms': round(duration_ms, 2),
            'success': success,
            'performance_log': True
        }
        perf_data.update(kwargs)
        
        level = LogLevel.INFO if success else LogLevel.WARNING
        self._log(level, f"PERF: {operation} completed in {duration_ms:.2f}ms", 
                 extra=perf_data)
    
    def business_event(self, event: str, entity_type: str, entity_id: str, 
                      metrics: Dict[str, Any] = None, **kwargs):
        """Log d'événement business pour analytics"""
        business_data = {
            'business_event': event,
            'entity_type': entity_type, 
            'entity_id': entity_id,
            'metrics': metrics or {},
            'business_log': True
        }
        business_data.update(kwargs)
        
        self._log(LogLevel.INFO, f"BUSINESS: {event} for {entity_type}:{entity_id}",
                 extra=business_data)
    
    def _log(self, level: LogLevel, message: str, extra: Dict[str, Any] = None):
        """Méthode interne de logging avec enrichissement"""
        
        # Statistiques
        self.log_counts[level.value] += 1
        
        # Contexte enrichi
        enriched_extra = self._enrich_context(extra)
        
        # Mapping vers niveau Python logging
        python_level = getattr(logging, level.value.replace('TRACE', 'DEBUG').replace('AUDIT', 'INFO'))
        
        # Log avec contexte enrichi
        self.logger.log(python_level, message, extra=enriched_extra)
        
        # Détection patterns d'erreur
        if level in [LogLevel.ERROR, LogLevel.CRITICAL]:
            self._track_error_pattern(message, enriched_extra)
    
    def _track_error_pattern(self, message: str, context: Dict[str, Any]):
        """Tracking des patterns d'erreur pour alerting"""
        error_key = f"{context.get('endpoint', 'unknown')}:{type(context.get('exception', {}).get('type', 'unknown'))}"
        
        if error_key not in self.error_patterns:
            self.error_patterns[error_key] = {'count': 0, 'last_seen': None}
            
        self.error_patterns[error_key]['count'] += 1
        self.error_patterns[error_key]['last_seen'] = datetime.now()
    
    def get_metrics(self) -> Dict[str, Any]:
        """Métriques du logger pour monitoring"""
        return {
            'log_counts': self.log_counts.copy(),
            'error_patterns': dict(self.error_patterns),
            'total_logs': sum(self.log_counts.values())
        }


class StructuredFormatter(logging.Formatter):
    """Formatter JSON structuré pour logs applicatifs"""
    
    def format(self, record):
        log_data = {
            'timestamp': datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            'level': record.levelname,
            'message': record.getMessage(),
            'logger': record.name,
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Ajout du contexte extra
        if hasattr(record, '__dict__'):
            for key, value in record.__dict__.items():
                if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 
                              'filename', 'module', 'lineno', 'funcName', 'created', 
                              'msecs', 'relativeCreated', 'thread', 'threadName', 
                              'processName', 'process', 'message']:
                    log_data[key] = value
        
        return json.dumps(log_data, ensure_ascii=False, default=str)


class AuditFormatter(logging.Formatter):
    """Formatter spécialisé pour audit trail"""
    
    def format(self, record):
        if hasattr(record, 'audit_trail') and record.audit_trail:
            # Format spécial pour audit
            audit_data = {
                'audit_timestamp': datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
                'audit_level': 'AUDIT',
                'audit_message': record.getMessage()
            }
            
            # Copie des données d'audit
            for key, value in record.__dict__.items():
                if key.startswith(('action', 'resource', 'user', 'details', 'success', 'risk')):
                    audit_data[key] = value
                    
            return json.dumps(audit_data, ensure_ascii=False, default=str)
        else:
            return super().format(record)


# Factory et utilitaires

_loggers: Dict[str, StructuredLogger] = {}


def get_logger(name: str, category: LogCategory = LogCategory.SYSTEM) -> StructuredLogger:
    """Factory pour obtenir un logger structuré"""
    if name not in _loggers:
        _loggers[name] = StructuredLogger(name, category)
    return _loggers[name]


def set_request_context(request_id: str = None, user_id: str = None, 
                       session_id: str = None):
    """Définit le contexte pour la request courante"""
    if request_id:
        request_id_var.set(request_id)
    if user_id:
        user_id_var.set(user_id)
    if session_id:
        session_id_var.set(session_id)


def clear_request_context():
    """Nettoie le contexte de la request"""
    request_id_var.set('')
    user_id_var.set('')
    session_id_var.set('')


def generate_request_id() -> str:
    """Génère un ID unique pour traçage des requests"""
    return str(uuid.uuid4())


# Loggers pré-configurés pour modules principaux

api_logger = get_logger("api", LogCategory.API)
security_logger = get_logger("security", LogCategory.SECURITY)
business_logger = get_logger("business", LogCategory.BUSINESS)
performance_logger = get_logger("performance", LogCategory.PERFORMANCE)
audit_logger = get_logger("audit", LogCategory.AUDIT)
cache_logger = get_logger("cache", LogCategory.CACHE)
scraping_logger = get_logger("scraping", LogCategory.SCRAPING)
logger = get_logger("advanced_ai_engine", LogCategory.PERFORMANCE)



# Décorateurs utilitaires

def log_performance(operation_name: str = None):
    """Décorateur pour logging automatique des performances"""
    def decorator(func):
        import time
        import asyncio
        from functools import wraps
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            operation = operation_name or f"{func.__module__}.{func.__name__}"
            
            try:
                result = await func(*args, **kwargs)
                duration_ms = (time.time() - start_time) * 1000
                performance_logger.performance(operation, duration_ms, success=True)
                return result
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                performance_logger.performance(operation, duration_ms, success=False, 
                                             error=str(e))
                raise
                
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            operation = operation_name or f"{func.__module__}.{func.__name__}"
            
            try:
                result = func(*args, **kwargs)
                duration_ms = (time.time() - start_time) * 1000
                performance_logger.performance(operation, duration_ms, success=True)
                return result
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                performance_logger.performance(operation, duration_ms, success=False,
                                             error=str(e))
                raise
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator


def log_audit(action: str, resource_type: str):
    """Décorateur pour audit trail automatique"""
    def decorator(func):
        from functools import wraps
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                result = await func(*args, **kwargs)
                # Essayer d'extraire resource_id du résultat ou des args
                resource_id = getattr(result, 'id', str(args[1]) if len(args) > 1 else 'unknown')
                audit_logger.audit(action, resource_type, resource_id, success=True)
                return result
            except Exception as e:
                resource_id = str(args[1]) if len(args) > 1 else 'unknown'
                audit_logger.audit(action, resource_type, resource_id, success=False,
                                  details={'error': str(e)})
                raise
                
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                result = func(*args, **kwargs)
                resource_id = getattr(result, 'id', str(args[1]) if len(args) > 1 else 'unknown')
                audit_logger.audit(action, resource_type, resource_id, success=True)
                return result
            except Exception as e:
                resource_id = str(args[1]) if len(args) > 1 else 'unknown'
                audit_logger.audit(action, resource_type, resource_id, success=False,
                                  details={'error': str(e)})
                raise
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator