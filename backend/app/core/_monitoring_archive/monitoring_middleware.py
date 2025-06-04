"""
Middleware d'intégration monitoring pour M&A Intelligence Platform
US-003: Middleware FastAPI pour collecte automatique métriques et logs

Features:
- Collecte automatique métriques API (temps réponse, codes statut)
- Logging contextualisé avec request ID
- Intégration avec système d'alerting
- Tracking business events automatique
- Health check intégré
- Rate limiting avec métriques
"""

import time
import uuid
import asyncio
from typing import Callable, Dict, Any
from fastapi import Request, Response, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from app.core.logging_system import (
    get_logger, LogCategory, api_logger, performance_logger, security_logger,
    set_request_context, clear_request_context, generate_request_id
)
from app.core.metrics_collector import get_metrics_collector
from app.core.health_monitor import get_health_monitor
from app.core.alerting_system import get_alerting_system
from app.config import settings

logger = get_logger("monitoring_middleware", LogCategory.SYSTEM)


class MonitoringMiddleware(BaseHTTPMiddleware):
    """
    Middleware de monitoring centralisé
    
    Collecte automatiquement:
    - Métriques API (requests, response time, status codes)
    - Logs contextualisés avec request ID
    - Events business selon endpoint
    - Erreurs et exceptions
    - Performance tracking
    """
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.metrics_collector = get_metrics_collector()
        
        # Configuration
        self.track_business_events = True
        self.log_slow_requests_ms = 2000
        self.log_request_bodies = False  # Pour debugging (attention GDPR)
        
        # Patterns d'endpoints business
        self.business_endpoints = {
            '/api/v1/companies': 'company_access',
            '/api/v1/scraping': 'scraping_operation',
            '/api/v1/stats': 'analytics_access',
            '/api/v1/auth': 'authentication'
        }
        
        # Métriques à exclure (endpoints internes)
        self.excluded_paths = {
            '/health', '/metrics', '/docs', '/redoc', '/openapi.json'
        }
        
        logger.info("🔍 Middleware monitoring initialisé")
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Traitement principal du middleware"""
        
        # Générer request ID unique pour traçage
        request_id = generate_request_id()
        
        # Configuration contexte logging
        user_id = await self._extract_user_id(request)
        set_request_context(
            request_id=request_id,
            user_id=user_id,
            session_id=request.headers.get('x-session-id', '')
        )
        
        # Informations request
        start_time = time.time()
        method = request.method
        path = str(request.url.path)
        client_ip = self._get_client_ip(request)
        user_agent = request.headers.get('user-agent', '')
        
        # Skip monitoring pour certains endpoints
        should_monitor = path not in self.excluded_paths
        
        # Headers response
        response_headers = {
            'X-Request-ID': request_id,
            'X-Monitored': 'true' if should_monitor else 'false'
        }
        
        try:
            # Log début request
            if should_monitor:
                api_logger.info(f"Request started: {method} {path}",
                               endpoint=path, method=method, client_ip=client_ip,
                               user_agent=user_agent[:100])  # Limiter taille
            
            # Exécuter request
            response = await call_next(request)
            
            # Calculer durée
            duration_ms = (time.time() - start_time) * 1000
            status_code = response.status_code
            
            # Ajouter headers
            for key, value in response_headers.items():
                response.headers[key] = value
            
            # Collecte métriques
            if should_monitor:
                await self._collect_request_metrics(
                    method, path, status_code, duration_ms, user_id
                )
                
                # Log completion
                await self._log_request_completion(
                    method, path, status_code, duration_ms, client_ip, user_agent
                )
                
                # Business events
                if self.track_business_events:
                    await self._track_business_event(request, response, duration_ms)
                
                # Alerting sur erreurs
                if status_code >= 500:
                    await self._handle_server_error(method, path, status_code, duration_ms)
            
            return response
            
        except Exception as e:
            # Durée même en cas d'erreur
            duration_ms = (time.time() - start_time) * 1000
            
            # Log erreur
            api_logger.error(f"Request failed: {method} {path}",
                           exception=e, endpoint=path, method=method,
                           duration_ms=duration_ms, client_ip=client_ip)
            
            # Métriques erreur
            if should_monitor:
                self.metrics_collector.increment("api_errors",
                                                labels={
                                                    'method': method,
                                                    'endpoint': self._normalize_endpoint(path),
                                                    'error_type': type(e).__name__
                                                })
                
                # Alerte sur erreurs critiques
                await self._handle_exception_alert(method, path, e)
            
            # Réponse erreur formatée
            if isinstance(e, HTTPException):
                error_response = JSONResponse(
                    status_code=e.status_code,
                    content={"detail": e.detail, "request_id": request_id},
                    headers=response_headers
                )
            else:
                error_response = JSONResponse(
                    status_code=500,
                    content={"detail": "Internal server error", "request_id": request_id},
                    headers=response_headers
                )
            
            return error_response
            
        finally:
            # Nettoyage contexte
            clear_request_context()
    
    async def _extract_user_id(self, request: Request) -> str:
        """Extrait l'ID utilisateur depuis JWT ou session"""
        try:
            # Essayer d'extraire depuis Authorization header
            auth_header = request.headers.get('authorization', '')
            if auth_header.startswith('Bearer '):
                token = auth_header[7:]
                # Ici on pourrait décoder le JWT pour extraire user_id
                # Pour l'instant, retourner un placeholder
                return 'authenticated_user'
            
            return 'anonymous'
            
        except Exception:
            return 'unknown'
    
    def _get_client_ip(self, request: Request) -> str:
        """Récupère l'IP client en tenant compte des proxies"""
        # X-Forwarded-For (load balancer/proxy)
        forwarded_for = request.headers.get('x-forwarded-for')
        if forwarded_for:
            return forwarded_for.split(',')[0].strip()
        
        # X-Real-IP (Nginx)
        real_ip = request.headers.get('x-real-ip')
        if real_ip:
            return real_ip
        
        # IP directe
        return request.client.host if request.client else 'unknown'
    
    async def _collect_request_metrics(self, method: str, path: str, 
                                     status_code: int, duration_ms: float, 
                                     user_id: str):
        """Collecte métriques de la request"""
        
        # Normaliser endpoint (remplacer IDs par patterns)
        normalized_endpoint = self._normalize_endpoint(path)
        
        # Labels pour métriques
        labels = {
            'method': method,
            'endpoint': normalized_endpoint,
            'status': str(status_code)
        }
        
        # Compteur requests
        self.metrics_collector.increment("api_requests", labels=labels)
        
        # Histogramme temps de réponse
        self.metrics_collector.histogram("api_response_time", duration_ms, 
                                        labels={'method': method, 'endpoint': normalized_endpoint})
        
        # Compteur erreurs
        if status_code >= 400:
            error_labels = labels.copy()
            error_labels['error_type'] = self._classify_error(status_code)
            self.metrics_collector.increment("api_errors", labels=error_labels)
        
        # Métrique utilisateurs actifs
        if user_id != 'anonymous':
            self.metrics_collector.set_gauge("active_users", 1, 
                                            labels={'time_window': 'current'})
    
    def _normalize_endpoint(self, path: str) -> str:
        """Normalise les endpoints pour grouper les métriques"""
        # Remplacer les IDs par des patterns
        import re
        
        # Pattern pour IDs numériques
        path = re.sub(r'/\d+', '/{id}', path)
        
        # Pattern pour UUIDs
        path = re.sub(r'/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', 
                     '/{uuid}', path)
        
        # Pattern pour SIRENs (9 chiffres)
        path = re.sub(r'/\d{9}', '/{siren}', path)
        
        return path
    
    def _classify_error(self, status_code: int) -> str:
        """Classifie le type d'erreur selon le code statut"""
        if status_code == 400:
            return 'bad_request'
        elif status_code == 401:
            return 'unauthorized'
        elif status_code == 403:
            return 'forbidden'
        elif status_code == 404:
            return 'not_found'
        elif status_code == 422:
            return 'validation_error'
        elif status_code == 429:
            return 'rate_limited'
        elif 400 <= status_code < 500:
            return 'client_error'
        elif 500 <= status_code < 600:
            return 'server_error'
        else:
            return 'unknown'
    
    async def _log_request_completion(self, method: str, path: str, 
                                    status_code: int, duration_ms: float,
                                    client_ip: str, user_agent: str):
        """Log de completion de request avec contexte"""
        
        # Niveau de log selon performance et statut
        if duration_ms > self.log_slow_requests_ms:
            log_level = 'warning'
            message = f"Slow request: {method} {path} - {duration_ms:.2f}ms"
        elif status_code >= 500:
            log_level = 'error'
            message = f"Server error: {method} {path} - {status_code}"
        elif status_code >= 400:
            log_level = 'warning'
            message = f"Client error: {method} {path} - {status_code}"
        else:
            log_level = 'info'
            message = f"Request completed: {method} {path}"
        
        # Log avec contexte enrichi
        log_data = {
            'endpoint': path,
            'method': method,
            'status_code': status_code,
            'duration_ms': round(duration_ms, 2),
            'client_ip': client_ip,
            'user_agent': user_agent[:100],
            'api_request': True
        }
        
        if log_level == 'error':
            api_logger.error(message, **log_data)
        elif log_level == 'warning':
            api_logger.warning(message, **log_data)
        else:
            api_logger.info(message, **log_data)
        
        # Performance logging séparé
        if duration_ms > 100:  # > 100ms
            performance_logger.performance(
                operation=f"{method} {path}",
                duration_ms=duration_ms,
                success=status_code < 400,
                status_code=status_code
            )
    
    async def _track_business_event(self, request: Request, response: Response, 
                                  duration_ms: float):
        """Track business events selon l'endpoint"""
        path = str(request.url.path)
        method = request.method
        
        # Identifier le type d'événement business
        business_event = None
        entity_type = "api_endpoint"
        entity_id = path
        
        for endpoint_pattern, event_type in self.business_endpoints.items():
            if path.startswith(endpoint_pattern):
                business_event = event_type
                break
        
        if business_event:
            # Métriques business spécifiques
            metrics = {
                'duration_ms': duration_ms,
                'status_code': response.status_code,
                'method': method
            }
            
            # Events spécifiques selon endpoint
            if business_event == 'company_access':
                if method == 'GET':
                    self.metrics_collector.increment("companies_viewed")
                elif method == 'POST':
                    self.metrics_collector.increment("companies_created")
                    
            elif business_event == 'scraping_operation':
                if method == 'POST':
                    self.metrics_collector.increment("scraping_operations",
                                                   labels={'operation': 'start'})
                    
            elif business_event == 'analytics_access':
                self.metrics_collector.increment("analytics_requests")
                
            elif business_event == 'authentication':
                if method == 'POST' and response.status_code == 200:
                    self.metrics_collector.increment("successful_logins")
                elif method == 'POST' and response.status_code == 401:
                    self.metrics_collector.increment("failed_logins")
            
            # Business event log
            from app.core.logging_system import business_logger
            business_logger.business_event(
                event=business_event,
                entity_type=entity_type,
                entity_id=entity_id,
                metrics=metrics
            )
    
    async def _handle_server_error(self, method: str, path: str, 
                                 status_code: int, duration_ms: float):
        """Gère les erreurs serveur pour alerting"""
        
        # Incrémenter compteur erreurs
        self.metrics_collector.increment("server_errors_5xx",
                                        labels={
                                            'method': method,
                                            'endpoint': self._normalize_endpoint(path),
                                            'status_code': str(status_code)
                                        })
        
        # Log sécurité si 500 fréquent (possible attaque)
        security_logger.warning(f"Server error 5xx: {method} {path}",
                              endpoint=path, method=method, status_code=status_code,
                              duration_ms=duration_ms, security_event=True)
    
    async def _handle_exception_alert(self, method: str, path: str, exception: Exception):
        """Déclenche alerte en cas d'exception critique"""
        
        # Types d'exceptions critiques
        critical_exceptions = (
            ConnectionError, TimeoutError, MemoryError, 
            PermissionError, OSError
        )
        
        if isinstance(exception, critical_exceptions):
            # Import local pour éviter dépendance circulaire
            from app.core.alerting_system import send_custom_alert, AlertSeverity, AlertCategory
            
            await send_custom_alert(
                title=f"Critical Exception: {type(exception).__name__}",
                description=f"Exception in {method} {path}: {str(exception)}",
                severity=AlertSeverity.CRITICAL,
                category=AlertCategory.APPLICATION
            )


class HealthCheckMiddleware(BaseHTTPMiddleware):
    """
    Middleware pour health check endpoint automatique
    """
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.health_endpoint = '/health'
        
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Intercepte les health checks"""
        
        if request.url.path == self.health_endpoint:
            try:
                # Import local pour éviter dépendance circulaire
                from app.core.health_monitor import quick_health_check
                
                health_data = await quick_health_check()
                
                return JSONResponse(
                    content=health_data,
                    status_code=200 if health_data.get('status') == 'healthy' else 503
                )
                
            except Exception as e:
                logger.error("Health check failed", exception=e)
                return JSONResponse(
                    content={'status': 'unhealthy', 'error': str(e)},
                    status_code=503
                )
        
        return await call_next(request)


class RateLimitingMiddleware(BaseHTTPMiddleware):
    """
    Middleware de rate limiting avec métriques
    """
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.rate_limits = {
            '/api/v1/auth/login': {'requests': 5, 'window': 300},  # 5 req/5min
            '/api/v1/scraping': {'requests': 10, 'window': 60},   # 10 req/min
            'default': {'requests': 100, 'window': 60}           # 100 req/min
        }
        self.request_counts = {}
        self.metrics_collector = get_metrics_collector()
        
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Applique rate limiting avec métriques"""
        
        client_ip = self._get_client_ip(request)
        path = str(request.url.path)
        
        # Vérifier rate limit
        if await self._is_rate_limited(client_ip, path):
            # Métriques rate limiting
            self.metrics_collector.increment("rate_limit_exceeded",
                                            labels={
                                                'endpoint': path,
                                                'client_ip': client_ip[:10]  # Partial IP pour privacy
                                            })
            
            return JSONResponse(
                status_code=429,
                content={
                    'detail': 'Rate limit exceeded',
                    'retry_after': 60
                },
                headers={'Retry-After': '60'}
            )
        
        return await call_next(request)
    
    def _get_client_ip(self, request: Request) -> str:
        """Récupère IP client"""
        forwarded_for = request.headers.get('x-forwarded-for')
        if forwarded_for:
            return forwarded_for.split(',')[0].strip()
        return request.client.host if request.client else 'unknown'
    
    async def _is_rate_limited(self, client_ip: str, path: str) -> bool:
        """Vérifie si la request dépasse les limites"""
        
        # Trouver limite applicable
        limits = self.rate_limits.get(path, self.rate_limits['default'])
        
        # Clé unique pour tracking
        key = f"{client_ip}:{path}"
        current_time = time.time()
        window_start = current_time - limits['window']
        
        # Nettoyer anciennes requests
        if key in self.request_counts:
            self.request_counts[key] = [
                req_time for req_time in self.request_counts[key]
                if req_time > window_start
            ]
        else:
            self.request_counts[key] = []
        
        # Vérifier limite
        if len(self.request_counts[key]) >= limits['requests']:
            return True
        
        # Ajouter request actuelle
        self.request_counts[key].append(current_time)
        return False


# Factory functions pour faciliter l'intégration

def create_monitoring_middleware(app: ASGIApp) -> MonitoringMiddleware:
    """Crée middleware monitoring avec configuration"""
    return MonitoringMiddleware(app)


def create_health_middleware(app: ASGIApp) -> HealthCheckMiddleware:
    """Crée middleware health check"""
    return HealthCheckMiddleware(app)


def create_rate_limiting_middleware(app: ASGIApp) -> RateLimitingMiddleware:
    """Crée middleware rate limiting"""
    return RateLimitingMiddleware(app)


# Fonction d'intégration complète

def setup_monitoring_middlewares(app):
    """
    Configure tous les middlewares de monitoring sur l'app FastAPI
    
    Usage:
        from app.core.monitoring_middleware import setup_monitoring_middlewares
        setup_monitoring_middlewares(app)
    """
    
    # Ordre important: du plus externe au plus interne
    app.add_middleware(RateLimitingMiddleware)
    app.add_middleware(HealthCheckMiddleware) 
    app.add_middleware(MonitoringMiddleware)
    
    logger.info("🚀 Middlewares de monitoring configurés")