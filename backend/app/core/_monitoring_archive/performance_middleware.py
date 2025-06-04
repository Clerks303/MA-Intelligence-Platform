"""
Middleware de performance pour M&A Intelligence Platform
US-005: Compression, rate limiting, et optimisations réponses HTTP

Features:
- Compression Gzip/Brotli automatique
- Rate limiting adaptatif par utilisateur
- Headers de cache optimisés
- Minification JSON
- Static files optimization
- Response streaming pour gros datasets
- Performance monitoring intégré
"""

import asyncio
import gzip
import brotli
import json
import time
import hashlib
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Callable, Set
from dataclasses import dataclass, field
import re
from functools import wraps

from fastapi import Request, Response, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import psutil

from app.config import settings
from app.core.logging_system import get_logger, LogCategory, performance_logger
from app.core.cache_manager import get_cache_manager
from app.core.metrics_collector import get_metrics_collector

logger = get_logger("performance_middleware", LogCategory.PERFORMANCE)


@dataclass
class CompressionStats:
    """Statistiques de compression"""
    original_size: int = 0
    compressed_size: int = 0
    compression_ratio: float = 0.0
    compression_time_ms: float = 0.0
    algorithm: str = ""
    
    @property
    def savings_percent(self) -> float:
        if self.original_size > 0:
            return (1 - self.compressed_size / self.original_size) * 100
        return 0.0


@dataclass
class RateLimitInfo:
    """Informations de rate limiting"""
    key: str
    limit: int
    window_seconds: int
    requests_count: int
    reset_time: datetime
    
    @property
    def remaining(self) -> int:
        return max(0, self.limit - self.requests_count)
    
    @property
    def is_exceeded(self) -> bool:
        return self.requests_count >= self.limit


class CompressionMiddleware(BaseHTTPMiddleware):
    """Middleware de compression automatique des réponses"""
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.min_size = 500  # Compresser si > 500 bytes
        self.compression_stats: List[CompressionStats] = []
        
        # Types MIME à compresser
        self.compressible_types = {
            'application/json',
            'application/javascript',
            'text/html',
            'text/css',
            'text/plain',
            'text/xml',
            'application/xml',
            'application/rss+xml',
            'application/atom+xml',
            'image/svg+xml'
        }
        
        logger.info("🗜️ CompressionMiddleware initialisé")
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Traitement de compression"""
        
        # Exécuter la requête
        response = await call_next(request)
        
        # Vérifier si compression applicable
        if not self._should_compress(request, response):
            return response
        
        # Déterminer algorithme de compression
        compression_algo = self._get_compression_algorithm(request)
        if not compression_algo:
            return response
        
        # Récupérer contenu de la réponse
        content = await self._get_response_content(response)
        if not content or len(content) < self.min_size:
            return response
        
        # Compression
        start_time = time.time()
        compressed_content, compression_type = await self._compress_content(content, compression_algo)
        compression_time_ms = (time.time() - start_time) * 1000
        
        if compressed_content and len(compressed_content) < len(content):
            # Statistiques
            stats = CompressionStats(
                original_size=len(content),
                compressed_size=len(compressed_content),
                compression_ratio=len(compressed_content) / len(content),
                compression_time_ms=compression_time_ms,
                algorithm=compression_type
            )
            self.compression_stats.append(stats)
            
            # Limiter historique stats
            if len(self.compression_stats) > 1000:
                self.compression_stats = self.compression_stats[-1000:]
            
            # Créer nouvelle réponse compressée
            headers = dict(response.headers)
            headers['content-encoding'] = compression_type
            headers['content-length'] = str(len(compressed_content))
            headers['x-compression-ratio'] = f"{stats.compression_ratio:.3f}"
            headers['x-compression-savings'] = f"{stats.savings_percent:.1f}%"
            
            return Response(
                content=compressed_content,
                status_code=response.status_code,
                headers=headers,
                media_type=response.headers.get('content-type')
            )
        
        return response
    
    def _should_compress(self, request: Request, response: Response) -> bool:
        """Détermine si la réponse doit être compressée"""
        
        # Déjà compressé
        if response.headers.get('content-encoding'):
            return False
        
        # Type MIME non compressible
        content_type = response.headers.get('content-type', '').split(';')[0]
        if content_type not in self.compressible_types:
            return False
        
        # Client ne supporte pas la compression
        accept_encoding = request.headers.get('accept-encoding', '')
        if 'gzip' not in accept_encoding and 'br' not in accept_encoding:
            return False
        
        return True
    
    def _get_compression_algorithm(self, request: Request) -> Optional[str]:
        """Détermine l'algorithme de compression optimal"""
        accept_encoding = request.headers.get('accept-encoding', '').lower()
        
        # Préférer Brotli si supporté (meilleur ratio)
        if 'br' in accept_encoding:
            return 'br'
        elif 'gzip' in accept_encoding:
            return 'gzip'
        
        return None
    
    async def _get_response_content(self, response: Response) -> Optional[bytes]:
        """Récupère le contenu de la réponse"""
        try:
            if hasattr(response, 'body'):
                content = response.body
                if isinstance(content, str):
                    return content.encode('utf-8')
                elif isinstance(content, bytes):
                    return content
            return None
        except Exception as e:
            logger.warning(f"Erreur récupération contenu réponse: {e}")
            return None
    
    async def _compress_content(self, content: bytes, algorithm: str) -> tuple[Optional[bytes], str]:
        """Compresse le contenu avec l'algorithme spécifié"""
        try:
            if algorithm == 'br':
                # Brotli compression
                compressed = brotli.compress(content, quality=6)  # Balance qualité/vitesse
                return compressed, 'br'
            elif algorithm == 'gzip':
                # Gzip compression
                compressed = gzip.compress(content, compresslevel=6)
                return compressed, 'gzip'
        except Exception as e:
            logger.warning(f"Erreur compression {algorithm}: {e}")
        
        return None, algorithm
    
    def get_compression_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques de compression"""
        if not self.compression_stats:
            return {'total_compressions': 0}
        
        recent_stats = [
            s for s in self.compression_stats 
            if time.time() - s.compression_time_ms/1000 < 3600  # Dernière heure
        ]
        
        if not recent_stats:
            return {'total_compressions': len(self.compression_stats)}
        
        total_original = sum(s.original_size for s in recent_stats)
        total_compressed = sum(s.compressed_size for s in recent_stats)
        avg_compression_time = sum(s.compression_time_ms for s in recent_stats) / len(recent_stats)
        
        # Stats par algorithme
        algo_stats = {}
        for algo in ['gzip', 'br']:
            algo_compressions = [s for s in recent_stats if s.algorithm == algo]
            if algo_compressions:
                algo_stats[algo] = {
                    'count': len(algo_compressions),
                    'avg_ratio': sum(s.compression_ratio for s in algo_compressions) / len(algo_compressions),
                    'avg_savings_percent': sum(s.savings_percent for s in algo_compressions) / len(algo_compressions)
                }
        
        return {
            'total_compressions': len(self.compression_stats),
            'last_hour': {
                'compressions': len(recent_stats),
                'total_bytes_saved': total_original - total_compressed,
                'avg_compression_ratio': total_compressed / total_original if total_original > 0 else 0,
                'avg_compression_time_ms': avg_compression_time,
                'algorithms': algo_stats
            }
        }


class RateLimitingMiddleware(BaseHTTPMiddleware):
    """Middleware de rate limiting adaptatif"""
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        
        # Configuration rate limits par endpoint
        self.rate_limits = {
            # Auth endpoints - plus restrictifs
            'POST:/api/v1/auth/login': {'requests': 5, 'window': 300, 'burst': 2},
            'POST:/api/v1/auth/register': {'requests': 3, 'window': 3600, 'burst': 1},
            
            # API endpoints - modérés
            'GET:/api/v1/companies': {'requests': 100, 'window': 60, 'burst': 20},
            'POST:/api/v1/companies': {'requests': 20, 'window': 60, 'burst': 5},
            'GET:/api/v1/stats': {'requests': 30, 'window': 60, 'burst': 10},
            
            # Scraping - plus restrictif (opération coûteuse)
            'POST:/api/v1/scraping/start': {'requests': 5, 'window': 300, 'burst': 1},
            
            # Monitoring - modéré
            'GET:/api/v1/monitoring': {'requests': 60, 'window': 60, 'burst': 15},
            
            # Défaut
            'default': {'requests': 100, 'window': 60, 'burst': 20}
        }
        
        # Stockage des compteurs (en production, utiliser Redis)
        self.request_counts: Dict[str, List[float]] = {}
        self.blocked_ips: Dict[str, datetime] = {}
        
        logger.info("🚦 RateLimitingMiddleware initialisé")
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Traitement rate limiting"""
        
        # Identifier client
        client_ip = self._get_client_ip(request)
        
        # Vérifier si IP bloquée
        if self._is_ip_blocked(client_ip):
            logger.warning(f"🚫 IP bloquée: {client_ip}")
            return JSONResponse(
                status_code=429,
                content={
                    'detail': 'IP temporairement bloquée',
                    'retry_after': 3600
                },
                headers={'Retry-After': '3600'}
            )
        
        # Vérifier rate limit
        endpoint_key = f"{request.method}:{request.url.path}"
        rate_limit_info = self._check_rate_limit(client_ip, endpoint_key)
        
        if rate_limit_info.is_exceeded:
            # Enregistrer tentative excessive
            await self._record_rate_limit_violation(client_ip, endpoint_key, request)
            
            # Bloquer IP si trop de violations
            if self._should_block_ip(client_ip):
                self.blocked_ips[client_ip] = datetime.now() + timedelta(hours=1)
                logger.warning(f"🔒 IP bloquée pour 1h: {client_ip}")
            
            return JSONResponse(
                status_code=429,
                content={
                    'detail': 'Rate limit exceeded',
                    'limit': rate_limit_info.limit,
                    'remaining': rate_limit_info.remaining,
                    'reset_time': rate_limit_info.reset_time.isoformat(),
                    'retry_after': rate_limit_info.window_seconds
                },
                headers={
                    'X-RateLimit-Limit': str(rate_limit_info.limit),
                    'X-RateLimit-Remaining': str(rate_limit_info.remaining),
                    'X-RateLimit-Reset': str(int(rate_limit_info.reset_time.timestamp())),
                    'Retry-After': str(rate_limit_info.window_seconds)
                }
            )
        
        # Enregistrer la requête
        self._record_request(client_ip, endpoint_key)
        
        # Exécuter requête avec headers rate limit
        response = await call_next(request)
        
        # Ajouter headers informatifs
        response.headers['X-RateLimit-Limit'] = str(rate_limit_info.limit)
        response.headers['X-RateLimit-Remaining'] = str(rate_limit_info.remaining - 1)
        response.headers['X-RateLimit-Reset'] = str(int(rate_limit_info.reset_time.timestamp()))
        
        return response
    
    def _get_client_ip(self, request: Request) -> str:
        """Récupère l'IP cliente en tenant compte des proxies"""
        # X-Forwarded-For (proxy/load balancer)
        forwarded = request.headers.get('x-forwarded-for')
        if forwarded:
            return forwarded.split(',')[0].strip()
        
        # X-Real-IP (Nginx)
        real_ip = request.headers.get('x-real-ip')
        if real_ip:
            return real_ip
        
        # IP directe
        return request.client.host if request.client else 'unknown'
    
    def _check_rate_limit(self, client_ip: str, endpoint_key: str) -> RateLimitInfo:
        """Vérifie les limites de taux pour un client/endpoint"""
        
        # Récupérer configuration
        config = self.rate_limits.get(endpoint_key, self.rate_limits['default'])
        
        # Clé de tracking
        tracking_key = f"{client_ip}:{endpoint_key}"
        current_time = time.time()
        window_start = current_time - config['window']
        
        # Nettoyer anciennes requêtes
        if tracking_key in self.request_counts:
            self.request_counts[tracking_key] = [
                req_time for req_time in self.request_counts[tracking_key]
                if req_time > window_start
            ]
        else:
            self.request_counts[tracking_key] = []
        
        # Compter requêtes dans la fenêtre
        requests_in_window = len(self.request_counts[tracking_key])
        
        return RateLimitInfo(
            key=tracking_key,
            limit=config['requests'],
            window_seconds=config['window'],
            requests_count=requests_in_window,
            reset_time=datetime.fromtimestamp(current_time + config['window'])
        )
    
    def _record_request(self, client_ip: str, endpoint_key: str):
        """Enregistre une requête"""
        tracking_key = f"{client_ip}:{endpoint_key}"
        current_time = time.time()
        
        if tracking_key not in self.request_counts:
            self.request_counts[tracking_key] = []
        
        self.request_counts[tracking_key].append(current_time)
    
    def _is_ip_blocked(self, client_ip: str) -> bool:
        """Vérifie si une IP est bloquée"""
        if client_ip in self.blocked_ips:
            unblock_time = self.blocked_ips[client_ip]
            if datetime.now() > unblock_time:
                # Débloquer IP
                del self.blocked_ips[client_ip]
                return False
            return True
        return False
    
    def _should_block_ip(self, client_ip: str) -> bool:
        """Détermine si une IP doit être bloquée"""
        # Compter violations récentes (dernières 10 minutes)
        recent_violations = 0
        cutoff_time = time.time() - 600  # 10 minutes
        
        for key, requests in self.request_counts.items():
            if key.startswith(f"{client_ip}:"):
                # Compter requêtes dans la fenêtre qui dépassent les limites
                endpoint = key.split(':', 1)[1]
                config = self.rate_limits.get(endpoint, self.rate_limits['default'])
                
                recent_requests = [r for r in requests if r > cutoff_time]
                if len(recent_requests) > config['requests']:
                    recent_violations += 1
        
        # Bloquer si plus de 3 violations sur différents endpoints
        return recent_violations >= 3
    
    async def _record_rate_limit_violation(self, client_ip: str, endpoint_key: str, request: Request):
        """Enregistre une violation de rate limit"""
        
        # Métriques
        metrics_collector = await get_metrics_collector()
        metrics_collector.increment("rate_limit_violations", labels={
            'endpoint': endpoint_key,
            'client_ip': client_ip[:10]  # IP partielle pour privacy
        })
        
        # Log sécurité
        logger.warning(
            f"🚦 Rate limit dépassé: {client_ip} -> {endpoint_key}",
            extra={
                'client_ip': client_ip,
                'endpoint': endpoint_key,
                'user_agent': request.headers.get('user-agent', '')[:100],
                'referer': request.headers.get('referer', ''),
                'security_event': True
            }
        )


class ResponseOptimizationMiddleware(BaseHTTPMiddleware):
    """Middleware d'optimisation des réponses"""
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.minify_json = True
        self.add_cache_headers = True
        self.stream_threshold = 1024 * 1024  # 1MB
        
        # Cache TTL par type de route
        self.cache_ttl = {
            '/api/v1/stats': 300,        # 5 minutes
            '/api/v1/companies': 600,    # 10 minutes
            '/api/v1/monitoring': 30,    # 30 secondes
            '/health': 60,               # 1 minute
            'default': 0                 # Pas de cache par défaut
        }
        
        logger.info("⚡ ResponseOptimizationMiddleware initialisé")
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Optimisation des réponses"""
        
        start_time = time.time()
        
        # Exécuter requête
        response = await call_next(request)
        
        # Optimiser réponse
        response = await self._optimize_response(request, response)
        
        # Ajouter headers de performance
        processing_time = (time.time() - start_time) * 1000
        response.headers['X-Processing-Time'] = f"{processing_time:.2f}ms"
        response.headers['X-Served-By'] = 'ma-intelligence-api'
        
        # Métriques performance
        await self._record_performance_metrics(request, response, processing_time)
        
        return response
    
    async def _optimize_response(self, request: Request, response: Response) -> Response:
        """Applique les optimisations de réponse"""
        
        # Headers de cache
        if self.add_cache_headers:
            self._add_cache_headers(request, response)
        
        # Optimisation JSON
        if (self.minify_json and 
            response.headers.get('content-type', '').startswith('application/json')):
            response = await self._optimize_json_response(response)
        
        # Headers de sécurité
        self._add_security_headers(response)
        
        return response
    
    def _add_cache_headers(self, request: Request, response: Response):
        """Ajoute les headers de cache appropriés"""
        
        path = request.url.path
        ttl = self.cache_ttl.get(path, self.cache_ttl['default'])
        
        if ttl > 0:
            response.headers['Cache-Control'] = f'public, max-age={ttl}'
            response.headers['Expires'] = (
                datetime.now() + timedelta(seconds=ttl)
            ).strftime('%a, %d %b %Y %H:%M:%S GMT')
        else:
            response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
            response.headers['Pragma'] = 'no-cache'
            response.headers['Expires'] = '0'
        
        # ETag pour validation cache
        if hasattr(response, 'body') and response.body:
            etag = hashlib.md5(str(response.body).encode()).hexdigest()[:16]
            response.headers['ETag'] = f'"{etag}"'
    
    async def _optimize_json_response(self, response: Response) -> Response:
        """Optimise les réponses JSON"""
        try:
            if hasattr(response, 'body') and response.body:
                # Récupérer contenu JSON
                if isinstance(response.body, bytes):
                    json_str = response.body.decode('utf-8')
                else:
                    json_str = str(response.body)
                
                # Parser et re-sérialiser sans espaces (minification)
                json_data = json.loads(json_str)
                minified_json = json.dumps(json_data, separators=(',', ':'), ensure_ascii=False)
                
                # Nouvelle réponse optimisée
                return Response(
                    content=minified_json,
                    status_code=response.status_code,
                    headers=dict(response.headers),
                    media_type='application/json'
                )
        
        except Exception as e:
            logger.warning(f"Erreur optimisation JSON: {e}")
        
        return response
    
    def _add_security_headers(self, response: Response):
        """Ajoute les headers de sécurité"""
        security_headers = {
            'X-Content-Type-Options': 'nosniff',
            'X-Frame-Options': 'DENY',
            'X-XSS-Protection': '1; mode=block',
            'Referrer-Policy': 'strict-origin-when-cross-origin',
            'Permissions-Policy': 'geolocation=(), microphone=(), camera=()'
        }
        
        for header, value in security_headers.items():
            if header not in response.headers:
                response.headers[header] = value
    
    async def _record_performance_metrics(self, request: Request, response: Response, processing_time_ms: float):
        """Enregistre les métriques de performance"""
        try:
            metrics_collector = await get_metrics_collector()
            
            # Métrique temps de traitement
            metrics_collector.histogram("api_processing_time", processing_time_ms, labels={
                'method': request.method,
                'endpoint': request.url.path,
                'status_code': str(response.status_code)
            })
            
            # Métrique taille réponse
            content_length = response.headers.get('content-length')
            if content_length:
                metrics_collector.histogram("api_response_size", int(content_length), labels={
                    'endpoint': request.url.path
                })
            
            # Alertes performance
            if processing_time_ms > 5000:  # > 5 secondes
                performance_logger.performance(
                    operation=f"{request.method} {request.url.path}",
                    duration_ms=processing_time_ms,
                    success=response.status_code < 400,
                    details={
                        'status_code': response.status_code,
                        'slow_request': True
                    }
                )
        
        except Exception as e:
            logger.error(f"Erreur enregistrement métriques performance: {e}")


# Factory functions

def setup_performance_middlewares(app):
    """Configure tous les middlewares de performance"""
    
    # Ordre important: du plus externe au plus interne
    app.add_middleware(ResponseOptimizationMiddleware)
    app.add_middleware(RateLimitingMiddleware)
    app.add_middleware(CompressionMiddleware)
    
    logger.info("🚀 Middlewares de performance configurés")


# Utilitaires

async def stream_large_response(data: List[Dict], chunk_size: int = 1000) -> StreamingResponse:
    """Stream une grande réponse par chunks"""
    
    async def generate_chunks():
        yield '{"data": ['
        
        for i, item in enumerate(data):
            if i > 0:
                yield ','
            yield json.dumps(item, separators=(',', ':'))
            
            # Yield contrôle periodiquement
            if i % chunk_size == 0:
                await asyncio.sleep(0.001)  # Permet autres tâches
        
        yield '], "total": ' + str(len(data)) + '}'
    
    return StreamingResponse(
        generate_chunks(),
        media_type='application/json',
        headers={
            'X-Streaming': 'true',
            'X-Total-Items': str(len(data))
        }
    )


def performance_monitor(operation_name: str = None):
    """Décorateur pour monitorer la performance d'une fonction"""
    
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = await func(*args, **kwargs)
                success = True
            except Exception as e:
                success = False
                raise
            finally:
                duration_ms = (time.time() - start_time) * 1000
                
                performance_logger.performance(
                    operation=operation_name or func.__name__,
                    duration_ms=duration_ms,
                    success=success
                )
        
        return wrapper
    return decorator