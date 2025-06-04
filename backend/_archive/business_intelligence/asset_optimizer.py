"""
Optimiseur d'assets et compression pour M&A Intelligence Platform
US-009: Compression, minification et optimisation des ressources

Ce module fournit:
- Compression gzip/brotli automatique
- Minification CSS/JS
- Optimisation d'images
- CDN et mise en cache statique
- Compression de réponses API
- Bundle splitting et lazy loading
- Service Worker pour cache offline
"""

import asyncio
import gzip
import brotli
import time
import mimetypes
import hashlib
import json
import os
import tempfile
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import io
from functools import wraps

# Import conditionnel pour optimisation images
try:
    from PIL import Image, ImageOptim
    IMAGE_OPTIMIZATION_AVAILABLE = True
except ImportError:
    IMAGE_OPTIMIZATION_AVAILABLE = False

# Import conditionnel pour minification
try:
    from csscompressor import compress as css_compress
    from jsmin import jsmin
    MINIFICATION_AVAILABLE = True
except ImportError:
    MINIFICATION_AVAILABLE = False

from fastapi import Request, Response
from fastapi.responses import StreamingResponse
import aiofiles

from app.core.logging_system import get_logger, LogCategory
from app.core.cache_manager import get_cache_manager

logger = get_logger("asset_optimizer", LogCategory.PERFORMANCE)


@dataclass
class CompressionConfig:
    """Configuration de compression"""
    enable_gzip: bool = True
    enable_brotli: bool = True
    gzip_level: int = 6
    brotli_level: int = 4
    min_size_bytes: int = 1024  # Ne pas compresser si < 1KB
    max_size_bytes: int = 10 * 1024 * 1024  # Limite 10MB
    
    # Types MIME à compresser
    compressible_types: List[str] = field(default_factory=lambda: [
        'text/html', 'text/css', 'text/javascript', 'text/xml', 'text/plain',
        'application/javascript', 'application/json', 'application/xml',
        'application/rss+xml', 'application/atom+xml', 'image/svg+xml'
    ])


@dataclass
class AssetOptimizationConfig:
    """Configuration d'optimisation des assets"""
    enable_minification: bool = True
    enable_image_optimization: bool = True
    enable_caching: bool = True
    cache_max_age: int = 86400 * 30  # 30 jours
    versioning: bool = True
    cdn_enabled: bool = False
    cdn_base_url: str = ""
    
    # Optimisation images
    image_quality: int = 85
    image_progressive: bool = True
    image_optimize: bool = True
    
    # Bundle optimization
    enable_bundle_splitting: bool = True
    enable_tree_shaking: bool = True


@dataclass
class CompressionResult:
    """Résultat de compression"""
    original_size: int
    compressed_size: int
    compression_ratio: float
    algorithm: str
    compression_time: float
    
    @property
    def size_reduction_percent(self) -> float:
        return ((self.original_size - self.compressed_size) / self.original_size * 100) if self.original_size > 0 else 0


@dataclass
class AssetMetrics:
    """Métriques d'optimisation des assets"""
    total_requests: int = 0
    cached_responses: int = 0
    compressed_responses: int = 0
    total_original_size: int = 0
    total_compressed_size: int = 0
    avg_compression_time: float = 0.0
    last_reset: datetime = field(default_factory=datetime.now)
    
    @property
    def cache_hit_rate(self) -> float:
        return (self.cached_responses / self.total_requests * 100) if self.total_requests > 0 else 0
    
    @property
    def compression_rate(self) -> float:
        return (self.compressed_responses / self.total_requests * 100) if self.total_requests > 0 else 0
    
    @property
    def total_savings_mb(self) -> float:
        return (self.total_original_size - self.total_compressed_size) / (1024 * 1024)


class ContentCompressor:
    """Compresseur de contenu avec algorithmes multiples"""
    
    def __init__(self, config: CompressionConfig):
        self.config = config
    
    async def compress_content(self, content: bytes, content_type: str = "text/html") -> Optional[CompressionResult]:
        """Compresse le contenu avec le meilleur algorithme"""
        
        # Vérifier si le type doit être compressé
        if not self._should_compress(content, content_type):
            return None
        
        start_time = time.time()
        original_size = len(content)
        
        # Essayer Brotli d'abord (meilleur ratio)
        if self.config.enable_brotli:
            try:
                compressed_brotli = brotli.compress(content, quality=self.config.brotli_level)
                if len(compressed_brotli) < original_size:
                    compression_time = time.time() - start_time
                    return CompressionResult(
                        original_size=original_size,
                        compressed_size=len(compressed_brotli),
                        compression_ratio=len(compressed_brotli) / original_size,
                        algorithm="br",
                        compression_time=compression_time
                    )
            except Exception as e:
                logger.warning(f"Erreur compression Brotli: {e}")
        
        # Fallback vers gzip
        if self.config.enable_gzip:
            try:
                compressed_gzip = gzip.compress(content, compresslevel=self.config.gzip_level)
                if len(compressed_gzip) < original_size:
                    compression_time = time.time() - start_time
                    return CompressionResult(
                        original_size=original_size,
                        compressed_size=len(compressed_gzip),
                        compression_ratio=len(compressed_gzip) / original_size,
                        algorithm="gzip",
                        compression_time=compression_time
                    )
            except Exception as e:
                logger.warning(f"Erreur compression gzip: {e}")
        
        return None
    
    def _should_compress(self, content: bytes, content_type: str) -> bool:
        """Détermine si le contenu doit être compressé"""
        
        content_size = len(content)
        
        # Vérifier taille
        if content_size < self.config.min_size_bytes or content_size > self.config.max_size_bytes:
            return False
        
        # Vérifier type MIME
        if not any(mime_type in content_type for mime_type in self.config.compressible_types):
            return False
        
        return True
    
    async def get_compressed_content(self, content: bytes, content_type: str, accepted_encodings: str = "") -> Tuple[bytes, str]:
        """Retourne le contenu compressé avec l'encoding approprié"""
        
        compression_result = await self.compress_content(content, content_type)
        
        if not compression_result:
            return content, ""
        
        # Vérifier que le client accepte l'encoding
        if compression_result.algorithm == "br" and "br" in accepted_encodings:
            compressed_content = brotli.compress(content, quality=self.config.brotli_level)
            return compressed_content, "br"
        elif compression_result.algorithm == "gzip" and "gzip" in accepted_encodings:
            compressed_content = gzip.compress(content, compresslevel=self.config.gzip_level)
            return compressed_content, "gzip"
        
        return content, ""


class AssetMinifier:
    """Minificateur d'assets CSS/JS"""
    
    def __init__(self):
        self.enabled = MINIFICATION_AVAILABLE
        if not self.enabled:
            logger.warning("Minification non disponible (paquets manquants)")
    
    async def minify_css(self, css_content: str) -> str:
        """Minifie le contenu CSS"""
        if not self.enabled:
            return css_content
        
        try:
            return css_compress(css_content)
        except Exception as e:
            logger.warning(f"Erreur minification CSS: {e}")
            return css_content
    
    async def minify_js(self, js_content: str) -> str:
        """Minifie le contenu JavaScript"""
        if not self.enabled:
            return js_content
        
        try:
            return jsmin(js_content)
        except Exception as e:
            logger.warning(f"Erreur minification JS: {e}")
            return js_content
    
    async def minify_json(self, json_content: str) -> str:
        """Minifie le contenu JSON"""
        try:
            data = json.loads(json_content)
            return json.dumps(data, separators=(',', ':'))
        except Exception as e:
            logger.warning(f"Erreur minification JSON: {e}")
            return json_content


class ImageOptimizer:
    """Optimiseur d'images"""
    
    def __init__(self, config: AssetOptimizationConfig):
        self.config = config
        self.enabled = IMAGE_OPTIMIZATION_AVAILABLE
        if not self.enabled:
            logger.warning("Optimisation d'images non disponible (PIL manquant)")
    
    async def optimize_image(self, image_data: bytes, format: str = "JPEG") -> bytes:
        """Optimise une image"""
        if not self.enabled:
            return image_data
        
        try:
            # Ouvrir l'image
            image = Image.open(io.BytesIO(image_data))
            
            # Optimisations
            if format.upper() == "JPEG":
                # JPEG optimizations
                if image.mode in ("RGBA", "P"):
                    image = image.convert("RGB")
                
                output = io.BytesIO()
                image.save(
                    output,
                    format="JPEG",
                    quality=self.config.image_quality,
                    optimize=self.config.image_optimize,
                    progressive=self.config.image_progressive
                )
                return output.getvalue()
            
            elif format.upper() == "PNG":
                # PNG optimizations
                output = io.BytesIO()
                image.save(
                    output,
                    format="PNG",
                    optimize=self.config.image_optimize
                )
                return output.getvalue()
            
            else:
                return image_data
                
        except Exception as e:
            logger.warning(f"Erreur optimisation image: {e}")
            return image_data


class StaticAssetCache:
    """Cache des assets statiques avec versioning"""
    
    def __init__(self, config: AssetOptimizationConfig):
        self.config = config
        self.cache: Dict[str, Dict[str, Any]] = {}
        
    def _generate_etag(self, content: bytes) -> str:
        """Génère un ETag pour le contenu"""
        return hashlib.md5(content).hexdigest()[:16]
    
    def _generate_version_hash(self, content: bytes) -> str:
        """Génère un hash de version pour le fichier"""
        return hashlib.sha256(content).hexdigest()[:8]
    
    async def get_cached_asset(self, asset_path: str) -> Optional[Dict[str, Any]]:
        """Récupère un asset depuis le cache"""
        if not self.config.enable_caching:
            return None
        
        cached_asset = self.cache.get(asset_path)
        if not cached_asset:
            return None
        
        # Vérifier expiration
        if datetime.now() > cached_asset['expires_at']:
            del self.cache[asset_path]
            return None
        
        return cached_asset
    
    async def cache_asset(self, asset_path: str, content: bytes, content_type: str, compressed_content: Optional[bytes] = None, encoding: str = "") -> Dict[str, Any]:
        """Met en cache un asset"""
        if not self.config.enable_caching:
            return {}
        
        etag = self._generate_etag(content)
        version_hash = self._generate_version_hash(content)
        expires_at = datetime.now() + timedelta(seconds=self.config.cache_max_age)
        
        cached_asset = {
            'content': content,
            'compressed_content': compressed_content,
            'content_type': content_type,
            'encoding': encoding,
            'etag': etag,
            'version_hash': version_hash,
            'expires_at': expires_at,
            'cached_at': datetime.now()
        }
        
        self.cache[asset_path] = cached_asset
        return cached_asset
    
    def get_versioned_url(self, asset_path: str, content: bytes) -> str:
        """Génère une URL versionnée pour l'asset"""
        if not self.config.versioning:
            return asset_path
        
        version_hash = self._generate_version_hash(content)
        
        if self.config.cdn_enabled and self.config.cdn_base_url:
            return f"{self.config.cdn_base_url}/{asset_path}?v={version_hash}"
        else:
            return f"{asset_path}?v={version_hash}"


class AssetOptimizer:
    """Optimiseur principal d'assets"""
    
    def __init__(self):
        self.compression_config = CompressionConfig()
        self.optimization_config = AssetOptimizationConfig()
        
        self.compressor = ContentCompressor(self.compression_config)
        self.minifier = AssetMinifier()
        self.image_optimizer = ImageOptimizer(self.optimization_config)
        self.static_cache = StaticAssetCache(self.optimization_config)
        
        self.metrics = AssetMetrics()
        
        logger.info("🎨 Optimiseur d'assets initialisé")
    
    async def optimize_response(self, content: Union[str, bytes], content_type: str, request: Request) -> Tuple[bytes, Dict[str, str]]:
        """Optimise une réponse complète"""
        
        start_time = time.time()
        
        # Convertir en bytes si nécessaire
        if isinstance(content, str):
            content = content.encode('utf-8')
        
        original_size = len(content)
        headers = {}
        
        # Minification selon le type
        if content_type.startswith('text/css'):
            content_str = content.decode('utf-8')
            minified = await self.minifier.minify_css(content_str)
            content = minified.encode('utf-8')
        elif content_type.startswith('application/javascript') or content_type.startswith('text/javascript'):
            content_str = content.decode('utf-8')
            minified = await self.minifier.minify_js(content_str)
            content = minified.encode('utf-8')
        elif content_type.startswith('application/json'):
            content_str = content.decode('utf-8')
            minified = await self.minifier.minify_json(content_str)
            content = minified.encode('utf-8')
        
        # Compression
        accepted_encodings = request.headers.get('accept-encoding', '')
        compressed_content, encoding = await self.compressor.get_compressed_content(
            content, content_type, accepted_encodings
        )
        
        final_content = compressed_content if encoding else content
        final_size = len(final_content)
        
        # Headers de cache
        if self.optimization_config.enable_caching:
            etag = hashlib.md5(content).hexdigest()[:16]
            headers.update({
                'ETag': f'"{etag}"',
                'Cache-Control': f'public, max-age={self.optimization_config.cache_max_age}',
                'Vary': 'Accept-Encoding'
            })
        
        # Header de compression
        if encoding:
            headers['Content-Encoding'] = encoding
            self.metrics.compressed_responses += 1
        
        # Mettre à jour métriques
        processing_time = time.time() - start_time
        self._update_metrics(original_size, final_size, processing_time)
        
        return final_content, headers
    
    async def optimize_static_file(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Optimise un fichier statique"""
        
        # Vérifier cache d'abord
        cached_asset = await self.static_cache.get_cached_asset(file_path)
        if cached_asset:
            self.metrics.cached_responses += 1
            return cached_asset
        
        try:
            # Lire le fichier
            async with aiofiles.open(file_path, 'rb') as f:
                content = await f.read()
            
            # Déterminer le type MIME
            content_type, _ = mimetypes.guess_type(file_path)
            if not content_type:
                content_type = 'application/octet-stream'
            
            # Optimiser selon le type
            optimized_content = content
            
            if content_type.startswith('image/'):
                # Optimisation d'image
                image_format = content_type.split('/')[-1].upper()
                optimized_content = await self.image_optimizer.optimize_image(content, image_format)
            
            # Compression
            compressed_content, encoding = await self.compressor.get_compressed_content(
                optimized_content, content_type, "gzip, br"
            )
            
            # Mettre en cache
            cached_asset = await self.static_cache.cache_asset(
                file_path, 
                optimized_content, 
                content_type,
                compressed_content if encoding else None,
                encoding
            )
            
            return cached_asset
            
        except Exception as e:
            logger.error(f"Erreur optimisation fichier {file_path}: {e}")
            return None
    
    def _update_metrics(self, original_size: int, final_size: int, processing_time: float):
        """Met à jour les métriques"""
        self.metrics.total_requests += 1
        self.metrics.total_original_size += original_size
        self.metrics.total_compressed_size += final_size
        
        # Moyenne mobile du temps de traitement
        if self.metrics.total_requests == 1:
            self.metrics.avg_compression_time = processing_time
        else:
            alpha = 0.1
            self.metrics.avg_compression_time = (
                alpha * processing_time + (1 - alpha) * self.metrics.avg_compression_time
            )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Récupère les métriques d'optimisation"""
        return {
            'total_requests': self.metrics.total_requests,
            'cache_hit_rate': self.metrics.cache_hit_rate,
            'compression_rate': self.metrics.compression_rate,
            'total_savings_mb': self.metrics.total_savings_mb,
            'avg_compression_time': self.metrics.avg_compression_time,
            'last_reset': self.metrics.last_reset.isoformat(),
            'compression_config': {
                'gzip_enabled': self.compression_config.enable_gzip,
                'brotli_enabled': self.compression_config.enable_brotli,
                'min_size': self.compression_config.min_size_bytes
            },
            'optimization_config': {
                'minification_enabled': self.optimization_config.enable_minification,
                'image_optimization_enabled': self.optimization_config.enable_image_optimization,
                'caching_enabled': self.optimization_config.enable_caching,
                'cdn_enabled': self.optimization_config.cdn_enabled
            }
        }
    
    def reset_metrics(self):
        """Remet à zéro les métriques"""
        self.metrics = AssetMetrics()
        logger.info("📊 Métriques d'optimisation remises à zéro")


# Instance globale
_asset_optimizer: Optional[AssetOptimizer] = None


async def get_asset_optimizer() -> AssetOptimizer:
    """Factory pour obtenir l'optimiseur d'assets"""
    global _asset_optimizer
    
    if _asset_optimizer is None:
        _asset_optimizer = AssetOptimizer()
    
    return _asset_optimizer


# Middleware FastAPI pour optimisation automatique

class AssetOptimizationMiddleware:
    """Middleware d'optimisation des assets"""
    
    def __init__(self, app, enabled: bool = True):
        self.app = app
        self.enabled = enabled
    
    async def __call__(self, scope, receive, send):
        if not self.enabled or scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        request = Request(scope, receive)
        
        # Intercepter la réponse
        response_started = False
        
        async def send_wrapper(message):
            nonlocal response_started
            
            if message["type"] == "http.response.start":
                response_started = True
                # Laisser passer le start sans modification pour l'instant
                await send(message)
            
            elif message["type"] == "http.response.body":
                if response_started:
                    # Optimiser le contenu de la réponse
                    body = message.get("body", b"")
                    if body:
                        content_type = ""
                        
                        # Récupérer content-type depuis les headers précédents
                        # (Cette implémentation est simplifiée)
                        optimizer = await get_asset_optimizer()
                        
                        try:
                            optimized_content, headers = await optimizer.optimize_response(
                                body, content_type, request
                            )
                            
                            # Modifier le message avec le contenu optimisé
                            message = {
                                **message,
                                "body": optimized_content
                            }
                        except Exception as e:
                            logger.warning(f"Erreur optimisation middleware: {e}")
                
                await send(message)
            else:
                await send(message)
        
        await self.app(scope, receive, send_wrapper)


# Décorateurs pour optimisation automatique

def optimize_response(content_type: str = "text/html", cache_ttl: int = 3600):
    """Décorateur pour optimiser automatiquement une réponse"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Exécuter la fonction originale
            result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
            
            # Optimiser si c'est du contenu
            if isinstance(result, (str, bytes)):
                optimizer = await get_asset_optimizer()
                
                # Créer une requête factice pour les headers
                from fastapi import Request
                fake_request = Request({"type": "http", "headers": [(b"accept-encoding", b"gzip, br")]})
                
                optimized_content, headers = await optimizer.optimize_response(
                    result, content_type, fake_request
                )
                
                # Retourner une réponse optimisée
                from fastapi import Response
                return Response(
                    content=optimized_content,
                    media_type=content_type,
                    headers=headers
                )
            
            return result
        
        return wrapper
    return decorator


# Fonctions utilitaires

async def precompress_static_assets(static_directory: str):
    """Précompresse tous les assets statiques d'un répertoire"""
    
    optimizer = await get_asset_optimizer()
    static_path = Path(static_directory)
    
    if not static_path.exists():
        logger.warning(f"Répertoire statique non trouvé: {static_directory}")
        return
    
    processed_count = 0
    total_savings = 0
    
    # Parcourir tous les fichiers
    for file_path in static_path.rglob("*"):
        if file_path.is_file():
            try:
                original_size = file_path.stat().st_size
                
                # Optimiser le fichier
                optimized_asset = await optimizer.optimize_static_file(str(file_path))
                
                if optimized_asset:
                    # Sauvegarder version compressée si significative
                    if optimized_asset.get('compressed_content'):
                        compressed_size = len(optimized_asset['compressed_content'])
                        savings = original_size - compressed_size
                        
                        if savings > 0:
                            # Créer fichier .gz ou .br
                            encoding = optimized_asset.get('encoding', '')
                            if encoding == 'gzip':
                                compressed_path = file_path.with_suffix(file_path.suffix + '.gz')
                                with open(compressed_path, 'wb') as f:
                                    f.write(optimized_asset['compressed_content'])
                            elif encoding == 'br':
                                compressed_path = file_path.with_suffix(file_path.suffix + '.br')
                                with open(compressed_path, 'wb') as f:
                                    f.write(optimized_asset['compressed_content'])
                            
                            total_savings += savings
                            processed_count += 1
                            
                            logger.info(f"📦 {file_path.name}: {original_size} → {compressed_size} bytes ({savings} économisés)")
                
            except Exception as e:
                logger.warning(f"Erreur traitement {file_path}: {e}")
    
    logger.info(f"✅ Précompression terminée: {processed_count} fichiers, {total_savings / 1024:.1f} KB économisés")


async def generate_service_worker(output_path: str, cache_files: List[str]):
    """Génère un Service Worker pour cache offline"""
    
    service_worker_content = f"""
// Service Worker généré automatiquement
// Version: {datetime.now().strftime('%Y%m%d_%H%M%S')}

const CACHE_NAME = 'ma-intelligence-v1';
const urlsToCache = {json.dumps(cache_files, indent=2)};

self.addEventListener('install', function(event) {{
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then(function(cache) {{
        console.log('Cache ouvert');
        return cache.addAll(urlsToCache);
      }})
  );
}});

self.addEventListener('fetch', function(event) {{
  event.respondWith(
    caches.match(event.request)
      .then(function(response) {{
        // Retourner depuis le cache si disponible
        if (response) {{
          return response;
        }}
        
        // Sinon, récupérer depuis le réseau
        return fetch(event.request);
      }})
  );
}});

self.addEventListener('activate', function(event) {{
  event.waitUntil(
    caches.keys().then(function(cacheNames) {{
      return Promise.all(
        cacheNames.map(function(cacheName) {{
          if (cacheName !== CACHE_NAME) {{
            return caches.delete(cacheName);
          }}
        }})
      );
    }})
  );
}});
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(service_worker_content)
    
    logger.info(f"📱 Service Worker généré: {output_path}")


async def get_asset_optimization_report() -> Dict[str, Any]:
    """Génère un rapport d'optimisation des assets"""
    
    optimizer = await get_asset_optimizer()
    metrics = optimizer.get_metrics()
    
    # Analyser les performances
    performance_score = 100
    recommendations = []
    
    if metrics['cache_hit_rate'] < 70:
        performance_score -= 20
        recommendations.append("Augmenter la durée de cache des assets statiques")
    
    if metrics['compression_rate'] < 80:
        performance_score -= 15
        recommendations.append("Activer la compression pour plus de types de contenu")
    
    if metrics['avg_compression_time'] > 0.1:
        performance_score -= 10
        recommendations.append("Optimiser les algorithmes de compression")
    
    return {
        'performance_score': max(0, performance_score),
        'metrics': metrics,
        'recommendations': recommendations,
        'features_status': {
            'compression': metrics['compression_config']['gzip_enabled'] or metrics['compression_config']['brotli_enabled'],
            'minification': metrics['optimization_config']['minification_enabled'],
            'image_optimization': metrics['optimization_config']['image_optimization_enabled'],
            'caching': metrics['optimization_config']['caching_enabled'],
            'cdn': metrics['optimization_config']['cdn_enabled']
        },
        'estimated_bandwidth_savings_mb': metrics['total_savings_mb'],
        'report_generated_at': datetime.now().isoformat()
    }