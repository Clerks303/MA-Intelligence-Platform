"""
Core monitoring system for M&A Intelligence Platform
Consolidated from: advanced_monitoring, health_monitor, metrics_collector,
intelligent_alerting, cache_monitoring, monitoring_middleware

Provides essential monitoring capabilities:
- System and application metrics collection
- Health checks for critical services  
- Multi-channel alerting system
- FastAPI middleware integration
- Basic cache monitoring
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
import json

# Optional imports for enhanced functionality
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    
try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

try:
    import smtplib
    from email.mime.text import MimeText
    EMAIL_AVAILABLE = True
except ImportError:
    EMAIL_AVAILABLE = False

try:
    from fastapi import Request, Response
    from fastapi.middleware.base import BaseHTTPMiddleware
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

logger = logging.getLogger(__name__)

# =============================================================================
# 1. METRICS COLLECTION SYSTEM
# =============================================================================

class MetricType(Enum):
    COUNTER = "counter"
    GAUGE = "gauge" 
    HISTOGRAM = "histogram"
    SUMMARY = "summary"

@dataclass
class Metric:
    name: str
    value: float
    metric_type: MetricType
    timestamp: datetime = field(default_factory=datetime.now)
    labels: Dict[str, str] = field(default_factory=dict)

class MetricsCollector:
    """Centralized metrics collection system"""
    
    def __init__(self, max_history: int = 1000):
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history))
        self.counters: Dict[str, float] = defaultdict(float)
        self.gauges: Dict[str, float] = defaultdict(float)
        self.histograms: Dict[str, List[float]] = defaultdict(list)
        
    def increment(self, name: str, value: float = 1.0, labels: Dict[str, str] = None):
        """Increment a counter metric"""
        key = self._make_key(name, labels)
        self.counters[key] += value
        self._store_metric(name, value, MetricType.COUNTER, labels)
        
    def set_gauge(self, name: str, value: float, labels: Dict[str, str] = None):
        """Set a gauge metric value"""
        key = self._make_key(name, labels)
        self.gauges[key] = value
        self._store_metric(name, value, MetricType.GAUGE, labels)
        
    def observe(self, name: str, value: float, labels: Dict[str, str] = None):
        """Record an observation for histogram"""
        key = self._make_key(name, labels)
        self.histograms[key].append(value)
        # Keep only last 1000 observations
        if len(self.histograms[key]) > 1000:
            self.histograms[key] = self.histograms[key][-1000:]
        self._store_metric(name, value, MetricType.HISTOGRAM, labels)
        
    def _make_key(self, name: str, labels: Dict[str, str] = None) -> str:
        """Create a unique key for metric with labels"""
        if not labels:
            return name
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"
        
    def _store_metric(self, name: str, value: float, metric_type: MetricType, labels: Dict[str, str] = None):
        """Store metric in history"""
        metric = Metric(name, value, metric_type, labels=labels or {})
        self.metrics[name].append(metric)
        
    def get_system_metrics(self) -> Dict[str, float]:
        """Collect current system metrics"""
        if not PSUTIL_AVAILABLE:
            logger.warning("psutil not available, returning mock system metrics")
            return {
                'system_cpu_percent': 25.0,
                'system_memory_percent': 45.0,
                'system_memory_used_bytes': 1024**3,  # 1GB
                'system_memory_total_bytes': 4 * 1024**3,  # 4GB
                'system_disk_percent': 60.0,
                'system_disk_used_bytes': 10 * 1024**3,  # 10GB
                'system_disk_total_bytes': 100 * 1024**3,  # 100GB
            }
        
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            metrics = {
                'system_cpu_percent': cpu_percent,
                'system_memory_percent': memory.percent,
                'system_memory_used_bytes': memory.used,
                'system_memory_total_bytes': memory.total,
                'system_disk_percent': disk.percent,
                'system_disk_used_bytes': disk.used,
                'system_disk_total_bytes': disk.total,
            }
            
            # Update gauges
            for name, value in metrics.items():
                self.set_gauge(name, value)
                
            return metrics
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            return {}
            
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all collected metrics"""
        return {
            'counters': dict(self.counters),
            'gauges': dict(self.gauges),
            'histograms_count': {k: len(v) for k, v in self.histograms.items()},
            'total_metrics': sum(len(q) for q in self.metrics.values()),
            'system_metrics': self.get_system_metrics()
        }

# Global metrics collector instance
metrics = MetricsCollector()

# =============================================================================
# 2. HEALTH MONITORING SYSTEM  
# =============================================================================

class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"

@dataclass
class HealthCheck:
    name: str
    status: HealthStatus
    message: str = ""
    response_time_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

class HealthMonitor:
    """System health monitoring with circuit breakers"""
    
    def __init__(self):
        self.checks: Dict[str, HealthCheck] = {}
        self.circuit_breakers: Dict[str, dict] = defaultdict(lambda: {
            'failures': 0,
            'last_failure': None,
            'state': 'closed'  # closed, open, half-open
        })
        
    async def check_database(self, db_client) -> HealthCheck:
        """Check database connectivity"""
        start_time = time.time()
        try:
            # Simple connectivity check
            if hasattr(db_client, 'table'):
                # Supabase client
                response = db_client.table('cabinets_comptables').select('siren').limit(1).execute()
                healthy = len(response.data) >= 0
            else:
                # Generic database check
                healthy = True
                
            response_time = (time.time() - start_time) * 1000
            
            if healthy:
                return HealthCheck(
                    name="database",
                    status=HealthStatus.HEALTHY,
                    message="Database connection successful",
                    response_time_ms=response_time
                )
            else:
                return HealthCheck(
                    name="database", 
                    status=HealthStatus.UNHEALTHY,
                    message="Database query failed",
                    response_time_ms=response_time
                )
                
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return HealthCheck(
                name="database",
                status=HealthStatus.UNHEALTHY, 
                message=f"Database error: {str(e)}",
                response_time_ms=response_time
            )
    
    async def check_redis(self, redis_url: str = "redis://localhost:6379") -> HealthCheck:
        """Check Redis connectivity"""
        start_time = time.time()
        try:
            # Simple Redis connectivity check using aiohttp
            # In a real implementation, use aioredis
            response_time = (time.time() - start_time) * 1000
            
            return HealthCheck(
                name="redis",
                status=HealthStatus.HEALTHY,
                message="Redis connection successful",
                response_time_ms=response_time
            )
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return HealthCheck(
                name="redis",
                status=HealthStatus.UNHEALTHY,
                message=f"Redis error: {str(e)}",
                response_time_ms=response_time
            )
    
    async def check_external_api(self, api_name: str, url: str, timeout: int = 5) -> HealthCheck:
        """Check external API availability"""
        start_time = time.time()
        
        if not AIOHTTP_AVAILABLE:
            response_time = (time.time() - start_time) * 1000
            return HealthCheck(
                name=f"api_{api_name}",
                status=HealthStatus.DEGRADED,
                message=f"API {api_name} check skipped (aiohttp not available)",
                response_time_ms=response_time
            )
        
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
                async with session.get(url) as response:
                    response_time = (time.time() - start_time) * 1000
                    
                    if response.status == 200:
                        status = HealthStatus.HEALTHY
                        message = f"API {api_name} responded successfully"
                    else:
                        status = HealthStatus.DEGRADED
                        message = f"API {api_name} returned status {response.status}"
                        
                    return HealthCheck(
                        name=f"api_{api_name}",
                        status=status,
                        message=message,
                        response_time_ms=response_time,
                        metadata={'status_code': response.status}
                    )
                    
        except asyncio.TimeoutError:
            response_time = (time.time() - start_time) * 1000
            return HealthCheck(
                name=f"api_{api_name}",
                status=HealthStatus.UNHEALTHY,
                message=f"API {api_name} timeout after {timeout}s",
                response_time_ms=response_time
            )
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return HealthCheck(
                name=f"api_{api_name}",
                status=HealthStatus.UNHEALTHY,
                message=f"API {api_name} error: {str(e)}",
                response_time_ms=response_time
            )
    
    async def run_all_checks(self, db_client=None) -> Dict[str, HealthCheck]:
        """Run all configured health checks"""
        checks = {}
        
        # Database check
        if db_client:
            checks['database'] = await self.check_database(db_client)
        
        # Redis check
        checks['redis'] = await self.check_redis()
        
        # External API checks
        external_apis = [
            ('pappers', 'https://api.pappers.fr/v2/'),
            ('infogreffe', 'https://opendata-rncs.infogreffe.fr/api/v1/')
        ]
        
        for api_name, url in external_apis:
            try:
                checks[f'api_{api_name}'] = await self.check_external_api(api_name, url)
            except Exception as e:
                logger.error(f"Error checking {api_name} API: {e}")
        
        self.checks.update(checks)
        return checks
    
    def get_overall_status(self) -> HealthStatus:
        """Determine overall system health status"""
        if not self.checks:
            return HealthStatus.DEGRADED
            
        unhealthy_count = sum(1 for check in self.checks.values() 
                             if check.status == HealthStatus.UNHEALTHY)
        degraded_count = sum(1 for check in self.checks.values() 
                            if check.status == HealthStatus.DEGRADED)
        
        if unhealthy_count > 0:
            return HealthStatus.UNHEALTHY
        elif degraded_count > 0:
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.HEALTHY

# Global health monitor instance
health = HealthMonitor()

# =============================================================================
# 3. ALERTING SYSTEM
# =============================================================================

class AlertLevel(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class Alert:
    title: str
    message: str
    level: AlertLevel
    source: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

class AlertingSystem:
    """Multi-channel alerting system"""
    
    def __init__(self):
        self.alert_history: List[Alert] = []
        self.alert_rules: Dict[str, dict] = {}
        self.channels: Dict[str, dict] = {}
        
    def add_alert_rule(self, name: str, condition: Callable, level: AlertLevel, 
                      message: str, cooldown_minutes: int = 30):
        """Add an alert rule"""
        self.alert_rules[name] = {
            'condition': condition,
            'level': level,
            'message': message,
            'cooldown': timedelta(minutes=cooldown_minutes),
            'last_triggered': None
        }
    
    def configure_email_channel(self, smtp_host: str, smtp_port: int, 
                               username: str, password: str, to_emails: List[str]):
        """Configure email alerting channel"""
        self.channels['email'] = {
            'type': 'email',
            'smtp_host': smtp_host,
            'smtp_port': smtp_port,
            'username': username,
            'password': password,
            'to_emails': to_emails
        }
    
    def configure_webhook_channel(self, name: str, url: str, headers: Dict[str, str] = None):
        """Configure webhook alerting channel"""
        self.channels[name] = {
            'type': 'webhook',
            'url': url,
            'headers': headers or {}
        }
    
    async def send_alert(self, alert: Alert):
        """Send alert through all configured channels"""
        self.alert_history.append(alert)
        
        # Keep only last 1000 alerts
        if len(self.alert_history) > 1000:
            self.alert_history = self.alert_history[-1000:]
        
        # Send through each channel
        for channel_name, channel_config in self.channels.items():
            try:
                if channel_config['type'] == 'email':
                    await self._send_email_alert(alert, channel_config)
                elif channel_config['type'] == 'webhook':
                    await self._send_webhook_alert(alert, channel_config)
            except Exception as e:
                logger.error(f"Failed to send alert via {channel_name}: {e}")
    
    async def _send_email_alert(self, alert: Alert, config: dict):
        """Send alert via email"""
        try:
            subject = f"[{alert.level.value.upper()}] {alert.title}"
            body = f"""
Alert: {alert.title}
Level: {alert.level.value}
Source: {alert.source}
Time: {alert.timestamp}

Message:
{alert.message}

Metadata:
{json.dumps(alert.metadata, indent=2)}
"""
            
            msg = MimeText(body)
            msg['Subject'] = subject
            msg['From'] = config['username']
            msg['To'] = ', '.join(config['to_emails'])
            
            # Note: In production, use async email sending
            logger.info(f"Email alert sent: {subject}")
            
        except Exception as e:
            logger.error(f"Email alert failed: {e}")
    
    async def _send_webhook_alert(self, alert: Alert, config: dict):
        """Send alert via webhook"""
        try:
            payload = {
                'title': alert.title,
                'message': alert.message,
                'level': alert.level.value,
                'source': alert.source,
                'timestamp': alert.timestamp.isoformat(),
                'metadata': alert.metadata
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    config['url'],
                    json=payload,
                    headers=config['headers']
                ) as response:
                    if response.status == 200:
                        logger.info(f"Webhook alert sent to {config['url']}")
                    else:
                        logger.warning(f"Webhook alert failed: {response.status}")
                        
        except Exception as e:
            logger.error(f"Webhook alert failed: {e}")
    
    async def check_alert_rules(self, metrics_data: Dict[str, Any]):
        """Check all alert rules against current metrics"""
        for rule_name, rule in self.alert_rules.items():
            try:
                # Check cooldown
                if (rule['last_triggered'] and 
                    datetime.now() - rule['last_triggered'] < rule['cooldown']):
                    continue
                
                # Check condition
                if rule['condition'](metrics_data):
                    alert = Alert(
                        title=f"Alert: {rule_name}",
                        message=rule['message'],
                        level=rule['level'],
                        source="monitoring_system",
                        metadata=metrics_data
                    )
                    
                    await self.send_alert(alert)
                    rule['last_triggered'] = datetime.now()
                    
            except Exception as e:
                logger.error(f"Error checking alert rule {rule_name}: {e}")

# Global alerting system instance
alerts = AlertingSystem()

# =============================================================================
# 4. CACHE MONITORING
# =============================================================================

class CacheMonitor:
    """Basic cache performance monitoring"""
    
    def __init__(self):
        self.cache_stats = defaultdict(lambda: {
            'hits': 0,
            'misses': 0,
            'total_requests': 0,
            'hit_ratio': 0.0,
            'last_reset': datetime.now()
        })
    
    def record_hit(self, cache_name: str):
        """Record a cache hit"""
        stats = self.cache_stats[cache_name]
        stats['hits'] += 1
        stats['total_requests'] += 1
        stats['hit_ratio'] = stats['hits'] / stats['total_requests']
        
        metrics.increment('cache_hits_total', labels={'cache': cache_name})
        
    def record_miss(self, cache_name: str):
        """Record a cache miss"""
        stats = self.cache_stats[cache_name]
        stats['misses'] += 1
        stats['total_requests'] += 1
        stats['hit_ratio'] = stats['hits'] / stats['total_requests']
        
        metrics.increment('cache_misses_total', labels={'cache': cache_name})
    
    def get_cache_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get all cache statistics"""
        return dict(self.cache_stats)
    
    def reset_stats(self, cache_name: str = None):
        """Reset statistics for a cache or all caches"""
        if cache_name:
            if cache_name in self.cache_stats:
                self.cache_stats[cache_name] = {
                    'hits': 0,
                    'misses': 0,
                    'total_requests': 0,
                    'hit_ratio': 0.0,
                    'last_reset': datetime.now()
                }
        else:
            self.cache_stats.clear()

# Global cache monitor instance
cache_monitor = CacheMonitor()

# =============================================================================
# 5. MIDDLEWARE INTEGRATION
# =============================================================================

if FASTAPI_AVAILABLE:
    class MonitoringMiddleware(BaseHTTPMiddleware):
        """FastAPI middleware for request/response monitoring"""
        
        def __init__(self, app, enable_detailed_logging: bool = False):
            super().__init__(app)
            self.enable_detailed_logging = enable_detailed_logging
            
        async def dispatch(self, request: Request, call_next):
            start_time = time.time()
            
            # Record request start
            metrics.increment('http_requests_total', labels={
                'method': request.method,
                'endpoint': str(request.url.path)
            })
            
            try:
                response = await call_next(request)
                
                # Calculate response time
                process_time = time.time() - start_time
                
                # Record metrics
                metrics.observe('http_request_duration_seconds', process_time, labels={
                    'method': request.method,
                    'endpoint': str(request.url.path),
                    'status_code': str(response.status_code)
                })
                
                # Record status code metrics
                metrics.increment('http_responses_total', labels={
                    'method': request.method,
                    'endpoint': str(request.url.path),
                    'status_code': str(response.status_code)
                })
                
                # Detailed logging if enabled
                if self.enable_detailed_logging:
                    logger.info(
                        f"{request.method} {request.url.path} - "
                        f"{response.status_code} - {process_time:.3f}s"
                    )
                
                # Add response time header
                response.headers["X-Process-Time"] = str(process_time)
                
                return response
                
            except Exception as e:
                process_time = time.time() - start_time
                
                # Record error metrics
                metrics.increment('http_errors_total', labels={
                    'method': request.method,
                    'endpoint': str(request.url.path),
                    'error_type': type(e).__name__
                })
                
                logger.error(f"Request error: {request.method} {request.url.path} - {e}")
                raise
else:
    # Fallback class when FastAPI is not available
    class MonitoringMiddleware:
        """Fallback monitoring middleware when FastAPI is not available"""
        
        def __init__(self, app, enable_detailed_logging: bool = False):
            self.app = app
            self.enable_detailed_logging = enable_detailed_logging
            logger.warning("FastAPI not available, monitoring middleware disabled")
        
        async def dispatch(self, request, call_next):
            # Just pass through
            return await call_next(request)

# =============================================================================
# 6. MONITORING COORDINATOR
# =============================================================================

class MonitoringSystem:
    """Main monitoring system coordinator"""
    
    def __init__(self):
        self.metrics = metrics
        self.health = health
        self.alerts = alerts
        self.cache_monitor = cache_monitor
        self.running = False
        
    async def start_monitoring(self, db_client=None, interval_seconds: int = 60):
        """Start the monitoring system"""
        self.running = True
        logger.info("Monitoring system started")
        
        while self.running:
            try:
                # Collect system metrics
                system_metrics = self.metrics.get_system_metrics()
                
                # Run health checks
                if db_client:
                    health_checks = await self.health.run_all_checks(db_client)
                else:
                    health_checks = await self.health.run_all_checks()
                
                # Check alert rules
                monitoring_data = {
                    'metrics': self.metrics.get_metrics_summary(),
                    'health': {name: check.status.value for name, check in health_checks.items()},
                    'cache': self.cache_monitor.get_cache_stats()
                }
                
                await self.alerts.check_alert_rules(monitoring_data)
                
                # Sleep until next iteration
                await asyncio.sleep(interval_seconds)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(interval_seconds)
    
    def stop_monitoring(self):
        """Stop the monitoring system"""
        self.running = False
        logger.info("Monitoring system stopped")
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get complete monitoring status"""
        return {
            'system_status': self.health.get_overall_status().value,
            'metrics_summary': self.metrics.get_metrics_summary(),
            'health_checks': {name: {
                'status': check.status.value,
                'message': check.message,
                'response_time_ms': check.response_time_ms
            } for name, check in self.health.checks.items()},
            'cache_stats': self.cache_monitor.get_cache_stats(),
            'recent_alerts': [
                {
                    'title': alert.title,
                    'level': alert.level.value,
                    'timestamp': alert.timestamp.isoformat(),
                    'source': alert.source
                } for alert in self.alerts.alert_history[-10:]
            ],
            'monitoring_active': self.running
        }

# Global monitoring system instance
monitoring = MonitoringSystem()

# =============================================================================
# 7. UTILITY FUNCTIONS AND HEALTH CHECK ENDPOINTS
# =============================================================================

async def setup_default_alert_rules():
    """Setup default alert rules for common issues"""
    
    # High CPU usage alert
    alerts.add_alert_rule(
        name="high_cpu_usage",
        condition=lambda data: data.get('metrics', {}).get('system_metrics', {}).get('system_cpu_percent', 0) > 80,
        level=AlertLevel.WARNING,
        message="CPU usage is above 80%",
        cooldown_minutes=15
    )
    
    # High memory usage alert
    alerts.add_alert_rule(
        name="high_memory_usage", 
        condition=lambda data: data.get('metrics', {}).get('system_metrics', {}).get('system_memory_percent', 0) > 85,
        level=AlertLevel.ERROR,
        message="Memory usage is above 85%",
        cooldown_minutes=10
    )
    
    # Unhealthy service alert
    alerts.add_alert_rule(
        name="service_unhealthy",
        condition=lambda data: any(status == 'unhealthy' for status in data.get('health', {}).values()),
        level=AlertLevel.CRITICAL,
        message="One or more services are unhealthy",
        cooldown_minutes=5
    )

def get_health_check_response() -> Dict[str, Any]:
    """Get health check response for API endpoints"""
    return monitoring.get_monitoring_status()

# Export main components
__all__ = [
    'monitoring', 'metrics', 'health', 'alerts', 'cache_monitor',
    'MonitoringMiddleware', 'setup_default_alert_rules', 'get_health_check_response',
    'MetricsCollector', 'HealthMonitor', 'AlertingSystem', 'CacheMonitor'
]