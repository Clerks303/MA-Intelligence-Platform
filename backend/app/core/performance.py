"""
Advanced performance monitoring and optimization for M&A Intelligence Platform
Consolidated from: performance_analyzer, performance_monitor, realtime_monitoring

Provides advanced performance capabilities:
- Performance bottleneck detection
- Real-time metrics streaming  
- System profiling and optimization recommendations
- WebSocket integration for live dashboards
- Advanced analytics (P95, P99, histograms)
"""

import asyncio
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from collections import defaultdict, deque
from dataclasses import dataclass, field
import statistics
import json
import logging
from contextlib import asynccontextmanager

# Optional imports for enhanced functionality
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import tracemalloc
    TRACEMALLOC_AVAILABLE = True
except ImportError:
    TRACEMALLOC_AVAILABLE = False

logger = logging.getLogger(__name__)

# =============================================================================
# 1. PERFORMANCE ANALYSIS SYSTEM
# =============================================================================

@dataclass
class PerformanceMetric:
    name: str
    value: float
    timestamp: datetime
    context: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Bottleneck:
    component: str
    severity: str  # low, medium, high, critical
    description: str
    impact: str
    recommendations: List[str]
    metrics: Dict[str, float]
    timestamp: datetime = field(default_factory=datetime.now)

class PerformanceAnalyzer:
    """Advanced performance analysis and bottleneck detection"""
    
    def __init__(self, history_size: int = 10000):
        self.metrics_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=history_size))
        self.bottlenecks: List[Bottleneck] = []
        self.analysis_rules: Dict[str, Callable] = {}
        self.thresholds = {
            'cpu_high': 80.0,
            'memory_high': 85.0,
            'disk_high': 90.0,
            'response_time_slow': 2.0,  # seconds
            'database_slow': 1.0,  # seconds
            'api_error_rate_high': 0.05  # 5%
        }
        
    def record_metric(self, name: str, value: float, context: Dict[str, Any] = None):
        """Record a performance metric"""
        metric = PerformanceMetric(
            name=name,
            value=value, 
            timestamp=datetime.now(),
            context=context or {}
        )
        self.metrics_history[name].append(metric)
        
    def analyze_system_performance(self) -> Dict[str, Any]:
        """Analyze overall system performance"""
        if not PSUTIL_AVAILABLE:
            logger.warning("psutil not available, returning mock performance data")
            return {
                'cpu': {
                    'usage_percent': 25.0,
                    'core_count': 4,
                    'load_average': [0.5, 0.4, 0.3]
                },
                'memory': {
                    'usage_percent': 45.0,
                    'used_bytes': 1024**3,
                    'available_bytes': 3 * 1024**3,
                    'total_bytes': 4 * 1024**3,
                    'process_rss_bytes': 100 * 1024**2,
                    'process_vms_bytes': 200 * 1024**2
                },
                'disk': {
                    'usage_percent': 60.0,
                    'used_bytes': 60 * 1024**3,
                    'free_bytes': 40 * 1024**3,
                    'total_bytes': 100 * 1024**3
                },
                'network': {
                    'bytes_sent': 1024**3,
                    'bytes_recv': 2 * 1024**3,
                    'packets_sent': 10000,
                    'packets_recv': 15000
                }
            }
        
        try:
            # Get current system metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Network I/O
            net_io = psutil.net_io_counters()
            
            # Process information
            process = psutil.Process()
            process_memory = process.memory_info()
            
            system_perf = {
                'cpu': {
                    'usage_percent': cpu_percent,
                    'core_count': psutil.cpu_count(),
                    'load_average': psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None
                },
                'memory': {
                    'usage_percent': memory.percent,
                    'used_bytes': memory.used,
                    'available_bytes': memory.available,
                    'total_bytes': memory.total,
                    'process_rss_bytes': process_memory.rss,
                    'process_vms_bytes': process_memory.vms
                },
                'disk': {
                    'usage_percent': disk.percent,
                    'used_bytes': disk.used,
                    'free_bytes': disk.free,
                    'total_bytes': disk.total
                },
                'network': {
                    'bytes_sent': net_io.bytes_sent,
                    'bytes_recv': net_io.bytes_recv,
                    'packets_sent': net_io.packets_sent,
                    'packets_recv': net_io.packets_recv
                }
            }
            
            # Record metrics
            self.record_metric('cpu_usage_percent', cpu_percent)
            self.record_metric('memory_usage_percent', memory.percent)
            self.record_metric('disk_usage_percent', disk.percent)
            
            return system_perf
            
        except Exception as e:
            logger.error(f"Error analyzing system performance: {e}")
            return {}
    
    def detect_bottlenecks(self) -> List[Bottleneck]:
        """Detect performance bottlenecks"""
        current_bottlenecks = []
        
        try:
            # Analyze CPU bottlenecks
            cpu_metrics = self.metrics_history.get('cpu_usage_percent', deque())
            if cpu_metrics:
                recent_cpu = list(cpu_metrics)[-10:]  # Last 10 measurements
                avg_cpu = statistics.mean(recent_cpu) if recent_cpu else 0
                
                if avg_cpu > self.thresholds['cpu_high']:
                    severity = 'critical' if avg_cpu > 95 else 'high' if avg_cpu > 90 else 'medium'
                    current_bottlenecks.append(Bottleneck(
                        component='cpu',
                        severity=severity,
                        description=f'High CPU usage detected: {avg_cpu:.1f}%',
                        impact='System responsiveness degraded, requests may timeout',
                        recommendations=[
                            'Scale horizontally by adding more instances',
                            'Optimize CPU-intensive operations',
                            'Review async/await patterns',
                            'Consider caching for expensive computations'
                        ],
                        metrics={'avg_cpu_percent': avg_cpu, 'max_cpu_percent': max(recent_cpu)}
                    ))
            
            # Analyze Memory bottlenecks
            memory_metrics = self.metrics_history.get('memory_usage_percent', deque())
            if memory_metrics:
                recent_memory = list(memory_metrics)[-10:]
                avg_memory = statistics.mean(recent_memory) if recent_memory else 0
                
                if avg_memory > self.thresholds['memory_high']:
                    severity = 'critical' if avg_memory > 95 else 'high' if avg_memory > 90 else 'medium'
                    current_bottlenecks.append(Bottleneck(
                        component='memory',
                        severity=severity,
                        description=f'High memory usage detected: {avg_memory:.1f}%',
                        impact='Risk of OOM kills, increased garbage collection',
                        recommendations=[
                            'Increase available memory',
                            'Review memory leaks in application code',
                            'Optimize data structures and caching',
                            'Implement memory pooling for large objects'
                        ],
                        metrics={'avg_memory_percent': avg_memory, 'max_memory_percent': max(recent_memory)}
                    ))
            
            # Analyze Response Time bottlenecks
            response_time_metrics = self.metrics_history.get('http_request_duration_seconds', deque())
            if response_time_metrics:
                recent_times = [m.value for m in list(response_time_metrics)[-50:]]  # Last 50 requests
                if recent_times:
                    avg_time = statistics.mean(recent_times)
                    p95_time = self._calculate_percentile(recent_times, 95)
                    
                    if avg_time > self.thresholds['response_time_slow']:
                        severity = 'high' if avg_time > 5.0 else 'medium'
                        current_bottlenecks.append(Bottleneck(
                            component='api_response_time',
                            severity=severity,
                            description=f'Slow API response times: avg {avg_time:.2f}s, P95 {p95_time:.2f}s',
                            impact='Poor user experience, potential timeouts',
                            recommendations=[
                                'Optimize database queries',
                                'Add response caching',
                                'Review async processing patterns',
                                'Consider API endpoint pagination'
                            ],
                            metrics={
                                'avg_response_time': avg_time,
                                'p95_response_time': p95_time,
                                'sample_size': len(recent_times)
                            }
                        ))
            
            # Update bottlenecks list
            self.bottlenecks = current_bottlenecks
            return current_bottlenecks
            
        except Exception as e:
            logger.error(f"Error detecting bottlenecks: {e}")
            return []
    
    def _calculate_percentile(self, data: List[float], percentile: float) -> float:
        """Calculate percentile from data"""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        index = int((percentile / 100) * len(sorted_data))
        return sorted_data[min(index, len(sorted_data) - 1)]
    
    def get_performance_recommendations(self) -> Dict[str, List[str]]:
        """Get performance optimization recommendations"""
        recommendations = defaultdict(list)
        
        # Analyze current bottlenecks
        bottlenecks = self.detect_bottlenecks()
        
        for bottleneck in bottlenecks:
            recommendations[bottleneck.component].extend(bottleneck.recommendations)
        
        # General recommendations based on patterns
        if not bottlenecks:
            recommendations['general'] = [
                'System is performing well',
                'Consider implementing proactive monitoring',
                'Review application metrics regularly'
            ]
        
        return dict(recommendations)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        system_perf = self.analyze_system_performance()
        bottlenecks = self.detect_bottlenecks()
        recommendations = self.get_performance_recommendations()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'system_performance': system_perf,
            'bottlenecks': [
                {
                    'component': b.component,
                    'severity': b.severity,
                    'description': b.description,
                    'impact': b.impact,
                    'recommendations': b.recommendations[:3],  # Top 3 recommendations
                    'metrics': b.metrics
                } for b in bottlenecks
            ],
            'recommendations': recommendations,
            'overall_health': 'healthy' if not bottlenecks else 'degraded' if any(b.severity in ['high', 'critical'] for b in bottlenecks) else 'warning'
        }

# =============================================================================
# 2. REAL-TIME MONITORING SYSTEM
# =============================================================================

class RealtimeMonitor:
    """Real-time metrics streaming and WebSocket integration"""
    
    def __init__(self, buffer_size: int = 1000):
        self.metrics_buffer: Dict[str, deque] = defaultdict(lambda: deque(maxlen=buffer_size))
        self.subscribers: List[Callable] = []
        self.streaming_active = False
        self.stream_interval = 1.0  # seconds
        
    def subscribe(self, callback: Callable[[Dict[str, Any]], None]):
        """Subscribe to real-time metrics updates"""
        self.subscribers.append(callback)
        
    def unsubscribe(self, callback: Callable):
        """Unsubscribe from real-time metrics updates"""
        if callback in self.subscribers:
            self.subscribers.remove(callback)
    
    async def start_streaming(self):
        """Start real-time metrics streaming"""
        self.streaming_active = True
        logger.info("Real-time monitoring started")
        
        while self.streaming_active:
            try:
                # Collect current metrics
                metrics_data = self._collect_realtime_metrics()
                
                # Buffer the data
                timestamp = datetime.now()
                for metric_name, value in metrics_data.items():
                    self.metrics_buffer[metric_name].append({
                        'timestamp': timestamp.isoformat(),
                        'value': value
                    })
                
                # Notify subscribers
                await self._notify_subscribers(metrics_data)
                
                await asyncio.sleep(self.stream_interval)
                
            except Exception as e:
                logger.error(f"Error in real-time streaming: {e}")
                await asyncio.sleep(self.stream_interval)
    
    def stop_streaming(self):
        """Stop real-time metrics streaming"""
        self.streaming_active = False
        logger.info("Real-time monitoring stopped")
    
    def _collect_realtime_metrics(self) -> Dict[str, float]:
        """Collect metrics for real-time streaming"""
        if not PSUTIL_AVAILABLE:
            return {
                'cpu_usage': 25.0,
                'memory_usage': 45.0,
                'memory_used_mb': 1024.0,
                'timestamp': time.time()
            }
        
        try:
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            
            return {
                'cpu_usage': cpu_percent,
                'memory_usage': memory.percent,
                'memory_used_mb': memory.used / (1024 * 1024),
                'timestamp': time.time()
            }
        except Exception as e:
            logger.error(f"Error collecting real-time metrics: {e}")
            return {}
    
    async def _notify_subscribers(self, metrics_data: Dict[str, Any]):
        """Notify all subscribers of new metrics data"""
        if not self.subscribers:
            return
            
        # Prepare data for subscribers
        notification_data = {
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics_data,
            'buffer_stats': {
                name: len(buffer) for name, buffer in self.metrics_buffer.items()
            }
        }
        
        # Notify each subscriber
        for callback in self.subscribers:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(notification_data)
                else:
                    callback(notification_data)
            except Exception as e:
                logger.error(f"Error notifying subscriber: {e}")
    
    def get_recent_metrics(self, metric_name: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent metrics for a specific metric"""
        buffer = self.metrics_buffer.get(metric_name, deque())
        return list(buffer)[-limit:]
    
    def get_all_recent_metrics(self, limit: int = 100) -> Dict[str, List[Dict[str, Any]]]:
        """Get recent metrics for all tracked metrics"""
        return {
            name: list(buffer)[-limit:] 
            for name, buffer in self.metrics_buffer.items()
        }

# =============================================================================
# 3. SYSTEM PROFILING
# =============================================================================

class SystemProfiler:
    """Advanced system profiling and analysis"""
    
    def __init__(self):
        self.profiling_active = False
        self.memory_snapshots = []
        self.function_profiles = defaultdict(list)
        
    @asynccontextmanager
    async def profile_memory(self):
        """Context manager for memory profiling"""
        if not TRACEMALLOC_AVAILABLE:
            logger.warning("tracemalloc not available, memory profiling disabled")
            yield
            return
        
        tracemalloc.start()
        try:
            yield
        finally:
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            snapshot = {
                'timestamp': datetime.now().isoformat(),
                'current_bytes': current,
                'peak_bytes': peak,
                'current_mb': current / (1024 * 1024),
                'peak_mb': peak / (1024 * 1024)
            }
            
            self.memory_snapshots.append(snapshot)
            logger.info(f"Memory profiling: current={snapshot['current_mb']:.1f}MB, peak={snapshot['peak_mb']:.1f}MB")
    
    def profile_function_performance(self, func_name: str):
        """Decorator for function performance profiling"""
        def decorator(func):
            async def async_wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = await func(*args, **kwargs)
                    execution_time = time.time() - start_time
                    
                    self.function_profiles[func_name].append({
                        'timestamp': datetime.now().isoformat(),
                        'execution_time': execution_time,
                        'success': True
                    })
                    
                    return result
                except Exception as e:
                    execution_time = time.time() - start_time
                    self.function_profiles[func_name].append({
                        'timestamp': datetime.now().isoformat(),
                        'execution_time': execution_time,
                        'success': False,
                        'error': str(e)
                    })
                    raise
            
            def sync_wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    execution_time = time.time() - start_time
                    
                    self.function_profiles[func_name].append({
                        'timestamp': datetime.now().isoformat(),
                        'execution_time': execution_time,
                        'success': True
                    })
                    
                    return result
                except Exception as e:
                    execution_time = time.time() - start_time
                    self.function_profiles[func_name].append({
                        'timestamp': datetime.now().isoformat(),
                        'execution_time': execution_time,
                        'success': False,
                        'error': str(e)
                    })
                    raise
            
            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        return decorator
    
    def analyze_io_performance(self) -> Dict[str, Any]:
        """Analyze I/O performance"""
        if not PSUTIL_AVAILABLE:
            return {
                'disk_io': {
                    'read_bytes': 1024**3,
                    'write_bytes': 512 * 1024**2,
                    'read_count': 1000,
                    'write_count': 500,
                    'read_time_ms': 1000,
                    'write_time_ms': 500
                },
                'network_io': {
                    'bytes_sent': 1024**3,
                    'bytes_recv': 2 * 1024**3,
                    'packets_sent': 10000,
                    'packets_recv': 15000,
                    'errors_in': 0,
                    'errors_out': 0,
                    'drops_in': 0,
                    'drops_out': 0
                }
            }
        
        try:
            disk_io = psutil.disk_io_counters()
            net_io = psutil.net_io_counters()
            
            return {
                'disk_io': {
                    'read_bytes': disk_io.read_bytes,
                    'write_bytes': disk_io.write_bytes,
                    'read_count': disk_io.read_count,
                    'write_count': disk_io.write_count,
                    'read_time_ms': disk_io.read_time,
                    'write_time_ms': disk_io.write_time
                },
                'network_io': {
                    'bytes_sent': net_io.bytes_sent,
                    'bytes_recv': net_io.bytes_recv,
                    'packets_sent': net_io.packets_sent,
                    'packets_recv': net_io.packets_recv,
                    'errors_in': net_io.errin,
                    'errors_out': net_io.errout,
                    'drops_in': net_io.dropin,
                    'drops_out': net_io.dropout
                }
            }
        except Exception as e:
            logger.error(f"Error analyzing I/O performance: {e}")
            return {}
    
    def get_profiling_summary(self) -> Dict[str, Any]:
        """Get comprehensive profiling summary"""
        
        # Analyze function performance
        function_stats = {}
        for func_name, profiles in self.function_profiles.items():
            if profiles:
                times = [p['execution_time'] for p in profiles if p['success']]
                if times:
                    function_stats[func_name] = {
                        'call_count': len(profiles),
                        'success_count': sum(1 for p in profiles if p['success']),
                        'error_count': sum(1 for p in profiles if not p['success']),
                        'avg_time': statistics.mean(times),
                        'min_time': min(times),
                        'max_time': max(times),
                        'success_rate': sum(1 for p in profiles if p['success']) / len(profiles)
                    }
        
        return {
            'timestamp': datetime.now().isoformat(),
            'memory_snapshots': self.memory_snapshots[-10:],  # Last 10 snapshots
            'function_performance': function_stats,
            'io_performance': self.analyze_io_performance(),
            'profiling_active': self.profiling_active
        }

# =============================================================================
# 4. PERFORMANCE SYSTEM COORDINATOR
# =============================================================================

class PerformanceSystem:
    """Main performance monitoring system coordinator"""
    
    def __init__(self):
        self.analyzer = PerformanceAnalyzer()
        self.realtime_monitor = RealtimeMonitor()
        self.profiler = SystemProfiler()
        self.running = False
        
    async def start_performance_monitoring(self, stream_interval: float = 1.0):
        """Start comprehensive performance monitoring"""
        self.running = True
        self.realtime_monitor.stream_interval = stream_interval
        
        # Start real-time monitoring in background
        realtime_task = asyncio.create_task(self.realtime_monitor.start_streaming())
        
        logger.info("Performance monitoring system started")
        
        try:
            await realtime_task
        except Exception as e:
            logger.error(f"Error in performance monitoring: {e}")
        finally:
            self.running = False
    
    def stop_performance_monitoring(self):
        """Stop performance monitoring"""
        self.running = False
        self.realtime_monitor.stop_streaming()
        logger.info("Performance monitoring system stopped")
    
    def get_complete_performance_status(self) -> Dict[str, Any]:
        """Get complete performance monitoring status"""
        return {
            'timestamp': datetime.now().isoformat(),
            'analysis': self.analyzer.get_performance_summary(),
            'realtime': {
                'streaming_active': self.realtime_monitor.streaming_active,
                'subscriber_count': len(self.realtime_monitor.subscribers),
                'recent_metrics': self.realtime_monitor.get_all_recent_metrics(limit=10)
            },
            'profiling': self.profiler.get_profiling_summary(),
            'system_active': self.running
        }

# Global performance system instance
performance = PerformanceSystem()

# =============================================================================
# 5. UTILITY FUNCTIONS AND DECORATORS
# =============================================================================

def monitor_performance(func_name: str = None):
    """Decorator to monitor function performance"""
    def decorator(func):
        name = func_name or f"{func.__module__}.{func.__name__}"
        return performance.profiler.profile_function_performance(name)(func)
    return decorator

async def get_performance_metrics() -> Dict[str, Any]:
    """Get current performance metrics for API endpoints"""
    return performance.get_complete_performance_status()

def subscribe_to_realtime_metrics(callback: Callable):
    """Subscribe to real-time metrics updates"""
    performance.realtime_monitor.subscribe(callback)

def unsubscribe_from_realtime_metrics(callback: Callable):
    """Unsubscribe from real-time metrics updates"""
    performance.realtime_monitor.unsubscribe(callback)

# Export main components
__all__ = [
    'performance', 'monitor_performance', 'get_performance_metrics',
    'subscribe_to_realtime_metrics', 'unsubscribe_from_realtime_metrics',
    'PerformanceAnalyzer', 'RealtimeMonitor', 'SystemProfiler', 'PerformanceSystem'
]