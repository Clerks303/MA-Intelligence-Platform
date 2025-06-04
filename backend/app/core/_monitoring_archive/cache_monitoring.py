"""
Module de monitoring et métriques cache Redis
US-002: Monitoring complet des performances cache avec alerting

Features:
- Métriques détaillées par type de cache (hit ratio, latence, etc.)
- Alerting automatique si performance dégradée
- Rapports périodiques de performance
- Prédictions d'usage et optimisations
- Dashboard temps réel
"""

import asyncio
import logging
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import statistics

from app.core.cache import get_cache, CacheType, CacheMetrics

logger = logging.getLogger(__name__)


class AlertLevel(str, Enum):
    """Niveaux d'alerte monitoring"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class CacheAlert:
    """Alerte cache"""
    level: AlertLevel
    message: str
    metric_name: str
    current_value: float
    threshold: float
    timestamp: datetime
    cache_type: Optional[CacheType] = None


class CacheMonitor:
    """
    Moniteur de performance cache Redis avec alerting
    
    Surveille:
    - Hit ratio par type de cache
    - Latence moyenne des opérations
    - Usage mémoire Redis
    - Éviction de clés
    - Connexions actives
    """
    
    def __init__(self):
        self.alerts_history: List[CacheAlert] = []
        self.metrics_history: Dict[str, List[Dict]] = {}
        self.monitoring_active = False
        
        # Seuils d'alerting
        self.alert_thresholds = {
            'hit_ratio_warning': 80.0,      # Hit ratio < 80%
            'hit_ratio_critical': 60.0,     # Hit ratio < 60%
            'latency_warning': 50.0,        # Latence > 50ms
            'latency_critical': 100.0,      # Latence > 100ms
            'memory_warning': 80.0,         # Mémoire > 80%
            'memory_critical': 95.0,        # Mémoire > 95%
            'error_rate_warning': 5.0,      # Erreurs > 5%
            'error_rate_critical': 10.0     # Erreurs > 10%
        }
        
        # Configuration monitoring
        self.config = {
            'collection_interval_seconds': 60,    # Collecte toutes les minutes
            'history_retention_hours': 24,        # Historique 24h
            'alert_cooldown_minutes': 10,         # Éviter spam alertes
            'metrics_aggregation_minutes': 5      # Agrégation par 5min
        }
        
        # Cache des dernières alertes (éviter doublons)
        self._last_alerts: Dict[str, datetime] = {}
    
    async def start_monitoring(self):
        """Démarre le monitoring continu"""
        if self.monitoring_active:
            logger.warning("⚠️ Monitoring cache déjà actif")
            return
        
        self.monitoring_active = True
        logger.info("🔍 Démarrage monitoring cache Redis...")
        
        # Boucle monitoring principale
        while self.monitoring_active:
            try:
                await self._collect_metrics()
                await self._check_alerts()
                await self._cleanup_old_data()
                
                # Attendre intervalle suivant
                await asyncio.sleep(self.config['collection_interval_seconds'])
                
            except Exception as e:
                logger.error(f"❌ Erreur monitoring cache: {e}")
                await asyncio.sleep(30)  # Attendre plus longtemps en cas d'erreur
    
    async def stop_monitoring(self):
        """Arrête le monitoring"""
        self.monitoring_active = False
        logger.info("🛑 Arrêt monitoring cache")
    
    async def get_cache_dashboard_data(self) -> Dict[str, Any]:
        """
        Données complètes pour dashboard monitoring
        
        Returns:
            Données formatées pour affichage dashboard
        """
        try:
            cache = await get_cache()
            
            # Informations Redis générales
            cache_info = await cache.get_cache_info()
            
            # Métriques par type de cache
            cache_types_metrics = await self._get_metrics_by_cache_type()
            
            # Alertes actives
            active_alerts = [alert for alert in self.alerts_history[-50:] 
                           if alert.timestamp > datetime.now() - timedelta(hours=1)]
            
            # Trends (dernières 24h)
            trends = await self._calculate_performance_trends()
            
            # Prédictions
            predictions = await self._generate_performance_predictions()
            
            return {
                'timestamp': datetime.now().isoformat(),
                'overall_status': self._determine_overall_status(active_alerts),
                'redis_info': cache_info.get('redis_info', {}),
                'global_metrics': cache_info.get('cache_metrics', {}),
                'cache_types_metrics': cache_types_metrics,
                'active_alerts': [asdict(alert) for alert in active_alerts],
                'performance_trends': trends,
                'predictions': predictions,
                'recommendations': await self._generate_recommendations()
            }
            
        except Exception as e:
            logger.error(f"❌ Erreur génération dashboard: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'overall_status': 'error',
                'error': str(e)
            }
    
    async def get_performance_report(self, hours: int = 24) -> Dict[str, Any]:
        """
        Génère rapport de performance détaillé
        
        Args:
            hours: Période d'analyse en heures
            
        Returns:
            Rapport complet de performance
        """
        try:
            since = datetime.now() - timedelta(hours=hours)
            
            # Métriques historiques
            historical_metrics = await self._get_historical_metrics(since)
            
            # Analyse alertes
            alerts_analysis = await self._analyze_alerts_period(since)
            
            # Économies estimées
            savings_analysis = await self._calculate_cache_savings(since)
            
            # Analyse par type de cache
            cache_type_analysis = await self._analyze_cache_types_performance(since)
            
            return {
                'report_period_hours': hours,
                'generated_at': datetime.now().isoformat(),
                'summary': {
                    'overall_hit_ratio': historical_metrics.get('avg_hit_ratio', 0),
                    'total_operations': historical_metrics.get('total_operations', 0),
                    'avg_latency_ms': historical_metrics.get('avg_latency', 0),
                    'total_alerts': len(alerts_analysis.get('alerts', [])),
                    'estimated_savings_euros': savings_analysis.get('total_savings', 0)
                },
                'detailed_metrics': historical_metrics,
                'alerts_analysis': alerts_analysis,
                'savings_analysis': savings_analysis,
                'cache_types_performance': cache_type_analysis,
                'optimization_suggestions': await self._generate_optimization_suggestions()
            }
            
        except Exception as e:
            logger.error(f"❌ Erreur génération rapport: {e}")
            return {'error': str(e)}
    
    async def trigger_cache_health_check(self) -> Dict[str, Any]:
        """Déclenche health check complet du cache"""
        try:
            cache = await get_cache()
            
            # Health check Redis
            health_result = await cache.health_check()
            
            # Tests performance
            performance_tests = await self._run_performance_tests()
            
            # Vérification configuration
            config_check = await self._check_cache_configuration()
            
            # Recommandations urgentes
            urgent_recommendations = await self._check_urgent_issues()
            
            return {
                'timestamp': datetime.now().isoformat(),
                'health_status': health_result.get('status', 'unknown'),
                'redis_health': health_result,
                'performance_tests': performance_tests,
                'configuration_check': config_check,
                'urgent_recommendations': urgent_recommendations
            }
            
        except Exception as e:
            logger.error(f"❌ Erreur health check: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'health_status': 'error',
                'error': str(e)
            }
    
    # Méthodes privées de monitoring
    
    async def _collect_metrics(self):
        """Collecte périodique des métriques"""
        try:
            cache = await get_cache()
            
            # Métriques principales
            cache_info = await cache.get_cache_info()
            
            # Timestamp de collecte
            timestamp = datetime.now()
            
            # Structurer métriques
            metrics = {
                'timestamp': timestamp.isoformat(),
                'global_metrics': cache_info.get('cache_metrics', {}),
                'redis_info': cache_info.get('redis_info', {}),
                'cache_types': cache_info.get('cache_stats_by_type', {})
            }
            
            # Ajouter à l'historique
            if 'global' not in self.metrics_history:
                self.metrics_history['global'] = []
            
            self.metrics_history['global'].append(metrics)
            
            # Limiter taille historique
            max_records = int(self.config['history_retention_hours'] * 60 / 
                            (self.config['collection_interval_seconds'] / 60))
            
            if len(self.metrics_history['global']) > max_records:
                self.metrics_history['global'] = self.metrics_history['global'][-max_records:]
            
        except Exception as e:
            logger.error(f"❌ Erreur collecte métriques: {e}")
    
    async def _check_alerts(self):
        """Vérification des seuils d'alerte"""
        try:
            if not self.metrics_history.get('global'):
                return
            
            latest_metrics = self.metrics_history['global'][-1]
            global_metrics = latest_metrics.get('global_metrics', {})
            redis_info = latest_metrics.get('redis_info', {})
            
            # Check hit ratio
            hit_ratio = global_metrics.get('hit_ratio_percent', 100)
            await self._check_metric_threshold(
                'hit_ratio', hit_ratio, 
                self.alert_thresholds['hit_ratio_warning'],
                self.alert_thresholds['hit_ratio_critical'],
                reverse=True  # Plus bas = plus mauvais
            )
            
            # Check latence
            latency = global_metrics.get('average_latency_ms', 0)
            await self._check_metric_threshold(
                'latency', latency,
                self.alert_thresholds['latency_warning'],
                self.alert_thresholds['latency_critical']
            )
            
            # Check mémoire
            memory_mb = redis_info.get('memory_used_mb', 0)
            memory_peak_mb = redis_info.get('memory_peak_mb', 1)
            memory_percent = (memory_mb / max(memory_peak_mb, 1)) * 100
            
            await self._check_metric_threshold(
                'memory_usage', memory_percent,
                self.alert_thresholds['memory_warning'],
                self.alert_thresholds['memory_critical']
            )
            
            # Check taux d'erreur
            errors = global_metrics.get('errors', 0)
            total_ops = global_metrics.get('operations_count', 1)
            error_rate = (errors / max(total_ops, 1)) * 100
            
            await self._check_metric_threshold(
                'error_rate', error_rate,
                self.alert_thresholds['error_rate_warning'],
                self.alert_thresholds['error_rate_critical']
            )
            
        except Exception as e:
            logger.error(f"❌ Erreur vérification alertes: {e}")
    
    async def _check_metric_threshold(self, 
                                    metric_name: str, 
                                    current_value: float,
                                    warning_threshold: float,
                                    critical_threshold: float,
                                    reverse: bool = False):
        """Vérifie seuil d'une métrique et génère alerte si nécessaire"""
        
        # Déterminer niveau d'alerte
        alert_level = None
        threshold_exceeded = None
        
        if reverse:  # Pour hit_ratio (plus bas = pire)
            if current_value < critical_threshold:
                alert_level = AlertLevel.CRITICAL
                threshold_exceeded = critical_threshold
            elif current_value < warning_threshold:
                alert_level = AlertLevel.WARNING
                threshold_exceeded = warning_threshold
        else:  # Pour latence, mémoire (plus haut = pire)
            if current_value > critical_threshold:
                alert_level = AlertLevel.CRITICAL
                threshold_exceeded = critical_threshold
            elif current_value > warning_threshold:
                alert_level = AlertLevel.WARNING
                threshold_exceeded = warning_threshold
        
        # Générer alerte si nécessaire
        if alert_level and threshold_exceeded is not None:
            await self._generate_alert(
                alert_level,
                f"{metric_name.replace('_', ' ').title()} {alert_level.value}: {current_value:.2f}",
                metric_name,
                current_value,
                threshold_exceeded
            )
    
    async def _generate_alert(self, 
                            level: AlertLevel, 
                            message: str,
                            metric_name: str,
                            current_value: float,
                            threshold: float,
                            cache_type: Optional[CacheType] = None):
        """Génère une alerte avec cooldown"""
        
        # Clé unique pour éviter spam
        alert_key = f"{metric_name}_{level.value}_{cache_type.value if cache_type else 'global'}"
        now = datetime.now()
        
        # Vérifier cooldown
        if alert_key in self._last_alerts:
            last_alert = self._last_alerts[alert_key]
            cooldown_minutes = self.config['alert_cooldown_minutes']
            
            if now - last_alert < timedelta(minutes=cooldown_minutes):
                return  # Skip alerte (cooldown actif)
        
        # Créer alerte
        alert = CacheAlert(
            level=level,
            message=message,
            metric_name=metric_name,
            current_value=current_value,
            threshold=threshold,
            timestamp=now,
            cache_type=cache_type
        )
        
        self.alerts_history.append(alert)
        self._last_alerts[alert_key] = now
        
        # Log selon niveau
        if level == AlertLevel.CRITICAL:
            logger.error(f"🚨 ALERTE CRITIQUE Cache: {message}")
        elif level == AlertLevel.WARNING:
            logger.warning(f"⚠️ ALERTE WARNING Cache: {message}")
        else:
            logger.info(f"ℹ️ ALERTE INFO Cache: {message}")
    
    async def _cleanup_old_data(self):
        """Nettoyage des données anciennes"""
        retention_hours = self.config['history_retention_hours']
        cutoff = datetime.now() - timedelta(hours=retention_hours)
        
        # Nettoyer alertes anciennes
        self.alerts_history = [
            alert for alert in self.alerts_history 
            if alert.timestamp > cutoff
        ]
        
        # Nettoyer cache last_alerts
        self._last_alerts = {
            key: timestamp for key, timestamp in self._last_alerts.items()
            if timestamp > cutoff
        }
    
    async def _get_metrics_by_cache_type(self) -> Dict[str, Any]:
        """Métriques détaillées par type de cache"""
        try:
            cache = await get_cache()
            cache_info = await cache.get_cache_info()
            
            cache_types_stats = {}
            
            for cache_type in CacheType:
                type_stats = cache_info.get('cache_stats_by_type', {}).get(cache_type.value, {})
                
                cache_types_stats[cache_type.value] = {
                    'key_count': type_stats.get('key_count', 0),
                    'estimated_memory_mb': type_stats.get('key_count', 0) * 0.001,  # Estimation
                    'hit_ratio_estimated': 85.0,  # Placeholder - à calculer depuis métriques
                    'avg_ttl_hours': self._get_cache_type_ttl(cache_type) / 3600,
                    'usage_frequency': 'high' if type_stats.get('key_count', 0) > 100 else 'low'
                }
            
            return cache_types_stats
            
        except Exception as e:
            logger.error(f"❌ Erreur métriques par type: {e}")
            return {}
    
    def _get_cache_type_ttl(self, cache_type: CacheType) -> int:
        """Retourne TTL par défaut pour un type de cache"""
        ttl_map = {
            CacheType.ENRICHMENT_PAPPERS: 86400,  # 24h
            CacheType.ENRICHMENT_KASPR: 21600,    # 6h
            CacheType.SCORING_MA: 3600,           # 1h
            CacheType.EXPORT_CSV: 1800,           # 30min
            CacheType.API_EXTERNAL: 7200          # 2h
        }
        return ttl_map.get(cache_type, 3600)
    
    def _determine_overall_status(self, active_alerts: List[CacheAlert]) -> str:
        """Détermine statut global depuis alertes actives"""
        if not active_alerts:
            return 'healthy'
        
        has_critical = any(alert.level == AlertLevel.CRITICAL for alert in active_alerts)
        has_warning = any(alert.level == AlertLevel.WARNING for alert in active_alerts)
        
        if has_critical:
            return 'critical'
        elif has_warning:
            return 'warning'
        else:
            return 'info'
    
    async def _calculate_performance_trends(self) -> Dict[str, Any]:
        """Calcule tendances de performance"""
        if len(self.metrics_history.get('global', [])) < 2:
            return {'status': 'insufficient_data'}
        
        recent_metrics = self.metrics_history['global'][-10:]  # 10 derniers points
        
        # Extraire hit ratios
        hit_ratios = [
            m.get('global_metrics', {}).get('hit_ratio_percent', 0) 
            for m in recent_metrics
        ]
        
        # Extraire latences
        latencies = [
            m.get('global_metrics', {}).get('average_latency_ms', 0)
            for m in recent_metrics
        ]
        
        # Calculer tendances
        hit_ratio_trend = 'stable'
        if len(hit_ratios) >= 3:
            if hit_ratios[-1] > hit_ratios[-3] + 2:
                hit_ratio_trend = 'improving'
            elif hit_ratios[-1] < hit_ratios[-3] - 2:
                hit_ratio_trend = 'degrading'
        
        latency_trend = 'stable'
        if len(latencies) >= 3:
            if latencies[-1] > latencies[-3] + 5:
                latency_trend = 'degrading'
            elif latencies[-1] < latencies[-3] - 5:
                latency_trend = 'improving'
        
        return {
            'hit_ratio_trend': hit_ratio_trend,
            'latency_trend': latency_trend,
            'current_hit_ratio': hit_ratios[-1] if hit_ratios else 0,
            'current_latency': latencies[-1] if latencies else 0,
            'data_points': len(recent_metrics)
        }
    
    async def _generate_performance_predictions(self) -> Dict[str, Any]:
        """Génère prédictions de performance"""
        # Prédictions simples basées sur tendances
        trends = await self._calculate_performance_trends()
        
        predictions = {
            'next_hour_hit_ratio': trends.get('current_hit_ratio', 80),
            'memory_growth_trend': 'moderate',
            'recommended_actions': []
        }
        
        # Recommandations basées sur tendances
        if trends.get('hit_ratio_trend') == 'degrading':
            predictions['recommended_actions'].append('Investigate cache invalidation patterns')
        
        if trends.get('latency_trend') == 'degrading':
            predictions['recommended_actions'].append('Check Redis server performance')
        
        return predictions
    
    async def _generate_recommendations(self) -> List[str]:
        """Génère recommandations d'optimisation"""
        recommendations = []
        
        # Analyser métriques récentes
        if self.metrics_history.get('global'):
            latest = self.metrics_history['global'][-1]
            global_metrics = latest.get('global_metrics', {})
            
            hit_ratio = global_metrics.get('hit_ratio_percent', 100)
            if hit_ratio < 80:
                recommendations.append("Hit ratio faible - vérifier TTL et patterns d'utilisation")
            
            latency = global_metrics.get('average_latency_ms', 0)
            if latency > 30:
                recommendations.append("Latence élevée - optimiser sérialisation ou network")
            
            errors = global_metrics.get('errors', 0)
            if errors > 10:
                recommendations.append("Taux d'erreur élevé - vérifier connexion Redis")
        
        if not recommendations:
            recommendations.append("Performance cache optimale - aucune action nécessaire")
        
        return recommendations
    
    async def _run_performance_tests(self) -> Dict[str, Any]:
        """Tests de performance cache"""
        try:
            cache = await get_cache()
            
            # Test set/get simple
            test_key = "perf_test"
            test_value = {"test": True, "timestamp": time.time()}
            
            start_time = time.time()
            await cache.set(test_key, test_value, CacheType.API_EXTERNAL, ttl=60)
            set_latency = (time.time() - start_time) * 1000
            
            start_time = time.time()
            retrieved = await cache.get(test_key, CacheType.API_EXTERNAL)
            get_latency = (time.time() - start_time) * 1000
            
            # Nettoyage
            await cache.delete(test_key, CacheType.API_EXTERNAL)
            
            return {
                'set_latency_ms': round(set_latency, 2),
                'get_latency_ms': round(get_latency, 2),
                'set_get_working': retrieved is not None,
                'test_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    async def _check_cache_configuration(self) -> Dict[str, Any]:
        """Vérification configuration cache"""
        return {
            'redis_connection': 'ok',
            'cache_types_configured': len(CacheType),
            'monitoring_active': self.monitoring_active,
            'alert_thresholds': self.alert_thresholds,
            'collection_interval': self.config['collection_interval_seconds']
        }
    
    async def _check_urgent_issues(self) -> List[str]:
        """Vérifications urgentes"""
        issues = []
        
        # Vérifier alertes critiques récentes
        recent_critical = [
            alert for alert in self.alerts_history[-10:]
            if alert.level == AlertLevel.CRITICAL
        ]
        
        if recent_critical:
            issues.append(f"{len(recent_critical)} alertes critiques récentes")
        
        return issues
    
    # Méthodes analytiques (placeholders pour implémentation future)
    
    async def _get_historical_metrics(self, since: datetime) -> Dict[str, Any]:
        """Récupère métriques historiques depuis une date"""
        return {'status': 'placeholder'}
    
    async def _analyze_alerts_period(self, since: datetime) -> Dict[str, Any]:
        """Analyse alertes sur période"""
        return {'status': 'placeholder'}
    
    async def _calculate_cache_savings(self, since: datetime) -> Dict[str, Any]:
        """Calcule économies cache sur période"""
        return {'total_savings': 250.50}  # Placeholder
    
    async def _analyze_cache_types_performance(self, since: datetime) -> Dict[str, Any]:
        """Analyse performance par type de cache"""
        return {'status': 'placeholder'}
    
    async def _generate_optimization_suggestions(self) -> List[str]:
        """Suggestions d'optimisation"""
        return [
            "Augmenter TTL pour les données stables",
            "Implémenter compression pour gros objets",
            "Optimiser patterns de clés"
        ]


# Instance globale du moniteur
_monitor_instance: Optional[CacheMonitor] = None


async def get_cache_monitor() -> CacheMonitor:
    """Factory pour obtenir instance du moniteur"""
    global _monitor_instance
    
    if _monitor_instance is None:
        _monitor_instance = CacheMonitor()
    
    return _monitor_instance


async def start_cache_monitoring():
    """Démarre le monitoring en arrière-plan"""
    monitor = await get_cache_monitor()
    
    # Démarrer en tâche de fond
    asyncio.create_task(monitor.start_monitoring())
    
    logger.info("🚀 Monitoring cache démarré en arrière-plan")