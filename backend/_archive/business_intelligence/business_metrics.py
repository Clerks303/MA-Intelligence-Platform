"""
Système de métriques business pour M&A Intelligence Platform
US-006: Tracking KPIs business avec calculs automatisés et insights

Features:
- KPIs M&A: Deal flow, conversion rates, prospect quality
- Métriques scraping: Success rates, data quality, coverage
- Analytics utilisateurs: Engagement, features usage
- ROI tracking: Cost per lead, revenue attribution
- Predictive analytics: Trend analysis, forecasting
- Custom business rules et alertes
"""

import asyncio
import statistics
from datetime import datetime, timedelta, date
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict, deque
import json
import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd

from app.config import settings
from app.core.logging_system import get_logger, LogCategory
from app.core.cache_manager import get_cache_manager, cached
from app.core.database_optimizer import get_database_optimizer
from app.core.advanced_monitoring import get_advanced_monitoring

logger = get_logger("business_metrics", LogCategory.BUSINESS)


class MetricCategory(str, Enum):
    """Catégories de métriques business"""
    SCRAPING = "scraping"
    PROSPECTS = "prospects"
    CONVERSION = "conversion"
    DATA_QUALITY = "data_quality"
    USER_ENGAGEMENT = "user_engagement"
    OPERATIONAL = "operational"
    FINANCIAL = "financial"


class TimeGranularity(str, Enum):
    """Granularité temporelle"""
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"


@dataclass
class BusinessKPI:
    """Indicateur clé de performance business"""
    name: str
    category: MetricCategory
    value: float
    target: Optional[float] = None
    unit: str = ""
    description: str = ""
    calculation_method: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def target_achievement(self) -> Optional[float]:
        """Pourcentage d'atteinte de l'objectif"""
        if self.target and self.target > 0:
            return (self.value / self.target) * 100
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        result['target_achievement'] = self.target_achievement
        return result


@dataclass
class TrendAnalysis:
    """Analyse de tendance d'une métrique"""
    metric_name: str
    period_days: int
    trend_direction: str  # "increasing", "decreasing", "stable"
    trend_strength: float  # 0-1
    growth_rate_percent: float
    forecast_7d: float
    forecast_30d: float
    confidence_score: float
    analysis_timestamp: datetime = field(default_factory=datetime.now)


class BusinessMetricsCalculator:
    """Calculateur de métriques business"""
    
    def __init__(self):
        self.cache_ttl = 300  # 5 minutes
        self.historical_data: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        logger.info("📊 Business metrics calculator initialisé")
    
    @cached('business_metrics', ttl_seconds=300)
    async def calculate_scraping_metrics(self) -> Dict[str, BusinessKPI]:
        """Calcule les métriques de scraping"""
        
        try:
            # Simuler récupération données DB (à remplacer par vraies requêtes)
            db_optimizer = await get_database_optimizer()
            
            # Métriques scraping des 24 dernières heures
            now = datetime.now()
            yesterday = now - timedelta(days=1)
            
            # Simuler données (remplacer par vraies requêtes SQL)
            scraping_data = await self._get_scraping_data_mock(yesterday, now)
            
            metrics = {}
            
            # 1. Volume de scraping
            total_attempts = scraping_data['total_attempts']
            successful_scrapes = scraping_data['successful_scrapes']
            
            metrics['scraping_volume_24h'] = BusinessKPI(
                name='scraping_volume_24h',
                category=MetricCategory.SCRAPING,
                value=total_attempts,
                target=500,  # Objectif: 500 scrapes par jour
                unit='companies',
                description='Nombre total de tentatives de scraping sur 24h',
                calculation_method='COUNT(*) FROM scraping_logs WHERE created_at >= NOW() - INTERVAL 24 HOUR'
            )
            
            # 2. Taux de succès scraping
            success_rate = (successful_scrapes / total_attempts * 100) if total_attempts > 0 else 0
            
            metrics['scraping_success_rate'] = BusinessKPI(
                name='scraping_success_rate',
                category=MetricCategory.SCRAPING,
                value=success_rate,
                target=95.0,  # Objectif: 95% de succès
                unit='%',
                description='Pourcentage de scraping réussis',
                calculation_method='(successful_scrapes / total_attempts) * 100'
            )
            
            # 3. Temps moyen de scraping
            avg_scraping_time = scraping_data['avg_duration_seconds']
            
            metrics['avg_scraping_time'] = BusinessKPI(
                name='avg_scraping_time',
                category=MetricCategory.SCRAPING,
                value=avg_scraping_time,
                target=30.0,  # Objectif: <30s par company
                unit='seconds',
                description='Temps moyen de scraping par entreprise',
                calculation_method='AVG(duration_seconds) FROM scraping_logs WHERE status = "success"'
            )
            
            # 4. Couverture des sources
            sources_coverage = scraping_data['sources_used']
            total_sources = 3  # Pappers, Société.com, Infogreffe
            
            metrics['sources_coverage'] = BusinessKPI(
                name='sources_coverage',
                category=MetricCategory.SCRAPING,
                value=sources_coverage,
                target=total_sources,
                unit='sources',
                description='Nombre de sources de données utilisées',
                calculation_method='COUNT(DISTINCT source) FROM scraping_logs WHERE status = "success"',
                metadata={'total_available': total_sources}
            )
            
            return metrics
            
        except Exception as e:
            logger.error(f"Erreur calcul métriques scraping: {e}")
            return {}
    
    @cached('business_metrics', ttl_seconds=600)
    async def calculate_prospect_metrics(self) -> Dict[str, BusinessKPI]:
        """Calcule les métriques de prospection"""
        
        try:
            # Données prospects (simulées)
            prospect_data = await self._get_prospect_data_mock()
            
            metrics = {}
            
            # 1. Nouveaux prospects identifiés
            new_prospects_24h = prospect_data['new_prospects_24h']
            
            metrics['new_prospects_24h'] = BusinessKPI(
                name='new_prospects_24h',
                category=MetricCategory.PROSPECTS,
                value=new_prospects_24h,
                target=50,  # Objectif: 50 nouveaux prospects/jour
                unit='prospects',
                description='Nouveaux prospects identifiés dans les 24h',
                calculation_method='COUNT(*) FROM companies WHERE created_at >= NOW() - INTERVAL 24 HOUR AND statut = "à contacter"'
            )
            
            # 2. Score moyen de prospection
            avg_prospect_score = prospect_data['avg_prospect_score']
            
            metrics['avg_prospect_score'] = BusinessKPI(
                name='avg_prospect_score',
                category=MetricCategory.PROSPECTS,
                value=avg_prospect_score,
                target=75.0,  # Objectif: score moyen >75
                unit='points',
                description='Score moyen de prospection (0-100)',
                calculation_method='AVG(score_prospection) FROM companies WHERE score_prospection IS NOT NULL'
            )
            
            # 3. Prospects qualifiés (score >80)
            high_quality_prospects = prospect_data['high_quality_prospects']
            total_prospects = prospect_data['total_prospects']
            
            qualification_rate = (high_quality_prospects / total_prospects * 100) if total_prospects > 0 else 0
            
            metrics['prospect_qualification_rate'] = BusinessKPI(
                name='prospect_qualification_rate',
                category=MetricCategory.PROSPECTS,
                value=qualification_rate,
                target=25.0,  # Objectif: 25% de prospects qualifiés
                unit='%',
                description='Pourcentage de prospects avec score >80',
                calculation_method='(COUNT(*) WHERE score_prospection > 80) / COUNT(*) * 100'
            )
            
            # 4. Taille moyenne du chiffre d'affaires
            avg_turnover = prospect_data['avg_turnover']
            
            metrics['avg_prospect_turnover'] = BusinessKPI(
                name='avg_prospect_turnover',
                category=MetricCategory.PROSPECTS,
                value=avg_turnover,
                target=2000000,  # Objectif: CA moyen 2M€
                unit='euros',
                description='Chiffre d\'affaires moyen des prospects',
                calculation_method='AVG(chiffre_affaires) FROM companies WHERE chiffre_affaires > 0'
            )
            
            return metrics
            
        except Exception as e:
            logger.error(f"Erreur calcul métriques prospects: {e}")
            return {}
    
    @cached('business_metrics', ttl_seconds=900)
    async def calculate_conversion_metrics(self) -> Dict[str, BusinessKPI]:
        """Calcule les métriques de conversion"""
        
        try:
            # Données conversion (simulées)
            conversion_data = await self._get_conversion_data_mock()
            
            metrics = {}
            
            # 1. Taux de conversion prospect -> contact
            contacted_prospects = conversion_data['contacted_prospects']
            total_prospects = conversion_data['total_prospects']
            
            contact_rate = (contacted_prospects / total_prospects * 100) if total_prospects > 0 else 0
            
            metrics['prospect_to_contact_rate'] = BusinessKPI(
                name='prospect_to_contact_rate',
                category=MetricCategory.CONVERSION,
                value=contact_rate,
                target=15.0,  # Objectif: 15% des prospects contactés
                unit='%',
                description='Taux de conversion prospect vers contact',
                calculation_method='(COUNT(*) WHERE statut = "contacté") / COUNT(*) * 100'
            )
            
            # 2. Taux de conversion contact -> qualifié
            qualified_prospects = conversion_data['qualified_prospects']
            
            qualification_rate = (qualified_prospects / contacted_prospects * 100) if contacted_prospects > 0 else 0
            
            metrics['contact_to_qualified_rate'] = BusinessKPI(
                name='contact_to_qualified_rate',
                category=MetricCategory.CONVERSION,
                value=qualification_rate,
                target=30.0,  # Objectif: 30% des contacts qualifiés
                unit='%',
                description='Taux de conversion contact vers qualifié',
                calculation_method='(COUNT(*) WHERE statut = "qualifié") / (COUNT(*) WHERE statut IN ("contacté", "qualifié")) * 100'
            )
            
            # 3. Temps moyen de conversion
            avg_conversion_days = conversion_data['avg_conversion_days']
            
            metrics['avg_conversion_time'] = BusinessKPI(
                name='avg_conversion_time',
                category=MetricCategory.CONVERSION,
                value=avg_conversion_days,
                target=14.0,  # Objectif: <14 jours
                unit='days',
                description='Temps moyen de conversion prospect -> qualifié',
                calculation_method='AVG(DATEDIFF(updated_at, created_at)) FROM companies WHERE statut = "qualifié"'
            )
            
            # 4. ROI estimé
            estimated_roi = conversion_data['estimated_deal_value'] / conversion_data['acquisition_cost']
            
            metrics['estimated_roi'] = BusinessKPI(
                name='estimated_roi',
                category=MetricCategory.FINANCIAL,
                value=estimated_roi,
                target=5.0,  # Objectif: ROI >5x
                unit='ratio',
                description='ROI estimé des prospects qualifiés',
                calculation_method='estimated_deal_value / acquisition_cost',
                metadata={
                    'deal_value': conversion_data['estimated_deal_value'],
                    'acquisition_cost': conversion_data['acquisition_cost']
                }
            )
            
            return metrics
            
        except Exception as e:
            logger.error(f"Erreur calcul métriques conversion: {e}")
            return {}
    
    @cached('business_metrics', ttl_seconds=1800)
    async def calculate_data_quality_metrics(self) -> Dict[str, BusinessKPI]:
        """Calcule les métriques de qualité des données"""
        
        try:
            quality_data = await self._get_data_quality_mock()
            
            metrics = {}
            
            # 1. Complétude des données
            complete_profiles = quality_data['complete_profiles']
            total_companies = quality_data['total_companies']
            
            completeness_rate = (complete_profiles / total_companies * 100) if total_companies > 0 else 0
            
            metrics['data_completeness'] = BusinessKPI(
                name='data_completeness',
                category=MetricCategory.DATA_QUALITY,
                value=completeness_rate,
                target=85.0,  # Objectif: 85% de profils complets
                unit='%',
                description='Pourcentage de profils d\'entreprise complets',
                calculation_method='Profiles with >=80% fields filled'
            )
            
            # 2. Fraîcheur des données
            fresh_data_rate = quality_data['fresh_data_rate']
            
            metrics['data_freshness'] = BusinessKPI(
                name='data_freshness',
                category=MetricCategory.DATA_QUALITY,
                value=fresh_data_rate,
                target=90.0,  # Objectif: 90% de données <30j
                unit='%',
                description='Pourcentage de données mises à jour <30 jours',
                calculation_method='COUNT(*) WHERE last_scraped_at > NOW() - INTERVAL 30 DAY / COUNT(*) * 100'
            )
            
            # 3. Taux de validation des données
            validation_rate = quality_data['validation_success_rate']
            
            metrics['data_validation_rate'] = BusinessKPI(
                name='data_validation_rate',
                category=MetricCategory.DATA_QUALITY,
                value=validation_rate,
                target=98.0,  # Objectif: 98% de validation
                unit='%',
                description='Pourcentage de données validées avec succès',
                calculation_method='Successful validations / Total validations * 100'
            )
            
            # 4. Score de fiabilité des sources
            source_reliability = quality_data['source_reliability_score']
            
            metrics['source_reliability'] = BusinessKPI(
                name='source_reliability',
                category=MetricCategory.DATA_QUALITY,
                value=source_reliability,
                target=95.0,  # Objectif: score >95
                unit='points',
                description='Score de fiabilité moyen des sources de données',
                calculation_method='Weighted average of source reliability scores'
            )
            
            return metrics
            
        except Exception as e:
            logger.error(f"Erreur calcul métriques qualité: {e}")
            return {}
    
    async def calculate_user_engagement_metrics(self) -> Dict[str, BusinessKPI]:
        """Calcule les métriques d'engagement utilisateur"""
        
        try:
            engagement_data = await self._get_user_engagement_mock()
            
            metrics = {}
            
            # 1. Utilisateurs actifs quotidiens
            daily_active_users = engagement_data['daily_active_users']
            
            metrics['daily_active_users'] = BusinessKPI(
                name='daily_active_users',
                category=MetricCategory.USER_ENGAGEMENT,
                value=daily_active_users,
                target=25,  # Objectif: 25 utilisateurs actifs/jour
                unit='users',
                description='Nombre d\'utilisateurs actifs dans les 24h',
                calculation_method='COUNT(DISTINCT user_id) FROM user_sessions WHERE created_at >= NOW() - INTERVAL 24 HOUR'
            )
            
            # 2. Durée moyenne de session
            avg_session_duration = engagement_data['avg_session_duration_minutes']
            
            metrics['avg_session_duration'] = BusinessKPI(
                name='avg_session_duration',
                category=MetricCategory.USER_ENGAGEMENT,
                value=avg_session_duration,
                target=15.0,  # Objectif: >15 min par session
                unit='minutes',
                description='Durée moyenne des sessions utilisateur',
                calculation_method='AVG(session_duration_seconds) / 60 FROM user_sessions'
            )
            
            # 3. Actions par session
            actions_per_session = engagement_data['avg_actions_per_session']
            
            metrics['actions_per_session'] = BusinessKPI(
                name='actions_per_session',
                category=MetricCategory.USER_ENGAGEMENT,
                value=actions_per_session,
                target=12.0,  # Objectif: >12 actions/session
                unit='actions',
                description='Nombre moyen d\'actions par session',
                calculation_method='AVG(action_count) FROM user_sessions'
            )
            
            # 4. Taux de rétention 7 jours
            retention_rate_7d = engagement_data['retention_rate_7d']
            
            metrics['user_retention_7d'] = BusinessKPI(
                name='user_retention_7d',
                category=MetricCategory.USER_ENGAGEMENT,
                value=retention_rate_7d,
                target=70.0,  # Objectif: 70% de rétention
                unit='%',
                description='Taux de rétention utilisateur à 7 jours',
                calculation_method='Users active in last 7 days / Total users * 100'
            )
            
            return metrics
            
        except Exception as e:
            logger.error(f"Erreur calcul métriques engagement: {e}")
            return {}
    
    async def analyze_metric_trends(self, metric_name: str, days: int = 30) -> Optional[TrendAnalysis]:
        """Analyse les tendances d'une métrique"""
        
        try:
            # Récupérer historique métrique
            historical_values = self.historical_data.get(metric_name, deque())
            
            if len(historical_values) < 7:  # Minimum 7 points
                return None
            
            # Préparer données pour analyse
            values = [point['value'] for point in historical_values]
            dates = [point['timestamp'] for point in historical_values]
            
            # Conversion en format numérique pour regression
            x_numeric = np.array([(d - dates[0]).days for d in dates]).reshape(-1, 1)
            y_values = np.array(values)
            
            # Régression linéaire pour tendance
            model = LinearRegression()
            model.fit(x_numeric, y_values)
            
            # Coefficient de la pente
            slope = model.coef_[0]
            
            # Direction de la tendance
            if abs(slope) < 0.01:  # Seuil de stabilité
                trend_direction = "stable"
                trend_strength = 0.0
            elif slope > 0:
                trend_direction = "increasing"
                trend_strength = min(1.0, abs(slope) / (np.std(values) + 0.001))
            else:
                trend_direction = "decreasing"
                trend_strength = min(1.0, abs(slope) / (np.std(values) + 0.001))
            
            # Taux de croissance
            if len(values) >= 2:
                first_value = values[0]
                last_value = values[-1]
                growth_rate = ((last_value - first_value) / abs(first_value + 0.001)) * 100
            else:
                growth_rate = 0.0
            
            # Prédictions simples
            next_7d_x = len(values) + 7
            next_30d_x = len(values) + 30
            
            forecast_7d = model.predict([[next_7d_x]])[0]
            forecast_30d = model.predict([[next_30d_x]])[0]
            
            # Score de confiance basé sur R²
            r2_score = model.score(x_numeric, y_values)
            confidence = max(0.1, min(1.0, r2_score))
            
            return TrendAnalysis(
                metric_name=metric_name,
                period_days=days,
                trend_direction=trend_direction,
                trend_strength=trend_strength,
                growth_rate_percent=growth_rate,
                forecast_7d=forecast_7d,
                forecast_30d=forecast_30d,
                confidence_score=confidence
            )
            
        except Exception as e:
            logger.error(f"Erreur analyse tendance {metric_name}: {e}")
            return None
    
    async def update_metric_history(self, metric_name: str, value: float):
        """Met à jour l'historique d'une métrique"""
        
        self.historical_data[metric_name].append({
            'timestamp': datetime.now(),
            'value': value
        })
    
    # Méthodes mockées (à remplacer par vraies requêtes DB)
    
    async def _get_scraping_data_mock(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Données de scraping simulées"""
        return {
            'total_attempts': 342,
            'successful_scrapes': 322,
            'failed_scrapes': 20,
            'avg_duration_seconds': 28.5,
            'sources_used': 3,
            'companies_enriched': 298
        }
    
    async def _get_prospect_data_mock(self) -> Dict[str, Any]:
        """Données prospects simulées"""
        return {
            'new_prospects_24h': 28,
            'total_prospects': 15420,
            'avg_prospect_score': 73.2,
            'high_quality_prospects': 3855,  # Score >80
            'avg_turnover': 1850000
        }
    
    async def _get_conversion_data_mock(self) -> Dict[str, Any]:
        """Données conversion simulées"""
        return {
            'total_prospects': 15420,
            'contacted_prospects': 2313,  # 15%
            'qualified_prospects': 694,   # 30% des contactés
            'avg_conversion_days': 12.5,
            'estimated_deal_value': 450000,
            'acquisition_cost': 75000
        }
    
    async def _get_data_quality_mock(self) -> Dict[str, Any]:
        """Données qualité simulées"""
        return {
            'total_companies': 15420,
            'complete_profiles': 13107,  # 85%
            'fresh_data_rate': 92.3,     # 92.3% <30j
            'validation_success_rate': 97.8,
            'source_reliability_score': 94.2
        }
    
    async def _get_user_engagement_mock(self) -> Dict[str, Any]:
        """Données engagement simulées"""
        return {
            'daily_active_users': 18,
            'avg_session_duration_minutes': 22.5,
            'avg_actions_per_session': 14.2,
            'retention_rate_7d': 76.3
        }


class BusinessMetricsAggregator:
    """Agrégateur de métriques business avec KPIs consolidés"""
    
    def __init__(self):
        self.calculator = BusinessMetricsCalculator()
        self.cache_manager = None
        
        logger.info("📈 Business metrics aggregator initialisé")
    
    async def get_all_business_kpis(self) -> Dict[str, Dict[str, BusinessKPI]]:
        """Récupère tous les KPIs business organisés par catégorie"""
        
        try:
            # Calculer toutes les métriques en parallèle
            scraping_metrics, prospect_metrics, conversion_metrics, quality_metrics, engagement_metrics = await asyncio.gather(
                self.calculator.calculate_scraping_metrics(),
                self.calculator.calculate_prospect_metrics(), 
                self.calculator.calculate_conversion_metrics(),
                self.calculator.calculate_data_quality_metrics(),
                self.calculator.calculate_user_engagement_metrics()
            )
            
            return {
                'scraping': scraping_metrics,
                'prospects': prospect_metrics,
                'conversion': conversion_metrics,
                'data_quality': quality_metrics,
                'user_engagement': engagement_metrics
            }
            
        except Exception as e:
            logger.error(f"Erreur agrégation KPIs: {e}")
            return {}
    
    async def get_executive_dashboard(self) -> Dict[str, Any]:
        """Dashboard exécutif avec KPIs principaux"""
        
        all_kpis = await self.get_all_business_kpis()
        
        # Sélectionner KPIs clés pour le dashboard exécutif
        executive_kpis = {}
        
        # Revenue/Business impact
        if 'conversion' in all_kpis and 'estimated_roi' in all_kpis['conversion']:
            executive_kpis['roi'] = all_kpis['conversion']['estimated_roi']
        
        if 'prospects' in all_kpis and 'new_prospects_24h' in all_kpis['prospects']:
            executive_kpis['new_prospects'] = all_kpis['prospects']['new_prospects_24h']
        
        # Operational efficiency
        if 'scraping' in all_kpis and 'scraping_success_rate' in all_kpis['scraping']:
            executive_kpis['scraping_performance'] = all_kpis['scraping']['scraping_success_rate']
        
        if 'data_quality' in all_kpis and 'data_completeness' in all_kpis['data_quality']:
            executive_kpis['data_quality'] = all_kpis['data_quality']['data_completeness']
        
        # User adoption
        if 'user_engagement' in all_kpis and 'daily_active_users' in all_kpis['user_engagement']:
            executive_kpis['user_adoption'] = all_kpis['user_engagement']['daily_active_users']
        
        # Calculer score global de santé business
        health_score = await self._calculate_business_health_score(all_kpis)
        
        return {
            'timestamp': datetime.now().isoformat(),
            'business_health_score': health_score,
            'key_metrics': {k: v.to_dict() for k, v in executive_kpis.items()},
            'alerts': await self._get_business_alerts(all_kpis),
            'trends': await self._get_key_trends(),
            'recommendations': await self._generate_business_recommendations(all_kpis)
        }
    
    async def _calculate_business_health_score(self, all_kpis: Dict[str, Dict[str, BusinessKPI]]) -> float:
        """Calcule un score global de santé business (0-100)"""
        
        scores = []
        weights = {
            'scraping': 0.2,
            'prospects': 0.25,
            'conversion': 0.3,
            'data_quality': 0.15,
            'user_engagement': 0.1
        }
        
        for category, kpis in all_kpis.items():
            category_scores = []
            
            for kpi in kpis.values():
                if kpi.target and kpi.target > 0:
                    achievement = min(100, (kpi.value / kpi.target) * 100)
                    category_scores.append(achievement)
            
            if category_scores:
                category_avg = sum(category_scores) / len(category_scores)
                weight = weights.get(category, 0.1)
                scores.append(category_avg * weight)
        
        return sum(scores) if scores else 50.0
    
    async def _get_business_alerts(self, all_kpis: Dict[str, Dict[str, BusinessKPI]]) -> List[Dict[str, Any]]:
        """Identifie les alertes business"""
        
        alerts = []
        
        for category, kpis in all_kpis.items():
            for kpi in kpis.values():
                if kpi.target and kpi.target > 0:
                    achievement = (kpi.value / kpi.target) * 100
                    
                    if achievement < 70:  # Performance critique
                        alerts.append({
                            'type': 'performance_critical',
                            'metric': kpi.name,
                            'category': category,
                            'current': kpi.value,
                            'target': kpi.target,
                            'achievement_percent': achievement,
                            'severity': 'high' if achievement < 50 else 'medium'
                        })
                    elif achievement > 120:  # Performance exceptionnelle
                        alerts.append({
                            'type': 'performance_excellent',
                            'metric': kpi.name,
                            'category': category,
                            'current': kpi.value,
                            'target': kpi.target,
                            'achievement_percent': achievement,
                            'severity': 'info'
                        })
        
        return alerts
    
    async def _get_key_trends(self) -> Dict[str, TrendAnalysis]:
        """Analyse des tendances clés"""
        
        key_metrics = [
            'new_prospects_24h',
            'scraping_success_rate',
            'estimated_roi',
            'data_completeness'
        ]
        
        trends = {}
        
        for metric in key_metrics:
            trend = await self.calculator.analyze_metric_trends(metric)
            if trend:
                trends[metric] = asdict(trend)
        
        return trends
    
    async def _generate_business_recommendations(self, all_kpis: Dict[str, Dict[str, BusinessKPI]]) -> List[str]:
        """Génère des recommandations business"""
        
        recommendations = []
        
        # Analyser chaque catégorie
        for category, kpis in all_kpis.items():
            low_performers = [
                kpi for kpi in kpis.values()
                if kpi.target and (kpi.value / kpi.target) < 0.8
            ]
            
            if low_performers:
                if category == 'scraping':
                    recommendations.append(
                        f"Optimiser le processus de scraping: {len(low_performers)} métriques sous-performantes"
                    )
                elif category == 'conversion':
                    recommendations.append(
                        f"Améliorer le funnel de conversion: taux actuel {low_performers[0].value:.1f}% vs objectif {low_performers[0].target:.1f}%"
                    )
                elif category == 'data_quality':
                    recommendations.append(
                        f"Renforcer la qualité des données: {len(low_performers)} indicateurs en dessous de l'objectif"
                    )
        
        if not recommendations:
            recommendations.append("Performance globale satisfaisante - continuer sur cette lancée")
        
        return recommendations


# Instance globale
_business_metrics: Optional[BusinessMetricsAggregator] = None


async def get_business_metrics() -> BusinessMetricsAggregator:
    """Factory pour obtenir l'agrégateur de métriques business"""
    global _business_metrics
    
    if _business_metrics is None:
        _business_metrics = BusinessMetricsAggregator()
    
    return _business_metrics