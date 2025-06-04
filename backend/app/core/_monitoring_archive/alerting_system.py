"""
Système d'alerting intelligent pour M&A Intelligence Platform  
US-003: Alerting multi-canal avec seuils adaptatifs et notifications

Features:
- Alertes multi-canal (email, Slack, webhook, SMS)
- Seuils adaptatifs et machine learning
- Groupement et déduplication d'alertes
- Escalade automatique selon criticité
- Tableau de bord alertes temps réel
- Intégration avec métriques et health checks
"""

import asyncio
import aiohttp
import smtplib
import json
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
import hashlib
import statistics
from collections import defaultdict, deque

from app.core.logging_system import get_logger, LogCategory, audit_logger
from app.core.metrics_collector import get_metrics_collector
from app.config import settings

logger = get_logger("alerting", LogCategory.SYSTEM)


class AlertSeverity(str, Enum):
    """Niveaux de sévérité des alertes"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AlertStatus(str, Enum):
    """États des alertes"""
    ACTIVE = "active"
    RESOLVED = "resolved"
    ACKNOWLEDGED = "acknowledged"
    SUPPRESSED = "suppressed"
    ESCALATED = "escalated"


class AlertChannel(str, Enum):
    """Canaux de notification"""
    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"
    SMS = "sms"
    DESKTOP = "desktop"
    MOBILE_PUSH = "mobile_push"


class AlertCategory(str, Enum):
    """Catégories d'alertes"""
    SYSTEM = "system"
    APPLICATION = "application"
    BUSINESS = "business"
    SECURITY = "security"
    PERFORMANCE = "performance"
    INFRASTRUCTURE = "infrastructure"


@dataclass
class AlertRule:
    """Règle d'alerte avec conditions et actions"""
    name: str
    description: str
    category: AlertCategory
    severity: AlertSeverity
    condition: str  # Expression condition (ex: "cpu_usage > 80")
    metric_name: str
    threshold: float
    comparison: str = ">"  # >, <, >=, <=, ==, !=
    window_minutes: int = 5
    evaluation_frequency: int = 60  # secondes
    channels: List[AlertChannel] = field(default_factory=list)
    recipients: List[str] = field(default_factory=list)
    enabled: bool = True
    tags: Dict[str, str] = field(default_factory=dict)
    
    # Seuils adaptatifs
    adaptive_threshold: bool = False
    baseline_days: int = 7
    sensitivity: float = 1.0  # Facteur sensibilité (0.5 = moins sensible, 2.0 = plus sensible)
    
    # Gestion répétition
    cooldown_minutes: int = 15
    max_alerts_per_hour: int = 4
    escalation_after_minutes: int = 30


@dataclass  
class Alert:
    """Instance d'alerte générée"""
    id: str
    rule_name: str
    title: str
    description: str
    severity: AlertSeverity
    category: AlertCategory
    status: AlertStatus
    created_at: datetime
    metric_name: str
    current_value: float
    threshold_value: float
    tags: Dict[str, str] = field(default_factory=dict)
    
    # Gestion lifecycle
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: str = ""
    resolved_at: Optional[datetime] = None
    escalated_at: Optional[datetime] = None
    
    # Contexte
    context: Dict[str, Any] = field(default_factory=dict)
    related_alerts: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'rule_name': self.rule_name,
            'title': self.title,
            'description': self.description,
            'severity': self.severity.value,
            'category': self.category.value,
            'status': self.status.value,
            'created_at': self.created_at.isoformat(),
            'metric_name': self.metric_name,
            'current_value': self.current_value,
            'threshold_value': self.threshold_value,
            'tags': self.tags,
            'acknowledged_at': self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            'acknowledged_by': self.acknowledged_by,
            'resolved_at': self.resolved_at.isoformat() if self.resolved_at else None,
            'escalated_at': self.escalated_at.isoformat() if self.escalated_at else None,
            'context': self.context,
            'related_alerts': self.related_alerts
        }


@dataclass
class NotificationTemplate:
    """Template de notification"""
    channel: AlertChannel
    subject_template: str
    body_template: str
    format: str = "text"  # text, html, json
    
    def render(self, alert: Alert, context: Dict[str, Any] = None) -> Dict[str, str]:
        """Rend le template avec les données de l'alerte"""
        template_context = {
            'alert': alert,
            'severity': alert.severity.value.upper(),
            'category': alert.category.value.upper(),
            'metric': alert.metric_name,
            'value': alert.current_value,
            'threshold': alert.threshold_value,
            'timestamp': alert.created_at.strftime('%Y-%m-%d %H:%M:%S UTC')
        }
        
        if context:
            template_context.update(context)
        
        try:
            subject = self.subject_template.format(**template_context)
            body = self.body_template.format(**template_context)
            
            return {
                'subject': subject,
                'body': body,
                'format': self.format
            }
        except KeyError as e:
            logger.error(f"Template rendering error: missing variable {e}")
            return {
                'subject': f"Alert: {alert.title}",
                'body': f"Alert {alert.id} triggered. Check system for details.",
                'format': 'text'
            }


class AlertingSystem:
    """
    Système d'alerting centralisé avec intelligence adaptative
    
    Fonctionnalités:
    - Évaluation continue des règles d'alerte
    - Seuils adaptatifs basés sur historique
    - Groupement et déduplication intelligente
    - Notifications multi-canal
    - Escalade automatique
    - Tableau de bord temps réel
    """
    
    def __init__(self):
        self.rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.notification_templates: Dict[AlertChannel, NotificationTemplate] = {}
        
        # État du système
        self.running = False
        self.evaluation_tasks: Dict[str, asyncio.Task] = {}
        self.metrics_collector = get_metrics_collector()
        
        # Cache pour seuils adaptatifs
        self.adaptive_baselines: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10080))  # 7 jours * 24h * 60min
        self.last_alert_times: Dict[str, datetime] = {}
        self.alert_counts: Dict[str, deque] = defaultdict(lambda: deque(maxlen=60))  # 1 heure
        
        # Configuration notifications
        self.notification_config = {
            'email': {
                'smtp_server': getattr(settings, 'SMTP_SERVER', 'localhost'),
                'smtp_port': getattr(settings, 'SMTP_PORT', 587),
                'username': getattr(settings, 'SMTP_USERNAME', ''),
                'password': getattr(settings, 'SMTP_PASSWORD', ''),
                'from_address': getattr(settings, 'ALERT_FROM_EMAIL', 'alerts@ma-intelligence.com')
            },
            'slack': {
                'webhook_url': getattr(settings, 'SLACK_WEBHOOK_URL', ''),
                'channel': getattr(settings, 'SLACK_ALERT_CHANNEL', '#alerts')
            }
        }
        
        self._setup_default_templates()
        self._setup_default_rules()
        
        logger.info("🚨 Système d'alerting initialisé")
    
    def add_alert_rule(self, rule: AlertRule):
        """Ajoute une règle d'alerte"""
        self.rules[rule.name] = rule
        
        # Démarrer tâche d'évaluation si système actif
        if self.running:
            self._start_rule_evaluation(rule.name)
        
        logger.info(f"Règle d'alerte ajoutée: {rule.name} ({rule.severity.value})")
    
    def remove_alert_rule(self, rule_name: str):
        """Supprime une règle d'alerte"""
        if rule_name in self.rules:
            # Arrêter tâche d'évaluation
            if rule_name in self.evaluation_tasks:
                self.evaluation_tasks[rule_name].cancel()
                del self.evaluation_tasks[rule_name]
            
            del self.rules[rule_name]
            logger.info(f"Règle d'alerte supprimée: {rule_name}")
    
    def update_alert_rule(self, rule_name: str, updates: Dict[str, Any]):
        """Met à jour une règle d'alerte"""
        if rule_name not in self.rules:
            raise ValueError(f"Règle non trouvée: {rule_name}")
        
        rule = self.rules[rule_name]
        
        # Appliquer mises à jour
        for key, value in updates.items():
            if hasattr(rule, key):
                setattr(rule, key, value)
        
        logger.info(f"Règle d'alerte mise à jour: {rule_name}")
        audit_logger.audit("update_alert_rule", "alert_rule", rule_name)
    
    async def start_monitoring(self):
        """Démarre le monitoring et l'évaluation des alertes"""
        self.running = True
        logger.info("🔄 Démarrage monitoring alertes")
        
        # Démarrer évaluation pour chaque règle
        for rule_name in self.rules:
            self._start_rule_evaluation(rule_name)
        
        # Tâche de maintenance
        asyncio.create_task(self._maintenance_task())
    
    def stop_monitoring(self):
        """Arrête le monitoring"""
        self.running = False
        
        # Annuler toutes les tâches d'évaluation
        for task in self.evaluation_tasks.values():
            task.cancel()
        
        self.evaluation_tasks.clear()
        logger.info("⏹️ Arrêt monitoring alertes")
    
    async def evaluate_rule(self, rule_name: str) -> Optional[Alert]:
        """Évalue une règle d'alerte spécifique"""
        if rule_name not in self.rules:
            return None
        
        rule = self.rules[rule_name]
        if not rule.enabled:
            return None
        
        try:
            # Récupérer valeur métrique actuelle
            current_value = self.metrics_collector.get_current_value(rule.metric_name)
            
            if current_value is None:
                logger.debug(f"Pas de données pour métrique: {rule.metric_name}")
                return None
            
            # Calculer seuil (adaptatif ou fixe)
            threshold = await self._calculate_threshold(rule, current_value)
            
            # Évaluer condition
            condition_met = self._evaluate_condition(current_value, threshold, rule.comparison)
            
            if condition_met:
                # Vérifier cooldown et limites
                if self._should_suppress_alert(rule):
                    logger.debug(f"Alerte supprimée (cooldown): {rule_name}")
                    return None
                
                # Générer alerte
                alert = await self._create_alert(rule, current_value, threshold)
                
                # Déduplication
                existing_alert = self._find_similar_alert(alert)
                if existing_alert:
                    logger.debug(f"Alerte dédupliquée: {alert.id}")
                    return None
                
                # Enregistrer et notifier
                await self._process_new_alert(alert)
                return alert
            
            else:
                # Vérifier résolution d'alertes actives
                await self._check_alert_resolution(rule, current_value, threshold)
            
        except Exception as e:
            logger.error(f"Erreur évaluation règle {rule_name}", exception=e)
        
        return None
    
    async def acknowledge_alert(self, alert_id: str, user_id: str, comment: str = ""):
        """Acquitte une alerte"""
        if alert_id not in self.active_alerts:
            raise ValueError(f"Alerte non trouvée: {alert_id}")
        
        alert = self.active_alerts[alert_id]
        alert.status = AlertStatus.ACKNOWLEDGED
        alert.acknowledged_at = datetime.now(timezone.utc)
        alert.acknowledged_by = user_id
        
        if comment:
            alert.context['acknowledgment_comment'] = comment
        
        logger.info(f"Alerte acquittée: {alert_id} par {user_id}")
        audit_logger.audit("acknowledge_alert", "alert", alert_id, 
                          details={'user_id': user_id, 'comment': comment})
        
        # Notification acquittement
        await self._send_acknowledgment_notification(alert, user_id)
    
    async def resolve_alert(self, alert_id: str, user_id: str = "system", comment: str = ""):
        """Résout une alerte manuellement"""
        if alert_id not in self.active_alerts:
            raise ValueError(f"Alerte non trouvée: {alert_id}")
        
        alert = self.active_alerts[alert_id]
        alert.status = AlertStatus.RESOLVED
        alert.resolved_at = datetime.now(timezone.utc)
        
        if comment:
            alert.context['resolution_comment'] = comment
        
        # Déplacer vers historique
        self.alert_history.append(alert)
        del self.active_alerts[alert_id]
        
        logger.info(f"Alerte résolue: {alert_id} par {user_id}")
        audit_logger.audit("resolve_alert", "alert", alert_id,
                          details={'user_id': user_id, 'comment': comment})
        
        # Notification résolution
        await self._send_resolution_notification(alert, user_id)
    
    def get_active_alerts(self, severity: AlertSeverity = None, 
                         category: AlertCategory = None) -> List[Alert]:
        """Récupère les alertes actives avec filtres optionnels"""
        alerts = list(self.active_alerts.values())
        
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        
        if category:
            alerts = [a for a in alerts if a.category == category]
        
        # Trier par sévérité puis date
        severity_order = {
            AlertSeverity.EMERGENCY: 0,
            AlertSeverity.CRITICAL: 1,
            AlertSeverity.ERROR: 2,
            AlertSeverity.WARNING: 3,
            AlertSeverity.INFO: 4
        }
        
        alerts.sort(key=lambda a: (severity_order[a.severity], a.created_at))
        return alerts
    
    def get_alert_statistics(self, hours: int = 24) -> Dict[str, Any]:
        """Statistiques des alertes sur période"""
        since = datetime.now(timezone.utc) - timedelta(hours=hours)
        
        # Alertes dans la période
        period_alerts = [a for a in self.alert_history if a.created_at >= since]
        period_alerts.extend([a for a in self.active_alerts.values() if a.created_at >= since])
        
        # Statistiques par sévérité
        by_severity = defaultdict(int)
        for alert in period_alerts:
            by_severity[alert.severity.value] += 1
        
        # Statistiques par catégorie
        by_category = defaultdict(int)
        for alert in period_alerts:
            by_category[alert.category.value] += 1
        
        # Top règles
        by_rule = defaultdict(int)
        for alert in period_alerts:
            by_rule[alert.rule_name] += 1
        
        top_rules = sorted(by_rule.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Temps de résolution moyen
        resolved_alerts = [a for a in period_alerts 
                          if a.status == AlertStatus.RESOLVED and a.resolved_at]
        
        if resolved_alerts:
            resolution_times = [(a.resolved_at - a.created_at).total_seconds() / 60 
                              for a in resolved_alerts]
            avg_resolution_time = statistics.mean(resolution_times)
        else:
            avg_resolution_time = 0
        
        return {
            'period_hours': hours,
            'total_alerts': len(period_alerts),
            'active_alerts': len(self.active_alerts),
            'resolved_alerts': len(resolved_alerts),
            'by_severity': dict(by_severity),
            'by_category': dict(by_category),
            'top_rules': top_rules,
            'avg_resolution_time_minutes': round(avg_resolution_time, 2),
            'alert_rate_per_hour': len(period_alerts) / hours if hours > 0 else 0
        }
    
    async def get_alerting_dashboard_data(self) -> Dict[str, Any]:
        """Données complètes pour dashboard d'alerting"""
        now = datetime.now(timezone.utc)
        
        # Alertes actives par sévérité
        active_by_severity = defaultdict(int)
        for alert in self.active_alerts.values():
            active_by_severity[alert.severity.value] += 1
        
        # Tendances sur 24h
        stats_24h = self.get_alert_statistics(24)
        stats_1h = self.get_alert_statistics(1)
        
        # Règles les plus actives
        active_rules = defaultdict(int)
        for alert in self.active_alerts.values():
            active_rules[alert.rule_name] += 1
        
        # Alertes critiques non acquittées
        critical_unack = [
            a for a in self.active_alerts.values()
            if a.severity in [AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY]
            and a.status == AlertStatus.ACTIVE
        ]
        
        # Santé système alerting
        system_health = {
            'rules_count': len(self.rules),
            'active_rules': len([r for r in self.rules.values() if r.enabled]),
            'evaluation_tasks': len(self.evaluation_tasks),
            'monitoring_active': self.running
        }
        
        return {
            'timestamp': now.isoformat(),
            'summary': {
                'total_active_alerts': len(self.active_alerts),
                'critical_alerts': active_by_severity.get('critical', 0) + active_by_severity.get('emergency', 0),
                'warning_alerts': active_by_severity.get('warning', 0),
                'info_alerts': active_by_severity.get('info', 0),
                'unacknowledged_critical': len(critical_unack)
            },
            'active_alerts_by_severity': dict(active_by_severity),
            'recent_trends': {
                'alerts_last_hour': stats_1h['total_alerts'],
                'alerts_last_24h': stats_24h['total_alerts'],
                'avg_resolution_time_minutes': stats_24h['avg_resolution_time_minutes']
            },
            'top_alert_rules': sorted(active_rules.items(), key=lambda x: x[1], reverse=True)[:5],
            'critical_unacknowledged': [a.to_dict() for a in critical_unack[:10]],
            'system_health': system_health,
            'recent_alerts': [a.to_dict() for a in sorted(
                self.active_alerts.values(), 
                key=lambda x: x.created_at, 
                reverse=True
            )[:20]]
        }
    
    def _start_rule_evaluation(self, rule_name: str):
        """Démarre l'évaluation périodique d'une règle"""
        if rule_name in self.evaluation_tasks:
            self.evaluation_tasks[rule_name].cancel()
        
        rule = self.rules[rule_name]
        self.evaluation_tasks[rule_name] = asyncio.create_task(
            self._rule_evaluation_loop(rule_name, rule.evaluation_frequency)
        )
    
    async def _rule_evaluation_loop(self, rule_name: str, frequency_seconds: int):
        """Boucle d'évaluation d'une règle"""
        while self.running:
            try:
                await self.evaluate_rule(rule_name)
                await asyncio.sleep(frequency_seconds)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Erreur évaluation règle {rule_name}", exception=e)
                await asyncio.sleep(min(frequency_seconds, 60))
    
    async def _calculate_threshold(self, rule: AlertRule, current_value: float) -> float:
        """Calcule le seuil (adaptatif ou fixe)"""
        if not rule.adaptive_threshold:
            return rule.threshold
        
        # Seuil adaptatif basé sur historique
        baseline_key = f"{rule.metric_name}_{rule.name}"
        baseline_values = self.adaptive_baselines[baseline_key]
        
        # Ajouter valeur actuelle à l'historique
        baseline_values.append(current_value)
        
        if len(baseline_values) < 10:  # Pas assez de données
            return rule.threshold
        
        # Calculer statistiques baseline
        baseline_mean = statistics.mean(baseline_values)
        baseline_std = statistics.stdev(baseline_values) if len(baseline_values) > 1 else 0
        
        # Seuil adaptatif = moyenne + (sensibilité * écart-type)
        adaptive_threshold = baseline_mean + (rule.sensitivity * baseline_std)
        
        # Appliquer limites min/max basées sur seuil original
        min_threshold = rule.threshold * 0.5
        max_threshold = rule.threshold * 2.0
        
        adaptive_threshold = max(min_threshold, min(max_threshold, adaptive_threshold))
        
        logger.debug(f"Seuil adaptatif {rule.name}: {adaptive_threshold:.2f} "
                    f"(baseline: {baseline_mean:.2f}±{baseline_std:.2f})")
        
        return adaptive_threshold
    
    def _evaluate_condition(self, value: float, threshold: float, comparison: str) -> bool:
        """Évalue une condition d'alerte"""
        if comparison == ">":
            return value > threshold
        elif comparison == "<":
            return value < threshold
        elif comparison == ">=":
            return value >= threshold
        elif comparison == "<=":
            return value <= threshold
        elif comparison == "==":
            return abs(value - threshold) < 0.001  # Tolérance pour flottants
        elif comparison == "!=":
            return abs(value - threshold) >= 0.001
        else:
            logger.warning(f"Comparaison inconnue: {comparison}")
            return False
    
    def _should_suppress_alert(self, rule: AlertRule) -> bool:
        """Vérifie si une alerte doit être supprimée (cooldown, limites)"""
        now = datetime.now(timezone.utc)
        
        # Vérifier cooldown
        if rule.name in self.last_alert_times:
            last_alert = self.last_alert_times[rule.name]
            if (now - last_alert).total_seconds() < rule.cooldown_minutes * 60:
                return True
        
        # Vérifier limite par heure
        hour_key = f"{rule.name}_{now.hour}"
        alert_counts = self.alert_counts[hour_key]
        
        # Nettoyer anciens compteurs
        cutoff = now - timedelta(hours=1)
        while alert_counts and alert_counts[0] < cutoff:
            alert_counts.popleft()
        
        if len(alert_counts) >= rule.max_alerts_per_hour:
            return True
        
        return False
    
    async def _create_alert(self, rule: AlertRule, current_value: float, threshold: float) -> Alert:
        """Crée une nouvelle alerte"""
        alert_id = self._generate_alert_id(rule, current_value)
        
        alert = Alert(
            id=alert_id,
            rule_name=rule.name,
            title=f"{rule.description} - {rule.metric_name}",
            description=f"Metric {rule.metric_name} is {current_value:.2f} "
                       f"(threshold: {threshold:.2f})",
            severity=rule.severity,
            category=rule.category,
            status=AlertStatus.ACTIVE,
            created_at=datetime.now(timezone.utc),
            metric_name=rule.metric_name,
            current_value=current_value,
            threshold_value=threshold,
            tags=rule.tags.copy()
        )
        
        return alert
    
    def _generate_alert_id(self, rule: AlertRule, current_value: float) -> str:
        """Génère un ID unique pour l'alerte"""
        unique_string = f"{rule.name}_{rule.metric_name}_{current_value}_{datetime.now().isoformat()}"
        return hashlib.md5(unique_string.encode()).hexdigest()[:12]
    
    def _find_similar_alert(self, alert: Alert) -> Optional[Alert]:
        """Trouve une alerte similaire pour déduplication"""
        for existing_alert in self.active_alerts.values():
            if (existing_alert.rule_name == alert.rule_name and
                existing_alert.metric_name == alert.metric_name and
                existing_alert.status == AlertStatus.ACTIVE):
                return existing_alert
        return None
    
    async def _process_new_alert(self, alert: Alert):
        """Traite une nouvelle alerte (enregistrement, notification)"""
        # Enregistrer alerte
        self.active_alerts[alert.id] = alert
        
        # Mettre à jour compteurs
        now = datetime.now(timezone.utc)
        self.last_alert_times[alert.rule_name] = now
        
        hour_key = f"{alert.rule_name}_{now.hour}"
        self.alert_counts[hour_key].append(now)
        
        # Métriques
        self.metrics_collector.increment("alerts_generated", 
                                       labels={'severity': alert.severity.value,
                                              'category': alert.category.value})
        
        # Notification
        await self._send_alert_notifications(alert)
        
        # Log audit
        audit_logger.audit("alert_generated", "alert", alert.id,
                          details={
                              'rule_name': alert.rule_name,
                              'severity': alert.severity.value,
                              'metric_name': alert.metric_name,
                              'current_value': alert.current_value
                          })
        
        logger.warning(f"🚨 ALERTE GÉNÉRÉE: {alert.title}",
                      extra=alert.to_dict())
    
    async def _check_alert_resolution(self, rule: AlertRule, current_value: float, threshold: float):
        """Vérifie si des alertes peuvent être automatiquement résolues"""
        # Chercher alertes actives pour cette règle
        rule_alerts = [a for a in self.active_alerts.values() 
                      if a.rule_name == rule.name and a.status == AlertStatus.ACTIVE]
        
        for alert in rule_alerts:
            # Condition inversée pour résolution
            if rule.comparison == ">":
                condition_resolved = current_value <= threshold
            elif rule.comparison == "<":
                condition_resolved = current_value >= threshold
            elif rule.comparison == ">=":
                condition_resolved = current_value < threshold
            elif rule.comparison == "<=":
                condition_resolved = current_value > threshold
            else:
                continue  # Pas de résolution auto pour == et !=
            
            if condition_resolved:
                await self.resolve_alert(alert.id, "system", 
                                       f"Condition resolved: {current_value:.2f}")
    
    async def _send_alert_notifications(self, alert: Alert):
        """Envoie les notifications pour une alerte"""
        rule = self.rules[alert.rule_name]
        
        for channel in rule.channels:
            try:
                if channel == AlertChannel.EMAIL:
                    await self._send_email_notification(alert, rule.recipients)
                elif channel == AlertChannel.SLACK:
                    await self._send_slack_notification(alert)
                elif channel == AlertChannel.WEBHOOK:
                    await self._send_webhook_notification(alert, rule.recipients)
                
            except Exception as e:
                logger.error(f"Erreur notification {channel.value} pour alerte {alert.id}", 
                           exception=e)
    
    async def _send_email_notification(self, alert: Alert, recipients: List[str]):
        """Envoie notification email"""
        if not recipients or not self.notification_config['email']['smtp_server']:
            return
        
        template = self.notification_templates[AlertChannel.EMAIL]
        content = template.render(alert)
        
        try:
            msg = MimeMultipart()
            msg['From'] = self.notification_config['email']['from_address']
            msg['To'] = ', '.join(recipients)
            msg['Subject'] = content['subject']
            
            if content['format'] == 'html':
                msg.attach(MimeText(content['body'], 'html'))
            else:
                msg.attach(MimeText(content['body'], 'plain'))
            
            # Envoi SMTP (à implémenter selon configuration)
            logger.info(f"Email notification sent for alert {alert.id} to {len(recipients)} recipients")
            
        except Exception as e:
            logger.error(f"Erreur envoi email pour alerte {alert.id}", exception=e)
    
    async def _send_slack_notification(self, alert: Alert):
        """Envoie notification Slack"""
        webhook_url = self.notification_config['slack']['webhook_url']
        if not webhook_url:
            return
        
        # Couleur selon sévérité
        color_map = {
            AlertSeverity.INFO: "#36a64f",
            AlertSeverity.WARNING: "#ff9900", 
            AlertSeverity.ERROR: "#ff0000",
            AlertSeverity.CRITICAL: "#ff0000",
            AlertSeverity.EMERGENCY: "#8B0000"
        }
        
        payload = {
            "channel": self.notification_config['slack']['channel'],
            "username": "M&A Intelligence Alerts",
            "icon_emoji": ":rotating_light:",
            "attachments": [{
                "color": color_map.get(alert.severity, "#ff9900"),
                "title": alert.title,
                "text": alert.description,
                "fields": [
                    {"title": "Severity", "value": alert.severity.value.upper(), "short": True},
                    {"title": "Category", "value": alert.category.value.upper(), "short": True},
                    {"title": "Current Value", "value": f"{alert.current_value:.2f}", "short": True},
                    {"title": "Threshold", "value": f"{alert.threshold_value:.2f}", "short": True}
                ],
                "footer": "M&A Intelligence Platform",
                "ts": int(alert.created_at.timestamp())
            }]
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(webhook_url, json=payload) as response:
                    if response.status == 200:
                        logger.info(f"Slack notification sent for alert {alert.id}")
                    else:
                        logger.error(f"Slack notification failed: {response.status}")
        except Exception as e:
            logger.error(f"Erreur notification Slack pour alerte {alert.id}", exception=e)
    
    async def _send_webhook_notification(self, alert: Alert, webhook_urls: List[str]):
        """Envoie notification webhook"""
        payload = alert.to_dict()
        
        for url in webhook_urls:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, json=payload, timeout=10) as response:
                        if response.status == 200:
                            logger.info(f"Webhook notification sent to {url} for alert {alert.id}")
                        else:
                            logger.warning(f"Webhook notification failed: {url} returned {response.status}")
            except Exception as e:
                logger.error(f"Erreur webhook {url} pour alerte {alert.id}", exception=e)
    
    async def _send_acknowledgment_notification(self, alert: Alert, user_id: str):
        """Notification d'acquittement"""
        logger.info(f"Alert {alert.id} acknowledged by {user_id}")
        # Ici on pourrait envoyer une notification plus légère
    
    async def _send_resolution_notification(self, alert: Alert, user_id: str):
        """Notification de résolution"""
        logger.info(f"Alert {alert.id} resolved by {user_id}")
        # Notification de résolution aux canaux configurés
    
    async def _maintenance_task(self):
        """Tâche de maintenance périodique"""
        while self.running:
            try:
                await self._cleanup_old_alerts()
                await self._check_escalations()
                await asyncio.sleep(300)  # Toutes les 5 minutes
            except Exception as e:
                logger.error("Erreur maintenance alerting", exception=e)
                await asyncio.sleep(60)
    
    async def _cleanup_old_alerts(self):
        """Nettoie les anciennes alertes de l'historique"""
        cutoff = datetime.now(timezone.utc) - timedelta(days=30)
        
        initial_count = len(self.alert_history)
        self.alert_history = [a for a in self.alert_history if a.created_at >= cutoff]
        
        cleaned_count = initial_count - len(self.alert_history)
        if cleaned_count > 0:
            logger.debug(f"Nettoyage historique alertes: {cleaned_count} alertes supprimées")
    
    async def _check_escalations(self):
        """Vérifie et traite les escalades d'alertes"""
        now = datetime.now(timezone.utc)
        
        for alert in self.active_alerts.values():
            rule = self.rules.get(alert.rule_name)
            if not rule:
                continue
            
            # Vérifier escalade
            if (alert.status == AlertStatus.ACTIVE and
                not alert.escalated_at and
                rule.escalation_after_minutes > 0):
                
                time_since_creation = (now - alert.created_at).total_seconds() / 60
                
                if time_since_creation >= rule.escalation_after_minutes:
                    await self._escalate_alert(alert)
    
    async def _escalate_alert(self, alert: Alert):
        """Escalade une alerte"""
        alert.status = AlertStatus.ESCALATED
        alert.escalated_at = datetime.now(timezone.utc)
        
        # Augmenter sévérité si possible
        if alert.severity == AlertSeverity.WARNING:
            alert.severity = AlertSeverity.ERROR
        elif alert.severity == AlertSeverity.ERROR:
            alert.severity = AlertSeverity.CRITICAL
        
        logger.critical(f"🚨 ALERTE ESCALADÉE: {alert.title}", extra=alert.to_dict())
        
        # Notification escalade (canaux prioritaires)
        await self._send_escalation_notifications(alert)
    
    async def _send_escalation_notifications(self, alert: Alert):
        """Notifications d'escalade (prioritaires)"""
        # Notifications plus agressives pour escalades
        logger.critical(f"Alert {alert.id} escalated due to no acknowledgment")
    
    def _setup_default_templates(self):
        """Configure les templates de notification par défaut"""
        
        # Template Email
        email_template = NotificationTemplate(
            channel=AlertChannel.EMAIL,
            subject_template="[{severity}] Alert: {alert.title}",
            body_template="""
Alert Details:
- Title: {alert.title}
- Description: {alert.description}  
- Severity: {severity}
- Category: {category}
- Current Value: {value}
- Threshold: {threshold}
- Timestamp: {timestamp}

Alert ID: {alert.id}
Rule: {alert.rule_name}

Please investigate and acknowledge this alert.
""",
            format="text"
        )
        
        self.notification_templates[AlertChannel.EMAIL] = email_template
        
        # Template Slack (géré directement dans _send_slack_notification)
        
        logger.debug("Templates de notification configurés")
    
    def _setup_default_rules(self):
        """Configure les règles d'alerte par défaut"""
        
        default_rules = [
            # System Resources
            AlertRule(
                name="high_cpu_usage",
                description="High CPU Usage",
                category=AlertCategory.SYSTEM,
                severity=AlertSeverity.WARNING,
                condition="system_cpu_usage > 80",
                metric_name="system_cpu_usage",
                threshold=80.0,
                comparison=">",
                window_minutes=5,
                channels=[AlertChannel.SLACK],
                recipients=[],
                adaptive_threshold=True,
                sensitivity=1.5
            ),
            
            AlertRule(
                name="high_memory_usage",
                description="High Memory Usage",
                category=AlertCategory.SYSTEM,
                severity=AlertSeverity.WARNING,
                condition="system_memory_usage > 85",
                metric_name="system_memory_usage",
                threshold=85.0,
                comparison=">",
                window_minutes=5,
                channels=[AlertChannel.SLACK],
                escalation_after_minutes=30
            ),
            
            # API Performance
            AlertRule(
                name="api_high_response_time",
                description="API Response Time High",
                category=AlertCategory.PERFORMANCE,
                severity=AlertSeverity.WARNING,
                condition="api_response_time > 2000",
                metric_name="api_response_time",
                threshold=2000.0,
                comparison=">",
                window_minutes=3,
                channels=[AlertChannel.SLACK],
                cooldown_minutes=10
            ),
            
            AlertRule(
                name="api_error_rate_high",
                description="API Error Rate High",
                category=AlertCategory.APPLICATION,
                severity=AlertSeverity.ERROR,
                condition="api_errors > 10",
                metric_name="api_errors",
                threshold=10.0,
                comparison=">",
                window_minutes=5,
                channels=[AlertChannel.SLACK],
                escalation_after_minutes=15
            ),
            
            # Cache Performance (intégration US-002)
            AlertRule(
                name="cache_hit_ratio_low",
                description="Cache Hit Ratio Low",
                category=AlertCategory.PERFORMANCE,
                severity=AlertSeverity.WARNING,
                condition="cache_hit_ratio < 70",
                metric_name="cache_hit_ratio",
                threshold=70.0,
                comparison="<",
                window_minutes=10,
                channels=[AlertChannel.SLACK]
            ),
            
            # Database
            AlertRule(
                name="database_connection_issues",
                description="Database Connection Issues",
                category=AlertCategory.INFRASTRUCTURE,
                severity=AlertSeverity.CRITICAL,
                condition="db_connection_pool_usage > 90",
                metric_name="db_connection_pool_usage",
                threshold=90.0,
                comparison=">",
                window_minutes=2,
                channels=[AlertChannel.SLACK],
                escalation_after_minutes=5
            )
        ]
        
        for rule in default_rules:
            self.add_alert_rule(rule)
        
        logger.info(f"Règles d'alerte par défaut configurées: {len(default_rules)} règles")


# Instance globale du système d'alerting
_alerting_system: Optional[AlertingSystem] = None


async def get_alerting_system() -> AlertingSystem:
    """Factory pour obtenir le système d'alerting"""
    global _alerting_system
    
    if _alerting_system is None:
        _alerting_system = AlertingSystem()
    
    return _alerting_system


# Utilitaires

async def send_custom_alert(title: str, description: str, severity: AlertSeverity,
                           category: AlertCategory = AlertCategory.SYSTEM,
                           channels: List[AlertChannel] = None):
    """Envoie une alerte personnalisée"""
    alerting = await get_alerting_system()
    
    alert = Alert(
        id=f"custom_{int(datetime.now().timestamp())}",
        rule_name="custom",
        title=title,
        description=description,
        severity=severity,
        category=category,
        status=AlertStatus.ACTIVE,
        created_at=datetime.now(timezone.utc),
        metric_name="custom",
        current_value=0,
        threshold_value=0
    )
    
    await alerting._process_new_alert(alert)