"""
Système d'alertes intelligentes pour M&A Intelligence Platform  
US-006: Alerting avec ML, escalation automatique et notifications multi-canaux

Features:
- Détection intelligente d'anomalies avec ML
- Escalation automatique selon la sévérité
- Notifications multi-canaux (Email, Slack, SMS)
- Corrélation d'alertes et déduplication
- Alertes business et techniques
- Templates personnalisables et i18n
- Circuit breaker pour éviter le spam
"""

import asyncio
import json
import smtplib
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Callable, Union, Set
from dataclasses import dataclass, field, asdict
from enum import Enum
from functools import wraps
from collections import defaultdict, deque
import hashlib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
from jinja2 import Template, Environment, BaseLoader
import pytz

# Slack SDK
try:
    from slack_sdk import WebClient
    from slack_sdk.errors import SlackApiError
    SLACK_AVAILABLE = True
except ImportError:
    SLACK_AVAILABLE = False

from app.config import settings
from app.core.logging_system import get_logger, LogCategory, audit_logger
from app.core.advanced_monitoring import AnomalyDetection, SeverityLevel
from app.core.cache_manager import get_cache_manager

logger = get_logger("intelligent_alerting", LogCategory.MONITORING)


class AlertType(str, Enum):
    """Types d'alertes"""
    SYSTEM = "system"
    PERFORMANCE = "performance"
    BUSINESS = "business"
    SECURITY = "security"
    DATA_QUALITY = "data_quality"
    SCRAPING = "scraping"
    USER_ACTION = "user_action"


class AlertStatus(str, Enum):
    """Statuts d'alerte"""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"
    ESCALATED = "escalated"


class NotificationChannel(str, Enum):
    """Canaux de notification"""
    EMAIL = "email"
    SLACK = "slack"
    SMS = "sms"
    WEBHOOK = "webhook"
    CONSOLE = "console"


class EscalationLevel(str, Enum):
    """Niveaux d'escalation"""
    L1_MONITORING = "l1_monitoring"
    L2_ENGINEERING = "l2_engineering"
    L3_MANAGEMENT = "l3_management"
    L4_EXECUTIVE = "l4_executive"


@dataclass
class AlertRule:
    """Règle d'alerte configurable"""
    name: str
    description: str
    alert_type: AlertType
    severity: SeverityLevel
    condition: str  # Expression à évaluer
    threshold_value: float
    operator: str  # >, <, ==, >=, <=
    time_window_minutes: int = 5
    min_occurrences: int = 1
    channels: List[NotificationChannel] = field(default_factory=list)
    escalation_minutes: int = 30
    tags: Set[str] = field(default_factory=set)
    enabled: bool = True
    cooldown_minutes: int = 60  # Période de silence après alerte


@dataclass
class AlertInstance:
    """Instance d'alerte générée"""
    id: str
    rule_name: str
    alert_type: AlertType
    severity: SeverityLevel
    title: str
    description: str
    current_value: float
    threshold_value: float
    first_triggered: datetime
    last_triggered: datetime
    status: AlertStatus = AlertStatus.ACTIVE
    escalation_level: EscalationLevel = EscalationLevel.L1_MONITORING
    acknowledgments: List[Dict[str, Any]] = field(default_factory=list)
    notifications_sent: List[Dict[str, Any]] = field(default_factory=list)
    correlation_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: Set[str] = field(default_factory=set)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'rule_name': self.rule_name,
            'alert_type': self.alert_type.value,
            'severity': self.severity.value,
            'title': self.title,
            'description': self.description,
            'current_value': self.current_value,
            'threshold_value': self.threshold_value,
            'first_triggered': self.first_triggered.isoformat(),
            'last_triggered': self.last_triggered.isoformat(),
            'status': self.status.value,
            'escalation_level': self.escalation_level.value,
            'acknowledgments': self.acknowledgments,
            'notifications_sent': self.notifications_sent,
            'correlation_id': self.correlation_id,
            'metadata': self.metadata,
            'tags': list(self.tags)
        }


@dataclass
class NotificationTemplate:
    """Template de notification"""
    name: str
    channel: NotificationChannel
    subject_template: str
    body_template: str
    language: str = "fr"
    format: str = "text"  # text, html, markdown


class CircuitBreaker:
    """Circuit breaker pour éviter spam de notifications"""
    
    def __init__(self, failure_threshold: int = 5, timeout_seconds: int = 300):
        self.failure_threshold = failure_threshold
        self.timeout_seconds = timeout_seconds
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open
    
    def call(self, func: Callable, *args, **kwargs):
        """Exécute une fonction avec circuit breaker"""
        if self.state == "open":
            if time.time() - self.last_failure_time > self.timeout_seconds:
                self.state = "half-open"
            else:
                raise Exception("Circuit breaker is open")
        
        try:
            result = func(*args, **kwargs)
            if self.state == "half-open":
                self.state = "closed"
                self.failure_count = 0
            return result
        
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "open"
            
            raise


class NotificationService:
    """Service de notifications multi-canaux"""
    
    def __init__(self):
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.templates: Dict[str, NotificationTemplate] = {}
        self.jinja_env = Environment(loader=BaseLoader())
        
        self._setup_default_templates()
        self._setup_clients()
        
        logger.info("📢 Service de notifications initialisé")
    
    def _setup_default_templates(self):
        """Configure les templates par défaut"""
        
        # Template Email
        self.templates['email_alert'] = NotificationTemplate(
            name='email_alert',
            channel=NotificationChannel.EMAIL,
            subject_template='🚨 Alerte {{ severity.upper() }}: {{ title }}',
            body_template='''
Alerte détectée sur M&A Intelligence Platform

Type: {{ alert_type }}
Sévérité: {{ severity }}
Titre: {{ title }}
Description: {{ description }}

Détails:
- Valeur actuelle: {{ current_value }}
- Seuil: {{ threshold_value }}
- Première occurrence: {{ first_triggered }}
- Dernière occurrence: {{ last_triggered }}

Métadonnées:
{% for key, value in metadata.items() %}
- {{ key }}: {{ value }}
{% endfor %}

---
Alerte générée automatiquement le {{ now.strftime('%Y-%m-%d %H:%M:%S') }}
            '''.strip(),
            format='text'
        )
        
        # Template Slack
        self.templates['slack_alert'] = NotificationTemplate(
            name='slack_alert',
            channel=NotificationChannel.SLACK,
            subject_template='',
            body_template='''
{
    "blocks": [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": "🚨 Alerte {{ severity.upper() }}"
            }
        },
        {
            "type": "section",
            "fields": [
                {
                    "type": "mrkdwn",
                    "text": "*Type:*\\n{{ alert_type }}"
                },
                {
                    "type": "mrkdwn",
                    "text": "*Sévérité:*\\n{{ severity }}"
                },
                {
                    "type": "mrkdwn",
                    "text": "*Valeur:*\\n{{ current_value }}"
                },
                {
                    "type": "mrkdwn",
                    "text": "*Seuil:*\\n{{ threshold_value }}"
                }
            ]
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": "*Description:*\\n{{ description }}"
            }
        },
        {
            "type": "actions",
            "elements": [
                {
                    "type": "button",
                    "text": {
                        "type": "plain_text",
                        "text": "Acknowledge"
                    },
                    "value": "ack_{{ alert_id }}",
                    "action_id": "acknowledge_alert"
                },
                {
                    "type": "button",
                    "text": {
                        "type": "plain_text",
                        "text": "View Dashboard"
                    },
                    "url": "{{ dashboard_url }}",
                    "action_id": "view_dashboard"
                }
            ]
        }
    ]
}
            '''.strip(),
            format='json'
        )
    
    def _setup_clients(self):
        """Configure les clients de notification"""
        # Slack client
        if SLACK_AVAILABLE and hasattr(settings, 'SLACK_BOT_TOKEN'):
            try:
                self.slack_client = WebClient(token=settings.SLACK_BOT_TOKEN)
                logger.info("✅ Client Slack configuré")
            except Exception as e:
                logger.warning(f"Slack client non configuré: {e}")
                self.slack_client = None
        else:
            self.slack_client = None
        
        # Email configuration
        self.smtp_config = {
            'host': getattr(settings, 'SMTP_HOST', 'localhost'),
            'port': getattr(settings, 'SMTP_PORT', 587),
            'username': getattr(settings, 'SMTP_USERNAME', ''),
            'password': getattr(settings, 'SMTP_PASSWORD', ''),
            'use_tls': getattr(settings, 'SMTP_USE_TLS', True),
            'from_email': getattr(settings, 'SMTP_FROM_EMAIL', 'alerts@ma-intelligence.com')
        }
    
    def _get_circuit_breaker(self, channel: str) -> CircuitBreaker:
        """Récupère ou crée un circuit breaker pour un canal"""
        if channel not in self.circuit_breakers:
            self.circuit_breakers[channel] = CircuitBreaker()
        return self.circuit_breakers[channel]
    
    async def send_notification(self, 
                               alert: AlertInstance, 
                               channel: NotificationChannel,
                               recipients: List[str]) -> bool:
        """Envoie une notification sur un canal"""
        try:
            circuit_breaker = self._get_circuit_breaker(channel.value)
            
            if channel == NotificationChannel.EMAIL:
                return circuit_breaker.call(self._send_email, alert, recipients)
            elif channel == NotificationChannel.SLACK:
                return circuit_breaker.call(self._send_slack, alert, recipients)
            elif channel == NotificationChannel.CONSOLE:
                return circuit_breaker.call(self._send_console, alert, recipients)
            else:
                logger.warning(f"Canal de notification non supporté: {channel}")
                return False
        
        except Exception as e:
            logger.error(f"Erreur envoi notification {channel}: {e}")
            return False
    
    def _send_email(self, alert: AlertInstance, recipients: List[str]) -> bool:
        """Envoie une notification email"""
        try:
            template = self.templates['email_alert']
            
            # Rendu du template
            subject = self._render_template(template.subject_template, alert)
            body = self._render_template(template.body_template, alert)
            
            # Configuration SMTP
            msg = MimeMultipart()
            msg['From'] = self.smtp_config['from_email']
            msg['To'] = ', '.join(recipients)
            msg['Subject'] = subject
            
            msg.attach(MimeText(body, 'plain'))
            
            # Envoi
            with smtplib.SMTP(self.smtp_config['host'], self.smtp_config['port']) as server:
                if self.smtp_config['use_tls']:
                    server.starttls()
                
                if self.smtp_config['username']:
                    server.login(self.smtp_config['username'], self.smtp_config['password'])
                
                server.send_message(msg)
            
            logger.info(f"📧 Email envoyé: {alert.title} -> {recipients}")
            return True
        
        except Exception as e:
            logger.error(f"Erreur envoi email: {e}")
            return False
    
    def _send_slack(self, alert: AlertInstance, channels: List[str]) -> bool:
        """Envoie une notification Slack"""
        if not self.slack_client:
            logger.warning("Client Slack non configuré")
            return False
        
        try:
            template = self.templates['slack_alert']
            
            # Rendu du template JSON
            message_json = self._render_template(template.body_template, alert)
            message_data = json.loads(message_json)
            
            success = True
            for channel in channels:
                try:
                    response = self.slack_client.chat_postMessage(
                        channel=channel,
                        blocks=message_data['blocks']
                    )
                    
                    if not response['ok']:
                        success = False
                        logger.error(f"Erreur Slack: {response.get('error', 'Unknown')}")
                
                except SlackApiError as e:
                    success = False
                    logger.error(f"Erreur API Slack: {e}")
            
            if success:
                logger.info(f"📱 Slack envoyé: {alert.title} -> {channels}")
            
            return success
        
        except Exception as e:
            logger.error(f"Erreur envoi Slack: {e}")
            return False
    
    def _send_console(self, alert: AlertInstance, recipients: List[str]) -> bool:
        """Envoie une notification console (logs)"""
        try:
            severity_emoji = {
                SeverityLevel.LOW: "🟡",
                SeverityLevel.MEDIUM: "🟠", 
                SeverityLevel.HIGH: "🔴",
                SeverityLevel.CRITICAL: "🚨"
            }
            
            emoji = severity_emoji.get(alert.severity, "⚪")
            
            message = (
                f"{emoji} ALERTE {alert.severity.value.upper()}: {alert.title}\n"
                f"   Type: {alert.alert_type.value}\n"
                f"   Valeur: {alert.current_value} (seuil: {alert.threshold_value})\n"
                f"   Description: {alert.description}\n"
                f"   ID: {alert.id}"
            )
            
            # Log selon la sévérité
            if alert.severity in [SeverityLevel.CRITICAL, SeverityLevel.HIGH]:
                logger.error(message)
            elif alert.severity == SeverityLevel.MEDIUM:
                logger.warning(message)
            else:
                logger.info(message)
            
            return True
        
        except Exception as e:
            logger.error(f"Erreur notification console: {e}")
            return False
    
    def _render_template(self, template_str: str, alert: AlertInstance) -> str:
        """Rend un template avec les données de l'alerte"""
        template = self.jinja_env.from_string(template_str)
        
        context = {
            **alert.to_dict(),
            'now': datetime.now(),
            'dashboard_url': f"{getattr(settings, 'FRONTEND_URL', 'http://localhost:3000')}/dashboard",
            'alert_id': alert.id
        }
        
        return template.render(**context)


class AlertCorrelationEngine:
    """Moteur de corrélation d'alertes"""
    
    def __init__(self):
        self.correlation_window_minutes = 10
        self.similarity_threshold = 0.8
        
        logger.info("🔗 Moteur de corrélation d'alertes initialisé")
    
    def correlate_alerts(self, new_alert: AlertInstance, existing_alerts: List[AlertInstance]) -> Optional[str]:
        """Corrèle une nouvelle alerte avec les existantes"""
        
        # Fenêtre temporelle
        cutoff_time = datetime.now() - timedelta(minutes=self.correlation_window_minutes)
        recent_alerts = [
            alert for alert in existing_alerts
            if alert.last_triggered > cutoff_time and alert.status == AlertStatus.ACTIVE
        ]
        
        # Recherche de corrélations
        for existing_alert in recent_alerts:
            similarity = self._calculate_similarity(new_alert, existing_alert)
            
            if similarity > self.similarity_threshold:
                logger.info(f"🔗 Corrélation trouvée: {new_alert.id} <-> {existing_alert.id} (similarité: {similarity:.2f})")
                return existing_alert.correlation_id or existing_alert.id
        
        return None
    
    def _calculate_similarity(self, alert1: AlertInstance, alert2: AlertInstance) -> float:
        """Calcule la similarité entre deux alertes"""
        score = 0.0
        
        # Type d'alerte (poids: 30%)
        if alert1.alert_type == alert2.alert_type:
            score += 0.3
        
        # Règle (poids: 25%)
        if alert1.rule_name == alert2.rule_name:
            score += 0.25
        
        # Sévérité (poids: 15%)
        if alert1.severity == alert2.severity:
            score += 0.15
        
        # Tags communs (poids: 20%)
        if alert1.tags and alert2.tags:
            common_tags = alert1.tags.intersection(alert2.tags)
            tag_similarity = len(common_tags) / len(alert1.tags.union(alert2.tags))
            score += 0.2 * tag_similarity
        
        # Valeurs proches (poids: 10%)
        if alert1.current_value and alert2.current_value:
            value_diff = abs(alert1.current_value - alert2.current_value)
            max_value = max(abs(alert1.current_value), abs(alert2.current_value), 1)
            value_similarity = 1 - min(value_diff / max_value, 1)
            score += 0.1 * value_similarity
        
        return score


class IntelligentAlertingSystem:
    """Système d'alertes intelligentes principal"""
    
    def __init__(self):
        self.notification_service = NotificationService()
        self.correlation_engine = AlertCorrelationEngine()
        
        # Configuration des règles d'alerte
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, AlertInstance] = {}
        self.alert_history: List[AlertInstance] = []
        
        # Configuration escalation
        self.escalation_contacts = {
            EscalationLevel.L1_MONITORING: {
                'email': ['monitoring@ma-intelligence.com'],
                'slack': ['#alerts']
            },
            EscalationLevel.L2_ENGINEERING: {
                'email': ['engineering@ma-intelligence.com'],
                'slack': ['#engineering-alerts']
            },
            EscalationLevel.L3_MANAGEMENT: {
                'email': ['management@ma-intelligence.com'],
                'slack': ['#management']
            },
            EscalationLevel.L4_EXECUTIVE: {
                'email': ['executive@ma-intelligence.com'],
                'slack': ['#executive']
            }
        }
        
        # Cooldown tracking
        self.rule_cooldowns: Dict[str, datetime] = {}
        
        self._setup_default_rules()
        
        # Thread d'escalation
        self.escalation_active = False
        self.escalation_thread: Optional[asyncio.Task] = None
        
        logger.info("🧠 Système d'alertes intelligentes initialisé")
    
    def _setup_default_rules(self):
        """Configure les règles d'alerte par défaut"""
        
        # Règles système
        self.alert_rules['high_cpu'] = AlertRule(
            name='high_cpu',
            description='Usage CPU élevé',
            alert_type=AlertType.SYSTEM,
            severity=SeverityLevel.HIGH,
            condition='system_cpu_usage',
            threshold_value=80.0,
            operator='>',
            time_window_minutes=5,
            min_occurrences=2,
            channels=[NotificationChannel.SLACK, NotificationChannel.EMAIL],
            escalation_minutes=15,
            tags={'system', 'cpu'}
        )
        
        self.alert_rules['high_memory'] = AlertRule(
            name='high_memory',
            description='Usage mémoire élevé',
            alert_type=AlertType.SYSTEM,
            severity=SeverityLevel.HIGH,
            condition='system_memory_usage',
            threshold_value=85.0,
            operator='>',
            time_window_minutes=5,
            min_occurrences=2,
            channels=[NotificationChannel.SLACK, NotificationChannel.EMAIL],
            escalation_minutes=15,
            tags={'system', 'memory'}
        )
        
        # Règles performance
        self.alert_rules['slow_api'] = AlertRule(
            name='slow_api',
            description='Temps de réponse API lent',
            alert_type=AlertType.PERFORMANCE,
            severity=SeverityLevel.MEDIUM,
            condition='http_request_duration_p95',
            threshold_value=2.0,
            operator='>',
            time_window_minutes=10,
            min_occurrences=3,
            channels=[NotificationChannel.SLACK],
            escalation_minutes=30,
            tags={'performance', 'api'}
        )
        
        # Règles business
        self.alert_rules['low_scraping_success'] = AlertRule(
            name='low_scraping_success',
            description='Taux de succès scraping faible',
            alert_type=AlertType.BUSINESS,
            severity=SeverityLevel.HIGH,
            condition='scraping_success_rate',
            threshold_value=90.0,
            operator='<',
            time_window_minutes=30,
            min_occurrences=1,
            channels=[NotificationChannel.SLACK, NotificationChannel.EMAIL],
            escalation_minutes=20,
            tags={'business', 'scraping'}
        )
        
        # Règles sécurité
        self.alert_rules['failed_authentications'] = AlertRule(
            name='failed_authentications',
            description='Tentatives d\'authentification échouées',
            alert_type=AlertType.SECURITY,
            severity=SeverityLevel.CRITICAL,
            condition='failed_auth_attempts',
            threshold_value=10.0,
            operator='>',
            time_window_minutes=5,
            min_occurrences=1,
            channels=[NotificationChannel.SLACK, NotificationChannel.EMAIL],
            escalation_minutes=5,
            tags={'security', 'auth'}
        )
    
    async def process_anomaly(self, anomaly: AnomalyDetection) -> Optional[AlertInstance]:
        """Traite une anomalie détectée"""
        
        # Chercher une règle correspondante
        matching_rule = self._find_matching_rule(anomaly.metric_name, anomaly.current_value)
        
        if not matching_rule:
            return None
        
        # Vérifier cooldown
        if self._is_in_cooldown(matching_rule.name):
            return None
        
        # Générer une alerte
        alert = self._create_alert_from_anomaly(anomaly, matching_rule)
        
        # Corrélation
        correlation_id = self.correlation_engine.correlate_alerts(
            alert, list(self.active_alerts.values())
        )
        
        if correlation_id:
            alert.correlation_id = correlation_id
            # Mettre à jour l'alerte existante plutôt que créer une nouvelle
            existing_alert = self._find_alert_by_correlation(correlation_id)
            if existing_alert:
                existing_alert.last_triggered = datetime.now()
                existing_alert.current_value = alert.current_value
                return existing_alert
        
        # Nouvelle alerte
        self.active_alerts[alert.id] = alert
        
        # Envoyer notifications
        await self._send_alert_notifications(alert, matching_rule)
        
        # Programmer escalation
        self._schedule_escalation(alert, matching_rule)
        
        # Audit log
        audit_logger.audit(
            action="alert_triggered",
            resource_type="alert",
            resource_id=alert.id,
            success=True,
            details={
                'rule': matching_rule.name,
                'severity': alert.severity.value,
                'metric': anomaly.metric_name,
                'value': anomaly.current_value
            }
        )
        
        logger.info(f"🚨 Alerte générée: {alert.title} (ID: {alert.id})")
        
        return alert
    
    def _find_matching_rule(self, metric_name: str, value: float) -> Optional[AlertRule]:
        """Trouve une règle correspondant à une métrique"""
        
        for rule in self.alert_rules.values():
            if not rule.enabled:
                continue
            
            # Correspondance basique nom de métrique
            if rule.condition in metric_name or metric_name in rule.condition:
                
                # Vérifier opérateur et seuil
                if self._evaluate_condition(value, rule.operator, rule.threshold_value):
                    return rule
        
        return None
    
    def _evaluate_condition(self, value: float, operator: str, threshold: float) -> bool:
        """Évalue une condition d'alerte"""
        if operator == '>':
            return value > threshold
        elif operator == '<':
            return value < threshold
        elif operator == '>=':
            return value >= threshold
        elif operator == '<=':
            return value <= threshold
        elif operator == '==':
            return abs(value - threshold) < 0.001
        return False
    
    def _is_in_cooldown(self, rule_name: str) -> bool:
        """Vérifie si une règle est en cooldown"""
        if rule_name not in self.rule_cooldowns:
            return False
        
        rule = self.alert_rules[rule_name]
        cooldown_until = self.rule_cooldowns[rule_name] + timedelta(minutes=rule.cooldown_minutes)
        
        return datetime.now() < cooldown_until
    
    def _create_alert_from_anomaly(self, anomaly: AnomalyDetection, rule: AlertRule) -> AlertInstance:
        """Crée une alerte à partir d'une anomalie"""
        
        alert_id = hashlib.md5(
            f"{rule.name}_{anomaly.metric_name}_{datetime.now().isoformat()}".encode()
        ).hexdigest()[:12]
        
        title = f"{rule.description} - {anomaly.metric_name}"
        
        description = (
            f"Anomalie détectée sur {anomaly.metric_name}. "
            f"Valeur actuelle: {anomaly.current_value:.2f}, "
            f"seuil: {rule.threshold_value}, "
            f"score d'anomalie: {anomaly.anomaly_score:.3f}"
        )
        
        return AlertInstance(
            id=alert_id,
            rule_name=rule.name,
            alert_type=rule.alert_type,
            severity=anomaly.severity,
            title=title,
            description=description,
            current_value=anomaly.current_value,
            threshold_value=rule.threshold_value,
            first_triggered=datetime.now(),
            last_triggered=datetime.now(),
            tags=rule.tags.copy(),
            metadata={
                'anomaly_score': anomaly.anomaly_score,
                'expected_range': anomaly.expected_range,
                'context': anomaly.context
            }
        )
    
    def _find_alert_by_correlation(self, correlation_id: str) -> Optional[AlertInstance]:
        """Trouve une alerte par ID de corrélation"""
        for alert in self.active_alerts.values():
            if alert.correlation_id == correlation_id or alert.id == correlation_id:
                return alert
        return None
    
    async def _send_alert_notifications(self, alert: AlertInstance, rule: AlertRule):
        """Envoie les notifications pour une alerte"""
        
        # Contacts du niveau d'escalation actuel
        contacts = self.escalation_contacts.get(alert.escalation_level, {})
        
        for channel in rule.channels:
            recipients = contacts.get(channel.value, [])
            
            if recipients:
                success = await self.notification_service.send_notification(
                    alert, channel, recipients
                )
                
                # Enregistrer notification
                alert.notifications_sent.append({
                    'channel': channel.value,
                    'recipients': recipients,
                    'success': success,
                    'timestamp': datetime.now().isoformat()
                })
    
    def _schedule_escalation(self, alert: AlertInstance, rule: AlertRule):
        """Programme l'escalation d'une alerte"""
        
        async def escalate_later():
            await asyncio.sleep(rule.escalation_minutes * 60)
            
            # Vérifier si l'alerte est toujours active
            if alert.id in self.active_alerts and alert.status == AlertStatus.ACTIVE:
                await self._escalate_alert(alert)
        
        # Programmer la tâche d'escalation
        asyncio.create_task(escalate_later())
    
    async def _escalate_alert(self, alert: AlertInstance):
        """Escalade une alerte au niveau supérieur"""
        
        # Déterminer le prochain niveau
        escalation_order = [
            EscalationLevel.L1_MONITORING,
            EscalationLevel.L2_ENGINEERING,
            EscalationLevel.L3_MANAGEMENT,
            EscalationLevel.L4_EXECUTIVE
        ]
        
        current_index = escalation_order.index(alert.escalation_level)
        
        if current_index < len(escalation_order) - 1:
            alert.escalation_level = escalation_order[current_index + 1]
            alert.status = AlertStatus.ESCALATED
            
            # Augmenter la sévérité
            if alert.severity == SeverityLevel.LOW:
                alert.severity = SeverityLevel.MEDIUM
            elif alert.severity == SeverityLevel.MEDIUM:
                alert.severity = SeverityLevel.HIGH
            elif alert.severity == SeverityLevel.HIGH:
                alert.severity = SeverityLevel.CRITICAL
            
            # Renvoyer notifications au nouveau niveau
            rule = self.alert_rules[alert.rule_name]
            await self._send_alert_notifications(alert, rule)
            
            logger.warning(f"📈 Alerte escaladée: {alert.id} -> {alert.escalation_level.value}")
            
            # Programmer prochaine escalation
            self._schedule_escalation(alert, rule)
    
    async def acknowledge_alert(self, alert_id: str, user_id: str, comment: str = "") -> bool:
        """Acquitte une alerte"""
        
        if alert_id not in self.active_alerts:
            return False
        
        alert = self.active_alerts[alert_id]
        alert.status = AlertStatus.ACKNOWLEDGED
        
        acknowledgment = {
            'user_id': user_id,
            'timestamp': datetime.now().isoformat(),
            'comment': comment
        }
        
        alert.acknowledgments.append(acknowledgment)
        
        logger.info(f"✅ Alerte acquittée: {alert_id} par {user_id}")
        
        return True
    
    async def resolve_alert(self, alert_id: str, user_id: str, resolution_note: str = "") -> bool:
        """Résout une alerte"""
        
        if alert_id not in self.active_alerts:
            return False
        
        alert = self.active_alerts.pop(alert_id)
        alert.status = AlertStatus.RESOLVED
        
        alert.metadata['resolution'] = {
            'user_id': user_id,
            'timestamp': datetime.now().isoformat(),
            'note': resolution_note
        }
        
        # Déplacer vers l'historique
        self.alert_history.append(alert)
        
        # Définir cooldown
        rule = self.alert_rules.get(alert.rule_name)
        if rule:
            self.rule_cooldowns[rule.name] = datetime.now()
        
        logger.info(f"✅ Alerte résolue: {alert_id} par {user_id}")
        
        return True
    
    def get_active_alerts(self, filters: Dict[str, Any] = None) -> List[AlertInstance]:
        """Retourne les alertes actives avec filtres optionnels"""
        alerts = list(self.active_alerts.values())
        
        if not filters:
            return alerts
        
        # Filtrer par type
        if 'alert_type' in filters:
            alerts = [a for a in alerts if a.alert_type == filters['alert_type']]
        
        # Filtrer par sévérité
        if 'severity' in filters:
            alerts = [a for a in alerts if a.severity == filters['severity']]
        
        # Filtrer par statut
        if 'status' in filters:
            alerts = [a for a in alerts if a.status == filters['status']]
        
        return alerts
    
    def get_alert_statistics(self) -> Dict[str, Any]:
        """Retourne les statistiques d'alertes"""
        
        active_alerts = list(self.active_alerts.values())
        total_alerts = len(active_alerts) + len(self.alert_history)
        
        # Stats par sévérité
        severity_stats = defaultdict(int)
        for alert in active_alerts:
            severity_stats[alert.severity.value] += 1
        
        # Stats par type
        type_stats = defaultdict(int)
        for alert in active_alerts:
            type_stats[alert.alert_type.value] += 1
        
        # MTTR (Mean Time To Resolution)
        resolved_alerts = [a for a in self.alert_history if a.status == AlertStatus.RESOLVED]
        if resolved_alerts:
            resolution_times = []
            for alert in resolved_alerts:
                if 'resolution' in alert.metadata:
                    resolution_time = datetime.fromisoformat(alert.metadata['resolution']['timestamp'])
                    duration = (resolution_time - alert.first_triggered).total_seconds() / 60  # minutes
                    resolution_times.append(duration)
            
            mttr = statistics.mean(resolution_times) if resolution_times else 0
        else:
            mttr = 0
        
        return {
            'active_alerts': len(active_alerts),
            'total_alerts_24h': total_alerts,
            'severity_distribution': dict(severity_stats),
            'type_distribution': dict(type_stats),
            'mttr_minutes': round(mttr, 2),
            'escalation_rate': len([a for a in active_alerts if a.status == AlertStatus.ESCALATED]) / max(len(active_alerts), 1) * 100
        }


# Instance globale
_intelligent_alerting: Optional[IntelligentAlertingSystem] = None


async def get_intelligent_alerting() -> IntelligentAlertingSystem:
    """Factory pour obtenir le système d'alertes"""
    global _intelligent_alerting
    
    if _intelligent_alerting is None:
        _intelligent_alerting = IntelligentAlertingSystem()
    
    return _intelligent_alerting


# Fonctions utilitaires

async def trigger_alert_from_anomaly(anomaly: AnomalyDetection) -> Optional[AlertInstance]:
    """Déclenche une alerte à partir d'une anomalie"""
    alerting = await get_intelligent_alerting()
    return await alerting.process_anomaly(anomaly)


async def create_custom_alert(
    title: str,
    description: str,
    alert_type: AlertType,
    severity: SeverityLevel,
    current_value: float,
    threshold_value: float,
    metadata: Dict[str, Any] = None
) -> AlertInstance:
    """Crée une alerte personnalisée"""
    
    alerting = await get_intelligent_alerting()
    
    alert_id = hashlib.md5(
        f"custom_{title}_{datetime.now().isoformat()}".encode()
    ).hexdigest()[:12]
    
    alert = AlertInstance(
        id=alert_id,
        rule_name="custom",
        alert_type=alert_type,
        severity=severity,
        title=title,
        description=description,
        current_value=current_value,
        threshold_value=threshold_value,
        first_triggered=datetime.now(),
        last_triggered=datetime.now(),
        metadata=metadata or {}
    )
    
    alerting.active_alerts[alert.id] = alert
    
    # Envoyer notification basique
    await alerting.notification_service.send_notification(
        alert,
        NotificationChannel.CONSOLE,
        []
    )
    
    return alert