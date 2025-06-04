"""
Système d'audit et de compliance pour M&A Intelligence Platform
US-007: Audit de sécurité, compliance et traçabilité

Features:
- Audit trail complet des actions
- Compliance RGPD et réglementaire
- Détection d'intrusions et anomalies
- Rapports de sécurité automatisés
- Analyse forensique et investigation
- Alertes de sécurité en temps réel
"""

import asyncio
import json
import hashlib
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import ipaddress
import re
import geoip2.database
import geoip2.errors

from fastapi import Request
from pydantic import BaseModel, validator
import pandas as pd

from app.config import settings
from app.core.logging_system import get_logger, LogCategory
from app.core.rbac_system import get_rbac_system, ResourceType, Action
from app.models.user import User

logger = get_logger("security_audit", LogCategory.SECURITY)


class AuditEventType(str, Enum):
    """Types d'événements d'audit"""
    # Authentification
    LOGIN_SUCCESS = "login_success"
    LOGIN_FAILED = "login_failed"
    LOGOUT = "logout"
    PASSWORD_CHANGE = "password_change"
    MFA_ENABLED = "mfa_enabled"
    MFA_DISABLED = "mfa_disabled"
    
    # Autorisation
    ACCESS_GRANTED = "access_granted"
    ACCESS_DENIED = "access_denied"
    PERMISSION_CHANGE = "permission_change"
    ROLE_ASSIGNED = "role_assigned"
    ROLE_REVOKED = "role_revoked"
    
    # Données
    DATA_ACCESS = "data_access"
    DATA_MODIFICATION = "data_modification"
    DATA_DELETION = "data_deletion"
    DATA_EXPORT = "data_export"
    DATA_IMPORT = "data_import"
    
    # Système
    SYSTEM_CONFIG_CHANGE = "system_config_change"
    SECURITY_ALERT = "security_alert"
    INTRUSION_DETECTED = "intrusion_detected"
    ANOMALY_DETECTED = "anomaly_detected"
    
    # Compliance
    GDPR_REQUEST = "gdpr_request"
    DATA_RETENTION_ACTION = "data_retention_action"
    CONSENT_CHANGE = "consent_change"


class SeverityLevel(str, Enum):
    """Niveaux de sévérité"""
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ComplianceFramework(str, Enum):
    """Frameworks de compliance"""
    GDPR = "gdpr"
    ISO27001 = "iso27001"
    SOC2 = "soc2"
    HIPAA = "hipaa"
    PCI_DSS = "pci_dss"


@dataclass
class AuditEvent:
    """Événement d'audit"""
    event_id: str
    event_type: AuditEventType
    timestamp: datetime
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    resource_type: Optional[ResourceType] = None
    resource_id: Optional[str] = None
    action: Optional[Action] = None
    severity: SeverityLevel = SeverityLevel.INFO
    success: bool = True
    
    # Contexte technique
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    endpoint: Optional[str] = None
    method: Optional[str] = None
    
    # Géolocalisation
    country: Optional[str] = None
    city: Optional[str] = None
    is_vpn: bool = False
    is_tor: bool = False
    
    # Détails et métadonnées
    details: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Compliance
    compliance_tags: Set[ComplianceFramework] = field(default_factory=set)
    retention_until: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire"""
        return asdict(self)
    
    def get_risk_score(self) -> float:
        """Calcule un score de risque (0-1)"""
        base_score = {
            SeverityLevel.INFO: 0.1,
            SeverityLevel.LOW: 0.3,
            SeverityLevel.MEDIUM: 0.5,
            SeverityLevel.HIGH: 0.7,
            SeverityLevel.CRITICAL: 0.9
        }.get(self.severity, 0.1)
        
        # Facteurs aggravants
        if not self.success:
            base_score += 0.2
        
        if self.is_vpn or self.is_tor:
            base_score += 0.1
        
        if self.event_type in [AuditEventType.INTRUSION_DETECTED, AuditEventType.SECURITY_ALERT]:
            base_score += 0.3
        
        return min(1.0, base_score)


@dataclass
class SecurityThreat:
    """Menace de sécurité détectée"""
    threat_id: str
    threat_type: str
    detected_at: datetime
    source_ip: Optional[str] = None
    target_resource: Optional[str] = None
    severity: SeverityLevel = SeverityLevel.MEDIUM
    confidence: float = 0.5  # 0-1
    indicators: List[str] = field(default_factory=list)
    mitigation_actions: List[str] = field(default_factory=list)
    resolved: bool = False
    resolved_at: Optional[datetime] = None


@dataclass
class ComplianceReport:
    """Rapport de compliance"""
    report_id: str
    framework: ComplianceFramework
    generated_at: datetime
    period_start: datetime
    period_end: datetime
    total_events: int
    compliant_events: int
    non_compliant_events: int
    compliance_score: float  # 0-100
    recommendations: List[str] = field(default_factory=list)
    violations: List[Dict[str, Any]] = field(default_factory=list)


class ThreatDetector:
    """Détecteur de menaces et d'intrusions"""
    
    def __init__(self):
        # Patterns d'attaque connus
        self.attack_patterns = {
            'sql_injection': [
                r"(?i)(union.*select|select.*from|insert.*into|drop.*table)",
                r"(?i)(\'+.*or.*\'+|\"+.*or.*\"+)",
                r"(?i)(;.*drop|;.*delete|;.*update)"
            ],
            'xss': [
                r"(?i)(<script.*>|javascript:|onload=|onerror=)",
                r"(?i)(alert\(|confirm\(|prompt\()",
                r"(?i)(<img.*onerror|<iframe.*src)"
            ],
            'path_traversal': [
                r"(\.\./|\.\.\\ |%2e%2e%2f|%2e%2e\\)",
                r"(etc/passwd|windows/system32|boot\.ini)"
            ],
            'command_injection': [
                r"(?i)(;.*cat|;.*ls|;.*dir|;.*type)",
                r"(\|.*cat|\|.*ls|\|.*dir|\|.*type)",
                r"(`.*cat|`.*ls|`.*dir|`.*type)"
            ]
        }
        
        # Seuils d'alerte
        self.thresholds = {
            'failed_logins_per_minute': 5,
            'failed_logins_per_hour': 20,
            'requests_per_minute': 100,
            'different_ips_per_user': 3,
            'admin_actions_per_hour': 50
        }
        
        # Cache des événements récents
        self.recent_events: Dict[str, List[AuditEvent]] = {}
        
        logger.info("🛡️ Détecteur de menaces initialisé")
    
    def analyze_event(self, event: AuditEvent) -> List[SecurityThreat]:
        """Analyse un événement pour détecter des menaces"""
        threats = []
        
        try:
            # 1. Détection d'attaques par injection
            if event.details:
                content_to_check = str(event.details)
                threats.extend(self._detect_injection_attacks(event, content_to_check))
            
            # 2. Analyse des tentatives de connexion
            if event.event_type in [AuditEventType.LOGIN_FAILED, AuditEventType.LOGIN_SUCCESS]:
                threats.extend(self._analyze_login_patterns(event))
            
            # 3. Détection d'anomalies de comportement
            if event.user_id:
                threats.extend(self._detect_behavioral_anomalies(event))
            
            # 4. Analyse géographique
            if event.ip_address:
                threats.extend(self._analyze_geographic_anomalies(event))
            
            # 5. Détection d'escalade de privilèges
            if event.event_type in [AuditEventType.ACCESS_DENIED, AuditEventType.PERMISSION_CHANGE]:
                threats.extend(self._detect_privilege_escalation(event))
            
            # Mettre à jour cache
            self._update_event_cache(event)
            
        except Exception as e:
            logger.error(f"Erreur analyse menaces pour événement {event.event_id}: {e}")
        
        return threats
    
    def _detect_injection_attacks(self, event: AuditEvent, content: str) -> List[SecurityThreat]:
        """Détecte les attaques par injection"""
        threats = []
        
        for attack_type, patterns in self.attack_patterns.items():
            for pattern in patterns:
                if re.search(pattern, content):
                    threat = SecurityThreat(
                        threat_id=f"injection_{attack_type}_{event.event_id}",
                        threat_type=f"{attack_type}_injection",
                        detected_at=datetime.now(),
                        source_ip=event.ip_address,
                        target_resource=event.resource_id,
                        severity=SeverityLevel.HIGH,
                        confidence=0.8,
                        indicators=[f"Pattern détecté: {pattern}", f"Contenu: {content[:100]}"],
                        mitigation_actions=[
                            "Bloquer l'IP source",
                            "Analyser les logs récents",
                            "Vérifier l'intégrité des données"
                        ]
                    )
                    threats.append(threat)
                    break  # Un seul threat par type d'attaque
        
        return threats
    
    def _analyze_login_patterns(self, event: AuditEvent) -> List[SecurityThreat]:
        """Analyse les patterns de connexion"""
        threats = []
        
        if not event.ip_address:
            return threats
        
        # Récupérer événements récents de connexion
        recent_logins = self._get_recent_events_by_type([
            AuditEventType.LOGIN_FAILED,
            AuditEventType.LOGIN_SUCCESS
        ], minutes=60)
        
        # Compter échecs par IP
        failures_by_ip = {}
        for login_event in recent_logins:
            if (login_event.event_type == AuditEventType.LOGIN_FAILED and 
                login_event.ip_address):
                ip = login_event.ip_address
                failures_by_ip[ip] = failures_by_ip.get(ip, 0) + 1
        
        # Attaque par force brute détectée
        if failures_by_ip.get(event.ip_address, 0) >= self.thresholds['failed_logins_per_hour']:
            threat = SecurityThreat(
                threat_id=f"brute_force_{event.ip_address}_{event.event_id}",
                threat_type="brute_force_attack",
                detected_at=datetime.now(),
                source_ip=event.ip_address,
                severity=SeverityLevel.HIGH,
                confidence=0.9,
                indicators=[
                    f"{failures_by_ip[event.ip_address]} tentatives échouées en 1h",
                    f"IP source: {event.ip_address}"
                ],
                mitigation_actions=[
                    "Bloquer temporairement l'IP",
                    "Notifier l'utilisateur cible",
                    "Analyser la source de l'attaque"
                ]
            )
            threats.append(threat)
        
        # Connexions depuis plusieurs pays
        if event.user_id and event.country:
            user_countries = set()
            for login_event in recent_logins:
                if (login_event.user_id == event.user_id and 
                    login_event.country and
                    login_event.event_type == AuditEventType.LOGIN_SUCCESS):
                    user_countries.add(login_event.country)
            
            if len(user_countries) > 2:
                threat = SecurityThreat(
                    threat_id=f"impossible_travel_{event.user_id}_{event.event_id}",
                    threat_type="impossible_travel",
                    detected_at=datetime.now(),
                    source_ip=event.ip_address,
                    severity=SeverityLevel.MEDIUM,
                    confidence=0.7,
                    indicators=[
                        f"Connexions depuis {len(user_countries)} pays: {list(user_countries)}",
                        f"Utilisateur: {event.user_id}"
                    ],
                    mitigation_actions=[
                        "Vérifier avec l'utilisateur",
                        "Demander une re-authentification",
                        "Analyser les sessions actives"
                    ]
                )
                threats.append(threat)
        
        return threats
    
    def _detect_behavioral_anomalies(self, event: AuditEvent) -> List[SecurityThreat]:
        """Détecte les anomalies comportementales"""
        threats = []
        
        if not event.user_id:
            return threats
        
        # Récupérer activité récente de l'utilisateur
        user_events = self._get_recent_events_by_user(event.user_id, hours=24)
        
        # Analyser volume d'activité
        if len(user_events) > 200:  # Seuil élevé d'activité
            threat = SecurityThreat(
                threat_id=f"unusual_activity_{event.user_id}_{event.event_id}",
                threat_type="unusual_activity_volume",
                detected_at=datetime.now(),
                source_ip=event.ip_address,
                severity=SeverityLevel.MEDIUM,
                confidence=0.6,
                indicators=[
                    f"{len(user_events)} actions en 24h",
                    f"Utilisateur: {event.user_id}"
                ],
                mitigation_actions=[
                    "Analyser le détail des actions",
                    "Vérifier avec l'utilisateur",
                    "Surveiller les prochaines actions"
                ]
            )
            threats.append(threat)
        
        # Analyser actions administratives
        admin_actions = [
            e for e in user_events
            if e.event_type in [
                AuditEventType.PERMISSION_CHANGE,
                AuditEventType.ROLE_ASSIGNED,
                AuditEventType.SYSTEM_CONFIG_CHANGE
            ]
        ]
        
        if len(admin_actions) > 10:  # Beaucoup d'actions admin
            threat = SecurityThreat(
                threat_id=f"excessive_admin_{event.user_id}_{event.event_id}",
                threat_type="excessive_admin_actions",
                detected_at=datetime.now(),
                source_ip=event.ip_address,
                severity=SeverityLevel.HIGH,
                confidence=0.8,
                indicators=[
                    f"{len(admin_actions)} actions administratives en 24h",
                    f"Types: {set(a.event_type.value for a in admin_actions)}"
                ],
                mitigation_actions=[
                    "Vérifier l'autorisation des actions",
                    "Examiner les changements effectués",
                    "Contacter l'administrateur"
                ]
            )
            threats.append(threat)
        
        return threats
    
    def _analyze_geographic_anomalies(self, event: AuditEvent) -> List[SecurityThreat]:
        """Analyse les anomalies géographiques"""
        threats = []
        
        # VPN/Tor détecté
        if event.is_vpn or event.is_tor:
            threat = SecurityThreat(
                threat_id=f"anonymizer_{event.event_id}",
                threat_type="anonymizer_usage",
                detected_at=datetime.now(),
                source_ip=event.ip_address,
                severity=SeverityLevel.MEDIUM,
                confidence=0.9,
                indicators=[
                    f"VPN: {event.is_vpn}, Tor: {event.is_tor}",
                    f"IP: {event.ip_address}"
                ],
                mitigation_actions=[
                    "Vérifier l'identité de l'utilisateur",
                    "Appliquer des contrôles supplémentaires",
                    "Surveiller les actions suivantes"
                ]
            )
            threats.append(threat)
        
        return threats
    
    def _detect_privilege_escalation(self, event: AuditEvent) -> List[SecurityThreat]:
        """Détecte les tentatives d'escalade de privilèges"""
        threats = []
        
        if event.event_type == AuditEventType.ACCESS_DENIED:
            # Compter les refus d'accès récents
            denied_events = self._get_recent_events_by_user(
                event.user_id, hours=1,
                event_types=[AuditEventType.ACCESS_DENIED]
            )
            
            if len(denied_events) >= 5:
                threat = SecurityThreat(
                    threat_id=f"privilege_escalation_{event.user_id}_{event.event_id}",
                    threat_type="privilege_escalation_attempt",
                    detected_at=datetime.now(),
                    source_ip=event.ip_address,
                    severity=SeverityLevel.HIGH,
                    confidence=0.7,
                    indicators=[
                        f"{len(denied_events)} tentatives d'accès refusées en 1h",
                        f"Ressources ciblées: {set(e.resource_type.value for e in denied_events if e.resource_type)}"
                    ],
                    mitigation_actions=[
                        "Vérifier les permissions utilisateur",
                        "Examiner les ressources ciblées",
                        "Possibilité de suspension temporaire"
                    ]
                )
                threats.append(threat)
        
        return threats
    
    def _get_recent_events_by_type(self, event_types: List[AuditEventType], 
                                  minutes: int = 60) -> List[AuditEvent]:
        """Récupère les événements récents par type"""
        cutoff = datetime.now() - timedelta(minutes=minutes)
        recent_events = []
        
        for events_list in self.recent_events.values():
            for event in events_list:
                if (event.timestamp >= cutoff and 
                    event.event_type in event_types):
                    recent_events.append(event)
        
        return recent_events
    
    def _get_recent_events_by_user(self, user_id: str, hours: int = 24,
                                  event_types: List[AuditEventType] = None) -> List[AuditEvent]:
        """Récupère les événements récents d'un utilisateur"""
        cutoff = datetime.now() - timedelta(hours=hours)
        user_events = []
        
        events_list = self.recent_events.get(user_id, [])
        for event in events_list:
            if event.timestamp >= cutoff:
                if event_types is None or event.event_type in event_types:
                    user_events.append(event)
        
        return user_events
    
    def _update_event_cache(self, event: AuditEvent):
        """Met à jour le cache des événements récents"""
        if not event.user_id:
            return
        
        if event.user_id not in self.recent_events:
            self.recent_events[event.user_id] = []
        
        self.recent_events[event.user_id].append(event)
        
        # Limiter la taille du cache (garder dernières 24h)
        cutoff = datetime.now() - timedelta(hours=24)
        self.recent_events[event.user_id] = [
            e for e in self.recent_events[event.user_id]
            if e.timestamp >= cutoff
        ]
        
        # Limiter le nombre total d'utilisateurs en cache
        if len(self.recent_events) > 1000:
            # Supprimer les utilisateurs les moins actifs
            user_activity = {
                user_id: len(events)
                for user_id, events in self.recent_events.items()
            }
            
            least_active = sorted(user_activity.items(), key=lambda x: x[1])[:100]
            for user_id, _ in least_active:
                del self.recent_events[user_id]


class ComplianceManager:
    """Gestionnaire de compliance"""
    
    def __init__(self):
        self.frameworks = {
            ComplianceFramework.GDPR: {
                'name': 'Règlement Général sur la Protection des Données',
                'retention_default': timedelta(days=2555),  # 7 ans
                'required_events': [
                    AuditEventType.DATA_ACCESS,
                    AuditEventType.DATA_MODIFICATION,
                    AuditEventType.DATA_DELETION,
                    AuditEventType.GDPR_REQUEST,
                    AuditEventType.CONSENT_CHANGE
                ]
            },
            ComplianceFramework.ISO27001: {
                'name': 'ISO/IEC 27001',
                'retention_default': timedelta(days=2190),  # 6 ans
                'required_events': [
                    AuditEventType.LOGIN_SUCCESS,
                    AuditEventType.LOGIN_FAILED,
                    AuditEventType.ACCESS_DENIED,
                    AuditEventType.SECURITY_ALERT,
                    AuditEventType.SYSTEM_CONFIG_CHANGE
                ]
            }
        }
        
        logger.info("📋 Gestionnaire de compliance initialisé")
    
    def tag_event_for_compliance(self, event: AuditEvent) -> AuditEvent:
        """Tag un événement selon les frameworks de compliance"""
        
        for framework, config in self.frameworks.items():
            if event.event_type in config['required_events']:
                event.compliance_tags.add(framework)
                
                # Définir période de rétention
                if not event.retention_until:
                    event.retention_until = event.timestamp + config['retention_default']
        
        return event
    
    def check_compliance_violations(self, event: AuditEvent) -> List[Dict[str, Any]]:
        """Vérifie les violations de compliance"""
        violations = []
        
        # RGPD - Accès non autorisé aux données personnelles
        if (ComplianceFramework.GDPR in event.compliance_tags and
            event.event_type == AuditEventType.DATA_ACCESS and
            not event.success):
            
            violations.append({
                'framework': ComplianceFramework.GDPR.value,
                'violation_type': 'unauthorized_personal_data_access',
                'description': 'Tentative d\'accès non autorisé aux données personnelles',
                'severity': 'high',
                'event_id': event.event_id
            })
        
        # ISO27001 - Échecs de connexion répétés
        if (ComplianceFramework.ISO27001 in event.compliance_tags and
            event.event_type == AuditEventType.LOGIN_FAILED):
            
            # Cette vérification serait normalement plus sophistiquée
            violations.append({
                'framework': ComplianceFramework.ISO27001.value,
                'violation_type': 'repeated_authentication_failures',
                'description': 'Échecs d\'authentification répétés',
                'severity': 'medium',
                'event_id': event.event_id
            })
        
        return violations
    
    def generate_compliance_report(self, framework: ComplianceFramework,
                                 period_start: datetime,
                                 period_end: datetime,
                                 events: List[AuditEvent]) -> ComplianceReport:
        """Génère un rapport de compliance"""
        
        # Filtrer événements pertinents
        relevant_events = [
            e for e in events
            if framework in e.compliance_tags and
            period_start <= e.timestamp <= period_end
        ]
        
        # Analyser compliance
        compliant_events = [e for e in relevant_events if e.success]
        non_compliant_events = [e for e in relevant_events if not e.success]
        
        compliance_score = (
            len(compliant_events) / len(relevant_events) * 100
            if relevant_events else 100
        )
        
        # Recommandations basées sur l'analyse
        recommendations = self._generate_compliance_recommendations(
            framework, relevant_events
        )
        
        # Violations détectées
        violations = []
        for event in relevant_events:
            violations.extend(self.check_compliance_violations(event))
        
        report = ComplianceReport(
            report_id=f"{framework.value}_{period_start.strftime('%Y%m%d')}_{period_end.strftime('%Y%m%d')}",
            framework=framework,
            generated_at=datetime.now(),
            period_start=period_start,
            period_end=period_end,
            total_events=len(relevant_events),
            compliant_events=len(compliant_events),
            non_compliant_events=len(non_compliant_events),
            compliance_score=compliance_score,
            recommendations=recommendations,
            violations=violations
        )
        
        return report
    
    def _generate_compliance_recommendations(self, framework: ComplianceFramework,
                                          events: List[AuditEvent]) -> List[str]:
        """Génère des recommandations de compliance"""
        recommendations = []
        
        if framework == ComplianceFramework.GDPR:
            # Analyser les accès aux données
            data_access_events = [
                e for e in events
                if e.event_type == AuditEventType.DATA_ACCESS
            ]
            
            if len(data_access_events) > 1000:
                recommendations.append(
                    "Volume élevé d'accès aux données - considérer l'anonymisation"
                )
            
            # Vérifier les exports de données
            export_events = [
                e for e in events
                if e.event_type == AuditEventType.DATA_EXPORT
            ]
            
            if len(export_events) > 50:
                recommendations.append(
                    "Nombreux exports de données - renforcer les contrôles d'autorisation"
                )
        
        elif framework == ComplianceFramework.ISO27001:
            # Analyser les échecs de sécurité
            security_failures = [
                e for e in events
                if e.event_type in [
                    AuditEventType.LOGIN_FAILED,
                    AuditEventType.ACCESS_DENIED,
                    AuditEventType.INTRUSION_DETECTED
                ]
            ]
            
            failure_rate = len(security_failures) / len(events) if events else 0
            
            if failure_rate > 0.1:  # Plus de 10% d'échecs
                recommendations.append(
                    "Taux d'échecs de sécurité élevé - revoir les politiques d'accès"
                )
        
        return recommendations


class SecurityAuditSystem:
    """Système d'audit de sécurité principal"""
    
    def __init__(self):
        self.threat_detector = ThreatDetector()
        self.compliance_manager = ComplianceManager()
        
        # Stockage des événements (en production: base de données)
        self.audit_events: List[AuditEvent] = []
        self.security_threats: List[SecurityThreat] = []
        
        # Configuration
        self.max_events_in_memory = 10000
        self.auto_cleanup_enabled = True
        
        logger.info("🔒 Système d'audit de sécurité initialisé")
    
    async def log_event(self, event_type: AuditEventType, 
                       user_id: str = None,
                       session_id: str = None,
                       resource_type: ResourceType = None,
                       resource_id: str = None,
                       action: Action = None,
                       success: bool = True,
                       severity: SeverityLevel = SeverityLevel.INFO,
                       request: Request = None,
                       details: Dict[str, Any] = None,
                       metadata: Dict[str, Any] = None) -> str:
        """Enregistre un événement d'audit"""
        
        try:
            # Générer ID unique
            event_id = hashlib.md5(
                f"{event_type.value}_{datetime.now().isoformat()}_{user_id}".encode()
            ).hexdigest()[:16]
            
            # Créer événement
            event = AuditEvent(
                event_id=event_id,
                event_type=event_type,
                timestamp=datetime.now(),
                user_id=user_id,
                session_id=session_id,
                resource_type=resource_type,
                resource_id=resource_id,
                action=action,
                severity=severity,
                success=success,
                details=details or {},
                metadata=metadata or {}
            )
            
            # Enrichir avec informations de requête
            if request:
                event.ip_address = request.client.host if request.client else None
                event.user_agent = request.headers.get('user-agent', '')
                event.endpoint = request.url.path
                event.method = request.method
                
                # Géolocalisation (simulation)
                if event.ip_address:
                    await self._enrich_with_geolocation(event)
            
            # Tag pour compliance
            event = self.compliance_manager.tag_event_for_compliance(event)
            
            # Analyser menaces
            threats = self.threat_detector.analyze_event(event)
            if threats:
                self.security_threats.extend(threats)
                
                # Log des menaces détectées
                for threat in threats:
                    logger.warning(
                        f"🚨 Menace détectée: {threat.threat_type} "
                        f"(confiance: {threat.confidence:.2f})"
                    )
            
            # Sauvegarder événement
            self.audit_events.append(event)
            
            # Nettoyage automatique
            if self.auto_cleanup_enabled and len(self.audit_events) > self.max_events_in_memory:
                await self._cleanup_old_events()
            
            logger.debug(f"📝 Événement d'audit enregistré: {event_id}")
            
            return event_id
            
        except Exception as e:
            logger.error(f"Erreur enregistrement événement audit: {e}")
            return ""
    
    async def _enrich_with_geolocation(self, event: AuditEvent):
        """Enrichit un événement avec des informations de géolocalisation"""
        
        try:
            if not event.ip_address:
                return
            
            # Simulation de géolocalisation
            # En production, utiliser une vraie base GeoIP2
            
            # Détection VPN/Tor basique
            if event.ip_address.startswith('10.') or event.ip_address.startswith('192.168.'):
                event.country = "Local"
                event.city = "Local"
            elif event.ip_address.startswith('185.'):
                event.country = "France"
                event.city = "Paris"
            elif event.ip_address.startswith('8.8.'):
                event.country = "United States"
                event.city = "Mountain View"
            else:
                event.country = "Unknown"
                event.city = "Unknown"
            
            # Simulation détection VPN/Tor
            known_vpn_ranges = ['198.', '203.', '172.']
            known_tor_ranges = ['199.']
            
            event.is_vpn = any(event.ip_address.startswith(r) for r in known_vpn_ranges)
            event.is_tor = any(event.ip_address.startswith(r) for r in known_tor_ranges)
            
        except Exception as e:
            logger.error(f"Erreur enrichissement géolocalisation: {e}")
    
    async def _cleanup_old_events(self):
        """Nettoie les anciens événements"""
        
        try:
            # Garder événements des 30 derniers jours
            cutoff_date = datetime.now() - timedelta(days=30)
            
            initial_count = len(self.audit_events)
            self.audit_events = [
                event for event in self.audit_events
                if event.timestamp >= cutoff_date
            ]
            
            cleaned_count = initial_count - len(self.audit_events)
            
            if cleaned_count > 0:
                logger.info(f"🧹 Nettoyage audit: {cleaned_count} événements supprimés")
            
        except Exception as e:
            logger.error(f"Erreur nettoyage événements: {e}")
    
    def get_events_for_user(self, user_id: str, 
                           start_date: datetime = None,
                           end_date: datetime = None,
                           event_types: List[AuditEventType] = None) -> List[AuditEvent]:
        """Récupère les événements pour un utilisateur"""
        
        events = [e for e in self.audit_events if e.user_id == user_id]
        
        if start_date:
            events = [e for e in events if e.timestamp >= start_date]
        
        if end_date:
            events = [e for e in events if e.timestamp <= end_date]
        
        if event_types:
            events = [e for e in events if e.event_type in event_types]
        
        return sorted(events, key=lambda x: x.timestamp, reverse=True)
    
    def get_security_summary(self, days: int = 7) -> Dict[str, Any]:
        """Génère un résumé de sécurité"""
        
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_events = [e for e in self.audit_events if e.timestamp >= cutoff_date]
        recent_threats = [t for t in self.security_threats if t.detected_at >= cutoff_date]
        
        # Statistiques par type d'événement
        event_stats = {}
        for event in recent_events:
            event_type = event.event_type.value
            event_stats[event_type] = event_stats.get(event_type, 0) + 1
        
        # Statistiques par sévérité
        severity_stats = {}
        for event in recent_events:
            severity = event.severity.value
            severity_stats[severity] = severity_stats.get(severity, 0) + 1
        
        # Top IPs
        ip_stats = {}
        for event in recent_events:
            if event.ip_address:
                ip_stats[event.ip_address] = ip_stats.get(event.ip_address, 0) + 1
        
        top_ips = sorted(ip_stats.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Menaces par type
        threat_stats = {}
        for threat in recent_threats:
            threat_type = threat.threat_type
            threat_stats[threat_type] = threat_stats.get(threat_type, 0) + 1
        
        return {
            'period_days': days,
            'total_events': len(recent_events),
            'total_threats': len(recent_threats),
            'event_by_type': event_stats,
            'event_by_severity': severity_stats,
            'top_source_ips': top_ips,
            'threats_by_type': threat_stats,
            'success_rate': len([e for e in recent_events if e.success]) / len(recent_events) if recent_events else 0
        }
    
    def export_audit_log(self, start_date: datetime, end_date: datetime,
                        format: str = 'json') -> Union[str, Dict[str, Any]]:
        """Exporte le log d'audit"""
        
        events = [
            e for e in self.audit_events
            if start_date <= e.timestamp <= end_date
        ]
        
        if format == 'json':
            return {
                'export_date': datetime.now().isoformat(),
                'period_start': start_date.isoformat(),
                'period_end': end_date.isoformat(),
                'total_events': len(events),
                'events': [event.to_dict() for event in events]
            }
        
        elif format == 'csv':
            import io
            import csv
            
            output = io.StringIO()
            writer = csv.writer(output)
            
            # Headers
            headers = [
                'event_id', 'event_type', 'timestamp', 'user_id', 'session_id',
                'resource_type', 'resource_id', 'action', 'severity', 'success',
                'ip_address', 'user_agent', 'country', 'city'
            ]
            writer.writerow(headers)
            
            # Data
            for event in events:
                row = [
                    event.event_id,
                    event.event_type.value,
                    event.timestamp.isoformat(),
                    event.user_id,
                    event.session_id,
                    event.resource_type.value if event.resource_type else '',
                    event.resource_id,
                    event.action.value if event.action else '',
                    event.severity.value,
                    event.success,
                    event.ip_address,
                    event.user_agent,
                    event.country,
                    event.city
                ]
                writer.writerow(row)
            
            return output.getvalue()
        
        else:
            raise ValueError(f"Format non supporté: {format}")


# Instance globale
_security_audit_system: Optional[SecurityAuditSystem] = None


def get_security_audit_system() -> SecurityAuditSystem:
    """Factory pour obtenir le système d'audit de sécurité"""
    global _security_audit_system
    
    if _security_audit_system is None:
        _security_audit_system = SecurityAuditSystem()
    
    return _security_audit_system


# Décorateurs et utilitaires

def audit_action(event_type: AuditEventType, 
                resource_type: ResourceType = None,
                action: Action = None,
                severity: SeverityLevel = SeverityLevel.INFO):
    """Décorateur pour auditer une action"""
    
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            audit_system = get_security_audit_system()
            
            # Extraire contexte
            current_user = kwargs.get('current_user')
            request = kwargs.get('request')
            
            # Exécuter fonction
            try:
                result = await func(*args, **kwargs)
                success = True
                details = {'result_type': type(result).__name__}
            except Exception as e:
                success = False
                details = {'error': str(e)}
                raise
            finally:
                # Enregistrer événement
                await audit_system.log_event(
                    event_type=event_type,
                    user_id=current_user.id if current_user else None,
                    resource_type=resource_type,
                    action=action,
                    success=success,
                    severity=severity,
                    request=request,
                    details=details
                )
            
            return result
        
        return wrapper
    return decorator