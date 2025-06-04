"""
Fonctionnalités de sécurité avancées pour M&A Intelligence Platform
US-007: WAF, détection d'intrusions, protection DDoS et sécurité renforcée

Features:
- Web Application Firewall (WAF) intégré
- Détection d'intrusions en temps réel
- Protection anti-DDoS et rate limiting avancé
- Headers de sécurité HTTP
- Validation et sanitisation stricte
- Monitoring de sécurité en temps réel
"""

import asyncio
import re
import time
import ipaddress
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import json
import hashlib

from fastapi import Request, Response, HTTPException, status
from fastapi.middleware.base import BaseHTTPMiddleware
from starlette.middleware.base import BaseHTTPMiddleware as StarletteBaseHTTPMiddleware
from starlette.responses import JSONResponse
import user_agents

from app.config import settings
from app.core.logging_system import get_logger, LogCategory
from app.core.security_audit import get_security_audit_system, AuditEventType, SeverityLevel

logger = get_logger("advanced_security", LogCategory.SECURITY)


class ThreatLevel(str, Enum):
    """Niveaux de menace"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AttackType(str, Enum):
    """Types d'attaques détectées"""
    SQL_INJECTION = "sql_injection"
    XSS = "xss"
    PATH_TRAVERSAL = "path_traversal"
    COMMAND_INJECTION = "command_injection"
    XXSS = "xxss"
    CSRF = "csrf"
    DDOS = "ddos"
    BRUTE_FORCE = "brute_force"
    BOT_ATTACK = "bot_attack"
    MALFORMED_REQUEST = "malformed_request"
    SUSPICIOUS_PAYLOAD = "suspicious_payload"


class SecurityAction(str, Enum):
    """Actions de sécurité"""
    ALLOW = "allow"
    BLOCK = "block"
    RATE_LIMIT = "rate_limit"
    CHALLENGE = "challenge"
    LOG_ONLY = "log_only"


@dataclass
class SecurityRule:
    """Règle de sécurité WAF"""
    rule_id: str
    name: str
    description: str
    pattern: str
    attack_type: AttackType
    threat_level: ThreatLevel
    action: SecurityAction
    enabled: bool = True
    score: int = 10  # Score de menace (1-100)
    
    def matches(self, content: str) -> bool:
        """Vérifie si le contenu correspond à la règle"""
        try:
            return bool(re.search(self.pattern, content, re.IGNORECASE | re.MULTILINE))
        except re.error:
            logger.error(f"Erreur regex dans règle {self.rule_id}: {self.pattern}")
            return False


@dataclass
class SecurityEvent:
    """Événement de sécurité détecté"""
    event_id: str
    timestamp: datetime
    source_ip: str
    user_agent: str
    request_path: str
    request_method: str
    attack_type: AttackType
    threat_level: ThreatLevel
    rule_id: str
    matched_content: str
    action_taken: SecurityAction
    score: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IPStats:
    """Statistiques d'une adresse IP"""
    ip_address: str
    first_seen: datetime
    last_seen: datetime
    request_count: int = 0
    blocked_count: int = 0
    threat_score: float = 0.0
    user_agents: Set[str] = field(default_factory=set)
    endpoints: Set[str] = field(default_factory=set)
    countries: Set[str] = field(default_factory=set)
    is_whitelisted: bool = False
    is_blacklisted: bool = False


class WAFEngine:
    """Moteur Web Application Firewall"""
    
    def __init__(self):
        self.rules: Dict[str, SecurityRule] = {}
        self.ip_whitelist: Set[str] = set()
        self.ip_blacklist: Set[str] = set()
        
        # Configuration
        self.max_request_size = 10 * 1024 * 1024  # 10MB
        self.max_header_size = 8192  # 8KB
        self.max_url_length = 2048
        
        # Initialiser règles par défaut
        self._load_default_rules()
        
        logger.info("🔥 Moteur WAF initialisé")
    
    def _load_default_rules(self):
        """Charge les règles de sécurité par défaut"""
        
        # Règles SQL Injection
        sql_rules = [
            SecurityRule(
                rule_id="SQL001",
                name="SQL Injection - Union Select",
                description="Détecte les tentatives d'injection SQL avec UNION SELECT",
                pattern=r"(?i)(union\s+select|union\s+all\s+select)",
                attack_type=AttackType.SQL_INJECTION,
                threat_level=ThreatLevel.HIGH,
                action=SecurityAction.BLOCK,
                score=80
            ),
            SecurityRule(
                rule_id="SQL002",
                name="SQL Injection - Comments",
                description="Détecte les commentaires SQL malveillants",
                pattern=r"(?i)(\/\*|\*\/|--|\#)",
                attack_type=AttackType.SQL_INJECTION,
                threat_level=ThreatLevel.MEDIUM,
                action=SecurityAction.BLOCK,
                score=60
            ),
            SecurityRule(
                rule_id="SQL003",
                name="SQL Injection - Functions",
                description="Détecte les fonctions SQL dangereuses",
                pattern=r"(?i)(exec\s*\(|sp_|xp_|cmdshell)",
                attack_type=AttackType.SQL_INJECTION,
                threat_level=ThreatLevel.CRITICAL,
                action=SecurityAction.BLOCK,
                score=95
            )
        ]
        
        # Règles XSS
        xss_rules = [
            SecurityRule(
                rule_id="XSS001",
                name="XSS - Script Tags",
                description="Détecte les balises script malveillantes",
                pattern=r"(?i)<\s*script[^>]*>",
                attack_type=AttackType.XSS,
                threat_level=ThreatLevel.HIGH,
                action=SecurityAction.BLOCK,
                score=85
            ),
            SecurityRule(
                rule_id="XSS002",
                name="XSS - JavaScript Events",
                description="Détecte les événements JavaScript malveillants",
                pattern=r"(?i)(onload|onerror|onclick|onmouseover)\s*=",
                attack_type=AttackType.XSS,
                threat_level=ThreatLevel.HIGH,
                action=SecurityAction.BLOCK,
                score=75
            ),
            SecurityRule(
                rule_id="XSS003",
                name="XSS - JavaScript URLs",
                description="Détecte les URLs JavaScript malveillantes",
                pattern=r"(?i)javascript\s*:",
                attack_type=AttackType.XSS,
                threat_level=ThreatLevel.MEDIUM,
                action=SecurityAction.BLOCK,
                score=65
            )
        ]
        
        # Règles Path Traversal
        path_rules = [
            SecurityRule(
                rule_id="PATH001",
                name="Path Traversal - Directory Traversal",
                description="Détecte les tentatives de traversée de répertoires",
                pattern=r"(\.\./|\.\.\\ |%2e%2e%2f|%2e%2e\\)",
                attack_type=AttackType.PATH_TRAVERSAL,
                threat_level=ThreatLevel.HIGH,
                action=SecurityAction.BLOCK,
                score=80
            ),
            SecurityRule(
                rule_id="PATH002",
                name="Path Traversal - System Files",
                description="Détecte l'accès aux fichiers système",
                pattern=r"(?i)(etc/passwd|windows/system32|boot\.ini|web\.config)",
                attack_type=AttackType.PATH_TRAVERSAL,
                threat_level=ThreatLevel.CRITICAL,
                action=SecurityAction.BLOCK,
                score=90
            )
        ]
        
        # Règles Command Injection
        cmd_rules = [
            SecurityRule(
                rule_id="CMD001",
                name="Command Injection - Shell Commands",
                description="Détecte les tentatives d'injection de commandes",
                pattern=r"(?i)(;|\|)(cat|ls|dir|type|net|ping|wget|curl)",
                attack_type=AttackType.COMMAND_INJECTION,
                threat_level=ThreatLevel.CRITICAL,
                action=SecurityAction.BLOCK,
                score=95
            ),
            SecurityRule(
                rule_id="CMD002",
                name="Command Injection - Backticks",
                description="Détecte l'exécution de commandes avec backticks",
                pattern=r"`[^`]+`",
                attack_type=AttackType.COMMAND_INJECTION,
                threat_level=ThreatLevel.HIGH,
                action=SecurityAction.BLOCK,
                score=85
            )
        ]
        
        # Règles Bot Detection
        bot_rules = [
            SecurityRule(
                rule_id="BOT001",
                name="Malicious Bots - Common Patterns",
                description="Détecte les bots malveillants connus",
                pattern=r"(?i)(sqlmap|nmap|nikto|dirb|masscan|zap)",
                attack_type=AttackType.BOT_ATTACK,
                threat_level=ThreatLevel.HIGH,
                action=SecurityAction.BLOCK,
                score=75
            ),
            SecurityRule(
                rule_id="BOT002",
                name="Scanning Tools",
                description="Détecte les outils de scan",
                pattern=r"(?i)(burp|acunetix|nessus|openvas|w3af)",
                attack_type=AttackType.BOT_ATTACK,
                threat_level=ThreatLevel.HIGH,
                action=SecurityAction.BLOCK,
                score=80
            )
        ]
        
        # Ajouter toutes les règles
        all_rules = sql_rules + xss_rules + path_rules + cmd_rules + bot_rules
        
        for rule in all_rules:
            self.rules[rule.rule_id] = rule
        
        logger.info(f"✅ {len(all_rules)} règles WAF chargées")
    
    def add_ip_to_whitelist(self, ip_address: str):
        """Ajoute une IP à la whitelist"""
        try:
            ipaddress.ip_address(ip_address)
            self.ip_whitelist.add(ip_address)
            logger.info(f"✅ IP ajoutée à la whitelist: {ip_address}")
        except ValueError:
            logger.error(f"Adresse IP invalide pour whitelist: {ip_address}")
    
    def add_ip_to_blacklist(self, ip_address: str):
        """Ajoute une IP à la blacklist"""
        try:
            ipaddress.ip_address(ip_address)
            self.ip_blacklist.add(ip_address)
            logger.info(f"🚫 IP ajoutée à la blacklist: {ip_address}")
        except ValueError:
            logger.error(f"Adresse IP invalide pour blacklist: {ip_address}")
    
    def is_ip_whitelisted(self, ip_address: str) -> bool:
        """Vérifie si une IP est dans la whitelist"""
        return ip_address in self.ip_whitelist
    
    def is_ip_blacklisted(self, ip_address: str) -> bool:
        """Vérifie si une IP est dans la blacklist"""
        return ip_address in self.ip_blacklist
    
    def analyze_request(self, request_content: str, 
                       request_headers: Dict[str, str],
                       request_path: str,
                       request_method: str,
                       source_ip: str) -> Tuple[List[SecurityEvent], int]:
        """Analyse une requête contre les règles WAF"""
        
        events = []
        total_score = 0
        
        # Vérifier IP blacklistée
        if self.is_ip_blacklisted(source_ip):
            event = SecurityEvent(
                event_id=f"waf_{datetime.now().timestamp()}",
                timestamp=datetime.now(),
                source_ip=source_ip,
                user_agent=request_headers.get('user-agent', ''),
                request_path=request_path,
                request_method=request_method,
                attack_type=AttackType.DDOS,
                threat_level=ThreatLevel.CRITICAL,
                rule_id="BLACKLIST",
                matched_content="IP blacklistée",
                action_taken=SecurityAction.BLOCK,
                score=100
            )
            events.append(event)
            total_score += 100
        
        # Vérifier taille de la requête
        if len(request_content) > self.max_request_size:
            event = SecurityEvent(
                event_id=f"waf_{datetime.now().timestamp()}",
                timestamp=datetime.now(),
                source_ip=source_ip,
                user_agent=request_headers.get('user-agent', ''),
                request_path=request_path,
                request_method=request_method,
                attack_type=AttackType.MALFORMED_REQUEST,
                threat_level=ThreatLevel.MEDIUM,
                rule_id="SIZE001",
                matched_content="Requête trop volumineux",
                action_taken=SecurityAction.BLOCK,
                score=50
            )
            events.append(event)
            total_score += 50
        
        # Vérifier longueur URL
        if len(request_path) > self.max_url_length:
            event = SecurityEvent(
                event_id=f"waf_{datetime.now().timestamp()}",
                timestamp=datetime.now(),
                source_ip=source_ip,
                user_agent=request_headers.get('user-agent', ''),
                request_path=request_path,
                request_method=request_method,
                attack_type=AttackType.MALFORMED_REQUEST,
                threat_level=ThreatLevel.MEDIUM,
                rule_id="SIZE002",
                matched_content="URL trop longue",
                action_taken=SecurityAction.BLOCK,
                score=40
            )
            events.append(event)
            total_score += 40
        
        # Analyser contre les règles
        content_to_analyze = f"{request_path} {request_content}"
        
        for rule in self.rules.values():
            if not rule.enabled:
                continue
            
            if rule.matches(content_to_analyze):
                event = SecurityEvent(
                    event_id=f"waf_{datetime.now().timestamp()}_{rule.rule_id}",
                    timestamp=datetime.now(),
                    source_ip=source_ip,
                    user_agent=request_headers.get('user-agent', ''),
                    request_path=request_path,
                    request_method=request_method,
                    attack_type=rule.attack_type,
                    threat_level=rule.threat_level,
                    rule_id=rule.rule_id,
                    matched_content=content_to_analyze[:200],  # Limiter taille
                    action_taken=rule.action,
                    score=rule.score
                )
                events.append(event)
                total_score += rule.score
        
        # Analyser User-Agent
        user_agent = request_headers.get('user-agent', '')
        if user_agent:
            ua_events = self._analyze_user_agent(
                user_agent, source_ip, request_path, request_method
            )
            events.extend(ua_events)
            total_score += sum(e.score for e in ua_events)
        
        return events, total_score
    
    def _analyze_user_agent(self, user_agent: str, source_ip: str, 
                          request_path: str, request_method: str) -> List[SecurityEvent]:
        """Analyse le User-Agent pour détecter des anomalies"""
        
        events = []
        
        # User-Agent vide ou suspect
        if not user_agent or len(user_agent) < 10:
            event = SecurityEvent(
                event_id=f"waf_{datetime.now().timestamp()}_ua",
                timestamp=datetime.now(),
                source_ip=source_ip,
                user_agent=user_agent,
                request_path=request_path,
                request_method=request_method,
                attack_type=AttackType.BOT_ATTACK,
                threat_level=ThreatLevel.MEDIUM,
                rule_id="UA001",
                matched_content=f"User-Agent suspect: {user_agent}",
                action_taken=SecurityAction.LOG_ONLY,
                score=30
            )
            events.append(event)
        
        # Vérifier patterns de bots malveillants dans User-Agent
        malicious_patterns = [
            r"(?i)(python|curl|wget|scanner|bot)",
            r"(?i)(sqlmap|nmap|nikto|dirb)",
            r"(?i)(masscan|zap|burp|acunetix)"
        ]
        
        for pattern in malicious_patterns:
            if re.search(pattern, user_agent):
                event = SecurityEvent(
                    event_id=f"waf_{datetime.now().timestamp()}_ua_bot",
                    timestamp=datetime.now(),
                    source_ip=source_ip,
                    user_agent=user_agent,
                    request_path=request_path,
                    request_method=request_method,
                    attack_type=AttackType.BOT_ATTACK,
                    threat_level=ThreatLevel.HIGH,
                    rule_id="UA002",
                    matched_content=f"User-Agent malveillant détecté",
                    action_taken=SecurityAction.BLOCK,
                    score=70
                )
                events.append(event)
                break
        
        return events


class DDoSProtection:
    """Protection anti-DDoS avec rate limiting avancé"""
    
    def __init__(self):
        # Configuration rate limiting
        self.rate_limits = {
            'requests_per_second': 10,
            'requests_per_minute': 100,
            'requests_per_hour': 1000,
            'burst_threshold': 50,  # Pic de requêtes
            'burst_window': 10,     # Fenêtre pour burst (secondes)
        }
        
        # Compteurs par IP
        self.ip_counters: Dict[str, Dict[str, deque]] = defaultdict(lambda: {
            'requests_per_second': deque(),
            'requests_per_minute': deque(),
            'requests_per_hour': deque()
        })
        
        # Détection de patterns d'attaque
        self.attack_patterns = {
            'slowloris': {'connection_time': 300, 'requests_threshold': 5},
            'http_flood': {'requests_per_second': 50},
            'bandwidth_exhaustion': {'bytes_per_second': 10 * 1024 * 1024}  # 10MB/s
        }
        
        # IPs temporairement bloquées
        self.temp_blocked_ips: Dict[str, datetime] = {}
        self.block_duration = timedelta(minutes=15)
        
        logger.info("🛡️ Protection DDoS initialisée")
    
    def is_rate_limited(self, ip_address: str) -> Tuple[bool, str]:
        """Vérifie si une IP est rate limitée"""
        
        now = time.time()
        current_time = datetime.now()
        
        # Vérifier blocage temporaire
        if ip_address in self.temp_blocked_ips:
            block_until = self.temp_blocked_ips[ip_address]
            if current_time < block_until:
                remaining = (block_until - current_time).seconds
                return True, f"IP bloquée temporairement ({remaining}s restantes)"
            else:
                # Débloquer IP
                del self.temp_blocked_ips[ip_address]
        
        # Nettoyer compteurs anciens
        self._cleanup_counters(ip_address, now)
        
        # Ajouter requête actuelle
        counters = self.ip_counters[ip_address]
        counters['requests_per_second'].append(now)
        counters['requests_per_minute'].append(now)
        counters['requests_per_hour'].append(now)
        
        # Vérifier limites
        if len(counters['requests_per_second']) > self.rate_limits['requests_per_second']:
            self._block_ip_temporarily(ip_address)
            return True, "Limite par seconde dépassée"
        
        if len(counters['requests_per_minute']) > self.rate_limits['requests_per_minute']:
            self._block_ip_temporarily(ip_address)
            return True, "Limite par minute dépassée"
        
        if len(counters['requests_per_hour']) > self.rate_limits['requests_per_hour']:
            self._block_ip_temporarily(ip_address)
            return True, "Limite par heure dépassée"
        
        # Vérifier burst
        burst_count = len([
            t for t in counters['requests_per_second'] 
            if now - t <= self.rate_limits['burst_window']
        ])
        
        if burst_count > self.rate_limits['burst_threshold']:
            self._block_ip_temporarily(ip_address)
            return True, "Burst détecté"
        
        return False, ""
    
    def _cleanup_counters(self, ip_address: str, current_time: float):
        """Nettoie les compteurs anciens"""
        
        counters = self.ip_counters[ip_address]
        
        # Nettoyer requêtes par seconde (garder dernière seconde)
        while (counters['requests_per_second'] and 
               current_time - counters['requests_per_second'][0] > 1):
            counters['requests_per_second'].popleft()
        
        # Nettoyer requêtes par minute (garder dernière minute)
        while (counters['requests_per_minute'] and 
               current_time - counters['requests_per_minute'][0] > 60):
            counters['requests_per_minute'].popleft()
        
        # Nettoyer requêtes par heure (garder dernière heure)
        while (counters['requests_per_hour'] and 
               current_time - counters['requests_per_hour'][0] > 3600):
            counters['requests_per_hour'].popleft()
    
    def _block_ip_temporarily(self, ip_address: str):
        """Bloque temporairement une IP"""
        
        block_until = datetime.now() + self.block_duration
        self.temp_blocked_ips[ip_address] = block_until
        
        logger.warning(f"🚫 IP bloquée temporairement: {ip_address} jusqu'à {block_until}")
    
    def get_ip_stats(self, ip_address: str) -> Dict[str, Any]:
        """Retourne les statistiques d'une IP"""
        
        counters = self.ip_counters.get(ip_address, {})
        
        return {
            'ip_address': ip_address,
            'requests_last_second': len(counters.get('requests_per_second', [])),
            'requests_last_minute': len(counters.get('requests_per_minute', [])),
            'requests_last_hour': len(counters.get('requests_per_hour', [])),
            'is_blocked': ip_address in self.temp_blocked_ips,
            'blocked_until': self.temp_blocked_ips.get(ip_address, '').isoformat() if ip_address in self.temp_blocked_ips else None
        }


class SecurityHeadersManager:
    """Gestionnaire des headers de sécurité HTTP"""
    
    def __init__(self):
        # Headers de sécurité par défaut
        self.security_headers = {
            'X-Content-Type-Options': 'nosniff',
            'X-Frame-Options': 'DENY',
            'X-XSS-Protection': '1; mode=block',
            'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
            'Content-Security-Policy': (
                "default-src 'self'; "
                "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
                "style-src 'self' 'unsafe-inline'; "
                "img-src 'self' data: https:; "
                "font-src 'self' https:; "
                "connect-src 'self' https:; "
                "media-src 'none'; "
                "object-src 'none'; "
                "child-src 'none'; "
                "worker-src 'none'; "
                "frame-ancestors 'none'; "
                "form-action 'self'; "
                "base-uri 'self'"
            ),
            'Referrer-Policy': 'strict-origin-when-cross-origin',
            'Permissions-Policy': (
                "accelerometer=(), camera=(), geolocation=(), "
                "gyroscope=(), magnetometer=(), microphone=(), "
                "payment=(), usb=()"
            )
        }
        
        logger.info("🛡️ Gestionnaire headers de sécurité initialisé")
    
    def add_security_headers(self, response: Response):
        """Ajoute les headers de sécurité à une réponse"""
        
        for header_name, header_value in self.security_headers.items():
            response.headers[header_name] = header_value
    
    def validate_request_headers(self, headers: Dict[str, str]) -> List[str]:
        """Valide les headers de requête et retourne les anomalies"""
        
        anomalies = []
        
        # Vérifier présence headers requis
        required_headers = ['host', 'user-agent']
        for header in required_headers:
            if header not in headers:
                anomalies.append(f"Header requis manquant: {header}")
        
        # Vérifier taille des headers
        for header_name, header_value in headers.items():
            if len(header_value) > 8192:  # 8KB max par header
                anomalies.append(f"Header trop volumineux: {header_name}")
        
        # Vérifier caractères suspects dans headers
        for header_name, header_value in headers.items():
            if re.search(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', header_value):
                anomalies.append(f"Caractères de contrôle dans header: {header_name}")
        
        return anomalies


class IntrusionDetectionSystem:
    """Système de détection d'intrusions"""
    
    def __init__(self):
        self.waf_engine = WAFEngine()
        self.ddos_protection = DDoSProtection()
        self.security_headers = SecurityHeadersManager()
        
        # Statistiques globales
        self.ip_stats: Dict[str, IPStats] = {}
        self.security_events: List[SecurityEvent] = []
        self.max_events_in_memory = 10000
        
        # Seuils d'alerte
        self.alert_thresholds = {
            'high_threat_score': 200,
            'multiple_attack_types': 3,
            'rapid_requests': 100,
            'geographic_anomaly': True
        }
        
        logger.info("🚨 Système de détection d'intrusions initialisé")
    
    async def analyze_request(self, request: Request) -> Tuple[bool, str, List[SecurityEvent]]:
        """Analyse complète d'une requête"""
        
        try:
            # Extraire informations de la requête
            source_ip = self._get_client_ip(request)
            user_agent = request.headers.get('user-agent', '')
            request_path = str(request.url.path)
            request_method = request.method
            
            # Lire le body de la requête
            body = await request.body()
            request_content = body.decode('utf-8', errors='ignore') if body else ''
            
            # Headers comme dictionnaire
            headers = dict(request.headers)
            
            # 1. Vérifier rate limiting
            is_limited, limit_reason = self.ddos_protection.is_rate_limited(source_ip)
            if is_limited:
                await self._log_security_event(
                    source_ip, user_agent, request_path, request_method,
                    AttackType.DDOS, ThreatLevel.HIGH, "RATE_LIMIT",
                    limit_reason, SecurityAction.BLOCK, 50
                )
                return False, f"Rate limit: {limit_reason}", []
            
            # 2. Analyser avec WAF
            waf_events, threat_score = self.waf_engine.analyze_request(
                request_content, headers, request_path, request_method, source_ip
            )
            
            # 3. Valider headers
            header_anomalies = self.security_headers.validate_request_headers(headers)
            if header_anomalies:
                for anomaly in header_anomalies:
                    await self._log_security_event(
                        source_ip, user_agent, request_path, request_method,
                        AttackType.MALFORMED_REQUEST, ThreatLevel.MEDIUM, "HEADER_ANOMALY",
                        anomaly, SecurityAction.LOG_ONLY, 20
                    )
                    threat_score += 20
            
            # 4. Mettre à jour statistiques IP
            await self._update_ip_stats(source_ip, user_agent, request_path, headers)
            
            # 5. Analyser patterns comportementaux
            behavioral_events = await self._analyze_behavioral_patterns(source_ip)
            waf_events.extend(behavioral_events)
            threat_score += sum(e.score for e in behavioral_events)
            
            # 6. Décision finale
            should_block = False
            block_reason = ""
            
            # Bloquer si score de menace élevé
            if threat_score >= self.alert_thresholds['high_threat_score']:
                should_block = True
                block_reason = f"Score de menace élevé: {threat_score}"
            
            # Bloquer si IP blacklistée
            if self.waf_engine.is_ip_blacklisted(source_ip):
                should_block = True
                block_reason = "IP blacklistée"
            
            # Bloquer si action BLOCK dans les événements
            for event in waf_events:
                if event.action_taken == SecurityAction.BLOCK:
                    should_block = True
                    block_reason = f"Règle WAF: {event.rule_id}"
                    break
            
            # Enregistrer tous les événements
            for event in waf_events:
                self.security_events.append(event)
                
                # Audit système
                audit_system = get_security_audit_system()
                await audit_system.log_event(
                    event_type=AuditEventType.SECURITY_ALERT,
                    details={
                        'attack_type': event.attack_type.value,
                        'threat_level': event.threat_level.value,
                        'rule_id': event.rule_id,
                        'source_ip': source_ip,
                        'score': event.score
                    }
                )
            
            # Nettoyer anciens événements
            if len(self.security_events) > self.max_events_in_memory:
                self.security_events = self.security_events[-self.max_events_in_memory:]
            
            return not should_block, block_reason, waf_events
            
        except Exception as e:
            logger.error(f"Erreur analyse sécurité: {e}")
            return True, "", []  # En cas d'erreur, autoriser par défaut
    
    def _get_client_ip(self, request: Request) -> str:
        """Extrait l'IP réelle du client"""
        
        # Vérifier headers de proxy
        forwarded_for = request.headers.get('x-forwarded-for')
        if forwarded_for:
            return forwarded_for.split(',')[0].strip()
        
        real_ip = request.headers.get('x-real-ip')
        if real_ip:
            return real_ip
        
        # IP directe
        return request.client.host if request.client else 'unknown'
    
    async def _log_security_event(self, source_ip: str, user_agent: str,
                                request_path: str, request_method: str,
                                attack_type: AttackType, threat_level: ThreatLevel,
                                rule_id: str, matched_content: str,
                                action: SecurityAction, score: int):
        """Enregistre un événement de sécurité"""
        
        event = SecurityEvent(
            event_id=f"ids_{datetime.now().timestamp()}",
            timestamp=datetime.now(),
            source_ip=source_ip,
            user_agent=user_agent,
            request_path=request_path,
            request_method=request_method,
            attack_type=attack_type,
            threat_level=threat_level,
            rule_id=rule_id,
            matched_content=matched_content,
            action_taken=action,
            score=score
        )
        
        self.security_events.append(event)
    
    async def _update_ip_stats(self, ip_address: str, user_agent: str,
                             request_path: str, headers: Dict[str, str]):
        """Met à jour les statistiques d'une IP"""
        
        now = datetime.now()
        
        if ip_address not in self.ip_stats:
            self.ip_stats[ip_address] = IPStats(
                ip_address=ip_address,
                first_seen=now,
                last_seen=now
            )
        
        stats = self.ip_stats[ip_address]
        stats.last_seen = now
        stats.request_count += 1
        stats.user_agents.add(user_agent)
        stats.endpoints.add(request_path)
        
        # Simuler géolocalisation
        if ip_address.startswith('185.'):
            stats.countries.add('France')
        elif ip_address.startswith('8.8.'):
            stats.countries.add('United States')
        else:
            stats.countries.add('Unknown')
    
    async def _analyze_behavioral_patterns(self, ip_address: str) -> List[SecurityEvent]:
        """Analyse les patterns comportementaux d'une IP"""
        
        events = []
        
        if ip_address not in self.ip_stats:
            return events
        
        stats = self.ip_stats[ip_address]
        
        # Trop de User-Agents différents (bot rotatif)
        if len(stats.user_agents) > 10:
            event = SecurityEvent(
                event_id=f"behavior_{datetime.now().timestamp()}",
                timestamp=datetime.now(),
                source_ip=ip_address,
                user_agent="",
                request_path="",
                request_method="",
                attack_type=AttackType.BOT_ATTACK,
                threat_level=ThreatLevel.MEDIUM,
                rule_id="BEHAVIOR001",
                matched_content=f"{len(stats.user_agents)} User-Agents différents",
                action_taken=SecurityAction.LOG_ONLY,
                score=40
            )
            events.append(event)
        
        # Trop d'endpoints différents (scan)
        if len(stats.endpoints) > 50:
            event = SecurityEvent(
                event_id=f"behavior_{datetime.now().timestamp()}_scan",
                timestamp=datetime.now(),
                source_ip=ip_address,
                user_agent="",
                request_path="",
                request_method="",
                attack_type=AttackType.BOT_ATTACK,
                threat_level=ThreatLevel.HIGH,
                rule_id="BEHAVIOR002",
                matched_content=f"{len(stats.endpoints)} endpoints différents",
                action_taken=SecurityAction.RATE_LIMIT,
                score=60
            )
            events.append(event)
        
        # Requêtes depuis plusieurs pays (géographique suspect)
        if len(stats.countries) > 2:
            event = SecurityEvent(
                event_id=f"behavior_{datetime.now().timestamp()}_geo",
                timestamp=datetime.now(),
                source_ip=ip_address,
                user_agent="",
                request_path="",
                request_method="",
                attack_type=AttackType.SUSPICIOUS_PAYLOAD,
                threat_level=ThreatLevel.MEDIUM,
                rule_id="BEHAVIOR003",
                matched_content=f"Requêtes depuis {len(stats.countries)} pays",
                action_taken=SecurityAction.LOG_ONLY,
                score=30
            )
            events.append(event)
        
        return events
    
    def get_security_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Génère un résumé de sécurité"""
        
        cutoff = datetime.now() - timedelta(hours=hours)
        recent_events = [e for e in self.security_events if e.timestamp >= cutoff]
        
        # Statistiques par type d'attaque
        attack_stats = defaultdict(int)
        for event in recent_events:
            attack_stats[event.attack_type.value] += 1
        
        # Top IPs attaquantes
        ip_attack_counts = defaultdict(int)
        for event in recent_events:
            ip_attack_counts[event.source_ip] += 1
        
        top_attacking_ips = sorted(
            ip_attack_counts.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:10]
        
        # Score de menace moyen
        avg_threat_score = (
            sum(e.score for e in recent_events) / len(recent_events)
            if recent_events else 0
        )
        
        return {
            'period_hours': hours,
            'total_security_events': len(recent_events),
            'attack_types': dict(attack_stats),
            'top_attacking_ips': top_attacking_ips,
            'average_threat_score': avg_threat_score,
            'blocked_requests': len([e for e in recent_events if e.action_taken == SecurityAction.BLOCK]),
            'high_severity_events': len([e for e in recent_events if e.threat_level == ThreatLevel.HIGH]),
            'critical_events': len([e for e in recent_events if e.threat_level == ThreatLevel.CRITICAL])
        }


class SecurityMiddleware(BaseHTTPMiddleware):
    """Middleware de sécurité principal"""
    
    def __init__(self, app, ids: IntrusionDetectionSystem):
        super().__init__(app)
        self.ids = ids
    
    async def dispatch(self, request: Request, call_next):
        """Traite chaque requête avec les contrôles de sécurité"""
        
        start_time = time.time()
        
        try:
            # Analyser la requête
            is_allowed, block_reason, security_events = await self.ids.analyze_request(request)
            
            if not is_allowed:
                # Bloquer la requête
                logger.warning(f"🚫 Requête bloquée: {block_reason}")
                
                return JSONResponse(
                    status_code=status.HTTP_403_FORBIDDEN,
                    content={
                        "error": "Accès refusé",
                        "reason": "Violation des politiques de sécurité",
                        "timestamp": datetime.now().isoformat()
                    }
                )
            
            # Traiter la requête
            response = await call_next(request)
            
            # Ajouter headers de sécurité
            self.ids.security_headers.add_security_headers(response)
            
            # Ajouter métadonnées de sécurité
            processing_time = time.time() - start_time
            response.headers["X-Security-Processing-Time"] = f"{processing_time:.3f}s"
            response.headers["X-Security-Events"] = str(len(security_events))
            
            return response
            
        except Exception as e:
            logger.error(f"Erreur middleware sécurité: {e}")
            
            # En cas d'erreur, continuer sans bloquer
            response = await call_next(request)
            self.ids.security_headers.add_security_headers(response)
            return response


# Instance globale
_intrusion_detection_system: Optional[IntrusionDetectionSystem] = None


def get_intrusion_detection_system() -> IntrusionDetectionSystem:
    """Factory pour obtenir le système de détection d'intrusions"""
    global _intrusion_detection_system
    
    if _intrusion_detection_system is None:
        _intrusion_detection_system = IntrusionDetectionSystem()
    
    return _intrusion_detection_system


def create_security_middleware():
    """Crée le middleware de sécurité"""
    ids = get_intrusion_detection_system()
    return SecurityMiddleware, ids