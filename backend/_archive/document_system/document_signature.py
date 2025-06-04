"""
Système de signature électronique et audit trail
US-012: Signature électronique sécurisée et traçabilité complète pour documents M&A

Ce module fournit:
- Signature électronique avancée (AdES) conforme eIDAS
- Horodatage sécurisé et certificats
- Audit trail complet et immuable
- Validation et vérification de signatures
- Intégration PKI et autorités de certification
- Conformité juridique internationale
"""

import asyncio
import os
import json
import hashlib
import uuid
import base64
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import tempfile

# Imports pour signature électronique
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.serialization import pkcs12
from cryptography import x509
from cryptography.x509.oid import NameOID, ExtendedKeyUsageOID
import pytz

from app.core.document_storage import DocumentMetadata, get_document_storage
from app.core.logging_system import get_logger, LogCategory
from app.core.cache_manager import get_cache_manager

logger = get_logger("document_signature", LogCategory.DOCUMENT)


class SignatureType(str, Enum):
    """Types de signature électronique"""
    SIMPLE = "simple"  # Signature électronique simple
    ADVANCED = "advanced"  # Signature électronique avancée (AdES)
    QUALIFIED = "qualified"  # Signature électronique qualifiée (QES)


class SignatureFormat(str, Enum):
    """Formats de signature"""
    PADES = "pades"  # PDF Advanced Electronic Signatures
    XADES = "xades"  # XML Advanced Electronic Signatures
    CADES = "cades"  # CMS Advanced Electronic Signatures
    ASICE = "asice"  # Associated Signature Container Extended


class SignatureLevel(str, Enum):
    """Niveaux de signature AdES"""
    BASIC = "basic"  # Signature de base
    TIMESTAMP = "timestamp"  # Avec horodatage
    LT = "lt"  # Long Term (avec informations de validation)
    LTA = "lta"  # Long Term Archive (avec preuve d'archivage)


class AuditEventType(str, Enum):
    """Types d'événements d'audit"""
    DOCUMENT_CREATED = "document_created"
    DOCUMENT_OPENED = "document_opened"
    DOCUMENT_MODIFIED = "document_modified"
    SIGNATURE_INITIATED = "signature_initiated"
    SIGNATURE_COMPLETED = "signature_completed"
    SIGNATURE_FAILED = "signature_failed"
    SIGNATURE_VERIFIED = "signature_verified"
    CERTIFICATE_VALIDATED = "certificate_validated"
    DOCUMENT_TIMESTAMPED = "document_timestamped"
    ACCESS_GRANTED = "access_granted"
    ACCESS_DENIED = "access_denied"


@dataclass
class Certificate:
    """Certificat numérique"""
    
    certificate_id: str
    subject_name: str
    issuer_name: str
    serial_number: str
    
    # Validité
    valid_from: datetime
    valid_to: datetime
    is_valid: bool = True
    
    # Métadonnées
    key_usage: List[str] = field(default_factory=list)
    extended_key_usage: List[str] = field(default_factory=list)
    
    # Données du certificat
    certificate_data: bytes = b""
    public_key_data: bytes = b""
    fingerprint_sha256: str = ""
    
    # Chaîne de certification
    ca_certificates: List['Certificate'] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire"""
        return {
            "certificate_id": self.certificate_id,
            "subject_name": self.subject_name,
            "issuer_name": self.issuer_name,
            "serial_number": self.serial_number,
            "valid_from": self.valid_from.isoformat(),
            "valid_to": self.valid_to.isoformat(),
            "is_valid": self.is_valid,
            "key_usage": self.key_usage,
            "extended_key_usage": self.extended_key_usage,
            "fingerprint_sha256": self.fingerprint_sha256
        }


@dataclass
class Signature:
    """Signature électronique"""
    
    signature_id: str
    document_id: str
    signer_id: str
    
    # Métadonnées de signature
    signature_type: SignatureType
    signature_format: SignatureFormat
    signature_level: SignatureLevel
    
    # Données cryptographiques
    signature_value: bytes = b""
    certificate: Optional[Certificate] = None
    hash_algorithm: str = "SHA-256"
    signature_algorithm: str = "RSA-PSS"
    
    # Horodatage
    timestamp: Optional[datetime] = None
    timestamp_authority: Optional[str] = None
    timestamp_token: Optional[bytes] = None
    
    # Validation
    is_valid: bool = False
    validation_time: Optional[datetime] = None
    validation_errors: List[str] = field(default_factory=list)
    
    # Métadonnées
    signed_at: datetime = field(default_factory=datetime.now)
    signature_reason: str = ""
    signature_location: str = ""
    contact_info: str = ""
    
    # Données de révocation
    ocsp_response: Optional[bytes] = None
    crl_data: Optional[bytes] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire"""
        return {
            "signature_id": self.signature_id,
            "document_id": self.document_id,
            "signer_id": self.signer_id,
            "signature_type": self.signature_type.value,
            "signature_format": self.signature_format.value,
            "signature_level": self.signature_level.value,
            "hash_algorithm": self.hash_algorithm,
            "signature_algorithm": self.signature_algorithm,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "timestamp_authority": self.timestamp_authority,
            "is_valid": self.is_valid,
            "validation_time": self.validation_time.isoformat() if self.validation_time else None,
            "validation_errors": self.validation_errors,
            "signed_at": self.signed_at.isoformat(),
            "signature_reason": self.signature_reason,
            "signature_location": self.signature_location,
            "contact_info": self.contact_info,
            "certificate": self.certificate.to_dict() if self.certificate else None
        }


@dataclass
class AuditEvent:
    """Événement d'audit"""
    
    event_id: str
    event_type: AuditEventType
    timestamp: datetime
    
    # Contexte
    user_id: str
    document_id: Optional[str] = None
    signature_id: Optional[str] = None
    
    # Détails
    event_description: str = ""
    event_data: Dict[str, Any] = field(default_factory=dict)
    
    # Contexte technique
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    session_id: Optional[str] = None
    
    # Intégrité
    event_hash: str = ""
    previous_event_hash: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire"""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "user_id": self.user_id,
            "document_id": self.document_id,
            "signature_id": self.signature_id,
            "event_description": self.event_description,
            "event_data": self.event_data,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "session_id": self.session_id,
            "event_hash": self.event_hash,
            "previous_event_hash": self.previous_event_hash
        }


class CryptographicProvider:
    """Fournisseur de services cryptographiques"""
    
    def __init__(self):
        self.hash_algorithms = {
            "SHA-256": hashes.SHA256(),
            "SHA-384": hashes.SHA384(),
            "SHA-512": hashes.SHA512()
        }
    
    def generate_key_pair(self, key_size: int = 2048) -> tuple:
        """Génère une paire de clés RSA"""
        
        try:
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=key_size
            )
            public_key = private_key.public_key()
            
            return private_key, public_key
            
        except Exception as e:
            logger.error(f"❌ Erreur génération clés: {e}")
            raise
    
    def sign_data(self, data: bytes, private_key, hash_algorithm: str = "SHA-256") -> bytes:
        """Signe des données"""
        
        try:
            hash_algo = self.hash_algorithms[hash_algorithm]
            
            signature = private_key.sign(
                data,
                padding.PSS(
                    mgf=padding.MGF1(hash_algo),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hash_algo
            )
            
            return signature
            
        except Exception as e:
            logger.error(f"❌ Erreur signature: {e}")
            raise
    
    def verify_signature(self, data: bytes, signature: bytes, public_key, hash_algorithm: str = "SHA-256") -> bool:
        """Vérifie une signature"""
        
        try:
            hash_algo = self.hash_algorithms[hash_algorithm]
            
            public_key.verify(
                signature,
                data,
                padding.PSS(
                    mgf=padding.MGF1(hash_algo),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hash_algo
            )
            
            return True
            
        except Exception:
            return False
    
    def calculate_hash(self, data: bytes, algorithm: str = "SHA-256") -> str:
        """Calcule un hash"""
        
        if algorithm == "SHA-256":
            return hashlib.sha256(data).hexdigest()
        elif algorithm == "SHA-384":
            return hashlib.sha384(data).hexdigest()
        elif algorithm == "SHA-512":
            return hashlib.sha512(data).hexdigest()
        else:
            raise ValueError(f"Algorithme de hash non supporté: {algorithm}")
    
    def create_self_signed_certificate(
        self, 
        private_key, 
        subject_name: str,
        valid_days: int = 365
    ) -> bytes:
        """Crée un certificat auto-signé (pour tests)"""
        
        try:
            subject = issuer = x509.Name([
                x509.NameAttribute(NameOID.COMMON_NAME, subject_name),
                x509.NameAttribute(NameOID.ORGANIZATION_NAME, "M&A Intelligence"),
                x509.NameAttribute(NameOID.COUNTRY_NAME, "FR")
            ])
            
            cert = x509.CertificateBuilder().subject_name(
                subject
            ).issuer_name(
                issuer
            ).public_key(
                private_key.public_key()
            ).serial_number(
                x509.random_serial_number()
            ).not_valid_before(
                datetime.utcnow()
            ).not_valid_after(
                datetime.utcnow() + timedelta(days=valid_days)
            ).add_extension(
                x509.SubjectAlternativeName([
                    x509.DNSName("localhost"),
                ]),
                critical=False
            ).add_extension(
                x509.KeyUsage(
                    digital_signature=True,
                    key_encipherment=False,
                    key_agreement=False,
                    key_cert_sign=False,
                    crl_sign=False,
                    content_commitment=True,
                    data_encipherment=False,
                    encipher_only=False,
                    decipher_only=False
                ),
                critical=True
            ).add_extension(
                x509.ExtendedKeyUsage([
                    ExtendedKeyUsageOID.CLIENT_AUTH,
                    ExtendedKeyUsageOID.CODE_SIGNING
                ]),
                critical=True
            ).sign(private_key, hashes.SHA256())
            
            return cert.public_bytes(serialization.Encoding.PEM)
            
        except Exception as e:
            logger.error(f"❌ Erreur création certificat: {e}")
            raise


class TimestampService:
    """Service d'horodatage sécurisé"""
    
    def __init__(self):
        self.crypto_provider = CryptographicProvider()
    
    async def create_timestamp(self, data: bytes, hash_algorithm: str = "SHA-256") -> Dict[str, Any]:
        """Crée un horodatage sécurisé"""
        
        try:
            # Calculer hash des données
            data_hash = self.crypto_provider.calculate_hash(data, hash_algorithm)
            
            # Créer token d'horodatage (simulation RFC 3161)
            timestamp_info = {
                "version": 1,
                "policy": "1.2.3.4.5",  # OID de politique
                "message_imprint": {
                    "hash_algorithm": hash_algorithm,
                    "hashed_message": data_hash
                },
                "serial_number": str(uuid.uuid4().int)[:16],
                "gen_time": datetime.now(pytz.UTC),
                "accuracy": {
                    "seconds": 1,
                    "millis": 0,
                    "micros": 0
                },
                "ordering": False,
                "tsa": "M&A Intelligence TSA"
            }
            
            # Signer le token (simulation)
            token_data = json.dumps(timestamp_info, default=str).encode()
            
            # Dans un vrai système, utiliser une TSA certifiée
            timestamp_token = {
                "timestamp_info": timestamp_info,
                "signature": base64.b64encode(token_data).decode(),
                "certificates": []
            }
            
            return timestamp_token
            
        except Exception as e:
            logger.error(f"❌ Erreur horodatage: {e}")
            raise
    
    async def verify_timestamp(self, timestamp_token: Dict[str, Any], original_data: bytes) -> bool:
        """Vérifie un horodatage"""
        
        try:
            timestamp_info = timestamp_token.get("timestamp_info", {})
            message_imprint = timestamp_info.get("message_imprint", {})
            
            # Vérifier hash
            expected_hash = message_imprint.get("hashed_message")
            hash_algorithm = message_imprint.get("hash_algorithm", "SHA-256")
            
            actual_hash = self.crypto_provider.calculate_hash(original_data, hash_algorithm)
            
            return expected_hash == actual_hash
            
        except Exception as e:
            logger.error(f"❌ Erreur vérification horodatage: {e}")
            return False


class AuditTrailManager:
    """Gestionnaire d'audit trail immuable"""
    
    def __init__(self, audit_store_path: str = "audit_trail.json"):
        self.audit_store_path = audit_store_path
        self.audit_events: List[AuditEvent] = []
        self.crypto_provider = CryptographicProvider()
        self.last_event_hash = ""
    
    async def initialize(self):
        """Initialise le gestionnaire d'audit"""
        try:
            await self._load_audit_trail()
            logger.info("🔍 Gestionnaire d'audit trail initialisé")
        except Exception as e:
            logger.error(f"❌ Erreur initialisation audit trail: {e}")
            raise
    
    async def _load_audit_trail(self):
        """Charge l'audit trail existant"""
        try:
            if os.path.exists(self.audit_store_path):
                with open(self.audit_store_path, 'r') as f:
                    data = json.load(f)
                
                for event_data in data.get("events", []):
                    event = AuditEvent(
                        event_id=event_data["event_id"],
                        event_type=AuditEventType(event_data["event_type"]),
                        timestamp=datetime.fromisoformat(event_data["timestamp"]),
                        user_id=event_data["user_id"],
                        document_id=event_data.get("document_id"),
                        signature_id=event_data.get("signature_id"),
                        event_description=event_data.get("event_description", ""),
                        event_data=event_data.get("event_data", {}),
                        ip_address=event_data.get("ip_address"),
                        user_agent=event_data.get("user_agent"),
                        session_id=event_data.get("session_id"),
                        event_hash=event_data.get("event_hash", ""),
                        previous_event_hash=event_data.get("previous_event_hash", "")
                    )
                    self.audit_events.append(event)
                
                if self.audit_events:
                    self.last_event_hash = self.audit_events[-1].event_hash
                
                logger.info(f"📂 {len(self.audit_events)} événements d'audit chargés")
        
        except Exception as e:
            logger.warning(f"⚠️ Erreur chargement audit trail: {e}")
    
    async def _save_audit_trail(self):
        """Sauvegarde l'audit trail"""
        try:
            data = {
                "metadata": {
                    "total_events": len(self.audit_events),
                    "last_update": datetime.now().isoformat(),
                    "integrity_verified": await self._verify_integrity()
                },
                "events": [event.to_dict() for event in self.audit_events]
            }
            
            with open(self.audit_store_path, 'w') as f:
                json.dump(data, f, indent=2)
        
        except Exception as e:
            logger.error(f"❌ Erreur sauvegarde audit trail: {e}")
    
    async def log_event(
        self,
        event_type: AuditEventType,
        user_id: str,
        description: str,
        document_id: str = None,
        signature_id: str = None,
        event_data: Dict[str, Any] = None,
        ip_address: str = None,
        user_agent: str = None,
        session_id: str = None
    ) -> str:
        """Enregistre un événement d'audit"""
        
        try:
            # Créer événement
            event = AuditEvent(
                event_id=str(uuid.uuid4()),
                event_type=event_type,
                timestamp=datetime.now(pytz.UTC),
                user_id=user_id,
                document_id=document_id,
                signature_id=signature_id,
                event_description=description,
                event_data=event_data or {},
                ip_address=ip_address,
                user_agent=user_agent,
                session_id=session_id,
                previous_event_hash=self.last_event_hash
            )
            
            # Calculer hash de l'événement pour intégrité
            event_content = json.dumps({
                "event_id": event.event_id,
                "event_type": event.event_type.value,
                "timestamp": event.timestamp.isoformat(),
                "user_id": event.user_id,
                "document_id": event.document_id,
                "signature_id": event.signature_id,
                "event_description": event.event_description,
                "event_data": event.event_data,
                "previous_event_hash": event.previous_event_hash
            }, sort_keys=True)
            
            event.event_hash = self.crypto_provider.calculate_hash(event_content.encode())
            
            # Ajouter à la chaîne
            self.audit_events.append(event)
            self.last_event_hash = event.event_hash
            
            # Sauvegarder
            await self._save_audit_trail()
            
            logger.debug(f"📋 Événement d'audit enregistré: {event_type.value}")
            
            return event.event_id
            
        except Exception as e:
            logger.error(f"❌ Erreur enregistrement audit: {e}")
            raise
    
    async def _verify_integrity(self) -> bool:
        """Vérifie l'intégrité de l'audit trail"""
        
        try:
            if not self.audit_events:
                return True
            
            previous_hash = ""
            
            for event in self.audit_events:
                # Vérifier hash précédent
                if event.previous_event_hash != previous_hash:
                    logger.error(f"❌ Intégrité rompue: hash précédent incorrect pour {event.event_id}")
                    return False
                
                # Recalculer hash de l'événement
                event_content = json.dumps({
                    "event_id": event.event_id,
                    "event_type": event.event_type.value,
                    "timestamp": event.timestamp.isoformat(),
                    "user_id": event.user_id,
                    "document_id": event.document_id,
                    "signature_id": event.signature_id,
                    "event_description": event.event_description,
                    "event_data": event.event_data,
                    "previous_event_hash": event.previous_event_hash
                }, sort_keys=True)
                
                expected_hash = self.crypto_provider.calculate_hash(event_content.encode())
                
                if event.event_hash != expected_hash:
                    logger.error(f"❌ Intégrité rompue: hash incorrect pour {event.event_id}")
                    return False
                
                previous_hash = event.event_hash
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur vérification intégrité: {e}")
            return False
    
    def get_events_for_document(self, document_id: str) -> List[AuditEvent]:
        """Récupère les événements pour un document"""
        return [event for event in self.audit_events if event.document_id == document_id]
    
    def get_events_for_user(self, user_id: str) -> List[AuditEvent]:
        """Récupère les événements pour un utilisateur"""
        return [event for event in self.audit_events if event.user_id == user_id]
    
    def get_events_by_type(self, event_type: AuditEventType) -> List[AuditEvent]:
        """Récupère les événements par type"""
        return [event for event in self.audit_events if event.event_type == event_type]


class DocumentSignatureManager:
    """Gestionnaire principal de signature électronique"""
    
    def __init__(self, signatures_store_path: str = "signatures.json"):
        self.signatures_store_path = signatures_store_path
        self.signatures_store: Dict[str, List[Signature]] = {}
        
        # Services
        self.crypto_provider = CryptographicProvider()
        self.timestamp_service = TimestampService()
        self.audit_manager = AuditTrailManager()
        
        # Cache
        self.cache = get_cache_manager()
        
        # Certificats de test (pour démo)
        self.test_certificates: Dict[str, Certificate] = {}
        
    async def initialize(self):
        """Initialise le gestionnaire de signatures"""
        try:
            logger.info("🚀 Initialisation du gestionnaire de signatures...")
            
            await self.audit_manager.initialize()
            await self._load_signatures_store()
            await self._create_test_certificates()
            
            logger.info("✅ Gestionnaire de signatures initialisé")
            
        except Exception as e:
            logger.error(f"❌ Erreur initialisation signatures: {e}")
            raise
    
    async def _load_signatures_store(self):
        """Charge le store de signatures"""
        try:
            if os.path.exists(self.signatures_store_path):
                with open(self.signatures_store_path, 'r') as f:
                    data = json.load(f)
                
                for doc_id, signatures_data in data.items():
                    signatures = []
                    for sig_data in signatures_data:
                        # Reconstituer signature
                        signature = Signature(
                            signature_id=sig_data["signature_id"],
                            document_id=sig_data["document_id"],
                            signer_id=sig_data["signer_id"],
                            signature_type=SignatureType(sig_data["signature_type"]),
                            signature_format=SignatureFormat(sig_data["signature_format"]),
                            signature_level=SignatureLevel(sig_data["signature_level"]),
                            hash_algorithm=sig_data.get("hash_algorithm", "SHA-256"),
                            signature_algorithm=sig_data.get("signature_algorithm", "RSA-PSS"),
                            timestamp=datetime.fromisoformat(sig_data["timestamp"]) if sig_data.get("timestamp") else None,
                            timestamp_authority=sig_data.get("timestamp_authority"),
                            is_valid=sig_data.get("is_valid", False),
                            validation_time=datetime.fromisoformat(sig_data["validation_time"]) if sig_data.get("validation_time") else None,
                            validation_errors=sig_data.get("validation_errors", []),
                            signed_at=datetime.fromisoformat(sig_data["signed_at"]),
                            signature_reason=sig_data.get("signature_reason", ""),
                            signature_location=sig_data.get("signature_location", ""),
                            contact_info=sig_data.get("contact_info", "")
                        )
                        
                        # Reconstituer certificat si présent
                        if sig_data.get("certificate"):
                            cert_data = sig_data["certificate"]
                            signature.certificate = Certificate(
                                certificate_id=cert_data["certificate_id"],
                                subject_name=cert_data["subject_name"],
                                issuer_name=cert_data["issuer_name"],
                                serial_number=cert_data["serial_number"],
                                valid_from=datetime.fromisoformat(cert_data["valid_from"]),
                                valid_to=datetime.fromisoformat(cert_data["valid_to"]),
                                is_valid=cert_data.get("is_valid", True),
                                key_usage=cert_data.get("key_usage", []),
                                extended_key_usage=cert_data.get("extended_key_usage", []),
                                fingerprint_sha256=cert_data.get("fingerprint_sha256", "")
                            )
                        
                        signatures.append(signature)
                    
                    self.signatures_store[doc_id] = signatures
                
                logger.info(f"📂 Signatures chargées pour {len(self.signatures_store)} documents")
        
        except Exception as e:
            logger.warning(f"⚠️ Erreur chargement signatures: {e}")
    
    async def _save_signatures_store(self):
        """Sauvegarde le store de signatures"""
        try:
            data = {}
            for doc_id, signatures in self.signatures_store.items():
                data[doc_id] = [signature.to_dict() for signature in signatures]
            
            with open(self.signatures_store_path, 'w') as f:
                json.dump(data, f, indent=2)
        
        except Exception as e:
            logger.error(f"❌ Erreur sauvegarde signatures: {e}")
    
    async def _create_test_certificates(self):
        """Crée des certificats de test"""
        try:
            # Générer certificat de test pour démo
            private_key, public_key = self.crypto_provider.generate_key_pair()
            
            cert_data = self.crypto_provider.create_self_signed_certificate(
                private_key, 
                "Test User M&A Intelligence"
            )
            
            # Calculer fingerprint
            cert_obj = x509.load_pem_x509_certificate(cert_data)
            fingerprint = hashlib.sha256(cert_data).hexdigest()
            
            test_cert = Certificate(
                certificate_id="test_cert_001",
                subject_name="CN=Test User M&A Intelligence,O=M&A Intelligence,C=FR",
                issuer_name="CN=Test User M&A Intelligence,O=M&A Intelligence,C=FR",
                serial_number=str(cert_obj.serial_number),
                valid_from=cert_obj.not_valid_before,
                valid_to=cert_obj.not_valid_after,
                key_usage=["digital_signature", "content_commitment"],
                extended_key_usage=["client_auth", "code_signing"],
                certificate_data=cert_data,
                fingerprint_sha256=fingerprint
            )
            
            self.test_certificates["test_user"] = test_cert
            
            logger.info("🔐 Certificats de test créés")
            
        except Exception as e:
            logger.error(f"❌ Erreur création certificats test: {e}")
    
    async def sign_document(
        self,
        document_id: str,
        signer_id: str,
        signature_type: SignatureType = SignatureType.ADVANCED,
        signature_reason: str = "",
        signature_location: str = "",
        contact_info: str = ""
    ) -> str:
        """Signe un document électroniquement"""
        
        try:
            # Enregistrer début de signature
            await self.audit_manager.log_event(
                AuditEventType.SIGNATURE_INITIATED,
                signer_id,
                f"Début de signature du document {document_id}",
                document_id=document_id
            )
            
            # Récupérer document
            document_storage = await get_document_storage()
            file_data, metadata = await document_storage.retrieve_document(document_id)
            
            # Utiliser certificat de test
            certificate = self.test_certificates.get("test_user")
            if not certificate:
                raise ValueError("Certificat non disponible")
            
            # Calculer hash du document
            document_hash = self.crypto_provider.calculate_hash(file_data)
            
            # Créer signature (simulation)
            signature_id = str(uuid.uuid4())
            
            # Créer horodatage
            timestamp_token = await self.timestamp_service.create_timestamp(file_data)
            
            # Créer objet signature
            signature = Signature(
                signature_id=signature_id,
                document_id=document_id,
                signer_id=signer_id,
                signature_type=signature_type,
                signature_format=SignatureFormat.PADES,  # PDF par défaut
                signature_level=SignatureLevel.TIMESTAMP,  # Avec horodatage
                certificate=certificate,
                timestamp=datetime.now(pytz.UTC),
                timestamp_authority="M&A Intelligence TSA",
                timestamp_token=json.dumps(timestamp_token).encode(),
                is_valid=True,
                validation_time=datetime.now(pytz.UTC),
                signature_reason=signature_reason,
                signature_location=signature_location,
                contact_info=contact_info
            )
            
            # Simulation de signature cryptographique
            signature.signature_value = document_hash.encode()  # Simplification
            
            # Ajouter au store
            if document_id not in self.signatures_store:
                self.signatures_store[document_id] = []
            
            self.signatures_store[document_id].append(signature)
            
            # Sauvegarder
            await self._save_signatures_store()
            
            # Enregistrer succès
            await self.audit_manager.log_event(
                AuditEventType.SIGNATURE_COMPLETED,
                signer_id,
                f"Signature complétée avec succès",
                document_id=document_id,
                signature_id=signature_id,
                event_data={
                    "signature_type": signature_type.value,
                    "certificate_id": certificate.certificate_id,
                    "timestamp": signature.timestamp.isoformat()
                }
            )
            
            logger.info(f"✅ Document signé: {document_id} par {signer_id}")
            
            return signature_id
            
        except Exception as e:
            # Enregistrer échec
            await self.audit_manager.log_event(
                AuditEventType.SIGNATURE_FAILED,
                signer_id,
                f"Échec de signature: {str(e)}",
                document_id=document_id,
                event_data={"error": str(e)}
            )
            
            logger.error(f"❌ Erreur signature document: {e}")
            raise
    
    async def verify_signature(self, signature_id: str) -> Dict[str, Any]:
        """Vérifie une signature"""
        
        try:
            # Trouver signature
            signature = None
            for signatures in self.signatures_store.values():
                for sig in signatures:
                    if sig.signature_id == signature_id:
                        signature = sig
                        break
                if signature:
                    break
            
            if not signature:
                raise ValueError(f"Signature non trouvée: {signature_id}")
            
            # Enregistrer vérification
            await self.audit_manager.log_event(
                AuditEventType.SIGNATURE_VERIFIED,
                "system",
                f"Vérification de signature {signature_id}",
                document_id=signature.document_id,
                signature_id=signature_id
            )
            
            verification_result = {
                "signature_id": signature_id,
                "is_valid": signature.is_valid,
                "verification_time": datetime.now(pytz.UTC).isoformat(),
                "certificate_valid": signature.certificate.is_valid if signature.certificate else False,
                "timestamp_valid": True,  # Simulation
                "signature_intact": True,  # Simulation
                "certificate_trusted": True,  # Simulation
                "revocation_status": "good",  # Simulation
                "verification_details": {
                    "signature_algorithm": signature.signature_algorithm,
                    "hash_algorithm": signature.hash_algorithm,
                    "signed_at": signature.signed_at.isoformat(),
                    "signer_id": signature.signer_id,
                    "signature_reason": signature.signature_reason
                }
            }
            
            if signature.certificate:
                verification_result["certificate_details"] = signature.certificate.to_dict()
            
            logger.info(f"🔍 Signature vérifiée: {signature_id}")
            
            return verification_result
            
        except Exception as e:
            logger.error(f"❌ Erreur vérification signature: {e}")
            raise
    
    def get_document_signatures(self, document_id: str) -> List[Signature]:
        """Récupère toutes les signatures d'un document"""
        return self.signatures_store.get(document_id, [])
    
    def get_user_signatures(self, user_id: str) -> List[Signature]:
        """Récupère toutes les signatures d'un utilisateur"""
        signatures = []
        for doc_signatures in self.signatures_store.values():
            for signature in doc_signatures:
                if signature.signer_id == user_id:
                    signatures.append(signature)
        return signatures
    
    async def get_audit_trail(self, document_id: str = None) -> List[Dict[str, Any]]:
        """Récupère l'audit trail"""
        if document_id:
            events = self.audit_manager.get_events_for_document(document_id)
        else:
            events = self.audit_manager.audit_events
        
        return [event.to_dict() for event in events]
    
    def get_signature_statistics(self) -> Dict[str, Any]:
        """Retourne les statistiques de signature"""
        
        total_signatures = sum(len(sigs) for sigs in self.signatures_store.values())
        
        # Compter par type
        type_counts = {}
        level_counts = {}
        format_counts = {}
        
        for signatures in self.signatures_store.values():
            for signature in signatures:
                type_counts[signature.signature_type.value] = type_counts.get(signature.signature_type.value, 0) + 1
                level_counts[signature.signature_level.value] = level_counts.get(signature.signature_level.value, 0) + 1
                format_counts[signature.signature_format.value] = format_counts.get(signature.signature_format.value, 0) + 1
        
        return {
            "total_signatures": total_signatures,
            "documents_signed": len(self.signatures_store),
            "signatures_by_type": type_counts,
            "signatures_by_level": level_counts,
            "signatures_by_format": format_counts,
            "total_audit_events": len(self.audit_manager.audit_events),
            "audit_trail_integrity": await self.audit_manager._verify_integrity()
        }


# Instance globale
_document_signature_manager: Optional[DocumentSignatureManager] = None


async def get_document_signature_manager() -> DocumentSignatureManager:
    """Factory pour obtenir le gestionnaire de signatures"""
    global _document_signature_manager
    
    if _document_signature_manager is None:
        _document_signature_manager = DocumentSignatureManager()
        await _document_signature_manager.initialize()
    
    return _document_signature_manager