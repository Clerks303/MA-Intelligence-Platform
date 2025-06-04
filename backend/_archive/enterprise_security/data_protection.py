"""
Système de protection et chiffrement des données pour M&A Intelligence Platform
US-007: Protection avancée des données sensibles et conformité RGPD

Features:
- Chiffrement AES-256 pour données sensibles
- Anonymisation et pseudonymisation
- Gestion des clés de chiffrement sécurisée
- Contrôle de rétention des données
- Audit des accès aux données
- Conformité RGPD (droit à l'oubli, portabilité)
"""

import asyncio
import base64
import hashlib
import secrets
import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import re

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
import bcrypt

from app.config import settings
from app.core.logging_system import get_logger, LogCategory
from app.core.security_audit import get_security_audit_system, AuditEventType, ResourceType, Action

logger = get_logger("data_protection", LogCategory.SECURITY)


class DataClassification(str, Enum):
    """Classification des données selon leur sensibilité"""
    PUBLIC = "public"              # Données publiques
    INTERNAL = "internal"          # Données internes
    CONFIDENTIAL = "confidential"  # Données confidentielles
    RESTRICTED = "restricted"      # Données à accès restreint
    SECRET = "secret"             # Données secrètes


class EncryptionMethod(str, Enum):
    """Méthodes de chiffrement disponibles"""
    FERNET = "fernet"             # Chiffrement symétrique Fernet
    AES_256_GCM = "aes_256_gcm"   # AES-256 en mode GCM
    RSA_2048 = "rsa_2048"         # RSA 2048 bits
    RSA_4096 = "rsa_4096"         # RSA 4096 bits


class DataProcessingPurpose(str, Enum):
    """Finalités de traitement des données RGPD"""
    BUSINESS_INTELLIGENCE = "business_intelligence"
    PROSPECTION = "prospection"
    ANALYTICS = "analytics"
    COMPLIANCE = "compliance"
    SECURITY = "security"
    TECHNICAL_OPERATION = "technical_operation"


@dataclass
class EncryptionKey:
    """Clé de chiffrement avec métadonnées"""
    key_id: str
    key_data: bytes
    algorithm: EncryptionMethod
    created_at: datetime
    expires_at: Optional[datetime] = None
    purpose: str = "general"
    rotation_count: int = 0
    is_active: bool = True
    
    def is_expired(self) -> bool:
        """Vérifie si la clé est expirée"""
        return self.expires_at is not None and datetime.now() > self.expires_at


@dataclass
class DataRecord:
    """Enregistrement de données avec métadonnées de protection"""
    record_id: str
    data_type: str
    classification: DataClassification
    owner_id: str
    created_at: datetime
    processing_purposes: Set[DataProcessingPurpose] = field(default_factory=set)
    retention_until: Optional[datetime] = None
    encrypted_fields: Set[str] = field(default_factory=set)
    anonymized_fields: Set[str] = field(default_factory=set)
    access_log: List[Dict[str, Any]] = field(default_factory=list)
    consent_given: bool = False
    consent_date: Optional[datetime] = None
    
    def is_retention_expired(self) -> bool:
        """Vérifie si la période de rétention est expirée"""
        return self.retention_until is not None and datetime.now() > self.retention_until


@dataclass
class AnonymizationRule:
    """Règle d'anonymisation pour un champ"""
    field_name: str
    anonymization_type: str  # "hash", "mask", "remove", "generalize"
    options: Dict[str, Any] = field(default_factory=dict)


class EncryptionManager:
    """Gestionnaire de chiffrement et de clés"""
    
    def __init__(self):
        self.keys: Dict[str, EncryptionKey] = {}
        self.master_key = self._derive_master_key()
        
        # Générer clés par défaut
        self._generate_default_keys()
        
        logger.info("🔐 Gestionnaire de chiffrement initialisé")
    
    def _derive_master_key(self) -> bytes:
        """Dérive la clé maître depuis la configuration"""
        
        # En production, utiliser un HSM ou coffre-fort sécurisé
        master_secret = getattr(settings, 'ENCRYPTION_MASTER_KEY', 'default-master-key-change-in-production')
        
        # Dérivation PBKDF2
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b'ma_intelligence_encryption_salt',  # En prod: salt unique stocké séparément
            iterations=100000,
            backend=default_backend()
        )
        
        return kdf.derive(master_secret.encode())
    
    def _generate_default_keys(self):
        """Génère les clés par défaut du système"""
        
        # Clé Fernet pour données générales
        fernet_key = Fernet.generate_key()
        self.keys["default_fernet"] = EncryptionKey(
            key_id="default_fernet",
            key_data=fernet_key,
            algorithm=EncryptionMethod.FERNET,
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(days=365),  # 1 an
            purpose="general_data"
        )
        
        # Clé AES pour données sensibles
        aes_key = secrets.token_bytes(32)  # 256 bits
        self.keys["sensitive_aes"] = EncryptionKey(
            key_id="sensitive_aes",
            key_data=aes_key,
            algorithm=EncryptionMethod.AES_256_GCM,
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(days=90),  # 3 mois
            purpose="sensitive_data"
        )
        
        # Clés RSA pour échange
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
            backend=default_backend()
        )
        
        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        
        self.keys["exchange_rsa"] = EncryptionKey(
            key_id="exchange_rsa",
            key_data=private_pem,
            algorithm=EncryptionMethod.RSA_2048,
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(days=1095),  # 3 ans
            purpose="key_exchange"
        )
        
        logger.info(f"✅ {len(self.keys)} clés de chiffrement générées")
    
    def get_key(self, key_id: str) -> Optional[EncryptionKey]:
        """Récupère une clé par son ID"""
        key = self.keys.get(key_id)
        
        if key and key.is_expired():
            logger.warning(f"Clé expirée utilisée: {key_id}")
            return None
        
        return key
    
    def encrypt_data(self, data: Union[str, bytes], 
                    key_id: str = "default_fernet",
                    additional_data: bytes = None) -> Tuple[bytes, str]:
        """Chiffre des données"""
        
        try:
            key = self.get_key(key_id)
            if not key:
                raise ValueError(f"Clé introuvable ou expirée: {key_id}")
            
            # Convertir en bytes si nécessaire
            if isinstance(data, str):
                data = data.encode('utf-8')
            
            if key.algorithm == EncryptionMethod.FERNET:
                cipher = Fernet(key.key_data)
                encrypted = cipher.encrypt(data)
                
            elif key.algorithm == EncryptionMethod.AES_256_GCM:
                # Générer IV aléatoire
                iv = secrets.token_bytes(12)  # 96 bits pour GCM
                
                cipher = Cipher(
                    algorithms.AES(key.key_data),
                    modes.GCM(iv),
                    backend=default_backend()
                )
                
                encryptor = cipher.encryptor()
                
                if additional_data:
                    encryptor.authenticate_additional_data(additional_data)
                
                encrypted_data = encryptor.update(data) + encryptor.finalize()
                
                # Combiner IV + tag + données chiffrées
                encrypted = iv + encryptor.tag + encrypted_data
                
            elif key.algorithm in [EncryptionMethod.RSA_2048, EncryptionMethod.RSA_4096]:
                # Charger clé RSA
                private_key = serialization.load_pem_private_key(
                    key.key_data,
                    password=None,
                    backend=default_backend()
                )
                
                public_key = private_key.public_key()
                
                # RSA avec OAEP
                encrypted = public_key.encrypt(
                    data,
                    padding.OAEP(
                        mgf=padding.MGF1(algorithm=hashes.SHA256()),
                        algorithm=hashes.SHA256(),
                        label=None
                    )
                )
            
            else:
                raise ValueError(f"Algorithme non supporté: {key.algorithm}")
            
            logger.debug(f"Données chiffrées avec {key.algorithm.value}")
            return encrypted, key_id
            
        except Exception as e:
            logger.error(f"Erreur chiffrement avec clé {key_id}: {e}")
            raise
    
    def decrypt_data(self, encrypted_data: bytes, 
                    key_id: str,
                    additional_data: bytes = None) -> bytes:
        """Déchiffre des données"""
        
        try:
            key = self.get_key(key_id)
            if not key:
                raise ValueError(f"Clé introuvable ou expirée: {key_id}")
            
            if key.algorithm == EncryptionMethod.FERNET:
                cipher = Fernet(key.key_data)
                decrypted = cipher.decrypt(encrypted_data)
                
            elif key.algorithm == EncryptionMethod.AES_256_GCM:
                # Extraire IV, tag et données
                iv = encrypted_data[:12]
                tag = encrypted_data[12:28]
                ciphertext = encrypted_data[28:]
                
                cipher = Cipher(
                    algorithms.AES(key.key_data),
                    modes.GCM(iv, tag),
                    backend=default_backend()
                )
                
                decryptor = cipher.decryptor()
                
                if additional_data:
                    decryptor.authenticate_additional_data(additional_data)
                
                decrypted = decryptor.update(ciphertext) + decryptor.finalize()
                
            elif key.algorithm in [EncryptionMethod.RSA_2048, EncryptionMethod.RSA_4096]:
                # Charger clé RSA
                private_key = serialization.load_pem_private_key(
                    key.key_data,
                    password=None,
                    backend=default_backend()
                )
                
                # RSA avec OAEP
                decrypted = private_key.decrypt(
                    encrypted_data,
                    padding.OAEP(
                        mgf=padding.MGF1(algorithm=hashes.SHA256()),
                        algorithm=hashes.SHA256(),
                        label=None
                    )
                )
            
            else:
                raise ValueError(f"Algorithme non supporté: {key.algorithm}")
            
            logger.debug(f"Données déchiffrées avec {key.algorithm.value}")
            return decrypted
            
        except Exception as e:
            logger.error(f"Erreur déchiffrement avec clé {key_id}: {e}")
            raise
    
    def rotate_key(self, key_id: str) -> bool:
        """Effectue la rotation d'une clé"""
        
        try:
            old_key = self.keys.get(key_id)
            if not old_key:
                return False
            
            # Générer nouvelle clé selon l'algorithme
            if old_key.algorithm == EncryptionMethod.FERNET:
                new_key_data = Fernet.generate_key()
                
            elif old_key.algorithm == EncryptionMethod.AES_256_GCM:
                new_key_data = secrets.token_bytes(32)
                
            elif old_key.algorithm in [EncryptionMethod.RSA_2048, EncryptionMethod.RSA_4096]:
                key_size = 2048 if old_key.algorithm == EncryptionMethod.RSA_2048 else 4096
                
                private_key = rsa.generate_private_key(
                    public_exponent=65537,
                    key_size=key_size,
                    backend=default_backend()
                )
                
                new_key_data = private_key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.PKCS8,
                    encryption_algorithm=serialization.NoEncryption()
                )
            
            else:
                return False
            
            # Créer nouvelle clé
            new_key = EncryptionKey(
                key_id=key_id,
                key_data=new_key_data,
                algorithm=old_key.algorithm,
                created_at=datetime.now(),
                expires_at=datetime.now() + timedelta(days=365),
                purpose=old_key.purpose,
                rotation_count=old_key.rotation_count + 1
            )
            
            # Archiver ancienne clé
            archived_key_id = f"{key_id}_archived_{old_key.rotation_count}"
            old_key.is_active = False
            self.keys[archived_key_id] = old_key
            
            # Remplacer par nouvelle clé
            self.keys[key_id] = new_key
            
            logger.info(f"✅ Rotation de clé effectuée: {key_id}")
            return True
            
        except Exception as e:
            logger.error(f"Erreur rotation clé {key_id}: {e}")
            return False


class DataAnonymizer:
    """Gestionnaire d'anonymisation des données"""
    
    def __init__(self):
        # Configurations d'anonymisation par type de données
        self.anonymization_rules = {
            'email': [
                AnonymizationRule('email', 'hash', {'salt': 'email_salt'}),
                AnonymizationRule('email', 'mask', {'pattern': '*****@*****.***'})
            ],
            'phone': [
                AnonymizationRule('phone', 'mask', {'pattern': '**-**-**-**-**'}),
                AnonymizationRule('phone', 'generalize', {'keep_country_code': True})
            ],
            'name': [
                AnonymizationRule('name', 'mask', {'keep_first_letter': True}),
                AnonymizationRule('name', 'remove', {})
            ],
            'address': [
                AnonymizationRule('address', 'generalize', {'keep_city': True}),
                AnonymizationRule('address', 'remove', {})
            ],
            'siret': [
                AnonymizationRule('siret', 'hash', {'salt': 'siret_salt'}),
                AnonymizationRule('siret', 'mask', {'keep_prefix': 3})
            ]
        }
        
        logger.info("🎭 Gestionnaire d'anonymisation initialisé")
    
    def anonymize_field(self, field_name: str, value: Any, 
                       anonymization_type: str = "hash",
                       options: Dict[str, Any] = None) -> Any:
        """Anonymise un champ selon les règles"""
        
        if value is None:
            return None
        
        if options is None:
            options = {}
        
        try:
            value_str = str(value)
            
            if anonymization_type == "hash":
                # Hash avec sel
                salt = options.get('salt', 'default_salt')
                combined = f"{value_str}:{salt}"
                return hashlib.sha256(combined.encode()).hexdigest()[:16]
            
            elif anonymization_type == "mask":
                pattern = options.get('pattern')
                if pattern:
                    return pattern
                
                # Masquage par défaut
                if len(value_str) <= 2:
                    return "*" * len(value_str)
                elif len(value_str) <= 6:
                    return value_str[0] + "*" * (len(value_str) - 2) + value_str[-1]
                else:
                    keep_first = options.get('keep_first_letter', False)
                    if keep_first:
                        return value_str[0] + "*" * (len(value_str) - 1)
                    else:
                        return "*" * len(value_str)
            
            elif anonymization_type == "remove":
                return "[SUPPRIMÉ]"
            
            elif anonymization_type == "generalize":
                # Généralisation selon le type
                if field_name == "email":
                    domain = value_str.split('@')[1] if '@' in value_str else "unknown.com"
                    return f"***@{domain}"
                
                elif field_name == "phone":
                    if options.get('keep_country_code') and value_str.startswith('+'):
                        return value_str[:3] + "*" * (len(value_str) - 3)
                    else:
                        return "*" * len(value_str)
                
                elif field_name == "address":
                    if options.get('keep_city'):
                        # Logique simplifiée pour garder la ville
                        parts = value_str.split(',')
                        if len(parts) > 1:
                            return f"*** {parts[-1].strip()}"
                    return "***"
                
                else:
                    return "***"
            
            else:
                logger.warning(f"Type d'anonymisation non supporté: {anonymization_type}")
                return value
                
        except Exception as e:
            logger.error(f"Erreur anonymisation champ {field_name}: {e}")
            return "[ERREUR]"
    
    def anonymize_record(self, data: Dict[str, Any], 
                        fields_to_anonymize: List[str],
                        anonymization_type: str = "hash") -> Dict[str, Any]:
        """Anonymise plusieurs champs d'un enregistrement"""
        
        anonymized_data = data.copy()
        
        for field_name in fields_to_anonymize:
            if field_name in anonymized_data:
                original_value = anonymized_data[field_name]
                
                # Obtenir options selon le type de champ
                options = {}
                if field_name in self.anonymization_rules:
                    rule = next(
                        (r for r in self.anonymization_rules[field_name] 
                         if r.anonymization_type == anonymization_type),
                        None
                    )
                    if rule:
                        options = rule.options
                
                anonymized_value = self.anonymize_field(
                    field_name, original_value, anonymization_type, options
                )
                
                anonymized_data[field_name] = anonymized_value
        
        return anonymized_data
    
    def pseudonymize_dataset(self, dataset: List[Dict[str, Any]], 
                           identifier_fields: List[str]) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
        """Pseudonymise un jeu de données avec mapping réversible"""
        
        pseudonym_mapping = {}
        pseudonymized_dataset = []
        
        for record in dataset:
            pseudonymized_record = record.copy()
            
            for field in identifier_fields:
                if field in record and record[field] is not None:
                    original_value = str(record[field])
                    
                    # Générer pseudonyme consistent
                    if original_value not in pseudonym_mapping:
                        # Utiliser un hash déterministe comme pseudonyme
                        pseudonym = hashlib.sha256(
                            f"{field}:{original_value}:pseudonym_salt".encode()
                        ).hexdigest()[:12]
                        
                        pseudonym_mapping[original_value] = pseudonym
                    
                    pseudonymized_record[field] = pseudonym_mapping[original_value]
            
            pseudonymized_dataset.append(pseudonymized_record)
        
        return pseudonymized_dataset, pseudonym_mapping


class DataRetentionManager:
    """Gestionnaire de rétention des données"""
    
    def __init__(self):
        # Politiques de rétention par type de données
        self.retention_policies = {
            'user_activity': timedelta(days=1095),  # 3 ans
            'company_data': timedelta(days=2555),   # 7 ans
            'audit_logs': timedelta(days=2555),     # 7 ans
            'temporary_data': timedelta(days=30),   # 1 mois
            'session_data': timedelta(days=7),      # 1 semaine
            'export_data': timedelta(days=90),      # 3 mois
        }
        
        self.data_records: Dict[str, DataRecord] = {}
        
        logger.info("📅 Gestionnaire de rétention initialisé")
    
    def register_data(self, record_id: str, data_type: str, 
                     classification: DataClassification,
                     owner_id: str,
                     processing_purposes: List[DataProcessingPurpose],
                     custom_retention: timedelta = None) -> DataRecord:
        """Enregistre des données avec métadonnées de rétention"""
        
        now = datetime.now()
        
        # Déterminer période de rétention
        if custom_retention:
            retention_until = now + custom_retention
        else:
            default_retention = self.retention_policies.get(data_type, timedelta(days=365))
            retention_until = now + default_retention
        
        record = DataRecord(
            record_id=record_id,
            data_type=data_type,
            classification=classification,
            owner_id=owner_id,
            created_at=now,
            processing_purposes=set(processing_purposes),
            retention_until=retention_until
        )
        
        self.data_records[record_id] = record
        
        logger.debug(f"📝 Données enregistrées: {record_id} (rétention jusqu'au {retention_until.date()})")
        
        return record
    
    def check_expired_data(self) -> List[str]:
        """Vérifie les données dont la rétention a expiré"""
        
        expired_records = []
        
        for record_id, record in self.data_records.items():
            if record.is_retention_expired():
                expired_records.append(record_id)
        
        return expired_records
    
    def extend_retention(self, record_id: str, 
                        additional_time: timedelta,
                        justification: str) -> bool:
        """Prolonge la rétention d'un enregistrement"""
        
        try:
            if record_id not in self.data_records:
                return False
            
            record = self.data_records[record_id]
            old_retention = record.retention_until
            
            if record.retention_until:
                record.retention_until += additional_time
            else:
                record.retention_until = datetime.now() + additional_time
            
            # Log de l'extension
            record.access_log.append({
                'action': 'retention_extended',
                'timestamp': datetime.now().isoformat(),
                'old_retention': old_retention.isoformat() if old_retention else None,
                'new_retention': record.retention_until.isoformat(),
                'justification': justification
            })
            
            logger.info(f"📅 Rétention prolongée pour {record_id}: {justification}")
            
            return True
            
        except Exception as e:
            logger.error(f"Erreur prolongation rétention {record_id}: {e}")
            return False
    
    async def cleanup_expired_data(self, dry_run: bool = True) -> Dict[str, Any]:
        """Nettoie les données expirées"""
        
        expired_records = self.check_expired_data()
        cleanup_results = {
            'total_expired': len(expired_records),
            'cleaned_records': [],
            'errors': [],
            'dry_run': dry_run
        }
        
        for record_id in expired_records:
            try:
                record = self.data_records[record_id]
                
                if not dry_run:
                    # Audit avant suppression
                    audit_system = get_security_audit_system()
                    await audit_system.log_event(
                        event_type=AuditEventType.DATA_RETENTION_ACTION,
                        user_id="system",
                        resource_type=ResourceType.COMPANY,  # ou autre selon le type
                        resource_id=record_id,
                        action=Action.DELETE,
                        details={
                            'data_type': record.data_type,
                            'retention_expired': record.retention_until.isoformat() if record.retention_until else None,
                            'classification': record.classification.value
                        }
                    )
                    
                    # Supprimer l'enregistrement
                    del self.data_records[record_id]
                
                cleanup_results['cleaned_records'].append({
                    'record_id': record_id,
                    'data_type': record.data_type,
                    'classification': record.classification.value,
                    'expired_since': (datetime.now() - record.retention_until).days if record.retention_until else 0
                })
                
            except Exception as e:
                cleanup_results['errors'].append({
                    'record_id': record_id,
                    'error': str(e)
                })
                logger.error(f"Erreur nettoyage {record_id}: {e}")
        
        if not dry_run and cleanup_results['cleaned_records']:
            logger.info(f"🧹 {len(cleanup_results['cleaned_records'])} enregistrements supprimés (rétention expirée)")
        
        return cleanup_results


class GDPRComplianceManager:
    """Gestionnaire de conformité RGPD"""
    
    def __init__(self, encryption_manager: EncryptionManager,
                 anonymizer: DataAnonymizer,
                 retention_manager: DataRetentionManager):
        
        self.encryption_manager = encryption_manager
        self.anonymizer = anonymizer
        self.retention_manager = retention_manager
        
        # Types de demandes RGPD
        self.gdpr_request_types = {
            'access': 'Droit d\'accès (Art. 15)',
            'rectification': 'Droit de rectification (Art. 16)',
            'erasure': 'Droit à l\'effacement (Art. 17)',
            'portability': 'Droit à la portabilité (Art. 20)',
            'restriction': 'Droit à la limitation du traitement (Art. 18)',
            'objection': 'Droit d\'opposition (Art. 21)'
        }
        
        logger.info("⚖️ Gestionnaire conformité RGPD initialisé")
    
    async def handle_access_request(self, user_id: str, 
                                  email: str) -> Dict[str, Any]:
        """Traite une demande d'accès aux données (Art. 15 RGPD)"""
        
        try:
            # Audit de la demande
            audit_system = get_security_audit_system()
            await audit_system.log_event(
                event_type=AuditEventType.GDPR_REQUEST,
                user_id=user_id,
                details={
                    'request_type': 'access',
                    'email': email
                }
            )
            
            # Rechercher toutes les données liées à l'utilisateur
            user_data = {
                'user_info': {
                    'user_id': user_id,
                    'email': email,
                    'request_date': datetime.now().isoformat()
                },
                'data_categories': {},
                'processing_purposes': [],
                'retention_periods': {},
                'data_sources': []
            }
            
            # Récupérer données depuis les enregistrements
            for record_id, record in self.retention_manager.data_records.items():
                if record.owner_id == user_id:
                    category = record.data_type
                    
                    if category not in user_data['data_categories']:
                        user_data['data_categories'][category] = []
                    
                    user_data['data_categories'][category].append({
                        'record_id': record_id,
                        'created_at': record.created_at.isoformat(),
                        'classification': record.classification.value,
                        'retention_until': record.retention_until.isoformat() if record.retention_until else None,
                        'processing_purposes': [p.value for p in record.processing_purposes]
                    })
                    
                    # Ajouter finalités de traitement
                    user_data['processing_purposes'].extend([p.value for p in record.processing_purposes])
                    
                    # Périodes de rétention
                    if category not in user_data['retention_periods']:
                        user_data['retention_periods'][category] = record.retention_until.isoformat() if record.retention_until else "Non définie"
            
            # Déduplication
            user_data['processing_purposes'] = list(set(user_data['processing_purposes']))
            
            return {
                'status': 'success',
                'request_id': f"access_{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'data': user_data
            }
            
        except Exception as e:
            logger.error(f"Erreur traitement demande d'accès {user_id}: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    async def handle_erasure_request(self, user_id: str, 
                                   email: str,
                                   justification: str = "") -> Dict[str, Any]:
        """Traite une demande d'effacement (Art. 17 RGPD)"""
        
        try:
            # Audit de la demande
            audit_system = get_security_audit_system()
            await audit_system.log_event(
                event_type=AuditEventType.GDPR_REQUEST,
                user_id=user_id,
                details={
                    'request_type': 'erasure',
                    'email': email,
                    'justification': justification
                }
            )
            
            # Rechercher données à supprimer
            records_to_delete = []
            records_to_anonymize = []
            
            for record_id, record in self.retention_manager.data_records.items():
                if record.owner_id == user_id:
                    # Vérifier si suppression possible ou si anonymisation requise
                    if self._can_delete_record(record):
                        records_to_delete.append(record_id)
                    else:
                        records_to_anonymize.append(record_id)
            
            deletion_results = {
                'deleted_records': [],
                'anonymized_records': [],
                'retained_records': [],
                'errors': []
            }
            
            # Supprimer les enregistrements éligibles
            for record_id in records_to_delete:
                try:
                    record = self.retention_manager.data_records[record_id]
                    
                    # Log avant suppression
                    await audit_system.log_event(
                        event_type=AuditEventType.DATA_DELETION,
                        user_id=user_id,
                        resource_id=record_id,
                        details={
                            'reason': 'gdpr_erasure_request',
                            'data_type': record.data_type
                        }
                    )
                    
                    del self.retention_manager.data_records[record_id]
                    deletion_results['deleted_records'].append(record_id)
                    
                except Exception as e:
                    deletion_results['errors'].append({
                        'record_id': record_id,
                        'error': str(e)
                    })
            
            # Anonymiser les enregistrements qui ne peuvent pas être supprimés
            for record_id in records_to_anonymize:
                try:
                    record = self.retention_manager.data_records[record_id]
                    
                    # Marquer comme anonymisé
                    record.anonymized_fields.update(['all_personal_data'])
                    record.owner_id = 'anonymized'
                    
                    await audit_system.log_event(
                        event_type=AuditEventType.DATA_MODIFICATION,
                        user_id=user_id,
                        resource_id=record_id,
                        details={
                            'reason': 'gdpr_erasure_request_anonymization',
                            'data_type': record.data_type
                        }
                    )
                    
                    deletion_results['anonymized_records'].append(record_id)
                    
                except Exception as e:
                    deletion_results['errors'].append({
                        'record_id': record_id,
                        'error': str(e)
                    })
            
            return {
                'status': 'success',
                'request_id': f"erasure_{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'results': deletion_results
            }
            
        except Exception as e:
            logger.error(f"Erreur traitement demande d'effacement {user_id}: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    async def handle_portability_request(self, user_id: str, 
                                       email: str,
                                       export_format: str = 'json') -> Dict[str, Any]:
        """Traite une demande de portabilité (Art. 20 RGPD)"""
        
        try:
            # Audit de la demande
            audit_system = get_security_audit_system()
            await audit_system.log_event(
                event_type=AuditEventType.GDPR_REQUEST,
                user_id=user_id,
                details={
                    'request_type': 'portability',
                    'email': email,
                    'format': export_format
                }
            )
            
            # Collecter données portables
            portable_data = {
                'export_info': {
                    'user_id': user_id,
                    'email': email,
                    'export_date': datetime.now().isoformat(),
                    'format': export_format
                },
                'data': {}
            }
            
            for record_id, record in self.retention_manager.data_records.items():
                if record.owner_id == user_id and self._is_data_portable(record):
                    category = record.data_type
                    
                    if category not in portable_data['data']:
                        portable_data['data'][category] = []
                    
                    # Ajouter métadonnées minimales (pas les données chiffrées)
                    portable_data['data'][category].append({
                        'record_id': record_id,
                        'created_at': record.created_at.isoformat(),
                        'processing_purposes': [p.value for p in record.processing_purposes]
                    })
            
            # Log de l'export
            await audit_system.log_event(
                event_type=AuditEventType.DATA_EXPORT,
                user_id=user_id,
                details={
                    'reason': 'gdpr_portability_request',
                    'format': export_format,
                    'records_count': sum(len(records) for records in portable_data['data'].values())
                }
            )
            
            return {
                'status': 'success',
                'request_id': f"portability_{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'data': portable_data
            }
            
        except Exception as e:
            logger.error(f"Erreur traitement demande de portabilité {user_id}: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def _can_delete_record(self, record: DataRecord) -> bool:
        """Vérifie si un enregistrement peut être supprimé"""
        
        # Ne pas supprimer si obligations légales
        legal_purposes = {
            DataProcessingPurpose.COMPLIANCE,
            DataProcessingPurpose.SECURITY
        }
        
        if any(purpose in legal_purposes for purpose in record.processing_purposes):
            return False
        
        # Ne pas supprimer si rétention légale non expirée
        if record.retention_until and datetime.now() < record.retention_until:
            return False
        
        return True
    
    def _is_data_portable(self, record: DataRecord) -> bool:
        """Vérifie si des données sont portables selon RGPD"""
        
        # Données portables : traitement automatisé, consentement ou contrat
        portable_purposes = {
            DataProcessingPurpose.BUSINESS_INTELLIGENCE,
            DataProcessingPurpose.PROSPECTION,
            DataProcessingPurpose.ANALYTICS
        }
        
        return any(purpose in portable_purposes for purpose in record.processing_purposes)


class DataProtectionSystem:
    """Système de protection des données principal"""
    
    def __init__(self):
        self.encryption_manager = EncryptionManager()
        self.anonymizer = DataAnonymizer()
        self.retention_manager = DataRetentionManager()
        self.gdpr_manager = GDPRComplianceManager(
            self.encryption_manager,
            self.anonymizer,
            self.retention_manager
        )
        
        logger.info("🛡️ Système de protection des données initialisé")
    
    async def protect_sensitive_data(self, data: Dict[str, Any], 
                                   data_type: str,
                                   classification: DataClassification,
                                   owner_id: str,
                                   sensitive_fields: List[str],
                                   processing_purposes: List[DataProcessingPurpose]) -> Dict[str, Any]:
        """Protège des données sensibles avec chiffrement et audit"""
        
        try:
            # Enregistrer dans le gestionnaire de rétention
            record_id = f"{data_type}_{owner_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            record = self.retention_manager.register_data(
                record_id=record_id,
                data_type=data_type,
                classification=classification,
                owner_id=owner_id,
                processing_purposes=processing_purposes
            )
            
            # Choisir méthode de chiffrement selon classification
            key_id = "default_fernet"
            if classification in [DataClassification.RESTRICTED, DataClassification.SECRET]:
                key_id = "sensitive_aes"
            
            protected_data = data.copy()
            
            # Chiffrer champs sensibles
            for field in sensitive_fields:
                if field in protected_data and protected_data[field] is not None:
                    original_value = str(protected_data[field])
                    
                    encrypted_data, used_key_id = self.encryption_manager.encrypt_data(
                        original_value, key_id
                    )
                    
                    # Encoder en base64 pour stockage
                    protected_data[field] = {
                        'encrypted': base64.b64encode(encrypted_data).decode('utf-8'),
                        'key_id': used_key_id,
                        'encrypted_at': datetime.now().isoformat()
                    }
                    
                    record.encrypted_fields.add(field)
            
            # Audit
            audit_system = get_security_audit_system()
            await audit_system.log_event(
                event_type=AuditEventType.DATA_ACCESS,
                user_id=owner_id,
                resource_type=ResourceType.COMPANY,
                resource_id=record_id,
                action=Action.CREATE,
                details={
                    'data_type': data_type,
                    'classification': classification.value,
                    'encrypted_fields': sensitive_fields,
                    'processing_purposes': [p.value for p in processing_purposes]
                }
            )
            
            protected_data['_protection_metadata'] = {
                'record_id': record_id,
                'protected_at': datetime.now().isoformat(),
                'classification': classification.value,
                'encrypted_fields': list(record.encrypted_fields)
            }
            
            return protected_data
            
        except Exception as e:
            logger.error(f"Erreur protection données: {e}")
            raise
    
    async def access_protected_data(self, protected_data: Dict[str, Any], 
                                  user_id: str,
                                  access_purpose: str) -> Dict[str, Any]:
        """Accède aux données protégées avec déchiffrement et audit"""
        
        try:
            if '_protection_metadata' not in protected_data:
                return protected_data  # Données non protégées
            
            metadata = protected_data['_protection_metadata']
            record_id = metadata['record_id']
            encrypted_fields = metadata.get('encrypted_fields', [])
            
            decrypted_data = protected_data.copy()
            
            # Déchiffrer champs
            for field in encrypted_fields:
                if field in decrypted_data and isinstance(decrypted_data[field], dict):
                    encrypted_info = decrypted_data[field]
                    
                    if 'encrypted' in encrypted_info and 'key_id' in encrypted_info:
                        encrypted_bytes = base64.b64decode(encrypted_info['encrypted'])
                        
                        decrypted_bytes = self.encryption_manager.decrypt_data(
                            encrypted_bytes,
                            encrypted_info['key_id']
                        )
                        
                        decrypted_data[field] = decrypted_bytes.decode('utf-8')
            
            # Audit de l'accès
            audit_system = get_security_audit_system()
            await audit_system.log_event(
                event_type=AuditEventType.DATA_ACCESS,
                user_id=user_id,
                resource_type=ResourceType.COMPANY,
                resource_id=record_id,
                action=Action.READ,
                details={
                    'access_purpose': access_purpose,
                    'decrypted_fields': encrypted_fields
                }
            )
            
            # Mettre à jour log d'accès du record
            if record_id in self.retention_manager.data_records:
                record = self.retention_manager.data_records[record_id]
                record.access_log.append({
                    'user_id': user_id,
                    'timestamp': datetime.now().isoformat(),
                    'purpose': access_purpose,
                    'accessed_fields': encrypted_fields
                })
            
            return decrypted_data
            
        except Exception as e:
            logger.error(f"Erreur accès données protégées: {e}")
            raise


# Instance globale
_data_protection_system: Optional[DataProtectionSystem] = None


def get_data_protection_system() -> DataProtectionSystem:
    """Factory pour obtenir le système de protection des données"""
    global _data_protection_system
    
    if _data_protection_system is None:
        _data_protection_system = DataProtectionSystem()
    
    return _data_protection_system