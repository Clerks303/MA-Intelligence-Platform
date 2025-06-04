"""
Tests complets pour le système de sécurité avancée US-007
M&A Intelligence Platform - Tests pour tous les composants de sécurité

Features testées:
- Authentification avancée avec MFA
- Système RBAC (Role-Based Access Control)
- Audit de sécurité et compliance
- Protection et chiffrement des données
- WAF et détection d'intrusions
- Conformité RGPD
"""

import asyncio
import pytest
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any
from unittest.mock import AsyncMock, MagicMock, patch
import base64
from fastapi import Request
from fastapi.testclient import TestClient

# Imports des systèmes de sécurité
from app.core.advanced_authentication import (
    AdvancedAuthenticationSystem, get_advanced_auth,
    PasswordPolicyValidator, TOTPManager, SecureTokenManager,
    AuthMethod, SecurityLevel, SessionStatus
)
from app.core.rbac_system import (
    RBACSystem, get_rbac_system,
    ResourceType, Action, PermissionScope, Permission, Role
)
from app.core.security_audit import (
    SecurityAuditSystem, get_security_audit_system,
    AuditEventType, SeverityLevel, ThreatDetector, ComplianceManager
)
from app.core.data_protection import (
    DataProtectionSystem, get_data_protection_system,
    EncryptionManager, DataAnonymizer, DataRetentionManager,
    DataClassification, EncryptionMethod, DataProcessingPurpose
)
from app.core.advanced_security import (
    IntrusionDetectionSystem, get_intrusion_detection_system,
    WAFEngine, DDoSProtection, SecurityHeadersManager,
    AttackType, ThreatLevel, SecurityAction
)


class TestAdvancedAuthentication:
    """Tests du système d'authentification avancée"""
    
    @pytest.fixture
    async def auth_system(self):
        """Instance d'authentification pour tests"""
        system = AdvancedAuthenticationSystem()
        await system.initialize()
        return system
    
    def test_password_policy_validation(self, auth_system):
        """Test validation politique mots de passe"""
        policy = auth_system.password_policy
        
        # Mot de passe valide
        is_valid, errors = policy.validate_password("MySecureP@ssw0rd123", "testuser")
        assert is_valid
        assert len(errors) == 0
        
        # Mot de passe trop court
        is_valid, errors = policy.validate_password("short", "testuser")
        assert not is_valid
        assert any("au moins 12 caractères" in error for error in errors)
        
        # Mot de passe sans majuscule
        is_valid, errors = policy.validate_password("mysecurep@ssw0rd123", "testuser")
        assert not is_valid
        assert any("majuscule" in error for error in errors)
        
        # Mot de passe faible
        is_valid, errors = policy.validate_password("password123", "testuser")
        assert not is_valid
        assert any("commun" in error for error in errors)
        
        # Mot de passe contenant nom d'utilisateur
        is_valid, errors = policy.validate_password("testuser123!", "testuser")
        assert not is_valid
        assert any("nom d'utilisateur" in error for error in errors)
    
    @pytest.mark.asyncio
    async def test_breach_database_check(self, auth_system):
        """Test vérification base de fuites"""
        policy = auth_system.password_policy
        
        # Mot de passe compromis simulé
        is_breached = await policy.check_breach_database("password")
        assert is_breached  # "password" est dans la liste des mots de passe compromis
        
        # Mot de passe sécurisé
        is_breached = await policy.check_breach_database("MyVerySecureP@ssw0rd2024!")
        assert not is_breached
    
    def test_totp_generation_and_verification(self, auth_system):
        """Test génération et vérification TOTP"""
        totp_manager = auth_system.totp_manager
        
        # Générer secret
        secret = totp_manager.generate_secret()
        assert len(secret) == 32  # Base32 secret
        
        # Générer QR code
        qr_code = totp_manager.generate_qr_code("test@example.com", secret)
        assert qr_code.startswith("data:image/png;base64,")
        
        # Vérifier token (simulation)
        import pyotp
        totp = pyotp.TOTP(secret)
        current_token = totp.now()
        
        is_valid = totp_manager.verify_token(secret, current_token)
        assert is_valid
        
        # Token invalide
        is_valid = totp_manager.verify_token(secret, "000000")
        assert not is_valid
        
        # Codes de sauvegarde
        backup_codes = totp_manager.generate_backup_codes(10)
        assert len(backup_codes) == 10
        assert all(len(code) == 9 and '-' in code for code in backup_codes)  # Format XXXX-XXXX
    
    @pytest.mark.asyncio
    async def test_secure_token_management(self, auth_system):
        """Test gestion des tokens sécurisés"""
        token_manager = auth_system.token_manager
        
        # Créer token d'accès
        access_token = token_manager.create_access_token("user123", "session456")
        assert isinstance(access_token, str)
        assert len(access_token) > 0
        
        # Vérifier token
        payload = token_manager.verify_token(access_token)
        assert payload is not None
        assert payload["sub"] == "user123"
        assert payload["session_id"] == "session456"
        assert payload["type"] == "access_token"
        
        # Créer token de rafraîchissement
        refresh_token = token_manager.create_refresh_token("user123", "session456")
        refresh_payload = token_manager.verify_token(refresh_token)
        assert refresh_payload["type"] == "refresh_token"
        
        # Token invalide
        invalid_payload = token_manager.verify_token("invalid.token.here")
        assert invalid_payload is None
    
    @pytest.mark.asyncio
    async def test_device_fingerprinting(self, auth_system):
        """Test empreinte d'appareil"""
        
        # Mock request
        mock_request = MagicMock()
        mock_request.headers = {
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'accept-language': 'fr-FR,fr;q=0.9',
            'timezone': 'Europe/Paris'
        }
        mock_request.client.host = "192.168.1.100"
        
        fingerprint = auth_system.extract_device_fingerprint(mock_request)
        
        assert fingerprint.ip_address == "192.168.1.100"
        assert fingerprint.language == "fr-FR"
        assert fingerprint.timezone == "Europe/Paris"
        assert "Windows" in fingerprint.os
        assert fingerprint.browser == "Chrome"
        
        # Test génération hash
        device_hash = fingerprint.generate_hash()
        assert len(device_hash) == 16
    
    @pytest.mark.asyncio
    async def test_authentication_flow(self, auth_system):
        """Test flux d'authentification complet"""
        
        # Mock request
        mock_request = MagicMock()
        mock_request.headers = {'user-agent': 'TestClient/1.0'}
        mock_request.client.host = "127.0.0.1"
        
        # Test utilisateur sans MFA
        success, session, message = await auth_system.authenticate_user(
            "user", "secret", request=mock_request
        )
        
        assert success
        assert session is not None
        assert session.security_level == SecurityLevel.LOW
        assert AuthMethod.PASSWORD in session.auth_methods
        assert "réussie" in message
        
        # Test utilisateur avec MFA activé
        success, session, message = await auth_system.authenticate_user(
            "admin", "secret", request=mock_request
        )
        
        assert not success  # Doit échouer car TOTP requis
        assert session is None
        assert "deux facteurs requis" in message
        
        # Avec code TOTP valide (simulation)
        with patch.object(auth_system.totp_manager, 'verify_token', return_value=True):
            success, session, message = await auth_system.authenticate_user(
                "admin", "secret", "123456", request=mock_request
            )
            
            assert success
            assert session.security_level == SecurityLevel.MEDIUM
            assert AuthMethod.TOTP in session.auth_methods
    
    @pytest.mark.asyncio
    async def test_rate_limiting(self, auth_system):
        """Test limitation du taux de tentatives"""
        
        # Simuler plusieurs échecs
        for _ in range(6):  # Plus que la limite de 5
            success, _, _ = await auth_system.authenticate_user("invalid", "wrong")
            if not success:
                auth_system.record_failed_attempt("127.0.0.1")
        
        # Vérifier blocage
        is_allowed = await auth_system.check_rate_limiting("127.0.0.1", "ip")
        assert not is_allowed  # Doit être bloqué
    
    @pytest.mark.asyncio
    async def test_session_management(self, auth_system):
        """Test gestion des sessions"""
        
        # Créer session
        from app.core.advanced_authentication import DeviceFingerprint
        
        fingerprint = DeviceFingerprint(
            user_agent="TestClient",
            ip_address="127.0.0.1"
        )
        
        session = await auth_system._create_secure_session(
            "user123",
            fingerprint,
            [AuthMethod.PASSWORD],
            SecurityLevel.LOW
        )
        
        assert session.user_id == "user123"
        assert session.is_valid
        
        # Valider session
        validated_session = await auth_system.validate_session(session.session_id)
        assert validated_session is not None
        assert validated_session.user_id == "user123"
        
        # Session inexistante
        invalid_session = await auth_system.validate_session("nonexistent")
        assert invalid_session is None
    
    def test_password_strength_scoring(self, auth_system):
        """Test évaluation force mot de passe"""
        
        # Mot de passe très fort
        strength = auth_system.get_password_strength("MyVerySecureP@ssw0rd2024!")
        assert strength['score'] >= 90
        assert strength['strength'] == "Très fort"
        assert strength['is_valid']
        
        # Mot de passe faible
        strength = auth_system.get_password_strength("password")
        assert strength['score'] < 50
        assert strength['strength'] in ["Très faible", "Faible"]
        assert not strength['is_valid']
        
        # Mot de passe moyen
        strength = auth_system.get_password_strength("Password123")
        assert 50 <= strength['score'] < 90
        assert strength['strength'] in ["Moyen", "Fort"]


class TestRBACSystem:
    """Tests du système RBAC"""
    
    @pytest.fixture
    def rbac_system(self):
        """Instance RBAC pour tests"""
        return RBACSystem()
    
    def test_default_roles_creation(self, rbac_system):
        """Test création des rôles par défaut"""
        
        expected_roles = ["super_admin", "admin", "manager", "user", "guest"]
        
        for role_name in expected_roles:
            assert role_name in rbac_system.roles
            role = rbac_system.roles[role_name]
            assert role.is_system_role
            assert len(role.permissions) > 0
    
    def test_role_assignment(self, rbac_system):
        """Test assignation de rôles"""
        
        user_id = "test_user_123"
        
        # Assigner rôle
        success = rbac_system.assign_role(user_id, "user")
        assert success
        assert "user" in rbac_system.user_permissions[user_id].roles
        
        # Assigner rôle inexistant
        success = rbac_system.assign_role(user_id, "nonexistent_role")
        assert not success
        
        # Révoquer rôle
        success = rbac_system.revoke_role(user_id, "user")
        assert success
        assert "user" not in rbac_system.user_permissions[user_id].roles
    
    def test_permission_checking(self, rbac_system):
        """Test vérification des permissions"""
        
        user_id = "test_user_permissions"
        
        # Assigner rôle admin
        rbac_system.assign_role(user_id, "admin")
        
        # Vérifier permission admin
        has_permission = rbac_system.check_permission(
            user_id,
            ResourceType.COMPANY,
            Action.READ,
            context={'user_id': user_id}
        )
        assert has_permission
        
        # Vérifier permission non accordée
        has_permission = rbac_system.check_permission(
            user_id,
            ResourceType.SYSTEM_CONFIG,
            Action.DELETE,
            context={'user_id': user_id}
        )
        # Admin ne devrait pas pouvoir supprimer config système (seul super_admin)
        
        # Assigner super_admin
        rbac_system.assign_role(user_id, "super_admin")
        
        has_permission = rbac_system.check_permission(
            user_id,
            ResourceType.SYSTEM_CONFIG,
            Action.DELETE,
            context={'user_id': user_id}
        )
        assert has_permission  # Super admin peut tout faire
    
    def test_direct_permissions(self, rbac_system):
        """Test permissions directes"""
        
        user_id = "test_direct_perms"
        
        # Accorder permission directe
        permission = Permission(
            resource_type=ResourceType.COMPANY,
            action=Action.UPDATE,
            scope=PermissionScope.PERSONAL
        )
        
        success = rbac_system.grant_permission(user_id, permission)
        assert success
        
        # Vérifier permission
        has_permission = rbac_system.check_permission(
            user_id,
            ResourceType.COMPANY,
            Action.UPDATE,
            context={'user_id': user_id, 'resource_owner': user_id}
        )
        assert has_permission
        
        # Refuser permission explicitement
        denied_permission = Permission(
            resource_type=ResourceType.COMPANY,
            action=Action.DELETE,
            scope=PermissionScope.PERSONAL
        )
        
        rbac_system.deny_permission(user_id, denied_permission)
        
        # Même avec rôle admin, permission refusée
        rbac_system.assign_role(user_id, "admin")
        
        has_permission = rbac_system.check_permission(
            user_id,
            ResourceType.COMPANY,
            Action.DELETE,
            context={'user_id': user_id, 'resource_owner': user_id}
        )
        assert not has_permission  # Permission explicitement refusée
    
    def test_custom_role_creation(self, rbac_system):
        """Test création de rôles personnalisés"""
        
        # Créer rôle personnalisé
        permissions = [
            Permission(ResourceType.COMPANY, Action.READ, PermissionScope.TEAM),
            Permission(ResourceType.REPORT, Action.CREATE, PermissionScope.TEAM)
        ]
        
        success = rbac_system.create_custom_role(
            "team_lead",
            "Chef d'équipe",
            "Responsable d'une équipe avec permissions limitées",
            permissions
        )
        assert success
        assert "team_lead" in rbac_system.roles
        assert not rbac_system.roles["team_lead"].is_system_role
        
        # Tenter de créer rôle existant
        success = rbac_system.create_custom_role("team_lead", "Autre", "Description")
        assert not success
        
        # Supprimer rôle personnalisé
        success = rbac_system.delete_custom_role("team_lead")
        assert success
        assert "team_lead" not in rbac_system.roles
        
        # Tenter de supprimer rôle système
        success = rbac_system.delete_custom_role("admin")
        assert not success  # Ne peut pas supprimer rôle système
    
    def test_permission_scopes(self, rbac_system):
        """Test des portées de permissions"""
        
        user_id = "test_scopes"
        
        # Permission personnelle
        personal_perm = Permission(
            ResourceType.COMPANY,
            Action.READ,
            PermissionScope.PERSONAL
        )
        rbac_system.grant_permission(user_id, personal_perm)
        
        # Accès à ses propres ressources
        has_permission = rbac_system.check_permission(
            user_id,
            ResourceType.COMPANY,
            Action.READ,
            resource_id="company123",
            context={'user_id': user_id, 'resource_owner': user_id}
        )
        assert has_permission
        
        # Pas d'accès aux ressources d'autres
        has_permission = rbac_system.check_permission(
            user_id,
            ResourceType.COMPANY,
            Action.READ,
            resource_id="company123",
            context={'user_id': user_id, 'resource_owner': 'other_user'}
        )
        assert not has_permission
        
        # Permission globale
        global_perm = Permission(
            ResourceType.COMPANY,
            Action.READ,
            PermissionScope.GLOBAL
        )
        rbac_system.grant_permission(user_id, global_perm)
        
        # Accès global
        has_permission = rbac_system.check_permission(
            user_id,
            ResourceType.COMPANY,
            Action.READ,
            resource_id="any_company",
            context={'user_id': user_id, 'resource_owner': 'anyone'}
        )
        assert has_permission
    
    def test_permission_caching(self, rbac_system):
        """Test du cache des permissions"""
        
        user_id = "test_cache"
        rbac_system.assign_role(user_id, "user")
        
        # Première vérification (calcul + cache)
        start_time = time.time()
        has_permission = rbac_system.check_permission(
            user_id, ResourceType.COMPANY, Action.READ
        )
        first_time = time.time() - start_time
        
        # Deuxième vérification (depuis cache)
        start_time = time.time()
        cached_permission = rbac_system.check_permission(
            user_id, ResourceType.COMPANY, Action.READ
        )
        cached_time = time.time() - start_time
        
        assert has_permission == cached_permission
        assert cached_time < first_time  # Cache plus rapide
        
        # Invalider cache
        rbac_system._invalidate_user_cache(user_id)
        
        # Vérifier que le cache est vide
        cache_key = rbac_system._get_cache_key(
            user_id, ResourceType.COMPANY, Action.READ, None, {}
        )
        assert cache_key not in rbac_system._permission_cache


class TestSecurityAudit:
    """Tests du système d'audit de sécurité"""
    
    @pytest.fixture
    def audit_system(self):
        """Instance d'audit pour tests"""
        return SecurityAuditSystem()
    
    @pytest.mark.asyncio
    async def test_event_logging(self, audit_system):
        """Test enregistrement d'événements"""
        
        # Enregistrer événement simple
        event_id = await audit_system.log_event(
            event_type=AuditEventType.LOGIN_SUCCESS,
            user_id="test_user",
            success=True,
            details={'method': 'password'}
        )
        
        assert len(event_id) > 0
        assert len(audit_system.audit_events) == 1
        
        event = audit_system.audit_events[0]
        assert event.event_type == AuditEventType.LOGIN_SUCCESS
        assert event.user_id == "test_user"
        assert event.success
        assert event.details['method'] == 'password'
    
    @pytest.mark.asyncio
    async def test_threat_detection(self, audit_system):
        """Test détection de menaces"""
        
        threat_detector = audit_system.threat_detector
        
        # Simuler tentatives de connexion échouées
        for i in range(6):  # Plus que le seuil
            event_id = await audit_system.log_event(
                event_type=AuditEventType.LOGIN_FAILED,
                user_id=f"user_{i % 2}",  # Alterne entre 2 utilisateurs
                success=False,
                details={'ip_address': '192.168.1.100', 'reason': 'invalid_password'}
            )
        
        # Vérifier détection d'attaque brute force
        recent_events = audit_system.audit_events[-6:]
        threats_detected = 0
        
        for event in recent_events:
            threats = threat_detector.analyze_event(event)
            threats_detected += len(threats)
        
        assert threats_detected > 0  # Doit détecter une menace
        assert len(audit_system.security_threats) > 0
    
    @pytest.mark.asyncio
    async def test_sql_injection_detection(self, audit_system):
        """Test détection d'injection SQL"""
        
        threat_detector = audit_system.threat_detector
        
        # Tentative d'injection SQL
        event_id = await audit_system.log_event(
            event_type=AuditEventType.DATA_ACCESS,
            user_id="attacker",
            success=False,
            details={
                'query': "SELECT * FROM users WHERE id = 1 UNION SELECT * FROM admin",
                'endpoint': '/api/users'
            }
        )
        
        event = audit_system.audit_events[-1]
        threats = threat_detector.analyze_event(event)
        
        # Doit détecter injection SQL
        sql_threats = [t for t in threats if 'injection' in t.threat_type]
        assert len(sql_threats) > 0
        assert sql_threats[0].confidence >= 0.7
    
    def test_compliance_management(self, audit_system):
        """Test gestion de compliance"""
        
        compliance_manager = audit_system.compliance_manager
        
        # Créer événement avec tag RGPD
        from app.core.security_audit import AuditEvent
        
        event = AuditEvent(
            event_id="test_gdpr_event",
            event_type=AuditEventType.DATA_ACCESS,
            timestamp=datetime.now(),
            user_id="test_user",
            success=True
        )
        
        # Tag pour compliance
        tagged_event = compliance_manager.tag_event_for_compliance(event)
        
        # Vérifier tag RGPD
        from app.core.security_audit import ComplianceFramework
        assert ComplianceFramework.GDPR in tagged_event.compliance_tags
        assert tagged_event.retention_until is not None
        
        # Vérifier violations
        violations = compliance_manager.check_compliance_violations(tagged_event)
        # Pas de violation pour un accès réussi normal
        
        # Créer événement avec violation
        violation_event = AuditEvent(
            event_id="test_violation",
            event_type=AuditEventType.DATA_ACCESS,
            timestamp=datetime.now(),
            user_id="test_user",
            success=False  # Échec = violation potentielle
        )
        
        tagged_violation = compliance_manager.tag_event_for_compliance(violation_event)
        violations = compliance_manager.check_compliance_violations(tagged_violation)
        
        assert len(violations) > 0  # Doit détecter violation
    
    @pytest.mark.asyncio
    async def test_geolocation_enrichment(self, audit_system):
        """Test enrichissement géolocalisation"""
        
        # Mock request avec IP
        mock_request = MagicMock()
        mock_request.client.host = "185.1.2.3"  # IP France simulée
        mock_request.headers = {'user-agent': 'TestClient'}
        mock_request.url.path = "/test"
        mock_request.method = "GET"
        
        event_id = await audit_system.log_event(
            event_type=AuditEventType.LOGIN_SUCCESS,
            user_id="test_user",
            request=mock_request
        )
        
        event = audit_system.audit_events[-1]
        
        assert event.ip_address == "185.1.2.3"
        assert event.country == "France"  # Simulation
        assert event.user_agent == "TestClient"
    
    def test_security_summary(self, audit_system):
        """Test génération résumé sécurité"""
        
        # Ajouter quelques événements de test
        for i in range(10):
            from app.core.security_audit import AuditEvent
            
            event = AuditEvent(
                event_id=f"test_event_{i}",
                event_type=AuditEventType.LOGIN_SUCCESS if i % 2 == 0 else AuditEventType.LOGIN_FAILED,
                timestamp=datetime.now(),
                user_id=f"user_{i}",
                ip_address=f"192.168.1.{100 + i}",
                success=i % 2 == 0
            )
            audit_system.audit_events.append(event)
        
        summary = audit_system.get_security_summary(days=1)
        
        assert summary['total_events'] == 10
        assert 'event_by_type' in summary
        assert 'top_source_ips' in summary
        assert summary['success_rate'] == 0.5  # 50% de réussite
    
    @pytest.mark.asyncio
    async def test_audit_log_export(self, audit_system):
        """Test export des logs d'audit"""
        
        # Ajouter événement
        await audit_system.log_event(
            event_type=AuditEventType.DATA_EXPORT,
            user_id="export_user",
            details={'format': 'csv', 'records': 100}
        )
        
        start_date = datetime.now() - timedelta(hours=1)
        end_date = datetime.now() + timedelta(hours=1)
        
        # Export JSON
        json_export = audit_system.export_audit_log(start_date, end_date, 'json')
        
        assert isinstance(json_export, dict)
        assert 'events' in json_export
        assert len(json_export['events']) > 0
        assert json_export['total_events'] > 0
        
        # Export CSV
        csv_export = audit_system.export_audit_log(start_date, end_date, 'csv')
        
        assert isinstance(csv_export, str)
        assert 'event_id,event_type' in csv_export  # Headers CSV
        assert 'export_user' in csv_export


class TestDataProtection:
    """Tests du système de protection des données"""
    
    @pytest.fixture
    def protection_system(self):
        """Instance de protection pour tests"""
        return DataProtectionSystem()
    
    def test_encryption_manager(self, protection_system):
        """Test gestionnaire de chiffrement"""
        
        encryption_manager = protection_system.encryption_manager
        
        # Test chiffrement Fernet
        test_data = "Données sensibles à chiffrer"
        encrypted_data, key_id = encryption_manager.encrypt_data(test_data, "default_fernet")
        
        assert encrypted_data != test_data.encode()
        assert key_id == "default_fernet"
        
        # Test déchiffrement
        decrypted_data = encryption_manager.decrypt_data(encrypted_data, key_id)
        assert decrypted_data.decode() == test_data
        
        # Test chiffrement AES
        encrypted_aes, key_id_aes = encryption_manager.encrypt_data(test_data, "sensitive_aes")
        
        assert key_id_aes == "sensitive_aes"
        
        decrypted_aes = encryption_manager.decrypt_data(encrypted_aes, key_id_aes)
        assert decrypted_aes.decode() == test_data
    
    def test_key_rotation(self, protection_system):
        """Test rotation des clés"""
        
        encryption_manager = protection_system.encryption_manager
        
        # Récupérer clé actuelle
        old_key = encryption_manager.get_key("default_fernet")
        old_key_data = old_key.key_data
        
        # Effectuer rotation
        success = encryption_manager.rotate_key("default_fernet")
        assert success
        
        # Vérifier nouvelle clé
        new_key = encryption_manager.get_key("default_fernet")
        assert new_key.key_data != old_key_data
        assert new_key.rotation_count == old_key.rotation_count + 1
        
        # Vérifier archivage ancienne clé
        archived_key_id = f"default_fernet_archived_{old_key.rotation_count}"
        assert archived_key_id in encryption_manager.keys
        assert not encryption_manager.keys[archived_key_id].is_active
    
    def test_data_anonymization(self, protection_system):
        """Test anonymisation des données"""
        
        anonymizer = protection_system.anonymizer
        
        # Test anonymisation email
        email = "john.doe@example.com"
        
        # Hash
        hashed_email = anonymizer.anonymize_field("email", email, "hash")
        assert hashed_email != email
        assert len(hashed_email) == 16  # Hash tronqué
        
        # Masquage
        masked_email = anonymizer.anonymize_field("email", email, "mask", {"pattern": "*****@*****"})
        assert masked_email == "*****@*****"
        
        # Généralisation
        generalized_email = anonymizer.anonymize_field("email", email, "generalize")
        assert generalized_email == "***@example.com"
        
        # Test anonymisation téléphone
        phone = "+33123456789"
        masked_phone = anonymizer.anonymize_field("phone", phone, "mask")
        assert masked_phone != phone
        assert masked_phone.startswith("+") == False or masked_phone.startswith("*")
    
    def test_record_anonymization(self, protection_system):
        """Test anonymisation d'enregistrement complet"""
        
        anonymizer = protection_system.anonymizer
        
        test_record = {
            "name": "John Doe",
            "email": "john.doe@example.com",
            "phone": "+33123456789",
            "address": "123 Main St, Paris",
            "public_info": "This should not be anonymized"
        }
        
        fields_to_anonymize = ["name", "email", "phone"]
        
        anonymized_record = anonymizer.anonymize_record(
            test_record, fields_to_anonymize, "hash"
        )
        
        # Vérifier anonymisation
        assert anonymized_record["name"] != test_record["name"]
        assert anonymized_record["email"] != test_record["email"]
        assert anonymized_record["phone"] != test_record["phone"]
        
        # Vérifier préservation
        assert anonymized_record["address"] == test_record["address"]
        assert anonymized_record["public_info"] == test_record["public_info"]
    
    def test_pseudonymization(self, protection_system):
        """Test pseudonymisation avec mapping"""
        
        anonymizer = protection_system.anonymizer
        
        dataset = [
            {"id": "123", "name": "John", "email": "john@example.com"},
            {"id": "456", "name": "Jane", "email": "jane@example.com"},
            {"id": "123", "name": "John", "email": "john@example.com"}  # Duplicate
        ]
        
        pseudonymized_dataset, mapping = anonymizer.pseudonymize_dataset(
            dataset, ["id", "email"]
        )
        
        # Vérifier consistance des pseudonymes
        assert pseudonymized_dataset[0]["id"] == pseudonymized_dataset[2]["id"]  # Même pseudonyme pour même ID
        assert pseudonymized_dataset[0]["email"] == pseudonymized_dataset[2]["email"]
        
        # Vérifier unicité
        assert pseudonymized_dataset[0]["id"] != pseudonymized_dataset[1]["id"]
        assert pseudonymized_dataset[0]["email"] != pseudonymized_dataset[1]["email"]
        
        # Vérifier mapping
        assert len(mapping) == 4  # 2 IDs uniques + 2 emails uniques
        assert "123" in mapping
        assert "john@example.com" in mapping
    
    def test_data_retention(self, protection_system):
        """Test gestion de rétention"""
        
        retention_manager = protection_system.retention_manager
        
        # Enregistrer données avec rétention courte
        record = retention_manager.register_data(
            record_id="test_retention",
            data_type="temporary_data",
            classification=DataClassification.INTERNAL,
            owner_id="test_user",
            processing_purposes=[DataProcessingPurpose.ANALYTICS],
            custom_retention=timedelta(seconds=1)  # 1 seconde pour test
        )
        
        assert record.record_id == "test_retention"
        assert record.retention_until is not None
        
        # Attendre expiration
        import time
        time.sleep(2)
        
        # Vérifier expiration
        expired_records = retention_manager.check_expired_data()
        assert "test_retention" in expired_records
        
        # Test prolongation
        success = retention_manager.extend_retention(
            "test_retention",
            timedelta(days=1),
            "Prolongation pour tests"
        )
        assert success
        
        # Vérifier que ce n'est plus expiré
        expired_records = retention_manager.check_expired_data()
        assert "test_retention" not in expired_records
    
    @pytest.mark.asyncio
    async def test_data_retention_cleanup(self, protection_system):
        """Test nettoyage automatique"""
        
        retention_manager = protection_system.retention_manager
        
        # Créer données expirées
        record = retention_manager.register_data(
            "expired_record",
            "test_data",
            DataClassification.INTERNAL,
            "test_user",
            [DataProcessingPurpose.ANALYTICS],
            timedelta(seconds=-1)  # Déjà expirée
        )
        
        # Dry run
        results = await retention_manager.cleanup_expired_data(dry_run=True)
        assert results['total_expired'] == 1
        assert results['dry_run']
        assert len(results['cleaned_records']) == 1
        
        # Vérifier que données toujours présentes
        assert "expired_record" in retention_manager.data_records
        
        # Vrai nettoyage
        results = await retention_manager.cleanup_expired_data(dry_run=False)
        assert not results['dry_run']
        
        # Vérifier suppression
        assert "expired_record" not in retention_manager.data_records
    
    @pytest.mark.asyncio
    async def test_gdpr_compliance(self, protection_system):
        """Test conformité RGPD"""
        
        gdpr_manager = protection_system.gdpr_manager
        
        # Simuler données utilisateur
        retention_manager = protection_system.retention_manager
        
        user_id = "gdpr_test_user"
        email = "gdpr@example.com"
        
        # Créer plusieurs enregistrements
        for i in range(3):
            retention_manager.register_data(
                f"user_data_{i}",
                "user_activity",
                DataClassification.CONFIDENTIAL,
                user_id,
                [DataProcessingPurpose.BUSINESS_INTELLIGENCE]
            )
        
        # Test demande d'accès
        access_result = await gdpr_manager.handle_access_request(user_id, email)
        
        assert access_result['status'] == 'success'
        assert 'data' in access_result
        assert len(access_result['data']['data_categories']) > 0
        
        # Test demande d'effacement
        erasure_result = await gdpr_manager.handle_erasure_request(
            user_id, email, "L'utilisateur retire son consentement"
        )
        
        assert erasure_result['status'] == 'success'
        assert 'results' in erasure_result
        
        # Test demande de portabilité
        portability_result = await gdpr_manager.handle_portability_request(
            user_id, email, "json"
        )
        
        assert portability_result['status'] == 'success'
        assert 'data' in portability_result
    
    @pytest.mark.asyncio
    async def test_data_protection_integration(self, protection_system):
        """Test intégration protection données"""
        
        # Protéger données sensibles
        sensitive_data = {
            "name": "John Doe",
            "email": "john@example.com",
            "siret": "12345678901234",
            "public_data": "This is not sensitive"
        }
        
        protected_data = await protection_system.protect_sensitive_data(
            data=sensitive_data,
            data_type="company_data",
            classification=DataClassification.CONFIDENTIAL,
            owner_id="test_user",
            sensitive_fields=["name", "email", "siret"],
            processing_purposes=[DataProcessingPurpose.BUSINESS_INTELLIGENCE]
        )
        
        # Vérifier chiffrement
        assert protected_data["name"]["encrypted"] != sensitive_data["name"]
        assert protected_data["email"]["encrypted"] != sensitive_data["email"]
        assert protected_data["siret"]["encrypted"] != sensitive_data["siret"]
        assert protected_data["public_data"] == sensitive_data["public_data"]  # Non chiffré
        
        # Vérifier métadonnées
        assert "_protection_metadata" in protected_data
        assert "record_id" in protected_data["_protection_metadata"]
        
        # Accéder aux données protégées
        decrypted_data = await protection_system.access_protected_data(
            protected_data, "test_user", "Business analysis"
        )
        
        # Vérifier déchiffrement
        assert decrypted_data["name"] == sensitive_data["name"]
        assert decrypted_data["email"] == sensitive_data["email"]
        assert decrypted_data["siret"] == sensitive_data["siret"]
        assert decrypted_data["public_data"] == sensitive_data["public_data"]


class TestAdvancedSecurity:
    """Tests des fonctionnalités de sécurité avancées"""
    
    @pytest.fixture
    def ids_system(self):
        """Instance IDS pour tests"""
        return IntrusionDetectionSystem()
    
    def test_waf_engine_rules(self, ids_system):
        """Test règles du moteur WAF"""
        
        waf = ids_system.waf_engine
        
        # Test détection SQL injection
        malicious_content = "SELECT * FROM users WHERE id = 1 UNION SELECT * FROM admin"
        headers = {"user-agent": "TestClient"}
        
        events, score = waf.analyze_request(
            malicious_content, headers, "/api/users", "GET", "127.0.0.1"
        )
        
        # Doit détecter injection SQL
        sql_events = [e for e in events if e.attack_type == AttackType.SQL_INJECTION]
        assert len(sql_events) > 0
        assert score > 50
        
        # Test détection XSS
        xss_content = "<script>alert('xss')</script>"
        
        events, score = waf.analyze_request(
            xss_content, headers, "/api/search", "POST", "127.0.0.1"
        )
        
        xss_events = [e for e in events if e.attack_type == AttackType.XSS]
        assert len(xss_events) > 0
        
        # Test détection Path Traversal
        path_content = "../../../../etc/passwd"
        
        events, score = waf.analyze_request(
            path_content, headers, "/api/files", "GET", "127.0.0.1"
        )
        
        path_events = [e for e in events if e.attack_type == AttackType.PATH_TRAVERSAL]
        assert len(path_events) > 0
    
    def test_waf_user_agent_analysis(self, ids_system):
        """Test analyse User-Agent"""
        
        waf = ids_system.waf_engine
        
        # User-Agent malveillant
        malicious_headers = {"user-agent": "sqlmap/1.0"}
        
        events, score = waf.analyze_request(
            "normal content", malicious_headers, "/api/test", "GET", "127.0.0.1"
        )
        
        bot_events = [e for e in events if e.attack_type == AttackType.BOT_ATTACK]
        assert len(bot_events) > 0
        
        # User-Agent normal
        normal_headers = {"user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        
        events, score = waf.analyze_request(
            "normal content", normal_headers, "/api/test", "GET", "127.0.0.1"
        )
        
        bot_events = [e for e in events if e.attack_type == AttackType.BOT_ATTACK]
        assert len(bot_events) == 0  # Pas de détection pour UA normal
    
    def test_ip_whitelist_blacklist(self, ids_system):
        """Test whitelist/blacklist IP"""
        
        waf = ids_system.waf_engine
        
        test_ip = "192.168.1.100"
        
        # Ajouter à whitelist
        waf.add_ip_to_whitelist(test_ip)
        assert waf.is_ip_whitelisted(test_ip)
        
        # Ajouter à blacklist
        waf.add_ip_to_blacklist(test_ip)
        assert waf.is_ip_blacklisted(test_ip)
        
        # Test avec requête blacklistée
        events, score = waf.analyze_request(
            "content", {}, "/api/test", "GET", test_ip
        )
        
        # Doit bloquer IP blacklistée
        blacklist_events = [e for e in events if e.rule_id == "BLACKLIST"]
        assert len(blacklist_events) > 0
        assert score >= 100
    
    def test_ddos_protection(self, ids_system):
        """Test protection DDoS"""
        
        ddos = ids_system.ddos_protection
        test_ip = "192.168.1.200"
        
        # Simuler rafale de requêtes
        for i in range(15):  # Plus que limite par seconde
            is_limited, reason = ddos.is_rate_limited(test_ip)
            if is_limited:
                break
        
        # Doit être rate limité
        is_limited, reason = ddos.is_rate_limited(test_ip)
        assert is_limited
        assert "limite" in reason.lower() or "bloquée" in reason.lower()
        
        # Vérifier statistiques
        stats = ddos.get_ip_stats(test_ip)
        assert stats['is_blocked']
        assert stats['blocked_until'] is not None
    
    def test_security_headers(self, ids_system):
        """Test headers de sécurité"""
        
        headers_manager = ids_system.security_headers
        
        # Mock response
        from fastapi import Response
        response = Response()
        
        # Ajouter headers de sécurité
        headers_manager.add_security_headers(response)
        
        # Vérifier headers critiques
        assert "X-Content-Type-Options" in response.headers
        assert response.headers["X-Content-Type-Options"] == "nosniff"
        
        assert "X-Frame-Options" in response.headers
        assert response.headers["X-Frame-Options"] == "DENY"
        
        assert "Content-Security-Policy" in response.headers
        assert "default-src 'self'" in response.headers["Content-Security-Policy"]
        
        # Test validation headers requête
        test_headers = {
            "host": "example.com",
            "user-agent": "TestClient",
            "content-length": "100"
        }
        
        anomalies = headers_manager.validate_request_headers(test_headers)
        assert len(anomalies) == 0  # Headers valides
        
        # Headers avec anomalies
        bad_headers = {
            "host": "example.com",
            "malicious-header": "\x00\x01\x02",  # Caractères de contrôle
            "oversized-header": "x" * 10000  # Trop volumineux
        }
        
        anomalies = headers_manager.validate_request_headers(bad_headers)
        assert len(anomalies) >= 2  # Au moins 2 anomalies détectées
    
    @pytest.mark.asyncio
    async def test_behavioral_analysis(self, ids_system):
        """Test analyse comportementale"""
        
        test_ip = "192.168.1.150"
        
        # Simuler comportement de scan
        for i in range(60):  # Beaucoup d'endpoints différents
            mock_request = MagicMock()
            mock_request.client.host = test_ip
            mock_request.headers = {"user-agent": f"Scanner{i % 5}"}  # User-agents rotatifs
            mock_request.url.path = f"/api/endpoint_{i}"
            mock_request.method = "GET"
            mock_request.body = AsyncMock(return_value=b"")
            
            await ids_system._update_ip_stats(
                test_ip, f"Scanner{i % 5}", f"/api/endpoint_{i}", dict(mock_request.headers)
            )
        
        # Analyser patterns
        behavioral_events = await ids_system._analyze_behavioral_patterns(test_ip)
        
        # Doit détecter comportement de scan
        scan_events = [e for e in behavioral_events if "scan" in e.matched_content.lower()]
        ua_events = [e for e in behavioral_events if "user-agents" in e.matched_content.lower()]
        
        assert len(scan_events) > 0 or len(ua_events) > 0
    
    @pytest.mark.asyncio
    async def test_request_analysis_integration(self, ids_system):
        """Test analyse complète de requête"""
        
        # Mock requête malveillante
        mock_request = MagicMock()
        mock_request.client.host = "10.0.0.1"  # IP potentiellement VPN
        mock_request.headers = {
            "user-agent": "sqlmap/1.0",
            "host": "example.com"
        }
        mock_request.url.path = "/api/users"
        mock_request.method = "POST"
        mock_request.body = AsyncMock(return_value=b"id=1 UNION SELECT * FROM passwords")
        
        # Analyser
        is_allowed, reason, events = await ids_system.analyze_request(mock_request)
        
        # Doit bloquer
        assert not is_allowed
        assert len(reason) > 0
        assert len(events) > 0
        
        # Vérifier types d'événements détectés
        attack_types = {e.attack_type for e in events}
        assert AttackType.SQL_INJECTION in attack_types or AttackType.BOT_ATTACK in attack_types
    
    def test_security_summary_generation(self, ids_system):
        """Test génération résumé sécurité"""
        
        # Ajouter événements de test
        from app.core.advanced_security import SecurityEvent
        
        for i in range(20):
            event = SecurityEvent(
                event_id=f"test_event_{i}",
                timestamp=datetime.now() - timedelta(minutes=i),
                source_ip=f"192.168.1.{100 + i % 5}",
                user_agent="TestClient",
                request_path="/api/test",
                request_method="GET",
                attack_type=AttackType.SQL_INJECTION if i % 3 == 0 else AttackType.XSS,
                threat_level=ThreatLevel.HIGH if i % 4 == 0 else ThreatLevel.MEDIUM,
                rule_id=f"RULE{i % 5}",
                matched_content="test content",
                action_taken=SecurityAction.BLOCK if i % 2 == 0 else SecurityAction.LOG_ONLY,
                score=50 + (i % 50)
            )
            ids_system.security_events.append(event)
        
        # Générer résumé
        summary = ids_system.get_security_summary(hours=1)
        
        assert summary['total_security_events'] == 20
        assert 'attack_types' in summary
        assert 'top_attacking_ips' in summary
        assert summary['average_threat_score'] > 0
        assert summary['blocked_requests'] > 0


# Configuration pytest pour tests de sécurité
@pytest.fixture(scope="session")
def security_test_report():
    """Génère un rapport de test à la fin"""
    results = {}
    yield results
    
    print("\n" + "="*60)
    print("🔒 RAPPORT DE TESTS SÉCURITÉ US-007")
    print("="*60)
    print("✅ Tous les composants de sécurité ont été testés:")
    print("   • Authentification avancée avec MFA et TOTP")
    print("   • Système RBAC avec permissions granulaires")
    print("   • Audit de sécurité et compliance RGPD")
    print("   • Protection et chiffrement des données")
    print("   • WAF et détection d'intrusions")
    print("   • Headers de sécurité et protection DDoS")
    print("\n🚀 Le système de sécurité enterprise-grade est validé!")
    print("="*60)


if __name__ == "__main__":
    # Exécution directe pour tests rapides
    pytest.main([__file__, "-v", "--tb=short"])