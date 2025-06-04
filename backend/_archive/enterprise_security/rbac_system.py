"""
Système de contrôle d'accès basé sur les rôles (RBAC) pour M&A Intelligence Platform
US-007: Gestion granulaire des permissions et autorisations

Features:
- Rôles et permissions granulaires
- Gestion hiérarchique des rôles
- Contrôle d'accès basé sur les ressources
- Audit des accès et permissions
- Support multi-tenant et contextes
- Policies dynamiques et conditions
"""

import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
import json
import hashlib
from pathlib import Path

from fastapi import HTTPException, Request, Depends, status
from pydantic import BaseModel, validator

from app.config import settings
from app.core.logging_system import get_logger, LogCategory
from app.models.user import User

logger = get_logger("rbac_system", LogCategory.SECURITY)


class ResourceType(str, Enum):
    """Types de ressources du système"""
    USER = "user"
    COMPANY = "company"
    SCRAPING_JOB = "scraping_job"
    REPORT = "report"
    DASHBOARD = "dashboard"
    ADMIN_PANEL = "admin_panel"
    API_ENDPOINT = "api_endpoint"
    DATA_EXPORT = "data_export"
    SYSTEM_CONFIG = "system_config"
    AUDIT_LOG = "audit_log"


class Action(str, Enum):
    """Actions possibles sur les ressources"""
    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    EXECUTE = "execute"
    EXPORT = "export"
    IMPORT = "import"
    APPROVE = "approve"
    REJECT = "reject"
    ASSIGN = "assign"
    UNASSIGN = "unassign"
    SHARE = "share"
    REVOKE = "revoke"


class PermissionScope(str, Enum):
    """Portées des permissions"""
    GLOBAL = "global"           # Toutes les ressources du type
    ORGANIZATION = "organization" # Ressources de l'organisation
    TEAM = "team"              # Ressources de l'équipe
    PERSONAL = "personal"      # Ressources personnelles
    SPECIFIC = "specific"      # Ressources spécifiques (avec IDs)


@dataclass
class Permission:
    """Permission granulaire"""
    resource_type: ResourceType
    action: Action
    scope: PermissionScope = PermissionScope.PERSONAL
    conditions: Dict[str, Any] = field(default_factory=dict)
    resource_ids: Set[str] = field(default_factory=set)
    granted_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    
    def __str__(self) -> str:
        return f"{self.resource_type.value}:{self.action.value}:{self.scope.value}"
    
    def is_expired(self) -> bool:
        """Vérifie si la permission est expirée"""
        return self.expires_at is not None and datetime.now() > self.expires_at
    
    def matches_resource(self, resource_id: str = None, context: Dict[str, Any] = None) -> bool:
        """Vérifie si la permission s'applique à une ressource"""
        if self.is_expired():
            return False
        
        # Scope global
        if self.scope == PermissionScope.GLOBAL:
            return True
        
        # Scope spécifique avec IDs
        if self.scope == PermissionScope.SPECIFIC:
            return resource_id in self.resource_ids if resource_id else len(self.resource_ids) == 0
        
        # Autres scopes nécessitent un contexte
        if not context:
            return False
        
        # Scope organisation
        if self.scope == PermissionScope.ORGANIZATION:
            user_org = context.get('user_organization')
            resource_org = context.get('resource_organization')
            return user_org and resource_org and user_org == resource_org
        
        # Scope équipe
        if self.scope == PermissionScope.TEAM:
            user_team = context.get('user_team')
            resource_team = context.get('resource_team')
            return user_team and resource_team and user_team == resource_team
        
        # Scope personnel
        if self.scope == PermissionScope.PERSONAL:
            user_id = context.get('user_id')
            resource_owner = context.get('resource_owner')
            return user_id and resource_owner and user_id == resource_owner
        
        return False


@dataclass
class Role:
    """Rôle avec permissions associées"""
    name: str
    display_name: str
    description: str
    permissions: Set[Permission] = field(default_factory=set)
    parent_roles: Set[str] = field(default_factory=set)  # Héritage
    is_system_role: bool = False
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def add_permission(self, permission: Permission):
        """Ajoute une permission au rôle"""
        self.permissions.add(permission)
        self.updated_at = datetime.now()
    
    def remove_permission(self, permission: Permission):
        """Supprime une permission du rôle"""
        self.permissions.discard(permission)
        self.updated_at = datetime.now()
    
    def has_permission(self, resource_type: ResourceType, action: Action, 
                      resource_id: str = None, context: Dict[str, Any] = None) -> bool:
        """Vérifie si le rôle a une permission spécifique"""
        for permission in self.permissions:
            if (permission.resource_type == resource_type and 
                permission.action == action and
                permission.matches_resource(resource_id, context)):
                return True
        return False


@dataclass
class UserPermissions:
    """Permissions effectives d'un utilisateur"""
    user_id: str
    roles: Set[str] = field(default_factory=set)
    direct_permissions: Set[Permission] = field(default_factory=set)
    denied_permissions: Set[Permission] = field(default_factory=set)
    context: Dict[str, Any] = field(default_factory=dict)
    last_updated: datetime = field(default_factory=datetime.now)


class PolicyCondition:
    """Condition dynamique pour les politiques"""
    
    def __init__(self, condition_func: Callable[[Dict[str, Any]], bool], 
                 description: str = ""):
        self.condition_func = condition_func
        self.description = description
    
    def evaluate(self, context: Dict[str, Any]) -> bool:
        """Évalue la condition"""
        try:
            return self.condition_func(context)
        except Exception as e:
            logger.error(f"Erreur évaluation condition de politique: {e}")
            return False


@dataclass
class Policy:
    """Politique de sécurité dynamique"""
    name: str
    description: str
    resource_type: ResourceType
    action: Action
    effect: str = "allow"  # "allow" ou "deny"
    conditions: List[PolicyCondition] = field(default_factory=list)
    priority: int = 0  # Plus élevé = plus prioritaire
    
    def evaluate(self, context: Dict[str, Any]) -> Optional[bool]:
        """Évalue la politique"""
        # Toutes les conditions doivent être vraies
        for condition in self.conditions:
            if not condition.evaluate(context):
                return None  # Politique ne s'applique pas
        
        return self.effect == "allow"


class RBACSystem:
    """Système RBAC principal"""
    
    def __init__(self):
        self.roles: Dict[str, Role] = {}
        self.user_permissions: Dict[str, UserPermissions] = {}
        self.policies: List[Policy] = []
        
        # Cache des permissions calculées
        self._permission_cache: Dict[str, Dict[str, bool]] = {}
        self._cache_ttl = timedelta(minutes=15)
        self._last_cache_cleanup = datetime.now()
        
        # Initialiser rôles par défaut
        self._setup_default_roles()
        self._setup_default_policies()
        
        logger.info("🔐 Système RBAC initialisé")
    
    def _setup_default_roles(self):
        """Configure les rôles par défaut du système"""
        
        # Super Administrateur
        super_admin = Role(
            name="super_admin",
            display_name="Super Administrateur",
            description="Accès complet au système",
            is_system_role=True
        )
        
        # Permissions globales pour super admin
        for resource_type in ResourceType:
            for action in Action:
                super_admin.add_permission(Permission(
                    resource_type=resource_type,
                    action=action,
                    scope=PermissionScope.GLOBAL
                ))
        
        self.roles["super_admin"] = super_admin
        
        # Administrateur
        admin = Role(
            name="admin",
            display_name="Administrateur",
            description="Administration générale",
            is_system_role=True
        )
        
        # Permissions pour admin
        admin_permissions = [
            (ResourceType.USER, Action.READ, PermissionScope.ORGANIZATION),
            (ResourceType.USER, Action.UPDATE, PermissionScope.ORGANIZATION),
            (ResourceType.COMPANY, Action.READ, PermissionScope.GLOBAL),
            (ResourceType.COMPANY, Action.UPDATE, PermissionScope.GLOBAL),
            (ResourceType.COMPANY, Action.DELETE, PermissionScope.GLOBAL),
            (ResourceType.SCRAPING_JOB, Action.CREATE, PermissionScope.GLOBAL),
            (ResourceType.SCRAPING_JOB, Action.READ, PermissionScope.GLOBAL),
            (ResourceType.SCRAPING_JOB, Action.EXECUTE, PermissionScope.GLOBAL),
            (ResourceType.REPORT, Action.READ, PermissionScope.GLOBAL),
            (ResourceType.REPORT, Action.CREATE, PermissionScope.GLOBAL),
            (ResourceType.DASHBOARD, Action.READ, PermissionScope.GLOBAL),
            (ResourceType.DATA_EXPORT, Action.EXPORT, PermissionScope.GLOBAL),
            (ResourceType.AUDIT_LOG, Action.READ, PermissionScope.ORGANIZATION)
        ]
        
        for resource_type, action, scope in admin_permissions:
            admin.add_permission(Permission(
                resource_type=resource_type,
                action=action,
                scope=scope
            ))
        
        self.roles["admin"] = admin
        
        # Manager / Responsable
        manager = Role(
            name="manager",
            display_name="Responsable",
            description="Gestion équipe et données",
            is_system_role=True
        )
        
        manager_permissions = [
            (ResourceType.COMPANY, Action.READ, PermissionScope.TEAM),
            (ResourceType.COMPANY, Action.UPDATE, PermissionScope.TEAM),
            (ResourceType.SCRAPING_JOB, Action.CREATE, PermissionScope.TEAM),
            (ResourceType.SCRAPING_JOB, Action.READ, PermissionScope.TEAM),
            (ResourceType.SCRAPING_JOB, Action.EXECUTE, PermissionScope.TEAM),
            (ResourceType.REPORT, Action.READ, PermissionScope.TEAM),
            (ResourceType.REPORT, Action.CREATE, PermissionScope.TEAM),
            (ResourceType.DASHBOARD, Action.READ, PermissionScope.TEAM),
            (ResourceType.DATA_EXPORT, Action.EXPORT, PermissionScope.TEAM),
            (ResourceType.USER, Action.READ, PermissionScope.TEAM)
        ]
        
        for resource_type, action, scope in manager_permissions:
            manager.add_permission(Permission(
                resource_type=resource_type,
                action=action,
                scope=scope
            ))
        
        self.roles["manager"] = manager
        
        # Utilisateur standard
        user = Role(
            name="user",
            display_name="Utilisateur",
            description="Accès en lecture et actions limitées",
            is_system_role=True
        )
        
        user_permissions = [
            (ResourceType.COMPANY, Action.READ, PermissionScope.PERSONAL),
            (ResourceType.COMPANY, Action.CREATE, PermissionScope.PERSONAL),
            (ResourceType.COMPANY, Action.UPDATE, PermissionScope.PERSONAL),
            (ResourceType.SCRAPING_JOB, Action.READ, PermissionScope.PERSONAL),
            (ResourceType.REPORT, Action.READ, PermissionScope.PERSONAL),
            (ResourceType.DASHBOARD, Action.READ, PermissionScope.PERSONAL),
            (ResourceType.DATA_EXPORT, Action.EXPORT, PermissionScope.PERSONAL),
            (ResourceType.USER, Action.READ, PermissionScope.PERSONAL),
            (ResourceType.USER, Action.UPDATE, PermissionScope.PERSONAL)
        ]
        
        for resource_type, action, scope in user_permissions:
            user.add_permission(Permission(
                resource_type=resource_type,
                action=action,
                scope=scope
            ))
        
        self.roles["user"] = user
        
        # Invité / Read-only
        guest = Role(
            name="guest",
            display_name="Invité",
            description="Accès en lecture uniquement",
            is_system_role=True
        )
        
        guest_permissions = [
            (ResourceType.COMPANY, Action.READ, PermissionScope.PERSONAL),
            (ResourceType.DASHBOARD, Action.READ, PermissionScope.PERSONAL),
            (ResourceType.REPORT, Action.READ, PermissionScope.PERSONAL)
        ]
        
        for resource_type, action, scope in guest_permissions:
            guest.add_permission(Permission(
                resource_type=resource_type,
                action=action,
                scope=scope
            ))
        
        self.roles["guest"] = guest
        
        logger.info(f"✅ {len(self.roles)} rôles par défaut configurés")
    
    def _setup_default_policies(self):
        """Configure les politiques par défaut"""
        
        # Politique: Heures ouvrables pour actions sensibles
        business_hours_condition = PolicyCondition(
            condition_func=lambda ctx: 8 <= datetime.now().hour <= 18,
            description="Heures ouvrables (8h-18h)"
        )
        
        business_hours_policy = Policy(
            name="business_hours_sensitive_actions",
            description="Actions sensibles uniquement en heures ouvrables",
            resource_type=ResourceType.SYSTEM_CONFIG,
            action=Action.UPDATE,
            effect="deny",
            conditions=[business_hours_condition],
            priority=100
        )
        
        # Politique: Approbation pour suppression de données
        data_deletion_condition = PolicyCondition(
            condition_func=lambda ctx: ctx.get('has_approval', False),
            description="Approbation requise pour suppression"
        )
        
        data_deletion_policy = Policy(
            name="require_approval_for_deletion",
            description="Suppression de données nécessite approbation",
            resource_type=ResourceType.COMPANY,
            action=Action.DELETE,
            effect="deny",
            conditions=[data_deletion_condition],
            priority=90
        )
        
        # Politique: Export limité par jour
        daily_export_limit_condition = PolicyCondition(
            condition_func=lambda ctx: ctx.get('daily_exports', 0) < 10,
            description="Maximum 10 exports par jour"
        )
        
        export_limit_policy = Policy(
            name="daily_export_limit",
            description="Limite quotidienne d'exports",
            resource_type=ResourceType.DATA_EXPORT,
            action=Action.EXPORT,
            effect="deny",
            conditions=[daily_export_limit_condition],
            priority=80
        )
        
        self.policies.extend([
            business_hours_policy,
            data_deletion_policy,
            export_limit_policy
        ])
        
        logger.info(f"✅ {len(self.policies)} politiques par défaut configurées")
    
    def assign_role(self, user_id: str, role_name: str) -> bool:
        """Assigne un rôle à un utilisateur"""
        try:
            if role_name not in self.roles:
                logger.warning(f"Tentative d'assignation de rôle inexistant: {role_name}")
                return False
            
            if user_id not in self.user_permissions:
                self.user_permissions[user_id] = UserPermissions(user_id=user_id)
            
            self.user_permissions[user_id].roles.add(role_name)
            self.user_permissions[user_id].last_updated = datetime.now()
            
            # Invalider cache
            self._invalidate_user_cache(user_id)
            
            logger.info(f"✅ Rôle {role_name} assigné à l'utilisateur {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Erreur assignation rôle {role_name} à {user_id}: {e}")
            return False
    
    def revoke_role(self, user_id: str, role_name: str) -> bool:
        """Révoque un rôle d'un utilisateur"""
        try:
            if user_id not in self.user_permissions:
                return False
            
            self.user_permissions[user_id].roles.discard(role_name)
            self.user_permissions[user_id].last_updated = datetime.now()
            
            # Invalider cache
            self._invalidate_user_cache(user_id)
            
            logger.info(f"✅ Rôle {role_name} révoqué pour l'utilisateur {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Erreur révocation rôle {role_name} pour {user_id}: {e}")
            return False
    
    def grant_permission(self, user_id: str, permission: Permission) -> bool:
        """Accorde une permission directe à un utilisateur"""
        try:
            if user_id not in self.user_permissions:
                self.user_permissions[user_id] = UserPermissions(user_id=user_id)
            
            self.user_permissions[user_id].direct_permissions.add(permission)
            self.user_permissions[user_id].last_updated = datetime.now()
            
            # Invalider cache
            self._invalidate_user_cache(user_id)
            
            logger.info(f"✅ Permission {permission} accordée à l'utilisateur {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Erreur accord permission à {user_id}: {e}")
            return False
    
    def deny_permission(self, user_id: str, permission: Permission) -> bool:
        """Refuse explicitement une permission à un utilisateur"""
        try:
            if user_id not in self.user_permissions:
                self.user_permissions[user_id] = UserPermissions(user_id=user_id)
            
            self.user_permissions[user_id].denied_permissions.add(permission)
            self.user_permissions[user_id].last_updated = datetime.now()
            
            # Invalider cache
            self._invalidate_user_cache(user_id)
            
            logger.info(f"✅ Permission {permission} refusée pour l'utilisateur {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Erreur refus permission pour {user_id}: {e}")
            return False
    
    def check_permission(self, user_id: str, resource_type: ResourceType, action: Action,
                        resource_id: str = None, context: Dict[str, Any] = None) -> bool:
        """Vérifie si un utilisateur a une permission"""
        
        # Préparer contexte
        if context is None:
            context = {}
        
        context['user_id'] = user_id
        
        # Vérifier cache
        cache_key = self._get_cache_key(user_id, resource_type, action, resource_id, context)
        
        if cache_key in self._permission_cache:
            cached_result = self._permission_cache[cache_key]
            if 'timestamp' in cached_result:
                cache_age = datetime.now() - cached_result['timestamp']
                if cache_age < self._cache_ttl:
                    return cached_result['result']
        
        # Calculer permission
        result = self._calculate_permission(user_id, resource_type, action, resource_id, context)
        
        # Mettre en cache
        self._permission_cache[cache_key] = {
            'result': result,
            'timestamp': datetime.now()
        }
        
        # Nettoyage périodique du cache
        self._cleanup_cache_if_needed()
        
        return result
    
    def _calculate_permission(self, user_id: str, resource_type: ResourceType, action: Action,
                            resource_id: str = None, context: Dict[str, Any] = None) -> bool:
        """Calcule si un utilisateur a une permission"""
        
        try:
            # Utilisateur inexistant = pas de permission
            if user_id not in self.user_permissions:
                return False
            
            user_perms = self.user_permissions[user_id]
            
            # 1. Vérifier les permissions explicitement refusées
            for denied_perm in user_perms.denied_permissions:
                if (denied_perm.resource_type == resource_type and 
                    denied_perm.action == action and
                    denied_perm.matches_resource(resource_id, context)):
                    logger.debug(f"Permission refusée explicitement pour {user_id}")
                    return False
            
            # 2. Évaluer les politiques (peuvent overrider)
            policy_result = self._evaluate_policies(resource_type, action, context)
            if policy_result is not None:
                if not policy_result:
                    logger.debug(f"Permission refusée par politique pour {user_id}")
                    return False
            
            # 3. Vérifier permissions directes
            for direct_perm in user_perms.direct_permissions:
                if (direct_perm.resource_type == resource_type and 
                    direct_perm.action == action and
                    direct_perm.matches_resource(resource_id, context)):
                    logger.debug(f"Permission accordée directement pour {user_id}")
                    return True
            
            # 4. Vérifier permissions via rôles (avec héritage)
            all_roles = self._get_effective_roles(user_perms.roles)
            
            for role_name in all_roles:
                if role_name in self.roles:
                    role = self.roles[role_name]
                    if role.has_permission(resource_type, action, resource_id, context):
                        logger.debug(f"Permission accordée via rôle {role_name} pour {user_id}")
                        return True
            
            # 5. Permission non trouvée
            logger.debug(f"Permission non trouvée pour {user_id}")
            return False
            
        except Exception as e:
            logger.error(f"Erreur calcul permission pour {user_id}: {e}")
            return False
    
    def _get_effective_roles(self, user_roles: Set[str]) -> Set[str]:
        """Récupère tous les rôles effectifs (avec héritage)"""
        effective_roles = set(user_roles)
        
        # Résoudre l'héritage
        to_process = list(user_roles)
        
        while to_process:
            current_role_name = to_process.pop()
            
            if current_role_name in self.roles:
                current_role = self.roles[current_role_name]
                
                for parent_role_name in current_role.parent_roles:
                    if parent_role_name not in effective_roles:
                        effective_roles.add(parent_role_name)
                        to_process.append(parent_role_name)
        
        return effective_roles
    
    def _evaluate_policies(self, resource_type: ResourceType, action: Action, 
                          context: Dict[str, Any]) -> Optional[bool]:
        """Évalue les politiques applicables"""
        
        applicable_policies = [
            policy for policy in self.policies
            if policy.resource_type == resource_type and policy.action == action
        ]
        
        # Trier par priorité (plus élevé en premier)
        applicable_policies.sort(key=lambda p: p.priority, reverse=True)
        
        for policy in applicable_policies:
            result = policy.evaluate(context)
            if result is not None:
                logger.debug(f"Politique {policy.name} évaluée: {result}")
                return result
        
        return None  # Aucune politique applicable
    
    def _get_cache_key(self, user_id: str, resource_type: ResourceType, action: Action,
                      resource_id: str = None, context: Dict[str, Any] = None) -> str:
        """Génère une clé de cache pour une permission"""
        
        key_data = {
            'user_id': user_id,
            'resource_type': resource_type.value,
            'action': action.value,
            'resource_id': resource_id,
            'context_hash': hashlib.md5(
                json.dumps(context or {}, sort_keys=True).encode()
            ).hexdigest()[:8]
        }
        
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _invalidate_user_cache(self, user_id: str):
        """Invalide le cache pour un utilisateur"""
        keys_to_remove = [
            key for key in self._permission_cache.keys()
            if user_id in key
        ]
        
        for key in keys_to_remove:
            del self._permission_cache[key]
    
    def _cleanup_cache_if_needed(self):
        """Nettoie le cache si nécessaire"""
        now = datetime.now()
        
        if now - self._last_cache_cleanup > timedelta(hours=1):
            expired_keys = []
            
            for key, cached_data in self._permission_cache.items():
                if 'timestamp' in cached_data:
                    cache_age = now - cached_data['timestamp']
                    if cache_age > self._cache_ttl:
                        expired_keys.append(key)
            
            for key in expired_keys:
                del self._permission_cache[key]
            
            self._last_cache_cleanup = now
            
            if expired_keys:
                logger.debug(f"Cache nettoyé: {len(expired_keys)} entrées expirées supprimées")
    
    def get_user_roles(self, user_id: str) -> List[str]:
        """Retourne les rôles d'un utilisateur"""
        if user_id not in self.user_permissions:
            return []
        
        return list(self.user_permissions[user_id].roles)
    
    def get_role_permissions(self, role_name: str) -> List[Permission]:
        """Retourne les permissions d'un rôle"""
        if role_name not in self.roles:
            return []
        
        return list(self.roles[role_name].permissions)
    
    def get_user_effective_permissions(self, user_id: str) -> List[Permission]:
        """Retourne toutes les permissions effectives d'un utilisateur"""
        if user_id not in self.user_permissions:
            return []
        
        user_perms = self.user_permissions[user_id]
        all_permissions = set(user_perms.direct_permissions)
        
        # Ajouter permissions des rôles
        effective_roles = self._get_effective_roles(user_perms.roles)
        
        for role_name in effective_roles:
            if role_name in self.roles:
                all_permissions.update(self.roles[role_name].permissions)
        
        # Retirer permissions refusées
        all_permissions -= user_perms.denied_permissions
        
        return list(all_permissions)
    
    def create_custom_role(self, name: str, display_name: str, description: str,
                          permissions: List[Permission] = None) -> bool:
        """Crée un rôle personnalisé"""
        try:
            if name in self.roles:
                logger.warning(f"Tentative de création d'un rôle existant: {name}")
                return False
            
            role = Role(
                name=name,
                display_name=display_name,
                description=description,
                is_system_role=False
            )
            
            if permissions:
                for permission in permissions:
                    role.add_permission(permission)
            
            self.roles[name] = role
            
            logger.info(f"✅ Rôle personnalisé créé: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Erreur création rôle {name}: {e}")
            return False
    
    def delete_custom_role(self, name: str) -> bool:
        """Supprime un rôle personnalisé"""
        try:
            if name not in self.roles:
                return False
            
            role = self.roles[name]
            
            if role.is_system_role:
                logger.warning(f"Tentative de suppression d'un rôle système: {name}")
                return False
            
            # Retirer le rôle de tous les utilisateurs
            for user_perms in self.user_permissions.values():
                user_perms.roles.discard(name)
            
            del self.roles[name]
            
            # Invalider tout le cache
            self._permission_cache.clear()
            
            logger.info(f"✅ Rôle personnalisé supprimé: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Erreur suppression rôle {name}: {e}")
            return False
    
    def get_access_summary(self, user_id: str) -> Dict[str, Any]:
        """Retourne un résumé des accès d'un utilisateur"""
        
        if user_id not in self.user_permissions:
            return {
                'user_id': user_id,
                'roles': [],
                'direct_permissions': 0,
                'denied_permissions': 0,
                'effective_permissions': 0,
                'last_updated': None
            }
        
        user_perms = self.user_permissions[user_id]
        effective_permissions = self.get_user_effective_permissions(user_id)
        
        return {
            'user_id': user_id,
            'roles': list(user_perms.roles),
            'direct_permissions': len(user_perms.direct_permissions),
            'denied_permissions': len(user_perms.denied_permissions),
            'effective_permissions': len(effective_permissions),
            'last_updated': user_perms.last_updated.isoformat()
        }


# Instance globale
_rbac_system: Optional[RBACSystem] = None


def get_rbac_system() -> RBACSystem:
    """Factory pour obtenir le système RBAC"""
    global _rbac_system
    
    if _rbac_system is None:
        _rbac_system = RBACSystem()
    
    return _rbac_system


# Décorateurs et utilitaires FastAPI

def require_permission(resource_type: ResourceType, action: Action, 
                      resource_id_param: str = None):
    """Décorateur pour requérir une permission spécifique"""
    
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extraire l'utilisateur actuel et la requête du contexte FastAPI
            current_user = None
            request = None
            resource_id = None
            
            # Chercher dans les arguments
            for arg in args:
                if isinstance(arg, User):
                    current_user = arg
                elif isinstance(arg, Request):
                    request = arg
            
            # Chercher dans les kwargs
            for key, value in kwargs.items():
                if key == 'current_user' and isinstance(value, User):
                    current_user = value
                elif key == 'request' and isinstance(value, Request):
                    request = value
                elif key == resource_id_param and resource_id_param:
                    resource_id = str(value)
            
            if not current_user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentification requise"
                )
            
            # Construire contexte
            context = {
                'user_id': current_user.id,
                'user_organization': getattr(current_user, 'organization_id', None),
                'user_team': getattr(current_user, 'team_id', None),
            }
            
            if request:
                context.update({
                    'ip_address': request.client.host if request.client else None,
                    'user_agent': request.headers.get('user-agent', ''),
                    'endpoint': request.url.path,
                    'method': request.method
                })
            
            # Vérifier permission
            rbac = get_rbac_system()
            
            has_permission = rbac.check_permission(
                user_id=current_user.id,
                resource_type=resource_type,
                action=action,
                resource_id=resource_id,
                context=context
            )
            
            if not has_permission:
                logger.warning(
                    f"Accès refusé pour {current_user.username}: "
                    f"{resource_type.value}:{action.value}"
                )
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Permission insuffisante"
                )
            
            # Ajouter contexte RBAC aux kwargs pour la fonction
            kwargs['rbac_context'] = context
            
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator


def require_role(role_name: str):
    """Décorateur pour requérir un rôle spécifique"""
    
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            current_user = None
            
            # Extraire l'utilisateur actuel
            for arg in args:
                if isinstance(arg, User):
                    current_user = arg
                    break
            
            for key, value in kwargs.items():
                if key == 'current_user' and isinstance(value, User):
                    current_user = value
                    break
            
            if not current_user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentification requise"
                )
            
            # Vérifier rôle
            rbac = get_rbac_system()
            user_roles = rbac.get_user_roles(current_user.id)
            
            if role_name not in user_roles:
                logger.warning(
                    f"Accès refusé pour {current_user.username}: "
                    f"rôle {role_name} requis"
                )
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Rôle {role_name} requis"
                )
            
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator


# Dépendances FastAPI

async def get_current_user_roles(current_user: User = Depends()) -> List[str]:
    """Dépendance pour obtenir les rôles de l'utilisateur actuel"""
    rbac = get_rbac_system()
    return rbac.get_user_roles(current_user.id)


async def get_current_user_permissions(current_user: User = Depends()) -> List[Permission]:
    """Dépendance pour obtenir les permissions de l'utilisateur actuel"""
    rbac = get_rbac_system()
    return rbac.get_user_effective_permissions(current_user.id)


# Utilitaires de setup

async def setup_user_rbac(user_id: str, roles: List[str] = None):
    """Configure RBAC pour un nouvel utilisateur"""
    rbac = get_rbac_system()
    
    # Rôle par défaut
    if not roles:
        roles = ["user"]
    
    for role_name in roles:
        rbac.assign_role(user_id, role_name)
    
    logger.info(f"✅ RBAC configuré pour utilisateur {user_id}: {roles}")


async def migrate_existing_users():
    """Migre les utilisateurs existants vers RBAC"""
    # Cette fonction devrait être appelée lors de la migration initiale
    # pour assigner des rôles aux utilisateurs existants
    pass