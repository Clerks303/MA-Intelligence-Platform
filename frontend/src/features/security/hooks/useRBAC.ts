/**
 * Hooks RBAC - M&A Intelligence Platform
 * Sprint 4 - Hooks pour rôles, permissions et gestion utilisateurs
 */

import { useState, useCallback, useMemo } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { 
  User, 
  PaginatedResponse,
  CreateUserRequest,
  UpdateUserRequest,
  Role,
  Permission,
  CreateRoleRequest,
  UpdateRoleRequest,
  SecurityFilters,
  ResourceType,
  ActionType,
  PermissionScope
} from '../types/api';
import { securityService } from '../services/securityService';

// === GESTION DES UTILISATEURS ===

type UserFilters = SecurityFilters['users'];

export const useUsers = (
  initialFilters?: UserFilters,
  initialPage = 1,
  initialPerPage = 20
) => {
  const [filters, setFilters] = useState<UserFilters>(initialFilters || {});
  const [page, setPage] = useState(initialPage);
  const [perPage, setPerPage] = useState(initialPerPage);

  const queryClient = useQueryClient();

  // Liste des utilisateurs
  const { 
    data: usersResponse, 
    isLoading,
    error,
    refetch 
  } = useQuery({
    queryKey: ['security', 'users', filters, page, perPage],
    queryFn: () => securityService.users.getUsers(filters, page, perPage),
    placeholderData: (previousData) => previousData,
  });

  // Créer utilisateur
  const createUserMutation = useMutation({
    mutationFn: (userData: any) => securityService.users.createUser(userData),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['security', 'users'] });
    },
  });

  // Modifier utilisateur
  const updateUserMutation = useMutation({
    mutationFn: ({ userId, userData }: { userId: string; userData: UpdateUserRequest }) =>
      securityService.users.updateUser(userId, userData),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['security', 'users'] });
    },
  });

  // Supprimer utilisateur
  const deleteUserMutation = useMutation({
    mutationFn: (userId: string) => securityService.users.deleteUser(userId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['security', 'users'] });
    },
  });

  // Toggle statut utilisateur
  const toggleStatusMutation = useMutation({
    mutationFn: (userId: string) => securityService.users.toggleUserStatus(userId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['security', 'users'] });
    },
  });

  // Reset mot de passe
  const resetPasswordMutation = useMutation({
    mutationFn: (userId: string) => securityService.users.resetPassword(userId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['security', 'users'] });
    },
  });

  // Révoquer toutes les sessions
  const revokeAllSessionsMutation = useMutation({
    mutationFn: (userId: string) => securityService.users.revokeAllSessions(userId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['security', 'sessions'] });
    },
  });

  // Actions
  const createUser = useCallback((userData: CreateUserRequest) => {
    createUserMutation.mutate(userData);
  }, [createUserMutation]);

  const updateUser = useCallback((userId: string, userData: UpdateUserRequest) => {
    updateUserMutation.mutate({ userId, userData });
  }, [updateUserMutation]);

  const deleteUser = useCallback((userId: string) => {
    deleteUserMutation.mutate(userId);
  }, [deleteUserMutation]);

  const toggleUserStatus = useCallback((userId: string) => {
    toggleStatusMutation.mutate(userId);
  }, [toggleStatusMutation]);

  const resetUserPassword = useCallback((userId: string) => {
    resetPasswordMutation.mutate(userId);
  }, [resetPasswordMutation]);

  const revokeUserSessions = useCallback((userId: string) => {
    revokeAllSessionsMutation.mutate(userId);
  }, [revokeAllSessionsMutation]);

  // Filtres et pagination
  const updateFilters = useCallback((newFilters: Partial<SecurityFilters['users']>) => {
    setFilters(prev => ({ ...prev, ...newFilters }));
    setPage(1); // Reset à la première page
  }, []);

  const clearFilters = useCallback(() => {
    setFilters({});
    setPage(1);
  }, []);

  const goToPage = useCallback((newPage: number) => {
    setPage(newPage);
  }, []);

  // Métriques des utilisateurs
  const userMetrics = useMemo(() => {
    const users = (usersResponse as PaginatedResponse<User>)?.items || [];
    return {
      total: (usersResponse as PaginatedResponse<User>)?.total || 0,
      active: users.filter((u: User) => u.is_active).length,
      inactive: users.filter((u: User) => !u.is_active).length,
      admins: users.filter((u: User) => u.is_superuser).length,
      withMFA: users.filter((u: User) => u.mfa_enabled).length,
      recentLogins: users.filter((u: User) => 
        u.last_login && 
        new Date(u.last_login) > new Date(Date.now() - 24 * 60 * 60 * 1000)
      ).length,
    };
  }, [usersResponse]);

  return {
    // Données
    users: (usersResponse as PaginatedResponse<User>)?.items || [],
    pagination: usersResponse ? {
      page: (usersResponse as PaginatedResponse<User>).page,
      pages: (usersResponse as PaginatedResponse<User>).pages,
      perPage: (usersResponse as PaginatedResponse<User>).per_page,
      total: (usersResponse as PaginatedResponse<User>).total,
      hasNext: (usersResponse as PaginatedResponse<User>).has_next,
      hasPrev: (usersResponse as PaginatedResponse<User>).has_prev,
    } : null,
    userMetrics,
    
    // Filtres et pagination
    filters,
    updateFilters,
    clearFilters,
    page,
    perPage,
    goToPage,
    setPerPage,
    
    // Chargement
    isLoading,
    isCreating: createUserMutation.isPending,
    isUpdating: updateUserMutation.isPending,
    isDeleting: deleteUserMutation.isPending,
    isTogglingStatus: toggleStatusMutation.isPending,
    isResettingPassword: resetPasswordMutation.isPending,
    isRevokingSessions: revokeAllSessionsMutation.isPending,
    
    // Erreurs
    error,
    createError: createUserMutation.error,
    updateError: updateUserMutation.error,
    deleteError: deleteUserMutation.error,
    toggleStatusError: toggleStatusMutation.error,
    resetPasswordError: resetPasswordMutation.error,
    revokeSessionsError: revokeAllSessionsMutation.error,
    
    // Actions
    createUser,
    updateUser,
    deleteUser,
    toggleUserStatus,
    resetUserPassword,
    revokeUserSessions,
    refetch,
    
    // Données mutations
    createData: createUserMutation.data,
    updateData: updateUserMutation.data,
    resetPasswordData: resetPasswordMutation.data,
    revokeSessionsData: revokeAllSessionsMutation.data,
  };
};

// === GESTION DES RÔLES ===

export const useRoles = (initialFilters?: SecurityFilters['roles']) => {
  const [filters, setFilters] = useState(initialFilters || {});
  const queryClient = useQueryClient();

  // Liste des rôles
  const { 
    data: roles = [], 
    isLoading,
    error,
    refetch 
  } = useQuery({
    queryKey: ['security', 'roles', filters],
    queryFn: () => securityService.rbac.getRoles(filters),
  });

  // Créer rôle
  const createRoleMutation = useMutation({
    mutationFn: (roleData: CreateRoleRequest) => securityService.rbac.createRole(roleData),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['security', 'roles'] });
      queryClient.invalidateQueries({ queryKey: ['security', 'permissions'] });
    },
  });

  // Modifier rôle
  const updateRoleMutation = useMutation({
    mutationFn: ({ roleId, roleData }: { roleId: string; roleData: UpdateRoleRequest }) =>
      securityService.rbac.updateRole(roleId, roleData),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['security', 'roles'] });
      queryClient.invalidateQueries({ queryKey: ['security', 'permissions'] });
    },
  });

  // Supprimer rôle
  const deleteRoleMutation = useMutation({
    mutationFn: (roleId: string) => securityService.rbac.deleteRole(roleId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['security', 'roles'] });
      queryClient.invalidateQueries({ queryKey: ['security', 'users'] });
    },
  });

  // Assigner rôle à utilisateur
  const assignRoleMutation = useMutation({
    mutationFn: ({ userId, roleId }: { userId: string; roleId: string }) =>
      securityService.rbac.assignRole(userId, roleId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['security', 'users'] });
    },
  });

  // Retirer rôle d'utilisateur
  const removeRoleMutation = useMutation({
    mutationFn: ({ userId, roleId }: { userId: string; roleId: string }) =>
      securityService.rbac.removeRole(userId, roleId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['security', 'users'] });
    },
  });

  // Actions
  const createRole = useCallback((roleData: CreateRoleRequest) => {
    createRoleMutation.mutate(roleData);
  }, [createRoleMutation]);

  const updateRole = useCallback((roleId: string, roleData: UpdateRoleRequest) => {
    updateRoleMutation.mutate({ roleId, roleData });
  }, [updateRoleMutation]);

  const deleteRole = useCallback((roleId: string) => {
    deleteRoleMutation.mutate(roleId);
  }, [deleteRoleMutation]);

  const assignRole = useCallback((userId: string, roleId: string) => {
    assignRoleMutation.mutate({ userId, roleId });
  }, [assignRoleMutation]);

  const removeRole = useCallback((userId: string, roleId: string) => {
    removeRoleMutation.mutate({ userId, roleId });
  }, [removeRoleMutation]);

  const updateFilters = useCallback((newFilters: Partial<SecurityFilters['roles']>) => {
    setFilters(prev => ({ ...prev, ...newFilters }));
  }, []);

  const clearFilters = useCallback(() => {
    setFilters({});
  }, []);

  // Métriques des rôles
  const roleMetrics = useMemo(() => ({
    total: roles.length,
    system: roles.filter((r: Role) => r.is_system_role).length,
    custom: roles.filter((r: Role) => !r.is_system_role).length,
    withUsers: roles.filter((r: Role) => (r.users_count || 0) > 0).length,
    hierarchy: roles.filter((r: Role) => r.parent_role || (r.child_roles && r.child_roles.length > 0)).length,
  }), [roles]);

  return {
    // Données
    roles,
    roleMetrics,
    
    // Filtres
    filters,
    updateFilters,
    clearFilters,
    
    // Chargement
    isLoading,
    isCreating: createRoleMutation.isPending,
    isUpdating: updateRoleMutation.isPending,
    isDeleting: deleteRoleMutation.isPending,
    isAssigning: assignRoleMutation.isPending,
    isRemoving: removeRoleMutation.isPending,
    
    // Erreurs
    error,
    createError: createRoleMutation.error,
    updateError: updateRoleMutation.error,
    deleteError: deleteRoleMutation.error,
    assignError: assignRoleMutation.error,
    removeError: removeRoleMutation.error,
    
    // Actions
    createRole,
    updateRole,
    deleteRole,
    assignRole,
    removeRole,
    refetch,
    
    // Données mutations
    createData: createRoleMutation.data,
    updateData: updateRoleMutation.data,
    assignData: assignRoleMutation.data,
    removeData: removeRoleMutation.data,
  };
};

// === GESTION DES PERMISSIONS ===

export const usePermissions = () => {
  const queryClient = useQueryClient();

  // Liste des permissions
  const { 
    data: permissions = [], 
    isLoading,
    error,
    refetch 
  } = useQuery({
    queryKey: ['security', 'permissions'],
    queryFn: securityService.rbac.getPermissions,
  });

  // Vérifier permission
  const checkPermissionMutation = useMutation({
    mutationFn: ({ resource, action, scope }: { 
      resource: string; 
      action: string; 
      scope?: string 
    }) => securityService.rbac.checkPermission(resource, action, scope),
  });

  // Actions
  const checkPermission = useCallback((
    resource: ResourceType, 
    action: ActionType, 
    scope?: PermissionScope
  ) => {
    checkPermissionMutation.mutate({ resource, action, scope });
  }, [checkPermissionMutation]);

  // Grouper les permissions par ressource
  const permissionsByResource = useMemo(() => {
    return permissions.reduce((acc: Record<ResourceType, Permission[]>, permission: Permission) => {
      if (!acc[permission.resource]) {
        acc[permission.resource] = [];
      }
      acc[permission.resource].push(permission);
      return acc;
    }, {} as Record<ResourceType, Permission[]>);
  }, [permissions]);

  // Grouper les permissions par action
  const permissionsByAction = useMemo(() => {
    return permissions.reduce((acc: Record<ActionType, Permission[]>, permission: Permission) => {
      if (!acc[permission.action]) {
        acc[permission.action] = [];
      }
      acc[permission.action].push(permission);
      return acc;
    }, {} as Record<ActionType, Permission[]>);
  }, [permissions]);

  // Métriques des permissions
  const permissionMetrics = useMemo(() => ({
    total: permissions.length,
    byResource: Object.keys(permissionsByResource).length,
    byAction: Object.keys(permissionsByAction).length,
    byScope: permissions.reduce((acc: Record<PermissionScope, number>, p: Permission) => {
      acc[p.scope] = (acc[p.scope] || 0) + 1;
      return acc;
    }, {} as Record<PermissionScope, number>),
  }), [permissions, permissionsByResource, permissionsByAction]);

  return {
    // Données
    permissions,
    permissionsByResource,
    permissionsByAction,
    permissionMetrics,
    
    // Chargement
    isLoading,
    isCheckingPermission: checkPermissionMutation.isPending,
    
    // Erreurs
    error,
    checkPermissionError: checkPermissionMutation.error,
    
    // Actions
    checkPermission,
    refetch,
    
    // Données mutations
    checkPermissionData: checkPermissionMutation.data,
  };
};

// === HOOK POUR PERMISSIONS UTILISATEUR SPÉCIFIQUE ===

export const useUserPermissions = (userId?: string) => {
  // Permissions de l'utilisateur
  const { 
    data: userPermissions = [], 
    isLoading,
    error 
  } = useQuery({
    queryKey: ['security', 'users', userId, 'permissions'],
    queryFn: () => securityService.rbac.getUserPermissions(userId!),
    enabled: !!userId,
  });

  // Vérifier si l'utilisateur a une permission
  const hasPermission = useCallback((
    resource: ResourceType, 
    action: ActionType, 
    scope?: PermissionScope
  ) => {
    return userPermissions.some((permission: Permission) => 
      permission.resource === resource &&
      permission.action === action &&
      (!scope || permission.scope === scope)
    );
  }, [userPermissions]);

  // Vérifier si l'utilisateur peut effectuer une action
  const canPerform = useCallback((
    resource: ResourceType, 
    actions: ActionType[]
  ) => {
    return actions.some((action: ActionType) => hasPermission(resource, action));
  }, [hasPermission]);

  // Obtenir les ressources accessibles
  const accessibleResources = useMemo(() => {
    const resources = new Set<ResourceType>();
    userPermissions.forEach((permission: Permission) => {
      resources.add(permission.resource);
    });
    return Array.from(resources);
  }, [userPermissions]);

  // Grouper les permissions par ressource
  const permissionsByResource = useMemo(() => {
    return userPermissions.reduce((acc: Record<ResourceType, Permission[]>, permission: Permission) => {
      if (!acc[permission.resource]) {
        acc[permission.resource] = [];
      }
      acc[permission.resource].push(permission);
      return acc;
    }, {} as Record<ResourceType, Permission[]>);
  }, [userPermissions]);

  return {
    // Données
    permissions: userPermissions,
    permissionsByResource,
    accessibleResources,
    
    // État
    isLoading,
    error,
    
    // Utilitaires
    hasPermission,
    canPerform,
  };
};

// === HOOK COMBINÉ RBAC ===

export const useRBAC = () => {
  const users = useUsers();
  const roles = useRoles();
  const permissions = usePermissions();

  // Statistiques globales RBAC
  const rbacStats = useMemo(() => ({
    users: users.userMetrics,
    roles: roles.roleMetrics,
    permissions: permissions.permissionMetrics,
    
    // Relations
    averageRolesPerUser: users.userMetrics.total > 0 ? 
      roles.roles.reduce((sum: number, role: Role) => sum + (role.users_count || 0), 0) / users.userMetrics.total : 0,
    averagePermissionsPerRole: roles.roles.length > 0 ?
      roles.roles.reduce((sum: number, role: Role) => sum + role.permissions.length, 0) / roles.roles.length : 0,
  }), [users.userMetrics, roles.roleMetrics, permissions.permissionMetrics, users.users, roles.roles]);

  return {
    // Modules
    users,
    roles,
    permissions,
    
    // Statistiques
    rbacStats,
    
    // États globaux
    isLoading: users.isLoading || roles.isLoading || permissions.isLoading,
    hasErrors: !!(users.error || roles.error || permissions.error),
  };
};

export default {
  useUsers,
  useRoles,
  usePermissions,
  useUserPermissions,
  useRBAC,
};