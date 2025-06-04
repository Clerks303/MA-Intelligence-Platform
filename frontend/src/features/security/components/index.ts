/**
 * Security Components Exports - M&A Intelligence Platform
 * Sprint 4 - Index des composants sécurité
 */

// Main Components
export { default as SecurityDashboard } from './SecurityDashboard';
export { default as UserManagement } from './UserManagement';
export { default as RoleManagement } from './RoleManagement';
export { default as MFAInterface } from './MFAInterface';
export { default as AuditVisualization } from './AuditVisualization';

// Dialog Components
export { default as UserFormDialog } from './dialogs/UserFormDialog';
export { default as UserDetailsDialog } from './dialogs/UserDetailsDialog';
export { default as RoleFormDialog } from './dialogs/RoleFormDialog';
export { default as PermissionDetailsDialog } from './dialogs/PermissionDetailsDialog';
export { default as ConfirmDialog } from './dialogs/ConfirmDialog';

// Re-export types for convenience
export type {
  User,
  Role,
  Permission,
  CreateUserRequest,
  UpdateUserRequest,
  CreateRoleRequest,
  UpdateRoleRequest,
  SecurityLevel,
  ResourceType,
  ActionType,
  PermissionScope
} from '../types';