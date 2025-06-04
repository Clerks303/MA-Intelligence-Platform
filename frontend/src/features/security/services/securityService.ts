// Service de sécurité complet pour éviter les erreurs
import { 
  User, 
  PaginatedResponse, 
  AuditEvent, 
  AuditFilters,
  ChangePasswordRequest,
  CreateUserRequest,
  UpdateUserRequest,
  MFAVerifyRequest,
  MFAStatus,
  MFASetupResponse,
  UserSession,
  Role,
  Permission,
  CreateRoleRequest,
  UpdateRoleRequest,
  LoginResponse,
  SecurityDashboard,
  SecurityMetrics,
  SecurityAlert,
  SecurityConfig,
  SystemMetrics,
  HealthStatus,
  SecurityFilters,
  SecurityLevel,
  ResourceType,
  ActionType,
  PermissionScope
} from '../types/api';

export const securityService = {
  // === AUTHENTIFICATION ===
  auth: {
    getProfile: async (): Promise<User> => {
      return {
        id: '1',
        username: 'admin',
        email: 'admin@example.com',
        is_active: true,
        is_superuser: true,
        mfa_enabled: false,
        created_at: new Date().toISOString()
      };
    },

    login: async (username: string, password: string): Promise<LoginResponse> => {
      return {
        access_token: 'mock_token',
        token_type: 'bearer',
        expires_in: 3600
      };
    },

    loginJSON: async (username: string, password: string): Promise<LoginResponse> => {
      return {
        access_token: 'mock_token',
        token_type: 'bearer',
        expires_in: 3600,
        requires_mfa: false
      };
    },

    logout: async (): Promise<void> => {
      localStorage.removeItem('access_token');
    },

    changePassword: async (data: ChangePasswordRequest): Promise<void> => {
      // Mock change password
    }
  },

  // === MFA ===
  mfa: {
    getStatus: async (): Promise<MFAStatus> => {
      return {
        enabled: false,
        setup_complete: false,
        devices: [],
        backup_codes_remaining: 0
      };
    },

    setup: async (): Promise<MFASetupResponse> => {
      return {
        secret: 'mock_secret',
        qr_code: 'mock_qr_code',
        backup_codes: ['code1', 'code2']
      };
    },

    verify: async (data: MFAVerifyRequest): Promise<void> => {
      // Mock verification
    },

    verifyLogin: async (mfaToken: string, code: string): Promise<LoginResponse> => {
      return {
        access_token: 'mock_token_mfa',
        token_type: 'bearer',
        expires_in: 3600
      };
    },

    disable: async (password: string): Promise<void> => {
      // Mock disable
    },

    generateBackupCodes: async (): Promise<string[]> => {
      return ['backup1', 'backup2', 'backup3'];
    }
  },

  // === SESSIONS ===
  sessions: {
    getActiveSessions: async (filters?: SecurityFilters['sessions']): Promise<UserSession[]> => {
      return [
        {
          id: '1',
          user_id: '1',
          is_current: true,
          is_suspicious: false,
          created_at: new Date().toISOString(),
          last_activity: new Date().toISOString(),
          expires_at: new Date(Date.now() + 3600000).toISOString(),
          ip_address: '192.168.1.1',
          user_agent: 'Mozilla/5.0',
          location: {
            city: 'Paris',
            country: 'France'
          },
          security_level: 'HIGH' as SecurityLevel
        }
      ];
    },

    getCurrentSession: async (): Promise<UserSession> => {
      return {
        id: '1',
        user_id: '1',
        is_current: true,
        is_suspicious: false,
        created_at: new Date().toISOString(),
        last_activity: new Date().toISOString(),
        expires_at: new Date(Date.now() + 3600000).toISOString(),
        ip_address: '192.168.1.1',
        user_agent: 'Mozilla/5.0',
        location: {
          city: 'Paris',
          country: 'France'
        },
        security_level: 'HIGH' as SecurityLevel
      };
    },

    revokeSession: async (sessionId: string): Promise<void> => {
      // Mock revoke
    },

    extendSession: async (sessionId: string): Promise<void> => {
      // Mock extend
    }
  },

  // === UTILISATEURS ===
  users: {
    getUsers: async (filters?: any, page?: number, perPage?: number): Promise<PaginatedResponse<User>> => {
      return {
        items: [],
        total: 0,
        page: 1,
        pages: 1,
        per_page: 10,
        has_next: false,
        has_prev: false
      };
    },

    createUser: async (userData: CreateUserRequest): Promise<User> => {
      return {
        id: '2',
        username: userData.username,
        email: userData.email,
        is_active: userData.is_active ?? true,
        is_superuser: userData.is_superuser ?? false,
        mfa_enabled: false,
        created_at: new Date().toISOString()
      };
    },

    updateUser: async (userId: string, userData: UpdateUserRequest): Promise<User> => {
      return {
        id: userId,
        username: userData.username ?? 'updated_user',
        email: userData.email ?? 'updated@example.com',
        is_active: userData.is_active ?? true,
        is_superuser: userData.is_superuser ?? false,
        mfa_enabled: false,
        created_at: new Date().toISOString()
      };
    },

    deleteUser: async (userId: string): Promise<void> => {
      // Mock delete
    },

    toggleUserStatus: async (userId: string): Promise<User> => {
      return {
        id: userId,
        username: 'user',
        email: 'user@example.com',
        is_active: false,
        is_superuser: false,
        mfa_enabled: false,
        created_at: new Date().toISOString()
      };
    },

    resetPassword: async (userId: string): Promise<{ temporary_password: string }> => {
      return { temporary_password: 'temp123' };
    },

    revokeAllSessions: async (userId: string): Promise<void> => {
      // Mock revoke all sessions
    }
  },

  // === RBAC ===
  rbac: {
    getRoles: async (filters?: SecurityFilters['roles']): Promise<Role[]> => {
      return [
        {
          id: '1',
          name: 'Admin',
          description: 'Administrator role',
          permissions: [],
          users_count: 1,
          is_system_role: true,
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString()
        }
      ];
    },

    createRole: async (roleData: CreateRoleRequest): Promise<Role> => {
      return {
        id: '2',
        name: roleData.name,
        description: roleData.description,
        permissions: [],
        users_count: 0,
        is_system_role: roleData.is_system_role ?? false,
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString()
      };
    },

    updateRole: async (roleId: string, roleData: UpdateRoleRequest): Promise<Role> => {
      return {
        id: roleId,
        name: roleData.name ?? 'Updated Role',
        description: roleData.description ?? 'Updated description',
        permissions: [],
        users_count: 0,
        is_system_role: false,
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString()
      };
    },

    deleteRole: async (roleId: string): Promise<void> => {
      // Mock delete
    },

    assignRole: async (userId: string, roleId: string): Promise<void> => {
      // Mock assign
    },

    removeRole: async (userId: string, roleId: string): Promise<void> => {
      // Mock remove
    },

    getPermissions: async (): Promise<Permission[]> => {
      return [
        {
          id: '1',
          resource: 'USER' as ResourceType,
          action: 'READ' as ActionType,
          scope: 'global' as PermissionScope,
          description: 'Read users'
        }
      ];
    },

    checkPermission: async (resource: string, action: string, scope?: string): Promise<boolean> => {
      return true; // Mock check
    },

    getUserPermissions: async (userId: string): Promise<Permission[]> => {
      return [
        {
          id: '1',
          resource: 'USER' as ResourceType,
          action: 'READ' as ActionType,
          scope: 'global' as PermissionScope,
          description: 'Read users'
        }
      ];
    }
  },

  // === AUDIT ===
  audit: {
    getAuditEvents: async (filters?: AuditFilters & { page?: number; per_page?: number; search_query?: string }): Promise<PaginatedResponse<AuditEvent>> => {
      return {
        items: [],
        total: 0,
        page: filters?.page || 1,
        pages: 1,
        per_page: filters?.per_page || 10,
        has_next: false,
        has_prev: false
      };
    },

    getAuditEvent: async (id: string): Promise<AuditEvent> => {
      return {
        id,
        event_type: 'LOGIN',
        severity: 'MEDIUM',
        success: true,
        timestamp: new Date().toISOString(),
        username: 'admin',
        user_id: '1',
        resource_type: 'USER',
        action: 'LOGIN'
      };
    },

    exportAuditLogs: async () => ({ download_url: 'mock' }),
    generateReport: async () => ({ summary: {} }),
    
    getAuditStats: async () => ({
      total_events: 100,
      success_rate: 95.5,
      failed_events: 5,
      critical_events: 2,
      events_today: 25,
      events_this_week: 180,
      top_users: ['admin', 'user1', 'user2'],
      top_events: ['LOGIN', 'CREATE', 'UPDATE'],
      events_by_severity: {
        LOW: 50,
        MEDIUM: 30,
        HIGH: 15,
        CRITICAL: 5
      },
      trend: {
        events_growth: 5.2,
        threats_growth: -2.1,
        users_growth: 8.5,
        errors_growth: -15.3,
        logins_growth: 12.8,
        actions_growth: 3.7
      }
    })
  },

  // === DASHBOARD ===
  dashboard: {
    getDashboard: async (): Promise<SecurityDashboard> => {
      return {
        active_sessions: 5,
        threats_blocked_today: 12,
        failed_logins_today: 3,
        successful_logins_today: 45,
        active_alerts: [],
        recent_activities: [],
        security_score: 85,
        uptime_percentage: 99.9,
        last_updated: new Date().toISOString()
      };
    },

    getMetrics: async (days: number): Promise<SecurityMetrics> => {
      return {
        login_attempts: [],
        failed_logins: [],
        threats_detected: [],
        security_alerts: [],
        user_activities: [],
        system_performance: []
      };
    },

    getRealTimeMetrics: async () => {
      return {
        active_users: 15,
        cpu_usage: 45,
        memory_usage: 60,
        threats_blocked_last_hour: 2
      };
    }
  },

  // === ALERTES ===
  alerts: {
    getAlerts: async (filters?: SecurityFilters['security_alerts']): Promise<SecurityAlert[]> => {
      return [];
    },

    assignAlert: async (alertId: string, userId: string): Promise<void> => {
      // Mock assign
    },

    resolveAlert: async (alertId: string, notes: string): Promise<void> => {
      // Mock resolve
    },

    markFalsePositive: async (alertId: string, reason: string): Promise<void> => {
      // Mock false positive
    }
  },

  // === MONITORING ===
  monitoring: {
    getDashboard: async () => {
      return {
        uptime_percentage: 99.9,
        performance_score: 85,
        system_health: 90,
        active_alerts: 2
      };
    },

    getSystemMetrics: async (): Promise<SystemMetrics> => {
      return {
        cpu_usage: 45,
        memory_usage: 60,
        disk_usage: 30,
        network_io: {
          bytes_sent: 1024000,
          bytes_received: 2048000
        },
        database_connections: 25,
        response_time_avg: 150,
        error_rate: 0.5,
        throughput: 1000
      };
    },

    getHealthStatus: async (): Promise<HealthStatus> => {
      return {
        status: 'healthy',
        components: {
          database: { status: 'healthy' },
          redis: { status: 'healthy' },
          storage: { status: 'healthy' },
          external_apis: { status: 'healthy' }
        },
        last_checked: new Date().toISOString()
      };
    }
  },

  // === CONFIGURATION ===
  config: {
    getSecurityConfig: async (): Promise<SecurityConfig> => {
      return {
        password_policy: {
          min_length: 8,
          require_uppercase: true,
          require_lowercase: true,
          require_numbers: true,
          require_special_chars: true,
          max_age_days: 90,
          history_limit: 5
        },
        session_config: {
          timeout_minutes: 60,
          max_concurrent_sessions: 5,
          require_reauth_for_sensitive: true,
          extend_on_activity: true
        },
        mfa_config: {
          required_for_admin: true,
          required_for_all: false,
          backup_codes_count: 10,
          token_validity_seconds: 30
        },
        rate_limiting: {
          login_attempts_per_minute: 5,
          api_requests_per_minute: 100,
          failed_login_lockout_minutes: 15
        },
        audit_config: {
          retention_days: 365,
          log_all_requests: false,
          log_sensitive_data: false,
          export_format: 'json'
        }
      };
    },

    updateSecurityConfig: async (newConfig: Partial<SecurityConfig>): Promise<SecurityConfig> => {
      return {
        password_policy: {
          min_length: 8,
          require_uppercase: true,
          require_lowercase: true,
          require_numbers: true,
          require_special_chars: true,
          max_age_days: 90,
          history_limit: 5
        },
        session_config: {
          timeout_minutes: 60,
          max_concurrent_sessions: 5,
          require_reauth_for_sensitive: true,
          extend_on_activity: true
        },
        mfa_config: {
          required_for_admin: true,
          required_for_all: false,
          backup_codes_count: 10,
          token_validity_seconds: 30
        },
        rate_limiting: {
          login_attempts_per_minute: 5,
          api_requests_per_minute: 100,
          failed_login_lockout_minutes: 15
        },
        audit_config: {
          retention_days: 365,
          log_all_requests: false,
          log_sensitive_data: false,
          export_format: 'json'
        }
      };
    },

    resetSecurityConfig: async (): Promise<void> => {
      // Mock reset
    }
  }
};