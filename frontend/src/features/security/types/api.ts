// Types API pour les features security
// Définit les interfaces pour les réponses API

export interface PaginatedResponse<T> {
  items: T[];
  total: number;
  page: number;
  pages: number;
  per_page: number;
  has_next: boolean;
  has_prev: boolean;
}

export interface AuditEvent {
  id: string;
  event_type: AuditEventType;
  severity: AuditSeverity;
  success: boolean;
  timestamp: string;
  username?: string;
  user_id?: string;
  resource_type: ResourceType;
  action: ActionType;
  details?: Record<string, any>;
}

export interface User {
  id: string;
  username: string;
  email: string;
  is_active: boolean;
  is_superuser: boolean;
  mfa_enabled: boolean;
  last_login?: string;
  created_at: string;
  roles?: string[];
}

// Enums
export type AuditEventType = 'LOGIN' | 'LOGOUT' | 'CREATE' | 'UPDATE' | 'DELETE' | 'ACCESS' | 'ERROR';
export type AuditSeverity = 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
export type ResourceType = 'USER' | 'COMPANY' | 'SETTINGS' | 'API' | 'DATABASE';
export type ActionType = 'CREATE' | 'READ' | 'UPDATE' | 'DELETE' | 'LOGIN' | 'LOGOUT';

export interface AuditFilters {
  event_type?: AuditEventType[];
  severity?: AuditSeverity[];
  success?: boolean;
  start_date?: string;
  end_date?: string;
  user_id?: string;
  resource_type?: ResourceType[];
  search?: string;
}

export interface AuditReport {
  summary: {
    total_events: number;
    success_rate: number;
    most_common_event: string;
    most_active_user: string;
  };
  timeline: Array<{
    date: string;
    count: number;
  }>;
  by_type: Record<AuditEventType, number>;
  by_severity: Record<AuditSeverity, number>;
}

// === TYPES MANQUANTS POUR LES HOOKS ===

// Requests types
export interface ChangePasswordRequest {
  current_password: string;
  new_password: string;
  confirm_password: string;
}

export interface CreateUserRequest {
  username: string;
  email: string;
  password: string;
  is_active?: boolean;
  is_superuser?: boolean;
  roles?: string[];
}

export interface UpdateUserRequest {
  username?: string;
  email?: string;
  is_active?: boolean;
  is_superuser?: boolean;
  roles?: string[];
}

export interface MFAVerifyRequest {
  token: string;
  code: string;
}

export interface CreateRoleRequest {
  name: string;
  description: string;
  permissions: string[];
  is_system_role?: boolean;
}

export interface UpdateRoleRequest {
  name?: string;
  description?: string;
  permissions?: string[];
}

// MFA Types
export interface MFAStatus {
  enabled: boolean;
  setup_complete: boolean;
  devices: MFADevice[];
  backup_codes_remaining: number;
}

export interface MFADevice {
  id: string;
  name: string;
  type: 'authenticator' | 'sms' | 'email';
  last_used?: string;
  created_at: string;
}

export interface MFASetupResponse {
  secret: string;
  qr_code: string;
  backup_codes: string[];
}

// Session Types
export interface UserSession {
  id: string;
  user_id: string;
  is_current: boolean;
  is_suspicious: boolean;
  created_at: string;
  last_activity: string;
  expires_at: string;
  ip_address: string;
  user_agent: string;
  location?: SessionLocation;
  security_level: SecurityLevel;
}

export interface SessionLocation {
  city: string;
  country: string;
  region?: string;
  latitude?: number;
  longitude?: number;
}

export type SecurityLevel = 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';

// RBAC Types
export interface Role {
  id: string;
  name: string;
  description: string;
  permissions: Permission[];
  users_count?: number;
  is_system_role: boolean;
  parent_role?: string;
  child_roles?: string[];
  created_at: string;
  updated_at: string;
}

export interface Permission {
  id: string;
  resource: ResourceType;
  action: ActionType;
  scope: PermissionScope;
  description: string;
}

export type PermissionScope = 'global' | 'own' | 'organization' | 'department';

// Auth responses
export interface LoginResponse {
  access_token: string;
  token_type: string;
  expires_in: number;
  requires_mfa?: boolean;
  mfa_token?: string;
}

// Security Dashboard Types
export interface SecurityDashboard {
  active_sessions: number;
  threats_blocked_today: number;
  failed_logins_today: number;
  successful_logins_today: number;
  active_alerts: SecurityAlert[];
  recent_activities: AuditEvent[];
  security_score: number;
  uptime_percentage: number;
  last_updated: string;
}

export interface SecurityMetrics {
  login_attempts: MetricPoint[];
  failed_logins: MetricPoint[];
  threats_detected: MetricPoint[];
  security_alerts: MetricPoint[];
  user_activities: MetricPoint[];
  system_performance: MetricPoint[];
}

export interface MetricPoint {
  timestamp: string;
  value: number;
  label?: string;
}

export interface SecurityAlert {
  id: string;
  type: SecurityAlertType;
  severity: AuditSeverity;
  status: SecurityAlertStatus;
  title: string;
  description: string;
  details: Record<string, any>;
  user_id?: string;
  resource_id?: string;
  created_at: string;
  updated_at: string;
  assigned_to?: string;
  resolved_at?: string;
  resolution_notes?: string;
}

export type SecurityAlertType = 
  | 'login_anomaly'
  | 'multiple_failed_logins'
  | 'suspicious_activity'
  | 'unauthorized_access'
  | 'data_breach_attempt'
  | 'privilege_escalation'
  | 'malware_detected'
  | 'phishing_attempt'
  | 'brute_force_attack'
  | 'sql_injection'
  | 'xss_attempt'
  | 'system_compromise';

export type SecurityAlertStatus = 'open' | 'investigating' | 'resolved' | 'false_positive';

// System Monitoring Types
export interface SystemMetrics {
  cpu_usage: number;
  memory_usage: number;
  disk_usage: number;
  network_io: {
    bytes_sent: number;
    bytes_received: number;
  };
  database_connections: number;
  response_time_avg: number;
  error_rate: number;
  throughput: number;
}

export interface HealthStatus {
  status: 'healthy' | 'degraded' | 'unhealthy';
  components: {
    database: ComponentHealth;
    redis: ComponentHealth;
    storage: ComponentHealth;
    external_apis: ComponentHealth;
  };
  last_checked: string;
}

export interface ComponentHealth {
  status: 'healthy' | 'degraded' | 'unhealthy';
  response_time?: number;
  error_rate?: number;
  message?: string;
}

// Security Configuration
export interface SecurityConfig {
  password_policy: PasswordPolicy;
  session_config: SessionConfig;
  mfa_config: MFAConfig;
  rate_limiting: RateLimitingConfig;
  audit_config: AuditConfig;
}

export interface PasswordPolicy {
  min_length: number;
  require_uppercase: boolean;
  require_lowercase: boolean;
  require_numbers: boolean;
  require_special_chars: boolean;
  max_age_days: number;
  history_limit: number;
}

export interface SessionConfig {
  timeout_minutes: number;
  max_concurrent_sessions: number;
  require_reauth_for_sensitive: boolean;
  extend_on_activity: boolean;
}

export interface MFAConfig {
  required_for_admin: boolean;
  required_for_all: boolean;
  backup_codes_count: number;
  token_validity_seconds: number;
}

export interface RateLimitingConfig {
  login_attempts_per_minute: number;
  api_requests_per_minute: number;
  failed_login_lockout_minutes: number;
}

export interface AuditConfig {
  retention_days: number;
  log_all_requests: boolean;
  log_sensitive_data: boolean;
  export_format: 'json' | 'csv' | 'pdf';
}

// Filters
export interface SecurityFilters {
  users: {
    search?: string;
    roles?: string[];
    is_active?: boolean;
    mfa_enabled?: boolean;
    last_login_after?: string;
    created_after?: string;
  };
  roles: {
    search?: string;
    is_system_role?: boolean;
    has_users?: boolean;
  };
  security_alerts: {
    types?: SecurityAlertType[];
    severities?: AuditSeverity[];
    statuses?: SecurityAlertStatus[];
    assigned_to?: string;
    created_after?: string;
    created_before?: string;
  };
  sessions: {
    user_id?: string;
    is_current?: boolean;
    is_suspicious?: boolean;
    security_level?: SecurityLevel[];
  };
}