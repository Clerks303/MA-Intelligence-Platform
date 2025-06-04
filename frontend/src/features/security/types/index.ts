/**
 * Types Module Sécurité - M&A Intelligence Platform
 * Sprint 4 - RBAC, MFA, Audit et Sécurité Avancée
 */

// === UTILISATEURS & AUTHENTIFICATION ===

export interface User {
  id: string;
  username: string;
  email: string;
  is_active: boolean;
  is_superuser: boolean;
  created_at: string;
  updated_at: string;
  last_login?: string;
  
  // Profil étendu
  profile?: UserProfile;
  roles?: Role[];
  permissions?: Permission[];
  sessions?: UserSession[];
  
  // Sécurité
  mfa_enabled?: boolean;
  mfa_setup_complete?: boolean;
  security_level?: SecurityLevel;
  failed_login_attempts?: number;
  locked_until?: string;
}

export interface UserProfile {
  first_name?: string;
  last_name?: string;
  phone?: string;
  department?: string;
  job_title?: string;
  avatar_url?: string;
  timezone?: string;
  language?: string;
  
  // Préférences sécurité
  password_expires_at?: string;
  must_change_password?: boolean;
  security_notifications_enabled?: boolean;
}

export interface CreateUserRequest {
  username: string;
  email: string;
  password: string;
  confirm_password: string;
  roles?: string[];
  profile?: Partial<UserProfile>;
  send_invitation_email?: boolean;
}

export interface UpdateUserRequest {
  username?: string;
  email?: string;
  is_active?: boolean;
  roles?: string[];
  profile?: Partial<UserProfile>;
}

export interface ChangePasswordRequest {
  current_password: string;
  new_password: string;
  confirm_password: string;
}

// === RBAC (RÔLES ET PERMISSIONS) ===

export interface Role {
  id: string;
  name: string;
  description: string;
  is_system_role: boolean;
  created_at: string;
  updated_at: string;
  
  // Relations
  permissions: Permission[];
  users_count?: number;
  parent_role?: Role;
  child_roles?: Role[];
  
  // Métadonnées
  color?: string;
  icon?: string;
  priority?: number;
}

export interface Permission {
  id: string;
  name: string;
  resource: ResourceType;
  action: ActionType;
  scope: PermissionScope;
  description: string;
  created_at: string;
  
  // Conditions
  conditions?: PermissionCondition[];
  is_granted?: boolean;
}

export type ResourceType = 
  | 'users' 
  | 'roles' 
  | 'permissions'
  | 'companies' 
  | 'documents' 
  | 'scraping'
  | 'analytics' 
  | 'settings' 
  | 'audit_logs'
  | 'api_clients'
  | 'monitoring'
  | 'security';

export type ActionType = 
  | 'create' 
  | 'read' 
  | 'update' 
  | 'delete'
  | 'export' 
  | 'import' 
  | 'approve' 
  | 'revoke'
  | 'assign' 
  | 'configure' 
  | 'monitor'
  | 'audit';

export type PermissionScope = 
  | 'own' 
  | 'department' 
  | 'organization'
  | 'system' 
  | 'global';

export interface PermissionCondition {
  field: string;
  operator: 'equals' | 'not_equals' | 'in' | 'not_in' | 'contains' | 'greater_than' | 'less_than';
  value: any;
  logic?: 'AND' | 'OR';
}

export interface CreateRoleRequest {
  name: string;
  description: string;
  permissions: string[];
  parent_role_id?: string;
  metadata?: {
    color?: string;
    icon?: string;
    priority?: number;
  };
}

export interface UpdateRoleRequest {
  name?: string;
  description?: string;
  permissions?: string[];
  parent_role_id?: string;
  metadata?: {
    color?: string;
    icon?: string;
    priority?: number;
  };
}

// === MFA (MULTI-FACTOR AUTHENTICATION) ===

export interface MFASetupResponse {
  secret: string;
  qr_code_url: string;
  backup_codes: string[];
  setup_token: string;
}

export interface MFAVerifyRequest {
  token: string;
  setup_token?: string;
  remember_device?: boolean;
}

export interface MFADevice {
  id: string;
  name: string;
  device_type: MFADeviceType;
  is_primary: boolean;
  created_at: string;
  last_used_at?: string;
  
  // Métadonnées device
  browser?: string;
  os?: string;
  ip_address?: string;
  location?: string;
  fingerprint?: string;
}

export type MFADeviceType = 
  | 'totp_app' 
  | 'sms' 
  | 'email' 
  | 'hardware_token'
  | 'backup_codes';

export interface MFAStatus {
  enabled: boolean;
  setup_complete: boolean;
  primary_method?: MFADeviceType;
  backup_codes_remaining?: number;
  devices: MFADevice[];
  recovery_options: MFADeviceType[];
}

// === SESSIONS ET SÉCURITÉ ===

export interface UserSession {
  session_id: string;
  user_id: string;
  created_at: string;
  last_activity: string;
  expires_at: string;
  is_current: boolean;
  
  // Métadonnées
  ip_address: string;
  user_agent: string;
  location?: GeoLocation;
  device_fingerprint: string;
  
  // Sécurité
  security_level: SecurityLevel;
  auth_methods: AuthMethod[];
  is_suspicious: boolean;
  risk_score?: number;
}

export interface GeoLocation {
  country: string;
  region: string;
  city: string;
  latitude?: number;
  longitude?: number;
  timezone?: string;
  isp?: string;
  is_vpn?: boolean;
  is_tor?: boolean;
}

export type SecurityLevel = 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';

export type AuthMethod = 
  | 'password' 
  | 'mfa_totp' 
  | 'mfa_sms' 
  | 'api_key'
  | 'oauth2' 
  | 'sso';

// === AUDIT ET LOGS ===

export interface AuditEvent {
  event_id: string;
  event_type: AuditEventType;
  timestamp: string;
  user_id?: string;
  username?: string;
  
  // Détails de l'événement
  resource_type: ResourceType;
  resource_id?: string;
  action: ActionType;
  success: boolean;
  severity: AuditSeverity;
  
  // Contexte technique
  ip_address: string;
  user_agent: string;
  geolocation?: GeoLocation;
  session_id?: string;
  
  // Détails additionnels
  details?: Record<string, any>;
  changes?: AuditChange[];
  error_message?: string;
  
  // Compliance
  compliance_tags: string[];
  retention_until: string;
  
  // Métadonnées
  system_generated: boolean;
  correlation_id?: string;
}

export type AuditEventType = 
  | 'authentication'
  | 'authorization' 
  | 'data_access'
  | 'data_modification'
  | 'system_configuration'
  | 'security_event'
  | 'compliance_event'
  | 'performance_event';

export type AuditSeverity = 'INFO' | 'WARNING' | 'ERROR' | 'CRITICAL';

export interface AuditChange {
  field: string;
  old_value: any;
  new_value: any;
  change_type: 'create' | 'update' | 'delete';
}

export interface AuditFilters {
  start_date?: string;
  end_date?: string;
  event_types?: AuditEventType[];
  severities?: AuditSeverity[];
  users?: string[];
  resources?: ResourceType[];
  actions?: ActionType[];
  success?: boolean;
  ip_address?: string;
  search_query?: string;
  compliance_tags?: string[];
}

export interface AuditReport {
  total_events: number;
  filtered_events: number;
  events: AuditEvent[];
  
  // Statistiques
  events_by_type: Record<AuditEventType, number>;
  events_by_severity: Record<AuditSeverity, number>;
  events_by_user: Record<string, number>;
  events_by_day: { date: string; count: number }[];
  
  // Métadonnées
  generated_at: string;
  generated_by: string;
  filters_applied: AuditFilters;
}

// === API CLIENTS ET TOKENS ===

export interface APIClient {
  client_id: string;
  name: string;
  description: string;
  created_at: string;
  updated_at: string;
  
  // Configuration
  auth_methods: APIAuthMethod[];
  scopes: APIScope[];
  rate_limits: RateLimit;
  
  // Sécurité
  is_active: boolean;
  is_verified: boolean;
  allowed_ips?: string[];
  webhook_config?: WebhookConfig;
  
  // Statistiques
  last_used_at?: string;
  total_requests: number;
  api_keys_count: number;
  
  // Métadonnées
  owner_email: string;
  tags?: string[];
}

export interface APIKey {
  key_id: string;
  name: string;
  scopes: APIScope[];
  created_at: string;
  expires_at?: string;
  last_used_at?: string;
  
  // Restrictions
  rate_limits: RateLimit;
  allowed_ips?: string[];
  quotas: APIQuota;
  
  // Statistiques
  requests_count: number;
  errors_count: number;
  
  // Clé masquée (pour affichage)
  masked_key: string;
  
  // Métadonnées
  is_active: boolean;
  tags?: string[];
}

export type APIAuthMethod = 
  | 'bearer_token' 
  | 'api_key' 
  | 'oauth2'
  | 'hmac_signature' 
  | 'basic_auth';

export type APIScope = 
  | 'read' 
  | 'write' 
  | 'admin'
  | 'webhook' 
  | 'integration';

export interface RateLimit {
  requests_per_second?: number;
  requests_per_minute?: number;
  requests_per_hour?: number;
  requests_per_day?: number;
  burst_limit?: number;
}

export interface APIQuota {
  monthly_requests?: number;
  daily_requests?: number;
  data_transfer_gb?: number;
  storage_gb?: number;
}

export interface WebhookConfig {
  url: string;
  secret: string;
  events: string[];
  is_active: boolean;
}

export interface CreateAPIClientRequest {
  name: string;
  description: string;
  auth_methods: APIAuthMethod[];
  scopes: APIScope[];
  rate_limits?: RateLimit;
  allowed_ips?: string[];
  webhook_config?: WebhookConfig;
  tags?: string[];
}

export interface CreateAPIKeyRequest {
  name: string;
  scopes: APIScope[];
  expires_at?: string;
  rate_limits?: RateLimit;
  allowed_ips?: string[];
  quotas?: APIQuota;
  tags?: string[];
}

// === SÉCURITÉ AVANCÉE ===

export interface SecurityDashboard {
  // Métriques temps réel
  active_sessions: number;
  failed_logins_today: number;
  mfa_enabled_users: number;
  security_alerts: number;
  
  // Menaces détectées
  threats_blocked_today: number;
  suspicious_activities: number;
  compromised_accounts: number;
  
  // Compliance
  compliance_score: number;
  policy_violations: number;
  audit_alerts: number;
  
  // Performance sécurité
  average_login_time: number;
  system_health_score: number;
  security_updates_pending: number;
}

export interface SecurityAlert {
  alert_id: string;
  type: SecurityAlertType;
  severity: AuditSeverity;
  title: string;
  description: string;
  created_at: string;
  updated_at: string;
  
  // État
  status: SecurityAlertStatus;
  assigned_to?: string;
  resolved_at?: string;
  resolution_notes?: string;
  
  // Détails
  affected_users?: string[];
  affected_resources?: string[];
  source_ip?: string;
  indicators: SecurityIndicator[];
  
  // Actions
  recommended_actions: SecurityAction[];
  automated_actions?: SecurityAction[];
}

export type SecurityAlertType = 
  | 'brute_force_attack'
  | 'suspicious_login'
  | 'privilege_escalation'
  | 'data_exfiltration'
  | 'malware_detected'
  | 'policy_violation'
  | 'system_compromise'
  | 'insider_threat';

export type SecurityAlertStatus = 
  | 'open' 
  | 'investigating' 
  | 'resolved' 
  | 'false_positive'
  | 'suppressed';

export interface SecurityIndicator {
  type: string;
  value: string;
  confidence: number;
  source: string;
  first_seen: string;
  last_seen: string;
}

export interface SecurityAction {
  action_type: SecurityActionType;
  description: string;
  executed: boolean;
  executed_at?: string;
  result?: string;
}

export type SecurityActionType = 
  | 'block_ip'
  | 'disable_user'
  | 'force_password_reset'
  | 'revoke_sessions'
  | 'require_mfa'
  | 'quarantine_device'
  | 'notify_admin'
  | 'log_event';

// === MONITORING ET MÉTRIQUES ===

export interface SecurityMetrics {
  // Authentification
  login_attempts: TimeSeriesData[];
  successful_logins: TimeSeriesData[];
  failed_logins: TimeSeriesData[];
  mfa_activations: TimeSeriesData[];
  
  // Autorisation
  access_grants: TimeSeriesData[];
  access_denials: TimeSeriesData[];
  privilege_escalations: TimeSeriesData[];
  
  // Sécurité
  threats_detected: TimeSeriesData[];
  security_alerts: TimeSeriesData[];
  compliance_violations: TimeSeriesData[];
  
  // Performance
  response_times: TimeSeriesData[];
  system_load: TimeSeriesData[];
  error_rates: TimeSeriesData[];
}

export interface TimeSeriesData {
  timestamp: string;
  value: number;
  label?: string;
}

// === ÉTAT ET CONTEXTE ===

export interface SecurityState {
  // Utilisateur connecté
  currentUser: User | null;
  isAuthenticated: boolean;
  userPermissions: Permission[];
  userRoles: Role[];
  
  // Session
  currentSession: UserSession | null;
  securityLevel: SecurityLevel;
  mfaRequired: boolean;
  
  // Contexte sécurité
  securityContext: SecurityContext;
  
  // États UI
  isLoading: boolean;
  error: string | null;
}

export interface SecurityContext {
  ip_address: string;
  user_agent: string;
  geolocation?: GeoLocation;
  device_fingerprint: string;
  
  // Menaces
  threat_level: SecurityLevel;
  risk_score: number;
  blocked_features: string[];
  
  // Compliance
  compliance_mode: boolean;
  audit_required: boolean;
  retention_policy: string;
}

// === FILTRES ET PAGINATION ===

export interface SecurityFilters {
  users?: {
    search?: string;
    roles?: string[];
    is_active?: boolean;
    mfa_enabled?: boolean;
    last_login_after?: string;
    created_after?: string;
  };
  
  roles?: {
    search?: string;
    is_system_role?: boolean;
  };
  
  sessions?: {
    user_id?: string;
    is_current?: boolean;
    security_level?: SecurityLevel[];
    created_after?: string;
  };
  
  audit?: AuditFilters;
  
  security_alerts?: {
    types?: SecurityAlertType[];
    severities?: AuditSeverity[];
    statuses?: SecurityAlertStatus[];
    assigned_to?: string;
    created_after?: string;
  };
}

export interface PaginatedResponse<T> {
  items: T[];
  total: number;
  page: number;
  pages: number;
  per_page: number;
  has_next: boolean;
  has_prev: boolean;
}

// === CONFIGURATION ET PARAMÈTRES ===

export interface SecurityConfig {
  // Authentification
  password_policy: PasswordPolicy;
  session_settings: SessionSettings;
  mfa_settings: MFASettings;
  
  // Sécurité
  security_headers: SecurityHeaders;
  rate_limiting: RateLimitingConfig;
  threat_detection: ThreatDetectionConfig;
  
  // Audit
  audit_settings: AuditSettings;
  compliance_settings: ComplianceSettings;
}

export interface PasswordPolicy {
  min_length: number;
  require_uppercase: boolean;
  require_lowercase: boolean;
  require_numbers: boolean;
  require_symbols: boolean;
  password_history: number;
  max_age_days: number;
}

export interface SessionSettings {
  max_duration_hours: number;
  idle_timeout_minutes: number;
  max_concurrent_sessions: number;
  require_fresh_auth_for_sensitive: boolean;
}

export interface MFASettings {
  enforce_for_all_users: boolean;
  enforce_for_admin_users: boolean;
  allowed_methods: MFADeviceType[];
  backup_codes_count: number;
  remember_device_days: number;
}

export interface SecurityHeaders {
  enable_hsts: boolean;
  enable_csp: boolean;
  enable_xss_protection: boolean;
  enable_frame_options: boolean;
  custom_headers: Record<string, string>;
}

export interface RateLimitingConfig {
  global_limits: RateLimit;
  per_user_limits: RateLimit;
  per_ip_limits: RateLimit;
  burst_tolerance: number;
  block_duration_minutes: number;
}

export interface ThreatDetectionConfig {
  enable_ml_detection: boolean;
  suspicious_login_threshold: number;
  brute_force_threshold: number;
  geo_velocity_threshold: number;
  device_fingerprint_required: boolean;
}

export interface AuditSettings {
  enable_audit_logging: boolean;
  log_successful_events: boolean;
  log_failed_events: boolean;
  retention_days: number;
  export_format: 'json' | 'csv' | 'siem';
}

export interface ComplianceSettings {
  enable_gdpr_mode: boolean;
  enable_iso27001_mode: boolean;
  data_retention_years: number;
  auto_anonymization: boolean;
  consent_tracking: boolean;
}

// === EXPORTS ===

// Ré-exporter tous les types depuis api.ts
export * from './api';