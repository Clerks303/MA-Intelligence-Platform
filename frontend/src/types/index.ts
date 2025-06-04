/**
 * Types centralisés pour M&A Intelligence Platform
 * Sprint 1 - Foundation moderne
 */

// ============================================================================
// BASE TYPES
// ============================================================================

export interface User {
  id: string;
  email: string;
  name: string;
  role: 'admin' | 'user' | 'viewer';
  permissions: string[];
  avatar?: string;
  isActive: boolean;
  lastLogin?: Date;
}

export interface Company {
  id: string;
  siren: string;
  siret?: string;
  name: string;
  address?: string;
  sector?: string;
  status: 'active' | 'inactive' | 'prospect';
  score?: number;
  chiffreAffaires?: number;
  effectifs?: number;
  createdAt: Date;
  updatedAt: Date;
}

// ============================================================================
// DOCUMENT MANAGEMENT TYPES
// ============================================================================

export type DocumentType = 
  | 'financial'
  | 'legal'
  | 'due_diligence'
  | 'communication'
  | 'technical'
  | 'hr'
  | 'commercial'
  | 'other';

export type DocumentStatus = 
  | 'draft'
  | 'under_review'
  | 'approved'
  | 'archived'
  | 'deleted';

export interface Document {
  id: string;
  title: string;
  description?: string;
  filename: string;
  type: DocumentType;
  status: DocumentStatus;
  size: number;
  mimeType: string;
  tags: string[];
  version: number;
  ownerId: string;
  createdAt: Date;
  updatedAt: Date;
  accessedAt?: Date;
  viewCount: number;
  downloadCount: number;
}

export interface DocumentVersion {
  id: string;
  documentId: string;
  versionNumber: number;
  majorVersion: number;
  minorVersion: number;
  patchVersion: number;
  comment: string;
  createdBy: string;
  createdAt: Date;
  isCurrent: boolean;
  isPublished: boolean;
}

// ============================================================================
// API TYPES
// ============================================================================

export interface ApiResponse<T = any> {
  success: boolean;
  data?: T;
  message?: string;
  errors?: string[];
  pagination?: {
    page: number;
    limit: number;
    total: number;
    totalPages: number;
  };
}

export interface ApiError {
  message: string;
  code?: string;
  details?: Record<string, any>;
}

// ============================================================================
// UI TYPES
// ============================================================================

export type Theme = 'light' | 'dark' | 'system';

export interface ToastMessage {
  id: string;
  type: 'success' | 'error' | 'warning' | 'info';
  title: string;
  description?: string;
  duration?: number;
}

export interface ModalState {
  isOpen: boolean;
  component?: React.ComponentType<any>;
  props?: Record<string, any>;
}

// ============================================================================
// DASHBOARD & ANALYTICS TYPES
// ============================================================================

export interface KPI {
  id: string;
  name: string;
  value: number;
  unit: string;
  trend: 'up' | 'down' | 'stable';
  trendValue: number;
  target?: number;
  status: 'good' | 'warning' | 'critical';
}

export interface ChartData {
  labels: string[];
  datasets: {
    label: string;
    data: number[];
    backgroundColor?: string[];
    borderColor?: string;
    borderWidth?: number;
  }[];
}

export interface SystemHealth {
  overall: number;
  components: {
    database: number;
    api: number;
    storage: number;
    search: number;
  };
  lastUpdate: Date;
}

// ============================================================================
// FORM TYPES
// ============================================================================

export interface FormFieldError {
  message: string;
  type: string;
}

export interface FormState<T = Record<string, any>> {
  values: T;
  errors: Record<keyof T, FormFieldError | undefined>;
  touched: Record<keyof T, boolean>;
  isSubmitting: boolean;
  isValid: boolean;
}

// ============================================================================
// COLLABORATION TYPES
// ============================================================================

export interface CollaborationSession {
  id: string;
  documentId: string;
  activeUsers: User[];
  createdAt: Date;
  lastActivity: Date;
}

export interface Comment {
  id: string;
  documentId: string;
  position: number;
  content: string;
  authorId: string;
  author: User;
  createdAt: Date;
  isResolved: boolean;
  replies: Comment[];
}

// ============================================================================
// SEARCH TYPES
// ============================================================================

export interface SearchFilter {
  field: string;
  operator: 'eq' | 'ne' | 'lt' | 'lte' | 'gt' | 'gte' | 'contains' | 'in';
  value: any;
}

export interface SearchQuery {
  query?: string;
  filters: SearchFilter[];
  sortBy?: string;
  sortOrder?: 'asc' | 'desc';
  page?: number;
  limit?: number;
}

export interface SearchResult<T = any> {
  items: T[];
  total: number;
  page: number;
  totalPages: number;
  hasNext: boolean;
  hasPrev: boolean;
}

// ============================================================================
// UTILITY TYPES
// ============================================================================

export type Nullable<T> = T | null;
export type Optional<T> = T | undefined;
export type DeepPartial<T> = {
  [P in keyof T]?: T[P] extends object ? DeepPartial<T[P]> : T[P];
};

export type LoadingState = 'idle' | 'loading' | 'success' | 'error';

export interface AsyncState<T = any> {
  data: T | null;
  loading: boolean;
  error: string | null;
  lastUpdated?: Date;
}