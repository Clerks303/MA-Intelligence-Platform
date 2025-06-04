/**
 * Types pour le module Documents - M&A Intelligence Platform
 * Sprint 3 - Gestion Documentaire
 */

// Types alignés avec backend document_storage.py
export interface Document {
  // Identifiants (backend DocumentMetadata)
  document_id: string;
  filename: string;
  original_filename: string;
  
  // Type et classification
  document_type: BackendDocumentType;
  mime_type: string;
  file_extension: string;
  
  // Taille et hachage
  file_size: number;
  md5_hash: string;
  sha256_hash: string;
  
  // Contenu et indexation
  title?: string;
  description?: string;
  tags: string[];
  extracted_text?: string;
  embedding_vector?: number[];
  
  // Sécurité et accès (backend AccessLevel)
  access_level: BackendAccessLevel;
  owner_id: string;
  allowed_users: string[];
  allowed_groups: string[];
  
  // Versioning
  version: number;
  parent_document_id?: string;
  is_latest_version: boolean;
  
  // Workflow (backend DocumentStatus)
  status: BackendDocumentStatus;
  reviewer_id?: string;
  approved_by?: string;
  approved_at?: Date;
  
  // Métadonnées temporelles
  created_at: Date;
  updated_at: Date;
  accessed_at: Date;
  
  // Contexte M&A
  company_id?: string;
  deal_id?: string;
  project_phase?: string;
  
  // Stockage (backend StorageBackend)
  storage_backend: BackendStorageBackend;
  storage_path: string;
  storage_url?: string;
  
  // Analytics
  download_count: number;
  view_count: number;
  last_downloaded?: Date;
  
  // Frontend computed properties
  displayName: string;
  typeIcon: string;
  sizeFormatted: string;
  canEdit: boolean;
  canDelete: boolean;
  canDownload: boolean;
}

export interface Folder {
  id: string;
  name: string;
  path: string;
  parentId?: string;
  level: number;
  children: Folder[];
  documentCount: number;
  totalSize: number;
  createdAt: Date;
  updatedAt: Date;
  createdBy: DocumentUser;
  permissions: FolderPermissions;
  isExpanded?: boolean;
  isLoading?: boolean;
}

export interface DocumentUser {
  id: string;
  name: string;
  email: string;
  avatar?: string;
  role: UserRole;
}

export interface DocumentPermissions {
  canView: boolean;
  canDownload: boolean;
  canEdit: boolean;
  canDelete: boolean;
  canShare: boolean;
  canAnnotate: boolean;
}

export interface FolderPermissions {
  canView: boolean;
  canCreate: boolean;
  canEdit: boolean;
  canDelete: boolean;
  canManagePermissions: boolean;
}

export interface AccessLogEntry {
  id: string;
  action: DocumentAction;
  userId: string;
  userName: string;
  timestamp: Date;
  ipAddress?: string;
  userAgent?: string;
  details?: Record<string, any>;
}

// Enums alignés avec backend
export type BackendDocumentType = 
  | 'financial'
  | 'legal' 
  | 'due_diligence'
  | 'communication'
  | 'technical'
  | 'hr'
  | 'commercial'
  | 'other';

export type BackendDocumentStatus = 
  | 'draft'
  | 'under_review'
  | 'approved' 
  | 'archived'
  | 'deleted';

export type BackendAccessLevel = 
  | 'public'
  | 'internal'
  | 'confidential'
  | 'restricted';

export type BackendStorageBackend = 
  | 'local'
  | 'aws_s3'
  | 'google_drive'
  | 'azure_blob';

// Frontend helper types
export type FrontendDocumentType = 
  | 'pdf' 
  | 'image' 
  | 'document' 
  | 'spreadsheet' 
  | 'presentation'
  | 'archive'
  | 'video'
  | 'audio'
  | 'other';

export type ProcessingStatus = 'pending' | 'processing' | 'completed' | 'failed';

export type OCRStatus = 'not_started' | 'processing' | 'completed' | 'failed';

export type VirusScanStatus = 'pending' | 'scanning' | 'clean' | 'infected' | 'failed';

export type UserRole = 'admin' | 'manager' | 'user' | 'viewer';

export type DocumentAction = 
  | 'view' 
  | 'download' 
  | 'upload' 
  | 'edit' 
  | 'delete' 
  | 'share' 
  | 'annotate'
  | 'move'
  | 'copy';

export type SortField = 'name' | 'size' | 'type' | 'createdAt' | 'updatedAt';

export type SortDirection = 'asc' | 'desc';

export type ViewMode = 'grid' | 'list' | 'tree';

// Upload & Processing
export interface UploadFile {
  id: string;
  file: File;
  name: string;
  size: number;
  type: string;
  progress: number;
  status: UploadStatus;
  error?: string;
  uploadedDocument?: Document;
  folderId?: string;
}

export type UploadStatus = 'pending' | 'uploading' | 'processing' | 'completed' | 'failed' | 'cancelled';

export interface UploadConfig {
  maxFileSize: number; // bytes
  maxFiles: number;
  allowedTypes: string[];
  autoStartUpload: boolean;
  chunkSize: number;
  enableThumbnails: boolean;
  enableOCR: boolean;
  enableVirusScan: boolean;
}

// Search & Filters avec recherche sémantique
export interface DocumentFilters {
  query?: string;
  semantic_search?: boolean; // Active la recherche vectorielle
  document_type?: BackendDocumentType[];
  access_level?: BackendAccessLevel[];
  status?: BackendDocumentStatus[];
  tags?: string[];
  dateRange?: {
    start: Date;
    end: Date;
    field: 'created_at' | 'updated_at' | 'accessed_at';
  };
  sizeRange?: {
    min: number;
    max: number;
  };
  owner_id?: string;
  company_id?: string;
  deal_id?: string;
  project_phase?: string;
  hasExtractedText?: boolean;
  storage_backend?: BackendStorageBackend[];
  min_relevance_score?: number; // Pour recherche sémantique
}

// Analytics documentaire aligné avec backend
export interface DocumentAnalyticsData {
  usage: {
    total_documents: number;
    active_documents: number;
    activity_rate: number;
    usage_by_type: Record<BackendDocumentType, number>;
    views_by_type: Record<BackendDocumentType, number>;
    top_documents: Array<{
      document_id: string;
      title: string;
      view_count: number;
      document_type: BackendDocumentType;
    }>;
    daily_activity: Record<string, number>;
    average_size_by_type: Record<BackendDocumentType, number>;
    total_storage_size: number;
  };
  performance: {
    document_processing: {
      average_upload_time: number;
      average_ocr_time: number;
      average_classification_time: number;
      average_search_time: number;
      indexing_throughput: number;
    };
    storage: {
      disk_usage_percent: number;
      read_latency_ms: number;
      write_latency_ms: number;
      iops: number;
    };
    api_performance: {
      average_response_time: number;
      p95_response_time: number;
      error_rate_percent: number;
      requests_per_second: number;
    };
    search_performance: {
      semantic_search_time: number;
      boolean_search_time: number;
      index_size_mb: number;
      cache_hit_ratio: number;
    };
  };
  quality: {
    completion_rate: number;
    metadata_completeness: number;
    ocr_quality_average: number;
    classification_confidence: number;
    duplicate_rate: number;
    quality_distribution: {
      high_quality: number;
      medium_quality: number;
      low_quality: number;
    };
  };
  business: {
    deal_pipeline: {
      total_active_deals: number;
      pipeline_stages: Record<string, number>;
      average_docs_per_deal: number;
    };
    document_velocity: {
      document_creation_rate: Record<string, number>;
      average_processing_times: Record<string, number>;
      velocity_trend: string;
    };
    due_diligence_progress: {
      total_dd_documents: number;
      dd_status_distribution: Record<BackendDocumentStatus, number>;
      coverage_by_domain: Record<BackendDocumentType, number>;
      completion_rate: number;
    };
    compliance_status: {
      compliance_scores: Record<string, number>;
      overall_compliance: number;
      documents_needing_attention: number;
      compliance_trend: string;
    };
    team_productivity: {
      active_users: number;
      avg_docs_per_user: number;
      most_productive_user: string | null;
      collaboration_rate: number;
    };
  };
}

// Résultat de recherche sémantique
export interface SemanticSearchResult {
  document_id: string;
  metadata: Document;
  relevance_score: number;
  snippet?: string;
  highlighted_text?: string;
}

export interface SearchFacets {
  types: FacetOption[];
  categories: FacetOption[];
  confidentiality: FacetOption[];
  tags: FacetOption[];
  uploaders: FacetOption[];
  sizes: SizeFacet[];
  dates: DateFacet[];
}

export interface FacetOption {
  value: string;
  label: string;
  count: number;
  selected: boolean;
}

export interface SizeFacet {
  range: string;
  min: number;
  max: number;
  count: number;
  selected: boolean;
}

export interface DateFacet {
  period: string;
  start: Date;
  end: Date;
  count: number;
  selected: boolean;
}

// Navigation & Tree
export interface TreeNode {
  id: string;
  name: string;
  type: 'folder' | 'document';
  parentId?: string;
  level: number;
  isExpanded: boolean;
  isLoading: boolean;
  children: TreeNode[];
  path: string;
  metadata?: Folder | Document;
}

export interface BreadcrumbItem {
  id: string;
  name: string;
  path: string;
  type: 'root' | 'folder';
}

// Preview & Viewer
export interface PreviewConfig {
  showThumbnails: boolean;
  enableZoom: boolean;
  enableRotation: boolean;
  enableAnnotations: boolean;
  showMetadata: boolean;
  autoPlay: boolean; // for videos
}

export interface Annotation {
  id: string;
  documentId: string;
  type: AnnotationType;
  position: AnnotationPosition;
  content: string;
  author: DocumentUser;
  createdAt: Date;
  replies: AnnotationReply[];
  resolved: boolean;
}

export type AnnotationType = 'highlight' | 'note' | 'comment' | 'stamp';

export interface AnnotationPosition {
  page?: number; // for PDFs
  x: number;
  y: number;
  width?: number;
  height?: number;
}

export interface AnnotationReply {
  id: string;
  content: string;
  author: DocumentUser;
  createdAt: Date;
}

// API Responses
export interface DocumentsResponse {
  documents: Document[];
  total: number;
  page: number;
  limit: number;
  hasMore: boolean;
  facets?: SearchFacets;
}

export interface FoldersResponse {
  folders: Folder[];
  total: number;
}

export interface UploadResponse {
  success: boolean;
  document?: Document;
  error?: string;
  processingId?: string;
}

// Store State
export interface DocumentsState {
  // Current view
  currentFolderId: string | null;
  viewMode: ViewMode;
  selectedDocuments: string[];
  
  // Filters & Search
  filters: DocumentFilters;
  sortField: SortField;
  sortDirection: SortDirection;
  searchQuery: string;
  
  // Upload
  uploadQueue: UploadFile[];
  uploadConfig: UploadConfig;
  
  // UI State
  isUploadModalOpen: boolean;
  isPreviewModalOpen: boolean;
  previewDocumentId: string | null;
  sidebarExpanded: boolean;
  
  // Navigation
  breadcrumbs: BreadcrumbItem[];
  folderTree: TreeNode[];
  
  // Loading states
  isLoading: boolean;
  isUploading: boolean;
  error: string | null;
}

// Utility types
export interface DocumentStats {
  totalDocuments: number;
  totalSize: number;
  documentsByType: Record<DocumentType, number>;
  documentsByCategory: Record<DocumentCategory, number>;
  recentActivity: AccessLogEntry[];
  topUploaders: Array<{
    user: DocumentUser;
    documentCount: number;
    totalSize: number;
  }>;
}