/**
 * Module Documents - M&A Intelligence Platform
 * Sprint 3 - Export principal du module documentaire complet
 */

// === PAGES ===
export { DocumentManagement } from './pages/DocumentManagement';

// === ROUTES ===
export { DocumentRoutes } from './routes';

// === COMPOSANTS PRINCIPAUX ===
export { AdvancedDocumentUpload } from './components/AdvancedDocumentUpload';
export { VirtualizedDocumentTree } from './components/VirtualizedDocumentTree';
export { ModernDocumentPreview } from './components/ModernDocumentPreview';
export { DocumentPreview } from './components/DocumentPreview';
export { DocumentUpload } from './components/DocumentUpload';
export { FolderTree } from './components/FolderTree';

// === COMPOSANTS PREVIEW ===
export { PDFPreview } from './components/previews/PDFPreview';
export { ImagePreview } from './components/previews/ImagePreview';
export { TextPreview } from './components/previews/TextPreview';
export { VideoPreview } from './components/previews/VideoPreview';

// === HOOKS ===
export { useDocuments } from './hooks/useDocuments';
export { 
  useAdvancedDocuments,
  useSemanticSearch,
  useDocumentAnalytics,
  useAdvancedUpload,
  useDocumentIndexing,
  useVirtualizedDocuments
} from './hooks/useAdvancedDocuments';

// === STORES ===
export { 
  useAdvancedDocumentStore,
  useDocumentSelection,
  useDocumentSearch,
  useDocumentFilters,
  useDocumentAnalytics as useDocumentAnalyticsStore,
  useDocumentPreview,
  useDocumentPerformance,
  useDocumentPreferences,
  useDocumentOperations
} from './stores/advancedDocumentStore';

export { 
  useDocumentStore,
  useDocumentSelection as useDocumentSelectionBasic,
  useDocumentFilters as useDocumentFiltersBasic,
  useDocumentNavigation,
  useDocumentUploadState
} from './stores/documentStore';

// === SERVICES ===
export { documentService } from './services/documentService';
export { advancedDocumentService } from './services/advancedDocumentService';

// === TYPES ===
export type {
  Document,
  DocumentFilters,
  DocumentsState,
  ViewMode,
  SortField,
  SortDirection,
  BreadcrumbItem,
  TreeNode,
  UploadConfig,
  UploadFile,
  PreviewConfig,
  DocumentAnalyticsData,
  SemanticSearchResult,
  BackendDocumentType,
  BackendDocumentStatus,
  BackendAccessLevel,
  FrontendDocumentType,
  FrontendDocumentStatus,
  DocumentMetadata,
  DocumentSearchRequest,
  DocumentUploadRequest,
  DocumentUpdateRequest,
  AnalyticsTimeRange,
  PerformanceMetrics
} from './types';

// === CONSTANTES ===
export { 
  DEFAULT_UPLOAD_CONFIG,
  SUPPORTED_MIME_TYPES,
  MAX_FILE_SIZE,
  SUPPORTED_DOCUMENT_TYPES 
} from './services/documentService';

// === CONFIGURATION PAR DÉFAUT ===
export const DOCUMENT_MODULE_CONFIG = {
  // Pagination
  defaultPageSize: 50,
  maxPageSize: 200,
  
  // Performance
  enableVirtualization: true,
  enableSemanticSearch: true,
  enableAnalytics: true,
  
  // Upload
  maxFileSize: 100 * 1024 * 1024, // 100MB
  allowedTypes: [
    'application/pdf',
    'image/jpeg',
    'image/png',
    'image/gif',
    'text/plain',
    'application/msword',
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    'application/vnd.ms-excel',
    'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
  ],
  
  // Cache
  cacheTimeout: 5 * 60 * 1000, // 5 minutes
  
  // Recherche
  searchDebounceMs: 300,
  minSearchLength: 2,
  
  // UI
  compactMode: false,
  showMetadata: true,
  enableAnimations: true,
} as const;

// === HELPERS ===
export const DocumentHelpers = {
  formatFileSize: (bytes: number): string => {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return `${parseFloat((bytes / Math.pow(k, i)).toFixed(1))} ${sizes[i]}`;
  },

  formatDate: (date: Date | string): string => {
    const d = typeof date === 'string' ? new Date(date) : date;
    return new Intl.DateTimeFormat('fr-FR', {
      year: 'numeric',
      month: 'long',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    }).format(d);
  },

  getDocumentTypeIcon: (type: string): string => {
    const icons: Record<string, string> = {
      financial: '💰',
      legal: '⚖️',
      due_diligence: '🔍',
      communication: '💬',
      technical: '🔧',
      hr: '👥',
      commercial: '📈',
      other: '📄',
    };
    return icons[type] || '📄';
  },

  isImageType: (mimeType: string): boolean => {
    return mimeType.startsWith('image/');
  },

  isPDFType: (mimeType: string): boolean => {
    return mimeType === 'application/pdf';
  },

  isVideoType: (mimeType: string): boolean => {
    return mimeType.startsWith('video/');
  },

  isTextType: (mimeType: string): boolean => {
    return mimeType.startsWith('text/') || 
           mimeType.includes('document') || 
           mimeType.includes('word');
  },

  getPreviewType: (mimeType: string, fileExtension: string): 'pdf' | 'image' | 'video' | 'text' | 'unsupported' => {
    if (mimeType === 'application/pdf') return 'pdf';
    if (mimeType.startsWith('image/')) return 'image';
    if (mimeType.startsWith('video/')) return 'video';
    if (mimeType.startsWith('text/') || 
        mimeType.includes('document') || 
        mimeType.includes('word') ||
        ['.txt', '.md', '.csv', '.json', '.xml'].includes(fileExtension.toLowerCase())) {
      return 'text';
    }
    return 'unsupported';
  },
};

// === VERSION ===
export const DOCUMENT_MODULE_VERSION = '3.0.0';

// === MÉTADONNÉES MODULE ===
export const DOCUMENT_MODULE_INFO = {
  name: 'M&A Intelligence Documents',
  version: DOCUMENT_MODULE_VERSION,
  description: 'Module de gestion documentaire avancé avec IA, analytics et performance optimisée',
  sprint: 3,
  features: [
    'Navigation arbre virtualisée',
    'Upload drag & drop multi-fichiers',
    'Preview multi-formats',
    'Recherche sémantique IA',
    'Analytics en temps réel',
    'State management hybride',
    'Performance optimisée',
    'Sécurité intégrée',
    'Mobile responsive',
    'Backend FastAPI intégré'
  ],
  technicalStack: {
    frontend: ['React 18', 'TypeScript', 'Tailwind CSS', 'ShadCN/UI'],
    stateManagement: ['Zustand', 'TanStack Query'],
    performance: ['React Window', 'Lazy Loading', 'Code Splitting'],
    backend: ['FastAPI', 'Document Storage Engine', 'Semantic Search'],
    testing: ['Jest', 'React Testing Library', 'E2E Tests']
  }
} as const;