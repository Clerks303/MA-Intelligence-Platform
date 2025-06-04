/**
 * Service API Avancé pour Documents - M&A Intelligence Platform
 * Sprint 3 - Connexion Backend Complète avec Analytics et Recherche Sémantique
 * 
 * Intégration parfaite avec:
 * - document_storage.py (backend)
 * - document_analytics.py (backend)
 * - Recherche vectorielle/sémantique
 * - Analytics temps réel
 */

import { 
  Document, 
  DocumentFilters,
  DocumentAnalyticsData,
  SemanticSearchResult,
  BackendDocumentType,
  BackendDocumentStatus,
  BackendAccessLevel,
  BackendStorageBackend
} from '../types';

// Configuration API alignée avec backend
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000/api/v1';

// Sanitization avancée pour sécurité
const sanitizeInput = (input: string): string => {
  return input
    .replace(/<[^>]*>/g, '') // Supprime HTML
    .replace(/[<>\"'&]/g, '') // Caractères dangereux
    .trim()
    .substring(0, 1000); // Limite longueur
};

const sanitizeFilename = (filename: string): string => {
  return filename
    .replace(/[^a-zA-Z0-9._-]/g, '_')
    .replace(/_{2,}/g, '_')
    .substring(0, 255);
};

/**
 * Client API avancé pour documents avec fonctionnalités M&A
 */
class AdvancedDocumentService {
  private authToken: string | null = null;

  constructor() {
    this.authToken = localStorage.getItem('auth_token');
  }

  private async fetchWithAuth(endpoint: string, options: RequestInit = {}): Promise<any> {
    const token = this.authToken || localStorage.getItem('auth_token');
    
    const config: RequestInit = {
      ...options,
      headers: {
        'Authorization': token ? `Bearer ${token}` : '',
        ...options.headers,
      },
    };

    // Gestion Content-Type intelligent
    if (!(options.body instanceof FormData)) {
      config.headers = {
        'Content-Type': 'application/json',
        ...config.headers,
      };
    }

    const response = await fetch(`${API_BASE_URL}${endpoint}`, config);

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || errorData.message || `HTTP ${response.status}`);
    }

    if (response.status === 204) {
      return null;
    }

    return response.json();
  }

  // === DOCUMENT STORAGE ENGINE API ===

  /**
   * Stockage de document avec métadonnées complètes
   */
  async storeDocument(
    file: File,
    documentType: BackendDocumentType,
    options: {
      title?: string;
      description?: string;
      tags?: string[];
      accessLevel?: BackendAccessLevel;
      companyId?: string;
      dealId?: string;
      projectPhase?: string;
    } = {},
    onProgress?: (progress: number) => void
  ): Promise<Document> {
    const formData = new FormData();
    formData.append('file', file, sanitizeFilename(file.name));
    formData.append('document_type', documentType);
    formData.append('owner_id', 'current_user'); // Sera remplacé côté backend

    // Métadonnées optionnelles
    if (options.title) formData.append('title', sanitizeInput(options.title));
    if (options.description) formData.append('description', sanitizeInput(options.description));
    if (options.tags) formData.append('tags', JSON.stringify(options.tags.map(sanitizeInput)));
    if (options.accessLevel) formData.append('access_level', options.accessLevel);
    if (options.companyId) formData.append('company_id', options.companyId);
    if (options.dealId) formData.append('deal_id', options.dealId);
    if (options.projectPhase) formData.append('project_phase', sanitizeInput(options.projectPhase));

    // Upload avec suivi de progression
    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      
      xhr.upload.addEventListener('progress', (event) => {
        if (event.lengthComputable && onProgress) {
          const progress = Math.round((event.loaded / event.total) * 100);
          onProgress(progress);
        }
      });

      xhr.addEventListener('load', () => {
        if (xhr.status >= 200 && xhr.status < 300) {
          try {
            const response = JSON.parse(xhr.responseText);
            resolve(this.transformBackendDocument(response));
          } catch (error) {
            reject(new Error('Format de réponse invalide'));
          }
        } else {
          try {
            const errorResponse = JSON.parse(xhr.responseText);
            reject(new Error(errorResponse.detail || `Upload échoué: ${xhr.statusText}`));
          } catch {
            reject(new Error(`Upload échoué: ${xhr.statusText}`));
          }
        }
      });

      xhr.addEventListener('error', () => {
        reject(new Error('Erreur réseau durant l\'upload'));
      });

      xhr.open('POST', `${API_BASE_URL}/documents/store`);
      xhr.setRequestHeader('Authorization', `Bearer ${this.authToken}`);
      xhr.send(formData);
    });
  }

  /**
   * Récupération de document avec métadonnées complètes
   */
  async retrieveDocument(documentId: string): Promise<{ data: ArrayBuffer; metadata: Document }> {
    const response = await fetch(`${API_BASE_URL}/documents/retrieve/${documentId}`, {
      headers: {
        'Authorization': `Bearer ${this.authToken}`,
      }
    });

    if (!response.ok) {
      throw new Error(`Erreur récupération document: ${response.statusText}`);
    }

    const data = await response.arrayBuffer();
    const metadata = this.transformBackendDocument(
      JSON.parse(response.headers.get('X-Document-Metadata') || '{}')
    );

    return { data, metadata };
  }

  /**
   * Recherche sémantique avancée avec embeddings vectoriels
   */
  async searchDocuments(
    query: string,
    options: {
      semanticSearch?: boolean;
      documentType?: BackendDocumentType;
      limit?: number;
      minRelevanceScore?: number;
    } = {}
  ): Promise<SemanticSearchResult[]> {
    const params = new URLSearchParams({
      query: sanitizeInput(query),
      limit: (options.limit || 10).toString(),
    });

    if (options.semanticSearch) params.append('semantic', 'true');
    if (options.documentType) params.append('document_type', options.documentType);
    if (options.minRelevanceScore) params.append('min_score', options.minRelevanceScore.toString());

    const data = await this.fetchWithAuth(`/documents/search?${params}`);
    
    return (data.results || []).map((result: any) => ({
      document_id: result.document_id,
      metadata: this.transformBackendDocument(result.metadata),
      relevance_score: result.relevance_score,
      snippet: result.snippet,
      highlighted_text: result.highlighted_text,
    }));
  }

  /**
   * Indexation du contenu textuel pour recherche sémantique
   */
  async indexDocumentContent(documentId: string, extractedText: string): Promise<void> {
    await this.fetchWithAuth(`/documents/${documentId}/index`, {
      method: 'POST',
      body: JSON.stringify({
        extracted_text: sanitizeInput(extractedText)
      })
    });
  }

  /**
   * Liste des documents avec filtres avancés
   */
  async listDocuments(
    filters: DocumentFilters = {},
    options: {
      limit?: number;
      offset?: number;
    } = {}
  ): Promise<Document[]> {
    const params = new URLSearchParams({
      limit: (options.limit || 50).toString(),
      offset: (options.offset || 0).toString(),
    });

    // Filtres avancés
    if (filters.document_type?.length) {
      params.append('document_type', filters.document_type.join(','));
    }
    if (filters.access_level?.length) {
      params.append('access_level', filters.access_level.join(','));
    }
    if (filters.status?.length) {
      params.append('status', filters.status.join(','));
    }
    if (filters.owner_id) {
      params.append('owner_id', filters.owner_id);
    }
    if (filters.company_id) {
      params.append('company_id', filters.company_id);
    }
    if (filters.deal_id) {
      params.append('deal_id', filters.deal_id);
    }

    const data = await this.fetchWithAuth(`/documents/list?${params}`);
    return (data.documents || []).map(this.transformBackendDocument);
  }

  /**
   * Suppression de document avec audit trail
   */
  async deleteDocument(documentId: string): Promise<boolean> {
    const data = await this.fetchWithAuth(`/documents/${documentId}`, {
      method: 'DELETE'
    });
    return data.success || false;
  }

  /**
   * Métadonnées d'un document
   */
  async getDocumentMetadata(documentId: string): Promise<Document | null> {
    try {
      const data = await this.fetchWithAuth(`/documents/${documentId}/metadata`);
      return data ? this.transformBackendDocument(data) : null;
    } catch (error) {
      return null;
    }
  }

  // === ANALYTICS API ===

  /**
   * Analytics complet documentaire
   */
  async getDocumentAnalytics(timePeriodDays: number = 30): Promise<DocumentAnalyticsData> {
    const data = await this.fetchWithAuth(`/documents/analytics?days=${timePeriodDays}`);
    return this.transformAnalyticsData(data);
  }

  /**
   * Dashboard temps réel
   */
  async getRealTimeDashboard(): Promise<{
    summary: {
      total_documents: number;
      documents_today: number;
      active_users: number;
      storage_used_mb: number;
    };
    recent_activity: Array<{
      document_id: string;
      title: string;
      action: string;
      timestamp: string;
      user_id: string;
    }>;
    type_distribution: Record<BackendDocumentType, number>;
    status_distribution: Record<BackendDocumentStatus, number>;
  }> {
    return this.fetchWithAuth('/documents/dashboard/realtime');
  }

  /**
   * Rapport analytics complet
   */
  async generateAnalyticsReport(options: {
    reportName?: string;
    timePeriodDays?: number;
    generatedBy?: string;
  } = {}): Promise<{
    report_id: string;
    report_name: string;
    metrics: Array<{
      metric_id: string;
      metric_name: string;
      value: number;
      unit: string;
    }>;
    insights: string[];
    recommendations: string[];
    generated_at: string;
  }> {
    return this.fetchWithAuth('/documents/analytics/report', {
      method: 'POST',
      body: JSON.stringify({
        report_name: options.reportName || 'Rapport Documents',
        time_period_days: options.timePeriodDays || 30,
        generated_by: options.generatedBy || 'user'
      })
    });
  }

  /**
   * Statistiques de stockage
   */
  async getStorageStatistics(): Promise<{
    total_documents: number;
    total_size: number;
    total_size_mb: number;
    average_file_size: number;
    backends_configured: BackendStorageBackend[];
    documents_by_type: Record<BackendDocumentType, number>;
    documents_by_status: Record<BackendDocumentStatus, number>;
    upload_count: number;
    download_count: number;
    search_count: number;
  }> {
    return this.fetchWithAuth('/documents/storage/stats');
  }

  // === TRANSFORMATION BACKEND → FRONTEND ===

  private transformBackendDocument(backendDoc: any): Document {
    return {
      document_id: backendDoc.document_id,
      filename: backendDoc.filename,
      original_filename: backendDoc.original_filename,
      
      document_type: backendDoc.document_type,
      mime_type: backendDoc.mime_type,
      file_extension: backendDoc.file_extension,
      
      file_size: backendDoc.file_size,
      md5_hash: backendDoc.md5_hash,
      sha256_hash: backendDoc.sha256_hash,
      
      title: backendDoc.title,
      description: backendDoc.description,
      tags: backendDoc.tags || [],
      extracted_text: backendDoc.extracted_text,
      embedding_vector: backendDoc.embedding_vector,
      
      access_level: backendDoc.access_level,
      owner_id: backendDoc.owner_id,
      allowed_users: backendDoc.allowed_users || [],
      allowed_groups: backendDoc.allowed_groups || [],
      
      version: backendDoc.version,
      parent_document_id: backendDoc.parent_document_id,
      is_latest_version: backendDoc.is_latest_version,
      
      status: backendDoc.status,
      reviewer_id: backendDoc.reviewer_id,
      approved_by: backendDoc.approved_by,
      approved_at: backendDoc.approved_at ? new Date(backendDoc.approved_at) : undefined,
      
      created_at: new Date(backendDoc.created_at),
      updated_at: new Date(backendDoc.updated_at),
      accessed_at: new Date(backendDoc.accessed_at),
      
      company_id: backendDoc.company_id,
      deal_id: backendDoc.deal_id,
      project_phase: backendDoc.project_phase,
      
      storage_backend: backendDoc.storage_backend,
      storage_path: backendDoc.storage_path,
      storage_url: backendDoc.storage_url,
      
      download_count: backendDoc.download_count,
      view_count: backendDoc.view_count,
      last_downloaded: backendDoc.last_downloaded ? new Date(backendDoc.last_downloaded) : undefined,
      
      // Computed properties
      displayName: backendDoc.title || backendDoc.filename,
      typeIcon: this.getTypeIcon(backendDoc.document_type),
      sizeFormatted: this.formatSize(backendDoc.file_size),
      canEdit: backendDoc.access_level !== 'restricted',
      canDelete: backendDoc.owner_id === 'current_user', // Sera vérifié côté backend
      canDownload: true,
    };
  }

  private transformAnalyticsData(backendData: any): DocumentAnalyticsData {
    return {
      usage: backendData.usage || {},
      performance: backendData.performance || {},
      quality: backendData.quality || {},
      business: backendData.business || {},
    };
  }

  // === UTILITAIRES ===

  private getTypeIcon(documentType: BackendDocumentType): string {
    const iconMap: Record<BackendDocumentType, string> = {
      financial: '💰',
      legal: '⚖️',
      due_diligence: '🔍',
      communication: '💬',
      technical: '🔧',
      hr: '👥',
      commercial: '📈',
      other: '📄',
    };
    return iconMap[documentType] || '📄';
  }

  private formatSize(bytes: number): string {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  }

  // === BATCH OPERATIONS ===

  /**
   * Upload multiple de fichiers avec parallélisme contrôlé
   */
  async uploadMultipleDocuments(
    files: File[],
    documentType: BackendDocumentType,
    options: {
      concurrency?: number;
      onProgress?: (fileIndex: number, progress: number) => void;
      onComplete?: (fileIndex: number, document: Document) => void;
      onError?: (fileIndex: number, error: Error) => void;
    } = {}
  ): Promise<Document[]> {
    const concurrency = options.concurrency || 3;
    const results: Document[] = [];
    
    // Traitement par lots
    for (let i = 0; i < files.length; i += concurrency) {
      const batch = files.slice(i, i + concurrency);
      const batchPromises = batch.map(async (file, batchIndex) => {
        const fileIndex = i + batchIndex;
        try {
          const document = await this.storeDocument(
            file,
            documentType,
            {},
            (progress) => options.onProgress?.(fileIndex, progress)
          );
          options.onComplete?.(fileIndex, document);
          return document;
        } catch (error) {
          options.onError?.(fileIndex, error as Error);
          throw error;
        }
      });

      const batchResults = await Promise.allSettled(batchPromises);
      batchResults.forEach((result) => {
        if (result.status === 'fulfilled') {
          results.push(result.value);
        }
      });
    }

    return results;
  }

  /**
   * Export de données documentaires
   */
  async exportDocuments(
    filters: DocumentFilters = {},
    format: 'json' | 'csv' | 'excel' = 'json'
  ): Promise<Blob> {
    const params = new URLSearchParams({ format });
    
    // Ajouter filtres à l'export
    Object.entries(filters).forEach(([key, value]) => {
      if (value) {
        params.append(key, Array.isArray(value) ? value.join(',') : value.toString());
      }
    });

    const response = await fetch(`${API_BASE_URL}/documents/export?${params}`, {
      headers: {
        'Authorization': `Bearer ${this.authToken}`,
      }
    });

    if (!response.ok) {
      throw new Error(`Erreur export: ${response.statusText}`);
    }

    return response.blob();
  }
}

// Instance singleton
export const advancedDocumentService = new AdvancedDocumentService();

// Configuration pour optimisation performance
export const DOCUMENT_CONFIG = {
  UPLOAD: {
    MAX_FILE_SIZE: 100 * 1024 * 1024, // 100MB
    MAX_CONCURRENT_UPLOADS: 3,
    CHUNK_SIZE: 1024 * 1024, // 1MB
    ALLOWED_TYPES: [
      'application/pdf',
      'image/jpeg', 'image/png', 'image/gif', 'image/webp',
      'application/msword', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
      'application/vnd.ms-excel', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
      'text/plain', 'text/csv'
    ],
  },
  SEARCH: {
    DEFAULT_LIMIT: 20,
    MAX_LIMIT: 100,
    MIN_RELEVANCE_SCORE: 0.5,
    ENABLE_SEMANTIC_BY_DEFAULT: true,
  },
  CACHE: {
    ANALYTICS_TTL: 5 * 60, // 5 minutes
    METADATA_TTL: 10 * 60, // 10 minutes
    SEARCH_TTL: 2 * 60, // 2 minutes
  },
};