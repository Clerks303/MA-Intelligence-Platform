/**
 * Service API pour les Documents - M&A Intelligence Platform
 * Sprint 3 - Connexion Backend + Transformateurs
 */

import { 
  Document, 
  Folder, 
  DocumentsResponse, 
  FoldersResponse, 
  UploadResponse,
  DocumentFilters,
  SearchFacets,
  DocumentStats,
  UploadFile,
  Annotation
} from '../types';

// Configuration API
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000/api/v1';
const UPLOAD_CHUNK_SIZE = 1024 * 1024; // 1MB chunks

// Utilitaires de sanitization sécurité
const sanitizeFileName = (filename: string): string => {
  return filename
    .replace(/[^a-zA-Z0-9._-]/g, '_') // Remplace caractères dangereux
    .replace(/_{2,}/g, '_') // Collapse underscores multiples
    .substring(0, 255); // Limite longueur
};

const sanitizeUserInput = (input: string): string => {
  return input
    .replace(/<[^>]*>/g, '') // Supprime HTML tags
    .trim()
    .substring(0, 1000); // Limite longueur
};

// API Client avec gestion d'erreurs et retry
class DocumentApiClient {
  private async fetchWithAuth(endpoint: string, options: RequestInit = {}): Promise<any> {
    const token = localStorage.getItem('auth_token');
    
    const config: RequestInit = {
      ...options,
      headers: {
        'Authorization': token ? `Bearer ${token}` : '',
        ...options.headers,
      },
    };

    // Ne pas ajouter Content-Type pour FormData (multipart)
    if (!(options.body instanceof FormData)) {
      config.headers = {
        'Content-Type': 'application/json',
        ...config.headers,
      };
    }

    const response = await fetch(`${API_BASE_URL}${endpoint}`, config);

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.message || `HTTP ${response.status}: ${response.statusText}`);
    }

    // Gestion des réponses vides (204 No Content)
    if (response.status === 204) {
      return null;
    }

    return response.json();
  }

  // === FOLDERS API ===
  
  async getFolders(parentId?: string): Promise<Folder[]> {
    const params = new URLSearchParams();
    if (parentId) params.append('parent_id', parentId);
    
    const data = await this.fetchWithAuth(`/documents/folders?${params}`);
    return this.transformFolders(data.folders || []);
  }

  async createFolder(name: string, parentId?: string): Promise<Folder> {
    const sanitizedName = sanitizeFileName(name);
    const data = await this.fetchWithAuth('/documents/folders', {
      method: 'POST',
      body: JSON.stringify({
        name: sanitizedName,
        parent_id: parentId,
      }),
    });
    return this.transformFolder(data.folder);
  }

  async updateFolder(id: string, updates: Partial<Pick<Folder, 'name'>>): Promise<Folder> {
    const sanitizedUpdates = {
      ...updates,
      name: updates.name ? sanitizeFileName(updates.name) : undefined,
    };
    
    const data = await this.fetchWithAuth(`/documents/folders/${id}`, {
      method: 'PATCH',
      body: JSON.stringify(sanitizedUpdates),
    });
    return this.transformFolder(data.folder);
  }

  async deleteFolder(id: string, recursive: boolean = false): Promise<void> {
    await this.fetchWithAuth(`/documents/folders/${id}?recursive=${recursive}`, {
      method: 'DELETE',
    });
  }

  async getFolderTree(): Promise<Folder[]> {
    const data = await this.fetchWithAuth('/documents/folders/tree');
    return this.transformFolders(data.folders || []);
  }

  // === DOCUMENTS API ===

  async getDocuments(
    folderId?: string,
    filters?: DocumentFilters,
    page: number = 1,
    limit: number = 50
  ): Promise<DocumentsResponse> {
    const params = new URLSearchParams({
      page: page.toString(),
      limit: limit.toString(),
    });

    if (folderId) params.append('folder_id', folderId);
    
    if (filters) {
      if (filters.query) params.append('q', sanitizeUserInput(filters.query));
      if (filters.type?.length) params.append('types', filters.type.join(','));
      if (filters.category?.length) params.append('categories', filters.category.join(','));
      if (filters.confidentiality?.length) params.append('confidentiality', filters.confidentiality.join(','));
      if (filters.tags?.length) params.append('tags', filters.tags.join(','));
      
      if (filters.dateRange) {
        params.append('date_from', filters.dateRange.start.toISOString());
        params.append('date_to', filters.dateRange.end.toISOString());
        params.append('date_field', filters.dateRange.field);
      }
      
      if (filters.sizeRange) {
        params.append('size_min', filters.sizeRange.min.toString());
        params.append('size_max', filters.sizeRange.max.toString());
      }
    }

    const data = await this.fetchWithAuth(`/documents?${params}`);
    
    return {
      documents: this.transformDocuments(data.documents || []),
      total: data.total || 0,
      page: data.page || 1,
      limit: data.limit || limit,
      hasMore: (data.page * data.limit) < data.total,
      facets: data.facets ? this.transformFacets(data.facets) : undefined,
    };
  }

  async getDocument(id: string): Promise<Document> {
    const data = await this.fetchWithAuth(`/documents/${id}`);
    return this.transformDocument(data.document);
  }

  async updateDocument(id: string, updates: Partial<Document>): Promise<Document> {
    // Sanitize user inputs
    const sanitizedUpdates = {
      ...updates,
      name: updates.name ? sanitizeFileName(updates.name) : undefined,
      description: updates.description ? sanitizeUserInput(updates.description) : undefined,
      tags: updates.tags?.map(tag => sanitizeUserInput(tag)) || undefined,
    };

    const data = await this.fetchWithAuth(`/documents/${id}`, {
      method: 'PATCH',
      body: JSON.stringify(sanitizedUpdates),
    });
    return this.transformDocument(data.document);
  }

  async deleteDocument(id: string): Promise<void> {
    await this.fetchWithAuth(`/documents/${id}`, {
      method: 'DELETE',
    });
  }

  async moveDocument(id: string, targetFolderId: string): Promise<Document> {
    const data = await this.fetchWithAuth(`/documents/${id}/move`, {
      method: 'POST',
      body: JSON.stringify({
        target_folder_id: targetFolderId,
      }),
    });
    return this.transformDocument(data.document);
  }

  async copyDocument(id: string, targetFolderId: string, newName?: string): Promise<Document> {
    const data = await this.fetchWithAuth(`/documents/${id}/copy`, {
      method: 'POST',
      body: JSON.stringify({
        target_folder_id: targetFolderId,
        new_name: newName ? sanitizeFileName(newName) : undefined,
      }),
    });
    return this.transformDocument(data.document);
  }

  // === UPLOAD API ===

  async uploadFile(
    file: File,
    folderId?: string,
    onProgress?: (progress: number) => void
  ): Promise<Document> {
    const formData = new FormData();
    formData.append('file', file, sanitizeFileName(file.name));
    if (folderId) formData.append('folder_id', folderId);

    // Upload avec progress tracking
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
            resolve(this.transformDocument(response.document));
          } catch (error) {
            reject(new Error('Invalid response format'));
          }
        } else {
          try {
            const errorResponse = JSON.parse(xhr.responseText);
            reject(new Error(errorResponse.message || `Upload failed: ${xhr.statusText}`));
          } catch {
            reject(new Error(`Upload failed: ${xhr.statusText}`));
          }
        }
      });

      xhr.addEventListener('error', () => {
        reject(new Error('Network error during upload'));
      });

      xhr.addEventListener('abort', () => {
        reject(new Error('Upload cancelled'));
      });

      xhr.open('POST', `${API_BASE_URL}/documents/upload`);
      xhr.setRequestHeader('Authorization', `Bearer ${localStorage.getItem('auth_token')}`);
      xhr.send(formData);
    });
  }

  async uploadMultipleFiles(
    files: File[],
    folderId?: string,
    onProgress?: (fileId: string, progress: number) => void
  ): Promise<Document[]> {
    const uploadPromises = files.map(async (file) => {
      const fileId = `${file.name}-${Date.now()}`;
      try {
        return await this.uploadFile(file, folderId, (progress) => {
          onProgress?.(fileId, progress);
        });
      } catch (error) {
        throw new Error(`Failed to upload ${file.name}: ${error}`);
      }
    });

    return Promise.all(uploadPromises);
  }

  // === SEARCH & STATS ===

  async searchDocuments(
    query: string,
    filters?: DocumentFilters,
    page: number = 1,
    limit: number = 20
  ): Promise<DocumentsResponse> {
    return this.getDocuments(undefined, { ...filters, query }, page, limit);
  }

  async getSearchFacets(query?: string, folderId?: string): Promise<SearchFacets> {
    const params = new URLSearchParams();
    if (query) params.append('q', sanitizeUserInput(query));
    if (folderId) params.append('folder_id', folderId);

    const data = await this.fetchWithAuth(`/documents/facets?${params}`);
    return this.transformFacets(data.facets);
  }

  async getDocumentStats(folderId?: string): Promise<DocumentStats> {
    const params = new URLSearchParams();
    if (folderId) params.append('folder_id', folderId);

    const data = await this.fetchWithAuth(`/documents/stats?${params}`);
    return this.transformStats(data.stats);
  }

  // === ANNOTATIONS API ===

  async getAnnotations(documentId: string): Promise<Annotation[]> {
    const data = await this.fetchWithAuth(`/documents/${documentId}/annotations`);
    return this.transformAnnotations(data.annotations || []);
  }

  async createAnnotation(documentId: string, annotation: Partial<Annotation>): Promise<Annotation> {
    const sanitizedAnnotation = {
      ...annotation,
      content: annotation.content ? sanitizeUserInput(annotation.content) : '',
    };

    const data = await this.fetchWithAuth(`/documents/${documentId}/annotations`, {
      method: 'POST',
      body: JSON.stringify(sanitizedAnnotation),
    });
    return this.transformAnnotation(data.annotation);
  }

  // === TRANSFORMERS ===

  private transformDocument(backendDoc: any): Document {
    return {
      id: backendDoc.id,
      name: backendDoc.name,
      originalName: backendDoc.original_name || backendDoc.name,
      type: this.mapDocumentType(backendDoc.mime_type),
      mimeType: backendDoc.mime_type,
      size: backendDoc.size,
      extension: backendDoc.extension || this.extractExtension(backendDoc.name),
      path: backendDoc.path,
      folderId: backendDoc.folder_id,
      parentPath: backendDoc.parent_path || '/',
      url: backendDoc.url,
      thumbnailUrl: backendDoc.thumbnail_url,
      previewUrl: backendDoc.preview_url,
      downloadUrl: backendDoc.download_url || backendDoc.url,
      
      createdAt: new Date(backendDoc.created_at),
      updatedAt: new Date(backendDoc.updated_at),
      uploadedBy: this.transformUser(backendDoc.uploaded_by),
      lastModifiedBy: backendDoc.last_modified_by ? this.transformUser(backendDoc.last_modified_by) : undefined,
      version: backendDoc.version || 1,
      
      extractedText: backendDoc.extracted_text,
      pageCount: backendDoc.page_count,
      dimensions: backendDoc.dimensions ? {
        width: backendDoc.dimensions.width,
        height: backendDoc.dimensions.height,
      } : undefined,
      
      companyId: backendDoc.company_id,
      tags: backendDoc.tags || [],
      description: backendDoc.description,
      category: backendDoc.category || 'other',
      confidentiality: backendDoc.confidentiality || 'internal',
      
      permissions: this.transformPermissions(backendDoc.permissions),
      accessLog: (backendDoc.access_log || []).map(this.transformAccessLog),
      
      processingStatus: backendDoc.processing_status || 'completed',
      ocrStatus: backendDoc.ocr_status,
      virusScanStatus: backendDoc.virus_scan_status,
    };
  }

  private transformDocuments(backendDocs: any[]): Document[] {
    return backendDocs.map(doc => this.transformDocument(doc));
  }

  private transformFolder(backendFolder: any): Folder {
    return {
      id: backendFolder.id,
      name: backendFolder.name,
      path: backendFolder.path,
      parentId: backendFolder.parent_id,
      level: backendFolder.level || 0,
      children: (backendFolder.children || []).map(this.transformFolder.bind(this)),
      documentCount: backendFolder.document_count || 0,
      totalSize: backendFolder.total_size || 0,
      createdAt: new Date(backendFolder.created_at),
      updatedAt: new Date(backendFolder.updated_at),
      createdBy: this.transformUser(backendFolder.created_by),
      permissions: this.transformFolderPermissions(backendFolder.permissions),
      isExpanded: false,
      isLoading: false,
    };
  }

  private transformFolders(backendFolders: any[]): Folder[] {
    return backendFolders.map(folder => this.transformFolder(folder));
  }

  private transformUser(backendUser: any): any {
    if (!backendUser) return { id: 'unknown', name: 'Unknown', email: '', role: 'user' };
    
    return {
      id: backendUser.id,
      name: backendUser.name || backendUser.username,
      email: backendUser.email,
      avatar: backendUser.avatar,
      role: backendUser.role || 'user',
    };
  }

  private transformPermissions(backendPerms: any): any {
    return {
      canView: backendPerms?.can_view !== false,
      canDownload: backendPerms?.can_download !== false,
      canEdit: backendPerms?.can_edit !== false,
      canDelete: backendPerms?.can_delete !== false,
      canShare: backendPerms?.can_share !== false,
      canAnnotate: backendPerms?.can_annotate !== false,
    };
  }

  private transformFolderPermissions(backendPerms: any): any {
    return {
      canView: backendPerms?.can_view !== false,
      canCreate: backendPerms?.can_create !== false,
      canEdit: backendPerms?.can_edit !== false,
      canDelete: backendPerms?.can_delete !== false,
      canManagePermissions: backendPerms?.can_manage_permissions !== false,
    };
  }

  private transformAccessLog(logEntry: any): any {
    return {
      id: logEntry.id,
      action: logEntry.action,
      userId: logEntry.user_id,
      userName: logEntry.user_name,
      timestamp: new Date(logEntry.timestamp),
      ipAddress: logEntry.ip_address,
      userAgent: logEntry.user_agent,
      details: logEntry.details,
    };
  }

  private transformFacets(backendFacets: any): SearchFacets {
    return {
      types: (backendFacets.types || []).map(this.transformFacetOption),
      categories: (backendFacets.categories || []).map(this.transformFacetOption),
      confidentiality: (backendFacets.confidentiality || []).map(this.transformFacetOption),
      tags: (backendFacets.tags || []).map(this.transformFacetOption),
      uploaders: (backendFacets.uploaders || []).map(this.transformFacetOption),
      sizes: backendFacets.sizes || [],
      dates: backendFacets.dates || [],
    };
  }

  private transformFacetOption(option: any): any {
    return {
      value: option.value,
      label: option.label || option.value,
      count: option.count || 0,
      selected: false,
    };
  }

  private transformStats(backendStats: any): DocumentStats {
    return {
      totalDocuments: backendStats.total_documents || 0,
      totalSize: backendStats.total_size || 0,
      documentsByType: backendStats.documents_by_type || {},
      documentsByCategory: backendStats.documents_by_category || {},
      recentActivity: (backendStats.recent_activity || []).map(this.transformAccessLog),
      topUploaders: backendStats.top_uploaders || [],
    };
  }

  private transformAnnotation(backendAnnotation: any): Annotation {
    return {
      id: backendAnnotation.id,
      documentId: backendAnnotation.document_id,
      type: backendAnnotation.type,
      position: backendAnnotation.position,
      content: backendAnnotation.content,
      author: this.transformUser(backendAnnotation.author),
      createdAt: new Date(backendAnnotation.created_at),
      replies: (backendAnnotation.replies || []).map((reply: any) => ({
        id: reply.id,
        content: reply.content,
        author: this.transformUser(reply.author),
        createdAt: new Date(reply.created_at),
      })),
      resolved: backendAnnotation.resolved || false,
    };
  }

  private transformAnnotations(backendAnnotations: any[]): Annotation[] {
    return backendAnnotations.map(annotation => this.transformAnnotation(annotation));
  }

  // === UTILITY METHODS ===

  private mapDocumentType(mimeType: string): any {
    if (mimeType.startsWith('image/')) return 'image';
    if (mimeType === 'application/pdf') return 'pdf';
    if (mimeType.includes('spreadsheet') || mimeType.includes('excel')) return 'spreadsheet';
    if (mimeType.includes('presentation') || mimeType.includes('powerpoint')) return 'presentation';
    if (mimeType.includes('document') || mimeType.includes('word') || mimeType.includes('text')) return 'document';
    if (mimeType.startsWith('video/')) return 'video';
    if (mimeType.startsWith('audio/')) return 'audio';
    if (mimeType.includes('zip') || mimeType.includes('archive')) return 'archive';
    return 'other';
  }

  private extractExtension(filename: string): string {
    const parts = filename.split('.');
    return parts.length > 1 ? parts.pop()!.toLowerCase() : '';
  }
}

// Instance singleton
export const documentService = new DocumentApiClient();

// Configuration upload par défaut
export const DEFAULT_UPLOAD_CONFIG = {
  maxFileSize: 100 * 1024 * 1024, // 100MB
  maxFiles: 10,
  allowedTypes: [
    'application/pdf',
    'image/jpeg',
    'image/png',
    'image/gif',
    'image/webp',
    'application/msword',
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    'application/vnd.ms-excel',
    'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    'application/vnd.ms-powerpoint',
    'application/vnd.openxmlformats-officedocument.presentationml.presentation',
    'text/plain',
    'text/csv',
  ],
  autoStartUpload: true,
  chunkSize: UPLOAD_CHUNK_SIZE,
  enableThumbnails: true,
  enableOCR: true,
  enableVirusScan: true,
};