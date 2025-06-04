/**
 * Store Avancé pour Documents - M&A Intelligence Platform
 * Sprint 3 - Integration complète avec backend, analytics et performance
 */

import { create } from 'zustand';
import { devtools, persist, subscribeWithSelector } from 'zustand/middleware';
import { immer } from 'zustand/middleware/immer';
import { 
  Document, 
  DocumentFilters,
  DocumentAnalyticsData,
  SemanticSearchResult,
  BackendDocumentType,
  BackendDocumentStatus,
  BackendAccessLevel
} from '../types';

interface AdvancedDocumentState {
  // États principaux
  documents: Document[];
  selectedDocuments: Set<string>;
  currentDocument: Document | null;
  
  // Vue et navigation
  viewMode: 'grid' | 'list' | 'tree';
  currentPath: string;
  navigationHistory: string[];
  
  // Recherche et filtres
  searchQuery: string;
  semanticSearchEnabled: boolean;
  semanticResults: SemanticSearchResult[];
  filters: DocumentFilters;
  activeFilters: Set<string>;
  
  // Analytics
  analyticsData: DocumentAnalyticsData | null;
  realTimeMetrics: {
    totalDocuments: number;
    documentsToday: number;
    activeUsers: number;
    storageUsedMB: number;
  } | null;
  
  // Performance et état
  isLoading: boolean;
  isSearching: boolean;
  isAnalyticsLoading: boolean;
  error: string | null;
  
  // Upload avancé
  uploadProgress: Record<string, number>;
  uploadErrors: Record<string, string>;
  batchOperations: Set<string>;
  
  // Preview et édition
  previewDocumentId: string | null;
  isPreviewOpen: boolean;
  editingDocument: Document | null;
  
  // Performance
  virtualizationEnabled: boolean;
  cacheEnabled: boolean;
  prefetchEnabled: boolean;
  
  // Préférences utilisateur
  preferences: {
    documentsPerPage: number;
    autoRefreshInterval: number;
    enableNotifications: boolean;
    defaultSortField: string;
    defaultSortDirection: 'asc' | 'desc';
    compactMode: boolean;
    showMetadata: boolean;
    enableAnimations: boolean;
  };
}

interface AdvancedDocumentActions {
  // Documents CRUD
  setDocuments: (documents: Document[]) => void;
  addDocument: (document: Document) => void;
  updateDocument: (documentId: string, updates: Partial<Document>) => void;
  removeDocument: (documentId: string) => void;
  refreshDocuments: () => void;
  
  // Sélection avancée
  selectDocument: (documentId: string, mode?: 'single' | 'multi' | 'range') => void;
  selectDocuments: (documentIds: string[]) => void;
  clearSelection: () => void;
  selectAll: () => void;
  selectByType: (documentType: BackendDocumentType) => void;
  selectByStatus: (status: BackendDocumentStatus) => void;
  invertSelection: () => void;
  
  // Navigation intelligente
  navigateTo: (path: string) => void;
  navigateBack: () => void;
  navigateForward: () => void;
  addToHistory: (path: string) => void;
  clearHistory: () => void;
  
  // Recherche sémantique
  setSearchQuery: (query: string) => void;
  enableSemanticSearch: () => void;
  disableSemanticSearch: () => void;
  setSemanticResults: (results: SemanticSearchResult[]) => void;
  clearSearch: () => void;
  
  // Filtres avancés
  setFilters: (filters: Partial<DocumentFilters>) => void;
  addFilter: (key: string, value: any) => void;
  removeFilter: (key: string) => void;
  clearAllFilters: () => void;
  saveFilterPreset: (name: string) => void;
  loadFilterPreset: (name: string) => void;
  
  // Analytics
  setAnalyticsData: (data: DocumentAnalyticsData) => void;
  setRealTimeMetrics: (metrics: any) => void;
  refreshAnalytics: () => void;
  
  // Operations par lots
  startBatchOperation: (operationId: string) => void;
  completeBatchOperation: (operationId: string) => void;
  cancelBatchOperation: (operationId: string) => void;
  
  // Upload et progression
  setUploadProgress: (fileName: string, progress: number) => void;
  setUploadError: (fileName: string, error: string) => void;
  clearUploadProgress: () => void;
  
  // Preview et édition
  openPreview: (documentId: string) => void;
  closePreview: () => void;
  startEditing: (document: Document) => void;
  stopEditing: () => void;
  
  // Performance
  toggleVirtualization: () => void;
  toggleCache: () => void;
  togglePrefetch: () => void;
  optimizePerformance: () => void;
  
  // Préférences
  updatePreferences: (preferences: Partial<AdvancedDocumentState['preferences']>) => void;
  resetPreferences: () => void;
  
  // États système
  setLoading: (loading: boolean) => void;
  setSearching: (searching: boolean) => void;
  setAnalyticsLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  
  // Actions utilitaires
  reset: () => void;
  exportState: () => string;
  importState: (state: string) => void;
}

const initialState: AdvancedDocumentState = {
  documents: [],
  selectedDocuments: new Set(),
  currentDocument: null,
  
  viewMode: 'grid',
  currentPath: '/',
  navigationHistory: ['/'],
  
  searchQuery: '',
  semanticSearchEnabled: false,
  semanticResults: [],
  filters: {},
  activeFilters: new Set(),
  
  analyticsData: null,
  realTimeMetrics: null,
  
  isLoading: false,
  isSearching: false,
  isAnalyticsLoading: false,
  error: null,
  
  uploadProgress: {},
  uploadErrors: {},
  batchOperations: new Set(),
  
  previewDocumentId: null,
  isPreviewOpen: false,
  editingDocument: null,
  
  virtualizationEnabled: true,
  cacheEnabled: true,
  prefetchEnabled: true,
  
  preferences: {
    documentsPerPage: 50,
    autoRefreshInterval: 30000, // 30 secondes
    enableNotifications: true,
    defaultSortField: 'created_at',
    defaultSortDirection: 'desc',
    compactMode: false,
    showMetadata: true,
    enableAnimations: true,
  },
};

export const useAdvancedDocumentStore = create<AdvancedDocumentState & AdvancedDocumentActions>()(
  devtools(
    persist(
      subscribeWithSelector(
        immer((set, get) => ({
          ...initialState,

          // === DOCUMENTS CRUD ===
          
          setDocuments: (documents) => set((state) => {
            state.documents = documents;
          }),

          addDocument: (document) => set((state) => {
            const existingIndex = state.documents.findIndex(d => d.document_id === document.document_id);
            if (existingIndex >= 0) {
              state.documents[existingIndex] = document;
            } else {
              state.documents.unshift(document);
            }
          }),

          updateDocument: (documentId, updates) => set((state) => {
            const index = state.documents.findIndex(d => d.document_id === documentId);
            if (index >= 0) {
              Object.assign(state.documents[index], updates);
            }
            
            if (state.currentDocument?.document_id === documentId) {
              Object.assign(state.currentDocument, updates);
            }
          }),

          removeDocument: (documentId) => set((state) => {
            state.documents = state.documents.filter(d => d.document_id !== documentId);
            state.selectedDocuments.delete(documentId);
            
            if (state.currentDocument?.document_id === documentId) {
              state.currentDocument = null;
            }
            
            if (state.previewDocumentId === documentId) {
              state.previewDocumentId = null;
              state.isPreviewOpen = false;
            }
          }),

          refreshDocuments: () => set((state) => {
            state.isLoading = true;
            state.error = null;
          }),

          // === SÉLECTION AVANCÉE ===

          selectDocument: (documentId, mode = 'single') => set((state) => {
            if (mode === 'single') {
              state.selectedDocuments.clear();
              state.selectedDocuments.add(documentId);
            } else if (mode === 'multi') {
              if (state.selectedDocuments.has(documentId)) {
                state.selectedDocuments.delete(documentId);
              } else {
                state.selectedDocuments.add(documentId);
              }
            } else if (mode === 'range') {
              // Implémentation de sélection par plage
              const lastSelected = Array.from(state.selectedDocuments).pop();
              if (lastSelected) {
                const lastIndex = state.documents.findIndex(d => d.document_id === lastSelected);
                const currentIndex = state.documents.findIndex(d => d.document_id === documentId);
                
                if (lastIndex >= 0 && currentIndex >= 0) {
                  const start = Math.min(lastIndex, currentIndex);
                  const end = Math.max(lastIndex, currentIndex);
                  
                  for (let i = start; i <= end; i++) {
                    state.selectedDocuments.add(state.documents[i].document_id);
                  }
                }
              } else {
                state.selectedDocuments.add(documentId);
              }
            }
          }),

          selectDocuments: (documentIds) => set((state) => {
            documentIds.forEach(id => state.selectedDocuments.add(id));
          }),

          clearSelection: () => set((state) => {
            state.selectedDocuments.clear();
          }),

          selectAll: () => set((state) => {
            state.documents.forEach(doc => {
              state.selectedDocuments.add(doc.document_id);
            });
          }),

          selectByType: (documentType) => set((state) => {
            state.documents
              .filter(doc => doc.document_type === documentType)
              .forEach(doc => state.selectedDocuments.add(doc.document_id));
          }),

          selectByStatus: (status) => set((state) => {
            state.documents
              .filter(doc => doc.status === status)
              .forEach(doc => state.selectedDocuments.add(doc.document_id));
          }),

          invertSelection: () => set((state) => {
            const newSelection = new Set<string>();
            state.documents.forEach(doc => {
              if (!state.selectedDocuments.has(doc.document_id)) {
                newSelection.add(doc.document_id);
              }
            });
            state.selectedDocuments = newSelection;
          }),

          // === NAVIGATION INTELLIGENTE ===

          navigateTo: (path) => set((state) => {
            if (state.currentPath !== path) {
              state.navigationHistory.push(path);
              state.currentPath = path;
              state.selectedDocuments.clear();
            }
          }),

          navigateBack: () => set((state) => {
            if (state.navigationHistory.length > 1) {
              state.navigationHistory.pop();
              const previousPath = state.navigationHistory[state.navigationHistory.length - 1];
              state.currentPath = previousPath;
              state.selectedDocuments.clear();
            }
          }),

          navigateForward: () => set((state) => {
            // Implémentation navigation forward si historique disponible
          }),

          addToHistory: (path) => set((state) => {
            if (state.navigationHistory[state.navigationHistory.length - 1] !== path) {
              state.navigationHistory.push(path);
            }
          }),

          clearHistory: () => set((state) => {
            state.navigationHistory = [state.currentPath];
          }),

          // === RECHERCHE SÉMANTIQUE ===

          setSearchQuery: (query) => set((state) => {
            state.searchQuery = query;
            if (!query.trim()) {
              state.semanticResults = [];
            }
          }),

          enableSemanticSearch: () => set((state) => {
            state.semanticSearchEnabled = true;
          }),

          disableSemanticSearch: () => set((state) => {
            state.semanticSearchEnabled = false;
            state.semanticResults = [];
          }),

          setSemanticResults: (results) => set((state) => {
            state.semanticResults = results;
          }),

          clearSearch: () => set((state) => {
            state.searchQuery = '';
            state.semanticResults = [];
          }),

          // === FILTRES AVANCÉS ===

          setFilters: (filters) => set((state) => {
            Object.assign(state.filters, filters);
            
            // Mettre à jour les filtres actifs
            Object.keys(filters).forEach(key => {
              if (filters[key as keyof DocumentFilters]) {
                state.activeFilters.add(key);
              } else {
                state.activeFilters.delete(key);
              }
            });
          }),

          addFilter: (key, value) => set((state) => {
            (state.filters as any)[key] = value;
            state.activeFilters.add(key);
          }),

          removeFilter: (key) => set((state) => {
            delete (state.filters as any)[key];
            state.activeFilters.delete(key);
          }),

          clearAllFilters: () => set((state) => {
            state.filters = {};
            state.activeFilters.clear();
          }),

          saveFilterPreset: (name) => {
            const filters = get().filters;
            localStorage.setItem(`document-filter-preset-${name}`, JSON.stringify(filters));
          },

          loadFilterPreset: (name) => {
            const preset = localStorage.getItem(`document-filter-preset-${name}`);
            if (preset) {
              const filters = JSON.parse(preset);
              get().setFilters(filters);
            }
          },

          // === ANALYTICS ===

          setAnalyticsData: (data) => set((state) => {
            state.analyticsData = data;
          }),

          setRealTimeMetrics: (metrics) => set((state) => {
            state.realTimeMetrics = metrics;
          }),

          refreshAnalytics: () => set((state) => {
            state.isAnalyticsLoading = true;
          }),

          // === OPERATIONS PAR LOTS ===

          startBatchOperation: (operationId) => set((state) => {
            state.batchOperations.add(operationId);
          }),

          completeBatchOperation: (operationId) => set((state) => {
            state.batchOperations.delete(operationId);
          }),

          cancelBatchOperation: (operationId) => set((state) => {
            state.batchOperations.delete(operationId);
          }),

          // === UPLOAD ET PROGRESSION ===

          setUploadProgress: (fileName, progress) => set((state) => {
            state.uploadProgress[fileName] = progress;
          }),

          setUploadError: (fileName, error) => set((state) => {
            state.uploadErrors[fileName] = error;
          }),

          clearUploadProgress: () => set((state) => {
            state.uploadProgress = {};
            state.uploadErrors = {};
          }),

          // === PREVIEW ET ÉDITION ===

          openPreview: (documentId) => set((state) => {
            state.previewDocumentId = documentId;
            state.isPreviewOpen = true;
            
            const document = state.documents.find(d => d.document_id === documentId);
            if (document) {
              state.currentDocument = document;
            }
          }),

          closePreview: () => set((state) => {
            state.previewDocumentId = null;
            state.isPreviewOpen = false;
          }),

          startEditing: (document) => set((state) => {
            state.editingDocument = { ...document };
          }),

          stopEditing: () => set((state) => {
            state.editingDocument = null;
          }),

          // === PERFORMANCE ===

          toggleVirtualization: () => set((state) => {
            state.virtualizationEnabled = !state.virtualizationEnabled;
          }),

          toggleCache: () => set((state) => {
            state.cacheEnabled = !state.cacheEnabled;
          }),

          togglePrefetch: () => set((state) => {
            state.prefetchEnabled = !state.prefetchEnabled;
          }),

          optimizePerformance: () => set((state) => {
            // Nettoyer les données non nécessaires
            if (state.documents.length > 1000) {
              state.documents = state.documents.slice(0, 500);
            }
            
            // Nettoyer l'historique
            if (state.navigationHistory.length > 20) {
              state.navigationHistory = state.navigationHistory.slice(-10);
            }
            
            // Nettoyer la sélection si trop importante
            if (state.selectedDocuments.size > 100) {
              state.selectedDocuments.clear();
            }
          }),

          // === PRÉFÉRENCES ===

          updatePreferences: (preferences) => set((state) => {
            Object.assign(state.preferences, preferences);
          }),

          resetPreferences: () => set((state) => {
            state.preferences = { ...initialState.preferences };
          }),

          // === ÉTATS SYSTÈME ===

          setLoading: (loading) => set((state) => {
            state.isLoading = loading;
          }),

          setSearching: (searching) => set((state) => {
            state.isSearching = searching;
          }),

          setAnalyticsLoading: (loading) => set((state) => {
            state.isAnalyticsLoading = loading;
          }),

          setError: (error) => set((state) => {
            state.error = error;
          }),

          // === UTILITAIRES ===

          reset: () => set(() => ({ ...initialState })),

          exportState: () => {
            const state = get();
            return JSON.stringify({
              preferences: state.preferences,
              filters: state.filters,
              viewMode: state.viewMode,
            });
          },

          importState: (stateString) => {
            try {
              const importedState = JSON.parse(stateString);
              set((state) => {
                Object.assign(state, importedState);
              });
            } catch (error) {
              console.error('Erreur import état:', error);
            }
          },
        }))
      ),
      {
        name: 'ma-intelligence-advanced-documents',
        partialize: (state) => ({
          preferences: state.preferences,
          viewMode: state.viewMode,
          semanticSearchEnabled: state.semanticSearchEnabled,
          virtualizationEnabled: state.virtualizationEnabled,
          cacheEnabled: state.cacheEnabled,
          prefetchEnabled: state.prefetchEnabled,
        }),
      }
    ),
    { name: 'AdvancedDocumentStore' }
  )
);

// === SÉLECTEURS SPÉCIALISÉS ===

export const useDocumentSelection = () => 
  useAdvancedDocumentStore(state => ({
    selectedDocuments: state.selectedDocuments,
    selectedCount: state.selectedDocuments.size,
    hasSelection: state.selectedDocuments.size > 0,
    isAllSelected: state.selectedDocuments.size === state.documents.length,
    selectDocument: state.selectDocument,
    clearSelection: state.clearSelection,
    selectAll: state.selectAll,
    invertSelection: state.invertSelection,
  }));

export const useDocumentSearch = () =>
  useAdvancedDocumentStore(state => ({
    searchQuery: state.searchQuery,
    semanticSearchEnabled: state.semanticSearchEnabled,
    semanticResults: state.semanticResults,
    isSearching: state.isSearching,
    setSearchQuery: state.setSearchQuery,
    enableSemanticSearch: state.enableSemanticSearch,
    disableSemanticSearch: state.disableSemanticSearch,
    clearSearch: state.clearSearch,
  }));

export const useDocumentFilters = () =>
  useAdvancedDocumentStore(state => ({
    filters: state.filters,
    activeFilters: state.activeFilters,
    hasActiveFilters: state.activeFilters.size > 0,
    setFilters: state.setFilters,
    addFilter: state.addFilter,
    removeFilter: state.removeFilter,
    clearAllFilters: state.clearAllFilters,
  }));

export const useDocumentAnalytics = () =>
  useAdvancedDocumentStore(state => ({
    analyticsData: state.analyticsData,
    realTimeMetrics: state.realTimeMetrics,
    isAnalyticsLoading: state.isAnalyticsLoading,
    setAnalyticsData: state.setAnalyticsData,
    refreshAnalytics: state.refreshAnalytics,
  }));

export const useDocumentPreview = () =>
  useAdvancedDocumentStore(state => ({
    previewDocumentId: state.previewDocumentId,
    isPreviewOpen: state.isPreviewOpen,
    currentDocument: state.currentDocument,
    editingDocument: state.editingDocument,
    openPreview: state.openPreview,
    closePreview: state.closePreview,
    startEditing: state.startEditing,
    stopEditing: state.stopEditing,
  }));

export const useDocumentPerformance = () =>
  useAdvancedDocumentStore(state => ({
    virtualizationEnabled: state.virtualizationEnabled,
    cacheEnabled: state.cacheEnabled,
    prefetchEnabled: state.prefetchEnabled,
    toggleVirtualization: state.toggleVirtualization,
    toggleCache: state.toggleCache,
    optimizePerformance: state.optimizePerformance,
  }));

export const useDocumentPreferences = () =>
  useAdvancedDocumentStore(state => ({
    preferences: state.preferences,
    updatePreferences: state.updatePreferences,
    resetPreferences: state.resetPreferences,
  }));

// === HOOKS COMPOSÉS ===

export const useDocumentOperations = () => {
  const store = useAdvancedDocumentStore();
  
  return {
    // Sélection intelligente
    selectByType: (type: BackendDocumentType) => store.selectByType(type),
    selectByStatus: (status: BackendDocumentStatus) => store.selectByStatus(status),
    
    // Operations par lots
    startBatchDelete: () => {
      const operationId = `delete-${Date.now()}`;
      store.startBatchOperation(operationId);
      return operationId;
    },
    
    startBatchMove: () => {
      const operationId = `move-${Date.now()}`;
      store.startBatchOperation(operationId);
      return operationId;
    },
    
    // Navigation intelligente
    navigateToDocument: (documentId: string) => {
      const document = store.documents.find(d => d.document_id === documentId);
      if (document) {
        store.openPreview(documentId);
      }
    },
    
    // Filtres prédéfinis
    showOnlyRecent: () => {
      const oneWeekAgo = new Date();
      oneWeekAgo.setDate(oneWeekAgo.getDate() - 7);
      store.setFilters({
        dateRange: {
          start: oneWeekAgo,
          end: new Date(),
          field: 'created_at',
        },
      });
    },
    
    showOnlyMyDocuments: (userId: string) => {
      store.setFilters({ owner_id: userId });
    },
    
    showHighPriorityOnly: () => {
      store.setFilters({ access_level: ['confidential', 'restricted'] });
    },
  };
};

// Subscription pour auto-refresh
if (typeof window !== 'undefined') {
  useAdvancedDocumentStore.subscribe(
    (state) => state.preferences.autoRefreshInterval,
    (autoRefreshInterval) => {
      // Auto-refresh logic ici
      if (autoRefreshInterval > 0) {
        setInterval(() => {
          const state = useAdvancedDocumentStore.getState();
          if (!state.isLoading && state.documents.length > 0) {
            state.refreshDocuments();
          }
        }, autoRefreshInterval);
      }
    }
  );
}