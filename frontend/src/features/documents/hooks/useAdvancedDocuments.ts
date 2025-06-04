/**
 * Hooks Avancés pour Documents - M&A Intelligence Platform
 * Sprint 3 - Analytics, Recherche Sémantique, Performance
 * 
 * Hooks spécialisés pour:
 * - Recherche sémantique vectorielle
 * - Analytics documentaire temps réel
 * - Upload avec optimisations performance
 * - Gestion d'état hybride (Zustand + TanStack Query)
 */

import { useQuery, useMutation, useQueryClient, useInfiniteQuery } from '@tanstack/react-query';
import { useCallback, useMemo, useState, useEffect } from 'react';
import { useVirtualizer } from '@tanstack/react-virtual';
import { debounce } from 'lodash-es';

import { 
  Document, 
  DocumentFilters,
  DocumentAnalyticsData,
  SemanticSearchResult,
  BackendDocumentType,
  BackendDocumentStatus,
  BackendAccessLevel
} from '../types';
import { advancedDocumentService, DOCUMENT_CONFIG } from '../services/advancedDocumentService';
import { useToast } from '../../../hooks';
import { useDocumentStore } from '../stores/documentStore';

// === QUERY KEYS AVANCÉES ===
export const ADVANCED_QUERY_KEYS = {
  analytics: ['documents', 'analytics'] as const,
  analyticsData: (days: number) => [...ADVANCED_QUERY_KEYS.analytics, { days }] as const,
  
  semantic: ['documents', 'semantic'] as const,
  semanticSearch: (query: string, options?: any) => 
    [...ADVANCED_QUERY_KEYS.semantic, 'search', { query, ...options }] as const,
  
  storage: ['documents', 'storage'] as const,
  storageStats: () => [...ADVANCED_QUERY_KEYS.storage, 'stats'] as const,
  
  realtime: ['documents', 'realtime'] as const,
  dashboard: () => [...ADVANCED_QUERY_KEYS.realtime, 'dashboard'] as const,
} as const;

// === RECHERCHE SÉMANTIQUE ===

/**
 * Hook pour recherche sémantique avec embeddings vectoriels
 */
export const useSemanticSearch = (options?: {
  autoSearch?: boolean;
  minQueryLength?: number;
  debounceMs?: number;
}) => {
  const { toast } = useToast();
  const [query, setQuery] = useState('');
  const [isSemanticEnabled, setIsSemanticEnabled] = useState(true);
  const [searchOptions, setSearchOptions] = useState<{
    documentType?: BackendDocumentType;
    limit?: number;
    minRelevanceScore?: number;
  }>({});

  const minQueryLength = options?.minQueryLength || 3;
  const debounceMs = options?.debounceMs || 300;

  // Recherche avec debouncing
  const debouncedSearch = useMemo(
    () => debounce((searchQuery: string) => {
      if (searchQuery.length >= minQueryLength && options?.autoSearch) {
        refetch();
      }
    }, debounceMs),
    [minQueryLength, options?.autoSearch, debounceMs]
  );

  const {
    data: results,
    isLoading,
    error,
    refetch,
  } = useQuery({
    queryKey: ADVANCED_QUERY_KEYS.semanticSearch(query, {
      semantic: isSemanticEnabled,
      ...searchOptions,
    }),
    queryFn: () => advancedDocumentService.searchDocuments(query, {
      semanticSearch: isSemanticEnabled,
      ...searchOptions,
    }),
    enabled: query.length >= minQueryLength,
    staleTime: DOCUMENT_CONFIG.CACHE.SEARCH_TTL * 1000,
  });

  // Effect pour debounced search
  useEffect(() => {
    debouncedSearch(query);
    return () => {
      debouncedSearch.cancel();
    };
  }, [query, debouncedSearch]);

  const search = useCallback(async (newQuery: string) => {
    setQuery(newQuery);
    if (newQuery.length >= minQueryLength) {
      try {
        await refetch();
      } catch (error) {
        toast('Erreur lors de la recherche', 'error');
      }
    }
  }, [refetch, minQueryLength, toast]);

  const toggleSemanticMode = useCallback(() => {
    setIsSemanticEnabled(prev => !prev);
    if (query.length >= minQueryLength) {
      refetch();
    }
  }, [query.length, minQueryLength, refetch]);

  const updateSearchOptions = useCallback((newOptions: typeof searchOptions) => {
    setSearchOptions(newOptions);
    if (query.length >= minQueryLength) {
      refetch();
    }
  }, [query.length, minQueryLength, refetch]);

  return {
    query,
    setQuery,
    results: results || [],
    isLoading,
    error,
    search,
    isSemanticEnabled,
    toggleSemanticMode,
    searchOptions,
    updateSearchOptions,
    hasResults: (results?.length || 0) > 0,
  };
};

// === ANALYTICS TEMPS RÉEL ===

/**
 * Hook pour analytics documentaire complète
 */
export const useDocumentAnalytics = (timePeriodDays: number = 30) => {
  const {
    data: analyticsData,
    isLoading,
    error,
    refetch,
  } = useQuery({
    queryKey: ADVANCED_QUERY_KEYS.analyticsData(timePeriodDays),
    queryFn: () => advancedDocumentService.getDocumentAnalytics(timePeriodDays),
    staleTime: DOCUMENT_CONFIG.CACHE.ANALYTICS_TTL * 1000,
    refetchInterval: 5 * 60 * 1000, // Refresh toutes les 5 minutes
  });

  // Métriques computées
  const computedMetrics = useMemo(() => {
    if (!analyticsData) return null;

    const { usage, performance, quality, business } = analyticsData;

    return {
      // Score de santé global
      overallHealth: Math.round(
        (performance.api_performance?.requests_per_second || 0) > 50 ? 90 :
        (performance.api_performance?.error_rate_percent || 0) < 1 ? 85 : 70
      ),
      
      // Tendances
      trends: {
        documents: usage.total_documents > 0 ? 'positive' : 'neutral',
        activity: usage.activity_rate > 0.5 ? 'positive' : 'negative',
        quality: quality.completion_rate > 0.8 ? 'positive' : 'negative',
      },
      
      // Top insights
      insights: [
        usage.activity_rate > 0.8 ? '📈 Excellente adoption documentaire' : null,
        quality.completion_rate > 0.9 ? '✅ Métadonnées de haute qualité' : null,
        performance.search_performance?.cache_hit_ratio > 0.8 ? '⚡ Performance de recherche optimale' : null,
        business.deal_pipeline?.total_active_deals > 10 ? '🎯 Pipeline M&A actif' : null,
      ].filter(Boolean),
    };
  }, [analyticsData]);

  return {
    analyticsData,
    computedMetrics,
    isLoading,
    error,
    refetch,
  };
};

/**
 * Hook pour dashboard temps réel
 */
export const useRealTimeDashboard = () => {
  const {
    data: dashboardData,
    isLoading,
    error,
  } = useQuery({
    queryKey: ADVANCED_QUERY_KEYS.dashboard(),
    queryFn: () => advancedDocumentService.getRealTimeDashboard(),
    staleTime: 60 * 1000, // 1 minute pour temps réel
    refetchInterval: 30 * 1000, // Refresh toutes les 30 secondes
  });

  return {
    dashboardData,
    isLoading,
    error,
  };
};

// === UPLOAD AVANCÉ AVEC PERFORMANCE ===

/**
 * Hook pour upload optimisé avec batch processing
 */
export const useAdvancedUpload = () => {
  const queryClient = useQueryClient();
  const { toast } = useToast();
  const [uploadProgress, setUploadProgress] = useState<Record<string, number>>({});
  const [uploadErrors, setUploadErrors] = useState<Record<string, string>>({});

  // Upload multiple optimisé
  const uploadMultipleMutation = useMutation({
    mutationFn: async ({ 
      files, 
      documentType, 
      options 
    }: { 
      files: File[];
      documentType: BackendDocumentType;
      options?: any;
    }) => {
      const results = await advancedDocumentService.uploadMultipleDocuments(
        files,
        documentType,
        {
          concurrency: DOCUMENT_CONFIG.UPLOAD.MAX_CONCURRENT_UPLOADS,
          onProgress: (fileIndex, progress) => {
            const fileName = files[fileIndex]?.name;
            if (fileName) {
              setUploadProgress(prev => ({
                ...prev,
                [fileName]: progress,
              }));
            }
          },
          onError: (fileIndex, error) => {
            const fileName = files[fileIndex]?.name;
            if (fileName) {
              setUploadErrors(prev => ({
                ...prev,
                [fileName]: error.message,
              }));
            }
          },
        }
      );

      return results;
    },
    onSuccess: (documents) => {
      // Invalidate relevant queries
      queryClient.invalidateQueries({ queryKey: ['documents'] });
      
      // Clear progress
      setUploadProgress({});
      setUploadErrors({});
      
      toast(`${documents.length} fichiers uploadés avec succès`, 'success');
    },
    onError: (error) => {
      toast(`Erreur upload: ${error.message}`, 'error');
    },
  });

  // Validation des fichiers
  const validateFiles = useCallback((files: File[]) => {
    const errors: string[] = [];
    const { MAX_FILE_SIZE, MAX_CONCURRENT_UPLOADS, ALLOWED_TYPES } = DOCUMENT_CONFIG.UPLOAD;

    if (files.length > MAX_CONCURRENT_UPLOADS) {
      errors.push(`Maximum ${MAX_CONCURRENT_UPLOADS} fichiers à la fois`);
    }

    files.forEach(file => {
      if (file.size > MAX_FILE_SIZE) {
        errors.push(`${file.name}: Taille maximale ${MAX_FILE_SIZE / (1024 * 1024)}MB`);
      }
      
      if (!ALLOWED_TYPES.includes(file.type)) {
        errors.push(`${file.name}: Type de fichier non autorisé`);
      }
    });

    return errors;
  }, []);

  const uploadFiles = useCallback(async (
    files: File[],
    documentType: BackendDocumentType,
    options?: any
  ) => {
    const validationErrors = validateFiles(files);
    if (validationErrors.length > 0) {
      validationErrors.forEach(error => toast(error, 'error'));
      return;
    }

    await uploadMultipleMutation.mutateAsync({ files, documentType, options });
  }, [validateFiles, uploadMultipleMutation, toast]);

  return {
    uploadFiles,
    uploadProgress,
    uploadErrors,
    isUploading: uploadMultipleMutation.isPending,
    validateFiles,
  };
};

// === VIRTUALISATION PERFORMANCE ===

/**
 * Hook pour virtualisation de longues listes de documents
 */
export const useVirtualizedDocuments = (
  documents: Document[],
  containerRef: React.RefObject<HTMLElement>
) => {
  const rowVirtualizer = useVirtualizer({
    count: documents.length,
    getScrollElement: () => containerRef.current,
    estimateSize: () => 80, // Hauteur estimée d'un item
    overscan: 10, // Nombre d'items à pré-rendre
  });

  const virtualItems = rowVirtualizer.getVirtualItems();
  const totalSize = rowVirtualizer.getTotalSize();

  // Métriques de performance
  const performanceMetrics = useMemo(() => ({
    totalItems: documents.length,
    visibleItems: virtualItems.length,
    renderRatio: virtualItems.length / Math.max(documents.length, 1),
    memoryOptimization: `${Math.round((1 - (virtualItems.length / Math.max(documents.length, 1))) * 100)}%`,
  }), [documents.length, virtualItems.length]);

  return {
    rowVirtualizer,
    virtualItems,
    totalSize,
    performanceMetrics,
  };
};

// === STORAGE AVANCÉ ===

/**
 * Hook pour statistiques de stockage
 */
export const useStorageStats = () => {
  const {
    data: storageStats,
    isLoading,
    error,
    refetch,
  } = useQuery({
    queryKey: ADVANCED_QUERY_KEYS.storageStats(),
    queryFn: () => advancedDocumentService.getStorageStatistics(),
    staleTime: DOCUMENT_CONFIG.CACHE.ANALYTICS_TTL * 1000,
  });

  // Calculs dérivés
  const computedStats = useMemo(() => {
    if (!storageStats) return null;

    const usagePercentage = (storageStats.total_size_mb / 1000) * 100; // Assuming 1GB limit
    const avgSizeMB = storageStats.average_file_size / (1024 * 1024);

    return {
      ...storageStats,
      usagePercentage: Math.min(usagePercentage, 100),
      avgSizeMB,
      healthStatus: usagePercentage > 90 ? 'warning' : usagePercentage > 70 ? 'good' : 'excellent',
    };
  }, [storageStats]);

  return {
    storageStats,
    computedStats,
    isLoading,
    error,
    refetch,
  };
};

// === INDEXATION ET CONTENU ===

/**
 * Hook pour indexation de contenu documentaire
 */
export const useDocumentIndexing = () => {
  const queryClient = useQueryClient();
  const { toast } = useToast();

  const indexContentMutation = useMutation({
    mutationFn: ({ documentId, extractedText }: { documentId: string; extractedText: string }) =>
      advancedDocumentService.indexDocumentContent(documentId, extractedText),
    onSuccess: (_, { documentId }) => {
      // Invalider les caches de recherche
      queryClient.invalidateQueries({ queryKey: ADVANCED_QUERY_KEYS.semantic });
      
      toast('Contenu indexé pour recherche sémantique', 'success');
    },
    onError: (error) => {
      toast(`Erreur indexation: ${error.message}`, 'error');
    },
  });

  return {
    indexContent: indexContentMutation.mutate,
    isIndexing: indexContentMutation.isPending,
  };
};

// === EXPORT AVANCÉ ===

/**
 * Hook pour export de données documentaires
 */
export const useDocumentExport = () => {
  const { toast } = useToast();
  const [isExporting, setIsExporting] = useState(false);

  const exportDocuments = useCallback(async (
    filters: DocumentFilters = {},
    format: 'json' | 'csv' | 'excel' = 'json'
  ) => {
    setIsExporting(true);
    try {
      const blob = await advancedDocumentService.exportDocuments(filters, format);
      
      // Téléchargement automatique
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `documents_export_${new Date().toISOString().split('T')[0]}.${format}`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
      
      toast(`Export ${format.toUpperCase()} généré avec succès`, 'success');
    } catch (error) {
      toast(`Erreur export: ${error}`, 'error');
    } finally {
      setIsExporting(false);
    }
  }, [toast]);

  return {
    exportDocuments,
    isExporting,
  };
};

// === HOOK COMPOSÉ PRINCIPAL ===

/**
 * Hook principal combinant toutes les fonctionnalités avancées
 */
export const useAdvancedDocuments = (options?: {
  enableAnalytics?: boolean;
  enableRealTime?: boolean;
  enableSemanticSearch?: boolean;
}) => {
  const store = useDocumentStore();
  
  // Conditionally enable features
  const analytics = useDocumentAnalytics(30);
  const dashboard = options?.enableRealTime ? useRealTimeDashboard() : null;
  const semanticSearch = options?.enableSemanticSearch ? useSemanticSearch({ autoSearch: false }) : null;
  const upload = useAdvancedUpload();
  const storageStats = useStorageStats();
  const exportFeature = useDocumentExport();
  const indexing = useDocumentIndexing();

  // Status global
  const globalStatus = useMemo(() => {
    const isLoading = analytics.isLoading || (dashboard?.isLoading || false);
    const hasErrors = analytics.error || dashboard?.error;
    
    return {
      isLoading,
      hasErrors: !!hasErrors,
      isHealthy: !hasErrors && (analytics.computedMetrics?.overallHealth || 0) > 70,
    };
  }, [analytics, dashboard]);

  return {
    // Store Zustand
    store,
    
    // Analytics
    analytics: options?.enableAnalytics ? analytics : null,
    dashboard,
    
    // Search
    semanticSearch,
    
    // Operations
    upload,
    exportFeature,
    indexing,
    
    // Stats
    storageStats,
    
    // Global state
    globalStatus,
  };
};