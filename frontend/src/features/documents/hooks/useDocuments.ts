/**
 * Hooks pour la gestion des Documents - M&A Intelligence Platform
 * Sprint 3 - Hooks centralisés avec React Query
 */

import { useQuery, useMutation, useQueryClient, useInfiniteQuery } from '@tanstack/react-query';
import { useCallback, useMemo, useState } from 'react';
import { 
  Document, 
  Folder, 
  DocumentFilters, 
  DocumentsResponse,
  UploadFile,
  UploadStatus,
  SearchFacets,
  DocumentStats
} from '../types';
import { documentService, DEFAULT_UPLOAD_CONFIG } from '../services/documentService';
import { useToast } from '../../../hooks';

// Query Keys pour React Query
export const DOCUMENTS_QUERY_KEYS = {
  all: ['documents'] as const,
  lists: () => [...DOCUMENTS_QUERY_KEYS.all, 'list'] as const,
  list: (folderId?: string, filters?: DocumentFilters) => 
    [...DOCUMENTS_QUERY_KEYS.lists(), { folderId, filters }] as const,
  infinite: (folderId?: string, filters?: DocumentFilters) =>
    [...DOCUMENTS_QUERY_KEYS.all, 'infinite', { folderId, filters }] as const,
  details: () => [...DOCUMENTS_QUERY_KEYS.all, 'detail'] as const,
  detail: (id: string) => [...DOCUMENTS_QUERY_KEYS.details(), id] as const,
  
  folders: ['folders'] as const,
  folderTree: () => [...DOCUMENTS_QUERY_KEYS.folders, 'tree'] as const,
  folderList: (parentId?: string) => [...DOCUMENTS_QUERY_KEYS.folders, 'list', { parentId }] as const,
  
  search: (query: string, filters?: DocumentFilters) =>
    [...DOCUMENTS_QUERY_KEYS.all, 'search', { query, filters }] as const,
  facets: (query?: string, folderId?: string) =>
    [...DOCUMENTS_QUERY_KEYS.all, 'facets', { query, folderId }] as const,
  stats: (folderId?: string) =>
    [...DOCUMENTS_QUERY_KEYS.all, 'stats', { folderId }] as const,
} as const;

// === DOCUMENTS HOOKS ===

export const useDocuments = (
  folderId?: string,
  filters?: DocumentFilters,
  options?: {
    enabled?: boolean;
    staleTime?: number;
  }
) => {
  const queryClient = useQueryClient();
  const { toast } = useToast();

  const {
    data: response,
    isLoading,
    error,
    refetch,
  } = useQuery({
    queryKey: DOCUMENTS_QUERY_KEYS.list(folderId, filters),
    queryFn: () => documentService.getDocuments(folderId, filters),
    staleTime: options?.staleTime || 2 * 60 * 1000, // 2 minutes
    enabled: options?.enabled !== false,
  });

  const documents = response?.documents || [];
  const total = response?.total || 0;
  const hasMore = response?.hasMore || false;
  const facets = response?.facets;

  // Refresh avec feedback utilisateur
  const refreshDocuments = useCallback(async () => {
    try {
      await refetch();
      toast('Documents actualisés', 'success');
    } catch (error) {
      toast('Erreur lors de l\'actualisation', 'error');
    }
  }, [refetch, toast]);

  // Invalidate cache quand nécessaire
  const invalidateDocuments = useCallback(() => {
    queryClient.invalidateQueries({ queryKey: DOCUMENTS_QUERY_KEYS.lists() });
  }, [queryClient]);

  return {
    documents,
    total,
    hasMore,
    facets,
    isLoading,
    error,
    refreshDocuments,
    invalidateDocuments,
  };
};

// Hook pour infinite scroll (grandes listes)
export const useInfiniteDocuments = (
  folderId?: string,
  filters?: DocumentFilters,
  pageSize: number = 50
) => {
  const {
    data,
    fetchNextPage,
    hasNextPage,
    isFetchingNextPage,
    isLoading,
    error,
  } = useInfiniteQuery({
    queryKey: DOCUMENTS_QUERY_KEYS.infinite(folderId, filters),
    queryFn: ({ pageParam = 1 }) => 
      documentService.getDocuments(folderId, filters, pageParam, pageSize),
    getNextPageParam: (lastPage, allPages) => {
      return lastPage.hasMore ? allPages.length + 1 : undefined;
    },
    initialPageParam: 1,
    staleTime: 2 * 60 * 1000,
  });

  // Flatten des pages pour usage simple
  const documents = useMemo(() => {
    return data?.pages.flatMap(page => page.documents) || [];
  }, [data]);

  const total = data?.pages[0]?.total || 0;

  return {
    documents,
    total,
    fetchNextPage,
    hasNextPage,
    isFetchingNextPage,
    isLoading,
    error,
  };
};

// Hook pour un document spécifique
export const useDocument = (documentId: string, options?: { enabled?: boolean }) => {
  const { toast } = useToast();

  const {
    data: document,
    isLoading,
    error,
    refetch,
  } = useQuery({
    queryKey: DOCUMENTS_QUERY_KEYS.detail(documentId),
    queryFn: () => documentService.getDocument(documentId),
    enabled: options?.enabled !== false && !!documentId,
    staleTime: 5 * 60 * 1000, // 5 minutes pour les détails
  });

  return {
    document,
    isLoading,
    error,
    refetch,
  };
};

// === FOLDERS HOOKS ===

export const useFolders = (parentId?: string) => {
  const {
    data: folders,
    isLoading,
    error,
    refetch,
  } = useQuery({
    queryKey: DOCUMENTS_QUERY_KEYS.folderList(parentId),
    queryFn: () => documentService.getFolders(parentId),
    staleTime: 5 * 60 * 1000,
  });

  return {
    folders: folders || [],
    isLoading,
    error,
    refetch,
  };
};

export const useFolderTree = () => {
  const {
    data: folderTree,
    isLoading,
    error,
    refetch,
  } = useQuery({
    queryKey: DOCUMENTS_QUERY_KEYS.folderTree(),
    queryFn: () => documentService.getFolderTree(),
    staleTime: 10 * 60 * 1000, // 10 minutes pour l'arbre
  });

  return {
    folderTree: folderTree || [],
    isLoading,
    error,
    refetch,
  };
};

// === UPLOAD HOOKS ===

export const useDocumentUpload = () => {
  const queryClient = useQueryClient();
  const { toast } = useToast();
  const [uploadQueue, setUploadQueue] = useState<UploadFile[]>([]);

  // Upload single file
  const uploadMutation = useMutation({
    mutationFn: ({ file, folderId }: { file: File; folderId?: string }) => {
      const uploadFile: UploadFile = {
        id: `${file.name}-${Date.now()}`,
        file,
        name: file.name,
        size: file.size,
        type: file.type,
        progress: 0,
        status: 'uploading',
        folderId,
      };

      // Ajouter à la queue
      setUploadQueue(prev => [...prev, uploadFile]);

      return documentService.uploadFile(file, folderId, (progress) => {
        setUploadQueue(prev => prev.map(item => 
          item.id === uploadFile.id 
            ? { ...item, progress }
            : item
        ));
      });
    },
    onSuccess: (document, variables) => {
      const fileName = variables.file.name;
      
      // Mettre à jour le statut
      setUploadQueue(prev => prev.map(item => 
        item.file.name === fileName 
          ? { ...item, status: 'completed' as UploadStatus, uploadedDocument: document }
          : item
      ));

      // Invalider les caches
      queryClient.invalidateQueries({ queryKey: DOCUMENTS_QUERY_KEYS.lists() });
      queryClient.invalidateQueries({ queryKey: DOCUMENTS_QUERY_KEYS.folders });
      
      toast(`Fichier "${fileName}" uploadé avec succès`, 'success');
    },
    onError: (error, variables) => {
      const fileName = variables.file.name;
      
      // Mettre à jour le statut d'erreur
      setUploadQueue(prev => prev.map(item => 
        item.file.name === fileName 
          ? { ...item, status: 'failed' as UploadStatus, error: error.message }
          : item
      ));

      toast(`Échec upload "${fileName}": ${error.message}`, 'error');
    },
  });

  // Upload multiple files
  const uploadMultiple = useCallback(async (files: File[], folderId?: string) => {
    const uploadPromises = files.map(file => 
      uploadMutation.mutateAsync({ file, folderId })
    );

    try {
      await Promise.all(uploadPromises);
      toast(`${files.length} fichiers uploadés avec succès`, 'success');
    } catch (error) {
      // Les erreurs individuelles sont déjà gérées
      console.error('Some uploads failed:', error);
    }
  }, [uploadMutation, toast]);

  // Retirer de la queue
  const removeFromQueue = useCallback((fileId: string) => {
    setUploadQueue(prev => prev.filter(item => item.id !== fileId));
  }, []);

  // Nettoyer la queue (fichiers complétés/échoués)
  const clearQueue = useCallback(() => {
    setUploadQueue(prev => prev.filter(item => 
      item.status === 'uploading' || item.status === 'pending'
    ));
  }, []);

  return {
    uploadFile: uploadMutation.mutate,
    uploadMultiple,
    uploadQueue,
    isUploading: uploadMutation.isPending,
    removeFromQueue,
    clearQueue,
  };
};

// === MUTATIONS HOOKS ===

export const useDocumentMutations = () => {
  const queryClient = useQueryClient();
  const { toast } = useToast();

  // Update document
  const updateMutation = useMutation({
    mutationFn: ({ id, updates }: { id: string; updates: Partial<Document> }) => 
      documentService.updateDocument(id, updates),
    onSuccess: (document) => {
      // Mettre à jour le cache
      queryClient.setQueryData(
        DOCUMENTS_QUERY_KEYS.detail(document.id),
        document
      );
      
      // Invalider les listes
      queryClient.invalidateQueries({ queryKey: DOCUMENTS_QUERY_KEYS.lists() });
      
      toast('Document mis à jour', 'success');
    },
    onError: (error) => {
      toast(`Erreur mise à jour: ${error.message}`, 'error');
    },
  });

  // Delete document
  const deleteMutation = useMutation({
    mutationFn: (id: string) => documentService.deleteDocument(id),
    onSuccess: (_, documentId) => {
      // Supprimer du cache
      queryClient.removeQueries({ queryKey: DOCUMENTS_QUERY_KEYS.detail(documentId) });
      
      // Invalider les listes
      queryClient.invalidateQueries({ queryKey: DOCUMENTS_QUERY_KEYS.lists() });
      queryClient.invalidateQueries({ queryKey: DOCUMENTS_QUERY_KEYS.folders });
      
      toast('Document supprimé', 'success');
    },
    onError: (error) => {
      toast(`Erreur suppression: ${error.message}`, 'error');
    },
  });

  // Move document
  const moveMutation = useMutation({
    mutationFn: ({ id, targetFolderId }: { id: string; targetFolderId: string }) => 
      documentService.moveDocument(id, targetFolderId),
    onSuccess: (document) => {
      // Mettre à jour le cache
      queryClient.setQueryData(
        DOCUMENTS_QUERY_KEYS.detail(document.id),
        document
      );
      
      // Invalider les listes
      queryClient.invalidateQueries({ queryKey: DOCUMENTS_QUERY_KEYS.lists() });
      
      toast('Document déplacé', 'success');
    },
    onError: (error) => {
      toast(`Erreur déplacement: ${error.message}`, 'error');
    },
  });

  // Copy document
  const copyMutation = useMutation({
    mutationFn: ({ id, targetFolderId, newName }: { 
      id: string; 
      targetFolderId: string; 
      newName?: string;
    }) => documentService.copyDocument(id, targetFolderId, newName),
    onSuccess: () => {
      // Invalider les listes
      queryClient.invalidateQueries({ queryKey: DOCUMENTS_QUERY_KEYS.lists() });
      
      toast('Document copié', 'success');
    },
    onError: (error) => {
      toast(`Erreur copie: ${error.message}`, 'error');
    },
  });

  return {
    updateDocument: updateMutation.mutate,
    deleteDocument: deleteMutation.mutate,
    moveDocument: moveMutation.mutate,
    copyDocument: copyMutation.mutate,
    isUpdating: updateMutation.isPending,
    isDeleting: deleteMutation.isPending,
    isMoving: moveMutation.isPending,
    isCopying: copyMutation.isPending,
  };
};

// === FOLDER MUTATIONS ===

export const useFolderMutations = () => {
  const queryClient = useQueryClient();
  const { toast } = useToast();

  // Create folder
  const createMutation = useMutation({
    mutationFn: ({ name, parentId }: { name: string; parentId?: string }) => 
      documentService.createFolder(name, parentId),
    onSuccess: () => {
      // Invalider les caches folders
      queryClient.invalidateQueries({ queryKey: DOCUMENTS_QUERY_KEYS.folders });
      
      toast('Dossier créé', 'success');
    },
    onError: (error) => {
      toast(`Erreur création: ${error.message}`, 'error');
    },
  });

  // Update folder
  const updateMutation = useMutation({
    mutationFn: ({ id, updates }: { id: string; updates: Partial<Folder> }) => 
      documentService.updateFolder(id, updates),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: DOCUMENTS_QUERY_KEYS.folders });
      
      toast('Dossier mis à jour', 'success');
    },
    onError: (error) => {
      toast(`Erreur mise à jour: ${error.message}`, 'error');
    },
  });

  // Delete folder
  const deleteMutation = useMutation({
    mutationFn: ({ id, recursive }: { id: string; recursive?: boolean }) => 
      documentService.deleteFolder(id, recursive),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: DOCUMENTS_QUERY_KEYS.folders });
      queryClient.invalidateQueries({ queryKey: DOCUMENTS_QUERY_KEYS.lists() });
      
      toast('Dossier supprimé', 'success');
    },
    onError: (error) => {
      toast(`Erreur suppression: ${error.message}`, 'error');
    },
  });

  return {
    createFolder: createMutation.mutate,
    updateFolder: updateMutation.mutate,
    deleteFolder: deleteMutation.mutate,
    isCreating: createMutation.isPending,
    isUpdating: updateMutation.isPending,
    isDeleting: deleteMutation.isPending,
  };
};

// === SEARCH & STATS HOOKS ===

export const useDocumentSearch = (
  query: string,
  filters?: DocumentFilters,
  options?: { enabled?: boolean }
) => {
  const {
    data: response,
    isLoading,
    error,
  } = useQuery({
    queryKey: DOCUMENTS_QUERY_KEYS.search(query, filters),
    queryFn: () => documentService.searchDocuments(query, filters),
    enabled: options?.enabled !== false && query.length > 0,
    staleTime: 1 * 60 * 1000, // 1 minute pour search
  });

  return {
    documents: response?.documents || [],
    total: response?.total || 0,
    facets: response?.facets,
    isLoading,
    error,
  };
};

export const useSearchFacets = (query?: string, folderId?: string) => {
  const {
    data: facets,
    isLoading,
    error,
  } = useQuery({
    queryKey: DOCUMENTS_QUERY_KEYS.facets(query, folderId),
    queryFn: () => documentService.getSearchFacets(query, folderId),
    staleTime: 5 * 60 * 1000,
  });

  return {
    facets,
    isLoading,
    error,
  };
};

export const useDocumentStats = (folderId?: string) => {
  const {
    data: stats,
    isLoading,
    error,
  } = useQuery({
    queryKey: DOCUMENTS_QUERY_KEYS.stats(folderId),
    queryFn: () => documentService.getDocumentStats(folderId),
    staleTime: 10 * 60 * 1000,
  });

  return {
    stats,
    isLoading,
    error,
  };
};