/**
 * Store Zustand pour Documents - M&A Intelligence Platform
 * Sprint 3 - État local UI
 */

import { create } from 'zustand';
import { devtools, persist } from 'zustand/middleware';
import { 
  DocumentsState, 
  ViewMode, 
  SortField, 
  SortDirection,
  DocumentFilters,
  BreadcrumbItem,
  TreeNode,
  UploadConfig,
  UploadFile
} from '../types';
import { DEFAULT_UPLOAD_CONFIG } from '../services/documentService';

interface DocumentStoreState extends DocumentsState {
  // Actions pour navigation
  setCurrentFolder: (folderId: string | null) => void;
  setBreadcrumbs: (breadcrumbs: BreadcrumbItem[]) => void;
  navigateToFolder: (folderId: string, folderName: string, folderPath: string) => void;
  navigateBack: () => void;
  
  // Actions pour vue et tri
  setViewMode: (mode: ViewMode) => void;
  setSorting: (field: SortField, direction?: SortDirection) => void;
  toggleSortDirection: () => void;
  
  // Actions pour sélection
  selectDocument: (documentId: string) => void;
  selectMultipleDocuments: (documentIds: string[]) => void;
  unselectDocument: (documentId: string) => void;
  clearSelection: () => void;
  toggleDocumentSelection: (documentId: string) => void;
  selectAll: (documentIds: string[]) => void;
  
  // Actions pour filtres et recherche
  setFilters: (filters: Partial<DocumentFilters>) => void;
  clearFilters: () => void;
  setSearchQuery: (query: string) => void;
  addTagFilter: (tag: string) => void;
  removeTagFilter: (tag: string) => void;
  toggleTypeFilter: (type: string) => void;
  
  // Actions pour upload
  setUploadConfig: (config: Partial<UploadConfig>) => void;
  openUploadModal: () => void;
  closeUploadModal: () => void;
  addToUploadQueue: (files: UploadFile[]) => void;
  removeFromUploadQueue: (fileId: string) => void;
  clearUploadQueue: () => void;
  updateUploadProgress: (fileId: string, progress: number) => void;
  
  // Actions pour preview
  openPreview: (documentId: string) => void;
  closePreview: () => void;
  
  // Actions pour sidebar
  toggleSidebar: () => void;
  setSidebarExpanded: (expanded: boolean) => void;
  
  // Actions pour arbre de dossiers
  setFolderTree: (tree: TreeNode[]) => void;
  expandTreeNode: (nodeId: string) => void;
  collapseTreeNode: (nodeId: string) => void;
  toggleTreeNode: (nodeId: string) => void;
  
  // Actions pour états de chargement
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  setUploading: (uploading: boolean) => void;
}

export const useDocumentStore = create<DocumentStoreState>()(
  devtools(
    persist(
      (set, get) => ({
        // État initial
        currentFolderId: null,
        viewMode: 'grid',
        selectedDocuments: [],
        
        filters: {
          query: '',
          type: [],
          category: [],
          confidentiality: [],
          tags: [],
        },
        sortField: 'updatedAt',
        sortDirection: 'desc',
        searchQuery: '',
        
        uploadQueue: [],
        uploadConfig: DEFAULT_UPLOAD_CONFIG,
        
        isUploadModalOpen: false,
        isPreviewModalOpen: false,
        previewDocumentId: null,
        sidebarExpanded: true,
        
        breadcrumbs: [{ id: 'root', name: 'Documents', path: '/', type: 'root' }],
        folderTree: [],
        
        isLoading: false,
        isUploading: false,
        error: null,

        // === NAVIGATION ===
        
        setCurrentFolder: (folderId) => set({ currentFolderId: folderId }),
        
        setBreadcrumbs: (breadcrumbs) => set({ breadcrumbs }),
        
        navigateToFolder: (folderId, folderName, folderPath) => {
          const state = get();
          const pathParts = folderPath.split('/').filter(Boolean);
          
          const newBreadcrumbs: BreadcrumbItem[] = [
            { id: 'root', name: 'Documents', path: '/', type: 'root' }
          ];
          
          // Construire les breadcrumbs basés sur le path
          let currentPath = '';
          pathParts.forEach((part, index) => {
            currentPath += `/${part}`;
            newBreadcrumbs.push({
              id: index === pathParts.length - 1 ? folderId : `folder-${index}`,
              name: index === pathParts.length - 1 ? folderName : part,
              path: currentPath,
              type: 'folder',
            });
          });
          
          set({ 
            currentFolderId: folderId,
            breadcrumbs: newBreadcrumbs,
            selectedDocuments: [], // Clear selection lors navigation
          });
        },
        
        navigateBack: () => {
          const state = get();
          if (state.breadcrumbs.length > 1) {
            const newBreadcrumbs = state.breadcrumbs.slice(0, -1);
            const previousFolder = newBreadcrumbs[newBreadcrumbs.length - 1];
            
            set({
              currentFolderId: previousFolder.type === 'root' ? null : previousFolder.id,
              breadcrumbs: newBreadcrumbs,
              selectedDocuments: [],
            });
          }
        },

        // === VUE ET TRI ===
        
        setViewMode: (mode) => set({ viewMode: mode }),
        
        setSorting: (field, direction) => set({ 
          sortField: field,
          sortDirection: direction || 'asc',
        }),
        
        toggleSortDirection: () => {
          const state = get();
          set({ 
            sortDirection: state.sortDirection === 'asc' ? 'desc' : 'asc' 
          });
        },

        // === SÉLECTION ===
        
        selectDocument: (documentId) => {
          const state = get();
          if (!state.selectedDocuments.includes(documentId)) {
            set({ 
              selectedDocuments: [...state.selectedDocuments, documentId] 
            });
          }
        },
        
        selectMultipleDocuments: (documentIds) => {
          const state = get();
          const newSelection = Array.from(new Set([
            ...state.selectedDocuments,
            ...documentIds
          ]));
          set({ selectedDocuments: newSelection });
        },
        
        unselectDocument: (documentId) => {
          const state = get();
          set({ 
            selectedDocuments: state.selectedDocuments.filter(id => id !== documentId) 
          });
        },
        
        clearSelection: () => set({ selectedDocuments: [] }),
        
        toggleDocumentSelection: (documentId) => {
          const state = get();
          if (state.selectedDocuments.includes(documentId)) {
            state.unselectDocument(documentId);
          } else {
            state.selectDocument(documentId);
          }
        },
        
        selectAll: (documentIds) => set({ selectedDocuments: documentIds }),

        // === FILTRES ET RECHERCHE ===
        
        setFilters: (newFilters) => {
          const state = get();
          set({ 
            filters: { ...state.filters, ...newFilters },
            selectedDocuments: [], // Clear selection lors filtrage
          });
        },
        
        clearFilters: () => set({ 
          filters: {
            query: '',
            type: [],
            category: [],
            confidentiality: [],
            tags: [],
          },
          searchQuery: '',
          selectedDocuments: [],
        }),
        
        setSearchQuery: (query) => set({ 
          searchQuery: query,
          filters: { ...get().filters, query },
          selectedDocuments: [],
        }),
        
        addTagFilter: (tag) => {
          const state = get();
          const currentTags = state.filters.tags || [];
          if (!currentTags.includes(tag)) {
            set({
              filters: {
                ...state.filters,
                tags: [...currentTags, tag],
              },
            });
          }
        },
        
        removeTagFilter: (tag) => {
          const state = get();
          const currentTags = state.filters.tags || [];
          set({
            filters: {
              ...state.filters,
              tags: currentTags.filter(t => t !== tag),
            },
          });
        },
        
        toggleTypeFilter: (type) => {
          const state = get();
          const currentTypes = state.filters.type || [];
          const newTypes = currentTypes.includes(type as any)
            ? currentTypes.filter(t => t !== type)
            : [...currentTypes, type as any];
          
          set({
            filters: {
              ...state.filters,
              type: newTypes,
            },
          });
        },

        // === UPLOAD ===
        
        setUploadConfig: (config) => {
          const state = get();
          set({ 
            uploadConfig: { ...state.uploadConfig, ...config } 
          });
        },
        
        openUploadModal: () => set({ isUploadModalOpen: true }),
        
        closeUploadModal: () => set({ isUploadModalOpen: false }),
        
        addToUploadQueue: (files) => {
          const state = get();
          set({ 
            uploadQueue: [...state.uploadQueue, ...files] 
          });
        },
        
        removeFromUploadQueue: (fileId) => {
          const state = get();
          set({ 
            uploadQueue: state.uploadQueue.filter(file => file.id !== fileId) 
          });
        },
        
        clearUploadQueue: () => set({ uploadQueue: [] }),
        
        updateUploadProgress: (fileId, progress) => {
          const state = get();
          set({
            uploadQueue: state.uploadQueue.map(file => 
              file.id === fileId 
                ? { ...file, progress }
                : file
            ),
          });
        },

        // === PREVIEW ===
        
        openPreview: (documentId) => set({ 
          isPreviewModalOpen: true,
          previewDocumentId: documentId,
        }),
        
        closePreview: () => set({ 
          isPreviewModalOpen: false,
          previewDocumentId: null,
        }),

        // === SIDEBAR ===
        
        toggleSidebar: () => {
          const state = get();
          set({ sidebarExpanded: !state.sidebarExpanded });
        },
        
        setSidebarExpanded: (expanded) => set({ sidebarExpanded: expanded }),

        // === ARBRE DE DOSSIERS ===
        
        setFolderTree: (tree) => set({ folderTree: tree }),
        
        expandTreeNode: (nodeId) => {
          const state = get();
          const updateNode = (nodes: TreeNode[]): TreeNode[] => {
            return nodes.map(node => {
              if (node.id === nodeId) {
                return { ...node, isExpanded: true };
              }
              if (node.children.length > 0) {
                return { ...node, children: updateNode(node.children) };
              }
              return node;
            });
          };
          
          set({ folderTree: updateNode(state.folderTree) });
        },
        
        collapseTreeNode: (nodeId) => {
          const state = get();
          const updateNode = (nodes: TreeNode[]): TreeNode[] => {
            return nodes.map(node => {
              if (node.id === nodeId) {
                return { ...node, isExpanded: false };
              }
              if (node.children.length > 0) {
                return { ...node, children: updateNode(node.children) };
              }
              return node;
            });
          };
          
          set({ folderTree: updateNode(state.folderTree) });
        },
        
        toggleTreeNode: (nodeId) => {
          const state = get();
          // Trouver le noeud et toggle son état
          const findAndToggle = (nodes: TreeNode[]): boolean => {
            for (const node of nodes) {
              if (node.id === nodeId) {
                if (node.isExpanded) {
                  state.collapseTreeNode(nodeId);
                } else {
                  state.expandTreeNode(nodeId);
                }
                return true;
              }
              if (node.children.length > 0 && findAndToggle(node.children)) {
                return true;
              }
            }
            return false;
          };
          
          findAndToggle(state.folderTree);
        },

        // === ÉTATS DE CHARGEMENT ===
        
        setLoading: (loading) => set({ isLoading: loading }),
        
        setError: (error) => set({ error }),
        
        setUploading: (uploading) => set({ isUploading: uploading }),
      }),
      {
        name: 'ma-intelligence-documents',
        partialize: (state) => ({
          // Persister seulement certains éléments
          viewMode: state.viewMode,
          sortField: state.sortField,
          sortDirection: state.sortDirection,
          uploadConfig: state.uploadConfig,
          sidebarExpanded: state.sidebarExpanded,
          // Ne pas persister: currentFolderId, selectedDocuments, filters, etc.
        }),
      }
    ),
    { name: 'DocumentStore' }
  )
);

// Sélecteurs pour éviter re-renders inutiles
export const useDocumentSelection = () => {
  return useDocumentStore(state => ({
    selectedDocuments: state.selectedDocuments,
    selectDocument: state.selectDocument,
    unselectDocument: state.unselectDocument,
    clearSelection: state.clearSelection,
    toggleDocumentSelection: state.toggleDocumentSelection,
    selectAll: state.selectAll,
  }));
};

export const useDocumentFilters = () => {
  return useDocumentStore(state => ({
    filters: state.filters,
    searchQuery: state.searchQuery,
    setFilters: state.setFilters,
    clearFilters: state.clearFilters,
    setSearchQuery: state.setSearchQuery,
    addTagFilter: state.addTagFilter,
    removeTagFilter: state.removeTagFilter,
    toggleTypeFilter: state.toggleTypeFilter,
  }));
};

export const useDocumentNavigation = () => {
  return useDocumentStore(state => ({
    currentFolderId: state.currentFolderId,
    breadcrumbs: state.breadcrumbs,
    setCurrentFolder: state.setCurrentFolder,
    setBreadcrumbs: state.setBreadcrumbs,
    navigateToFolder: state.navigateToFolder,
    navigateBack: state.navigateBack,
  }));
};

export const useDocumentUploadState = () => {
  return useDocumentStore(state => ({
    uploadQueue: state.uploadQueue,
    uploadConfig: state.uploadConfig,
    isUploadModalOpen: state.isUploadModalOpen,
    isUploading: state.isUploading,
    openUploadModal: state.openUploadModal,
    closeUploadModal: state.closeUploadModal,
    addToUploadQueue: state.addToUploadQueue,
    removeFromUploadQueue: state.removeFromUploadQueue,
    clearUploadQueue: state.clearUploadQueue,
    updateUploadProgress: state.updateUploadProgress,
  }));
};