/**
 * Tests Advanced Document Store - M&A Intelligence Platform
 * Sprint 3 - Tests state management Zustand
 */

import { renderHook, act } from '@testing-library/react';
import { 
  useAdvancedDocumentStore,
  useDocumentSelection,
  useDocumentSearch,
  useDocumentFilters,
  useDocumentAnalytics,
  useDocumentPreview,
  useDocumentOperations
} from '../stores/advancedDocumentStore';

// Mock document data
const mockDocument = {
  document_id: '1',
  title: 'Test Document',
  filename: 'test.pdf',
  document_type: 'financial' as const,
  access_level: 'public' as const,
  file_size: 1024000,
  view_count: 5,
  download_count: 2,
  created_at: '2024-01-15T10:00:00Z',
  updated_at: '2024-01-15T10:00:00Z',
  accessed_at: '2024-01-15T10:00:00Z',
  owner_id: 'user1',
  tags: ['important', 'financial'],
  version: 1,
  mime_type: 'application/pdf',
  file_extension: '.pdf',
  is_latest_version: true,
  canEdit: true,
  canDelete: true,
  description: 'Test document',
};

const mockDocuments = [
  mockDocument,
  {
    ...mockDocument,
    document_id: '2',
    title: 'Second Document',
    document_type: 'legal' as const,
    access_level: 'confidential' as const,
  },
  {
    ...mockDocument,
    document_id: '3',
    title: 'Third Document',
    document_type: 'technical' as const,
    access_level: 'internal' as const,
  },
];

describe('Advanced Document Store', () => {
  beforeEach(() => {
    // Reset store state
    useAdvancedDocumentStore.getState().reset();
  });

  describe('Basic Store Operations', () => {
    test('should initialize with default state', () => {
      const { result } = renderHook(() => useAdvancedDocumentStore());
      
      expect(result.current.documents).toEqual([]);
      expect(result.current.selectedDocuments).toEqual(new Set());
      expect(result.current.currentDocument).toBeNull();
      expect(result.current.isLoading).toBe(false);
      expect(result.current.error).toBeNull();
    });

    test('should set documents', () => {
      const { result } = renderHook(() => useAdvancedDocumentStore());
      
      act(() => {
        result.current.setDocuments(mockDocuments);
      });
      
      expect(result.current.documents).toEqual(mockDocuments);
      expect(result.current.documents).toHaveLength(3);
    });

    test('should add document', () => {
      const { result } = renderHook(() => useAdvancedDocumentStore());
      
      act(() => {
        result.current.addDocument(mockDocument);
      });
      
      expect(result.current.documents).toContain(mockDocument);
      expect(result.current.documents).toHaveLength(1);
    });

    test('should update existing document', () => {
      const { result } = renderHook(() => useAdvancedDocumentStore());
      
      act(() => {
        result.current.setDocuments([mockDocument]);
        result.current.updateDocument('1', { title: 'Updated Title' });
      });
      
      expect(result.current.documents[0].title).toBe('Updated Title');
    });

    test('should remove document', () => {
      const { result } = renderHook(() => useAdvancedDocumentStore());
      
      act(() => {
        result.current.setDocuments(mockDocuments);
        result.current.removeDocument('2');
      });
      
      expect(result.current.documents).toHaveLength(2);
      expect(result.current.documents.find(d => d.document_id === '2')).toBeUndefined();
    });
  });

  describe('Document Selection', () => {
    test('should select single document', () => {
      const { result } = renderHook(() => useDocumentSelection());
      
      act(() => {
        useAdvancedDocumentStore.getState().setDocuments(mockDocuments);
        result.current.selectDocument('1', 'single');
      });
      
      expect(result.current.selectedDocuments.has('1')).toBe(true);
      expect(result.current.selectedDocuments.size).toBe(1);
    });

    test('should handle multi-selection', () => {
      const { result } = renderHook(() => useDocumentSelection());
      
      act(() => {
        useAdvancedDocumentStore.getState().setDocuments(mockDocuments);
        result.current.selectDocument('1', 'multi');
        result.current.selectDocument('2', 'multi');
      });
      
      expect(result.current.selectedDocuments.has('1')).toBe(true);
      expect(result.current.selectedDocuments.has('2')).toBe(true);
      expect(result.current.selectedDocuments.size).toBe(2);
    });

    test('should handle range selection', () => {
      const { result } = renderHook(() => useDocumentSelection());
      
      act(() => {
        useAdvancedDocumentStore.getState().setDocuments(mockDocuments);
        result.current.selectDocument('1', 'single');
        result.current.selectDocument('3', 'range');
      });
      
      // Should select documents from index 0 to 2 (inclusive)
      expect(result.current.selectedDocuments.has('1')).toBe(true);
      expect(result.current.selectedDocuments.has('2')).toBe(true);
      expect(result.current.selectedDocuments.has('3')).toBe(true);
      expect(result.current.selectedDocuments.size).toBe(3);
    });

    test('should select all documents', () => {
      const { result } = renderHook(() => useDocumentSelection());
      
      act(() => {
        useAdvancedDocumentStore.getState().setDocuments(mockDocuments);
        result.current.selectAll();
      });
      
      expect(result.current.selectedDocuments.size).toBe(3);
      mockDocuments.forEach(doc => {
        expect(result.current.selectedDocuments.has(doc.document_id)).toBe(true);
      });
    });

    test('should clear selection', () => {
      const { result } = renderHook(() => useDocumentSelection());
      
      act(() => {
        useAdvancedDocumentStore.getState().setDocuments(mockDocuments);
        result.current.selectAll();
        result.current.clearSelection();
      });
      
      expect(result.current.selectedDocuments.size).toBe(0);
    });

    test('should invert selection', () => {
      const { result } = renderHook(() => useDocumentSelection());
      
      act(() => {
        useAdvancedDocumentStore.getState().setDocuments(mockDocuments);
        result.current.selectDocument('1', 'single');
        result.current.invertSelection();
      });
      
      expect(result.current.selectedDocuments.has('1')).toBe(false);
      expect(result.current.selectedDocuments.has('2')).toBe(true);
      expect(result.current.selectedDocuments.has('3')).toBe(true);
      expect(result.current.selectedDocuments.size).toBe(2);
    });
  });

  describe('Search Functionality', () => {
    test('should set search query', () => {
      const { result } = renderHook(() => useDocumentSearch());
      
      act(() => {
        result.current.setSearchQuery('test query');
      });
      
      expect(result.current.searchQuery).toBe('test query');
    });

    test('should clear search results when query is empty', () => {
      const { result } = renderHook(() => useDocumentSearch());
      
      act(() => {
        result.current.setSearchQuery('test');
        result.current.setSearchQuery('');
      });
      
      expect(result.current.searchQuery).toBe('');
      expect(result.current.semanticResults).toEqual([]);
    });

    test('should enable/disable semantic search', () => {
      const { result } = renderHook(() => useDocumentSearch());
      
      act(() => {
        result.current.enableSemanticSearch();
      });
      
      expect(result.current.semanticSearchEnabled).toBe(true);
      
      act(() => {
        result.current.disableSemanticSearch();
      });
      
      expect(result.current.semanticSearchEnabled).toBe(false);
      expect(result.current.semanticResults).toEqual([]);
    });

    test('should clear search', () => {
      const { result } = renderHook(() => useDocumentSearch());
      
      act(() => {
        result.current.setSearchQuery('test');
        result.current.enableSemanticSearch();
        result.current.clearSearch();
      });
      
      expect(result.current.searchQuery).toBe('');
      expect(result.current.semanticResults).toEqual([]);
    });
  });

  describe('Filters', () => {
    test('should set filters', () => {
      const { result } = renderHook(() => useDocumentFilters());
      
      const filters = {
        document_type: ['financial'],
        access_level: ['public'],
      };
      
      act(() => {
        result.current.setFilters(filters);
      });
      
      expect(result.current.filters).toEqual(filters);
      expect(result.current.activeFilters.has('document_type')).toBe(true);
      expect(result.current.activeFilters.has('access_level')).toBe(true);
    });

    test('should add and remove individual filters', () => {
      const { result } = renderHook(() => useDocumentFilters());
      
      act(() => {
        result.current.addFilter('document_type', 'financial');
      });
      
      expect(result.current.filters.document_type).toBe('financial');
      expect(result.current.activeFilters.has('document_type')).toBe(true);
      
      act(() => {
        result.current.removeFilter('document_type');
      });
      
      expect(result.current.filters.document_type).toBeUndefined();
      expect(result.current.activeFilters.has('document_type')).toBe(false);
    });

    test('should clear all filters', () => {
      const { result } = renderHook(() => useDocumentFilters());
      
      act(() => {
        result.current.addFilter('document_type', 'financial');
        result.current.addFilter('access_level', 'public');
        result.current.clearAllFilters();
      });
      
      expect(result.current.filters).toEqual({});
      expect(result.current.activeFilters.size).toBe(0);
    });
  });

  describe('Preview Functionality', () => {
    test('should open and close preview', () => {
      const { result } = renderHook(() => useDocumentPreview());
      
      act(() => {
        useAdvancedDocumentStore.getState().setDocuments([mockDocument]);
        result.current.openPreview('1');
      });
      
      expect(result.current.isPreviewOpen).toBe(true);
      expect(result.current.previewDocumentId).toBe('1');
      expect(result.current.currentDocument).toEqual(mockDocument);
      
      act(() => {
        result.current.closePreview();
      });
      
      expect(result.current.isPreviewOpen).toBe(false);
      expect(result.current.previewDocumentId).toBeNull();
    });

    test('should start and stop editing', () => {
      const { result } = renderHook(() => useDocumentPreview());
      
      act(() => {
        result.current.startEditing(mockDocument);
      });
      
      expect(result.current.editingDocument).toEqual(mockDocument);
      
      act(() => {
        result.current.stopEditing();
      });
      
      expect(result.current.editingDocument).toBeNull();
    });
  });

  describe('Analytics', () => {
    test('should set analytics data', () => {
      const { result } = renderHook(() => useDocumentAnalytics());
      
      const analyticsData = {
        totalDocuments: 100,
        totalSize: 1024000,
        documentsByType: { financial: 50, legal: 30, technical: 20 },
        uploadTrend: [],
        accessPatterns: [],
        popularDocuments: [],
      };
      
      act(() => {
        result.current.setAnalyticsData(analyticsData);
      });
      
      expect(result.current.analyticsData).toEqual(analyticsData);
    });

    test('should refresh analytics', () => {
      const { result } = renderHook(() => useDocumentAnalytics());
      
      act(() => {
        result.current.refreshAnalytics();
      });
      
      expect(result.current.isAnalyticsLoading).toBe(true);
    });
  });

  describe('Navigation', () => {
    test('should navigate to path', () => {
      const { result } = renderHook(() => useAdvancedDocumentStore());
      
      act(() => {
        result.current.navigateTo('/documents/financial');
      });
      
      expect(result.current.currentPath).toBe('/documents/financial');
      expect(result.current.navigationHistory).toContain('/documents/financial');
      expect(result.current.selectedDocuments.size).toBe(0); // Should clear selection
    });

    test('should navigate back', () => {
      const { result } = renderHook(() => useAdvancedDocumentStore());
      
      act(() => {
        result.current.navigateTo('/documents/financial');
        result.current.navigateTo('/documents/legal');
        result.current.navigateBack();
      });
      
      expect(result.current.currentPath).toBe('/documents/financial');
      expect(result.current.navigationHistory).toHaveLength(2); // ['/', '/documents/financial']
    });

    test('should not navigate back from root', () => {
      const { result } = renderHook(() => useAdvancedDocumentStore());
      
      const initialPath = result.current.currentPath;
      const initialHistoryLength = result.current.navigationHistory.length;
      
      act(() => {
        result.current.navigateBack();
      });
      
      expect(result.current.currentPath).toBe(initialPath);
      expect(result.current.navigationHistory).toHaveLength(initialHistoryLength);
    });
  });

  describe('Operations', () => {
    test('should select by document type', () => {
      const { result } = renderHook(() => useDocumentOperations());
      
      act(() => {
        useAdvancedDocumentStore.getState().setDocuments(mockDocuments);
        result.current.selectByType('financial');
      });
      
      const selection = useAdvancedDocumentStore.getState().selectedDocuments;
      expect(selection.size).toBe(1);
      expect(selection.has('1')).toBe(true);
    });

    test('should start batch operations', () => {
      const { result } = renderHook(() => useDocumentOperations());
      
      const operationId = result.current.startBatchDelete();
      
      expect(operationId).toMatch(/^delete-\d+$/);
      
      const batchOperations = useAdvancedDocumentStore.getState().batchOperations;
      expect(batchOperations.has(operationId)).toBe(true);
    });

    test('should show only recent documents', () => {
      const { result } = renderHook(() => useDocumentOperations());
      
      act(() => {
        result.current.showOnlyRecent();
      });
      
      const filters = useAdvancedDocumentStore.getState().filters;
      expect(filters.dateRange).toBeDefined();
      expect(filters.dateRange?.field).toBe('created_at');
    });
  });

  describe('Performance Optimization', () => {
    test('should optimize performance when needed', () => {
      const { result } = renderHook(() => useAdvancedDocumentStore());
      
      // Add many documents and selections
      const manyDocuments = Array.from({ length: 1500 }, (_, i) => ({
        ...mockDocument,
        document_id: `doc-${i}`,
      }));
      
      act(() => {
        result.current.setDocuments(manyDocuments);
        
        // Add many selections
        for (let i = 0; i < 150; i++) {
          result.current.selectDocument(`doc-${i}`, 'multi');
        }
        
        // Add navigation history
        for (let i = 0; i < 25; i++) {
          result.current.navigateTo(`/path-${i}`);
        }
        
        result.current.optimizePerformance();
      });
      
      // Should limit documents to 500
      expect(result.current.documents.length).toBe(500);
      
      // Should limit navigation history to 10
      expect(result.current.navigationHistory.length).toBe(10);
      
      // Should clear large selections
      expect(result.current.selectedDocuments.size).toBe(0);
    });
  });

  describe('State Persistence', () => {
    test('should export and import state', () => {
      const { result } = renderHook(() => useAdvancedDocumentStore());
      
      // Set some state
      act(() => {
        result.current.updatePreferences({ documentsPerPage: 100 });
        result.current.setFilters({ document_type: ['financial'] });
        result.current.setViewMode('list');
      });
      
      // Export state
      const exportedState = result.current.exportState();
      expect(exportedState).toBeDefined();
      
      // Reset state
      act(() => {
        result.current.reset();
      });
      
      expect(result.current.preferences.documentsPerPage).toBe(50); // Default value
      
      // Import state
      act(() => {
        result.current.importState(exportedState);
      });
      
      expect(result.current.preferences.documentsPerPage).toBe(100);
      expect(result.current.filters.document_type).toEqual(['financial']);
      expect(result.current.viewMode).toBe('list');
    });

    test('should handle invalid import state gracefully', () => {
      const { result } = renderHook(() => useAdvancedDocumentStore());
      
      // Mock console.error to verify error handling
      const consoleSpy = jest.spyOn(console, 'error').mockImplementation();
      
      act(() => {
        result.current.importState('invalid json');
      });
      
      expect(consoleSpy).toHaveBeenCalledWith('Erreur import état:', expect.any(Error));
      
      consoleSpy.mockRestore();
    });
  });

  describe('Selectors', () => {
    test('should provide selection information', () => {
      const { result } = renderHook(() => useDocumentSelection());
      
      act(() => {
        useAdvancedDocumentStore.getState().setDocuments(mockDocuments);
        result.current.selectDocument('1', 'single');
      });
      
      expect(result.current.selectedCount).toBe(1);
      expect(result.current.hasSelection).toBe(true);
      expect(result.current.isAllSelected).toBe(false);
    });

    test('should detect when all documents are selected', () => {
      const { result } = renderHook(() => useDocumentSelection());
      
      act(() => {
        useAdvancedDocumentStore.getState().setDocuments(mockDocuments);
        result.current.selectAll();
      });
      
      expect(result.current.isAllSelected).toBe(true);
    });
  });
});

// Performance tests
describe('Store Performance', () => {
  test('should handle large datasets efficiently', () => {
    const largeDataset = Array.from({ length: 10000 }, (_, i) => ({
      ...mockDocument,
      document_id: `doc-${i}`,
      title: `Document ${i}`,
    }));

    const start = performance.now();
    
    const { result } = renderHook(() => useAdvancedDocumentStore());
    
    act(() => {
      result.current.setDocuments(largeDataset);
    });
    
    const end = performance.now();
    const duration = end - start;
    
    // Should complete within reasonable time (< 100ms)
    expect(duration).toBeLessThan(100);
    expect(result.current.documents.length).toBe(10000);
  });

  test('should handle rapid state updates', () => {
    const { result } = renderHook(() => useAdvancedDocumentStore());
    
    const start = performance.now();
    
    act(() => {
      // Rapid updates
      for (let i = 0; i < 1000; i++) {
        result.current.updatePreferences({ documentsPerPage: i });
      }
    });
    
    const end = performance.now();
    const duration = end - start;
    
    // Should handle rapid updates efficiently
    expect(duration).toBeLessThan(50);
    expect(result.current.preferences.documentsPerPage).toBe(999);
  });
});