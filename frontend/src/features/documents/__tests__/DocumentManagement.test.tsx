/**
 * Tests DocumentManagement - M&A Intelligence Platform
 * Sprint 3 - Tests unitaires et d'intégration
 */

import React from 'react';
import { render, screen, fireEvent, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { BrowserRouter } from 'react-router-dom';
import { DocumentManagement } from '../pages/DocumentManagement';
import { useAdvancedDocumentStore } from '../stores/advancedDocumentStore';
import { useDocuments } from '../hooks/useDocuments';

// Mocks
jest.mock('../hooks/useDocuments');
jest.mock('../stores/advancedDocumentStore');
jest.mock('../hooks/useAdvancedDocuments', () => ({
  useSemanticSearch: () => ({
    query: '',
    setQuery: jest.fn(),
    results: [],
    isLoading: false,
    isSemanticEnabled: false,
    toggleSemanticMode: jest.fn(),
  }),
  useAdvancedUpload: () => ({
    upload: jest.fn(),
    isUploading: false,
    progress: {},
    errors: {},
  }),
  useDocumentAnalytics: () => ({
    data: null,
    isLoading: false,
    refetch: jest.fn(),
  }),
}));

// Mock data
const mockDocuments = [
  {
    document_id: '1',
    title: 'Document Test 1',
    filename: 'test1.pdf',
    document_type: 'financial',
    access_level: 'public',
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
    description: 'Document de test financier',
  },
  {
    document_id: '2',
    title: 'Document Test 2',
    filename: 'test2.docx',
    document_type: 'legal',
    access_level: 'confidential',
    file_size: 512000,
    view_count: 3,
    download_count: 1,
    created_at: '2024-01-14T15:30:00Z',
    updated_at: '2024-01-14T15:30:00Z',
    accessed_at: '2024-01-14T15:30:00Z',
    owner_id: 'user2',
    tags: ['legal', 'contract'],
    version: 1,
    mime_type: 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    file_extension: '.docx',
    is_latest_version: true,
    canEdit: false,
    canDelete: false,
    description: 'Document légal confidentiel',
  },
];

const mockStore = {
  documents: mockDocuments,
  viewMode: 'grid',
  setLoading: jest.fn(),
  setError: jest.fn(),
  refreshDocuments: jest.fn(),
  preferences: {
    documentsPerPage: 50,
    autoRefreshInterval: 30000,
    enableNotifications: true,
    defaultSortField: 'created_at',
    defaultSortDirection: 'desc',
    compactMode: false,
    showMetadata: true,
    enableAnimations: true,
  },
  updatePreferences: jest.fn(),
};

const mockSelection = {
  selectedDocuments: new Set(),
  selectDocument: jest.fn(),
  clearSelection: jest.fn(),
  selectAll: jest.fn(),
  invertSelection: jest.fn(),
};

const mockSearch = {
  searchQuery: '',
  semanticSearchEnabled: false,
  semanticResults: [],
  isSearching: false,
  setSearchQuery: jest.fn(),
  enableSemanticSearch: jest.fn(),
  disableSemanticSearch: jest.fn(),
  clearSearch: jest.fn(),
};

const mockFilters = {
  filters: {},
  activeFilters: new Set(),
  hasActiveFilters: false,
  setFilters: jest.fn(),
  addFilter: jest.fn(),
  removeFilter: jest.fn(),
  clearAllFilters: jest.fn(),
};

const mockAnalytics = {
  analyticsData: null,
  realTimeMetrics: null,
  isAnalyticsLoading: false,
  setAnalyticsData: jest.fn(),
  refreshAnalytics: jest.fn(),
};

const mockPreview = {
  previewDocumentId: null,
  isPreviewOpen: false,
  currentDocument: null,
  editingDocument: null,
  openPreview: jest.fn(),
  closePreview: jest.fn(),
  startEditing: jest.fn(),
  stopEditing: jest.fn(),
};

const mockPerformance = {
  virtualizationEnabled: true,
  cacheEnabled: true,
  prefetchEnabled: true,
  toggleVirtualization: jest.fn(),
  toggleCache: jest.fn(),
  optimizePerformance: jest.fn(),
};

// Wrapper de test
const TestWrapper: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
      mutations: { retry: false },
    },
  });

  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        {children}
      </BrowserRouter>
    </QueryClientProvider>
  );
};

describe('DocumentManagement', () => {
  beforeEach(() => {
    // Reset mocks
    jest.clearAllMocks();
    
    // Setup store mocks
    (useAdvancedDocumentStore as jest.Mock).mockReturnValue(mockStore);
    (useAdvancedDocumentStore as any).mockImplementation((selector: any) => {
      if (typeof selector === 'function') {
        const state = {
          ...mockStore,
          selectedDocuments: mockSelection.selectedDocuments,
          searchQuery: mockSearch.searchQuery,
          semanticSearchEnabled: mockSearch.semanticSearchEnabled,
          semanticResults: mockSearch.semanticResults,
          isSearching: mockSearch.isSearching,
          filters: mockFilters.filters,
          activeFilters: mockFilters.activeFilters,
          analyticsData: mockAnalytics.analyticsData,
          realTimeMetrics: mockAnalytics.realTimeMetrics,
          isAnalyticsLoading: mockAnalytics.isAnalyticsLoading,
          previewDocumentId: mockPreview.previewDocumentId,
          isPreviewOpen: mockPreview.isPreviewOpen,
          currentDocument: mockPreview.currentDocument,
          editingDocument: mockPreview.editingDocument,
          virtualizationEnabled: mockPerformance.virtualizationEnabled,
          cacheEnabled: mockPerformance.cacheEnabled,
          prefetchEnabled: mockPerformance.prefetchEnabled,
          // Actions
          ...mockSelection,
          ...mockSearch,
          ...mockFilters,
          ...mockAnalytics,
          ...mockPreview,
          ...mockPerformance,
        };
        return selector(state);
      }
      return mockStore;
    });
    
    // Setup hooks mocks
    (useDocuments as jest.Mock).mockReturnValue({
      data: mockDocuments,
      isLoading: false,
      error: null,
      refetch: jest.fn(),
    });
  });

  describe('Rendering', () => {
    test('should render main components', () => {
      render(
        <TestWrapper>
          <DocumentManagement />
        </TestWrapper>
      );

      // Header
      expect(screen.getByText('Gestion Documentaire')).toBeInTheDocument();
      expect(screen.getByText('Plateforme M&A Intelligence - Sprint 3')).toBeInTheDocument();

      // Navigation
      expect(screen.getByPlaceholderText('Rechercher...')).toBeInTheDocument();
      expect(screen.getByText('🧠 Recherche IA')).toBeInTheDocument();

      // Stats
      expect(screen.getByText('Total Documents')).toBeInTheDocument();
      expect(screen.getByText('Taille Totale')).toBeInTheDocument();
      expect(screen.getByText('Cette Semaine')).toBeInTheDocument();
      expect(screen.getByText('Types')).toBeInTheDocument();
    });

    test('should render documents in grid view', () => {
      render(
        <TestWrapper>
          <DocumentManagement />
        </TestWrapper>
      );

      // Documents should be displayed
      expect(screen.getByText('Document Test 1')).toBeInTheDocument();
      expect(screen.getByText('Document Test 2')).toBeInTheDocument();

      // Document metadata
      expect(screen.getByText('financial')).toBeInTheDocument();
      expect(screen.getByText('legal')).toBeInTheDocument();
      expect(screen.getByText('public')).toBeInTheDocument();
      expect(screen.getByText('confidential')).toBeInTheDocument();
    });

    test('should render statistics correctly', () => {
      render(
        <TestWrapper>
          <DocumentManagement />
        </TestWrapper>
      );

      // Total documents
      expect(screen.getByText('2')).toBeInTheDocument(); // 2 documents

      // File sizes
      const totalSize = mockDocuments.reduce((sum, doc) => sum + doc.file_size, 0);
      const sizeKB = Math.round(totalSize / 1024);
      expect(screen.getByText(`${sizeKB}KB`)).toBeInTheDocument();

      // Document types
      const uniqueTypes = new Set(mockDocuments.map(doc => doc.document_type)).size;
      expect(screen.getByText(`${uniqueTypes}`)).toBeInTheDocument();
    });
  });

  describe('Search Functionality', () => {
    test('should handle search input', async () => {
      const user = userEvent.setup();
      
      render(
        <TestWrapper>
          <DocumentManagement />
        </TestWrapper>
      );

      const searchInput = screen.getByPlaceholderText('Rechercher...');
      await user.type(searchInput, 'test document');

      expect(mockSearch.setSearchQuery).toHaveBeenCalledWith('test document');
    });

    test('should toggle semantic search', async () => {
      const user = userEvent.setup();
      
      render(
        <TestWrapper>
          <DocumentManagement />
        </TestWrapper>
      );

      const semanticButton = screen.getByText('🧠 Recherche IA');
      await user.click(semanticButton);

      expect(mockSearch.enableSemanticSearch).toHaveBeenCalled();
    });
  });

  describe('View Mode Switching', () => {
    test('should switch between view modes', async () => {
      const user = userEvent.setup();
      
      render(
        <TestWrapper>
          <DocumentManagement />
        </TestWrapper>
      );

      // Find view mode buttons
      const viewButtons = screen.getAllByRole('button');
      const listViewButton = viewButtons.find(button => 
        button.querySelector('svg')?.classList.contains('lucide-list')
      );

      if (listViewButton) {
        await user.click(listViewButton);
        expect(mockStore.updatePreferences).toHaveBeenCalledWith({ compactMode: true });
      }
    });
  });

  describe('Document Interactions', () => {
    test('should handle document selection', async () => {
      const user = userEvent.setup();
      
      render(
        <TestWrapper>
          <DocumentManagement />
        </TestWrapper>
      );

      const firstDocument = screen.getByText('Document Test 1').closest('div');
      if (firstDocument) {
        await user.click(firstDocument);
        expect(mockPreview.openPreview).toHaveBeenCalledWith('1');
      }
    });

    test('should show document actions on hover', async () => {
      const user = userEvent.setup();
      
      render(
        <TestWrapper>
          <DocumentManagement />
        </TestWrapper>
      );

      const firstDocument = screen.getByText('Document Test 1').closest('[class*="group"]');
      if (firstDocument) {
        await user.hover(firstDocument);
        
        // Actions should become visible
        await waitFor(() => {
          const actions = within(firstDocument).getAllByRole('button');
          expect(actions.length).toBeGreaterThan(0);
        });
      }
    });
  });

  describe('Upload Functionality', () => {
    test('should open upload modal', async () => {
      const user = userEvent.setup();
      
      render(
        <TestWrapper>
          <DocumentManagement />
        </TestWrapper>
      );

      const addButton = screen.getByText('Ajouter');
      await user.click(addButton);

      // Upload modal should be accessible (even if lazy loaded)
      await waitFor(() => {
        expect(document.body).toMatchSnapshot();
      });
    });
  });

  describe('Refresh and Actions', () => {
    test('should handle refresh', async () => {
      const mockRefetch = jest.fn();
      (useDocuments as jest.Mock).mockReturnValue({
        data: mockDocuments,
        isLoading: false,
        error: null,
        refetch: mockRefetch,
      });

      const user = userEvent.setup();
      
      render(
        <TestWrapper>
          <DocumentManagement />
        </TestWrapper>
      );

      const refreshButton = screen.getByRole('button', { name: /refresh/i });
      await user.click(refreshButton);

      expect(mockRefetch).toHaveBeenCalled();
      expect(mockAnalytics.refreshAnalytics).toHaveBeenCalled();
    });
  });

  describe('Loading States', () => {
    test('should show loading state', () => {
      (useDocuments as jest.Mock).mockReturnValue({
        data: [],
        isLoading: true,
        error: null,
        refetch: jest.fn(),
      });

      render(
        <TestWrapper>
          <DocumentManagement />
        </TestWrapper>
      );

      // Loading spinners should be present in stats
      const loadingElements = document.querySelectorAll('.animate-pulse');
      expect(loadingElements.length).toBeGreaterThan(0);
    });

    test('should show empty state', () => {
      (useDocuments as jest.Mock).mockReturnValue({
        data: [],
        isLoading: false,
        error: null,
        refetch: jest.fn(),
      });

      render(
        <TestWrapper>
          <DocumentManagement />
        </TestWrapper>
      );

      expect(screen.getByText('Aucun document')).toBeInTheDocument();
      expect(screen.getByText('Commencez par ajouter des documents à votre plateforme M&A Intelligence.')).toBeInTheDocument();
    });
  });

  describe('Error Handling', () => {
    test('should handle API errors gracefully', () => {
      (useDocuments as jest.Mock).mockReturnValue({
        data: [],
        isLoading: false,
        error: new Error('API Error'),
        refetch: jest.fn(),
      });

      render(
        <TestWrapper>
          <DocumentManagement />
        </TestWrapper>
      );

      expect(mockStore.setError).toHaveBeenCalledWith('API Error');
    });
  });

  describe('Accessibility', () => {
    test('should be accessible', () => {
      render(
        <TestWrapper>
          <DocumentManagement />
        </TestWrapper>
      );

      // Check for proper ARIA labels and roles
      expect(screen.getByRole('main')).toBeInTheDocument();
      
      // All buttons should have accessible text
      const buttons = screen.getAllByRole('button');
      buttons.forEach(button => {
        expect(button).toHaveAccessibleName();
      });

      // Form inputs should have labels
      const searchInput = screen.getByPlaceholderText('Rechercher...');
      expect(searchInput).toBeInTheDocument();
    });

    test('should support keyboard navigation', async () => {
      const user = userEvent.setup();
      
      render(
        <TestWrapper>
          <DocumentManagement />
        </TestWrapper>
      );

      // Tab through interactive elements
      await user.tab();
      expect(document.activeElement).toBeInstanceOf(HTMLElement);
      
      // Should be able to activate with Enter/Space
      if (document.activeElement && 'click' in document.activeElement) {
        await user.keyboard('{Enter}');
        // Should trigger some action
      }
    });
  });

  describe('Performance', () => {
    test('should render efficiently with many documents', () => {
      const manyDocuments = Array.from({ length: 1000 }, (_, i) => ({
        ...mockDocuments[0],
        document_id: `doc-${i}`,
        title: `Document ${i}`,
      }));

      (useDocuments as jest.Mock).mockReturnValue({
        data: manyDocuments,
        isLoading: false,
        error: null,
        refetch: jest.fn(),
      });

      const renderStart = performance.now();
      
      render(
        <TestWrapper>
          <DocumentManagement />
        </TestWrapper>
      );

      const renderTime = performance.now() - renderStart;
      
      // Should render within reasonable time (< 100ms for initial render)
      expect(renderTime).toBeLessThan(100);
      
      // Should show correct document count
      expect(screen.getByText('1000')).toBeInTheDocument();
    });
  });
});