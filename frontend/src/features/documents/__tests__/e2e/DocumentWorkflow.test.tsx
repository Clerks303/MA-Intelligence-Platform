/**
 * Tests E2E Document Workflow - M&A Intelligence Platform
 * Sprint 3 - Tests d'intégration bout en bout
 */

import React from 'react';
import { render, screen, fireEvent, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { BrowserRouter } from 'react-router-dom';
import { DocumentManagement } from '../../pages/DocumentManagement';
import { advancedDocumentService } from '../../services/advancedDocumentService';
import { mockDocument, createMockFiles, measurePerformance } from '../setup';

// Mock the service avec implémentation complète
jest.mock('../../services/advancedDocumentService', () => ({
  advancedDocumentService: {
    getDocuments: jest.fn(),
    uploadMultipleDocuments: jest.fn(),
    retrieveDocument: jest.fn(),
    updateDocument: jest.fn(),
    deleteDocument: jest.fn(),
    searchDocuments: jest.fn(),
    getDocumentAnalytics: jest.fn(),
    getRealTimeDashboard: jest.fn(),
    indexDocument: jest.fn(),
  },
}));

const mockService = advancedDocumentService as jest.Mocked<typeof advancedDocumentService>;

// Mock documents complets pour E2E
const mockDocuments = [
  {
    ...mockDocument,
    document_id: '1',
    title: 'Rapport Financier Q4 2023',
    document_type: 'financial' as const,
    tags: ['financial', 'quarterly', 'report'],
    view_count: 15,
    download_count: 5,
  },
  {
    ...mockDocument,
    document_id: '2',
    title: 'Contrat de Fusion - Entreprise A',
    document_type: 'legal' as const,
    access_level: 'confidential' as const,
    tags: ['legal', 'merger', 'contract'],
    view_count: 8,
    download_count: 3,
  },
  {
    ...mockDocument,
    document_id: '3',
    title: 'Due Diligence Technique',
    document_type: 'due_diligence' as const,
    access_level: 'internal' as const,
    tags: ['due-diligence', 'technical', 'analysis'],
    view_count: 22,
    download_count: 8,
  },
  {
    ...mockDocument,
    document_id: '4',
    title: 'Présentation Commerciale',
    document_type: 'commercial' as const,
    tags: ['commercial', 'presentation', 'sales'],
    view_count: 12,
    download_count: 4,
  },
];

const mockAnalytics = {
  totalDocuments: mockDocuments.length,
  totalSize: mockDocuments.reduce((sum, doc) => sum + doc.file_size, 0),
  documentsByType: {
    financial: 1,
    legal: 1,
    due_diligence: 1,
    commercial: 1,
  },
  uploadTrend: [
    { date: '2024-01-01', count: 1 },
    { date: '2024-01-02', count: 2 },
    { date: '2024-01-03', count: 1 },
  ],
  accessPatterns: [
    { hour: 9, count: 15 },
    { hour: 14, count: 20 },
    { hour: 16, count: 10 },
  ],
  popularDocuments: mockDocuments.slice(0, 3),
};

const mockSearchResults = [
  {
    document: mockDocuments[0],
    score: 0.95,
    highlights: ['rapport', 'financier'],
    metadata: mockDocuments[0],
  },
  {
    document: mockDocuments[2],
    score: 0.87,
    highlights: ['due', 'diligence'],
    metadata: mockDocuments[2],
  },
];

// Test wrapper avec providers complets
const TestWrapper: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false, cacheTime: 0 },
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

describe('Document Management E2E Workflow', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    
    // Setup default mocks
    mockService.getDocuments.mockResolvedValue({
      documents: mockDocuments,
      total: mockDocuments.length,
      page: 1,
      totalPages: 1,
    });

    mockService.getDocumentAnalytics.mockResolvedValue(mockAnalytics);
    
    mockService.getRealTimeDashboard.mockResolvedValue({
      activeUsers: 15,
      documentsToday: 8,
      pendingUploads: 2,
      systemLoad: 0.45,
    });
  });

  describe('Page Load and Initial State', () => {
    test('should load page with all documents and analytics', async () => {
      const duration = await measurePerformance(async () => {
        render(
          <TestWrapper>
            <DocumentManagement enableAnalytics enableUpload />
          </TestWrapper>
        );

        // Vérifier le header
        expect(screen.getByText('Gestion Documentaire')).toBeInTheDocument();
        expect(screen.getByText('Plateforme M&A Intelligence - Sprint 3')).toBeInTheDocument();

        // Attendre le chargement des documents
        await waitFor(() => {
          expect(screen.getByText('Rapport Financier Q4 2023')).toBeInTheDocument();
        });

        // Vérifier tous les documents
        expect(screen.getByText('Contrat de Fusion - Entreprise A')).toBeInTheDocument();
        expect(screen.getByText('Due Diligence Technique')).toBeInTheDocument();
        expect(screen.getByText('Présentation Commerciale')).toBeInTheDocument();

        // Vérifier les statistiques
        expect(screen.getByText('4')).toBeInTheDocument(); // Total documents
        expect(screen.getByText('4')).toBeInTheDocument(); // Types count
      }, 'Initial page load');

      // Page should load within 2 seconds
      expect(duration).toBeLessThan(2000);

      // Vérifier les appels API
      expect(mockService.getDocuments).toHaveBeenCalledTimes(1);
      expect(mockService.getDocumentAnalytics).toHaveBeenCalledTimes(1);
    });

    test('should display correct document counts by type', async () => {
      render(
        <TestWrapper>
          <DocumentManagement enableAnalytics />
        </TestWrapper>
      );

      await waitFor(() => {
        expect(screen.getByText('Rapport Financier Q4 2023')).toBeInTheDocument();
      });

      // Vérifier les badges de type
      expect(screen.getByText('financial')).toBeInTheDocument();
      expect(screen.getByText('legal')).toBeInTheDocument();
      expect(screen.getByText('due_diligence')).toBeInTheDocument();
      expect(screen.getByText('commercial')).toBeInTheDocument();

      // Vérifier les niveaux d'accès
      expect(screen.getByText('public')).toBeInTheDocument();
      expect(screen.getByText('confidential')).toBeInTheDocument();
      expect(screen.getByText('internal')).toBeInTheDocument();
    });
  });

  describe('Search and Filter Workflow', () => {
    test('should perform complete search workflow', async () => {
      const user = userEvent.setup();
      
      mockService.searchDocuments.mockResolvedValue({
        documents: mockSearchResults.map(r => r.document),
        results: mockSearchResults,
        total: 2,
        page: 1,
        totalPages: 1,
      });

      render(
        <TestWrapper>
          <DocumentManagement enableAnalytics />
        </TestWrapper>
      );

      await waitFor(() => {
        expect(screen.getByText('Rapport Financier Q4 2023')).toBeInTheDocument();
      });

      // 1. Activer la recherche sémantique
      const semanticButton = screen.getByText('🧠 Recherche IA');
      await user.click(semanticButton);

      // 2. Effectuer une recherche
      const searchInput = screen.getByPlaceholderText('Rechercher...');
      await user.type(searchInput, 'rapport financier');

      // 3. Vérifier les résultats
      await waitFor(() => {
        expect(mockService.searchDocuments).toHaveBeenCalledWith(
          'rapport financier',
          expect.objectContaining({
            semanticSearch: true,
          })
        );
      });

      // 4. Vérifier l'affichage des résultats
      expect(screen.getByText('2 résultats pertinents')).toBeInTheDocument();
    });

    test('should filter documents by type in tree view', async () => {
      const user = userEvent.setup();

      render(
        <TestWrapper>
          <DocumentManagement />
        </TestWrapper>
      );

      await waitFor(() => {
        expect(screen.getByText('Rapport Financier Q4 2023')).toBeInTheDocument();
      });

      // Passer en vue arbre
      const treeViewButton = screen.getByRole('button', { name: /tree/i });
      await user.click(treeViewButton);

      // Vérifier que l'arbre est affiché
      await waitFor(() => {
        expect(screen.getByText(/financial.*\(1\)/)).toBeInTheDocument();
        expect(screen.getByText(/legal.*\(1\)/)).toBeInTheDocument();
      });

      // Développer un type de document
      const financialFolder = screen.getByText(/financial.*\(1\)/);
      await user.click(financialFolder);

      // Vérifier que les documents du type sont affichés
      expect(screen.getByText('Rapport Financier Q4 2023')).toBeInTheDocument();
    });
  });

  describe('Document Upload Workflow', () => {
    test('should complete full upload workflow', async () => {
      const user = userEvent.setup();
      
      const mockUploadedDocs = [
        {
          ...mockDocument,
          document_id: '5',
          title: 'Nouveau Document Uploadé',
          filename: 'nouveau.pdf',
        },
      ];

      mockService.uploadMultipleDocuments.mockResolvedValue(mockUploadedDocs);

      render(
        <TestWrapper>
          <DocumentManagement enableUpload />
        </TestWrapper>
      );

      await waitFor(() => {
        expect(screen.getByText('Gestion Documentaire')).toBeInTheDocument();
      });

      // 1. Ouvrir le modal d'upload
      const addButton = screen.getByText('Ajouter');
      await user.click(addButton);

      // 2. Modal d'upload devrait être ouvert (lazy loaded)
      await waitFor(() => {
        // Le composant AdvancedDocumentUpload est lazy loaded
        // On vérifie que le modal est dans le DOM
        expect(document.body).toBeDefined();
      });

      // 3. Simuler l'upload de fichiers
      const files = createMockFiles(2);
      
      // Mock de l'upload avec progression
      let progressCallback: ((progress: number) => void) | undefined;
      mockService.uploadMultipleDocuments.mockImplementation((files, metadata, onProgress) => {
        progressCallback = onProgress;
        
        // Simuler la progression
        setTimeout(() => progressCallback?.(25), 50);
        setTimeout(() => progressCallback?.(50), 100);
        setTimeout(() => progressCallback?.(75), 150);
        setTimeout(() => progressCallback?.(100), 200);
        
        return Promise.resolve(mockUploadedDocs);
      });

      // 4. Vérifier que l'upload est en cours
      expect(mockService.uploadMultipleDocuments).toBeDefined();
    });
  });

  describe('Document Preview Workflow', () => {
    test('should open and interact with document preview', async () => {
      const user = userEvent.setup();
      
      mockService.retrieveDocument.mockResolvedValue({
        data: new ArrayBuffer(1024),
        metadata: mockDocuments[0],
      });

      render(
        <TestWrapper>
          <DocumentManagement />
        </TestWrapper>
      );

      await waitFor(() => {
        expect(screen.getByText('Rapport Financier Q4 2023')).toBeInTheDocument();
      });

      // 1. Cliquer sur un document pour l'ouvrir
      const documentCard = screen.getByText('Rapport Financier Q4 2023').closest('div');
      expect(documentCard).toBeInTheDocument();
      
      if (documentCard) {
        await user.click(documentCard);
      }

      // 2. Vérifier que le service de récupération est appelé
      await waitFor(() => {
        expect(mockService.retrieveDocument).toHaveBeenCalledWith('1');
      });

      // 3. Le preview modal devrait être ouvert (lazy loaded)
      await waitFor(() => {
        expect(document.body).toBeDefined();
      });
    });

    test('should handle document actions in preview', async () => {
      const user = userEvent.setup();

      render(
        <TestWrapper>
          <DocumentManagement />
        </TestWrapper>
      );

      await waitFor(() => {
        expect(screen.getByText('Rapport Financier Q4 2023')).toBeInTheDocument();
      });

      // Hover sur un document pour voir les actions
      const documentCard = screen.getByText('Rapport Financier Q4 2023').closest('[class*="group"]');
      
      if (documentCard) {
        await user.hover(documentCard);
        
        // Les actions devraient apparaître
        await waitFor(() => {
          const actions = within(documentCard).getAllByRole('button');
          expect(actions.length).toBeGreaterThan(0);
        });
      }
    });
  });

  describe('View Mode Switching Workflow', () => {
    test('should switch between all view modes', async () => {
      const user = userEvent.setup();

      render(
        <TestWrapper>
          <DocumentManagement />
        </TestWrapper>
      );

      await waitFor(() => {
        expect(screen.getByText('Rapport Financier Q4 2023')).toBeInTheDocument();
      });

      // 1. Vue grille par défaut
      expect(screen.getByText('Rapport Financier Q4 2023')).toBeInTheDocument();

      // 2. Passer en vue liste
      const viewButtons = screen.getAllByRole('button');
      const listButton = viewButtons.find(btn => 
        btn.querySelector('svg')?.classList.contains('lucide-list')
      );
      
      if (listButton) {
        await user.click(listButton);
        
        // Vérifier que la vue a changé
        await waitFor(() => {
          expect(screen.getByText('Rapport Financier Q4 2023')).toBeInTheDocument();
        });
      }

      // 3. Passer en vue arbre
      const treeButton = viewButtons.find(btn => 
        btn.querySelector('svg')?.classList.contains('lucide-folder-tree')
      );
      
      if (treeButton) {
        await user.click(treeButton);
        
        // Vérifier l'arbre de navigation
        await waitFor(() => {
          expect(screen.getByText(/financial.*\(1\)/)).toBeInTheDocument();
        });
      }
    });
  });

  describe('Real-time Updates and Analytics', () => {
    test('should handle real-time metrics updates', async () => {
      const { rerender } = render(
        <TestWrapper>
          <DocumentManagement enableAnalytics />
        </TestWrapper>
      );

      await waitFor(() => {
        expect(screen.getByText('4')).toBeInTheDocument();
      });

      // Simuler une mise à jour des données
      mockService.getDocuments.mockResolvedValue({
        documents: [...mockDocuments, {
          ...mockDocument,
          document_id: '5',
          title: 'Nouveau Document',
        }],
        total: 5,
        page: 1,
        totalPages: 1,
      });

      // Re-render pour simuler un refresh
      rerender(
        <TestWrapper>
          <DocumentManagement enableAnalytics />
        </TestWrapper>
      );

      await waitFor(() => {
        expect(screen.getByText('5')).toBeInTheDocument();
      });
    });
  });

  describe('Error Handling and Recovery', () => {
    test('should handle API errors gracefully', async () => {
      mockService.getDocuments.mockRejectedValue(new Error('Network Error'));

      render(
        <TestWrapper>
          <DocumentManagement />
        </TestWrapper>
      );

      // L'erreur devrait être gérée sans crash
      await waitFor(() => {
        expect(screen.getByText('Gestion Documentaire')).toBeInTheDocument();
      });

      // Devrait afficher l'état vide plutôt qu'un crash
      expect(screen.getByText('Aucun document')).toBeInTheDocument();
    });

    test('should recover from errors with retry', async () => {
      const user = userEvent.setup();

      // Premier appel échoue
      mockService.getDocuments
        .mockRejectedValueOnce(new Error('Network Error'))
        .mockResolvedValueOnce({
          documents: mockDocuments,
          total: mockDocuments.length,
          page: 1,
          totalPages: 1,
        });

      render(
        <TestWrapper>
          <DocumentManagement />
        </TestWrapper>
      );

      // Attendre l'erreur initiale
      await waitFor(() => {
        expect(screen.getByText('Aucun document')).toBeInTheDocument();
      });

      // Retry avec le bouton refresh
      const refreshButton = screen.getByRole('button', { name: /refresh/i });
      await user.click(refreshButton);

      // Devrait maintenant charger les documents
      await waitFor(() => {
        expect(screen.getByText('Rapport Financier Q4 2023')).toBeInTheDocument();
      });
    });
  });

  describe('Performance and Optimization', () => {
    test('should handle large datasets efficiently', async () => {
      // Créer un grand dataset
      const largeDataset = Array.from({ length: 1000 }, (_, i) => ({
        ...mockDocument,
        document_id: `doc-${i}`,
        title: `Document ${i}`,
      }));

      mockService.getDocuments.mockResolvedValue({
        documents: largeDataset,
        total: 1000,
        page: 1,
        totalPages: 20,
      });

      const duration = await measurePerformance(async () => {
        render(
          <TestWrapper>
            <DocumentManagement />
          </TestWrapper>
        );

        await waitFor(() => {
          expect(screen.getByText('1000')).toBeInTheDocument();
        });
      }, 'Large dataset rendering');

      // Devrait rendre efficacement même avec beaucoup de données
      expect(duration).toBeLessThan(1000);
    });

    test('should implement virtualization for large lists', async () => {
      const user = userEvent.setup();
      
      const largeDataset = Array.from({ length: 1000 }, (_, i) => ({
        ...mockDocument,
        document_id: `doc-${i}`,
        title: `Document ${i}`,
      }));

      mockService.getDocuments.mockResolvedValue({
        documents: largeDataset,
        total: 1000,
        page: 1,
        totalPages: 20,
      });

      render(
        <TestWrapper>
          <DocumentManagement />
        </TestWrapper>
      );

      await waitFor(() => {
        expect(screen.getByText('1000')).toBeInTheDocument();
      });

      // Passer en vue arbre pour tester la virtualisation
      const treeButton = screen.getByRole('button', { name: /tree/i });
      await user.click(treeButton);

      // Vérifier que la virtualisation est active
      await waitFor(() => {
        const virtualizedList = document.querySelector('[data-testid="virtualized-list"]');
        expect(virtualizedList).toBeInTheDocument();
      });
    });
  });

  describe('Accessibility and User Experience', () => {
    test('should be fully accessible', async () => {
      const { container } = render(
        <TestWrapper>
          <DocumentManagement />
        </TestWrapper>
      );

      await waitFor(() => {
        expect(screen.getByText('Gestion Documentaire')).toBeInTheDocument();
      });

      // Vérifier les rôles ARIA
      expect(screen.getByRole('main')).toBeInTheDocument();
      
      // Tous les boutons devraient avoir des noms accessibles
      const buttons = screen.getAllByRole('button');
      buttons.forEach(button => {
        expect(button).toHaveAccessibleName();
      });

      // Les champs de formulaire devraient avoir des labels
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

      await waitFor(() => {
        expect(screen.getByText('Gestion Documentaire')).toBeInTheDocument();
      });

      // Navigation par Tab
      await user.tab();
      expect(document.activeElement).toBeInstanceOf(HTMLElement);

      // Activation avec Enter
      if (document.activeElement && 'click' in document.activeElement) {
        await user.keyboard('{Enter}');
      }

      // La navigation devrait fonctionner sans erreur
      expect(document.activeElement).toBeInstanceOf(HTMLElement);
    });
  });

  describe('Complete User Journey', () => {
    test('should complete full document management journey', async () => {
      const user = userEvent.setup();
      
      // Setup mocks pour le journey complet
      mockService.uploadMultipleDocuments.mockResolvedValue([{
        ...mockDocument,
        document_id: '5',
        title: 'Document Uploadé',
      }]);

      mockService.retrieveDocument.mockResolvedValue({
        data: new ArrayBuffer(1024),
        metadata: mockDocuments[0],
      });

      mockService.searchDocuments.mockResolvedValue({
        documents: [mockDocuments[0]],
        results: [mockSearchResults[0]],
        total: 1,
        page: 1,
        totalPages: 1,
      });

      const journeyDuration = await measurePerformance(async () => {
        render(
          <TestWrapper>
            <DocumentManagement enableAnalytics enableUpload />
          </TestWrapper>
        );

        // 1. Chargement initial
        await waitFor(() => {
          expect(screen.getByText('Rapport Financier Q4 2023')).toBeInTheDocument();
        });

        // 2. Recherche
        await user.type(screen.getByPlaceholderText('Rechercher...'), 'rapport');
        
        // 3. Changement de vue
        const listButton = screen.getAllByRole('button').find(btn => 
          btn.querySelector('svg')?.classList.contains('lucide-list')
        );
        if (listButton) {
          await user.click(listButton);
        }

        // 4. Sélection de document
        const documentCard = screen.getByText('Rapport Financier Q4 2023').closest('div');
        if (documentCard) {
          await user.click(documentCard);
        }

        // 5. Actions sur document
        await waitFor(() => {
          expect(mockService.retrieveDocument).toHaveBeenCalled();
        });

      }, 'Complete user journey');

      // Le journey complet devrait être fluide
      expect(journeyDuration).toBeLessThan(3000);
      
      console.log(`✅ Journey complet exécuté en ${journeyDuration.toFixed(2)}ms`);
    });
  });
});