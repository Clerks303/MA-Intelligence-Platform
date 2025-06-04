/**
 * Page Principale de Gestion Documentaire - M&A Intelligence Platform
 * Sprint 3 - Integration complète avec routing, state management hybride et performance
 */

import React, { useState, useCallback, useEffect, Suspense } from 'react';
import { Button } from '../../../components/ui/button';
import { Card, CardHeader, CardContent } from '../../../components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../../../components/ui/tabs';
import { Badge } from '../../../components/ui/badge';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '../../../components/ui/dialog';
import { 
  Upload, 
  Search, 
  Filter, 
  Grid, 
  List,
  FolderTree,
  Settings,
  Download,
  Share2,
  MoreHorizontal,
  RefreshCw,
  BarChart3,
  FileText,
  AlertCircle,
  CheckCircle2,
  Clock,
  Eye,
  Plus,
  Trash2,
  Archive,
  Star,
  Menu,
  X
} from 'lucide-react';
import { cn } from '../../../lib/utils';

import { Document, DocumentFilters, ViewMode } from '../types';
import { 
  useAdvancedDocumentStore,
  useDocumentSelection,
  useDocumentSearch,
  useDocumentFilters,
  useDocumentAnalytics,
  useDocumentPreview,
  useDocumentPerformance,
  useDocumentOperations 
} from '../stores/advancedDocumentStore';
import { useDocuments } from '../hooks/useDocuments';
import { useSemanticSearch, useAdvancedUpload, useDocumentAnalytics as useAnalyticsHook } from '../hooks/useAdvancedDocuments';

// Lazy loading des composants
const VirtualizedDocumentTree = React.lazy(() => 
  import('../components/VirtualizedDocumentTree').then(module => ({ 
    default: module.VirtualizedDocumentTree 
  }))
);
const AdvancedDocumentUpload = React.lazy(() => 
  import('../components/AdvancedDocumentUpload').then(module => ({ 
    default: module.AdvancedDocumentUpload 
  }))
);
const ModernDocumentPreview = React.lazy(() => 
  import('../components/ModernDocumentPreview').then(module => ({ 
    default: module.ModernDocumentPreview 
  }))
);

interface DocumentManagementProps {
  initialFilters?: DocumentFilters;
  enableAnalytics?: boolean;
  enableUpload?: boolean;
  compactMode?: boolean;
}

interface QuickStatsProps {
  documents: Document[];
  analyticsData?: any;
  isLoading?: boolean;
}

interface DocumentGridProps {
  documents: Document[];
  onSelectDocument: (document: Document) => void;
  onDocumentAction: (action: string, document: Document) => void;
  viewMode: ViewMode;
}

// Composant des statistiques rapides
const QuickStats: React.FC<QuickStatsProps> = ({ documents, analyticsData, isLoading }) => {
  const stats = React.useMemo(() => {
    const total = documents.length;
    const byType = documents.reduce((acc, doc) => {
      acc[doc.document_type] = (acc[doc.document_type] || 0) + 1;
      return acc;
    }, {} as Record<string, number>);
    
    const totalSize = documents.reduce((sum, doc) => sum + doc.file_size, 0);
    const recentCount = documents.filter(doc => 
      new Date(doc.created_at) > new Date(Date.now() - 7 * 24 * 60 * 60 * 1000)
    ).length;

    return { total, byType, totalSize, recentCount };
  }, [documents]);

  const formatSize = (bytes: number): string => {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return `${parseFloat((bytes / Math.pow(k, i)).toFixed(1))} ${sizes[i]}`;
  };

  if (isLoading) {
    return (
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
        {[...Array(4)].map((_, i) => (
          <Card key={i} className="animate-pulse">
            <CardContent className="p-4">
              <div className="h-4 bg-gray-200 rounded mb-2"></div>
              <div className="h-6 bg-gray-200 rounded"></div>
            </CardContent>
          </Card>
        ))}
      </div>
    );
  }

  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
      <Card>
        <CardContent className="p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Total Documents</p>
              <p className="text-2xl font-bold text-blue-600">{stats.total}</p>
            </div>
            <FileText className="h-8 w-8 text-blue-500" />
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardContent className="p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Taille Totale</p>
              <p className="text-2xl font-bold text-green-600">{formatSize(stats.totalSize)}</p>
            </div>
            <Archive className="h-8 w-8 text-green-500" />
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardContent className="p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Cette Semaine</p>
              <p className="text-2xl font-bold text-purple-600">{stats.recentCount}</p>
            </div>
            <Clock className="h-8 w-8 text-purple-500" />
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardContent className="p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Types</p>
              <p className="text-2xl font-bold text-orange-600">{Object.keys(stats.byType).length}</p>
            </div>
            <BarChart3 className="h-8 w-8 text-orange-500" />
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

// Composant de grille de documents
const DocumentGrid: React.FC<DocumentGridProps> = ({ 
  documents, 
  onSelectDocument, 
  onDocumentAction,
  viewMode 
}) => {
  const [hoveredDocument, setHoveredDocument] = useState<string | null>(null);

  const getDocumentTypeIcon = (type: string) => {
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
  };

  const formatDate = (date: Date | string): string => {
    const d = typeof date === 'string' ? new Date(date) : date;
    return new Intl.DateTimeFormat('fr-FR', {
      day: 'numeric',
      month: 'short',
      year: 'numeric',
    }).format(d);
  };

  if (viewMode === 'list') {
    return (
      <div className="space-y-2">
        {documents.map((document) => (
          <Card 
            key={document.document_id} 
            className={cn(
              "cursor-pointer transition-all duration-200 hover:shadow-md",
              hoveredDocument === document.document_id && "ring-2 ring-blue-500"
            )}
            onMouseEnter={() => setHoveredDocument(document.document_id)}
            onMouseLeave={() => setHoveredDocument(null)}
            onClick={() => onSelectDocument(document)}
          >
            <CardContent className="p-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3 flex-1 min-w-0">
                  <div className="text-2xl">
                    {getDocumentTypeIcon(document.document_type)}
                  </div>
                  <div className="flex-1 min-w-0">
                    <h3 className="font-medium truncate">
                      {document.title || document.filename}
                    </h3>
                    <div className="flex items-center gap-2 text-sm text-gray-500 mt-1">
                      <span>{document.document_type}</span>
                      <span>•</span>
                      <span>{formatDate(document.created_at)}</span>
                      <span>•</span>
                      <span>{document.view_count} vues</span>
                    </div>
                  </div>
                </div>

                <div className="flex items-center gap-2">
                  <Badge 
                    variant="secondary" 
                    className={cn(
                      "text-xs",
                      document.access_level === 'public' && "bg-green-100 text-green-800",
                      document.access_level === 'internal' && "bg-blue-100 text-blue-800",
                      document.access_level === 'confidential' && "bg-orange-100 text-orange-800",
                      document.access_level === 'restricted' && "bg-red-100 text-red-800"
                    )}
                  >
                    {document.access_level}
                  </Badge>

                  {hoveredDocument === document.document_id && (
                    <div className="flex items-center gap-1">
                      <Button
                        variant="ghost"
                        size="icon"
                        className="h-8 w-8"
                        onClick={(e) => {
                          e.stopPropagation();
                          onDocumentAction('view', document);
                        }}
                      >
                        <Eye className="h-4 w-4" />
                      </Button>
                      <Button
                        variant="ghost"
                        size="icon"
                        className="h-8 w-8"
                        onClick={(e) => {
                          e.stopPropagation();
                          onDocumentAction('download', document);
                        }}
                      >
                        <Download className="h-4 w-4" />
                      </Button>
                      <Button
                        variant="ghost"
                        size="icon"
                        className="h-8 w-8"
                        onClick={(e) => {
                          e.stopPropagation();
                          onDocumentAction('share', document);
                        }}
                      >
                        <Share2 className="h-4 w-4" />
                      </Button>
                    </div>
                  )}
                </div>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
    );
  }

  // Vue grille
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
      {documents.map((document) => (
        <Card 
          key={document.document_id} 
          className={cn(
            "cursor-pointer transition-all duration-200 hover:shadow-lg group",
            hoveredDocument === document.document_id && "ring-2 ring-blue-500"
          )}
          onMouseEnter={() => setHoveredDocument(document.document_id)}
          onMouseLeave={() => setHoveredDocument(null)}
          onClick={() => onSelectDocument(document)}
        >
          <CardContent className="p-4">
            <div className="flex flex-col h-full">
              <div className="flex items-start justify-between mb-3">
                <div className="text-3xl">
                  {getDocumentTypeIcon(document.document_type)}
                </div>
                <div className="opacity-0 group-hover:opacity-100 transition-opacity">
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-6 w-6"
                    onClick={(e) => {
                      e.stopPropagation();
                      onDocumentAction('favorite', document);
                    }}
                  >
                    <Star className="h-3 w-3" />
                  </Button>
                </div>
              </div>

              <h3 className="font-medium text-sm mb-2 line-clamp-2">
                {document.title || document.filename}
              </h3>

              <div className="flex-1" />

              <div className="space-y-2">
                <div className="flex items-center justify-between text-xs text-gray-500">
                  <span>{document.document_type}</span>
                  <span>{formatDate(document.created_at)}</span>
                </div>

                <div className="flex items-center justify-between">
                  <Badge 
                    variant="secondary" 
                    className={cn(
                      "text-xs",
                      document.access_level === 'public' && "bg-green-100 text-green-800",
                      document.access_level === 'internal' && "bg-blue-100 text-blue-800",
                      document.access_level === 'confidential' && "bg-orange-100 text-orange-800",
                      document.access_level === 'restricted' && "bg-red-100 text-red-800"
                    )}
                  >
                    {document.access_level}
                  </Badge>
                  <span className="text-xs text-gray-500">
                    {document.view_count} vues
                  </span>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      ))}
    </div>
  );
};

// Composant principal
export const DocumentManagement: React.FC<DocumentManagementProps> = ({
  initialFilters = {},
  enableAnalytics = true,
  enableUpload = true,
  compactMode = false,
}) => {
  // État local
  const [sidebarOpen, setSidebarOpen] = useState(!compactMode);
  const [activeView, setActiveView] = useState<'grid' | 'list' | 'tree'>('grid');
  const [showUploadModal, setShowUploadModal] = useState(false);

  // Stores Zustand
  const { 
    documents, 
    viewMode, 
    setLoading, 
    setError, 
    refreshDocuments,
    preferences,
    updatePreferences
  } = useAdvancedDocumentStore();

  const selection = useDocumentSelection();
  const search = useDocumentSearch();
  const filters = useDocumentFilters();
  const analytics = useDocumentAnalytics();
  const preview = useDocumentPreview();
  const performance = useDocumentPerformance();
  const operations = useDocumentOperations();

  // TanStack Query hooks
  const { 
    data: documentsData, 
    isLoading, 
    error, 
    refetch 
  } = useDocuments({
    filters: { ...initialFilters, ...filters.filters },
    enabled: true,
  });

  const analyticsHook = useAnalyticsHook({
    enabled: enableAnalytics,
  });

  const uploadHook = useAdvancedUpload({
    onUploadComplete: (document) => {
      console.log('Document uploaded:', document);
      refetch();
    },
    onUploadError: (error) => {
      console.error('Upload error:', error);
    },
  });

  // Actions
  const handleSelectDocument = useCallback((document: Document) => {
    preview.openPreview(document.document_id);
  }, [preview]);

  const handleDocumentAction = useCallback((action: string, document: Document) => {
    switch (action) {
      case 'view':
        preview.openPreview(document.document_id);
        break;
      case 'download':
        // TODO: Implémenter le téléchargement
        console.log('Download document:', document.document_id);
        break;
      case 'share':
        // TODO: Implémenter le partage
        console.log('Share document:', document.document_id);
        break;
      case 'edit':
        preview.startEditing(document);
        break;
      case 'delete':
        // TODO: Implémenter la suppression
        console.log('Delete document:', document.document_id);
        break;
      case 'favorite':
        // TODO: Implémenter les favoris
        console.log('Favorite document:', document.document_id);
        break;
      default:
        console.log('Unknown action:', action, document);
    }
  }, [preview]);

  const handleRefresh = useCallback(() => {
    refetch();
    if (enableAnalytics) {
      analytics.refreshAnalytics();
    }
  }, [refetch, analytics, enableAnalytics]);

  const handleViewModeChange = useCallback((mode: 'grid' | 'list' | 'tree') => {
    setActiveView(mode);
    updatePreferences({ compactMode: mode === 'list' });
  }, [updatePreferences]);

  // Effets
  useEffect(() => {
    setLoading(isLoading);
  }, [isLoading, setLoading]);

  useEffect(() => {
    if (error) {
      setError(error.message);
    } else {
      setError(null);
    }
  }, [error, setError]);

  const currentDocuments = documentsData || [];
  const filteredDocuments = currentDocuments;

  return (
    <div className="h-full flex flex-col bg-gray-50">
      {/* Header */}
      <div className="bg-white border-b border-gray-200 px-6 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <Button
              variant="ghost"
              size="icon"
              onClick={() => setSidebarOpen(!sidebarOpen)}
              className="lg:hidden"
            >
              {sidebarOpen ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
            </Button>
            
            <div>
              <h1 className="text-2xl font-bold text-gray-900">Gestion Documentaire</h1>
              <p className="text-sm text-gray-600">
                Plateforme M&A Intelligence - Sprint 3
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            {/* Contrôles de vue */}
            <div className="flex items-center border rounded-lg p-1">
              <Button
                variant={activeView === 'grid' ? 'default' : 'ghost'}
                size="sm"
                onClick={() => handleViewModeChange('grid')}
              >
                <Grid className="h-4 w-4" />
              </Button>
              <Button
                variant={activeView === 'list' ? 'default' : 'ghost'}
                size="sm"
                onClick={() => handleViewModeChange('list')}
              >
                <List className="h-4 w-4" />
              </Button>
              <Button
                variant={activeView === 'tree' ? 'default' : 'ghost'}
                size="sm"
                onClick={() => handleViewModeChange('tree')}
              >
                <FolderTree className="h-4 w-4" />
              </Button>
            </div>

            <Button
              variant="outline"
              size="sm"
              onClick={handleRefresh}
              disabled={isLoading}
            >
              <RefreshCw className={cn("h-4 w-4", isLoading && "animate-spin")} />
            </Button>

            {enableUpload && (
              <Button
                onClick={() => setShowUploadModal(true)}
                size="sm"
              >
                <Plus className="h-4 w-4 mr-1" />
                Ajouter
              </Button>
            )}

            <Button variant="outline" size="sm">
              <Settings className="h-4 w-4" />
            </Button>
          </div>
        </div>
      </div>

      {/* Contenu principal */}
      <div className="flex-1 flex overflow-hidden">
        {/* Sidebar */}
        {sidebarOpen && (
          <div className="w-80 bg-white border-r border-gray-200 flex flex-col">
            <div className="p-4 border-b">
              <h2 className="font-medium text-gray-900 mb-3">Navigation</h2>
              
              {/* Recherche */}
              <div className="relative mb-4">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
                <input
                  type="text"
                  placeholder="Rechercher..."
                  value={search.searchQuery}
                  onChange={(e) => search.setSearchQuery(e.target.value)}
                  className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                />
              </div>

              {/* Mode recherche sémantique */}
              <div className="flex items-center gap-2 mb-4">
                <Button
                  variant={search.semanticSearchEnabled ? "default" : "outline"}
                  size="sm"
                  onClick={search.semanticSearchEnabled ? search.disableSemanticSearch : search.enableSemanticSearch}
                  className="text-xs"
                >
                  🧠 Recherche IA
                </Button>
                {search.semanticSearchEnabled && search.semanticResults.length > 0 && (
                  <Badge variant="secondary" className="text-xs">
                    {search.semanticResults.length} résultats
                  </Badge>
                )}
              </div>
            </div>

            {/* Arbre de navigation */}
            <div className="flex-1 overflow-hidden">
              <Suspense fallback={
                <div className="flex items-center justify-center h-40">
                  <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-500"></div>
                </div>
              }>
                <VirtualizedDocumentTree
                  documents={filteredDocuments}
                  onSelectDocument={handleSelectDocument}
                  onDocumentAction={handleDocumentAction}
                  filters={filters.filters}
                  onFiltersChange={filters.setFilters}
                  height={600}
                  enableSearch={false} // Recherche déjà dans la sidebar
                  enableVirtualization={performance.virtualizationEnabled}
                />
              </Suspense>
            </div>
          </div>
        )}

        {/* Zone principale */}
        <div className="flex-1 flex flex-col overflow-hidden">
          <div className="p-6 flex-1 overflow-auto">
            {/* Statistiques rapides */}
            <QuickStats 
              documents={filteredDocuments}
              analyticsData={analytics.analyticsData}
              isLoading={isLoading}
            />

            {/* Contenu selon la vue */}
            {activeView === 'tree' ? (
              <Card className="h-[600px]">
                <CardContent className="p-0">
                  <Suspense fallback={
                    <div className="flex items-center justify-center h-full">
                      <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
                    </div>
                  }>
                    <VirtualizedDocumentTree
                      documents={filteredDocuments}
                      onSelectDocument={handleSelectDocument}
                      onDocumentAction={handleDocumentAction}
                      filters={filters.filters}
                      onFiltersChange={filters.setFilters}
                      height={600}
                      enableSearch={true}
                      enableVirtualization={performance.virtualizationEnabled}
                    />
                  </Suspense>
                </CardContent>
              </Card>
            ) : (
              <DocumentGrid
                documents={filteredDocuments}
                onSelectDocument={handleSelectDocument}
                onDocumentAction={handleDocumentAction}
                viewMode={activeView}
              />
            )}

            {/* État vide */}
            {!isLoading && filteredDocuments.length === 0 && (
              <div className="flex flex-col items-center justify-center h-64 text-gray-500">
                <FileText className="h-16 w-16 mb-4" />
                <h3 className="text-lg font-medium mb-2">Aucun document</h3>
                <p className="text-center mb-4">
                  Commencez par ajouter des documents à votre plateforme M&A Intelligence.
                </p>
                {enableUpload && (
                  <Button onClick={() => setShowUploadModal(true)}>
                    <Upload className="h-4 w-4 mr-2" />
                    Ajouter des documents
                  </Button>
                )}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Modales */}
      {enableUpload && showUploadModal && (
        <Suspense fallback={null}>
          <AdvancedDocumentUpload
            isOpen={showUploadModal}
            onOpenChange={setShowUploadModal}
            onUploadComplete={(documents) => {
              console.log('Upload completed:', documents);
              refetch();
            }}
            onUploadError={(error) => {
              console.error('Upload error:', error);
            }}
          />
        </Suspense>
      )}

      {preview.isPreviewOpen && preview.previewDocumentId && (
        <Suspense fallback={null}>
          <ModernDocumentPreview
            document={filteredDocuments.find(d => d.document_id === preview.previewDocumentId) || null}
            isOpen={preview.isPreviewOpen}
            onOpenChange={preview.closePreview}
            onDocumentUpdate={(updatedDocument) => {
              console.log('Document updated:', updatedDocument);
              refetch();
            }}
            onDownload={handleDocumentAction.bind(null, 'download')}
            onShare={handleDocumentAction.bind(null, 'share')}
            enableEditing={true}
            enableAnnotations={false}
            enableAnalytics={enableAnalytics}
          />
        </Suspense>
      )}
    </div>
  );
};

export default DocumentManagement;