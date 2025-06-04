/**
 * Prévisualisation Documents Moderne - M&A Intelligence Platform
 * Sprint 3 - Preview complet avec backend intégré, analytics et performance
 */

import React, { useState, useCallback, useMemo, lazy, Suspense, useEffect } from 'react';
import { Button } from '../../../components/ui/button';
import { Card, CardHeader, CardContent } from '../../../components/ui/card';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '../../../components/ui/dialog';
import { Badge } from '../../../components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../../../components/ui/tabs';
import { Progress } from '../../../components/ui/progress';
import { Input } from '../../../components/ui/input';
import { Textarea } from '../../../components/ui/textarea';
import { 
  Download, 
  ZoomIn, 
  ZoomOut, 
  RotateCw, 
  Maximize2, 
  Minimize2,
  FileText,
  Image as ImageIcon,
  Video,
  Music,
  Archive,
  AlertCircle,
  Eye,
  Share2,
  Edit,
  MessageSquare,
  X,
  Save,
  History,
  Users,
  Lock,
  Globe,
  Shield,
  Key,
  Search
} from 'lucide-react';
import { cn } from '../../../lib/utils';

import { Document, BackendDocumentType, BackendAccessLevel } from '../types';
import { advancedDocumentService } from '../services/advancedDocumentService';
import { useDocumentIndexing } from '../hooks/useAdvancedDocuments';

// Lazy loading des composants de preview spécialisés
const PDFViewer = lazy(() => import('./previews/PDFPreview'));
const ImageViewer = lazy(() => import('./previews/ImagePreview'));
const VideoViewer = lazy(() => import('./previews/VideoPreview'));
const TextViewer = lazy(() => import('./previews/TextPreview'));

interface ModernDocumentPreviewProps {
  document: Document | null;
  isOpen: boolean;
  onOpenChange: (open: boolean) => void;
  onDocumentUpdate?: (document: Document) => void;
  onDownload?: (document: Document) => void;
  onShare?: (document: Document) => void;
  enableEditing?: boolean;
  enableAnnotations?: boolean;
  enableAnalytics?: boolean;
}

interface PreviewState {
  zoom: number;
  rotation: number;
  isFullscreen: boolean;
  currentPage: number;
  isLoading: boolean;
  error: string | null;
  documentData: ArrayBuffer | null;
  isEditing: boolean;
  editedMetadata: Partial<Document>;
}

interface DocumentAnalytics {
  viewCount: number;
  downloadCount: number;
  lastAccessed: Date;
  averageViewTime: number;
  popularSections: string[];
}

// Utilitaires
const getDocumentTypeIcon = (documentType: BackendDocumentType) => {
  const icons: Record<BackendDocumentType, React.ReactNode> = {
    financial: <span className="text-lg">💰</span>,
    legal: <span className="text-lg">⚖️</span>,
    due_diligence: <span className="text-lg">🔍</span>,
    communication: <span className="text-lg">💬</span>,
    technical: <span className="text-lg">🔧</span>,
    hr: <span className="text-lg">👥</span>,
    commercial: <span className="text-lg">📈</span>,
    other: <FileText className="w-5 h-5" />,
  };
  return icons[documentType] || <FileText className="w-5 h-5" />;
};

const getAccessLevelIcon = (accessLevel: BackendAccessLevel) => {
  const icons: Record<BackendAccessLevel, React.ReactNode> = {
    public: <Globe className="w-4 h-4" />,
    internal: <Users className="w-4 h-4" />,
    confidential: <Shield className="w-4 h-4" />,
    restricted: <Lock className="w-4 h-4" />,
  };
  return icons[accessLevel];
};

const getAccessLevelColor = (accessLevel: BackendAccessLevel) => {
  const colors: Record<BackendAccessLevel, string> = {
    public: 'bg-green-100 text-green-800 border-green-300',
    internal: 'bg-blue-100 text-blue-800 border-blue-300',
    confidential: 'bg-orange-100 text-orange-800 border-orange-300',
    restricted: 'bg-red-100 text-red-800 border-red-300',
  };
  return colors[accessLevel] || 'bg-gray-100 text-gray-800 border-gray-300';
};

const formatFileSize = (bytes: number): string => {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(1))} ${sizes[i]}`;
};

const formatDate = (date: Date | string): string => {
  const d = typeof date === 'string' ? new Date(date) : date;
  return new Intl.DateTimeFormat('fr-FR', {
    year: 'numeric',
    month: 'long',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  }).format(d);
};

// Détermine le type de preview basé sur le MIME type
const getPreviewType = (mimeType: string, fileExtension: string): 'pdf' | 'image' | 'video' | 'audio' | 'text' | 'unsupported' => {
  if (mimeType === 'application/pdf') return 'pdf';
  if (mimeType.startsWith('image/')) return 'image';
  if (mimeType.startsWith('video/')) return 'video';
  if (mimeType.startsWith('audio/')) return 'audio';
  if (mimeType.startsWith('text/') || 
      mimeType.includes('document') || 
      mimeType.includes('word') ||
      ['.txt', '.md', '.csv', '.json', '.xml'].includes(fileExtension.toLowerCase())) {
    return 'text';
  }
  return 'unsupported';
};

// Composant de métadonnées éditables
const EditableDocumentMetadata: React.FC<{ 
  document: Document; 
  isEditing: boolean;
  editedMetadata: Partial<Document>;
  onMetadataChange: (metadata: Partial<Document>) => void;
  onSave: () => void;
  onCancel: () => void;
}> = ({ document, isEditing, editedMetadata, onMetadataChange, onSave, onCancel }) => {
  
  const handleFieldChange = (field: keyof Document, value: any) => {
    onMetadataChange({ ...editedMetadata, [field]: value });
  };

  const addTag = (tag: string) => {
    const currentTags = editedMetadata.tags || document.tags;
    if (tag.trim() && !currentTags.includes(tag.trim())) {
      handleFieldChange('tags', [...currentTags, tag.trim()]);
    }
  };

  const removeTag = (tagToRemove: string) => {
    const currentTags = editedMetadata.tags || document.tags;
    handleFieldChange('tags', currentTags.filter(tag => tag !== tagToRemove));
  };

  if (!isEditing) {
    return (
      <div className="space-y-6">
        {/* En-tête document */}
        <div className="flex items-start gap-4">
          <div className="flex-shrink-0">
            {getDocumentTypeIcon(document.document_type)}
          </div>
          <div className="flex-1 min-w-0">
            <h2 className="text-xl font-semibold truncate">
              {document.title || document.filename}
            </h2>
            {document.description && (
              <p className="text-gray-600 mt-1">{document.description}</p>
            )}
            <div className="flex items-center gap-2 mt-2">
              <Badge className={cn("flex items-center gap-1", getAccessLevelColor(document.access_level))}>
                {getAccessLevelIcon(document.access_level)}
                {document.access_level}
              </Badge>
              <Badge variant="outline">
                Version {document.version}
              </Badge>
              {document.is_latest_version && (
                <Badge variant="default" className="bg-green-600">
                  Dernière version
                </Badge>
              )}
            </div>
          </div>
        </div>

        {/* Tags */}
        {document.tags.length > 0 && (
          <div>
            <h4 className="font-medium mb-2">Tags</h4>
            <div className="flex flex-wrap gap-1">
              {document.tags.map(tag => (
                <Badge key={tag} variant="secondary" className="text-xs">
                  {tag}
                </Badge>
              ))}
            </div>
          </div>
        )}

        {/* Informations techniques */}
        <div className="grid grid-cols-2 gap-4 text-sm">
          <div>
            <h4 className="font-medium mb-2">Fichier</h4>
            <div className="space-y-1">
              <div className="flex justify-between">
                <span className="text-gray-600">Nom:</span>
                <span className="font-mono text-xs">{document.filename}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Taille:</span>
                <span>{formatFileSize(document.file_size)}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Type:</span>
                <span>{document.mime_type}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Extension:</span>
                <span>{document.file_extension}</span>
              </div>
            </div>
          </div>

          <div>
            <h4 className="font-medium mb-2">Métadonnées</h4>
            <div className="space-y-1">
              <div className="flex justify-between">
                <span className="text-gray-600">Créé le:</span>
                <span className="text-xs">{formatDate(document.created_at)}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Modifié le:</span>
                <span className="text-xs">{formatDate(document.updated_at)}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Consulté le:</span>
                <span className="text-xs">{formatDate(document.accessed_at)}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Propriétaire:</span>
                <span className="text-xs">{document.owner_id}</span>
              </div>
            </div>
          </div>
        </div>

        {/* Analytics */}
        <div>
          <h4 className="font-medium mb-2">Utilisation</h4>
          <div className="grid grid-cols-3 gap-4 text-sm">
            <div className="text-center p-2 bg-blue-50 rounded">
              <div className="text-lg font-bold text-blue-600">{document.view_count}</div>
              <div className="text-xs text-gray-600">Vues</div>
            </div>
            <div className="text-center p-2 bg-green-50 rounded">
              <div className="text-lg font-bold text-green-600">{document.download_count}</div>
              <div className="text-xs text-gray-600">Téléchargements</div>
            </div>
            <div className="text-center p-2 bg-purple-50 rounded">
              <div className="text-lg font-bold text-purple-600">{document.version}</div>
              <div className="text-xs text-gray-600">Version</div>
            </div>
          </div>
        </div>

        {/* Contexte M&A */}
        {(document.company_id || document.deal_id || document.project_phase) && (
          <div className="border-t pt-4">
            <h4 className="font-medium mb-2">Contexte M&A</h4>
            <div className="space-y-2 text-sm">
              {document.company_id && (
                <div className="flex justify-between">
                  <span className="text-gray-600">Entreprise:</span>
                  <span className="font-medium">{document.company_id}</span>
                </div>
              )}
              {document.deal_id && (
                <div className="flex justify-between">
                  <span className="text-gray-600">Deal:</span>
                  <span className="font-medium">{document.deal_id}</span>
                </div>
              )}
              {document.project_phase && (
                <div className="flex justify-between">
                  <span className="text-gray-600">Phase:</span>
                  <span className="font-medium">{document.project_phase}</span>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Contenu extrait */}
        {document.extracted_text && (
          <div className="border-t pt-4">
            <h4 className="font-medium mb-2">Contenu extrait (OCR)</h4>
            <div className="bg-gray-50 p-3 rounded-lg text-xs max-h-40 overflow-y-auto">
              <pre className="whitespace-pre-wrap font-mono">
                {document.extracted_text.substring(0, 800)}
                {document.extracted_text.length > 800 && '...'}
              </pre>
            </div>
          </div>
        )}
      </div>
    );
  }

  // Mode édition
  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="font-medium">Modifier les métadonnées</h3>
        <div className="flex gap-2">
          <Button variant="outline" size="sm" onClick={onCancel}>
            Annuler
          </Button>
          <Button size="sm" onClick={onSave}>
            <Save className="w-4 h-4 mr-1" />
            Sauvegarder
          </Button>
        </div>
      </div>

      {/* Titre */}
      <div>
        <label className="block text-sm font-medium mb-1">Titre</label>
        <Input
          value={editedMetadata.title ?? document.title ?? ''}
          onChange={(e) => handleFieldChange('title', e.target.value)}
          placeholder="Titre du document"
        />
      </div>

      {/* Description */}
      <div>
        <label className="block text-sm font-medium mb-1">Description</label>
        <Textarea
          value={editedMetadata.description ?? document.description ?? ''}
          onChange={(e) => handleFieldChange('description', e.target.value)}
          placeholder="Description du document"
          rows={3}
        />
      </div>

      {/* Tags */}
      <div>
        <label className="block text-sm font-medium mb-1">Tags</label>
        <div className="space-y-2">
          <Input
            placeholder="Ajouter un tag et appuyer sur Entrée"
            onKeyDown={(e) => {
              if (e.key === 'Enter') {
                e.preventDefault();
                addTag(e.currentTarget.value);
                e.currentTarget.value = '';
              }
            }}
          />
          {(editedMetadata.tags || document.tags).length > 0 && (
            <div className="flex flex-wrap gap-1">
              {(editedMetadata.tags || document.tags).map(tag => (
                <Badge
                  key={tag}
                  variant="secondary"
                  className="cursor-pointer hover:bg-red-100"
                  onClick={() => removeTag(tag)}
                >
                  {tag}
                  <X className="w-3 h-3 ml-1" />
                </Badge>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

// Composant LoadingSpinner
const LoadingSpinner: React.FC<{ message?: string }> = ({ message = 'Chargement...' }) => (
  <div className="flex flex-col items-center justify-center h-64 space-y-4">
    <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
    <p className="text-gray-500 text-sm">{message}</p>
  </div>
);

// Composant ErrorDisplay
const ErrorDisplay: React.FC<{ error: string; onRetry?: () => void }> = ({ error, onRetry }) => (
  <div className="flex flex-col items-center justify-center h-64 space-y-4">
    <AlertCircle className="h-12 w-12 text-red-500" />
    <div className="text-center">
      <p className="text-red-600 font-medium">Erreur de chargement</p>
      <p className="text-gray-500 text-sm mt-1">{error}</p>
    </div>
    {onRetry && (
      <Button variant="outline" onClick={onRetry}>
        Réessayer
      </Button>
    )}
  </div>
);

// Composant principal
export const ModernDocumentPreview: React.FC<ModernDocumentPreviewProps> = ({
  document,
  isOpen,
  onOpenChange,
  onDocumentUpdate,
  onDownload,
  onShare,
  enableEditing = true,
  enableAnnotations = false,
  enableAnalytics = true,
}) => {
  const [previewState, setPreviewState] = useState<PreviewState>({
    zoom: 100,
    rotation: 0,
    isFullscreen: false,
    currentPage: 1,
    isLoading: false,
    error: null,
    documentData: null,
    isEditing: false,
    editedMetadata: {},
  });

  const { indexContent, isIndexing } = useDocumentIndexing();

  // Charger le document
  const loadDocument = useCallback(async () => {
    if (!document) return;

    setPreviewState(prev => ({ ...prev, isLoading: true, error: null }));
    
    try {
      const { data } = await advancedDocumentService.retrieveDocument(document.document_id);
      setPreviewState(prev => ({ 
        ...prev, 
        documentData: data, 
        isLoading: false 
      }));
    } catch (error) {
      setPreviewState(prev => ({ 
        ...prev, 
        error: error instanceof Error ? error.message : 'Erreur de chargement',
        isLoading: false 
      }));
    }
  }, [document]);

  // Actions de contrôle
  const handleZoomIn = useCallback(() => {
    setPreviewState(prev => ({ 
      ...prev, 
      zoom: Math.min(prev.zoom + 25, 400) 
    }));
  }, []);

  const handleZoomOut = useCallback(() => {
    setPreviewState(prev => ({ 
      ...prev, 
      zoom: Math.max(prev.zoom - 25, 25) 
    }));
  }, []);

  const handleRotate = useCallback(() => {
    setPreviewState(prev => ({ 
      ...prev, 
      rotation: (prev.rotation + 90) % 360 
    }));
  }, []);

  const handleToggleFullscreen = useCallback(() => {
    setPreviewState(prev => ({ 
      ...prev, 
      isFullscreen: !prev.isFullscreen 
    }));
  }, []);

  const handleEdit = useCallback(() => {
    setPreviewState(prev => ({ 
      ...prev, 
      isEditing: !prev.isEditing,
      editedMetadata: prev.isEditing ? {} : {}
    }));
  }, []);

  const handleSaveMetadata = useCallback(async () => {
    if (!document || !Object.keys(previewState.editedMetadata).length) return;

    try {
      // TODO: Implémenter la sauvegarde via le service
      // const updatedDocument = await advancedDocumentService.updateDocument(document.document_id, previewState.editedMetadata);
      // onDocumentUpdate?.(updatedDocument);
      
      setPreviewState(prev => ({ 
        ...prev, 
        isEditing: false, 
        editedMetadata: {} 
      }));
    } catch (error) {
      console.error('Erreur sauvegarde:', error);
    }
  }, [document, previewState.editedMetadata, onDocumentUpdate]);

  const handleDownload = useCallback(() => {
    if (document && onDownload) {
      onDownload(document);
    }
  }, [document, onDownload]);

  const handleShare = useCallback(() => {
    if (document && onShare) {
      onShare(document);
    }
  }, [document, onShare]);

  const handleIndexContent = useCallback(() => {
    if (document?.extracted_text) {
      indexContent({
        documentId: document.document_id,
        extractedText: document.extracted_text
      });
    }
  }, [document, indexContent]);

  // Type de preview
  const previewType = useMemo(() => {
    if (!document) return 'unsupported';
    return getPreviewType(document.mime_type, document.file_extension);
  }, [document]);

  // Charger le document quand il change
  useEffect(() => {
    if (document && isOpen) {
      loadDocument();
    }
  }, [document, isOpen, loadDocument]);

  // Reset state quand on ferme
  useEffect(() => {
    if (!isOpen) {
      setPreviewState({
        zoom: 100,
        rotation: 0,
        isFullscreen: false,
        currentPage: 1,
        isLoading: false,
        error: null,
        documentData: null,
        isEditing: false,
        editedMetadata: {},
      });
    }
  }, [isOpen]);

  if (!document) return null;

  return (
    <Dialog 
      open={isOpen} 
      onOpenChange={onOpenChange}
    >
      <DialogContent 
        className={cn(
          "max-w-7xl max-h-[95vh] overflow-hidden",
          previewState.isFullscreen && "max-w-full max-h-full w-full h-full"
        )}
      >
        <DialogHeader className="flex-shrink-0">
          <div className="flex items-center justify-between">
            <DialogTitle className="flex items-center gap-2">
              {getDocumentTypeIcon(document.document_type)}
              <span className="truncate">{document.title || document.filename}</span>
              <Badge className={cn("flex items-center gap-1", getAccessLevelColor(document.access_level))}>
                {getAccessLevelIcon(document.access_level)}
                {document.access_level}
              </Badge>
            </DialogTitle>
            
            {/* Actions principales */}
            <div className="flex items-center gap-2">
              <Button variant="outline" size="icon" onClick={handleZoomOut}>
                <ZoomOut className="h-4 w-4" />
              </Button>
              <span className="text-sm min-w-12 text-center">{previewState.zoom}%</span>
              <Button variant="outline" size="icon" onClick={handleZoomIn}>
                <ZoomIn className="h-4 w-4" />
              </Button>
              
              <Button variant="outline" size="icon" onClick={handleRotate}>
                <RotateCw className="h-4 w-4" />
              </Button>
              
              <Button variant="outline" size="icon" onClick={handleToggleFullscreen}>
                {previewState.isFullscreen ? (
                  <Minimize2 className="h-4 w-4" />
                ) : (
                  <Maximize2 className="h-4 w-4" />
                )}
              </Button>
              
              {onShare && (
                <Button variant="outline" size="icon" onClick={handleShare}>
                  <Share2 className="h-4 w-4" />
                </Button>
              )}
              
              {enableEditing && document.canEdit && (
                <Button 
                  variant={previewState.isEditing ? "default" : "outline"} 
                  size="icon" 
                  onClick={handleEdit}
                >
                  <Edit className="h-4 w-4" />
                </Button>
              )}
              
              <Button variant="outline" size="icon" onClick={handleDownload}>
                <Download className="h-4 w-4" />
              </Button>
            </div>
          </div>
        </DialogHeader>

        <div className="flex-1 overflow-hidden">
          <Tabs defaultValue="preview" className="h-full flex flex-col">
            <TabsList className="flex-shrink-0">
              <TabsTrigger value="preview" className="flex items-center gap-2">
                <Eye className="h-4 w-4" />
                Aperçu
              </TabsTrigger>
              <TabsTrigger value="metadata" className="flex items-center gap-2">
                <FileText className="h-4 w-4" />
                Métadonnées
              </TabsTrigger>
              {enableAnalytics && (
                <TabsTrigger value="analytics" className="flex items-center gap-2">
                  <History className="h-4 w-4" />
                  Analytics
                </TabsTrigger>
              )}
              {enableAnnotations && (
                <TabsTrigger value="annotations" className="flex items-center gap-2">
                  <MessageSquare className="h-4 w-4" />
                  Annotations
                </TabsTrigger>
              )}
            </TabsList>

            <TabsContent value="preview" className="flex-1 overflow-hidden mt-4">
              <div className="h-full border rounded-lg overflow-hidden bg-gray-50">
                {previewState.isLoading && (
                  <LoadingSpinner message="Chargement du document..." />
                )}
                
                {previewState.error && (
                  <ErrorDisplay 
                    error={previewState.error} 
                    onRetry={loadDocument}
                  />
                )}
                
                {previewState.documentData && !previewState.isLoading && !previewState.error && (
                  <Suspense fallback={<LoadingSpinner message="Initialisation du viewer..." />}>
                    <div 
                      className="h-full w-full"
                      style={{
                        transform: `scale(${previewState.zoom / 100}) rotate(${previewState.rotation}deg)`,
                        transformOrigin: 'center center',
                      }}
                    >
                      {previewType === 'pdf' && (
                        <PDFViewer 
                          data={previewState.documentData}
                          currentPage={previewState.currentPage}
                          onPageChange={(page) => 
                            setPreviewState(prev => ({ ...prev, currentPage: page }))
                          }
                        />
                      )}
                      
                      {previewType === 'image' && (
                        <ImageViewer 
                          data={previewState.documentData}
                          mimeType={document.mime_type}
                        />
                      )}
                      
                      {previewType === 'video' && (
                        <VideoViewer 
                          data={previewState.documentData}
                          mimeType={document.mime_type}
                        />
                      )}
                      
                      {previewType === 'text' && (
                        <TextViewer 
                          data={previewState.documentData}
                          mimeType={document.mime_type}
                          filename={document.filename}
                        />
                      )}
                      
                      {previewType === 'unsupported' && (
                        <div className="flex flex-col items-center justify-center h-full space-y-4">
                          <Archive className="h-16 w-16 text-gray-400" />
                          <div className="text-center">
                            <p className="font-medium text-gray-600">Aperçu non disponible</p>
                            <p className="text-sm text-gray-500 mt-1">
                              Type de fichier: {document.mime_type}
                            </p>
                            <Button 
                              variant="outline" 
                              onClick={handleDownload}
                              className="mt-4"
                            >
                              <Download className="h-4 w-4 mr-2" />
                              Télécharger pour voir
                            </Button>
                          </div>
                        </div>
                      )}
                    </div>
                  </Suspense>
                )}
              </div>
            </TabsContent>

            <TabsContent value="metadata" className="flex-1 overflow-auto">
              <EditableDocumentMetadata
                document={document}
                isEditing={previewState.isEditing}
                editedMetadata={previewState.editedMetadata}
                onMetadataChange={(metadata) => 
                  setPreviewState(prev => ({ 
                    ...prev, 
                    editedMetadata: { ...prev.editedMetadata, ...metadata }
                  }))
                }
                onSave={handleSaveMetadata}
                onCancel={() => setPreviewState(prev => ({ 
                  ...prev, 
                  isEditing: false, 
                  editedMetadata: {} 
                }))}
              />
              
              {/* Action d'indexation */}
              {document.extracted_text && (
                <div className="border-t pt-4 mt-4">
                  <Button 
                    variant="outline" 
                    onClick={handleIndexContent}
                    disabled={isIndexing}
                    className="w-full"
                  >
                    {isIndexing ? (
                      <>Indexation en cours...</>
                    ) : (
                      <>
                        <Search className="w-4 h-4 mr-2" />
                        Indexer pour recherche sémantique
                      </>
                    )}
                  </Button>
                </div>
              )}
            </TabsContent>

            {enableAnalytics && (
              <TabsContent value="analytics" className="flex-1 overflow-auto">
                <div className="space-y-6">
                  <div>
                    <h3 className="font-medium mb-4">Statistiques d'utilisation</h3>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                      <div className="bg-blue-50 p-4 rounded-lg text-center">
                        <div className="text-2xl font-bold text-blue-600">{document.view_count}</div>
                        <div className="text-sm text-gray-600">Vues totales</div>
                      </div>
                      <div className="bg-green-50 p-4 rounded-lg text-center">
                        <div className="text-2xl font-bold text-green-600">{document.download_count}</div>
                        <div className="text-sm text-gray-600">Téléchargements</div>
                      </div>
                      <div className="bg-purple-50 p-4 rounded-lg text-center">
                        <div className="text-2xl font-bold text-purple-600">{document.version}</div>
                        <div className="text-sm text-gray-600">Version</div>
                      </div>
                      <div className="bg-orange-50 p-4 rounded-lg text-center">
                        <div className="text-2xl font-bold text-orange-600">
                          {Math.round((new Date().getTime() - new Date(document.created_at).getTime()) / (1000 * 60 * 60 * 24))}
                        </div>
                        <div className="text-sm text-gray-600">Jours depuis création</div>
                      </div>
                    </div>
                  </div>

                  <div>
                    <h4 className="font-medium mb-2">Informations techniques</h4>
                    <div className="bg-gray-50 p-4 rounded-lg space-y-2 text-sm font-mono">
                      <div><strong>ID:</strong> {document.document_id}</div>
                      <div><strong>MD5:</strong> {document.md5_hash}</div>
                      <div><strong>SHA256:</strong> {document.sha256_hash.substring(0, 32)}...</div>
                      <div><strong>Stockage:</strong> {document.storage_backend}</div>
                      <div><strong>Chemin:</strong> {document.storage_path}</div>
                    </div>
                  </div>
                </div>
              </TabsContent>
            )}

            {enableAnnotations && (
              <TabsContent value="annotations" className="flex-1 overflow-auto">
                <div className="text-center text-gray-500 py-8">
                  <MessageSquare className="h-12 w-12 mx-auto mb-4 text-gray-300" />
                  <p>Fonctionnalité d'annotations à venir</p>
                  <p className="text-sm mt-2">
                    Permettra d'ajouter des commentaires et annotations directement sur le document.
                  </p>
                </div>
              </TabsContent>
            )}
          </Tabs>
        </div>
      </DialogContent>
    </Dialog>
  );
};