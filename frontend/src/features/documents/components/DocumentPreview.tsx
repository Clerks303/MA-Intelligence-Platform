/**
 * Prévisualisation Multi-Formats - M&A Intelligence Platform
 * Sprint 3 - Preview PDF, Images, Documents avec annotations et backend intégré
 */

import React, { useState, useCallback, useMemo, lazy, Suspense } from 'react';
import { Button } from '../../../components/ui/button';
import { Card, CardHeader, CardContent } from '../../../components/ui/card';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '../../../components/ui/dialog';
import { Badge } from '../../../components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../../../components/ui/tabs';
import { Progress } from '../../../components/ui/progress';
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
  X
} from 'lucide-react';
import { cn } from '../../../lib/utils';

import { Document, BackendDocumentType } from '../types';
import { advancedDocumentService } from '../services/advancedDocumentService';

// Lazy loading des composants de preview
const PDFPreview = lazy(() => import('./previews/PDFPreview'));
const ImagePreview = lazy(() => import('./previews/ImagePreview'));
const TextPreview = lazy(() => import('./previews/TextPreview'));
const VideoPreview = lazy(() => import('./previews/VideoPreview'));

interface DocumentPreviewProps {
  documentId: string;
  isOpen: boolean;
  onClose: () => void;
  config?: Partial<PreviewConfig>;
}

interface PreviewToolbarProps {
  document: Document;
  config: PreviewConfig;
  onConfigChange: (config: Partial<PreviewConfig>) => void;
  onDownload: () => void;
  onFullscreen: () => void;
}

interface PreviewContentProps {
  document: Document;
  config: PreviewConfig;
}

// Configuration par défaut
const DEFAULT_PREVIEW_CONFIG: PreviewConfig = {
  showThumbnails: true,
  enableZoom: true,
  enableRotation: true,
  enableAnnotations: false,
  showMetadata: true,
  autoPlay: false,
};

// Utilitaires
const formatFileSize = (bytes: number): string => {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(1))} ${sizes[i]}`;
};

const formatDate = (date: Date): string => {
  return new Intl.DateTimeFormat('fr-FR', {
    year: 'numeric',
    month: 'long',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  }).format(date);
};

// Toolbar de contrôles
const PreviewToolbar: React.FC<PreviewToolbarProps> = ({
  document,
  config,
  onConfigChange,
  onDownload,
  onFullscreen,
}) => {
  const toggleConfig = (key: keyof PreviewConfig) => {
    onConfigChange({ [key]: !config[key] });
  };

  return (
    <div className="flex items-center justify-between p-3 border-b border-ma-slate-200 bg-ma-slate-50">
      {/* Infos document */}
      <div className="flex items-center gap-3">
        <div className="flex-shrink-0">
          {document.type === 'pdf' && (
            <svg className="w-6 h-6 text-ma-red-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
          )}
          {document.type === 'image' && (
            <svg className="w-6 h-6 text-ma-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
            </svg>
          )}
          {document.type === 'document' && (
            <svg className="w-6 h-6 text-ma-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
          )}
        </div>
        
        <div>
          <h3 className="text-sm font-medium text-ma-slate-900 truncate max-w-xs">
            {document.name}
          </h3>
          <p className="text-xs text-ma-slate-500">
            {formatFileSize(document.size)} • {document.extension.toUpperCase()}
          </p>
        </div>
      </div>

      {/* Contrôles */}
      <div className="flex items-center gap-1">
        {/* Zoom */}
        {config.enableZoom && (
          <Button
            variant="ghost"
            size="icon"
            className="h-8 w-8"
            title="Zoom"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0zM10 7v3m0 0v3m0-3h3m-3 0H7" />
            </svg>
          </Button>
        )}

        {/* Rotation */}
        {config.enableRotation && document.type === 'image' && (
          <Button
            variant="ghost"
            size="icon"
            className="h-8 w-8"
            title="Rotation"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
            </svg>
          </Button>
        )}

        {/* Métadonnées */}
        <Button
          variant={config.showMetadata ? "default" : "ghost"}
          size="icon"
          className="h-8 w-8"
          onClick={() => toggleConfig('showMetadata')}
          title="Métadonnées"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
        </Button>

        {/* Plein écran */}
        <Button
          variant="ghost"
          size="icon"
          className="h-8 w-8"
          onClick={onFullscreen}
          title="Plein écran"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 8V4m0 0h4M4 4l5 5m11-1V4m0 0h-4m4 0l-5 5M4 16v4m0 0h4m-4 0l5-5m11 5l-5-5m5 5v-4m0 4h-4" />
          </svg>
        </Button>

        {/* Télécharger */}
        <Button
          variant="ghost"
          size="icon"
          className="h-8 w-8"
          onClick={onDownload}
          title="Télécharger"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
          </svg>
        </Button>
      </div>
    </div>
  );
};

// Contenu de preview selon le type
const PreviewContent: React.FC<PreviewContentProps> = ({ document, config }) => {
  // Loading fallback
  const LoadingPreview = () => (
    <div className="flex items-center justify-center h-96">
      <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-ma-blue-600"></div>
    </div>
  );

  // Erreur de preview
  const ErrorPreview = ({ message }: { message: string }) => (
    <div className="flex flex-col items-center justify-center h-96 text-ma-slate-500">
      <svg className="w-16 h-16 mb-4 text-ma-slate-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
      </svg>
      <p className="text-center">{message}</p>
      <Button
        variant="outline"
        size="sm"
        className="mt-4"
        onClick={() => window.open(document.downloadUrl, '_blank')}
      >
        Télécharger le fichier
      </Button>
    </div>
  );

  // Preview non supporté
  const UnsupportedPreview = () => (
    <div className="flex flex-col items-center justify-center h-96 text-ma-slate-500">
      <svg className="w-16 h-16 mb-4 text-ma-slate-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
      </svg>
      <p className="text-center mb-2">Aperçu non disponible pour ce type de fichier</p>
      <p className="text-sm text-center text-ma-slate-400 mb-4">
        Type: {document.mimeType} • Taille: {formatFileSize(document.size)}
      </p>
      <Button
        variant="ma"
        size="sm"
        onClick={() => window.open(document.downloadUrl, '_blank')}
      >
        Télécharger le fichier
      </Button>
    </div>
  );

  // Sélection du composant de preview
  switch (document.type) {
    case 'pdf':
      return (
        <Suspense fallback={<LoadingPreview />}>
          <PDFPreview document={document} config={config} />
        </Suspense>
      );
      
    case 'image':
      return (
        <Suspense fallback={<LoadingPreview />}>
          <ImagePreview document={document} config={config} />
        </Suspense>
      );
      
    case 'document':
    case 'spreadsheet':
      if (document.extractedText) {
        return (
          <Suspense fallback={<LoadingPreview />}>
            <TextPreview document={document} config={config} />
          </Suspense>
        );
      }
      return <UnsupportedPreview />;
      
    case 'video':
      return (
        <Suspense fallback={<LoadingPreview />}>
          <VideoPreview document={document} config={config} />
        </Suspense>
      );
      
    default:
      return <UnsupportedPreview />;
  }
};

// Sidebar métadonnées
const MetadataSidebar: React.FC<{ document: Document }> = ({ document }) => {
  return (
    <Card className="h-full">
      <CardHeader className="pb-3">
        <CardTitle className="text-base">Métadonnées</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4 text-sm">
        {/* Informations de base */}
        <div>
          <h4 className="font-medium text-ma-slate-900 mb-2">Informations</h4>
          <div className="space-y-1">
            <div className="flex justify-between">
              <span className="text-ma-slate-600">Nom:</span>
              <span className="font-medium">{document.name}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-ma-slate-600">Taille:</span>
              <span>{formatFileSize(document.size)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-ma-slate-600">Type:</span>
              <span>{document.mimeType}</span>
            </div>
            {document.pageCount && (
              <div className="flex justify-between">
                <span className="text-ma-slate-600">Pages:</span>
                <span>{document.pageCount}</span>
              </div>
            )}
            {document.dimensions && (
              <div className="flex justify-between">
                <span className="text-ma-slate-600">Dimensions:</span>
                <span>{document.dimensions.width}×{document.dimensions.height}</span>
              </div>
            )}
          </div>
        </div>

        {/* Dates */}
        <div>
          <h4 className="font-medium text-ma-slate-900 mb-2">Dates</h4>
          <div className="space-y-1">
            <div>
              <span className="text-ma-slate-600">Créé:</span>
              <p className="text-xs">{formatDate(document.createdAt)}</p>
            </div>
            <div>
              <span className="text-ma-slate-600">Modifié:</span>
              <p className="text-xs">{formatDate(document.updatedAt)}</p>
            </div>
          </div>
        </div>

        {/* Utilisateur */}
        <div>
          <h4 className="font-medium text-ma-slate-900 mb-2">Utilisateur</h4>
          <div className="flex items-center gap-2">
            {document.uploadedBy.avatar ? (
              <img 
                src={document.uploadedBy.avatar} 
                alt={document.uploadedBy.name}
                className="w-6 h-6 rounded-full"
              />
            ) : (
              <div className="w-6 h-6 rounded-full bg-ma-slate-200 flex items-center justify-center">
                <span className="text-xs font-medium text-ma-slate-600">
                  {document.uploadedBy.name.charAt(0).toUpperCase()}
                </span>
              </div>
            )}
            <div>
              <p className="font-medium">{document.uploadedBy.name}</p>
              <p className="text-xs text-ma-slate-500">{document.uploadedBy.email}</p>
            </div>
          </div>
        </div>

        {/* Tags */}
        {document.tags.length > 0 && (
          <div>
            <h4 className="font-medium text-ma-slate-900 mb-2">Tags</h4>
            <div className="flex flex-wrap gap-1">
              {document.tags.map(tag => (
                <span 
                  key={tag}
                  className="inline-flex items-center px-2 py-1 rounded text-xs bg-ma-slate-100 text-ma-slate-700"
                >
                  {tag}
                </span>
              ))}
            </div>
          </div>
        )}

        {/* Description */}
        {document.description && (
          <div>
            <h4 className="font-medium text-ma-slate-900 mb-2">Description</h4>
            <p className="text-ma-slate-600 text-sm">{document.description}</p>
          </div>
        )}

        {/* Statut traitement */}
        <div>
          <h4 className="font-medium text-ma-slate-900 mb-2">Traitement</h4>
          <div className="space-y-1">
            <div className="flex justify-between">
              <span className="text-ma-slate-600">Status:</span>
              <span className={cn(
                "text-xs px-2 py-1 rounded",
                document.processingStatus === 'completed' ? "bg-ma-green-100 text-ma-green-700" :
                document.processingStatus === 'failed' ? "bg-ma-red-100 text-ma-red-700" :
                "bg-ma-blue-100 text-ma-blue-700"
              )}>
                {document.processingStatus}
              </span>
            </div>
            {document.ocrStatus && (
              <div className="flex justify-between">
                <span className="text-ma-slate-600">OCR:</span>
                <span className="text-xs">{document.ocrStatus}</span>
              </div>
            )}
            {document.virusScanStatus && (
              <div className="flex justify-between">
                <span className="text-ma-slate-600">Antivirus:</span>
                <span className="text-xs">{document.virusScanStatus}</span>
              </div>
            )}
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

// Composant principal
export const DocumentPreview: React.FC<DocumentPreviewProps> = ({
  documentId,
  isOpen,
  onClose,
  config: configOverride = {},
}) => {
  const [config, setConfig] = useState<PreviewConfig>({
    ...DEFAULT_PREVIEW_CONFIG,
    ...configOverride,
  });
  const [isFullscreen, setIsFullscreen] = useState(false);

  const { document, isLoading, error } = useDocument(documentId, { enabled: isOpen });

  const handleConfigChange = useCallback((newConfig: Partial<PreviewConfig>) => {
    setConfig(prev => ({ ...prev, ...newConfig }));
  }, []);

  const handleDownload = useCallback(() => {
    if (document) {
      window.open(document.downloadUrl, '_blank');
    }
  }, [document]);

  const handleFullscreen = useCallback(() => {
    setIsFullscreen(!isFullscreen);
  }, [isFullscreen]);

  if (!isOpen) return null;

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className={cn(
        "max-w-6xl max-h-[90vh] p-0",
        isFullscreen && "max-w-[95vw] max-h-[95vh]"
      )}>
        {isLoading && (
          <div className="flex items-center justify-center h-96">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-ma-blue-600"></div>
          </div>
        )}

        {error && (
          <div className="flex flex-col items-center justify-center h-96 text-ma-red-500">
            <svg className="w-16 h-16 mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <p>Erreur de chargement du document</p>
          </div>
        )}

        {document && (
          <div className="flex flex-col h-full">
            {/* Toolbar */}
            <PreviewToolbar
              document={document}
              config={config}
              onConfigChange={handleConfigChange}
              onDownload={handleDownload}
              onFullscreen={handleFullscreen}
            />

            {/* Contenu principal */}
            <div className="flex flex-1 overflow-hidden">
              {/* Preview */}
              <div className="flex-1 overflow-auto">
                <PreviewContent document={document} config={config} />
              </div>

              {/* Sidebar métadonnées */}
              {config.showMetadata && (
                <div className="w-80 border-l border-ma-slate-200 overflow-y-auto">
                  <MetadataSidebar document={document} />
                </div>
              )}
            </div>
          </div>
        )}
      </DialogContent>
    </Dialog>
  );
};