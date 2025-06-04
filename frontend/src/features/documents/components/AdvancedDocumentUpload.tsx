/**
 * Composant Upload Avancé Documents - M&A Intelligence Platform
 * Sprint 3 - Integration backend complet avec analytics et performance
 */

import React, { useCallback, useState, useMemo } from 'react';
import { useDropzone } from 'react-dropzone';
import { Button } from '../../../components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '../../../components/ui/card';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '../../../components/ui/dialog';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '../../../components/ui/select';
import { Input } from '../../../components/ui/input';
import { Label } from '../../../components/ui/label';
import { Textarea } from '../../../components/ui/textarea';
import { Badge } from '../../../components/ui/badge';
import { Progress } from '../../../components/ui/progress';
import { AlertCircle, Upload, FileCheck, X, RotateCcw, Settings } from 'lucide-react';
import { cn } from '../../../lib/utils';

import { 
  BackendDocumentType, 
  BackendAccessLevel,
  Document
} from '../types';
import { useAdvancedUpload } from '../hooks/useAdvancedDocuments';
import { DOCUMENT_CONFIG } from '../services/advancedDocumentService';

interface AdvancedDocumentUploadProps {
  isOpen: boolean;
  onOpenChange: (open: boolean) => void;
  onUploadComplete?: (documents: Document[]) => void;
  defaultDocumentType?: BackendDocumentType;
  companyId?: string;
  dealId?: string;
  projectPhase?: string;
}

interface UploadMetadataFormProps {
  onSubmit: (metadata: UploadMetadata) => void;
  onCancel: () => void;
  defaultType?: BackendDocumentType;
  companyId?: string;
  dealId?: string;
  projectPhase?: string;
}

interface UploadMetadata {
  documentType: BackendDocumentType;
  title: string;
  description: string;
  tags: string[];
  accessLevel: BackendAccessLevel;
  companyId?: string;
  dealId?: string;
  projectPhase?: string;
}

// Utilitaires
const formatFileSize = (bytes: number): string => {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(1))} ${sizes[i]}`;
};

const getFileTypeIcon = (type: string) => {
  if (type.startsWith('image/')) return '🖼️';
  if (type === 'application/pdf') return '📄';
  if (type.includes('word') || type.includes('document')) return '📝';
  if (type.includes('excel') || type.includes('spreadsheet')) return '📊';
  if (type.includes('powerpoint') || type.includes('presentation')) return '📈';
  return '📁';
};

const documentTypeLabels: Record<BackendDocumentType, string> = {
  financial: 'Financier',
  legal: 'Juridique',
  due_diligence: 'Due Diligence',
  communication: 'Communication',
  technical: 'Technique',
  hr: 'Ressources Humaines',
  commercial: 'Commercial',
  other: 'Autre',
};

const accessLevelLabels: Record<BackendAccessLevel, { label: string; color: string }> = {
  public: { label: 'Public', color: 'bg-green-100 text-green-800' },
  internal: { label: 'Interne', color: 'bg-blue-100 text-blue-800' },
  confidential: { label: 'Confidentiel', color: 'bg-orange-100 text-orange-800' },
  restricted: { label: 'Restreint', color: 'bg-red-100 text-red-800' },
};

// Formulaire de métadonnées
const UploadMetadataForm: React.FC<UploadMetadataFormProps> = ({
  onSubmit,
  onCancel,
  defaultType = 'other',
  companyId,
  dealId,
  projectPhase,
}) => {
  const [metadata, setMetadata] = useState<UploadMetadata>({
    documentType: defaultType,
    title: '',
    description: '',
    tags: [],
    accessLevel: 'internal',
    companyId,
    dealId,
    projectPhase,
  });

  const [tagInput, setTagInput] = useState('');

  const addTag = useCallback((tag: string) => {
    const trimmedTag = tag.trim();
    if (trimmedTag && !metadata.tags.includes(trimmedTag)) {
      setMetadata(prev => ({
        ...prev,
        tags: [...prev.tags, trimmedTag],
      }));
    }
    setTagInput('');
  }, [metadata.tags]);

  const removeTag = useCallback((tagToRemove: string) => {
    setMetadata(prev => ({
      ...prev,
      tags: prev.tags.filter(tag => tag !== tagToRemove),
    }));
  }, []);

  const handleSubmit = useCallback((e: React.FormEvent) => {
    e.preventDefault();
    onSubmit(metadata);
  }, [metadata, onSubmit]);

  return (
    <form onSubmit={handleSubmit} className="space-y-6">
      {/* Type de document */}
      <div className="space-y-2">
        <Label htmlFor="documentType">Type de document *</Label>
        <Select
          value={metadata.documentType}
          onValueChange={(value: BackendDocumentType) => 
            setMetadata(prev => ({ ...prev, documentType: value }))
          }
        >
          <SelectTrigger>
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {Object.entries(documentTypeLabels).map(([value, label]) => (
              <SelectItem key={value} value={value}>
                {label}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      {/* Titre */}
      <div className="space-y-2">
        <Label htmlFor="title">Titre</Label>
        <Input
          id="title"
          value={metadata.title}
          onChange={(e) => setMetadata(prev => ({ ...prev, title: e.target.value }))}
          placeholder="Titre du document (optionnel)"
        />
      </div>

      {/* Description */}
      <div className="space-y-2">
        <Label htmlFor="description">Description</Label>
        <Textarea
          id="description"
          value={metadata.description}
          onChange={(e) => setMetadata(prev => ({ ...prev, description: e.target.value }))}
          placeholder="Description du document (optionnel)"
          rows={3}
        />
      </div>

      {/* Tags */}
      <div className="space-y-2">
        <Label htmlFor="tags">Tags</Label>
        <div className="space-y-2">
          <Input
            id="tags"
            value={tagInput}
            onChange={(e) => setTagInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter') {
                e.preventDefault();
                addTag(tagInput);
              }
            }}
            placeholder="Ajouter un tag et appuyer sur Entrée"
          />
          {metadata.tags.length > 0 && (
            <div className="flex flex-wrap gap-1">
              {metadata.tags.map(tag => (
                <Badge
                  key={tag}
                  variant="secondary"
                  className="cursor-pointer"
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

      {/* Niveau d'accès */}
      <div className="space-y-2">
        <Label htmlFor="accessLevel">Niveau d'accès *</Label>
        <Select
          value={metadata.accessLevel}
          onValueChange={(value: BackendAccessLevel) => 
            setMetadata(prev => ({ ...prev, accessLevel: value }))
          }
        >
          <SelectTrigger>
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {Object.entries(accessLevelLabels).map(([value, { label }]) => (
              <SelectItem key={value} value={value}>
                {label}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      {/* Contexte M&A */}
      {(companyId || dealId || projectPhase) && (
        <div className="p-4 bg-gray-50 rounded-lg space-y-2">
          <h4 className="font-medium text-sm text-gray-900">Contexte M&A</h4>
          {companyId && (
            <p className="text-sm text-gray-600">Entreprise: {companyId}</p>
          )}
          {dealId && (
            <p className="text-sm text-gray-600">Deal: {dealId}</p>
          )}
          {projectPhase && (
            <p className="text-sm text-gray-600">Phase: {projectPhase}</p>
          )}
        </div>
      )}

      {/* Actions */}
      <div className="flex justify-end gap-3">
        <Button type="button" variant="outline" onClick={onCancel}>
          Annuler
        </Button>
        <Button type="submit" disabled={!metadata.documentType}>
          Continuer l'upload
        </Button>
      </div>
    </form>
  );
};

// Composant principal
export const AdvancedDocumentUpload: React.FC<AdvancedDocumentUploadProps> = ({
  isOpen,
  onOpenChange,
  onUploadComplete,
  defaultDocumentType,
  companyId,
  dealId,
  projectPhase,
}) => {
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [step, setStep] = useState<'select' | 'metadata' | 'uploading'>('select');
  const [uploadMetadata, setUploadMetadata] = useState<UploadMetadata | null>(null);

  const { uploadFiles, uploadProgress, uploadErrors, isUploading, validateFiles } = useAdvancedUpload();

  // Validation et sélection de fichiers
  const onDrop = useCallback((acceptedFiles: File[]) => {
    const validationErrors = validateFiles(acceptedFiles);
    if (validationErrors.length > 0) {
      // Afficher les erreurs (ici on pourrait utiliser un toast)
      console.error('Validation errors:', validationErrors);
      return;
    }

    setSelectedFiles(acceptedFiles);
    setStep('metadata');
  }, [validateFiles]);

  const { getRootProps, getInputProps, isDragActive, isDragReject } = useDropzone({
    onDrop,
    accept: DOCUMENT_CONFIG.UPLOAD.ALLOWED_TYPES.reduce((acc, type) => {
      acc[type] = [];
      return acc;
    }, {} as Record<string, string[]>),
    maxSize: DOCUMENT_CONFIG.UPLOAD.MAX_FILE_SIZE,
    maxFiles: DOCUMENT_CONFIG.UPLOAD.MAX_CONCURRENT_UPLOADS,
  });

  // Gérer la soumission des métadonnées
  const handleMetadataSubmit = useCallback(async (metadata: UploadMetadata) => {
    setUploadMetadata(metadata);
    setStep('uploading');

    try {
      const uploadedDocuments = await uploadFiles(
        selectedFiles,
        metadata.documentType,
        {
          title: metadata.title,
          description: metadata.description,
          tags: metadata.tags,
          accessLevel: metadata.accessLevel,
          companyId: metadata.companyId,
          dealId: metadata.dealId,
          projectPhase: metadata.projectPhase,
        }
      );

      // Succès
      onUploadComplete?.(uploadedDocuments);
      handleClose();
    } catch (error) {
      console.error('Upload failed:', error);
      // Gérer l'erreur (toast, etc.)
    }
  }, [selectedFiles, uploadFiles, onUploadComplete]);

  // Fermer et reset
  const handleClose = useCallback(() => {
    setSelectedFiles([]);
    setStep('select');
    setUploadMetadata(null);
    onOpenChange(false);
  }, [onOpenChange]);

  // Calculs de progression
  const progressStats = useMemo(() => {
    const totalFiles = selectedFiles.length;
    const progressValues = Object.values(uploadProgress);
    const averageProgress = progressValues.length > 0 
      ? progressValues.reduce((sum, progress) => sum + progress, 0) / progressValues.length
      : 0;
    
    const errorCount = Object.keys(uploadErrors).length;
    
    return {
      totalFiles,
      averageProgress,
      errorCount,
      isComplete: averageProgress >= 100 && errorCount === 0,
    };
  }, [selectedFiles.length, uploadProgress, uploadErrors]);

  return (
    <Dialog open={isOpen} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-4xl max-h-[90vh] overflow-hidden">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Upload className="w-5 h-5" />
            Upload de documents M&A
            {step === 'metadata' && ` (${selectedFiles.length} fichier${selectedFiles.length > 1 ? 's' : ''})`}
            {step === 'uploading' && ` - En cours...`}
          </DialogTitle>
        </DialogHeader>

        {/* Étape 1: Sélection des fichiers */}
        {step === 'select' && (
          <div className="space-y-6">
            {/* Zone de drop */}
            <div
              {...getRootProps()}
              className={cn(
                "border-2 border-dashed rounded-lg p-12 text-center cursor-pointer transition-all",
                isDragActive && !isDragReject && "border-blue-500 bg-blue-50 scale-105",
                isDragReject && "border-red-500 bg-red-50",
                !isDragActive && "border-gray-300 hover:border-blue-400 hover:bg-gray-50"
              )}
            >
              <input {...getInputProps()} />
              
              <div className="space-y-4">
                <div className="mx-auto w-16 h-16 flex items-center justify-center">
                  {isDragActive ? (
                    <Upload className="w-12 h-12 text-blue-500 animate-bounce" />
                  ) : (
                    <Upload className="w-12 h-12 text-gray-400" />
                  )}
                </div>

                <div>
                  {isDragReject ? (
                    <p className="text-red-600 font-medium">
                      Fichiers non supportés
                    </p>
                  ) : isDragActive ? (
                    <p className="text-blue-600 font-medium">
                      Relâcher pour sélectionner les fichiers
                    </p>
                  ) : (
                    <>
                      <p className="text-gray-900 font-medium text-lg">
                        Glissez-déposez vos documents ici
                      </p>
                      <p className="text-gray-500 mt-2">
                        ou cliquez pour sélectionner depuis votre ordinateur
                      </p>
                    </>
                  )}
                </div>

                {/* Informations de configuration */}
                <div className="text-sm text-gray-500 bg-gray-50 p-4 rounded">
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <strong>Formats acceptés:</strong>
                      <br />PDF, Images, Documents Office, Texte
                    </div>
                    <div>
                      <strong>Limites:</strong>
                      <br />Max {formatFileSize(DOCUMENT_CONFIG.UPLOAD.MAX_FILE_SIZE)} par fichier
                      <br />Max {DOCUMENT_CONFIG.UPLOAD.MAX_CONCURRENT_UPLOADS} fichiers simultanés
                    </div>
                  </div>
                </div>
              </div>
            </div>

            <div className="flex justify-end">
              <Button variant="outline" onClick={handleClose}>
                Annuler
              </Button>
            </div>
          </div>
        )}

        {/* Étape 2: Métadonnées */}
        {step === 'metadata' && (
          <div className="space-y-6">
            {/* Liste des fichiers sélectionnés */}
            <div className="border rounded-lg p-4 bg-gray-50">
              <h3 className="font-medium mb-3">Fichiers sélectionnés ({selectedFiles.length})</h3>
              <div className="space-y-2 max-h-40 overflow-y-auto">
                {selectedFiles.map((file, index) => (
                  <div key={index} className="flex items-center gap-3 p-2 bg-white rounded border">
                    <span className="text-lg">{getFileTypeIcon(file.type)}</span>
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-medium truncate">{file.name}</p>
                      <p className="text-xs text-gray-500">{formatFileSize(file.size)}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Formulaire de métadonnées */}
            <UploadMetadataForm
              onSubmit={handleMetadataSubmit}
              onCancel={() => setStep('select')}
              defaultType={defaultDocumentType}
              companyId={companyId}
              dealId={dealId}
              projectPhase={projectPhase}
            />
          </div>
        )}

        {/* Étape 3: Upload en cours */}
        {step === 'uploading' && (
          <div className="space-y-6">
            {/* Progression globale */}
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <h3 className="font-medium">Upload en cours...</h3>
                <span className="text-sm text-gray-500">
                  {Math.round(progressStats.averageProgress)}%
                </span>
              </div>
              <Progress value={progressStats.averageProgress} className="w-full" />
              
              {progressStats.errorCount > 0 && (
                <div className="flex items-center gap-2 text-red-600 text-sm">
                  <AlertCircle className="w-4 h-4" />
                  {progressStats.errorCount} fichier{progressStats.errorCount > 1 ? 's' : ''} en erreur
                </div>
              )}
            </div>

            {/* Détail par fichier */}
            <div className="space-y-2 max-h-60 overflow-y-auto">
              {selectedFiles.map((file, index) => {
                const progress = uploadProgress[file.name] || 0;
                const error = uploadErrors[file.name];
                
                return (
                  <div key={index} className="flex items-center gap-3 p-3 border rounded">
                    <span className="text-lg">{getFileTypeIcon(file.type)}</span>
                    
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center justify-between mb-1">
                        <p className="text-sm font-medium truncate">{file.name}</p>
                        <div className="flex items-center gap-2">
                          {error ? (
                            <AlertCircle className="w-4 h-4 text-red-500" />
                          ) : progress >= 100 ? (
                            <FileCheck className="w-4 h-4 text-green-500" />
                          ) : (
                            <div className="w-4 h-4 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
                          )}
                          <span className="text-xs text-gray-500">
                            {error ? 'Erreur' : progress >= 100 ? 'Terminé' : `${Math.round(progress)}%`}
                          </span>
                        </div>
                      </div>
                      
                      {!error && (
                        <Progress value={progress} className="w-full h-2" />
                      )}
                      
                      {error && (
                        <p className="text-xs text-red-600 mt-1">{error}</p>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>

            {/* Actions */}
            <div className="flex justify-end gap-3">
              {progressStats.isComplete ? (
                <Button onClick={handleClose} className="bg-green-600 hover:bg-green-700">
                  <FileCheck className="w-4 h-4 mr-2" />
                  Terminé
                </Button>
              ) : (
                <Button variant="outline" onClick={handleClose} disabled={isUploading}>
                  Fermer
                </Button>
              )}
            </div>
          </div>
        )}
      </DialogContent>
    </Dialog>
  );
};