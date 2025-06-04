/**
 * Composant Upload Documents - M&A Intelligence Platform
 * Sprint 3 - Drag & Drop multi-formats avec progress
 */

import React, { useCallback, useState, useRef } from 'react';
import { useDropzone } from 'react-dropzone';
import { Button } from '../../../components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '../../../components/ui/card';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '../../../components/ui/dialog';
import { cn } from '../../../lib/utils';
import { UploadFile, UploadStatus } from '../types';
import { useDocumentUpload } from '../hooks/useDocuments';
import { useDocumentUploadState } from '../stores/documentStore';
import { DEFAULT_UPLOAD_CONFIG } from '../services/documentService';

interface DocumentUploadProps {
  folderId?: string;
  onUploadComplete?: (documents: any[]) => void;
  className?: string;
}

interface UploadProgressProps {
  file: UploadFile;
  onRemove: (fileId: string) => void;
  onRetry?: (fileId: string) => void;
}

// Utilitaires pour affichage
const formatFileSize = (bytes: number): string => {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(1))} ${sizes[i]}`;
};

const getFileIcon = (type: string): React.ReactNode => {
  if (type.startsWith('image/')) {
    return (
      <svg className="w-8 h-8 text-ma-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
      </svg>
    );
  }
  
  if (type === 'application/pdf') {
    return (
      <svg className="w-8 h-8 text-ma-red-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
      </svg>
    );
  }
  
  if (type.includes('document') || type.includes('word')) {
    return (
      <svg className="w-8 h-8 text-ma-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
      </svg>
    );
  }
  
  if (type.includes('spreadsheet') || type.includes('excel')) {
    return (
      <svg className="w-8 h-8 text-ma-green-700" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 17V7m0 10a2 2 0 01-2 2H5a2 2 0 01-2-2V7a2 2 0 012-2h2a2 2 0 012 2m0 10a2 2 0 002 2h2a2 2 0 002-2M9 7a2 2 0 012-2h2a2 2 0 012 2m0 10V7m0 10a2 2 0 002 2h2a2 2 0 002-2V7a2 2 0 00-2-2h-2a2 2 0 00-2 2" />
      </svg>
    );
  }
  
  return (
    <svg className="w-8 h-8 text-ma-slate-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
    </svg>
  );
};

const getStatusIcon = (status: UploadStatus): React.ReactNode => {
  switch (status) {
    case 'uploading':
      return (
        <svg className="w-4 h-4 animate-spin text-ma-blue-600" fill="none" viewBox="0 0 24 24">
          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
        </svg>
      );
    case 'completed':
      return (
        <svg className="w-4 h-4 text-ma-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
        </svg>
      );
    case 'failed':
      return (
        <svg className="w-4 h-4 text-ma-red-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
        </svg>
      );
    case 'cancelled':
      return (
        <svg className="w-4 h-4 text-ma-slate-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M18.364 18.364A9 9 0 005.636 5.636m12.728 12.728L5.636 5.636m12.728 12.728L18.364 5.636M5.636 18.364l12.728-12.728" />
        </svg>
      );
    default:
      return (
        <svg className="w-4 h-4 text-ma-slate-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
      );
  }
};

// Composant progress pour un fichier
const UploadProgress: React.FC<UploadProgressProps> = ({ file, onRemove, onRetry }) => {
  const getStatusText = (status: UploadStatus): string => {
    switch (status) {
      case 'pending': return 'En attente';
      case 'uploading': return 'Upload en cours...';
      case 'processing': return 'Traitement...';
      case 'completed': return 'Terminé';
      case 'failed': return 'Échec';
      case 'cancelled': return 'Annulé';
      default: return 'Inconnu';
    }
  };

  const getStatusColor = (status: UploadStatus): string => {
    switch (status) {
      case 'completed': return 'text-ma-green-600';
      case 'failed': return 'text-ma-red-600';
      case 'uploading': case 'processing': return 'text-ma-blue-600';
      case 'cancelled': return 'text-ma-slate-600';
      default: return 'text-ma-slate-600';
    }
  };

  const getProgressColor = (status: UploadStatus): string => {
    switch (status) {
      case 'completed': return 'bg-ma-green-500';
      case 'failed': return 'bg-ma-red-500';
      case 'uploading': case 'processing': return 'bg-ma-blue-500';
      default: return 'bg-ma-slate-500';
    }
  };

  return (
    <div className="flex items-center gap-3 p-3 border border-ma-slate-200 rounded-lg">
      {/* Icône fichier */}
      <div className="flex-shrink-0">
        {getFileIcon(file.type)}
      </div>

      {/* Infos fichier */}
      <div className="flex-1 min-w-0">
        <div className="flex items-center justify-between mb-1">
          <p className="text-sm font-medium text-ma-slate-900 truncate">
            {file.name}
          </p>
          <div className="flex items-center gap-2">
            {getStatusIcon(file.status)}
            <span className={cn("text-xs font-medium", getStatusColor(file.status))}>
              {getStatusText(file.status)}
            </span>
          </div>
        </div>

        <div className="flex items-center gap-2 text-xs text-ma-slate-500">
          <span>{formatFileSize(file.size)}</span>
          {file.status === 'uploading' && (
            <span>• {file.progress}%</span>
          )}
        </div>

        {/* Barre de progression */}
        {(file.status === 'uploading' || file.status === 'processing') && (
          <div className="mt-2">
            <div className="w-full bg-ma-slate-200 rounded-full h-1.5">
              <div
                className={cn("h-1.5 rounded-full transition-all duration-300", getProgressColor(file.status))}
                style={{ width: `${file.progress}%` }}
              />
            </div>
          </div>
        )}

        {/* Message d'erreur */}
        {file.status === 'failed' && file.error && (
          <div className="mt-1 text-xs text-ma-red-600">
            {file.error}
          </div>
        )}
      </div>

      {/* Actions */}
      <div className="flex-shrink-0 flex gap-1">
        {file.status === 'failed' && onRetry && (
          <Button
            variant="ghost"
            size="icon"
            className="h-6 w-6"
            onClick={() => onRetry(file.id)}
            title="Réessayer"
          >
            <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
            </svg>
          </Button>
        )}
        
        <Button
          variant="ghost"
          size="icon"
          className="h-6 w-6 hover:text-ma-red-600"
          onClick={() => onRemove(file.id)}
          title="Supprimer"
        >
          <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        </Button>
      </div>
    </div>
  );
};

// Zone de drop principale
export const DocumentUpload: React.FC<DocumentUploadProps> = ({
  folderId,
  onUploadComplete,
  className,
}) => {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const { uploadMultiple } = useDocumentUpload();
  const { 
    uploadQueue, 
    isUploadModalOpen, 
    openUploadModal, 
    closeUploadModal,
    removeFromUploadQueue,
    clearQueue 
  } = useDocumentUploadState();

  const [dragCounter, setDragCounter] = useState(0);

  const onDrop = useCallback((acceptedFiles: File[]) => {
    if (acceptedFiles.length > 0) {
      uploadMultiple(acceptedFiles, folderId);
      openUploadModal(); // Ouvrir le modal de progression
    }
  }, [uploadMultiple, folderId, openUploadModal]);

  const { getRootProps, getInputProps, isDragActive, isDragReject } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf'],
      'image/*': ['.png', '.jpg', '.jpeg', '.gif', '.webp'],
      'application/msword': ['.doc'],
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
      'application/vnd.ms-excel': ['.xls'],
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx'],
      'text/plain': ['.txt'],
      'text/csv': ['.csv'],
    },
    maxSize: DEFAULT_UPLOAD_CONFIG.maxFileSize,
    maxFiles: DEFAULT_UPLOAD_CONFIG.maxFiles,
  });

  const handleFileSelect = useCallback(() => {
    fileInputRef.current?.click();
  }, []);

  const completedUploads = uploadQueue.filter(f => f.status === 'completed');
  const failedUploads = uploadQueue.filter(f => f.status === 'failed');
  const inProgressUploads = uploadQueue.filter(f => 
    f.status === 'uploading' || f.status === 'processing' || f.status === 'pending'
  );

  return (
    <>
      {/* Zone de drop */}
      <div className={className}>
        <div
          {...getRootProps()}
          className={cn(
            "border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors",
            isDragActive && !isDragReject && "border-ma-blue-500 bg-ma-blue-50",
            isDragReject && "border-ma-red-500 bg-ma-red-50",
            !isDragActive && "border-ma-slate-300 hover:border-ma-blue-400 hover:bg-ma-slate-50"
          )}
        >
          <input {...getInputProps()} ref={fileInputRef} />
          
          <div className="space-y-4">
            {/* Icône */}
            <div className="mx-auto w-16 h-16 flex items-center justify-center">
              {isDragActive ? (
                <svg className="w-12 h-12 text-ma-blue-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                </svg>
              ) : (
                <svg className="w-12 h-12 text-ma-slate-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              )}
            </div>

            {/* Texte */}
            <div>
              {isDragReject ? (
                <p className="text-ma-red-600 font-medium">
                  Fichiers non supportés
                </p>
              ) : isDragActive ? (
                <p className="text-ma-blue-600 font-medium">
                  Relâcher pour uploader les fichiers
                </p>
              ) : (
                <>
                  <p className="text-ma-slate-900 font-medium">
                    Glissez-déposez vos fichiers ici
                  </p>
                  <p className="text-sm text-ma-slate-500 mt-1">
                    ou{' '}
                    <span 
                      className="text-ma-blue-600 hover:text-ma-blue-700 cursor-pointer font-medium"
                      onClick={handleFileSelect}
                    >
                      cliquez pour sélectionner
                    </span>
                  </p>
                </>
              )}
            </div>

            {/* Informations formats */}
            <div className="text-xs text-ma-slate-500 space-y-1">
              <p>Formats supportés: PDF, Images (PNG, JPG, GIF), Documents (DOC, DOCX), Feuilles de calcul (XLS, XLSX), Texte (TXT, CSV)</p>
              <p>Taille maximum: {formatFileSize(DEFAULT_UPLOAD_CONFIG.maxFileSize)} par fichier</p>
              <p>Maximum {DEFAULT_UPLOAD_CONFIG.maxFiles} fichiers simultanés</p>
            </div>
          </div>
        </div>

        {/* Résumé uploads en cours */}
        {uploadQueue.length > 0 && (
          <div className="mt-4 flex items-center justify-between text-sm">
            <div className="flex items-center gap-4">
              {inProgressUploads.length > 0 && (
                <span className="text-ma-blue-600">
                  {inProgressUploads.length} en cours
                </span>
              )}
              {completedUploads.length > 0 && (
                <span className="text-ma-green-600">
                  {completedUploads.length} terminés
                </span>
              )}
              {failedUploads.length > 0 && (
                <span className="text-ma-red-600">
                  {failedUploads.length} échoués
                </span>
              )}
            </div>
            
            <Button
              variant="ghost"
              size="sm"
              onClick={openUploadModal}
            >
              Voir progression
            </Button>
          </div>
        )}
      </div>

      {/* Modal de progression */}
      <Dialog open={isUploadModalOpen} onOpenChange={closeUploadModal}>
        <DialogContent className="max-w-2xl max-h-[80vh] overflow-hidden">
          <DialogHeader>
            <DialogTitle>
              Upload de fichiers ({uploadQueue.length})
            </DialogTitle>
          </DialogHeader>

          <div className="space-y-4">
            {/* Statistiques */}
            <div className="flex items-center justify-between text-sm bg-ma-slate-50 p-3 rounded">
              <div className="flex gap-4">
                <span className="text-ma-blue-600">
                  {inProgressUploads.length} en cours
                </span>
                <span className="text-ma-green-600">
                  {completedUploads.length} terminés
                </span>
                {failedUploads.length > 0 && (
                  <span className="text-ma-red-600">
                    {failedUploads.length} échoués
                  </span>
                )}
              </div>
              
              <Button
                variant="ghost"
                size="sm"
                onClick={clearQueue}
                disabled={inProgressUploads.length > 0}
              >
                Nettoyer
              </Button>
            </div>

            {/* Liste des fichiers */}
            <div className="space-y-2 max-h-96 overflow-y-auto custom-scrollbar">
              {uploadQueue.map(file => (
                <UploadProgress
                  key={file.id}
                  file={file}
                  onRemove={removeFromUploadQueue}
                />
              ))}
            </div>

            {/* Actions */}
            <div className="flex justify-end gap-2 pt-4 border-t">
              <Button
                variant="outline"
                onClick={closeUploadModal}
              >
                Fermer
              </Button>
              
              {completedUploads.length > 0 && onUploadComplete && (
                <Button
                  variant="ma"
                  onClick={() => {
                    const documents = completedUploads
                      .map(f => f.uploadedDocument)
                      .filter(Boolean);
                    onUploadComplete(documents);
                    closeUploadModal();
                  }}
                >
                  Continuer ({completedUploads.length})
                </Button>
              )}
            </div>
          </div>
        </DialogContent>
      </Dialog>
    </>
  );
};