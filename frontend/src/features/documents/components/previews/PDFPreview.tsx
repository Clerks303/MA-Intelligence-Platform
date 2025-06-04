/**
 * Composant Preview PDF - M&A Intelligence Platform
 * Sprint 3 - Visualisation PDF avec react-pdf
 */

import React, { useState, useCallback } from 'react';
import { Document as PDFDocument, Page, pdfjs } from 'react-pdf';
import { Button } from '../../../../components/ui/button';
import { Input } from '../../../../components/ui/input';
import { cn } from '../../../../lib/utils';
import { Document, PreviewConfig } from '../../types';

// Configuration PDF.js worker
pdfjs.GlobalWorkerOptions.workerSrc = `//unpkg.com/pdfjs-dist@${pdfjs.version}/build/pdf.worker.min.js`;

interface PDFPreviewProps {
  document: Document;
  config: PreviewConfig;
}

const PDFPreview: React.FC<PDFPreviewProps> = ({ document, config }) => {
  const [numPages, setNumPages] = useState<number>(0);
  const [currentPage, setCurrentPage] = useState<number>(1);
  const [scale, setScale] = useState<number>(1.0);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const onDocumentLoadSuccess = useCallback(({ numPages }: { numPages: number }) => {
    setNumPages(numPages);
    setIsLoading(false);
    setError(null);
  }, []);

  const onDocumentLoadError = useCallback((error: Error) => {
    setError(error.message);
    setIsLoading(false);
  }, []);

  const goToPreviousPage = useCallback(() => {
    setCurrentPage(prev => Math.max(1, prev - 1));
  }, []);

  const goToNextPage = useCallback(() => {
    setCurrentPage(prev => Math.min(numPages, prev + 1));
  }, [numPages]);

  const goToPage = useCallback((page: number) => {
    if (page >= 1 && page <= numPages) {
      setCurrentPage(page);
    }
  }, [numPages]);

  const zoomIn = useCallback(() => {
    setScale(prev => Math.min(3.0, prev + 0.2));
  }, []);

  const zoomOut = useCallback(() => {
    setScale(prev => Math.max(0.5, prev - 0.2));
  }, []);

  const resetZoom = useCallback(() => {
    setScale(1.0);
  }, []);

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="text-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-ma-blue-600 mx-auto mb-4"></div>
          <p className="text-ma-slate-600">Chargement du PDF...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex flex-col items-center justify-center h-96 text-ma-red-500">
        <svg className="w-16 h-16 mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
        <p className="text-center mb-2">Erreur de chargement du PDF</p>
        <p className="text-sm text-center text-ma-slate-500">{error}</p>
        <Button
          variant="outline"
          size="sm"
          className="mt-4"
          onClick={() => window.open(document.downloadUrl, '_blank')}
        >
          Ouvrir dans un nouvel onglet
        </Button>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full">
      {/* Contrôles PDF */}
      <div className="flex items-center justify-between p-3 border-b border-ma-slate-200 bg-white">
        {/* Navigation pages */}
        <div className="flex items-center gap-2">
          <Button
            variant="ghost"
            size="icon"
            className="h-8 w-8"
            onClick={goToPreviousPage}
            disabled={currentPage <= 1}
            title="Page précédente"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
            </svg>
          </Button>

          <div className="flex items-center gap-2 text-sm">
            <Input
              type="number"
              min={1}
              max={numPages}
              value={currentPage}
              onChange={(e) => goToPage(parseInt(e.target.value) || 1)}
              className="w-16 h-8 text-center"
            />
            <span className="text-ma-slate-600">/ {numPages}</span>
          </div>

          <Button
            variant="ghost"
            size="icon"
            className="h-8 w-8"
            onClick={goToNextPage}
            disabled={currentPage >= numPages}
            title="Page suivante"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
            </svg>
          </Button>
        </div>

        {/* Contrôles zoom */}
        {config.enableZoom && (
          <div className="flex items-center gap-2">
            <Button
              variant="ghost"
              size="icon"
              className="h-8 w-8"
              onClick={zoomOut}
              disabled={scale <= 0.5}
              title="Zoom arrière"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0zM13 10H7" />
              </svg>
            </Button>

            <span className="text-sm text-ma-slate-600 min-w-[4rem] text-center">
              {Math.round(scale * 100)}%
            </span>

            <Button
              variant="ghost"
              size="icon"
              className="h-8 w-8"
              onClick={zoomIn}
              disabled={scale >= 3.0}
              title="Zoom avant"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0zM10 7v3m0 0v3m0-3h3m-3 0H7" />
              </svg>
            </Button>

            <Button
              variant="ghost"
              size="sm"
              className="h-8"
              onClick={resetZoom}
              title="Réinitialiser zoom"
            >
              Reset
            </Button>
          </div>
        )}
      </div>

      {/* Contenu PDF */}
      <div className="flex-1 overflow-auto bg-ma-slate-100 p-4">
        <div className="flex justify-center">
          <div className="bg-white shadow-lg">
            <PDFDocument
              file={document.url}
              onLoadSuccess={onDocumentLoadSuccess}
              onLoadError={onDocumentLoadError}
              loading={
                <div className="flex items-center justify-center h-96">
                  <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-ma-blue-600"></div>
                </div>
              }
              error={
                <div className="flex flex-col items-center justify-center h-96 text-ma-red-500 p-8">
                  <svg className="w-16 h-16 mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  <p>Impossible de charger le PDF</p>
                </div>
              }
            >
              <Page
                pageNumber={currentPage}
                scale={scale}
                loading={
                  <div className="flex items-center justify-center h-96 w-full bg-ma-slate-50">
                    <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-ma-blue-600"></div>
                  </div>
                }
                error={
                  <div className="flex items-center justify-center h-96 w-full bg-ma-slate-50 text-ma-red-500">
                    <p>Erreur de chargement de la page</p>
                  </div>
                }
                renderTextLayer={false} // Désactiver pour de meilleures performances
                renderAnnotationLayer={false}
              />
            </PDFDocument>
          </div>
        </div>
      </div>

      {/* Thumbnails */}
      {config.showThumbnails && numPages > 1 && (
        <div className="border-t border-ma-slate-200 bg-white p-2">
          <div className="flex gap-2 overflow-x-auto custom-scrollbar">
            {Array.from({ length: Math.min(numPages, 20) }, (_, index) => {
              const pageNum = index + 1;
              return (
                <div
                  key={pageNum}
                  className={cn(
                    "flex-shrink-0 border-2 rounded cursor-pointer transition-colors",
                    currentPage === pageNum 
                      ? "border-ma-blue-500 bg-ma-blue-50" 
                      : "border-ma-slate-200 hover:border-ma-blue-300"
                  )}
                  onClick={() => goToPage(pageNum)}
                >
                  <PDFDocument file={document.url}>
                    <Page
                      pageNumber={pageNum}
                      scale={0.2}
                      loading={<div className="w-16 h-20 bg-ma-slate-100 animate-pulse"></div>}
                      error={<div className="w-16 h-20 bg-ma-red-50 flex items-center justify-center text-xs text-ma-red-600">Err</div>}
                      renderTextLayer={false}
                      renderAnnotationLayer={false}
                    />
                  </PDFDocument>
                  <div className="text-xs text-center py-1 text-ma-slate-600">
                    {pageNum}
                  </div>
                </div>
              );
            })}
            
            {numPages > 20 && (
              <div className="flex-shrink-0 flex items-center px-2 text-xs text-ma-slate-500">
                +{numPages - 20} pages
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default PDFPreview;