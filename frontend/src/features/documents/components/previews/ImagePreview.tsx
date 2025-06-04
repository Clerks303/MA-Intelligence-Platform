/**
 * Composant Preview Image - M&A Intelligence Platform
 * Sprint 3 - Visualisation images avec zoom et rotation
 */

import React, { useState, useCallback, useRef, useEffect } from 'react';
import { Button } from '../../../../components/ui/button';
import { cn } from '../../../../lib/utils';
import { Document, PreviewConfig } from '../../types';

interface ImagePreviewProps {
  document: Document;
  config: PreviewConfig;
}

const ImagePreview: React.FC<ImagePreviewProps> = ({ document, config }) => {
  const [scale, setScale] = useState<number>(1.0);
  const [rotation, setRotation] = useState<number>(0);
  const [position, setPosition] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const [dragStart, setDragStart] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [hasError, setHasError] = useState<boolean>(false);
  const [naturalSize, setNaturalSize] = useState<{ width: number; height: number } | null>(null);

  const imageRef = useRef<HTMLImageElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  // Gestion du chargement de l'image
  const handleImageLoad = useCallback((event: React.SyntheticEvent<HTMLImageElement>) => {
    const img = event.currentTarget;
    setNaturalSize({ width: img.naturalWidth, height: img.naturalHeight });
    setIsLoading(false);
    setHasError(false);
    
    // Ajuster le zoom initial pour s'adapter au conteneur
    if (containerRef.current) {
      const containerRect = containerRef.current.getBoundingClientRect();
      const scaleX = (containerRect.width - 40) / img.naturalWidth;
      const scaleY = (containerRect.height - 40) / img.naturalHeight;
      const initialScale = Math.min(1, Math.min(scaleX, scaleY));
      setScale(initialScale);
    }
  }, []);

  const handleImageError = useCallback(() => {
    setIsLoading(false);
    setHasError(true);
  }, []);

  // Contrôles zoom
  const zoomIn = useCallback(() => {
    setScale(prev => Math.min(5.0, prev * 1.2));
  }, []);

  const zoomOut = useCallback(() => {
    setScale(prev => Math.max(0.1, prev / 1.2));
  }, []);

  const resetZoom = useCallback(() => {
    setScale(1.0);
    setPosition({ x: 0, y: 0 });
  }, []);

  const fitToContainer = useCallback(() => {
    if (containerRef.current && naturalSize) {
      const containerRect = containerRef.current.getBoundingClientRect();
      const scaleX = (containerRect.width - 40) / naturalSize.width;
      const scaleY = (containerRect.height - 40) / naturalSize.height;
      const fitScale = Math.min(scaleX, scaleY);
      setScale(fitScale);
      setPosition({ x: 0, y: 0 });
    }
  }, [naturalSize]);

  // Contrôles rotation
  const rotateLeft = useCallback(() => {
    setRotation(prev => prev - 90);
  }, []);

  const rotateRight = useCallback(() => {
    setRotation(prev => prev + 90);
  }, []);

  const resetRotation = useCallback(() => {
    setRotation(0);
  }, []);

  // Gestion du drag
  const handleMouseDown = useCallback((event: React.MouseEvent) => {
    if (scale <= 1) return; // Pas de drag si pas de zoom
    
    setIsDragging(true);
    setDragStart({
      x: event.clientX - position.x,
      y: event.clientY - position.y,
    });
  }, [scale, position]);

  const handleMouseMove = useCallback((event: React.MouseEvent) => {
    if (!isDragging) return;
    
    setPosition({
      x: event.clientX - dragStart.x,
      y: event.clientY - dragStart.y,
    });
  }, [isDragging, dragStart]);

  const handleMouseUp = useCallback(() => {
    setIsDragging(false);
  }, []);

  // Zoom avec molette
  const handleWheel = useCallback((event: React.WheelEvent) => {
    if (!config.enableZoom) return;
    
    event.preventDefault();
    const delta = event.deltaY > 0 ? 0.9 : 1.1;
    setScale(prev => Math.max(0.1, Math.min(5.0, prev * delta)));
  }, [config.enableZoom]);

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="text-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-ma-blue-600 mx-auto mb-4"></div>
          <p className="text-ma-slate-600">Chargement de l'image...</p>
        </div>
      </div>
    );
  }

  if (hasError) {
    return (
      <div className="flex flex-col items-center justify-center h-96 text-ma-red-500">
        <svg className="w-16 h-16 mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
        <p className="text-center mb-2">Erreur de chargement de l'image</p>
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
      {/* Contrôles image */}
      <div className="flex items-center justify-between p-3 border-b border-ma-slate-200 bg-white">
        {/* Informations image */}
        <div className="flex items-center gap-4 text-sm text-ma-slate-600">
          {naturalSize && (
            <>
              <span>{naturalSize.width} × {naturalSize.height}</span>
              <span>•</span>
            </>
          )}
          <span>{Math.round(scale * 100)}%</span>
          {rotation !== 0 && (
            <>
              <span>•</span>
              <span>{rotation}°</span>
            </>
          )}
        </div>

        {/* Contrôles */}
        <div className="flex items-center gap-1">
          {/* Zoom */}
          {config.enableZoom && (
            <>
              <Button
                variant="ghost"
                size="icon"
                className="h-8 w-8"
                onClick={zoomOut}
                disabled={scale <= 0.1}
                title="Zoom arrière"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0zM13 10H7" />
                </svg>
              </Button>

              <Button
                variant="ghost"
                size="icon"
                className="h-8 w-8"
                onClick={zoomIn}
                disabled={scale >= 5.0}
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
                onClick={fitToContainer}
                title="Ajuster à la fenêtre"
              >
                Ajuster
              </Button>

              <Button
                variant="ghost"
                size="sm"
                className="h-8"
                onClick={resetZoom}
                title="Taille réelle"
              >
                100%
              </Button>
            </>
          )}

          {/* Rotation */}
          {config.enableRotation && (
            <>
              <div className="mx-2 h-4 w-px bg-ma-slate-300"></div>
              
              <Button
                variant="ghost"
                size="icon"
                className="h-8 w-8"
                onClick={rotateLeft}
                title="Rotation anti-horaire"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 10h10a8 8 0 018 8v2M3 10l6 6m-6-6l6-6" />
                </svg>
              </Button>

              <Button
                variant="ghost"
                size="icon"
                className="h-8 w-8"
                onClick={rotateRight}
                title="Rotation horaire"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 10H11a8 8 0 00-8 8v2m18-10l-6 6m6-6l-6-6" />
                </svg>
              </Button>

              <Button
                variant="ghost"
                size="sm"
                className="h-8"
                onClick={resetRotation}
                title="Réinitialiser rotation"
              >
                0°
              </Button>
            </>
          )}
        </div>
      </div>

      {/* Container image */}
      <div 
        ref={containerRef}
        className="flex-1 overflow-hidden bg-ma-slate-100 relative"
        onWheel={handleWheel}
      >
        <div 
          className="w-full h-full flex items-center justify-center"
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseUp}
          style={{ 
            cursor: scale > 1 ? (isDragging ? 'grabbing' : 'grab') : 'default',
          }}
        >
          <img
            ref={imageRef}
            src={document.previewUrl || document.url}
            alt={document.name}
            onLoad={handleImageLoad}
            onError={handleImageError}
            className={cn(
              "max-w-none transition-transform duration-200",
              isDragging && "transition-none"
            )}
            style={{
              transform: `scale(${scale}) rotate(${rotation}deg) translate(${position.x}px, ${position.y}px)`,
              transformOrigin: 'center center',
            }}
            draggable={false}
          />
        </div>

        {/* Overlay d'aide */}
        {scale > 1 && (
          <div className="absolute top-4 left-4 bg-black/75 text-white text-xs px-2 py-1 rounded">
            Cliquez et glissez pour déplacer
          </div>
        )}

        {/* Overlay zoom */}
        {config.enableZoom && (
          <div className="absolute bottom-4 right-4 bg-black/75 text-white text-xs px-2 py-1 rounded">
            Molette souris pour zoomer
          </div>
        )}
      </div>

      {/* Informations techniques */}
      {naturalSize && (
        <div className="border-t border-ma-slate-200 bg-white p-2 text-xs text-ma-slate-500">
          <div className="flex items-center justify-between">
            <span>
              {naturalSize.width} × {naturalSize.height} pixels
            </span>
            <span>
              {document.extension.toUpperCase()} • {Math.round(document.size / 1024)} KB
            </span>
          </div>
        </div>
      )}
    </div>
  );
};

export default ImagePreview;