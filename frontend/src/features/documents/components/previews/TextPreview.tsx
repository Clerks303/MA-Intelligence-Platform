/**
 * Composant Preview Texte - M&A Intelligence Platform
 * Sprint 3 - Visualisation documents texte avec OCR
 */

import React, { useState, useCallback, useMemo } from 'react';
import { Button } from '../../../../components/ui/button';
import { Input } from '../../../../components/ui/input';
import { cn } from '../../../../lib/utils';
import { Document, PreviewConfig } from '../../types';

interface TextPreviewProps {
  document: Document;
  config: PreviewConfig;
}

const TextPreview: React.FC<TextPreviewProps> = ({ document, config }) => {
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [currentMatch, setCurrentMatch] = useState<number>(0);
  const [fontSize, setFontSize] = useState<number>(14);
  const [showLineNumbers, setShowLineNumbers] = useState<boolean>(true);

  // Texte à afficher (OCR ou contenu direct)
  const textContent = document.extractedText || '';
  const lines = textContent.split('\n');

  // Recherche dans le texte
  const searchMatches = useMemo(() => {
    if (!searchQuery.trim()) return [];
    
    const query = searchQuery.toLowerCase();
    const matches: Array<{ lineIndex: number; startIndex: number; endIndex: number }> = [];
    
    lines.forEach((line, lineIndex) => {
      const lowerLine = line.toLowerCase();
      let startIndex = 0;
      
      while (true) {
        const matchIndex = lowerLine.indexOf(query, startIndex);
        if (matchIndex === -1) break;
        
        matches.push({
          lineIndex,
          startIndex: matchIndex,
          endIndex: matchIndex + query.length,
        });
        
        startIndex = matchIndex + 1;
      }
    });
    
    return matches;
  }, [lines, searchQuery]);

  // Navigation dans les résultats de recherche
  const goToNextMatch = useCallback(() => {
    if (searchMatches.length === 0) return;
    setCurrentMatch(prev => (prev + 1) % searchMatches.length);
  }, [searchMatches.length]);

  const goToPreviousMatch = useCallback(() => {
    if (searchMatches.length === 0) return;
    setCurrentMatch(prev => (prev - 1 + searchMatches.length) % searchMatches.length);
  }, [searchMatches.length]);

  // Contrôles police
  const increaseFontSize = useCallback(() => {
    setFontSize(prev => Math.min(24, prev + 2));
  }, []);

  const decreaseFontSize = useCallback(() => {
    setFontSize(prev => Math.max(10, prev - 2));
  }, []);

  const resetFontSize = useCallback(() => {
    setFontSize(14);
  }, []);

  // Rendu d'une ligne avec surlignage
  const renderLine = useCallback((line: string, lineIndex: number) => {
    if (!searchQuery.trim()) {
      return <span>{line}</span>;
    }

    const lineMatches = searchMatches.filter(match => match.lineIndex === lineIndex);
    if (lineMatches.length === 0) {
      return <span>{line}</span>;
    }

    const parts: React.ReactNode[] = [];
    let lastIndex = 0;

    lineMatches.forEach((match, matchIndex) => {
      // Texte avant le match
      if (match.startIndex > lastIndex) {
        parts.push(
          <span key={`before-${matchIndex}`}>
            {line.substring(lastIndex, match.startIndex)}
          </span>
        );
      }

      // Match surligné
      const globalMatchIndex = searchMatches.findIndex(
        m => m.lineIndex === lineIndex && m.startIndex === match.startIndex
      );
      const isCurrentMatch = globalMatchIndex === currentMatch;

      parts.push(
        <span
          key={`match-${matchIndex}`}
          className={cn(
            "px-1 rounded",
            isCurrentMatch 
              ? "bg-ma-blue-200 text-ma-blue-900 font-medium" 
              : "bg-yellow-200 text-yellow-900"
          )}
        >
          {line.substring(match.startIndex, match.endIndex)}
        </span>
      );

      lastIndex = match.endIndex;
    });

    // Texte après le dernier match
    if (lastIndex < line.length) {
      parts.push(
        <span key="after">
          {line.substring(lastIndex)}
        </span>
      );
    }

    return <>{parts}</>;
  }, [searchQuery, searchMatches, currentMatch]);

  // Scroll vers le match actuel
  React.useEffect(() => {
    if (searchMatches.length > 0 && currentMatch < searchMatches.length) {
      const match = searchMatches[currentMatch];
      const lineElement = document.getElementById(`line-${match.lineIndex}`);
      if (lineElement) {
        lineElement.scrollIntoView({ behavior: 'smooth', block: 'center' });
      }
    }
  }, [currentMatch, searchMatches]);

  if (!textContent) {
    return (
      <div className="flex flex-col items-center justify-center h-96 text-ma-slate-500">
        <svg className="w-16 h-16 mb-4 text-ma-slate-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
        </svg>
        <p className="text-center mb-2">Aucun texte disponible</p>
        <p className="text-sm text-center text-ma-slate-400 mb-4">
          Le texte n'a pas pu être extrait de ce document
        </p>
        {document.ocrStatus === 'not_started' && (
          <Button variant="outline" size="sm">
            Démarrer l'extraction OCR
          </Button>
        )}
        {document.ocrStatus === 'processing' && (
          <div className="flex items-center gap-2 text-ma-blue-600">
            <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-ma-blue-600"></div>
            <span className="text-sm">Extraction en cours...</span>
          </div>
        )}
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full">
      {/* Contrôles texte */}
      <div className="flex items-center justify-between p-3 border-b border-ma-slate-200 bg-white">
        {/* Recherche */}
        <div className="flex items-center gap-2 flex-1 max-w-md">
          <div className="relative flex-1">
            <Input
              placeholder="Rechercher dans le texte..."
              value={searchQuery}
              onChange={(e) => {
                setSearchQuery(e.target.value);
                setCurrentMatch(0);
              }}
              className="pr-8"
            />
            {searchQuery && (
              <Button
                variant="ghost"
                size="icon"
                className="absolute right-0 top-0 h-full w-8"
                onClick={() => {
                  setSearchQuery('');
                  setCurrentMatch(0);
                }}
              >
                <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </Button>
            )}
          </div>

          {/* Navigation résultats */}
          {searchMatches.length > 0 && (
            <div className="flex items-center gap-1">
              <span className="text-xs text-ma-slate-600 whitespace-nowrap">
                {currentMatch + 1} / {searchMatches.length}
              </span>
              <Button
                variant="ghost"
                size="icon"
                className="h-6 w-6"
                onClick={goToPreviousMatch}
                disabled={searchMatches.length <= 1}
              >
                <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
                </svg>
              </Button>
              <Button
                variant="ghost"
                size="icon"
                className="h-6 w-6"
                onClick={goToNextMatch}
                disabled={searchMatches.length <= 1}
              >
                <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                </svg>
              </Button>
            </div>
          )}
        </div>

        {/* Contrôles affichage */}
        <div className="flex items-center gap-1">
          {/* Numéros de ligne */}
          <Button
            variant={showLineNumbers ? "default" : "ghost"}
            size="sm"
            className="h-8"
            onClick={() => setShowLineNumbers(!showLineNumbers)}
            title="Numéros de ligne"
          >
            #
          </Button>

          {/* Police */}
          <div className="mx-2 h-4 w-px bg-ma-slate-300"></div>
          
          <Button
            variant="ghost"
            size="icon"
            className="h-8 w-8"
            onClick={decreaseFontSize}
            disabled={fontSize <= 10}
            title="Diminuer police"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20 12H4" />
            </svg>
          </Button>

          <span className="text-sm text-ma-slate-600 min-w-[2rem] text-center">
            {fontSize}px
          </span>

          <Button
            variant="ghost"
            size="icon"
            className="h-8 w-8"
            onClick={increaseFontSize}
            disabled={fontSize >= 24}
            title="Augmenter police"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
            </svg>
          </Button>

          <Button
            variant="ghost"
            size="sm"
            className="h-8"
            onClick={resetFontSize}
            title="Réinitialiser police"
          >
            Reset
          </Button>
        </div>
      </div>

      {/* Contenu texte */}
      <div className="flex-1 overflow-auto bg-white">
        <div className="p-4">
          <div 
            className="font-mono whitespace-pre-wrap"
            style={{ fontSize: `${fontSize}px`, lineHeight: 1.5 }}
          >
            {lines.map((line, index) => (
              <div
                key={index}
                id={`line-${index}`}
                className="flex"
              >
                {/* Numéro de ligne */}
                {showLineNumbers && (
                  <span className="text-ma-slate-400 text-right pr-4 select-none shrink-0 w-12">
                    {index + 1}
                  </span>
                )}
                
                {/* Contenu ligne */}
                <span className="flex-1 min-w-0">
                  {renderLine(line, index)}
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Informations texte */}
      <div className="border-t border-ma-slate-200 bg-white p-2 text-xs text-ma-slate-500">
        <div className="flex items-center justify-between">
          <span>
            {lines.length} lignes • {textContent.length} caractères
          </span>
          <div className="flex items-center gap-4">
            {searchMatches.length > 0 && (
              <span>{searchMatches.length} résultats trouvés</span>
            )}
            {document.ocrStatus && (
              <span className={cn(
                "px-2 py-1 rounded text-xs",
                document.ocrStatus === 'completed' ? "bg-ma-green-100 text-ma-green-700" :
                document.ocrStatus === 'failed' ? "bg-ma-red-100 text-ma-red-700" :
                "bg-ma-blue-100 text-ma-blue-700"
              )}>
                OCR: {document.ocrStatus}
              </span>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default TextPreview;