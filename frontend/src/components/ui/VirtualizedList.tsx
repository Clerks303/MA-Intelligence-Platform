/**
 * Liste Virtualisée - M&A Intelligence Platform
 * Composant optimisé pour afficher de grandes listes
 */

import React, { useState, useCallback, useMemo, useRef, useEffect } from 'react';
import { Card, CardContent } from './card';

// Types
export interface VirtualizedListProps<T> {
  data: T[];
  itemHeight: number;
  containerHeight: number;
  renderItem: (item: T, index: number) => React.ReactNode;
  className?: string;
  onLoadMore?: () => Promise<void>;
  hasNextPage?: boolean;
  isLoading?: boolean;
  overscan?: number;
}

// Composant Liste Virtualisée
export function VirtualizedList<T>({
  data,
  itemHeight,
  containerHeight,
  renderItem,
  className = '',
  onLoadMore,
  hasNextPage = false,
  isLoading = false,
  overscan = 3
}: VirtualizedListProps<T>) {
  const [scrollTop, setScrollTop] = useState(0);
  const containerRef = useRef<HTMLDivElement>(null);

  // Calculs de virtualisation
  const visibleCount = Math.ceil(containerHeight / itemHeight);
  const startIndex = Math.max(0, Math.floor(scrollTop / itemHeight) - overscan);
  const endIndex = Math.min(data.length - 1, startIndex + visibleCount + overscan * 2);

  const visibleItems = useMemo(() => {
    return data.slice(startIndex, endIndex + 1).map((item, index) => ({
      item,
      index: startIndex + index,
      top: (startIndex + index) * itemHeight
    }));
  }, [data, startIndex, endIndex, itemHeight]);

  const totalHeight = data.length * itemHeight;

  // Gestion du scroll
  const handleScroll = useCallback((event: React.UIEvent<HTMLDivElement>) => {
    const newScrollTop = event.currentTarget.scrollTop;
    setScrollTop(newScrollTop);

    // Détecter si on approche de la fin pour charger plus
    if (onLoadMore && hasNextPage && !isLoading) {
      const scrollableHeight = totalHeight - containerHeight;
      const scrollPercentage = newScrollTop / scrollableHeight;
      
      if (scrollPercentage > 0.8) {
        onLoadMore();
      }
    }
  }, [onLoadMore, hasNextPage, isLoading, totalHeight, containerHeight]);

  return (
    <Card className={`relative ${className}`}>
      <CardContent className="p-0">
        <div
          ref={containerRef}
          className="overflow-auto"
          style={{ height: containerHeight }}
          onScroll={handleScroll}
        >
          <div style={{ height: totalHeight, position: 'relative' }}>
            {visibleItems.map(({ item, index, top }) => (
              <div
                key={index}
                style={{
                  position: 'absolute',
                  top,
                  left: 0,
                  right: 0,
                  height: itemHeight,
                }}
              >
                {renderItem(item, index)}
              </div>
            ))}
          </div>
          
          {isLoading && (
            <div className="flex justify-center items-center p-4">
              <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-600"></div>
              <span className="ml-2">Chargement...</span>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}

// Hook utilitaire pour faciliter l'utilisation
export const useVirtualizedList = <T extends unknown>(data: T[]) => {
  const [isLoading, setIsLoading] = useState(false);
  const [hasNextPage, setHasNextPage] = useState(true);
  const [page, setPage] = useState(1);

  const loadMore = useCallback(async () => {
    setIsLoading(true);
    try {
      // Logique de chargement personnalisée
      await new Promise(resolve => setTimeout(resolve, 1000)); // Simulation
      setPage(prev => prev + 1);
      
      // Exemple: arrêter le chargement après 5 pages
      if (page >= 5) {
        setHasNextPage(false);
      }
    } finally {
      setIsLoading(false);
    }
  }, [page]);

  return {
    loadMore,
    hasNextPage,
    isLoading,
    page
  };
};

export default VirtualizedList;