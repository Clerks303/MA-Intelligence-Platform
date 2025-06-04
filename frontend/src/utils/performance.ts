/**
 * Utilitaires d'Optimisation Performance - M&A Intelligence Platform
 * Sprint 6 - Optimisations frontend pour performance maximale
 */

import { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import { debounce, throttle } from 'lodash';

// Types pour l'optimisation
export interface PerformanceMetrics {
  renderTime: number;
  memoryUsage: number;
  bundleSize: number;
  loadTime: number;
  interactionTime: number;
}

export interface VirtualizationConfig {
  itemHeight: number;
  containerHeight: number;
  overscan?: number;
  threshold?: number;
}

export interface LazyLoadConfig {
  threshold?: number;
  rootMargin?: string;
  triggerOnce?: boolean;
}

// Hook pour mesurer les performances de rendu
export const usePerformanceMetrics = () => {
  const [metrics, setMetrics] = useState<Partial<PerformanceMetrics>>({});
  const startTime = useRef<number>(0);

  const startMeasurement = useCallback(() => {
    startTime.current = performance.now();
  }, []);

  const endMeasurement = useCallback((type: keyof PerformanceMetrics) => {
    const endTime = performance.now();
    const duration = endTime - startTime.current;
    
    setMetrics(prev => ({
      ...prev,
      [type]: duration
    }));

    // Log pour développement
    if (process.env.NODE_ENV === 'development') {
      console.log(`Performance - ${type}: ${duration.toFixed(2)}ms`);
    }
  }, []);

  const getMemoryUsage = useCallback(() => {
    if ('memory' in performance) {
      const memory = (performance as any).memory;
      return {
        used: memory.usedJSHeapSize,
        total: memory.totalJSHeapSize,
        limit: memory.jsHeapSizeLimit
      };
    }
    return null;
  }, []);

  return {
    metrics,
    startMeasurement,
    endMeasurement,
    getMemoryUsage
  };
};

// Hook de virtualisation pour grandes listes
export const useVirtualization = <T>(
  items: T[],
  config: VirtualizationConfig
) => {
  const [scrollTop, setScrollTop] = useState(0);
  const { itemHeight, containerHeight, overscan = 5 } = config;

  const visibleRange = useMemo(() => {
    const start = Math.floor(scrollTop / itemHeight);
    const visibleCount = Math.ceil(containerHeight / itemHeight);
    const end = start + visibleCount;

    return {
      start: Math.max(0, start - overscan),
      end: Math.min(items.length, end + overscan)
    };
  }, [scrollTop, itemHeight, containerHeight, overscan, items.length]);

  const visibleItems = useMemo(() => {
    return items.slice(visibleRange.start, visibleRange.end).map((item, index) => ({
      item,
      index: visibleRange.start + index,
      top: (visibleRange.start + index) * itemHeight
    }));
  }, [items, visibleRange, itemHeight]);

  const totalHeight = items.length * itemHeight;

  const handleScroll = useCallback((e: React.UIEvent<HTMLDivElement>) => {
    setScrollTop(e.currentTarget.scrollTop);
  }, []);

  return {
    visibleItems,
    totalHeight,
    handleScroll,
    visibleRange
  };
};

// Hook de lazy loading intelligent
export const useLazyLoading = (config: LazyLoadConfig = {}) => {
  const [isVisible, setIsVisible] = useState(false);
  const [hasLoaded, setHasLoaded] = useState(false);
  const elementRef = useRef<HTMLDivElement>(null);

  const {
    threshold = 0.1,
    rootMargin = '50px',
    triggerOnce = true
  } = config;

  useEffect(() => {
    const element = elementRef.current;
    if (!element) return;

    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setIsVisible(true);
          setHasLoaded(true);
          
          if (triggerOnce) {
            observer.unobserve(element);
          }
        } else if (!triggerOnce) {
          setIsVisible(false);
        }
      },
      { threshold, rootMargin }
    );

    observer.observe(element);

    return () => {
      observer.unobserve(element);
    };
  }, [threshold, rootMargin, triggerOnce]);

  return {
    ref: elementRef,
    isVisible,
    hasLoaded
  };
};

// Hook pour optimisation des recherches et filtres
export const useOptimizedSearch = <T>(
  data: T[],
  searchFields: (keyof T)[],
  debounceMs: number = 300
) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [debouncedSearchTerm, setDebouncedSearchTerm] = useState('');

  // Debounce de la recherche
  const debouncedSetSearchTerm = useMemo(
    () => debounce((term: string) => {
      setDebouncedSearchTerm(term);
    }, debounceMs),
    [debounceMs]
  );

  useEffect(() => {
    debouncedSetSearchTerm(searchTerm);
    return () => {
      debouncedSetSearchTerm.cancel();
    };
  }, [searchTerm, debouncedSetSearchTerm]);

  // Recherche optimisée avec mémorisation
  const filteredData = useMemo(() => {
    if (!debouncedSearchTerm.trim()) return data;

    const lowerSearchTerm = debouncedSearchTerm.toLowerCase();
    
    return data.filter(item =>
      searchFields.some(field => {
        const value = item[field];
        if (typeof value === 'string') {
          return value.toLowerCase().includes(lowerSearchTerm);
        }
        if (typeof value === 'number') {
          return value.toString().includes(lowerSearchTerm);
        }
        return false;
      })
    );
  }, [data, debouncedSearchTerm, searchFields]);

  return {
    searchTerm,
    setSearchTerm,
    filteredData,
    isSearching: searchTerm !== debouncedSearchTerm
  };
};

// Hook pour pagination optimisée
export const useOptimizedPagination = <T>(
  data: T[],
  itemsPerPage: number = 20
) => {
  const [currentPage, setCurrentPage] = useState(1);

  const totalPages = Math.ceil(data.length / itemsPerPage);
  
  const paginatedData = useMemo(() => {
    const start = (currentPage - 1) * itemsPerPage;
    const end = start + itemsPerPage;
    return data.slice(start, end);
  }, [data, currentPage, itemsPerPage]);

  const goToPage = useCallback((page: number) => {
    setCurrentPage(Math.max(1, Math.min(page, totalPages)));
  }, [totalPages]);

  const nextPage = useCallback(() => {
    goToPage(currentPage + 1);
  }, [currentPage, goToPage]);

  const prevPage = useCallback(() => {
    goToPage(currentPage - 1);
  }, [currentPage, goToPage]);

  return {
    currentPage,
    totalPages,
    paginatedData,
    goToPage,
    nextPage,
    prevPage,
    hasNext: currentPage < totalPages,
    hasPrev: currentPage > 1
  };
};

// Hook pour gestion du cache optimisé
export const useOptimizedCache = <T>(
  key: string,
  fetcher: () => Promise<T>,
  options: {
    ttl?: number;
    maxSize?: number;
    staleWhileRevalidate?: boolean;
  } = {}
) => {
  const { ttl = 5 * 60 * 1000, maxSize = 100, staleWhileRevalidate = true } = options;
  
  const [data, setData] = useState<T | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const [lastFetch, setLastFetch] = useState<number>(0);

  // Cache simple en mémoire
  const cache = useRef(new Map<string, { data: T; timestamp: number }>());

  const isStale = useCallback(() => {
    return Date.now() - lastFetch > ttl;
  }, [lastFetch, ttl]);

  const getCachedData = useCallback(() => {
    const cached = cache.current.get(key);
    if (cached && Date.now() - cached.timestamp < ttl) {
      return cached.data;
    }
    return null;
  }, [key, ttl]);

  const setCachedData = useCallback((newData: T) => {
    // Gérer la taille maximale du cache
    if (cache.current.size >= maxSize) {
      const firstKey = cache.current.keys().next().value;
      if (firstKey) {
        cache.current.delete(firstKey);
      }
    }
    
    cache.current.set(key, {
      data: newData,
      timestamp: Date.now()
    });
  }, [key, maxSize]);

  const fetchData = useCallback(async (forceRefresh = false) => {
    // Vérifier le cache d'abord
    if (!forceRefresh) {
      const cachedData = getCachedData();
      if (cachedData) {
        setData(cachedData);
        setLastFetch(Date.now());
        return cachedData;
      }
    }

    setIsLoading(true);
    setError(null);

    try {
      const result = await fetcher();
      setData(result);
      setCachedData(result);
      setLastFetch(Date.now());
      return result;
    } catch (err) {
      setError(err as Error);
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, [fetcher, getCachedData, setCachedData]);

  // Fetch initial et gestion du stale-while-revalidate
  useEffect(() => {
    const cachedData = getCachedData();
    
    if (cachedData) {
      setData(cachedData);
      
      if (staleWhileRevalidate && isStale()) {
        // Revalidate en arrière-plan
        fetchData(true).catch(console.error);
      }
    } else {
      fetchData().catch(console.error);
    }
  }, [key, fetchData, getCachedData, isStale, staleWhileRevalidate]);

  return {
    data,
    isLoading,
    error,
    refetch: () => fetchData(true),
    isStale: isStale()
  };
};

// Utilitaires pour optimisation des images
export const useImageOptimization = () => {
  const [supportsWebP, setSupportsWebP] = useState<boolean | null>(null);
  const [supportsAVIF, setSupportsAVIF] = useState<boolean | null>(null);

  useEffect(() => {
    // Tester le support WebP
    const webpTest = new Image();
    webpTest.onload = () => setSupportsWebP(true);
    webpTest.onerror = () => setSupportsWebP(false);
    webpTest.src = 'data:image/webp;base64,UklGRhoAAABXRUJQVlA4TA0AAAAvAAAAEAcQERGIiP4HAA==';

    // Tester le support AVIF
    const avifTest = new Image();
    avifTest.onload = () => setSupportsAVIF(true);
    avifTest.onerror = () => setSupportsAVIF(false);
    avifTest.src = 'data:image/avif;base64,AAAAIGZ0eXBhdmlmAAAAAGF2aWZtaWYxbWlhZk1BMUIAAADybWV0YQAAAAAAAAAoaGRscgAAAAAAAAAAcGljdAAAAAAAAAAAAAAAAGxpYmF2aWYAAAAADnBpdG0AAAAAAAEAAAAeaWxvYwAAAABEAAABAAEAAAABAAABGgAAAB0AAAAoaWluZgAAAAAAAQAAABppbmZlAgAAAAABAABhdjAxQ29sb3IAAAAAamlwcnAAAABLaXBjbwAAABRpc3BlAAAAAAAAAAIAAAACAAAAEHBpeGkAAAAAAwgICAAAAAxhdjFDgQ0MAAAAABNjb2xybmNseAACAAIAAYAAAAAXaXBtYQAAAAAAAAABAAEEAQKDBAAAACVtZGF0EgAKCBgABogQEAwgMg8f8D///8WfhwB8+ErK42A=';
  }, []);

  const getOptimizedImageUrl = useCallback((
    baseUrl: string,
    width?: number,
    quality?: number
  ) => {
    let optimizedUrl = baseUrl;
    
    // Ajouter les paramètres d'optimisation si l'URL le supporte
    const url = new URL(baseUrl, window.location.origin);
    
    if (width) {
      url.searchParams.set('w', width.toString());
    }
    
    if (quality) {
      url.searchParams.set('q', quality.toString());
    }
    
    // Utiliser le format le plus optimisé supporté
    if (supportsAVIF) {
      url.searchParams.set('f', 'avif');
    } else if (supportsWebP) {
      url.searchParams.set('f', 'webp');
    }
    
    return url.toString();
  }, [supportsWebP, supportsAVIF]);

  return {
    supportsWebP,
    supportsAVIF,
    getOptimizedImageUrl
  };
};

// Hook pour throttling des événements
export const useThrottledCallback = <T extends (...args: any[]) => any>(
  callback: T,
  delay: number
): T => {
  return useMemo(
    () => throttle(callback, delay) as unknown as T,
    [callback, delay]
  );
};

// Hook pour debouncing des valeurs
export const useDebouncedValue = <T>(value: T, delay: number): T => {
  const [debouncedValue, setDebouncedValue] = useState<T>(value);

  useEffect(() => {
    const handler = setTimeout(() => {
      setDebouncedValue(value);
    }, delay);

    return () => {
      clearTimeout(handler);
    };
  }, [value, delay]);

  return debouncedValue;
};

// Utilitaires de détection des capacités du navigateur
export const getBrowserCapabilities = () => {
  return {
    supportsIntersectionObserver: 'IntersectionObserver' in window,
    supportsWebP: (() => {
      const canvas = document.createElement('canvas');
      canvas.width = 1;
      canvas.height = 1;
      return canvas.toDataURL('image/webp').indexOf('data:image/webp') === 0;
    })(),
    supportsWebGL: (() => {
      try {
        const canvas = document.createElement('canvas');
        return !!(canvas.getContext('webgl') || canvas.getContext('experimental-webgl'));
      } catch (e) {
        return false;
      }
    })(),
    supportsTouchEvents: 'ontouchstart' in window,
    supportsPointerEvents: 'onpointerdown' in window,
    prefersDarkMode: window.matchMedia('(prefers-color-scheme: dark)').matches,
    prefersReducedMotion: window.matchMedia('(prefers-reduced-motion: reduce)').matches,
    connectionType: (navigator as any)?.connection?.effectiveType,
    memoryLimit: (performance as any)?.memory?.jsHeapSizeLimit
  };
};

export default {
  usePerformanceMetrics,
  useVirtualization,
  useLazyLoading,
  useOptimizedSearch,
  useOptimizedPagination,
  useOptimizedCache,
  useImageOptimization,
  useThrottledCallback,
  useDebouncedValue,
  getBrowserCapabilities
};