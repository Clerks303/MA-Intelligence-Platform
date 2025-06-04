/**
 * Utilitaires de Performance - Version corrigée
 * Hooks et fonctions pour optimiser les performances React
 */

import { useState, useEffect, useCallback, useMemo, useRef } from 'react';

// Implémentations alternatives simples pour remplacer lodash
const debounce = <T extends (...args: any[]) => any>(func: T, wait: number) => {
  let timeout: NodeJS.Timeout;
  return (...args: Parameters<T>) => {
    clearTimeout(timeout);
    timeout = setTimeout(() => func.apply(this, args), wait);
  };
};

const throttle = <T extends (...args: any[]) => any>(func: T, limit: number) => {
  let inThrottle: boolean;
  return (...args: Parameters<T>) => {
    if (!inThrottle) {
      func.apply(this, args);
      inThrottle = true;
      setTimeout(() => inThrottle = false, limit);
    }
  };
};

// Types pour l'optimisation
export interface PerformanceMetrics {
  renderTime: number;
  memoryUsage: number;
  bundleSize: number;
  apiLatency: number;
}

export interface VirtualizationOptions {
  itemHeight: number;
  bufferSize?: number;
  overscan?: number;
}

/**
 * Hook pour debouncer une valeur
 */
export const useDebounce = <T>(value: T, delay: number): T => {
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

/**
 * Hook pour throttler une fonction
 */
export const useThrottle = <T extends (...args: any[]) => any>(
  fn: T,
  delay: number
): T => {
  const throttledFn = useMemo(() => throttle(fn, delay), [fn, delay]);
  return throttledFn as T;
};

/**
 * Hook pour mesurer les performances de rendu
 */
export const useRenderMetrics = (componentName: string) => {
  const renderCount = useRef(0);
  const startTime = useRef<number>(0);

  useEffect(() => {
    renderCount.current++;
    startTime.current = performance.now();
  });

  useEffect(() => {
    const renderTime = performance.now() - startTime.current;
    
    if (renderTime > 16) { // > 16ms (60fps threshold)
      console.warn(`Slow render detected in ${componentName}: ${renderTime.toFixed(2)}ms`);
    }

    // Log metrics pour analyse
    if (process.env.NODE_ENV === 'development') {
      console.log(`${componentName} - Render #${renderCount.current}: ${renderTime.toFixed(2)}ms`);
    }
  });

  return {
    renderCount: renderCount.current,
    componentName
  };
};

/**
 * Hook pour virtualisation de listes
 */
export const useVirtualization = <T>(
  items: T[],
  containerHeight: number,
  options: VirtualizationOptions
) => {
  const { itemHeight, bufferSize = 5, overscan = 3 } = options;
  const [scrollTop, setScrollTop] = useState(0);

  const visibleCount = Math.ceil(containerHeight / itemHeight);
  const startIndex = Math.max(0, Math.floor(scrollTop / itemHeight) - overscan);
  const endIndex = Math.min(items.length - 1, startIndex + visibleCount + overscan * 2);

  const visibleItems = useMemo(() => {
    return items.slice(startIndex, endIndex + 1).map((item, index) => ({
      item,
      index: startIndex + index,
      top: (startIndex + index) * itemHeight
    }));
  }, [items, startIndex, endIndex, itemHeight]);

  const totalHeight = items.length * itemHeight;

  const handleScroll = useCallback((event: React.UIEvent<HTMLDivElement>) => {
    setScrollTop(event.currentTarget.scrollTop);
  }, []);

  return {
    visibleItems,
    totalHeight,
    handleScroll,
    containerProps: {
      style: {
        height: containerHeight,
        overflow: 'auto'
      },
      onScroll: handleScroll
    }
  };
};

/**
 * Hook pour préchargement d'images
 */
export const useImagePreloader = (imageSrcs: string[]) => {
  const [loadedImages, setLoadedImages] = useState<Set<string>>(new Set());
  const [failedImages, setFailedImages] = useState<Set<string>>(new Set());

  useEffect(() => {
    const promises = imageSrcs.map(src => {
      return new Promise<string>((resolve, reject) => {
        const img = new Image();
        img.onload = () => resolve(src);
        img.onerror = () => reject(src);
        img.src = src;
      });
    });

    Promise.allSettled(promises).then(results => {
      const loaded = new Set<string>();
      const failed = new Set<string>();

      results.forEach((result, index) => {
        if (result.status === 'fulfilled') {
          loaded.add(result.value);
        } else {
          failed.add(imageSrcs[index]);
        }
      });

      setLoadedImages(loaded);
      setFailedImages(failed);
    });
  }, [imageSrcs]);

  return {
    loadedImages,
    failedImages,
    isAllLoaded: loadedImages.size === imageSrcs.length,
    progress: imageSrcs.length > 0 ? (loadedImages.size / imageSrcs.length) * 100 : 0
  };
};

/**
 * Hook pour optimisation mémoire
 */
export const useMemoryOptimization = () => {
  const [memoryInfo, setMemoryInfo] = useState<any>(null);

  useEffect(() => {
    const updateMemoryInfo = () => {
      if ('memory' in performance) {
        setMemoryInfo((performance as any).memory);
      }
    };

    updateMemoryInfo();
    const interval = setInterval(updateMemoryInfo, 5000);

    return () => clearInterval(interval);
  }, []);

  const forceGarbageCollection = useCallback(() => {
    if ('gc' in window) {
      (window as any).gc();
    }
  }, []);

  return {
    memoryInfo,
    forceGarbageCollection,
    isMemoryPressure: memoryInfo ? memoryInfo.usedJSHeapSize / memoryInfo.totalJSHeapSize > 0.8 : false
  };
};

/**
 * Hook pour intersection observer (lazy loading)
 */
export const useIntersectionObserver = (
  options: IntersectionObserverInit = {}
) => {
  const [isIntersecting, setIsIntersecting] = useState(false);
  const [hasIntersected, setHasIntersected] = useState(false);
  const targetRef = useRef<HTMLElement>(null);

  useEffect(() => {
    const element = targetRef.current;
    if (!element) return;

    const observer = new IntersectionObserver(
      ([entry]) => {
        setIsIntersecting(entry.isIntersecting);
        if (entry.isIntersecting && !hasIntersected) {
          setHasIntersected(true);
        }
      },
      {
        threshold: 0.1,
        ...options
      }
    );

    observer.observe(element);

    return () => {
      observer.unobserve(element);
    };
  }, [options, hasIntersected]);

  return {
    targetRef,
    isIntersecting,
    hasIntersected
  };
};

/**
 * Hook pour mesurer les Web Vitals
 */
export const useWebVitals = () => {
  const [vitals, setVitals] = useState<Record<string, number>>({});

  useEffect(() => {
    // Measure First Contentful Paint
    const observer = new PerformanceObserver((list) => {
      for (const entry of list.getEntries()) {
        if (entry.entryType === 'paint' && entry.name === 'first-contentful-paint') {
          setVitals(prev => ({ ...prev, FCP: entry.startTime }));
        }
      }
    });

    observer.observe({ entryTypes: ['paint'] });

    // Measure Largest Contentful Paint
    const lcpObserver = new PerformanceObserver((list) => {
      const entries = list.getEntries();
      const lastEntry = entries[entries.length - 1];
      setVitals(prev => ({ ...prev, LCP: lastEntry.startTime }));
    });

    lcpObserver.observe({ entryTypes: ['largest-contentful-paint'] });

    return () => {
      observer.disconnect();
      lcpObserver.disconnect();
    };
  }, []);

  return vitals;
};

/**
 * Utilitaires de performance générale
 */
export const performanceUtils = {
  // Mesurer le temps d'exécution d'une fonction
  measureTime: async <T>(fn: () => Promise<T> | T, label?: string): Promise<T> => {
    const start = performance.now();
    const result = await fn();
    const end = performance.now();
    
    if (label) {
      console.log(`${label}: ${(end - start).toFixed(2)}ms`);
    }
    
    return result;
  },

  // Créer un worker pour les calculs lourds
  createWorker: (workerFunction: Function) => {
    const blob = new Blob([`(${workerFunction.toString()})()`], {
      type: 'application/javascript'
    });
    return new Worker(URL.createObjectURL(blob));
  },

  // Optimiser les images
  optimizeImage: (file: File, maxWidth: number = 800, quality: number = 0.8): Promise<Blob> => {
    return new Promise((resolve) => {
      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d')!;
      const img = new Image();

      img.onload = () => {
        const ratio = Math.min(maxWidth / img.width, maxWidth / img.height);
        canvas.width = img.width * ratio;
        canvas.height = img.height * ratio;

        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
        canvas.toBlob(resolve as BlobCallback, 'image/jpeg', quality);
      };

      img.src = URL.createObjectURL(file);
    });
  }
};

export default {
  useDebounce,
  useThrottle,
  useRenderMetrics,
  useVirtualization,
  useImagePreloader,
  useMemoryOptimization,
  useIntersectionObserver,
  useWebVitals,
  performanceUtils
};