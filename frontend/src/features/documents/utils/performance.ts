/**
 * Utilitaires Performance - M&A Intelligence Platform
 * Sprint 3 - Optimisations bundle size, lazy loading et virtualisation
 */

import { lazy } from 'react';

// === CONFIGURATION PERFORMANCE ===

export const PERFORMANCE_CONFIG = {
  // Virtualisation
  virtualization: {
    itemHeight: 60,
    overscan: 5,
    enableSmooth: true,
    threshold: 100, // Active la virtualisation au-delà de N éléments
  },
  
  // Cache
  cache: {
    documentTtl: 5 * 60 * 1000, // 5 minutes
    analyticsTtl: 2 * 60 * 1000, // 2 minutes
    searchTtl: 1 * 60 * 1000, // 1 minute
    maxCacheSize: 50, // Nombre max d'éléments en cache
  },
  
  // Debounce
  debounce: {
    search: 300,
    filters: 200,
    analytics: 500,
  },
  
  // Bundle size
  bundle: {
    maxChunkSize: 500000, // 500KB
    enableCodeSplitting: true,
    enableTreeShaking: true,
  },
  
  // Préchargement
  prefetch: {
    enabled: true,
    batchSize: 10,
    delay: 1000,
  },
} as const;

// === LAZY LOADING COMPONENTS ===

// Components lourds en lazy loading
export const LazyComponents = {
  // Module principal
  DocumentManagement: lazy(() => 
    import('../pages/DocumentManagement').then(module => ({ 
      default: module.DocumentManagement 
    }))
  ),
  
  // Upload avancé
  AdvancedDocumentUpload: lazy(() => 
    import('../components/AdvancedDocumentUpload').then(module => ({ 
      default: module.AdvancedDocumentUpload 
    }))
  ),
  
  // Preview moderne
  ModernDocumentPreview: lazy(() => 
    import('../components/ModernDocumentPreview').then(module => ({ 
      default: module.ModernDocumentPreview 
    }))
  ),
  
  // Arbre virtualisé
  VirtualizedDocumentTree: lazy(() => 
    import('../components/VirtualizedDocumentTree').then(module => ({ 
      default: module.VirtualizedDocumentTree 
    }))
  ),
  
  // Preview spécialisés
  PDFPreview: lazy(() => 
    import('../components/previews/PDFPreview').then(module => ({ 
      default: module.PDFPreview 
    }))
  ),
  
  ImagePreview: lazy(() => 
    import('../components/previews/ImagePreview').then(module => ({ 
      default: module.ImagePreview 
    }))
  ),
  
  VideoPreview: lazy(() => 
    import('../components/previews/VideoPreview').then(module => ({ 
      default: module.VideoPreview 
    }))
  ),
  
  TextPreview: lazy(() => 
    import('../components/previews/TextPreview').then(module => ({ 
      default: module.TextPreview 
    }))
  ),
} as const;

// === PERFORMANCE HOOKS ===

export const usePerformanceOptimization = () => {
  const optimizeImages = (images: HTMLImageElement[]) => {
    // Optimisation des images avec lazy loading et compression
    images.forEach(img => {
      if ('loading' in HTMLImageElement.prototype) {
        img.loading = 'lazy';
      }
      
      // Observer pour intersection
      const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
          if (entry.isIntersecting) {
            const img = entry.target as HTMLImageElement;
            if (img.dataset.src) {
              img.src = img.dataset.src;
              observer.unobserve(img);
            }
          }
        });
      });
      
      observer.observe(img);
    });
  };

  const optimizeBundle = () => {
    // Vérification taille bundle
    if (typeof window !== 'undefined' && window.performance) {
      const navigation = window.performance.getEntriesByType('navigation')[0] as PerformanceNavigationTiming;
      if (navigation) {
        const transferSize = navigation.transferSize || 0;
        if (transferSize > PERFORMANCE_CONFIG.bundle.maxChunkSize) {
          console.warn(`Bundle size dépassé: ${transferSize} bytes`);
        }
      }
    }
  };

  const optimizeMemory = () => {
    // Nettoyage mémoire
    if (typeof window !== 'undefined' && 'gc' in window) {
      // @ts-ignore - garbage collection si disponible
      window.gc?.();
    }
  };

  return {
    optimizeImages,
    optimizeBundle,
    optimizeMemory,
  };
};

// === CACHE MANAGER ===

interface CacheEntry<T> {
  data: T;
  timestamp: number;
  ttl: number;
}

class PerformanceCache {
  private cache = new Map<string, CacheEntry<any>>();
  private maxSize = PERFORMANCE_CONFIG.cache.maxCacheSize;

  set<T>(key: string, data: T, ttl: number = PERFORMANCE_CONFIG.cache.documentTtl): void {
    // Nettoyer le cache si trop plein
    if (this.cache.size >= this.maxSize) {
      this.cleanup();
    }

    this.cache.set(key, {
      data,
      timestamp: Date.now(),
      ttl,
    });
  }

  get<T>(key: string): T | null {
    const entry = this.cache.get(key);
    if (!entry) return null;

    // Vérifier expiration
    if (Date.now() - entry.timestamp > entry.ttl) {
      this.cache.delete(key);
      return null;
    }

    return entry.data;
  }

  has(key: string): boolean {
    const entry = this.cache.get(key);
    if (!entry) return false;

    // Vérifier expiration
    if (Date.now() - entry.timestamp > entry.ttl) {
      this.cache.delete(key);
      return false;
    }

    return true;
  }

  delete(key: string): void {
    this.cache.delete(key);
  }

  clear(): void {
    this.cache.clear();
  }

  cleanup(): void {
    const now = Date.now();
    for (const [key, entry] of this.cache.entries()) {
      if (now - entry.timestamp > entry.ttl) {
        this.cache.delete(key);
      }
    }

    // Si encore trop plein, supprimer les plus anciens
    if (this.cache.size >= this.maxSize) {
      const entries = Array.from(this.cache.entries())
        .sort((a, b) => a[1].timestamp - b[1].timestamp);
      
      const toDelete = entries.slice(0, Math.floor(this.maxSize / 4));
      toDelete.forEach(([key]) => this.cache.delete(key));
    }
  }

  getStats() {
    return {
      size: this.cache.size,
      maxSize: this.maxSize,
      usage: (this.cache.size / this.maxSize) * 100,
    };
  }
}

export const performanceCache = new PerformanceCache();

// === DEBOUNCE UTILITIES ===

export const createDebounce = <T extends (...args: any[]) => any>(
  func: T,
  delay: number
): (...args: Parameters<T>) => void => {
  let timeoutId: NodeJS.Timeout;
  
  return (...args: Parameters<T>) => {
    clearTimeout(timeoutId);
    timeoutId = setTimeout(() => func(...args), delay);
  };
};

export const createThrottle = <T extends (...args: any[]) => any>(
  func: T,
  delay: number
): (...args: Parameters<T>) => void => {
  let lastCall = 0;
  
  return (...args: Parameters<T>) => {
    const now = Date.now();
    if (now - lastCall >= delay) {
      lastCall = now;
      func(...args);
    }
  };
};

// === VIRTUALIZATION HELPERS ===

export const calculateVirtualizedMetrics = (
  totalItems: number,
  itemHeight: number,
  containerHeight: number,
  overscan: number = PERFORMANCE_CONFIG.virtualization.overscan
) => {
  const visibleCount = Math.ceil(containerHeight / itemHeight);
  const startIndex = Math.max(0, Math.floor(window.scrollY / itemHeight) - overscan);
  const endIndex = Math.min(totalItems - 1, startIndex + visibleCount + overscan * 2);
  
  return {
    visibleCount,
    startIndex,
    endIndex,
    renderCount: endIndex - startIndex + 1,
    memoryOptimization: Math.round((1 - (endIndex - startIndex + 1) / totalItems) * 100),
  };
};

// === PREFETCHING ===

export const createPrefetcher = <T>(
  fetchFunction: (id: string) => Promise<T>,
  options: {
    batchSize?: number;
    delay?: number;
    enabled?: boolean;
  } = {}
) => {
  const {
    batchSize = PERFORMANCE_CONFIG.prefetch.batchSize,
    delay = PERFORMANCE_CONFIG.prefetch.delay,
    enabled = PERFORMANCE_CONFIG.prefetch.enabled,
  } = options;

  const prefetchQueue = new Set<string>();
  let prefetchTimeout: NodeJS.Timeout;

  const prefetch = (ids: string[]) => {
    if (!enabled) return;
    
    ids.forEach(id => prefetchQueue.add(id));
    
    clearTimeout(prefetchTimeout);
    prefetchTimeout = setTimeout(() => {
      const batch = Array.from(prefetchQueue).slice(0, batchSize);
      prefetchQueue.clear();
      
      batch.forEach(async (id) => {
        try {
          const data = await fetchFunction(id);
          performanceCache.set(`prefetch-${id}`, data);
        } catch (error) {
          console.warn(`Prefetch failed for ${id}:`, error);
        }
      });
    }, delay);
  };

  const getPrefetched = (id: string): T | null => {
    return performanceCache.get(`prefetch-${id}`);
  };

  return { prefetch, getPrefetched };
};

// === PERFORMANCE MONITORING ===

export const performanceMonitor = {
  start: (label: string) => {
    if (typeof window !== 'undefined' && window.performance) {
      window.performance.mark(`${label}-start`);
    }
  },

  end: (label: string) => {
    if (typeof window !== 'undefined' && window.performance) {
      window.performance.mark(`${label}-end`);
      window.performance.measure(label, `${label}-start`, `${label}-end`);
      
      const measure = window.performance.getEntriesByName(label, 'measure')[0];
      if (measure) {
        console.log(`Performance [${label}]: ${measure.duration.toFixed(2)}ms`);
      }
    }
  },

  getMetrics: () => {
    if (typeof window === 'undefined' || !window.performance) {
      return null;
    }

    const navigation = window.performance.getEntriesByType('navigation')[0] as PerformanceNavigationTiming;
    const memory = (window.performance as any).memory;

    return {
      // Navigation
      pageLoad: navigation ? navigation.loadEventEnd - navigation.navigationStart : 0,
      domReady: navigation ? navigation.domContentLoadedEventEnd - navigation.navigationStart : 0,
      
      // Mémoire (si disponible)
      memoryUsed: memory ? memory.usedJSHeapSize : 0,
      memoryTotal: memory ? memory.totalJSHeapSize : 0,
      memoryLimit: memory ? memory.jsHeapSizeLimit : 0,
      
      // Cache
      cache: performanceCache.getStats(),
    };
  },
};

// === EXPORT ===

export const PerformanceUtils = {
  config: PERFORMANCE_CONFIG,
  components: LazyComponents,
  cache: performanceCache,
  debounce: createDebounce,
  throttle: createThrottle,
  monitor: performanceMonitor,
  prefetcher: createPrefetcher,
  virtualization: calculateVirtualizedMetrics,
} as const;