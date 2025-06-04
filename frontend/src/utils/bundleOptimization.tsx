/**
 * Optimisation des Bundles - Version corrigée
 * Système de lazy loading et préchargement
 */

import { lazy, Suspense, ComponentType, ReactElement } from 'react';

// Types pour l'optimisation des bundles
export interface LazyComponentOptions {
  webpackChunkName?: string;
  fallback?: ReactElement | string;
  preload?: boolean;
  timeout?: number;
}

// Cache pour les composants lazy
const lazyComponentCache = new Map<string, ComponentType<any>>();

/**
 * Créer un composant lazy avec options avancées
 */
export const createLazyComponent = <P extends Record<string, any>>(
  importFn: () => Promise<{ default: ComponentType<P> }>,
  options: LazyComponentOptions = {}
): ComponentType<P> => {
  const {
    fallback = "Chargement...",
    preload = false,
    timeout = 30000
  } = options;

  // Cache key basé sur la fonction import
  const cacheKey = importFn.toString();
  
  // Vérifier le cache
  if (lazyComponentCache.has(cacheKey)) {
    return lazyComponentCache.get(cacheKey)!;
  }

  // Import avec retry et timeout
  const importWithRetry = async () => {
    const timeoutPromise = new Promise((_, reject) => {
      setTimeout(() => reject(new Error('Import timeout')), timeout);
    });

    try {
      return await Promise.race([importFn(), timeoutPromise]) as { default: ComponentType<P> };
    } catch (error) {
      // Retry une fois
      console.warn('Import failed, retrying...', error);
      return await importFn();
    }
  };

  // Créer le composant lazy
  const LazyComponent = lazy(importWithRetry);

  // Composant wrapper avec gestion d'erreurs
  const WrappedComponent = (props: P) => (
    <Suspense fallback={typeof fallback === 'string' ? <div>{fallback}</div> : fallback}>
      <LazyComponent {...(props as any)} />
    </Suspense>
  );

  // Ajouter la méthode preload
  (WrappedComponent as any).preload = () => importWithRetry();

  // Précharger si demandé
  if (preload) {
    (WrappedComponent as any).preload();
  }

  // Mettre en cache
  lazyComponentCache.set(cacheKey, WrappedComponent);

  return WrappedComponent;
};

// Fonction pour précharger des composants basée sur l'interaction utilisateur
export const preloadOnInteraction = (
  component: ComponentType<any>,
  events: string[] = ['mouseenter', 'touchstart', 'focus']
) => {
  const preload = () => {
    if ((component as any).preload) {
      (component as any).preload();
    }
  };

  const addListeners = (element: Element) => {
    events.forEach(event => {
      element.addEventListener(event, preload, { once: true, passive: true });
    });
  };

  const removeListeners = (element: Element) => {
    events.forEach(event => {
      element.removeEventListener(event, preload);
    });
  };

  return { addListeners, removeListeners, preload };
};

// Cache des routes pour éviter les re-créations
const routeComponents = new Map<string, ComponentType<any>>();

/**
 * Créer un composant de route avec lazy loading
 */
export const createRouteComponent = (
  routeName: string,
  importFn: () => Promise<{ default: ComponentType<any> }>
) => {
  if (routeComponents.has(routeName)) {
    return routeComponents.get(routeName)!;
  }

  const component = createLazyComponent(importFn, {
    webpackChunkName: `route-${routeName}`,
    fallback: 'Chargement de la page...'
    });

    routeComponents.set(routeName, component);
    return component;
};

// Stratégie de préchargement par priorité
export const preloadStrategy = {
  // Précharger immédiatement (composants critiques)
  immediate: (components: ComponentType<any>[]) => {
    components.forEach(component => {
      if ((component as any).preload) {
        (component as any).preload();
      }
    });
  },

  // Précharger quand le navigateur est inactif
  idle: (components: ComponentType<any>[]) => {
    if ('requestIdleCallback' in window) {
      (window as any).requestIdleCallback(() => {
        preloadStrategy.immediate(components);
      });
    } else {
      setTimeout(() => preloadStrategy.immediate(components), 0);
    }
  },

  // Précharger sur interaction
  onInteraction: (
    components: ComponentType<any>[],
    targetSelector: string,
    events: string[] = ['mouseenter']
  ) => {
    const targets = document.querySelectorAll(targetSelector);
    targets.forEach(target => {
      const preloader = preloadOnInteraction(components[0], events);
      preloader.addListeners(target);
    });
  }
};

// Bundles conditionnels pour fonctionnalités avancées
export const conditionalBundles = {
  // Analytics et tracking
  analytics: {
    gtag: () => Promise.resolve({ default: null }),
    mixpanel: () => import('mixpanel-browser')
  },

  // Import sélectif pour lodash
  lodash: {
    debounce: () => import('lodash/debounce'),
    throttle: () => import('lodash/throttle'),
    memoize: () => import('lodash/memoize')
  },
  
  // Import sélectif pour date-fns
  dateFns: {
    format: () => import('date-fns/format'),
    parseISO: () => import('date-fns/parseISO'),
    isValid: () => import('date-fns/isValid')
  },

  // Graphiques et visualisations
  charts: {
    recharts: () => import('recharts'),
    d3: () => import('d3')
  }
};

/**
 * Hook pour charger dynamiquement une bibliothèque
 */
import { useState, useEffect } from 'react';

export const useDynamicImport = <T extends unknown>(
  importFn: () => Promise<T>,
  deps: any[] = []
) => {
  const [state, setState] = useState<{
    loading: boolean;
    error: Error | null;
    data: T | null;
  }>({
    loading: false,
    error: null,
    data: null
  });

  useEffect(() => {
    let cancelled = false;

    const load = async () => {
      setState(prev => ({ ...prev, loading: true, error: null }));
      
      try {
        const result = await importFn();
        if (!cancelled) {
          setState({ loading: false, error: null, data: result });
        }
      } catch (error) {
        if (!cancelled) {
          setState({ loading: false, error: error as Error, data: null });
        }
      }
    };

    load();

    return () => {
      cancelled = true;
    };
  }, deps);

  return state;
};

// Fix de l'import mal placé - déplacer en haut
// import { useState, useEffect } from 'react'; // Déjà importé en haut

// Métriques de performance des bundles
export const bundleMetrics = {
  // Mesurer le temps de chargement d'un composant
  measureLoadTime: (componentName: string) => {
    const startTime = performance.now();
    
    return {
      end: () => {
        const loadTime = performance.now() - startTime;
        console.log(`Bundle ${componentName} loaded in ${loadTime.toFixed(2)}ms`);
        
        // Envoyer aux analytics si disponible
        if ((window as any).gtag) {
          (window as any).gtag('event', 'bundle_load_time', {
            bundle_name: componentName,
            load_time: Math.round(loadTime)
          });
        }
        
        return loadTime;
      }
    };
  },

  // Surveiller la taille des bundles
  trackBundleSize: (bundleName: string, sizeBytes: number) => {
    console.log(`Bundle ${bundleName}: ${(sizeBytes / 1024).toFixed(2)} KB`);
    
    if (sizeBytes > 250 * 1024) { // > 250KB
      console.warn(`Large bundle detected: ${bundleName} (${(sizeBytes / 1024).toFixed(2)} KB)`);
    }
  }
};

export default {
  createLazyComponent,
  createRouteComponent,
  preloadStrategy,
  conditionalBundles,
  bundleMetrics
};