/**
 * Custom Hooks centralisés pour M&A Intelligence Platform
 * Sprint 1 - Foundation moderne
 */

import { useEffect, useState, useCallback, useRef } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { useAuthStore, useUIStore, useAppStore } from '../stores';
import { ApiResponse, LoadingState } from '../types';

// ============================================================================
// AUTH HOOKS
// ============================================================================

export const useAuth = () => {
  const { 
    user, 
    token, 
    isAuthenticated, 
    loading, 
    login, 
    logout, 
    updateUser, 
    setLoading 
  } = useAuthStore();

  const checkAuthStatus = useCallback(async () => {
    const savedToken = localStorage.getItem('auth_token');
    if (savedToken && !isAuthenticated) {
      setLoading(true);
      try {
        // Validate token with backend
        // const response = await api.validateToken(savedToken);
        // if (response.success) {
        //   login(response.data.user, savedToken);
        // } else {
        //   logout();
        // }
      } catch (error) {
        logout();
      } finally {
        setLoading(false);
      }
    }
  }, [isAuthenticated, login, logout, setLoading]);

  useEffect(() => {
    checkAuthStatus();
  }, [checkAuthStatus]);

  return {
    user,
    token,
    isAuthenticated,
    loading,
    login,
    logout,
    updateUser,
    checkAuthStatus,
  };
};

// ============================================================================
// UI HOOKS
// ============================================================================

export const useToast = () => {
  const { addToast, removeToast } = useUIStore();

  const toast = useCallback((
    message: string,
    type: 'success' | 'error' | 'warning' | 'info' = 'info',
    options?: { title?: string; duration?: number }
  ) => {
    addToast({
      type,
      title: options?.title || type.charAt(0).toUpperCase() + type.slice(1),
      description: message,
      duration: options?.duration,
    });
  }, [addToast]);

  return {
    toast,
    success: (message: string, options?: { title?: string; duration?: number }) => 
      toast(message, 'success', options),
    error: (message: string, options?: { title?: string; duration?: number }) => 
      toast(message, 'error', options),
    warning: (message: string, options?: { title?: string; duration?: number }) => 
      toast(message, 'warning', options),
    info: (message: string, options?: { title?: string; duration?: number }) => 
      toast(message, 'info', options),
    dismiss: removeToast,
  };
};

export const useModal = () => {
  const { modal, openModal, closeModal } = useUIStore();

  return {
    isOpen: modal.isOpen,
    component: modal.component,
    props: modal.props,
    open: openModal,
    close: closeModal,
  };
};

export const useTheme = () => {
  const { theme, setTheme } = useUIStore();

  const toggleTheme = useCallback(() => {
    setTheme(theme === 'light' ? 'dark' : 'light');
  }, [theme, setTheme]);

  return {
    theme,
    setTheme,
    toggleTheme,
  };
};

// ============================================================================
// API HOOKS
// ============================================================================

export const useApi = <T = any>(
  key: string | string[],
  fetcher: () => Promise<ApiResponse<T>>,
  options?: {
    enabled?: boolean;
    staleTime?: number;
    cacheTime?: number;
    refetchOnWindowFocus?: boolean;
  }
) => {
  const { toast } = useToast();

  return useQuery({
    queryKey: Array.isArray(key) ? key : [key],
    queryFn: async () => {
      try {
        const response = await fetcher();
        if (response.success) {
          return response.data;
        } else {
          throw new Error(response.message || 'API Error');
        }
      } catch (error) {
        toast(error instanceof Error ? error.message : 'Unknown error', 'error');
        throw error;
      }
    },
    enabled: options?.enabled !== false,
    staleTime: options?.staleTime || 5 * 60 * 1000, // 5 minutes
    gcTime: options?.cacheTime || 10 * 60 * 1000, // 10 minutes
    refetchOnWindowFocus: options?.refetchOnWindowFocus || false,
  });
};

export const useApiMutation = <T = any, V = any>(
  mutationFn: (variables: V) => Promise<ApiResponse<T>>,
  options?: {
    onSuccess?: (data: T, variables: V) => void;
    onError?: (error: Error, variables: V) => void;
    invalidateKeys?: string[][];
  }
) => {
  const queryClient = useQueryClient();
  const { toast } = useToast();

  return useMutation({
    mutationFn: async (variables: V): Promise<T> => {
      try {
        const response = await mutationFn(variables);
        if (response.success) {
          return response.data as T;
        } else {
          throw new Error(response.message || 'Mutation failed');
        }
      } catch (error) {
        const message = error instanceof Error ? error.message : 'Unknown error';
        toast(message, 'error');
        throw error;
      }
    },
    onSuccess: (data: T, variables: V) => {
      if (options?.onSuccess) {
        options.onSuccess(data, variables);
      }
      if (options?.invalidateKeys) {
        options.invalidateKeys.forEach(key => {
          queryClient.invalidateQueries({ queryKey: key });
        });
      }
    },
    onError: (error: Error, variables) => {
      if (options?.onError) {
        options.onError(error, variables);
      }
    },
  });
};

// ============================================================================
// UTILITY HOOKS
// ============================================================================

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

export const useLocalStorage = <T>(
  key: string,
  initialValue: T
): [T, (value: T | ((val: T) => T)) => void] => {
  const [storedValue, setStoredValue] = useState<T>(() => {
    try {
      const item = window.localStorage.getItem(key);
      return item ? JSON.parse(item) : initialValue;
    } catch (error) {
      console.error(`Error reading localStorage key "${key}":`, error);
      return initialValue;
    }
  });

  const setValue = useCallback((value: T | ((val: T) => T)) => {
    try {
      const valueToStore = value instanceof Function ? value(storedValue) : value;
      setStoredValue(valueToStore);
      window.localStorage.setItem(key, JSON.stringify(valueToStore));
    } catch (error) {
      console.error(`Error setting localStorage key "${key}":`, error);
    }
  }, [key, storedValue]);

  return [storedValue, setValue];
};

export const useOnlineStatus = () => {
  const { isOnline, setOnlineStatus } = useAppStore();

  useEffect(() => {
    const handleOnline = () => setOnlineStatus(true);
    const handleOffline = () => setOnlineStatus(false);

    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);

    return () => {
      window.removeEventListener('online', handleOnline);
      window.removeEventListener('offline', handleOffline);
    };
  }, [setOnlineStatus]);

  return isOnline;
};

export const useClickOutside = (
  ref: React.RefObject<HTMLElement>,
  callback: () => void
) => {
  useEffect(() => {
    const handleClick = (event: MouseEvent) => {
      if (ref.current && !ref.current.contains(event.target as Node)) {
        callback();
      }
    };

    document.addEventListener('mousedown', handleClick);
    return () => {
      document.removeEventListener('mousedown', handleClick);
    };
  }, [ref, callback]);
};

export const useAsyncState = <T>(
  initialData: T | null = null
): [
  { data: T | null; loading: boolean; error: string | null },
  {
    setData: (data: T) => void;
    setLoading: (loading: boolean) => void;
    setError: (error: string | null) => void;
    reset: () => void;
  }
] => {
  const [state, setState] = useState<{
    data: T | null;
    loading: boolean;
    error: string | null;
  }>({
    data: initialData,
    loading: false,
    error: null,
  });

  const setData = useCallback((data: T) => {
    setState(prev => ({ ...prev, data, loading: false, error: null }));
  }, []);

  const setLoading = useCallback((loading: boolean) => {
    setState(prev => ({ ...prev, loading }));
  }, []);

  const setError = useCallback((error: string | null) => {
    setState(prev => ({ ...prev, error, loading: false }));
  }, []);

  const reset = useCallback(() => {
    setState({ data: initialData, loading: false, error: null });
  }, [initialData]);

  return [state, { setData, setLoading, setError, reset }];
};

// ============================================================================
// PERFORMANCE HOOKS
// ============================================================================

export const useIsomorphicLayoutEffect = 
  typeof window !== 'undefined' ? useEffect : useEffect;

export const usePrevious = <T>(value: T): T | undefined => {
  const ref = useRef<T>();
  useEffect(() => {
    ref.current = value;
  });
  return ref.current;
};

export const useUpdateEffect = (effect: React.EffectCallback, deps?: React.DependencyList) => {
  const isFirstMount = useRef(true);

  useEffect(() => {
    if (isFirstMount.current) {
      isFirstMount.current = false;
    } else {
      return effect();
    }
  }, deps);
};