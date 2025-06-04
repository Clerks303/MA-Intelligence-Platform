/**
 * Store centralisé Zustand pour M&A Intelligence Platform
 * Sprint 1 - Foundation moderne
 */

import { create } from 'zustand';
import { devtools, persist } from 'zustand/middleware';
import { User, ToastMessage, Theme, ModalState, LoadingState } from '../types';

// ============================================================================
// AUTH STORE
// ============================================================================

interface AuthState {
  user: User | null;
  token: string | null;
  isAuthenticated: boolean;
  loading: boolean;
  
  // Actions
  login: (user: User, token: string) => void;
  logout: () => void;
  updateUser: (updates: Partial<User>) => void;
  setLoading: (loading: boolean) => void;
}

export const useAuthStore = create<AuthState>()(
  devtools(
    persist(
      (set, get) => ({
        user: null,
        token: null,
        isAuthenticated: false,
        loading: false,

        login: (user, token) => {
          localStorage.setItem('auth_token', token);
          set({ 
            user, 
            token, 
            isAuthenticated: true, 
            loading: false 
          });
        },

        logout: () => {
          localStorage.removeItem('auth_token');
          set({ 
            user: null, 
            token: null, 
            isAuthenticated: false, 
            loading: false 
          });
        },

        updateUser: (updates) => {
          const currentUser = get().user;
          if (currentUser) {
            set({ user: { ...currentUser, ...updates } });
          }
        },

        setLoading: (loading) => set({ loading }),
      }),
      {
        name: 'ma-intelligence-auth',
        partialize: (state) => ({ 
          user: state.user, 
          token: state.token, 
          isAuthenticated: state.isAuthenticated 
        }),
      }
    ),
    { name: 'AuthStore' }
  )
);

// ============================================================================
// UI STORE
// ============================================================================

interface UIState {
  theme: Theme;
  sidebarOpen: boolean;
  toasts: ToastMessage[];
  modal: ModalState;
  
  // Actions
  setTheme: (theme: Theme) => void;
  toggleSidebar: () => void;
  setSidebarOpen: (open: boolean) => void;
  addToast: (toast: Omit<ToastMessage, 'id'>) => void;
  removeToast: (id: string) => void;
  openModal: (component: React.ComponentType<any>, props?: Record<string, any>) => void;
  closeModal: () => void;
}

export const useUIStore = create<UIState>()(
  devtools(
    persist(
      (set, get) => ({
        theme: 'system',
        sidebarOpen: true,
        toasts: [],
        modal: { isOpen: false },

        setTheme: (theme) => set({ theme }),

        toggleSidebar: () => set((state) => ({ sidebarOpen: !state.sidebarOpen })),

        setSidebarOpen: (open) => set({ sidebarOpen: open }),

        addToast: (toast) => {
          const id = Math.random().toString(36).substr(2, 9);
          const newToast: ToastMessage = {
            ...toast,
            id,
            duration: toast.duration || 5000,
          };
          
          set((state) => ({ toasts: [...state.toasts, newToast] }));
          
          // Auto remove after duration
          setTimeout(() => {
            get().removeToast(id);
          }, newToast.duration);
        },

        removeToast: (id) => set((state) => ({ 
          toasts: state.toasts.filter(toast => toast.id !== id) 
        })),

        openModal: (component, props = {}) => set({ 
          modal: { isOpen: true, component, props } 
        }),

        closeModal: () => set({ 
          modal: { isOpen: false, component: undefined, props: undefined } 
        }),
      }),
      {
        name: 'ma-intelligence-ui',
        partialize: (state) => ({ 
          theme: state.theme, 
          sidebarOpen: state.sidebarOpen 
        }),
      }
    ),
    { name: 'UIStore' }
  )
);

// ============================================================================
// DOCUMENT STORE
// ============================================================================

interface DocumentState {
  selectedDocumentId: string | null;
  uploadProgress: Record<string, number>;
  searchQuery: string;
  filters: Record<string, any>;
  
  // Actions
  setSelectedDocument: (id: string | null) => void;
  setUploadProgress: (fileId: string, progress: number) => void;
  removeUploadProgress: (fileId: string) => void;
  setSearchQuery: (query: string) => void;
  updateFilters: (filters: Record<string, any>) => void;
  clearFilters: () => void;
}

export const useDocumentStore = create<DocumentState>()(
  devtools(
    (set) => ({
      selectedDocumentId: null,
      uploadProgress: {},
      searchQuery: '',
      filters: {},

      setSelectedDocument: (id) => set({ selectedDocumentId: id }),

      setUploadProgress: (fileId, progress) => set((state) => ({
        uploadProgress: { ...state.uploadProgress, [fileId]: progress }
      })),

      removeUploadProgress: (fileId) => set((state) => {
        const { [fileId]: removed, ...rest } = state.uploadProgress;
        return { uploadProgress: rest };
      }),

      setSearchQuery: (query) => set({ searchQuery: query }),

      updateFilters: (newFilters) => set((state) => ({
        filters: { ...state.filters, ...newFilters }
      })),

      clearFilters: () => set({ filters: {}, searchQuery: '' }),
    }),
    { name: 'DocumentStore' }
  )
);

// ============================================================================
// APP STORE (Global App State)
// ============================================================================

interface AppState {
  isInitialized: boolean;
  isOnline: boolean;
  lastActivity: Date | null;
  notifications: any[];
  
  // Actions
  setInitialized: (initialized: boolean) => void;
  setOnlineStatus: (online: boolean) => void;
  updateLastActivity: () => void;
  addNotification: (notification: any) => void;
  removeNotification: (id: string) => void;
  clearNotifications: () => void;
}

export const useAppStore = create<AppState>()(
  devtools(
    (set) => ({
      isInitialized: false,
      isOnline: navigator.onLine,
      lastActivity: null,
      notifications: [],

      setInitialized: (initialized) => set({ isInitialized: initialized }),

      setOnlineStatus: (online) => set({ isOnline: online }),

      updateLastActivity: () => set({ lastActivity: new Date() }),

      addNotification: (notification) => set((state) => ({
        notifications: [...state.notifications, { ...notification, id: Date.now().toString() }]
      })),

      removeNotification: (id) => set((state) => ({
        notifications: state.notifications.filter(n => n.id !== id)
      })),

      clearNotifications: () => set({ notifications: [] }),
    }),
    { name: 'AppStore' }
  )
);

// ============================================================================
// STORE EXPORTS
// ============================================================================

export const stores = {
  auth: useAuthStore,
  ui: useUIStore,
  document: useDocumentStore,
  app: useAppStore,
};

export default stores;