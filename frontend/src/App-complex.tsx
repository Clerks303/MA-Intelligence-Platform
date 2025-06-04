import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ReactQueryDevtools } from '@tanstack/react-query-devtools';

// Hooks and stores
import { useAuth } from './hooks';

// Components
import Layout from './components/Layout';
import Login from './pages/Login';
import Dashboard from './pages/Dashboard';
import Companies from './pages/Companies';
import Scraping from './pages/Scraping';
import Settings from './pages/Settings';

// Design System Demo (Sprint 1)
import { DesignSystemDemo } from './components/DesignSystemDemo';

// Loading component
function LoadingSpinner() {
  return (
    <div className="min-h-screen bg-background flex items-center justify-center">
      <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
    </div>
  );
}

// Query client with optimized defaults for M&A Intelligence
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      retry: 1,
      staleTime: 5 * 60 * 1000, // 5 minutes
      gcTime: 10 * 60 * 1000, // 10 minutes (formerly cacheTime)
    },
    mutations: {
      retry: 1,
    },
  },
});

// Protected Route component using Zustand auth store
function ProtectedRoute({ children }: { children: React.ReactNode }) {
  const { isAuthenticated, loading } = useAuth();
  
  if (loading) {
    return <LoadingSpinner />;
  }
  
  return isAuthenticated ? <>{children}</> : <Navigate to="/login" />;
}

// Theme wrapper component
function ThemeWrapper({ children }: { children: React.ReactNode }) {
  return (
    <div className="min-h-screen bg-background text-foreground font-sans antialiased">
      {children}
    </div>
  );
}

// Main App component with modern architecture
function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <Router>
        <ThemeWrapper>
          <Routes>
            {/* Public routes */}
            <Route path="/login" element={<Login />} />
            
            {/* Design system demo for Sprint 1 */}
            <Route path="/design-demo" element={<DesignSystemDemo />} />
            
            {/* Protected routes */}
            <Route
              path="/"
              element={
                <ProtectedRoute>
                  <Layout />
                </ProtectedRoute>
              }
            >
              <Route index element={<Navigate to="/dashboard" />} />
              <Route path="dashboard" element={<Dashboard />} />
              <Route path="companies" element={<Companies />} />
              <Route path="scraping" element={<Scraping />} />
              <Route path="settings" element={<Settings />} />
            </Route>
            
            {/* Catch all route */}
            <Route path="*" element={<Navigate to="/dashboard" />} />
          </Routes>
        </ThemeWrapper>
      </Router>
      
      {/* React Query devtools - only in development */}
      {process.env.NODE_ENV === 'development' && (
        <ReactQueryDevtools initialIsOpen={false} />
      )}
    </QueryClientProvider>
  );
}

export default App;