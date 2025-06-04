import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ReactQueryDevtools } from '@tanstack/react-query-devtools';

// Simple components that should work
import Login from './pages/Login';
import Dashboard from './pages/Dashboard';
import Companies from './pages/Companies';

// Loading component
function LoadingSpinner() {
  return (
    <div className="min-h-screen bg-white flex items-center justify-center">
      <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
    </div>
  );
}

// Query client with basic defaults
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      retry: 1,
      staleTime: 5 * 60 * 1000,
      gcTime: 10 * 60 * 1000,
    },
  },
});

// Simple auth check - just check if token exists
function useSimpleAuth() {
  const [isAuthenticated, setIsAuthenticated] = React.useState(false);
  const [loading, setLoading] = React.useState(true);

  React.useEffect(() => {
    const token = localStorage.getItem('auth_token');
    setIsAuthenticated(!!token);
    setLoading(false);
  }, []);

  const login = (token: string) => {
    localStorage.setItem('auth_token', token);
    setIsAuthenticated(true);
  };

  const logout = () => {
    localStorage.removeItem('auth_token');
    setIsAuthenticated(false);
  };

  return { isAuthenticated, loading, login, logout };
}

// Protected Route component
function ProtectedRoute({ children }: { children: React.ReactNode }) {
  const { isAuthenticated, loading } = useSimpleAuth();
  
  if (loading) {
    return <LoadingSpinner />;
  }
  
  return isAuthenticated ? <>{children}</> : <Navigate to="/login" />;
}

// Simple Layout component
function SimpleLayout({ children }: { children: React.ReactNode }) {
  const { logout } = useSimpleAuth();
  
  return (
    <div className="min-h-screen bg-gray-50">
      <nav className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between h-16">
            <div className="flex items-center">
              <h1 className="text-xl font-semibold text-gray-900">M&A Intelligence</h1>
            </div>
            <div className="flex items-center space-x-4">
              <button
                onClick={logout}
                className="bg-red-600 text-white px-4 py-2 rounded-md text-sm hover:bg-red-700"
              >
                Logout
              </button>
            </div>
          </div>
        </div>
      </nav>
      <main className="max-w-7xl mx-auto py-6 sm:px-6 lg:px-8">
        {children}
      </main>
    </div>
  );
}

// Main App component
function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <Router>
        <Routes>
          {/* Public routes */}
          <Route path="/login" element={<Login />} />
          
          {/* Protected routes */}
          <Route
            path="/"
            element={
              <ProtectedRoute>
                <SimpleLayout>
                  <Dashboard />
                </SimpleLayout>
              </ProtectedRoute>
            }
          />
          <Route
            path="/dashboard"
            element={
              <ProtectedRoute>
                <SimpleLayout>
                  <Dashboard />
                </SimpleLayout>
              </ProtectedRoute>
            }
          />
          <Route
            path="/companies"
            element={
              <ProtectedRoute>
                <SimpleLayout>
                  <Companies />
                </SimpleLayout>
              </ProtectedRoute>
            }
          />
          
          {/* Catch all route */}
          <Route path="*" element={<Navigate to="/dashboard" />} />
        </Routes>
      </Router>
      
      {/* React Query devtools */}
      {process.env.NODE_ENV === 'development' && (
        <ReactQueryDevtools initialIsOpen={false} />
      )}
    </QueryClientProvider>
  );
}

export default App;