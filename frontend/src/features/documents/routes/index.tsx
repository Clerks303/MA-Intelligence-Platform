/**
 * Routes Documents - M&A Intelligence Platform
 * Sprint 3 - Routes avec lazy loading et authentification
 */

import React, { Suspense } from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import { ErrorBoundary } from 'react-error-boundary';

// Lazy loading des pages
const DocumentManagement = React.lazy(() => 
  import('../pages/DocumentManagement').then(module => ({ 
    default: module.DocumentManagement 
  }))
);

interface DocumentRoutesProps {
  basePath?: string;
}

// Composant de fallback pour le chargement
const DocumentLoadingFallback: React.FC = () => (
  <div className="h-full flex items-center justify-center">
    <div className="text-center">
      <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500 mx-auto mb-4"></div>
      <p className="text-gray-600">Chargement du module documentaire...</p>
    </div>
  </div>
);

// Composant de fallback pour les erreurs
const DocumentErrorFallback: React.FC<{ error: Error; resetErrorBoundary: () => void }> = ({ 
  error, 
  resetErrorBoundary 
}) => (
  <div className="h-full flex items-center justify-center">
    <div className="text-center max-w-md">
      <div className="text-red-500 mb-4">
        <svg className="w-16 h-16 mx-auto" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
      </div>
      <h2 className="text-xl font-semibold text-gray-900 mb-2">
        Erreur du module documentaire
      </h2>
      <p className="text-gray-600 mb-4">
        Une erreur est survenue lors du chargement du module de gestion documentaire.
      </p>
      <details className="text-sm text-left bg-gray-50 p-3 rounded mb-4">
        <summary className="cursor-pointer font-medium">Détails techniques</summary>
        <pre className="mt-2 text-xs overflow-auto">{error.message}</pre>
      </details>
      <button
        onClick={resetErrorBoundary}
        className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 transition-colors"
      >
        Réessayer
      </button>
    </div>
  </div>
);

// Composant principal des routes
export const DocumentRoutes: React.FC<DocumentRoutesProps> = ({ basePath = "/documents" }) => {
  return (
    <ErrorBoundary
      FallbackComponent={DocumentErrorFallback}
      onReset={() => window.location.reload()}
    >
      <Suspense fallback={<DocumentLoadingFallback />}>
        <Routes>
          {/* Route principale */}
          <Route 
            index 
            element={
              <DocumentManagement
                enableAnalytics={true}
                enableUpload={true}
                compactMode={false}
              />
            } 
          />
          
          {/* Route de gestion compacte (pour mobile ou sidebars) */}
          <Route 
            path="compact" 
            element={
              <DocumentManagement
                enableAnalytics={false}
                enableUpload={true}
                compactMode={true}
              />
            } 
          />
          
          {/* Route lecture seule */}
          <Route 
            path="readonly" 
            element={
              <DocumentManagement
                enableAnalytics={true}
                enableUpload={false}
                compactMode={false}
              />
            } 
          />
          
          {/* Redirection par défaut */}
          <Route path="*" element={<Navigate to={basePath} replace />} />
        </Routes>
      </Suspense>
    </ErrorBoundary>
  );
};

export default DocumentRoutes;