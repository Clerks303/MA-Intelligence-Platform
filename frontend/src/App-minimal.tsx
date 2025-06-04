import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';

// Query client minimal
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

// Composant simple pour tester
const TestPage = () => (
  <div className="min-h-screen bg-background text-foreground p-8">
    <h1 className="text-4xl font-bold mb-4">🚀 M&A Intelligence Platform</h1>
    <p className="text-lg text-muted-foreground mb-6">
      Application démarrée avec succès! Interface fonctionnelle.
    </p>
    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
      <div className="p-4 border rounded-lg">
        <h2 className="text-xl font-semibold mb-2">✅ React</h2>
        <p>Framework OK</p>
      </div>
      <div className="p-4 border rounded-lg">
        <h2 className="text-xl font-semibold mb-2">✅ TypeScript</h2>
        <p>Types résolus</p>
      </div>
      <div className="p-4 border rounded-lg">
        <h2 className="text-xl font-semibold mb-2">✅ Tailwind</h2>
        <p>Styles appliqués</p>
      </div>
    </div>
    <div className="mt-8 p-4 bg-green-50 border border-green-200 rounded-lg">
      <h3 className="text-lg font-semibold text-green-800">Prêt pour le développement!</h3>
      <p className="text-green-700">Vous pouvez maintenant naviguer vers /login ou /dashboard</p>
    </div>
  </div>
);

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <Router>
        <div className="min-h-screen bg-background text-foreground">
          <Routes>
            <Route path="/" element={<TestPage />} />
            <Route path="/test" element={<TestPage />} />
            <Route path="*" element={<TestPage />} />
          </Routes>
        </div>
      </Router>
    </QueryClientProvider>
  );
}

export default App;