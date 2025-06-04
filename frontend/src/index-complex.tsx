import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import App from './App';
import { validateEnvironment } from './utils/envValidation';

// Validate environment before app start
try {
  validateEnvironment();
} catch (error) {
  console.error('Failed to start app due to environment issues:', (error as Error).message);
  // In development, we can continue anyway
  if (process.env.NODE_ENV === 'production') {
    throw error;
  }
}

const root = ReactDOM.createRoot(
  document.getElementById('root') as HTMLElement
);

root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);