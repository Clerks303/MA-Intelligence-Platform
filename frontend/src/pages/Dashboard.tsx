/**
 * Page Dashboard Modernisée - M&A Intelligence Platform
 * Sprint 2 - Utilisation du nouveau module Dashboard
 */

import React from 'react';
import { Dashboard as DashboardModule } from '../features/dashboard';

export default function Dashboard() {
  return (
    <div className="min-h-screen bg-background p-6">
      <div className="max-w-7xl mx-auto">
        <DashboardModule />
      </div>
    </div>
  );
}