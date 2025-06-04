/**
 * Module Dashboard - M&A Intelligence Platform
 * Sprint 2 - Exports centralisés
 */

// Types
export * from './types';

// Hooks
export * from './hooks/useDashboardData';

// Services
export { dashboardService, mockDashboardData } from './services/dashboardService';

// Components
export { Dashboard } from './components/Dashboard';
export { KPIWidget, KPIGrid } from './components/KPIWidget';
export { ChartWidget, ChartGrid } from './components/ChartWidget';
export { AlertsWidget, AlertsBadge } from './components/AlertsWidget';
export { SLAWidget, DataQualityWidget } from './components/SLAWidget';
export { 
  DashboardLayout, 
  ResponsiveGrid, 
  ColumnsLayout 
} from './components/DashboardLayout';