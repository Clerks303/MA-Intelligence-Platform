/**
 * Exports centralisés composants monitoring
 * US-004: Index des composants dashboard
 */

export { default as MonitoringDashboard } from './MonitoringDashboard';
export { default as SystemOverviewCard } from './SystemOverviewCard';
export { default as AlertsOverview } from './AlertsOverview';
export { default as MetricsChart } from './MetricsChart';
export { default as HealthStatus } from './HealthStatus';
export { default as RealTimeIndicator } from './RealTimeIndicator';

// Réexport du context pour faciliter l'accès
export { MonitoringProvider, useMonitoring } from '../../contexts/MonitoringContext';