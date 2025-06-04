/**
 * Index du module Analytics Avancé - M&A Intelligence Platform
 * Sprint 6 - Exports pour le module d'analytics sophistiqué
 */

// Composants principaux
export { default as AnalyticsDashboard } from './components/AnalyticsDashboard';
export { default as AdvancedMetricsGrid } from './components/AdvancedMetricsGrid';
export { default as InteractiveChart } from './components/InteractiveChart';

// Hooks
export { useAdvancedAnalytics, useRealTimeMetrics, useBusinessAnalytics } from './hooks/useAdvancedAnalytics';

// Services
export { analyticsService, generateMockAnalyticsData } from './services/analyticsService';

// Types
export type {
  AnalyticsMetric,
  ChartData,
  ChartConfig,
  AnalyticsInsight,
  AnalyticsDashboard as AnalyticsDashboardType,
  AnalyticsSection,
  AnalyticsWidget,
  AnalyticsFilter,
  PredictiveModel,
  SegmentationAnalysis,
  BusinessMetrics,
  RealTimeMetrics,
  AnalyticsReport,
  UseAnalyticsReturn
} from './types';

// Constantes et utilitaires
export {
  CHART_COLORS,
  METRIC_CATEGORIES,
  INSIGHT_TYPES
} from './types';

// Fonctions utilitaires pour l'intégration
export const analyticsUtils = {
  // Formatage des valeurs selon l'unité
  formatValue: (value: number, unit: 'number' | 'currency' | 'percentage' | 'duration'): string => {
    switch (unit) {
      case 'currency':
        return new Intl.NumberFormat('fr-FR', {
          style: 'currency',
          currency: 'EUR',
          minimumFractionDigits: 0,
          maximumFractionDigits: 0
        }).format(value);
      case 'percentage':
        return `${value.toFixed(1)}%`;
      case 'duration':
        if (value < 1000) return `${value.toFixed(0)}ms`;
        if (value < 60000) return `${(value / 1000).toFixed(1)}s`;
        return `${(value / 60000).toFixed(1)}min`;
      default:
        return value.toLocaleString('fr-FR');
    }
  },

  // Calcul de tendance
  calculateTrend: (current: number, previous: number): { trend: 'up' | 'down' | 'stable', value: number } => {
    if (previous === 0) return { trend: 'stable', value: 0 };
    
    const change = ((current - previous) / previous) * 100;
    
    if (Math.abs(change) < 0.1) return { trend: 'stable', value: 0 };
    
    return {
      trend: change > 0 ? 'up' : 'down',
      value: Math.abs(change)
    };
  },

  // Génération de couleurs pour graphiques
  generateColors: (count: number): string[] => {
    const baseColors = [
      '#3B82F6', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6',
      '#06B6D4', '#84CC16', '#F97316', '#EC4899', '#6366F1'
    ];
    
    if (count <= baseColors.length) {
      return baseColors.slice(0, count);
    }
    
    // Générer des couleurs supplémentaires si nécessaire
    const additionalColors = [];
    for (let i = baseColors.length; i < count; i++) {
      const hue = (i * 137.5) % 360; // Golden angle pour distribution uniforme
      additionalColors.push(`hsl(${hue}, 70%, 50%)`);
    }
    
    return [...baseColors, ...additionalColors];
  },

  // Validation de configuration de graphique
  validateChartConfig: (config: Partial<ChartConfig>): ChartConfig => {
    const defaults: ChartConfig = {
      interactive: true,
      zoom: false,
      brush: false,
      tooltip: { enabled: true },
      legend: { enabled: true, position: 'top' },
      animation: { enabled: true, duration: 1000, easing: 'ease-in-out' },
      colors: CHART_COLORS
    };

    return { ...defaults, ...config };
  },

  // Détection d'anomalies simples
  detectAnomalies: (data: number[], threshold: number = 2): number[] => {
    if (data.length < 3) return [];
    
    const mean = data.reduce((sum, val) => sum + val, 0) / data.length;
    const variance = data.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / data.length;
    const stdDev = Math.sqrt(variance);
    
    return data
      .map((value, index) => ({ value, index }))
      .filter(item => Math.abs(item.value - mean) > threshold * stdDev)
      .map(item => item.index);
  },

  // Génération d'insights automatiques
  generateInsights: (metrics: AnalyticsMetric[]): AnalyticsInsight[] => {
    const insights: AnalyticsInsight[] = [];
    const now = new Date().toISOString();

    // Identifier les métriques avec forte croissance
    const highGrowthMetrics = metrics.filter(m => m.trend === 'up' && m.trendValue > 10);
    if (highGrowthMetrics.length > 0) {
      insights.push({
        id: `insight_growth_${Date.now()}`,
        type: 'trend',
        title: 'Forte croissance détectée',
        description: `${highGrowthMetrics.length} métrique(s) montrent une croissance supérieure à 10%.`,
        confidence: 0.85,
        severity: 'success',
        timestamp: now
      });
    }

    // Identifier les métriques critiques
    const criticalMetrics = metrics.filter(m => 
      m.threshold && m.value <= m.threshold.critical
    );
    if (criticalMetrics.length > 0) {
      insights.push({
        id: `insight_critical_${Date.now()}`,
        type: 'anomaly',
        title: 'Métriques critiques détectées',
        description: `${criticalMetrics.length} métrique(s) sont en dessous du seuil critique.`,
        confidence: 0.95,
        severity: 'critical',
        timestamp: now
      });
    }

    return insights;
  },

  // Export de données en différents formats
  exportToCSV: (data: any[], filename: string = 'analytics_data.csv'): void => {
    if (!data || data.length === 0) return;

    const headers = Object.keys(data[0]).join(',');
    const rows = data.map(row => 
      Object.values(row).map(value => 
        typeof value === 'string' && value.includes(',') ? `"${value}"` : value
      ).join(',')
    );

    const csvContent = [headers, ...rows].join('\n');
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    
    const link = document.createElement('a');
    const url = URL.createObjectURL(blob);
    link.setAttribute('href', url);
    link.setAttribute('download', filename);
    link.style.visibility = 'hidden';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  }
};

// Configuration par défaut pour le module
export const analyticsConfig = {
  // Couleurs par défaut
  defaultColors: CHART_COLORS,
  
  // Intervalles de rafraîchissement
  refreshIntervals: {
    realTime: 3000,      // 3 secondes
    metrics: 30000,      // 30 secondes  
    business: 300000,    // 5 minutes
    insights: 60000      // 1 minute
  },
  
  // Limites de données
  dataLimits: {
    maxDataPoints: 1000,
    maxCharts: 20,
    maxMetrics: 50
  },
  
  // Configuration cache
  cache: {
    enabled: true,
    duration: 5 * 60 * 1000, // 5 minutes
    maxSize: 100
  },
  
  // Configuration animations
  animations: {
    enabled: true,
    duration: 1000,
    easing: 'ease-in-out',
    staggerDelay: 100
  }
};

export default {
  AnalyticsDashboard,
  AdvancedMetricsGrid,
  InteractiveChart,
  useAdvancedAnalytics,
  analyticsService,
  analyticsUtils,
  analyticsConfig
};