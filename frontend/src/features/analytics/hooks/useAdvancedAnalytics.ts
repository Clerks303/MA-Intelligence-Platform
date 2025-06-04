/**
 * Hook Analytics Avancé - M&A Intelligence Platform
 * Sprint 6 - Hook principal pour gestion state analytics et données temps réel
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { 
  AnalyticsMetric, 
  ChartData, 
  AnalyticsInsight, 
  RealTimeMetrics,
  BusinessMetrics,
  AnalyticsFilter,
  UseAnalyticsReturn 
} from '../types';
import { analyticsService } from '../services/analyticsService';

interface UseAdvancedAnalyticsOptions {
  enableRealTime?: boolean;
  refreshInterval?: number;
  autoRefresh?: boolean;
  maxRetries?: number;
}

const DEFAULT_OPTIONS: UseAdvancedAnalyticsOptions = {
  enableRealTime: true,
  refreshInterval: 30000, // 30 secondes
  autoRefresh: true,
  maxRetries: 3
};

export const useAdvancedAnalytics = (
  initialFilters: AnalyticsFilter[] = [],
  options: UseAdvancedAnalyticsOptions = {}
): UseAnalyticsReturn => {
  const opts = { ...DEFAULT_OPTIONS, ...options };
  const queryClient = useQueryClient();
  
  // State local
  const [filters, setFilters] = useState<AnalyticsFilter[]>(initialFilters);
  const [isExporting, setIsExporting] = useState(false);
  const [lastUpdated, setLastUpdated] = useState<string | null>(null);
  
  // Refs pour gestion des intervals
  const realTimeIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const refreshIntervalRef = useRef<NodeJS.Timeout | null>(null);

  // Query pour les métriques principales
  const {
    data: metrics = [],
    isLoading: isLoadingMetrics,
    error: metricsError,
    refetch: refetchMetrics
  } = useQuery({
    queryKey: ['analytics', 'metrics', filters],
    queryFn: () => analyticsService.getMetrics(filters),
    staleTime: 5 * 60 * 1000, // 5 minutes
    gcTime: 10 * 60 * 1000, // 10 minutes
    retry: opts.maxRetries,
    enabled: true
  });

  // Query pour les graphiques
  const {
    data: charts = [],
    isLoading: isLoadingCharts,
    error: chartsError,
    refetch: refetchCharts
  } = useQuery({
    queryKey: ['analytics', 'charts', filters],
    queryFn: () => analyticsService.getCharts(filters),
    staleTime: 5 * 60 * 1000,
    gcTime: 10 * 60 * 1000,
    retry: opts.maxRetries,
    enabled: true
  });

  // Query pour les insights
  const {
    data: insights = [],
    isLoading: isLoadingInsights,
    error: insightsError,
    refetch: refetchInsights
  } = useQuery({
    queryKey: ['analytics', 'insights'],
    queryFn: () => analyticsService.getInsights(),
    staleTime: 2 * 60 * 1000, // 2 minutes pour les insights
    gcTime: 5 * 60 * 1000,
    retry: opts.maxRetries,
    enabled: true
  });

  // Query pour les métriques business
  const {
    data: businessMetrics,
    isLoading: isLoadingBusiness,
    error: businessError,
    refetch: refetchBusinessMetrics
  } = useQuery({
    queryKey: ['analytics', 'business'],
    queryFn: () => analyticsService.getBusinessMetrics(),
    staleTime: 10 * 60 * 1000, // 10 minutes
    gcTime: 20 * 60 * 1000,
    retry: opts.maxRetries,
    enabled: true
  });

  // State pour les métriques temps réel
  const [realTimeMetrics, setRealTimeMetrics] = useState<RealTimeMetrics>({
    activeUsers: 0,
    requestsPerSecond: 0,
    responseTime: 0,
    errorRate: 0,
    dataProcessingRate: 0,
    queueSize: 0,
    lastUpdated: new Date().toISOString()
  });

  // Fonction pour rafraîchir les métriques temps réel
  const updateRealTimeMetrics = useCallback(async () => {
    if (!opts.enableRealTime) return;
    
    try {
      const newMetrics = await analyticsService.getRealTimeMetrics();
      setRealTimeMetrics(newMetrics);
      setLastUpdated(new Date().toISOString());
    } catch (error) {
      console.error('Error updating real-time metrics:', error);
    }
  }, [opts.enableRealTime]);

  // Démarrer les updates temps réel
  useEffect(() => {
    if (opts.enableRealTime) {
      // Update initial
      updateRealTimeMetrics();
      
      // Interval pour updates réguliers
      realTimeIntervalRef.current = setInterval(updateRealTimeMetrics, 5000); // 5 secondes
      
      return () => {
        if (realTimeIntervalRef.current) {
          clearInterval(realTimeIntervalRef.current);
        }
      };
    }
  }, [opts.enableRealTime, updateRealTimeMetrics]);

  // Auto-refresh général
  useEffect(() => {
    if (opts.autoRefresh && opts.refreshInterval) {
      refreshIntervalRef.current = setInterval(() => {
        refetchMetrics();
        refetchCharts();
        refetchInsights();
        refetchBusinessMetrics();
      }, opts.refreshInterval);
      
      return () => {
        if (refreshIntervalRef.current) {
          clearInterval(refreshIntervalRef.current);
        }
      };
    }
  }, [opts.autoRefresh, opts.refreshInterval, refetchMetrics, refetchCharts, refetchInsights, refetchBusinessMetrics]);

  // Fonction de rafraîchissement global
  const refreshData = useCallback(async () => {
    try {
      await Promise.all([
        refetchMetrics(),
        refetchCharts(),
        refetchInsights(),
        refetchBusinessMetrics(),
        updateRealTimeMetrics()
      ]);
      setLastUpdated(new Date().toISOString());
    } catch (error) {
      console.error('Error refreshing analytics data:', error);
      throw error;
    }
  }, [refetchMetrics, refetchCharts, refetchInsights, refetchBusinessMetrics, updateRealTimeMetrics]);

  // Fonction d'export
  const exportData = useCallback(async (format: 'csv' | 'json' | 'excel') => {
    setIsExporting(true);
    
    try {
      const dataToExport = {
        metrics,
        charts: charts.map(chart => ({
          id: chart.id,
          title: chart.title,
          type: chart.type,
          data: chart.data
        })),
        insights,
        businessMetrics,
        realTimeMetrics,
        exportDate: new Date().toISOString(),
        filters
      };

      const blob = await analyticsService.exportData(format, dataToExport);
      
      // Télécharger le fichier
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `analytics_${new Date().toISOString().split('T')[0]}.${format}`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
      
    } catch (error) {
      console.error('Error exporting data:', error);
      throw error;
    } finally {
      setIsExporting(false);
    }
  }, [metrics, charts, insights, businessMetrics, realTimeMetrics, filters]);

  // Fonction pour créer une alerte
  const createAlert = useCallback(async (metric: string, condition: any) => {
    try {
      // Simulation de création d'alerte
      console.log('Creating alert for metric:', metric, 'with condition:', condition);
      
      // En production, appel API réel
      // await analyticsService.createAlert(metric, condition);
      
      // Rafraîchir les insights pour voir la nouvelle alerte
      await refetchInsights();
      
      return true;
    } catch (error) {
      console.error('Error creating alert:', error);
      throw error;
    }
  }, [refetchInsights]);

  // Gestion des filtres
  const updateFilter = useCallback((id: string, value: any) => {
    setFilters(prev => 
      prev.map(filter => 
        filter.id === id ? { ...filter, value } : filter
      )
    );
    
    // Invalider les queries pour forcer un refetch
    queryClient.invalidateQueries({ queryKey: ['analytics'] });
  }, [queryClient]);

  const resetFilters = useCallback(() => {
    setFilters(initialFilters);
    queryClient.invalidateQueries({ queryKey: ['analytics'] });
  }, [initialFilters, queryClient]);

  // État de chargement global
  const isLoading = isLoadingMetrics || isLoadingCharts || isLoadingInsights || isLoadingBusiness;
  
  // Erreur globale
  const error = metricsError || chartsError || insightsError || businessError;
  const errorMessage = error ? (error as Error).message : null;

  // Nettoyage des intervals au démontage
  useEffect(() => {
    return () => {
      if (realTimeIntervalRef.current) {
        clearInterval(realTimeIntervalRef.current);
      }
      if (refreshIntervalRef.current) {
        clearInterval(refreshIntervalRef.current);
      }
    };
  }, []);

  return {
    // Données
    metrics,
    charts,
    insights,
    realTimeMetrics,
    businessMetrics,
    
    // État
    isLoading,
    error: errorMessage,
    lastUpdated,
    isExporting,
    
    // Actions
    refreshData,
    exportData,
    createAlert,
    
    // Filtres
    filters,
    updateFilter,
    resetFilters
  };
};

// Hook spécialisé pour métriques temps réel uniquement
export const useRealTimeMetrics = (interval: number = 5000) => {
  const [metrics, setMetrics] = useState<RealTimeMetrics | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);

  const startUpdates = useCallback(() => {
    setIsConnected(true);
    
    const updateMetrics = async () => {
      try {
        const newMetrics = await analyticsService.getRealTimeMetrics();
        setMetrics(newMetrics);
      } catch (error) {
        console.error('Error fetching real-time metrics:', error);
        setIsConnected(false);
      }
    };

    updateMetrics(); // Update initial
    intervalRef.current = setInterval(updateMetrics, interval);
  }, [interval]);

  const stopUpdates = useCallback(() => {
    setIsConnected(false);
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
  }, []);

  useEffect(() => {
    startUpdates();
    return () => stopUpdates();
  }, [startUpdates, stopUpdates]);

  return {
    metrics,
    isConnected,
    startUpdates,
    stopUpdates
  };
};

// Hook pour analytics business
export const useBusinessAnalytics = () => {
  const { data, isLoading, error, refetch } = useQuery({
    queryKey: ['analytics', 'business', 'detailed'],
    queryFn: () => analyticsService.getBusinessMetrics(),
    staleTime: 15 * 60 * 1000, // 15 minutes
    gcTime: 30 * 60 * 1000, // 30 minutes
  });

  return {
    businessMetrics: data,
    isLoading,
    error: error ? (error as Error).message : null,
    refresh: refetch
  };
};

export default useAdvancedAnalytics;