/**
 * Hook centralisé pour les données du Dashboard
 * Sprint 2 - Dashboard Central
 */

import { useQuery, useQueryClient } from '@tanstack/react-query';
import { useCallback, useMemo } from 'react';
import { 
  DashboardDataResponse, 
  DashboardFilters, 
  DashboardKPI, 
  DashboardChart,
  AlertLevel,
  SLAIndicator,
  DataQualityMetric,
  ActivityItem
} from '../types';
import { useApi, useToast } from '../../../hooks';
import { dashboardService } from '../services/dashboardService';

// Query Keys pour React Query
export const DASHBOARD_QUERY_KEYS = {
  all: ['dashboard'] as const,
  data: (filters: DashboardFilters) => [...DASHBOARD_QUERY_KEYS.all, 'data', filters] as const,
  kpis: (filters: DashboardFilters) => [...DASHBOARD_QUERY_KEYS.all, 'kpis', filters] as const,
  charts: (filters: DashboardFilters) => [...DASHBOARD_QUERY_KEYS.all, 'charts', filters] as const,
  alerts: () => [...DASHBOARD_QUERY_KEYS.all, 'alerts'] as const,
  sla: () => [...DASHBOARD_QUERY_KEYS.all, 'sla'] as const,
  quality: () => [...DASHBOARD_QUERY_KEYS.all, 'quality'] as const,
  activity: (page: number) => [...DASHBOARD_QUERY_KEYS.all, 'activity', page] as const,
};

export const useDashboardData = (filters: DashboardFilters) => {
  const queryClient = useQueryClient();
  const { toast } = useToast();

  // Main dashboard data query
  const {
    data: dashboardData,
    isLoading,
    error,
    refetch,
  } = useQuery({
    queryKey: DASHBOARD_QUERY_KEYS.data(filters),
    queryFn: () => dashboardService.getDashboardData(filters),
    staleTime: 2 * 60 * 1000, // 2 minutes
    gcTime: 5 * 60 * 1000, // 5 minutes
    refetchInterval: 5 * 60 * 1000, // Auto-refresh every 5 minutes
  });

  // Individual data extractors with fallbacks
  const kpis = useMemo<DashboardKPI[]>(() => 
    dashboardData?.kpis || [], [dashboardData]
  );

  const charts = useMemo<DashboardChart[]>(() => 
    dashboardData?.charts || [], [dashboardData]
  );

  const alerts = useMemo<AlertLevel[]>(() => 
    dashboardData?.alerts || [], [dashboardData]
  );

  const slaIndicators = useMemo<SLAIndicator[]>(() => 
    dashboardData?.slaIndicators || [], [dashboardData]
  );

  const dataQuality = useMemo<DataQualityMetric[]>(() => 
    dashboardData?.dataQuality || [], [dashboardData]
  );

  const activity = useMemo<ActivityItem[]>(() => 
    dashboardData?.activity || [], [dashboardData]
  );

  const summary = useMemo(() => 
    dashboardData?.summary || {
      totalCompanies: 0,
      activeScrapings: 0,
      completedToday: 0,
      pendingActions: 0,
    }, [dashboardData]
  );

  // Manual refresh function
  const refreshDashboard = useCallback(async () => {
    try {
      await refetch();
      toast('Dashboard actualisé avec succès', 'success');
    } catch (error) {
      toast('Erreur lors de l\'actualisation', 'error');
    }
  }, [refetch, toast]);

  // Invalidate specific data sections
  const invalidateSection = useCallback((section: keyof DashboardDataResponse) => {
    switch (section) {
      case 'kpis':
        queryClient.invalidateQueries({ queryKey: DASHBOARD_QUERY_KEYS.kpis(filters) });
        break;
      case 'charts':
        queryClient.invalidateQueries({ queryKey: DASHBOARD_QUERY_KEYS.charts(filters) });
        break;
      case 'alerts':
        queryClient.invalidateQueries({ queryKey: DASHBOARD_QUERY_KEYS.alerts() });
        break;
      case 'slaIndicators':
        queryClient.invalidateQueries({ queryKey: DASHBOARD_QUERY_KEYS.sla() });
        break;
      case 'dataQuality':
        queryClient.invalidateQueries({ queryKey: DASHBOARD_QUERY_KEYS.quality() });
        break;
      case 'activity':
        queryClient.invalidateQueries({ queryKey: DASHBOARD_QUERY_KEYS.activity(1) });
        break;
      default:
        queryClient.invalidateQueries({ queryKey: DASHBOARD_QUERY_KEYS.all });
    }
  }, [queryClient, filters]);

  // Computed metrics
  const metrics = useMemo(() => ({
    totalAlerts: alerts.length,
    criticalAlerts: alerts.filter(a => a.level === 'critical').length,
    averageDataQuality: dataQuality.length > 0 
      ? Math.round(dataQuality.reduce((sum, dq) => sum + dq.overall, 0) / dataQuality.length)
      : 0,
    slaCompliance: slaIndicators.length > 0
      ? Math.round(slaIndicators.filter(sla => sla.status === 'excellent' || sla.status === 'good').length / slaIndicators.length * 100)
      : 0,
  }), [alerts, dataQuality, slaIndicators]);

  return {
    // Data
    dashboardData,
    kpis,
    charts,
    alerts,
    slaIndicators,
    dataQuality,
    activity,
    summary,
    metrics,
    
    // States
    isLoading,
    error,
    
    // Actions
    refreshDashboard,
    invalidateSection,
  };
};

// Hook spécialisé pour les KPIs seulement
export const useDashboardKPIs = (filters: DashboardFilters) => {
  return useQuery({
    queryKey: DASHBOARD_QUERY_KEYS.kpis(filters),
    queryFn: () => dashboardService.getKPIs(filters),
    staleTime: 1 * 60 * 1000, // 1 minute - plus fréquent pour les KPIs
    refetchInterval: 2 * 60 * 1000, // Auto-refresh every 2 minutes
  });
};

// Hook pour les alertes en temps réel
export const useDashboardAlerts = () => {
  return useQuery({
    queryKey: DASHBOARD_QUERY_KEYS.alerts(),
    queryFn: () => dashboardService.getAlerts(),
    staleTime: 30 * 1000, // 30 seconds
    refetchInterval: 1 * 60 * 1000, // Check every minute
  });
};

// Hook pour les métriques SLA
export const useSLAIndicators = () => {
  return useQuery({
    queryKey: DASHBOARD_QUERY_KEYS.sla(),
    queryFn: () => dashboardService.getSLAIndicators(),
    staleTime: 5 * 60 * 1000, // 5 minutes
    refetchInterval: 10 * 60 * 1000, // Check every 10 minutes
  });
};

// Hook pour la qualité des données
export const useDataQuality = () => {
  return useQuery({
    queryKey: DASHBOARD_QUERY_KEYS.quality(),
    queryFn: () => dashboardService.getDataQuality(),
    staleTime: 15 * 60 * 1000, // 15 minutes
    refetchInterval: 30 * 60 * 1000, // Check every 30 minutes
  });
};