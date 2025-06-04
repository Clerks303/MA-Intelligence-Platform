/**
 * Hooks Monitoring et Dashboard - M&A Intelligence Platform
 * Sprint 4 - Hooks pour dashboard sécurité et monitoring système
 */

import { useState, useCallback, useMemo } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { 
  SecurityDashboard, 
  SecurityMetrics,
  SecurityAlert,
  SecurityFilters,
  SecurityAlertType,
  SecurityAlertStatus,
  SecurityConfig,
  AuditSeverity,
  SystemMetrics,
  HealthStatus
} from '../types/api';
import { securityService } from '../services/securityService';

// === DASHBOARD SÉCURITÉ ===

export const useSecurityDashboard = () => {
  const queryClient = useQueryClient();

  // Dashboard principal
  const { 
    data: dashboard, 
    isLoading: isLoadingDashboard,
    error: dashboardError,
    refetch: refetchDashboard 
  } = useQuery({
    queryKey: ['security', 'dashboard'],
    queryFn: securityService.dashboard.getDashboard,
    refetchInterval: 30000, // Mise à jour toutes les 30 secondes
  });

  // Métriques détaillées
  const { 
    data: metrics, 
    isLoading: isLoadingMetrics,
    error: metricsError,
    refetch: refetchMetrics 
  } = useQuery({
    queryKey: ['security', 'metrics'],
    queryFn: () => securityService.dashboard.getMetrics(7), // 7 derniers jours
    refetchInterval: 60000, // Mise à jour toutes les minutes
  });

  // Métriques temps réel
  const { 
    data: realTimeMetrics, 
    isLoading: isLoadingRealTime 
  } = useQuery({
    queryKey: ['security', 'realtime-metrics'],
    queryFn: securityService.dashboard.getRealTimeMetrics,
    refetchInterval: 10000, // Mise à jour toutes les 10 secondes
  });

  // Calculer les tendances
  const trends = useMemo(() => {
    if (!metrics) return null;
    
    const calculateTrend = (data: Array<{ value: number }>) => {
      if (data.length < 2) return 0;
      const recent = data.slice(-3).reduce((sum: number, item: { value: number }) => sum + item.value, 0);
      const previous = data.slice(-6, -3).reduce((sum: number, item: { value: number }) => sum + item.value, 0);
      return previous > 0 ? ((recent - previous) / previous * 100) : 0;
    };

    return {
      loginAttempts: calculateTrend(metrics.login_attempts),
      failedLogins: calculateTrend(metrics.failed_logins),
      threatsDetected: calculateTrend(metrics.threats_detected),
      securityAlerts: calculateTrend(metrics.security_alerts),
    };
  }, [metrics]);

  return {
    // Données
    dashboard,
    metrics,
    realTimeMetrics,
    trends,
    
    // Chargement
    isLoading: isLoadingDashboard || isLoadingMetrics,
    isLoadingRealTime,
    
    // Erreurs
    error: dashboardError || metricsError,
    
    // Actions
    refetch: () => {
      refetchDashboard();
      refetchMetrics();
    },
    refetchDashboard,
    refetchMetrics,
  };
};

// === ALERTES DE SÉCURITÉ ===

export const useSecurityAlerts = (initialFilters?: SecurityFilters['security_alerts']) => {
  const [filters, setFilters] = useState(initialFilters || {});
  const queryClient = useQueryClient();

  // Liste des alertes
  const { 
    data: alerts = [], 
    isLoading,
    error,
    refetch 
  } = useQuery({
    queryKey: ['security', 'alerts', filters],
    queryFn: () => securityService.alerts.getAlerts(filters),
    refetchInterval: 30000, // Mise à jour toutes les 30 secondes
  });

  // Assigner une alerte
  const assignAlertMutation = useMutation({
    mutationFn: ({ alertId, userId }: { alertId: string; userId: string }) =>
      securityService.alerts.assignAlert(alertId, userId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['security', 'alerts'] });
    },
  });

  // Résoudre une alerte
  const resolveAlertMutation = useMutation({
    mutationFn: ({ alertId, notes }: { alertId: string; notes: string }) =>
      securityService.alerts.resolveAlert(alertId, notes),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['security', 'alerts'] });
    },
  });

  // Marquer comme faux positif
  const markFalsePositiveMutation = useMutation({
    mutationFn: ({ alertId, reason }: { alertId: string; reason: string }) =>
      securityService.alerts.markFalsePositive(alertId, reason),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['security', 'alerts'] });
    },
  });

  // Actions
  const assignAlert = useCallback((alertId: string, userId: string) => {
    assignAlertMutation.mutate({ alertId, userId });
  }, [assignAlertMutation]);

  const resolveAlert = useCallback((alertId: string, notes: string) => {
    resolveAlertMutation.mutate({ alertId, notes });
  }, [resolveAlertMutation]);

  const markFalsePositive = useCallback((alertId: string, reason: string) => {
    markFalsePositiveMutation.mutate({ alertId, reason });
  }, [markFalsePositiveMutation]);

  const updateFilters = useCallback((newFilters: Partial<SecurityFilters['security_alerts']>) => {
    setFilters(prev => ({ ...prev, ...newFilters }));
  }, []);

  const clearFilters = useCallback(() => {
    setFilters({});
  }, []);

  // Métriques des alertes
  const alertMetrics = useMemo(() => ({
    total: alerts.length,
    open: alerts.filter((a: SecurityAlert) => a.status === 'open').length,
    investigating: alerts.filter((a: SecurityAlert) => a.status === 'investigating').length,
    resolved: alerts.filter((a: SecurityAlert) => a.status === 'resolved').length,
    falsePositives: alerts.filter((a: SecurityAlert) => a.status === 'false_positive').length,
    
    // Par sévérité
    bySeverity: alerts.reduce((acc: Record<AuditSeverity, number>, alert: SecurityAlert) => {
      acc[alert.severity] = (acc[alert.severity] || 0) + 1;
      return acc;
    }, {} as Record<AuditSeverity, number>),
    
    // Par type
    byType: alerts.reduce((acc: Record<SecurityAlertType, number>, alert: SecurityAlert) => {
      acc[alert.type] = (acc[alert.type] || 0) + 1;
      return acc;
    }, {} as Record<SecurityAlertType, number>),
    
    // Critiques non résolues
    criticalUnresolved: alerts.filter((a: SecurityAlert) => 
      a.severity === 'CRITICAL' && a.status !== 'resolved'
    ).length,
  }), [alerts]);

  // Alertes prioritaires
  const priorityAlerts = useMemo(() => {
    return alerts
      .filter((alert: SecurityAlert) => alert.severity === 'CRITICAL' || alert.severity === 'HIGH')
      .filter((alert: SecurityAlert) => alert.status === 'open' || alert.status === 'investigating')
      .sort((a: SecurityAlert, b: SecurityAlert) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime())
      .slice(0, 10);
  }, [alerts]);

  return {
    // Données
    alerts,
    alertMetrics,
    priorityAlerts,
    
    // Filtres
    filters,
    updateFilters,
    clearFilters,
    
    // Chargement
    isLoading,
    isAssigning: assignAlertMutation.isPending,
    isResolving: resolveAlertMutation.isPending,
    isMarkingFalsePositive: markFalsePositiveMutation.isPending,
    
    // Erreurs
    error,
    assignError: assignAlertMutation.error,
    resolveError: resolveAlertMutation.error,
    falsePositiveError: markFalsePositiveMutation.error,
    
    // Actions
    assignAlert,
    resolveAlert,
    markFalsePositive,
    refetch,
    
    // Données mutations
    assignData: assignAlertMutation.data,
    resolveData: resolveAlertMutation.data,
    falsePositiveData: markFalsePositiveMutation.data,
  };
};

// === MONITORING SYSTÈME ===

export const useSystemMonitoring = () => {
  // Dashboard monitoring
  const { 
    data: monitoringDashboard, 
    isLoading: isLoadingDashboard 
  } = useQuery({
    queryKey: ['monitoring', 'dashboard'],
    queryFn: securityService.monitoring.getDashboard,
    refetchInterval: 15000, // 15 secondes
  });

  // Métriques système
  const { 
    data: systemMetrics, 
    isLoading: isLoadingMetrics 
  } = useQuery({
    queryKey: ['monitoring', 'system-metrics'],
    queryFn: securityService.monitoring.getSystemMetrics,
    refetchInterval: 10000, // 10 secondes
  });

  // Health status
  const { 
    data: healthStatus, 
    isLoading: isLoadingHealth 
  } = useQuery({
    queryKey: ['monitoring', 'health'],
    queryFn: securityService.monitoring.getHealthStatus,
    refetchInterval: 30000, // 30 secondes
  });

  // Calculer l'état global du système
  const systemStatus = useMemo(() => {
    if (!healthStatus || !systemMetrics) return 'unknown';
    
    const health = healthStatus as HealthStatus;
    const metrics = systemMetrics as SystemMetrics;
    
    if (health.status === 'unhealthy') return 'critical';
    if (health.status === 'degraded') return 'warning';
    
    // Vérifier les métriques critiques
    const cpuHigh = metrics.cpu_usage > 80;
    const memoryHigh = metrics.memory_usage > 85;
    const diskHigh = metrics.disk_usage > 90;
    
    if (diskHigh) return 'critical';
    if (cpuHigh || memoryHigh) return 'warning';
    
    return 'healthy';
  }, [healthStatus, systemMetrics]);

  return {
    // Données
    monitoringDashboard,
    systemMetrics,
    healthStatus,
    systemStatus,
    
    // Chargement
    isLoading: isLoadingDashboard || isLoadingMetrics || isLoadingHealth,
    
    // Métriques calculées
    metrics: {
      uptime: (monitoringDashboard as any)?.uptime_percentage || 0,
      performance: (monitoringDashboard as any)?.performance_score || 0,
      health: (monitoringDashboard as any)?.system_health || 0,
      alerts: (monitoringDashboard as any)?.active_alerts || 0,
    },
  };
};

// === CONFIGURATION SÉCURITÉ ===

export const useSecurityConfig = () => {
  const queryClient = useQueryClient();

  // Configuration actuelle
  const { 
    data: config, 
    isLoading,
    error,
    refetch 
  } = useQuery({
    queryKey: ['security', 'config'],
    queryFn: securityService.config.getSecurityConfig,
  });

  // Modifier la configuration
  const updateConfigMutation = useMutation({
    mutationFn: (newConfig: Partial<SecurityConfig>) =>
      securityService.config.updateSecurityConfig(newConfig),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['security', 'config'] });
    },
  });

  // Réinitialiser la configuration
  const resetConfigMutation = useMutation({
    mutationFn: securityService.config.resetSecurityConfig,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['security', 'config'] });
    },
  });

  // Actions
  const updateConfig = useCallback((newConfig: Partial<SecurityConfig>) => {
    updateConfigMutation.mutate(newConfig);
  }, [updateConfigMutation]);

  const resetConfig = useCallback(() => {
    resetConfigMutation.mutate();
  }, [resetConfigMutation]);

  return {
    // Données
    config,
    
    // Chargement
    isLoading,
    isUpdating: updateConfigMutation.isPending,
    isResetting: resetConfigMutation.isPending,
    
    // Erreurs
    error,
    updateError: updateConfigMutation.error,
    resetError: resetConfigMutation.error,
    
    // Actions
    updateConfig,
    resetConfig,
    refetch,
    
    // Données mutations
    updateData: updateConfigMutation.data,
    resetData: resetConfigMutation.data,
  };
};

// === HOOK MONITORING COMBINÉ ===

export const useMonitoring = () => {
  const dashboard = useSecurityDashboard();
  const alerts = useSecurityAlerts();
  const system = useSystemMonitoring();
  const config = useSecurityConfig();

  // Statut global
  const globalStatus = useMemo(() => {
    const hasHighPriorityAlerts = alerts.priorityAlerts.length > 0;
    const systemIssues = system.systemStatus === 'critical' || system.systemStatus === 'warning';
    
    if (system.systemStatus === 'critical' || alerts.alertMetrics.criticalUnresolved > 0) {
      return 'critical';
    }
    
    if (systemIssues || hasHighPriorityAlerts) {
      return 'warning';
    }
    
    return 'healthy';
  }, [alerts.priorityAlerts, alerts.alertMetrics, system.systemStatus]);

  // Actions rapides
  const quickActions = {
    showCriticalAlerts: () => alerts.updateFilters({ severities: ['CRITICAL'] }),
    showOpenAlerts: () => alerts.updateFilters({ statuses: ['open'] }),
    showMyAlerts: (userId: string) => alerts.updateFilters({ assigned_to: userId }),
    clearAlertFilters: () => alerts.clearFilters(),
    refreshAll: () => {
      dashboard.refetch();
      alerts.refetch();
    },
  };

  return {
    // Modules
    dashboard,
    alerts,
    system,
    config,
    
    // État global
    globalStatus,
    isLoading: dashboard.isLoading || alerts.isLoading || system.isLoading,
    hasErrors: !!(dashboard.error || alerts.error),
    
    // Actions
    quickActions,
    
    // Métriques consolidées
    overview: {
      totalAlerts: alerts.alertMetrics.total,
      criticalAlerts: alerts.alertMetrics.criticalUnresolved,
      systemHealth: system.metrics.health,
      uptime: system.metrics.uptime,
      activeSessions: (dashboard.dashboard as SecurityDashboard)?.active_sessions || 0,
      threatsBlocked: (dashboard.dashboard as SecurityDashboard)?.threats_blocked_today || 0,
    },
  };
};

export default {
  useSecurityDashboard,
  useSecurityAlerts,
  useSystemMonitoring,
  useSecurityConfig,
  useMonitoring,
};