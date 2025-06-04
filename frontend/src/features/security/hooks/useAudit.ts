/**
 * Hooks Audit et Logs - M&A Intelligence Platform
 * Sprint 4 - Hooks pour visualisation et analyse des logs d'audit
 */

import { useState, useCallback, useMemo } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { 
  AuditEvent, 
  AuditFilters, 
  AuditReport, 
  PaginatedResponse,
  AuditEventType,
  AuditSeverity,
  ResourceType,
  ActionType
} from '../types/api';
import { securityService } from '../services/securityService';

// === HOOKS AUDIT EVENTS ===

export const useAuditEvents = (
  initialFilters?: AuditFilters,
  initialPage = 1,
  initialPerPage = 50
) => {
  const [filters, setFilters] = useState<AuditFilters>(initialFilters || {});
  const [page, setPage] = useState(initialPage);
  const [perPage, setPerPage] = useState(initialPerPage);

  const queryClient = useQueryClient();

  // Obtenir les événements d'audit
  const { 
    data: auditResponse, 
    isLoading,
    error,
    refetch 
  } = useQuery({
    queryKey: ['security', 'audit', 'events', filters, page, perPage],
    queryFn: () => securityService.audit.getAuditEvents({ ...filters, page, per_page: perPage }),
    placeholderData: (previousData) => previousData,
  });

  // Obtenir un événement spécifique
  const getAuditEvent = useCallback((eventId: string) => {
    return useQuery({
      queryKey: ['security', 'audit', 'events', eventId],
      queryFn: () => securityService.audit.getAuditEvent(eventId),
    });
  }, []);

  // Export des logs
  const exportLogsMutation = useMutation({
    mutationFn: ({ filters: exportFilters, format }: { 
      filters?: AuditFilters; 
      format: 'json' | 'csv' 
    }) => securityService.audit.exportAuditLogs(),
  });

  // Générer un rapport
  const generateReportMutation = useMutation({
    mutationFn: (reportFilters?: AuditFilters) => 
      securityService.audit.generateReport(),
  });

  // Actions
  const updateFilters = useCallback((newFilters: Partial<AuditFilters>) => {
    setFilters(prev => ({ ...prev, ...newFilters }));
    setPage(1); // Reset à la première page
  }, []);

  const clearFilters = useCallback(() => {
    setFilters({});
    setPage(1);
  }, []);

  const goToPage = useCallback((newPage: number) => {
    setPage(newPage);
  }, []);

  const exportLogs = useCallback((format: 'json' | 'csv' = 'json') => {
    exportLogsMutation.mutate({ filters, format });
  }, [exportLogsMutation, filters]);

  const generateReport = useCallback(() => {
    generateReportMutation.mutate(filters);
  }, [generateReportMutation, filters]);

  // Métadonnées des événements
  const auditMetrics = useMemo(() => {
    const events = (auditResponse as PaginatedResponse<AuditEvent>)?.items || [];
    return {
      total: (auditResponse as PaginatedResponse<AuditEvent>)?.total || 0,
      success: events.filter((e: AuditEvent) => e.success).length,
      failures: events.filter((e: AuditEvent) => !e.success).length,
      
      // Par type
      byType: events.reduce((acc: Record<AuditEventType, number>, event: AuditEvent) => {
        acc[event.event_type] = (acc[event.event_type] || 0) + 1;
        return acc;
      }, {} as Record<AuditEventType, number>),
      
      // Par sévérité
      bySeverity: events.reduce((acc: Record<AuditSeverity, number>, event: AuditEvent) => {
        acc[event.severity] = (acc[event.severity] || 0) + 1;
        return acc;
      }, {} as Record<AuditSeverity, number>),
      
      // Par utilisateur
      byUser: events.reduce((acc: Record<string, number>, event: AuditEvent) => {
        const user = event.username || event.user_id || 'System';
        acc[user] = (acc[user] || 0) + 1;
        return acc;
      }, {} as Record<string, number>),
      
      // Par ressource
      byResource: events.reduce((acc: Record<ResourceType, number>, event: AuditEvent) => {
        acc[event.resource_type] = (acc[event.resource_type] || 0) + 1;
        return acc;
      }, {} as Record<ResourceType, number>),
      
      // Par action
      byAction: events.reduce((acc: Record<ActionType, number>, event: AuditEvent) => {
        acc[event.action] = (acc[event.action] || 0) + 1;
        return acc;
      }, {} as Record<ActionType, number>),
    };
  }, [auditResponse]);

  // Analyse temporelle
  const timelineData = useMemo(() => {
    const events = (auditResponse as PaginatedResponse<AuditEvent>)?.items || [];
    const grouped = events.reduce((acc: Record<string, { total: number; success: number; failures: number }>, event: AuditEvent) => {
      const date = new Date(event.timestamp).toISOString().split('T')[0];
      if (!acc[date]) {
        acc[date] = { total: 0, success: 0, failures: 0 };
      }
      acc[date].total++;
      if (event.success) acc[date].success++;
      else acc[date].failures++;
      return acc;
    }, {} as Record<string, { total: number; success: number; failures: number }>);

    return Object.entries(grouped)
      .map(([date, counts]) => ({ date, ...counts }))
      .sort((a, b) => a.date.localeCompare(b.date));
  }, [auditResponse]);

  return {
    // Données
    events: (auditResponse as PaginatedResponse<AuditEvent>)?.items || [],
    pagination: auditResponse ? {
      page: (auditResponse as PaginatedResponse<AuditEvent>).page,
      pages: (auditResponse as PaginatedResponse<AuditEvent>).pages,
      perPage: (auditResponse as PaginatedResponse<AuditEvent>).per_page,
      total: (auditResponse as PaginatedResponse<AuditEvent>).total,
      hasNext: (auditResponse as PaginatedResponse<AuditEvent>).has_next,
      hasPrev: (auditResponse as PaginatedResponse<AuditEvent>).has_prev,
    } : null,
    auditMetrics,
    timelineData,
    
    // Filtres et pagination
    filters,
    updateFilters,
    clearFilters,
    page,
    perPage,
    goToPage,
    setPerPage,
    
    // Chargement
    isLoading,
    isExporting: exportLogsMutation.isPending,
    isGeneratingReport: generateReportMutation.isPending,
    
    // Erreurs
    error,
    exportError: exportLogsMutation.error,
    reportError: generateReportMutation.error,
    
    // Actions
    exportLogs,
    generateReport,
    getAuditEvent,
    refetch,
    
    // Données mutations
    exportData: exportLogsMutation.data,
    reportData: generateReportMutation.data,
  };
};

// === HOOK STATISTIQUES AUDIT ===

export const useAuditStats = (days = 30) => {
  const { 
    data: stats, 
    isLoading,
    error,
    refetch 
  } = useQuery({
    queryKey: ['security', 'audit', 'stats', days],
    queryFn: () => securityService.audit.getAuditStats(),
    staleTime: 5 * 60 * 1000, // 5 minutes
  });

  // Métriques calculées
  const metrics = useMemo(() => {
    if (!stats) return null;
    
    return {
      ...stats,
      
      // Taux de succès
      successRate: stats.total_events > 0 ? 
        ((stats.total_events - (stats.events_by_severity?.ERROR || 0) - (stats.events_by_severity?.CRITICAL || 0)) / stats.total_events * 100).toFixed(1)
        : '100',
      
      // Événements par jour moyen
      avgEventsPerDay: stats.total_events / days,
      
      // Tendance (croissance/décroissance)
      trend: stats.trend.length > 1 ? {
        direction: stats.trend[stats.trend.length - 1].count > stats.trend[0].count ? 'up' : 'down',
        percentage: stats.trend[0].count > 0 ? 
          ((stats.trend[stats.trend.length - 1].count - stats.trend[0].count) / stats.trend[0].count * 100).toFixed(1)
          : '0',
      } : null,
    };
  }, [stats, days]);

  return {
    // Données
    stats: metrics,
    rawStats: stats,
    
    // État
    isLoading,
    error,
    
    // Actions
    refetch,
  };
};

// === HOOK RECHERCHE AVANCÉE ===

export const useAuditSearch = () => {
  const [searchQuery, setSearchQuery] = useState('');
  const [searchFilters, setSearchFilters] = useState<AuditFilters>({});
  const [isSearching, setIsSearching] = useState(false);

  const queryClient = useQueryClient();

  // Recherche avec debounce
  const searchMutation = useMutation({
    mutationFn: async ({ query, filters }: { query: string; filters: AuditFilters }) => {
      setIsSearching(true);
      try {
        const searchFiltersWithQuery = {
          ...filters,
          search_query: query || undefined,
        };
        return await securityService.audit.getAuditEvents({ ...searchFiltersWithQuery, page: 1, per_page: 50 });
      } finally {
        setIsSearching(false);
      }
    },
  });

  // Actions
  const search = useCallback((query: string, filters: AuditFilters = {}) => {
    setSearchQuery(query);
    setSearchFilters(filters);
    searchMutation.mutate({ query, filters });
  }, [searchMutation]);

  const clearSearch = useCallback(() => {
    setSearchQuery('');
    setSearchFilters({});
    searchMutation.reset();
  }, [searchMutation]);

  // Suggestions de recherche
  const searchSuggestions = useMemo(() => {
    const suggestions = [
      'failed login',
      'successful login',
      'password change',
      'role assignment',
      'permission denied',
      'data export',
      'user creation',
      'suspicious activity',
    ];
    
    if (searchQuery.length > 2) {
      return suggestions.filter(s => 
        s.toLowerCase().includes(searchQuery.toLowerCase())
      );
    }
    
    return suggestions;
  }, [searchQuery]);

  return {
    // État de recherche
    searchQuery,
    searchFilters,
    isSearching: isSearching || searchMutation.isPending,
    
    // Résultats
    searchResults: searchMutation.data,
    searchError: searchMutation.error,
    
    // Suggestions
    searchSuggestions,
    
    // Actions
    search,
    clearSearch,
    setSearchQuery,
    setSearchFilters,
  };
};

// === HOOK SURVEILLANCE TEMPS RÉEL ===

export const useAuditRealTime = (enabled = true, pollingInterval = 30000) => {
  // Événements récents (dernières 24h)
  const { 
    data: recentEvents, 
    isLoading: isLoadingRecent 
  } = useQuery({
    queryKey: ['security', 'audit', 'realtime', 'recent'],
    queryFn: () => {
      const yesterday = new Date();
      yesterday.setDate(yesterday.getDate() - 1);
      
      return securityService.audit.getAuditEvents({
        start_date: yesterday.toISOString(),
        end_date: new Date().toISOString(),
      });
    },
    enabled,
    refetchInterval: pollingInterval,
  });

  // Statistiques temps réel
  const { 
    data: realtimeStats, 
    isLoading: isLoadingStats 
  } = useQuery({
    queryKey: ['security', 'audit', 'realtime', 'stats'],
    queryFn: () => securityService.audit.getAuditStats(), // Dernières 24h
    enabled,
    refetchInterval: pollingInterval,
  });

  // Alertes critiques récentes
  const criticalEvents = useMemo(() => {
    return recentEvents?.items?.filter(event => 
      event.severity === 'CRITICAL' || event.severity === 'HIGH'
    ) || [];
  }, [recentEvents]);

  // Activité par heure
  const hourlyActivity = useMemo(() => {
    if (!recentEvents?.items) return [];
    
    const hourly = recentEvents.items.reduce((acc, event) => {
      const hour = new Date(event.timestamp).getHours();
      acc[hour] = (acc[hour] || 0) + 1;
      return acc;
    }, {} as Record<number, number>);
    
    return Array.from({ length: 24 }, (_, hour) => ({
      hour,
      count: hourly[hour] || 0,
    }));
  }, [recentEvents]);

  return {
    // Données temps réel
    recentEvents: recentEvents?.items || [],
    criticalEvents,
    realtimeStats,
    hourlyActivity,
    
    // État
    isLoading: isLoadingRecent || isLoadingStats,
    
    // Métriques temps réel
    metrics: {
      totalEventsToday: realtimeStats?.total_events || 0,
      criticalEventsCount: criticalEvents.length,
      successRate: realtimeStats?.total_events ? 
        ((realtimeStats.total_events - (realtimeStats.events_by_severity?.ERROR || 0) - (realtimeStats.events_by_severity?.CRITICAL || 0)) / realtimeStats.total_events * 100).toFixed(1)
        : '100',
      topUsers: realtimeStats?.top_users || [],
    },
  };
};

// === HOOK AUDIT COMBINÉ ===

export const useAudit = (initialFilters?: AuditFilters) => {
  const events = useAuditEvents(initialFilters);
  const stats = useAuditStats();
  const search = useAuditSearch();
  const realtime = useAuditRealTime();

  return {
    // Modules
    events,
    stats,
    search,
    realtime,
    
    // État global
    isLoading: events.isLoading || stats.isLoading,
    hasErrors: !!(events.error || stats.error),
    
    // Actions rapides
    quickFilters: {
      showCritical: () => events.updateFilters({ severity: ['CRITICAL'] }),
      showErrors: () => events.updateFilters({ severity: ['HIGH', 'CRITICAL'] }),
      showToday: () => {
        const today = new Date().toISOString().split('T')[0];
        events.updateFilters({ start_date: today });
      },
      showFailures: () => events.updateFilters({ success: false }),
      showLogins: () => events.updateFilters({ event_type: ['LOGIN'] }),
      clearAll: () => events.clearFilters(),
    },
  };
};

export default {
  useAuditEvents,
  useAuditStats,
  useAuditSearch,
  useAuditRealTime,
  useAudit,
};