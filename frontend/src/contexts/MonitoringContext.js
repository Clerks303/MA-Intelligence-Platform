/**
 * MonitoringContext - Gestion état global dashboard monitoring
 * US-004: Context provider pour données monitoring temps réel
 * 
 * Features:
 * - État centralisé monitoring
 * - Polling automatique données temps réel
 * - Cache intelligent avec React Query
 * - Gestion alertes et notifications
 * - WebSocket support (future)
 */

import React, { createContext, useContext, useReducer, useEffect, useCallback } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import api from '../services/api';

// Types d'actions pour le reducer
const MONITORING_ACTIONS = {
  SET_LOADING: 'SET_LOADING',
  SET_ERROR: 'SET_ERROR',
  SET_OVERVIEW: 'SET_OVERVIEW',
  SET_ALERTS: 'SET_ALERTS',
  SET_METRICS: 'SET_METRICS',
  SET_HEALTH: 'SET_HEALTH',
  UPDATE_ALERT: 'UPDATE_ALERT',
  SET_REAL_TIME_ENABLED: 'SET_REAL_TIME_ENABLED',
  SET_REFRESH_INTERVAL: 'SET_REFRESH_INTERVAL',
  ADD_NOTIFICATION: 'ADD_NOTIFICATION',
  REMOVE_NOTIFICATION: 'REMOVE_NOTIFICATION'
};

// État initial du monitoring
const initialState = {
  // Données monitoring
  overview: null,
  alerts: null,
  metrics: null,
  health: null,
  
  // État UI
  loading: false,
  error: null,
  lastUpdated: null,
  
  // Configuration temps réel
  realTimeEnabled: true,
  refreshInterval: 30000, // 30 secondes
  
  // Notifications
  notifications: [],
  
  // Filtres et préférences
  alertFilters: {
    severity: 'all',
    status: 'active',
    category: 'all'
  },
  dashboardLayout: 'grid', // grid, list
  autoRefresh: true
};

// Reducer pour gérer les actions
function monitoringReducer(state, action) {
  switch (action.type) {
    case MONITORING_ACTIONS.SET_LOADING:
      return {
        ...state,
        loading: action.payload,
        error: action.payload ? null : state.error
      };
      
    case MONITORING_ACTIONS.SET_ERROR:
      return {
        ...state,
        error: action.payload,
        loading: false
      };
      
    case MONITORING_ACTIONS.SET_OVERVIEW:
      return {
        ...state,
        overview: action.payload,
        lastUpdated: new Date().toISOString(),
        loading: false,
        error: null
      };
      
    case MONITORING_ACTIONS.SET_ALERTS:
      return {
        ...state,
        alerts: action.payload,
        loading: false
      };
      
    case MONITORING_ACTIONS.SET_METRICS:
      return {
        ...state,
        metrics: action.payload,
        loading: false
      };
      
    case MONITORING_ACTIONS.SET_HEALTH:
      return {
        ...state,
        health: action.payload,
        loading: false
      };
      
    case MONITORING_ACTIONS.UPDATE_ALERT:
      if (!state.alerts) return state;
      
      return {
        ...state,
        alerts: {
          ...state.alerts,
          filtered_active_alerts: state.alerts.filtered_active_alerts.map(alert =>
            alert.id === action.payload.id 
              ? { ...alert, ...action.payload.updates }
              : alert
          )
        }
      };
      
    case MONITORING_ACTIONS.SET_REAL_TIME_ENABLED:
      return {
        ...state,
        realTimeEnabled: action.payload
      };
      
    case MONITORING_ACTIONS.SET_REFRESH_INTERVAL:
      return {
        ...state,
        refreshInterval: action.payload
      };
      
    case MONITORING_ACTIONS.ADD_NOTIFICATION:
      return {
        ...state,
        notifications: [
          ...state.notifications,
          {
            id: Date.now(),
            timestamp: new Date().toISOString(),
            ...action.payload
          }
        ]
      };
      
    case MONITORING_ACTIONS.REMOVE_NOTIFICATION:
      return {
        ...state,
        notifications: state.notifications.filter(n => n.id !== action.payload)
      };
      
    default:
      return state;
  }
}

// Context
const MonitoringContext = createContext();

// Hook pour utiliser le context
export const useMonitoring = () => {
  const context = useContext(MonitoringContext);
  if (!context) {
    throw new Error('useMonitoring must be used within a MonitoringProvider');
  }
  return context;
};

// API calls
const monitoringApi = {
  // Vue d'ensemble système
  getOverview: () => api.get('/monitoring/overview').then(res => res.data),
  
  // Alertes avec filtres
  getAlerts: (filters = {}) => {
    const params = new URLSearchParams();
    Object.entries(filters).forEach(([key, value]) => {
      if (value && value !== 'all') {
        params.append(key, value);
      }
    });
    return api.get(`/monitoring/alerts?${params}`).then(res => res.data);
  },
  
  // Métriques détaillées
  getMetrics: (window_minutes = 5, category = null) => {
    const params = new URLSearchParams({ window_minutes });
    if (category && category !== 'all') {
      params.append('category', category);
    }
    return api.get(`/monitoring/metrics?${params}`).then(res => res.data);
  },
  
  // Health status
  getHealth: (detailed = false) => {
    const params = detailed ? '?detailed=true' : '';
    return api.get(`/monitoring/health${params}`).then(res => res.data);
  },
  
  // Performance par composant
  getPerformance: (component = null, timerange = '15m') => {
    const params = new URLSearchParams({ timerange });
    if (component) {
      params.append('component', component);
    }
    return api.get(`/monitoring/performance?${params}`).then(res => res.data);
  },
  
  // Dashboard complet
  getDashboard: () => api.get('/monitoring/dashboard').then(res => res.data),
  
  // Actions alertes
  acknowledgeAlert: (alertId, comment = '') => 
    api.post(`/monitoring/alerts/${alertId}/acknowledge`, null, {
      params: { comment }
    }).then(res => res.data),
    
  resolveAlert: (alertId, comment = '') =>
    api.post(`/monitoring/alerts/${alertId}/resolve`, null, {
      params: { comment }
    }).then(res => res.data)
};

// Provider component
export const MonitoringProvider = ({ children }) => {
  const [state, dispatch] = useReducer(monitoringReducer, initialState);
  const queryClient = useQueryClient();

  // Query keys
  const QUERY_KEYS = {
    overview: 'monitoring-overview',
    alerts: 'monitoring-alerts',
    metrics: 'monitoring-metrics',
    health: 'monitoring-health',
    performance: 'monitoring-performance',
    dashboard: 'monitoring-dashboard'
  };

  // Queries avec React Query pour cache intelligent
  const overviewQuery = useQuery({
    queryKey: [QUERY_KEYS.overview],
    queryFn: monitoringApi.getOverview,
    enabled: state.realTimeEnabled,
    refetchInterval: state.realTimeEnabled ? state.refreshInterval : false,
    onSuccess: (data) => {
      dispatch({ type: MONITORING_ACTIONS.SET_OVERVIEW, payload: data });
      
      // Vérifier nouvelles alertes critiques
      if (data.critical_issues?.immediate_attention_required) {
        checkForNewCriticalAlerts(data);
      }
    },
    onError: (error) => {
      dispatch({ type: MONITORING_ACTIONS.SET_ERROR, payload: error.message });
    }
  });

  const alertsQuery = useQuery({
    queryKey: [QUERY_KEYS.alerts, state.alertFilters],
    queryFn: () => monitoringApi.getAlerts(state.alertFilters),
    enabled: state.realTimeEnabled,
    refetchInterval: state.realTimeEnabled ? state.refreshInterval : false,
    onSuccess: (data) => {
      dispatch({ type: MONITORING_ACTIONS.SET_ALERTS, payload: data });
    }
  });

  const metricsQuery = useQuery({
    queryKey: [QUERY_KEYS.metrics],
    queryFn: () => monitoringApi.getMetrics(5), // 5 minutes window
    enabled: state.realTimeEnabled,
    refetchInterval: state.realTimeEnabled ? state.refreshInterval * 2 : false, // Moins fréquent
    onSuccess: (data) => {
      dispatch({ type: MONITORING_ACTIONS.SET_METRICS, payload: data });
    }
  });

  const healthQuery = useQuery({
    queryKey: [QUERY_KEYS.health],
    queryFn: () => monitoringApi.getHealth(false),
    enabled: state.realTimeEnabled,
    refetchInterval: state.realTimeEnabled ? state.refreshInterval : false,
    onSuccess: (data) => {
      dispatch({ type: MONITORING_ACTIONS.SET_HEALTH, payload: data });
    }
  });

  // Fonctions d'action
  const actions = {
    // Toggle temps réel
    toggleRealTime: useCallback(() => {
      const newEnabled = !state.realTimeEnabled;
      dispatch({ type: MONITORING_ACTIONS.SET_REAL_TIME_ENABLED, payload: newEnabled });
      
      if (newEnabled) {
        // Relancer les queries
        queryClient.invalidateQueries();
      }
    }, [state.realTimeEnabled, queryClient]),

    // Changer intervalle refresh
    setRefreshInterval: useCallback((interval) => {
      dispatch({ type: MONITORING_ACTIONS.SET_REFRESH_INTERVAL, payload: interval });
    }, []),

    // Refresh manuel
    refreshData: useCallback(() => {
      dispatch({ type: MONITORING_ACTIONS.SET_LOADING, payload: true });
      return Promise.all([
        queryClient.invalidateQueries(QUERY_KEYS.overview),
        queryClient.invalidateQueries(QUERY_KEYS.alerts),
        queryClient.invalidateQueries(QUERY_KEYS.metrics),
        queryClient.invalidateQueries(QUERY_KEYS.health)
      ]);
    }, [queryClient]),

    // Gestion alertes
    acknowledgeAlert: useCallback(async (alertId, comment = '') => {
      try {
        const result = await monitoringApi.acknowledgeAlert(alertId, comment);
        
        dispatch({
          type: MONITORING_ACTIONS.UPDATE_ALERT,
          payload: {
            id: alertId,
            updates: { status: 'acknowledged', acknowledged_by: result.acknowledged_by }
          }
        });

        dispatch({
          type: MONITORING_ACTIONS.ADD_NOTIFICATION,
          payload: {
            type: 'success',
            title: 'Alerte acquittée',
            message: `Alerte ${alertId} acquittée avec succès`
          }
        });

        // Refresh alertes
        queryClient.invalidateQueries(QUERY_KEYS.alerts);
        
        return result;
      } catch (error) {
        dispatch({
          type: MONITORING_ACTIONS.ADD_NOTIFICATION,
          payload: {
            type: 'error',
            title: 'Erreur acquittement',
            message: error.response?.data?.detail || error.message
          }
        });
        throw error;
      }
    }, [queryClient]),

    resolveAlert: useCallback(async (alertId, comment = '') => {
      try {
        const result = await monitoringApi.resolveAlert(alertId, comment);
        
        dispatch({
          type: MONITORING_ACTIONS.UPDATE_ALERT,
          payload: {
            id: alertId,
            updates: { status: 'resolved', resolved_by: result.resolved_by }
          }
        });

        dispatch({
          type: MONITORING_ACTIONS.ADD_NOTIFICATION,
          payload: {
            type: 'success',
            title: 'Alerte résolue',
            message: `Alerte ${alertId} résolue avec succès`
          }
        });

        queryClient.invalidateQueries(QUERY_KEYS.alerts);
        
        return result;
      } catch (error) {
        dispatch({
          type: MONITORING_ACTIONS.ADD_NOTIFICATION,
          payload: {
            type: 'error',
            title: 'Erreur résolution',
            message: error.response?.data?.detail || error.message
          }
        });
        throw error;
      }
    }, [queryClient]),

    // Filtres alertes
    setAlertFilters: useCallback((filters) => {
      dispatch({
        type: MONITORING_ACTIONS.SET_ALERTS,
        payload: { ...state.alertFilters, ...filters }
      });
    }, [state.alertFilters]),

    // Notifications
    addNotification: useCallback((notification) => {
      dispatch({ type: MONITORING_ACTIONS.ADD_NOTIFICATION, payload: notification });
      
      // Auto-remove après 5 secondes pour les success
      if (notification.type === 'success') {
        setTimeout(() => {
          dispatch({ type: MONITORING_ACTIONS.REMOVE_NOTIFICATION, payload: notification.id });
        }, 5000);
      }
    }, []),

    removeNotification: useCallback((id) => {
      dispatch({ type: MONITORING_ACTIONS.REMOVE_NOTIFICATION, payload: id });
    }, [])
  };

  // Fonction pour détecter nouvelles alertes critiques
  const checkForNewCriticalAlerts = useCallback((data) => {
    const criticalCount = data.critical_issues?.critical_alerts_count || 0;
    const emergencyCount = data.critical_issues?.emergency_alerts_count || 0;
    
    if (criticalCount > 0 || emergencyCount > 0) {
      // Notification browser si supporté
      if ('Notification' in window && Notification.permission === 'granted') {
        new Notification('M&A Intelligence - Alerte Critique', {
          body: `${criticalCount + emergencyCount} alerte(s) critique(s) détectée(s)`,
          icon: '/favicon.ico',
          tag: 'critical-alert'
        });
      }
      
      actions.addNotification({
        type: 'error',
        title: 'Alertes Critiques',
        message: `${criticalCount + emergencyCount} alerte(s) critique(s) nécessitent votre attention`,
        persistent: true
      });
    }
  }, [actions]);

  // Demander permission notifications au démarrage
  useEffect(() => {
    if ('Notification' in window && Notification.permission === 'default') {
      Notification.requestPermission();
    }
  }, []);

  // Nettoyage notifications anciennes
  useEffect(() => {
    const interval = setInterval(() => {
      const oneHourAgo = new Date(Date.now() - 60 * 60 * 1000);
      
      state.notifications.forEach(notification => {
        if (new Date(notification.timestamp) < oneHourAgo && !notification.persistent) {
          actions.removeNotification(notification.id);
        }
      });
    }, 5 * 60 * 1000); // Toutes les 5 minutes

    return () => clearInterval(interval);
  }, [state.notifications, actions]);

  // État loading global
  const loading = overviewQuery.isLoading || alertsQuery.isLoading || 
                 metricsQuery.isLoading || healthQuery.isLoading;

  const value = {
    // État
    ...state,
    loading,
    
    // Données queries
    queries: {
      overview: overviewQuery,
      alerts: alertsQuery,
      metrics: metricsQuery,
      health: healthQuery
    },
    
    // Actions
    ...actions,
    
    // Statuts
    isConnected: !state.error && (state.overview !== null),
    hasNewAlerts: state.alerts?.summary?.unacknowledged_critical > 0,
    systemHealthy: state.overview?.system_status?.overall_health === 'healthy'
  };

  return (
    <MonitoringContext.Provider value={value}>
      {children}
    </MonitoringContext.Provider>
  );
};

export default MonitoringContext;