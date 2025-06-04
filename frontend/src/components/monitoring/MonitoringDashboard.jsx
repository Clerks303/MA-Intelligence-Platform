/**
 * MonitoringDashboard - Dashboard principal monitoring temps réel
 * US-004: Interface complète monitoring avec widgets et charts
 */

import React, { useState } from 'react';
import {
  Box,
  Grid,
  Paper,
  Typography,
  Button,
  Switch,
  FormControlLabel,
  Alert,
  Chip,
  IconButton,
  Menu,
  MenuItem,
  Tooltip
} from '@mui/material';
import {
  Refresh,
  Settings,
  PlayArrow,
  Pause,
  Notifications,
  Dashboard,
  Timeline
} from '@mui/icons-material';
import { useMonitoring } from '../../contexts/MonitoringContext';
import SystemOverviewCard from './SystemOverviewCard';
import AlertsOverview from './AlertsOverview';
import MetricsChart from './MetricsChart';
import HealthStatus from './HealthStatus';
import RealTimeIndicator from './RealTimeIndicator';

const MonitoringDashboard = () => {
  const {
    overview,
    loading,
    error,
    realTimeEnabled,
    lastUpdated,
    notifications,
    toggleRealTime,
    refreshData,
    removeNotification,
    systemHealthy,
    hasNewAlerts
  } = useMonitoring();

  const [settingsAnchor, setSettingsAnchor] = useState(null);
  const [viewMode, setViewMode] = useState('grid'); // grid, compact

  const handleRefresh = async () => {
    try {
      await refreshData();
    } catch (error) {
      console.error('Erreur refresh dashboard:', error);
    }
  };

  const formatLastUpdated = (timestamp) => {
    if (!timestamp) return 'Jamais';
    const date = new Date(timestamp);
    return `${date.toLocaleTimeString()}`;
  };

  // Indicateur santé système
  const getSystemHealthColor = () => {
    if (!overview) return 'grey';
    const health = overview.system_status?.overall_health;
    switch (health) {
      case 'healthy': return 'success';
      case 'degraded': return 'warning';
      case 'unhealthy': return 'error';
      default: return 'grey';
    }
  };

  return (
    <Box sx={{ p: 3 }}>
      {/* Header avec contrôles */}
      <Box sx={{ mb: 3, display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
          <Dashboard sx={{ fontSize: 32, color: 'primary.main' }} />
          <Typography variant="h4" component="h1">
            Monitoring Dashboard
          </Typography>
          <Chip
            label={systemHealthy ? 'Système Sain' : 'Alertes Actives'}
            color={getSystemHealthColor()}
            variant={systemHealthy ? 'filled' : 'outlined'}
            icon={hasNewAlerts ? <Notifications /> : undefined}
          />
        </Box>

        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          {/* Indicateur temps réel */}
          <RealTimeIndicator 
            enabled={realTimeEnabled}
            lastUpdated={lastUpdated}
          />

          {/* Toggle temps réel */}
          <FormControlLabel
            control={
              <Switch
                checked={realTimeEnabled}
                onChange={toggleRealTime}
                color="primary"
              />
            }
            label="Temps réel"
          />

          {/* Bouton refresh */}
          <Tooltip title="Actualiser">
            <IconButton
              onClick={handleRefresh}
              disabled={loading}
              color="primary"
            >
              <Refresh />
            </IconButton>
          </Tooltip>

          {/* Menu paramètres */}
          <Tooltip title="Paramètres">
            <IconButton
              onClick={(e) => setSettingsAnchor(e.currentTarget)}
              color="primary"
            >
              <Settings />
            </IconButton>
          </Tooltip>
        </Box>
      </Box>

      {/* Notifications */}
      {notifications.length > 0 && (
        <Box sx={{ mb: 3 }}>
          {notifications.slice(0, 3).map((notification) => (
            <Alert
              key={notification.id}
              severity={notification.type}
              onClose={() => removeNotification(notification.id)}
              sx={{ mb: 1 }}
            >
              <Typography variant="subtitle2">{notification.title}</Typography>
              {notification.message}
            </Alert>
          ))}
        </Box>
      )}

      {/* Erreur globale */}
      {error && (
        <Alert severity="error" sx={{ mb: 3 }}>
          <Typography variant="subtitle2">Erreur de connexion</Typography>
          {error}
        </Alert>
      )}

      {/* Dashboard principal */}
      <Grid container spacing={3}>
        {/* Vue d'ensemble système */}
        <Grid item xs={12} lg={8}>
          <SystemOverviewCard 
            data={overview}
            loading={loading}
            onRefresh={handleRefresh}
          />
        </Grid>

        {/* Alertes overview */}
        <Grid item xs={12} lg={4}>
          <AlertsOverview />
        </Grid>

        {/* Health Status */}
        <Grid item xs={12} md={6}>
          <HealthStatus />
        </Grid>

        {/* Métriques temps réel */}
        <Grid item xs={12} md={6}>
          <MetricsChart
            title="API Performance"
            metric="api_response_time"
            timeWindow="15m"
          />
        </Grid>

        {/* Charts métriques business */}
        <Grid item xs={12} lg={6}>
          <MetricsChart
            title="Activité Business"
            metric="business_activity"
            timeWindow="1h"
            chartType="area"
          />
        </Grid>

        <Grid item xs={12} lg={6}>
          <MetricsChart
            title="Système Resources"
            metric="system_resources"
            timeWindow="1h"
            chartType="line"
          />
        </Grid>
      </Grid>

      {/* Menu paramètres */}
      <Menu
        anchorEl={settingsAnchor}
        open={Boolean(settingsAnchor)}
        onClose={() => setSettingsAnchor(null)}
      >
        <MenuItem onClick={() => setViewMode('grid')}>
          <Dashboard sx={{ mr: 1 }} />
          Vue Grille
        </MenuItem>
        <MenuItem onClick={() => setViewMode('compact')}>
          <Timeline sx={{ mr: 1 }} />
          Vue Compacte
        </MenuItem>
      </Menu>
    </Box>
  );
};

export default MonitoringDashboard;