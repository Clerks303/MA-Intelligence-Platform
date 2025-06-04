/**
 * SystemOverviewCard - Vue d'ensemble santé système
 * US-004: Widget principal dashboard avec KPIs temps réel
 */

import React from 'react';
import {
  Card,
  CardContent,
  CardHeader,
  Grid,
  Typography,
  Box,
  Chip,
  LinearProgress,
  IconButton,
  Tooltip,
  CircularProgress
} from '@mui/material';
import {
  CheckCircle,
  Warning,
  Error,
  Refresh,
  TrendingUp,
  TrendingDown,
  Speed,
  Memory,
  Storage
} from '@mui/icons-material';

const SystemOverviewCard = ({ data, loading, onRefresh }) => {
  if (loading && !data) {
    return (
      <Card>
        <CardContent sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
          <CircularProgress />
        </CardContent>
      </Card>
    );
  }

  if (!data) {
    return (
      <Card>
        <CardContent>
          <Typography color="textSecondary">
            Aucune donnée disponible
          </Typography>
        </CardContent>
      </Card>
    );
  }

  const systemStatus = data.system_status || {};
  const keyMetrics = data.key_metrics || {};
  const performance = data.performance_summary || {};
  const businessHealth = data.business_health || {};

  // Fonctions utilitaires
  const getHealthIcon = (status) => {
    switch (status) {
      case 'healthy':
        return <CheckCircle color="success" />;
      case 'degraded':
        return <Warning color="warning" />;
      case 'unhealthy':
        return <Error color="error" />;
      default:
        return <Warning color="disabled" />;
    }
  };

  const getHealthColor = (status) => {
    switch (status) {
      case 'healthy': return 'success';
      case 'degraded': return 'warning';
      case 'unhealthy': return 'error';
      default: return 'default';
    }
  };

  const formatNumber = (num) => {
    if (num >= 1000) {
      return (num / 1000).toFixed(1) + 'k';
    }
    return num?.toString() || '0';
  };

  const formatPercent = (num) => {
    return `${num?.toFixed(1) || 0}%`;
  };

  const formatMs = (ms) => {
    return `${ms?.toFixed(0) || 0}ms`;
  };

  return (
    <Card sx={{ height: '100%' }}>
      <CardHeader
        title="Vue d'ensemble Système"
        action={
          <Tooltip title="Actualiser">
            <IconButton onClick={onRefresh} disabled={loading}>
              <Refresh />
            </IconButton>
          </Tooltip>
        }
        subheader={`Dernière mise à jour: ${new Date().toLocaleTimeString()}`}
      />
      
      <CardContent>
        {/* Statut global système */}
        <Box sx={{ mb: 3 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 2 }}>
            {getHealthIcon(systemStatus.overall_health)}
            <Typography variant="h6">
              Statut Global: {systemStatus.overall_health || 'Inconnu'}
            </Typography>
            <Chip
              label={`${systemStatus.services_healthy || 0}/${systemStatus.services_total || 0} Services`}
              color={getHealthColor(systemStatus.overall_health)}
              variant="outlined"
            />
          </Box>
          
          {systemStatus.availability_percent !== undefined && (
            <Box sx={{ mb: 2 }}>
              <Typography variant="body2" color="textSecondary" gutterBottom>
                Disponibilité: {formatPercent(systemStatus.availability_percent)}
              </Typography>
              <LinearProgress
                variant="determinate"
                value={systemStatus.availability_percent}
                color={systemStatus.availability_percent > 95 ? 'success' : 'warning'}
              />
            </Box>
          )}
        </Box>

        {/* Métriques clés */}
        <Grid container spacing={3}>
          {/* API Performance */}
          <Grid item xs={12} sm={6} md={3}>
            <Box sx={{ textAlign: 'center', p: 2, bgcolor: 'background.paper', borderRadius: 1 }}>
              <Speed color="primary" sx={{ fontSize: 32, mb: 1 }} />
              <Typography variant="h4" color="primary">
                {formatMs(keyMetrics.avg_response_time_ms)}
              </Typography>
              <Typography variant="body2" color="textSecondary">
                Temps Réponse API
              </Typography>
              {keyMetrics.avg_response_time_ms > 1000 && (
                <Chip
                  label="Lent"
                  size="small"
                  color="warning"
                  sx={{ mt: 1 }}
                />
              )}
            </Box>
          </Grid>

          {/* Requests par heure */}
          <Grid item xs={12} sm={6} md={3}>
            <Box sx={{ textAlign: 'center', p: 2, bgcolor: 'background.paper', borderRadius: 1 }}>
              <TrendingUp color="success" sx={{ fontSize: 32, mb: 1 }} />
              <Typography variant="h4" color="success.main">
                {formatNumber(keyMetrics.api_requests_last_hour)}
              </Typography>
              <Typography variant="body2" color="textSecondary">
                Requests/Heure
              </Typography>
            </Box>
          </Grid>

          {/* Taux d'erreur */}
          <Grid item xs={12} sm={6} md={3}>
            <Box sx={{ textAlign: 'center', p: 2, bgcolor: 'background.paper', borderRadius: 1 }}>
              {keyMetrics.error_rate_percent > 5 ? (
                <TrendingDown color="error" sx={{ fontSize: 32, mb: 1 }} />
              ) : (
                <CheckCircle color="success" sx={{ fontSize: 32, mb: 1 }} />
              )}
              <Typography
                variant="h4"
                color={keyMetrics.error_rate_percent > 5 ? 'error.main' : 'success.main'}
              >
                {formatPercent(keyMetrics.error_rate_percent)}
              </Typography>
              <Typography variant="body2" color="textSecondary">
                Taux d'Erreur
              </Typography>
            </Box>
          </Grid>

          {/* Utilisateurs actifs */}
          <Grid item xs={12} sm={6} md={3}>
            <Box sx={{ textAlign: 'center', p: 2, bgcolor: 'background.paper', borderRadius: 1 }}>
              <Memory color="info" sx={{ fontSize: 32, mb: 1 }} />
              <Typography variant="h4" color="info.main">
                {keyMetrics.active_users || 0}
              </Typography>
              <Typography variant="body2" color="textSecondary">
                Utilisateurs Actifs
              </Typography>
            </Box>
          </Grid>
        </Grid>

        {/* Ressources système */}
        <Box sx={{ mt: 3 }}>
          <Typography variant="h6" gutterBottom>
            Ressources Système
          </Typography>
          
          <Grid container spacing={2}>
            <Grid item xs={12} sm={6}>
              <Box sx={{ mb: 2 }}>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                  <Typography variant="body2">CPU Usage</Typography>
                  <Typography variant="body2">
                    {formatPercent(performance.system_cpu_percent)}
                  </Typography>
                </Box>
                <LinearProgress
                  variant="determinate"
                  value={performance.system_cpu_percent || 0}
                  color={performance.system_cpu_percent > 80 ? 'error' : 
                         performance.system_cpu_percent > 60 ? 'warning' : 'success'}
                />
              </Box>
            </Grid>

            <Grid item xs={12} sm={6}>
              <Box sx={{ mb: 2 }}>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                  <Typography variant="body2">Memory Usage</Typography>
                  <Typography variant="body2">
                    {formatPercent(performance.system_memory_percent)}
                  </Typography>
                </Box>
                <LinearProgress
                  variant="determinate"
                  value={performance.system_memory_percent || 0}
                  color={performance.system_memory_percent > 85 ? 'error' : 
                         performance.system_memory_percent > 70 ? 'warning' : 'success'}
                />
              </Box>
            </Grid>
          </Grid>
        </Box>

        {/* Santé business */}
        <Box sx={{ mt: 3 }}>
          <Typography variant="h6" gutterBottom>
            Activité Business
          </Typography>
          
          <Grid container spacing={2}>
            <Grid item xs={6} sm={3}>
              <Typography variant="body2" color="textSecondary">
                Companies Scrapées
              </Typography>
              <Typography variant="h6">
                {businessHealth.companies_scraped_today || 0}
              </Typography>
            </Grid>
            
            <Grid item xs={6} sm={3}>
              <Typography variant="body2" color="textSecondary">
                Scores M&A
              </Typography>
              <Typography variant="h6">
                {businessHealth.ma_scores_calculated || 0}
              </Typography>
            </Grid>
            
            <Grid item xs={6} sm={3}>
              <Typography variant="body2" color="textSecondary">
                Exports Générés
              </Typography>
              <Typography variant="h6">
                {businessHealth.exports_generated || 0}
              </Typography>
            </Grid>
            
            <Grid item xs={6} sm={3}>
              <Typography variant="body2" color="textSecondary">
                Taux Succès Scraping
              </Typography>
              <Typography variant="h6" color={
                businessHealth.scraping_success_rate > 90 ? 'success.main' : 'warning.main'
              }>
                {formatPercent(businessHealth.scraping_success_rate)}
              </Typography>
            </Grid>
          </Grid>
        </Box>
      </CardContent>
    </Card>
  );
};

export default SystemOverviewCard;