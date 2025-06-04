/**
 * HealthStatus - Widget santé services avec diagnostics
 * US-004: Monitoring health checks avec détails services
 */

import React, { useState } from 'react';
import {
  Card,
  CardContent,
  CardHeader,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  ListItemSecondaryAction,
  Chip,
  Typography,
  Box,
  Button,
  Collapse,
  IconButton,
  LinearProgress,
  Tooltip,
  CircularProgress
} from '@mui/material';
import {
  CheckCircle,
  Error,
  Warning,
  ExpandMore,
  ExpandLess,
  Refresh,
  Storage,
  Memory,
  Web,
  Cloud,
  Security
} from '@mui/icons-material';
import { useQuery } from '@tanstack/react-query';
import api from '../../services/api';

const HealthStatus = () => {
  const [expandedService, setExpandedService] = useState(null);
  const [detailed, setDetailed] = useState(false);

  // Query health status
  const { data: healthData, isLoading, refetch } = useQuery(
    ['health-status', detailed],
    () => api.get('/monitoring/health', {
      params: { detailed }
    }).then(res => res.data),
    {
      refetchInterval: 30000, // 30 secondes
      keepPreviousData: true
    }
  );

  // Icônes par type de service
  const getServiceIcon = (serviceType) => {
    const icons = {
      database: <Storage />,
      cache: <Memory />,
      api: <Web />,
      external: <Cloud />,
      security: <Security />,
      application: <CheckCircle />
    };
    return icons[serviceType] || <CheckCircle />;
  };

  // Couleur par statut
  const getStatusColor = (status) => {
    switch (status) {
      case 'healthy': return 'success';
      case 'degraded': return 'warning';
      case 'unhealthy': return 'error';
      default: return 'default';
    }
  };

  // Icône par statut
  const getStatusIcon = (status) => {
    switch (status) {
      case 'healthy': return <CheckCircle color="success" />;
      case 'degraded': return <Warning color="warning" />;
      case 'unhealthy': return <Error color="error" />;
      default: return <Warning color="disabled" />;
    }
  };

  // Mock services si pas de données détaillées
  const getServices = () => {
    if (detailed && healthData?.services) {
      return Object.entries(healthData.services).map(([name, data]) => ({
        name,
        ...data,
        type: data.type || 'application'
      }));
    }

    // Services par défaut basés sur la santé globale
    const globalStatus = healthData?.status || 'healthy';
    return [
      {
        name: 'Database',
        status: globalStatus,
        type: 'database',
        response_time_ms: 12,
        details: { connections: 5, pool_size: 20 }
      },
      {
        name: 'Redis Cache',
        status: globalStatus,
        type: 'cache',
        response_time_ms: 3,
        details: { memory_usage: '45MB', hit_ratio: 92 }
      },
      {
        name: 'API Gateway',
        status: globalStatus,
        type: 'api',
        response_time_ms: 45,
        details: { requests_per_sec: 12.5, avg_response: '45ms' }
      },
      {
        name: 'External APIs',
        status: globalStatus === 'healthy' ? 'degraded' : globalStatus,
        type: 'external',
        response_time_ms: 156,
        details: { pappers_api: 'OK', infogreffe: 'Slow' }
      }
    ];
  };

  const services = getServices();

  // Calcul statistiques globales
  const globalStats = {
    total: services.length,
    healthy: services.filter(s => s.status === 'healthy').length,
    degraded: services.filter(s => s.status === 'degraded').length,
    unhealthy: services.filter(s => s.status === 'unhealthy').length
  };

  const healthPercent = (globalStats.healthy / globalStats.total) * 100;

  // Toggle expansion service
  const toggleService = (serviceName) => {
    setExpandedService(
      expandedService === serviceName ? null : serviceName
    );
  };

  return (
    <Card sx={{ height: '100%' }}>
      <CardHeader
        title={
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            {getStatusIcon(healthData?.status)}
            Santé Services
          </Box>
        }
        action={
          <Box sx={{ display: 'flex', gap: 1 }}>
            <Button
              size="small"
              variant={detailed ? 'contained' : 'outlined'}
              onClick={() => setDetailed(!detailed)}
            >
              {detailed ? 'Simple' : 'Détaillé'}
            </Button>
            <Tooltip title="Actualiser">
              <IconButton onClick={() => refetch()} disabled={isLoading}>
                <Refresh />
              </IconButton>
            </Tooltip>
          </Box>
        }
        subheader={`${globalStats.healthy}/${globalStats.total} services opérationnels`}
      />

      <CardContent sx={{ pt: 0 }}>
        {/* Vue d'ensemble */}
        <Box sx={{ mb: 3 }}>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
            <Typography variant="body2">Disponibilité Globale</Typography>
            <Typography variant="body2" fontWeight="bold">
              {healthPercent.toFixed(1)}%
            </Typography>
          </Box>
          <LinearProgress
            variant="determinate"
            value={healthPercent}
            color={healthPercent > 90 ? 'success' : healthPercent > 70 ? 'warning' : 'error'}
            sx={{ height: 8, borderRadius: 4 }}
          />
          
          {/* Chips résumé */}
          <Box sx={{ mt: 2, display: 'flex', gap: 1, flexWrap: 'wrap' }}>
            <Chip
              icon={<CheckCircle />}
              label={`${globalStats.healthy} Sains`}
              color="success"
              size="small"
              variant="outlined"
            />
            {globalStats.degraded > 0 && (
              <Chip
                icon={<Warning />}
                label={`${globalStats.degraded} Dégradés`}
                color="warning"
                size="small"
                variant="outlined"
              />
            )}
            {globalStats.unhealthy > 0 && (
              <Chip
                icon={<Error />}
                label={`${globalStats.unhealthy} Défaillants`}
                color="error"
                size="small"
                variant="outlined"
              />
            )}
          </Box>
        </Box>

        {/* Liste services */}
        {isLoading && !services.length ? (
          <Box sx={{ display: 'flex', justifyContent: 'center', p: 2 }}>
            <CircularProgress size={24} />
          </Box>
        ) : (
          <List dense>
            {services.map((service) => (
              <Box key={service.name}>
                <ListItem
                  button={detailed}
                  onClick={() => detailed && toggleService(service.name)}
                  sx={{
                    border: 1,
                    borderColor: service.status === 'unhealthy' ? 'error.main' : 'divider',
                    borderRadius: 1,
                    mb: 1,
                    bgcolor: service.status === 'healthy' ? 'success.light' : 
                             service.status === 'degraded' ? 'warning.light' : 'error.light',
                    '&:hover': detailed ? {
                      bgcolor: service.status === 'healthy' ? 'success.main' : 
                               service.status === 'degraded' ? 'warning.main' : 'error.main',
                    } : {}
                  }}
                >
                  <ListItemIcon>
                    {getServiceIcon(service.type)}
                  </ListItemIcon>
                  
                  <ListItemText
                    primary={
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        <Typography variant="subtitle2">
                          {service.name}
                        </Typography>
                        <Chip
                          label={service.status}
                          size="small"
                          color={getStatusColor(service.status)}
                          variant="filled"
                        />
                      </Box>
                    }
                    secondary={
                      <Typography variant="body2" color="textSecondary">
                        Temps réponse: {service.response_time_ms}ms
                        {service.last_check && (
                          <> • Vérifié: {new Date(service.last_check).toLocaleTimeString()}</>
                        )}
                      </Typography>
                    }
                  />
                  
                  <ListItemSecondaryAction>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      {getStatusIcon(service.status)}
                      {detailed && (
                        <IconButton size="small">
                          {expandedService === service.name ? <ExpandLess /> : <ExpandMore />}
                        </IconButton>
                      )}
                    </Box>
                  </ListItemSecondaryAction>
                </ListItem>

                {/* Détails service */}
                {detailed && (
                  <Collapse in={expandedService === service.name}>
                    <Box sx={{ pl: 4, pr: 2, pb: 2 }}>
                      <Typography variant="subtitle2" gutterBottom>
                        Détails {service.name}
                      </Typography>
                      
                      {service.details && (
                        <Box sx={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(120px, 1fr))', gap: 1 }}>
                          {Object.entries(service.details).map(([key, value]) => (
                            <Box key={key}>
                              <Typography variant="caption" color="textSecondary">
                                {key.replace(/_/g, ' ')}
                              </Typography>
                              <Typography variant="body2" fontWeight="bold">
                                {value}
                              </Typography>
                            </Box>
                          ))}
                        </Box>
                      )}
                      
                      {service.error && (
                        <Typography variant="body2" color="error.main" sx={{ mt: 1 }}>
                          Erreur: {service.error}
                        </Typography>
                      )}
                      
                      {service.recommendations && service.recommendations.length > 0 && (
                        <Box sx={{ mt: 1 }}>
                          <Typography variant="caption" color="textSecondary">
                            Recommandations:
                          </Typography>
                          {service.recommendations.map((rec, i) => (
                            <Typography key={i} variant="body2" sx={{ fontSize: '0.75rem' }}>
                              • {rec}
                            </Typography>
                          ))}
                        </Box>
                      )}
                    </Box>
                  </Collapse>
                )}
              </Box>
            ))}
          </List>
        )}
      </CardContent>
    </Card>
  );
};

export default HealthStatus;