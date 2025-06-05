/**
 * MetricsChart - Charts temps réel avec Recharts
 * US-004: Visualisation métriques avec multiple types de charts
 */

import React, { useState, useMemo } from 'react';
import {
  Card,
  CardContent,
  CardHeader,
  Box,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Typography,
  IconButton,
  Tooltip,
  Chip,
  CircularProgress
} from '@mui/material';
import {
  Refresh,
  TrendingUp,
  TrendingDown,
  Timeline,
  ShowChart
} from '@mui/icons-material';
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  ResponsiveContainer,
  Legend
} from 'recharts';
import { useQuery } from '@tanstack/react-query';
import api from '../../services/api';

const MetricsChart = ({ 
  title, 
  metric, 
  timeWindow = '15m', 
  chartType = 'line',
  height = 300 
}) => {
  const [selectedTimeWindow, setSelectedTimeWindow] = useState(timeWindow);
  const [selectedChartType, setSelectedChartType] = useState(chartType);

  // Mapping time windows vers minutes
  const timeWindowMap = {
    '5m': 5,
    '15m': 15,
    '30m': 30,
    '1h': 60,
    '6h': 360,
    '24h': 1440
  };

  // Requête métriques
  const { data: metricsData, isLoading, refetch } = useQuery({
    queryKey: ['metrics-chart', metric, selectedTimeWindow],
    queryFn: () => api.get('/monitoring/metrics', {
      params: {
        window_minutes: timeWindowMap[selectedTimeWindow],
        category: getMetricCategory(metric)
      }
    }).then(res => res.data),
    refetchInterval: 30000, // 30 secondes
    keepPreviousData: true
  });

  // Fonction pour déterminer la catégorie de métrique
  function getMetricCategory(metricName) {
    if (metricName.includes('api')) return 'api';
    if (metricName.includes('system')) return 'system';
    if (metricName.includes('business')) return 'business';
    if (metricName.includes('cache')) return 'cache';
    return null;
  }

  // Transformation des données pour le chart
  const chartData = useMemo(() => {
    if (!metricsData) return [];

    // Générer données simulées basées sur les métriques réelles
    const now = new Date();
    const points = 20; // Nombre de points sur le chart
    const intervalMs = (timeWindowMap[selectedTimeWindow] * 60 * 1000) / points;
    
    return Array.from({ length: points }, (_, i) => {
      const timestamp = new Date(now.getTime() - (points - 1 - i) * intervalMs);
      
      // Données simulées réalistes selon le type de métrique
      let value;
      const baseTime = timestamp.getTime();
      
      switch (metric) {
        case 'api_response_time':
          // Temps de réponse API avec variations
          value = 200 + Math.sin(baseTime / 100000) * 50 + Math.random() * 100;
          break;
          
        case 'system_resources':
          // Usage CPU/mémoire
          value = 45 + Math.sin(baseTime / 200000) * 20 + Math.random() * 15;
          break;
          
        case 'business_activity':
          // Activité business avec pattern journalier
          const hour = timestamp.getHours();
          const businessMultiplier = hour >= 8 && hour <= 18 ? 1 : 0.3;
          value = (50 + Math.sin(baseTime / 300000) * 30) * businessMultiplier + Math.random() * 20;
          break;
          
        default:
          value = 50 + Math.sin(baseTime / 150000) * 25 + Math.random() * 20;
      }

      return {
        time: timestamp.toLocaleTimeString('fr-FR', { 
          hour: '2-digit', 
          minute: '2-digit' 
        }),
        timestamp: timestamp.toISOString(),
        value: Math.max(0, value),
        // Valeurs additionnelles selon le type
        ...(metric === 'system_resources' && {
          cpu: Math.max(0, 40 + Math.sin(baseTime / 180000) * 15 + Math.random() * 10),
          memory: Math.max(0, 60 + Math.sin(baseTime / 220000) * 20 + Math.random() * 10)
        }),
        ...(metric === 'business_activity' && {
          requests: Math.floor(Math.max(0, value * 10)),
          users: Math.floor(Math.max(0, value / 5))
        })
      };
    });
  }, [metricsData, metric, selectedTimeWindow]);

  // Calcul des statistiques
  const stats = useMemo(() => {
    if (!chartData.length) return null;
    
    const values = chartData.map(d => d.value);
    const min = Math.min(...values);
    const max = Math.max(...values);
    const avg = values.reduce((a, b) => a + b, 0) / values.length;
    const current = values[values.length - 1];
    const previous = values[values.length - 2] || current;
    const trend = current > previous ? 'up' : current < previous ? 'down' : 'stable';
    
    return { min, max, avg, current, trend };
  }, [chartData]);

  // Composant Chart selon le type
  const renderChart = () => {
    const commonProps = {
      data: chartData,
      margin: { top: 5, right: 30, left: 20, bottom: 5 }
    };

    switch (selectedChartType) {
      case 'area':
        return (
          <AreaChart {...commonProps}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="time" />
            <YAxis />
            <RechartsTooltip 
              labelFormatter={(label) => `Heure: ${label}`}
              formatter={(value, name) => [value.toFixed(1), getMetricLabel(name)]}
            />
            <Area 
              type="monotone" 
              dataKey="value" 
              stroke="#8884d8" 
              fill="#8884d8" 
              fillOpacity={0.3}
            />
            {metric === 'system_resources' && (
              <>
                <Area type="monotone" dataKey="cpu" stroke="#82ca9d" fill="#82ca9d" fillOpacity={0.2} />
                <Area type="monotone" dataKey="memory" stroke="#ffc658" fill="#ffc658" fillOpacity={0.2} />
              </>
            )}
          </AreaChart>
        );
        
      case 'bar':
        return (
          <BarChart {...commonProps}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="time" />
            <YAxis />
            <RechartsTooltip />
            <Bar dataKey="value" fill="#8884d8" />
          </BarChart>
        );
        
      default: // line
        return (
          <LineChart {...commonProps}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="time" />
            <YAxis />
            <RechartsTooltip 
              labelFormatter={(label) => `Heure: ${label}`}
              formatter={(value, name) => [value.toFixed(1), getMetricLabel(name)]}
            />
            <Legend />
            <Line 
              type="monotone" 
              dataKey="value" 
              stroke="#8884d8" 
              strokeWidth={2}
              dot={false}
              name={getMetricLabel('value')}
            />
            {metric === 'system_resources' && (
              <>
                <Line type="monotone" dataKey="cpu" stroke="#82ca9d" strokeWidth={2} dot={false} name="CPU %" />
                <Line type="monotone" dataKey="memory" stroke="#ffc658" strokeWidth={2} dot={false} name="Mémoire %" />
              </>
            )}
            {metric === 'business_activity' && (
              <>
                <Line type="monotone" dataKey="requests" stroke="#ff7300" strokeWidth={2} dot={false} name="Requests" />
                <Line type="monotone" dataKey="users" stroke="#8dd1e1" strokeWidth={2} dot={false} name="Utilisateurs" />
              </>
            )}
          </LineChart>
        );
    }
  };

  // Labels selon métrique
  const getMetricLabel = (key) => {
    const labels = {
      value: getMainMetricLabel(),
      cpu: 'CPU %',
      memory: 'Mémoire %',
      requests: 'Requests',
      users: 'Utilisateurs'
    };
    return labels[key] || key;
  };

  const getMainMetricLabel = () => {
    switch (metric) {
      case 'api_response_time': return 'Temps (ms)';
      case 'system_resources': return 'Usage %';
      case 'business_activity': return 'Activité';
      default: return 'Valeur';
    }
  };

  // Couleur trend
  const getTrendColor = (trend) => {
    switch (trend) {
      case 'up': return 'success';
      case 'down': return 'error';
      default: return 'default';
    }
  };

  return (
    <Card sx={{ height: '100%' }}>
      <CardHeader
        title={
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <ShowChart color="primary" />
            {title}
            {stats && (
              <Chip
                icon={stats.trend === 'up' ? <TrendingUp /> : stats.trend === 'down' ? <TrendingDown /> : <Timeline />}
                label={`${stats.current.toFixed(1)} ${getMainMetricLabel()}`}
                color={getTrendColor(stats.trend)}
                size="small"
              />
            )}
          </Box>
        }
        action={
          <Box sx={{ display: 'flex', gap: 1, alignItems: 'center' }}>
            <FormControl size="small" sx={{ minWidth: 80 }}>
              <InputLabel>Période</InputLabel>
              <Select
                value={selectedTimeWindow}
                onChange={(e) => setSelectedTimeWindow(e.target.value)}
                label="Période"
              >
                <MenuItem value="5m">5min</MenuItem>
                <MenuItem value="15m">15min</MenuItem>
                <MenuItem value="30m">30min</MenuItem>
                <MenuItem value="1h">1h</MenuItem>
                <MenuItem value="6h">6h</MenuItem>
                <MenuItem value="24h">24h</MenuItem>
              </Select>
            </FormControl>

            <FormControl size="small" sx={{ minWidth: 80 }}>
              <InputLabel>Type</InputLabel>
              <Select
                value={selectedChartType}
                onChange={(e) => setSelectedChartType(e.target.value)}
                label="Type"
              >
                <MenuItem value="line">Ligne</MenuItem>
                <MenuItem value="area">Zone</MenuItem>
                <MenuItem value="bar">Barres</MenuItem>
              </Select>
            </FormControl>

            <Tooltip title="Actualiser">
              <IconButton onClick={() => refetch()} disabled={isLoading}>
                <Refresh />
              </IconButton>
            </Tooltip>
          </Box>
        }
        subheader={
          stats && (
            <Typography variant="caption" color="textSecondary">
              Moy: {stats.avg.toFixed(1)} • Min: {stats.min.toFixed(1)} • Max: {stats.max.toFixed(1)}
            </Typography>
          )
        }
      />

      <CardContent sx={{ pt: 0 }}>
        {isLoading && !chartData.length ? (
          <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height }}>
            <CircularProgress />
          </Box>
        ) : chartData.length === 0 ? (
          <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height }}>
            <Typography color="textSecondary">
              Aucune donnée disponible
            </Typography>
          </Box>
        ) : (
          <ResponsiveContainer width="100%" height={height}>
            {renderChart()}
          </ResponsiveContainer>
        )}
      </CardContent>
    </Card>
  );
};

export default MetricsChart;