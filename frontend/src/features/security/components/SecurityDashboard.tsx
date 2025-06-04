/**
 * Security Dashboard - M&A Intelligence Platform
 * Sprint 4 - Dashboard principal du module sécurité avec navigation vers toutes les fonctionnalités
 */

import React, { useState } from 'react';
import {
  Box,
  Paper,
  Typography,
  Button,
  Card,
  CardContent,
  CardActions,
  Grid,
  Tabs,
  Tab,
  Alert,
  Chip,
  LinearProgress,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Divider,
  IconButton,
  Tooltip,
  Avatar
} from '@mui/material';
import {
  Security as SecurityIcon,
  People as PeopleIcon,
  AdminPanelSettings as RolesIcon,
  VpnKey as PermissionsIcon,
  History as AuditIcon,
  Shield as ShieldIcon,
  Warning as WarningIcon,
  CheckCircle as CheckCircleIcon,
  TrendingUp as TrendingUpIcon,
  TrendingDown as TrendingDownIcon,
  Refresh as RefreshIcon,
  Settings as SettingsIcon,
  Notifications as NotificationsIcon,
  DeviceHub as DevicesIcon,
  Assessment as AssessmentIcon,
  Error as ErrorIcon,
  Info as InfoIcon
} from '@mui/icons-material';
import { useSecurityDashboard, useSecurityAlerts, useSystemMonitoring } from '../hooks/useMonitoring';
import { useUsers, useRoles } from '../hooks/useRBAC';
import { useAuditStats } from '../hooks/useAudit';
import { UserManagement } from './UserManagement';
import { RoleManagement } from './RoleManagement';
import { AuditVisualization } from './AuditVisualization';
import { MFAInterface } from './MFAInterface';

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel({ children, value, index }: TabPanelProps) {
  return (
    <div role="tabpanel" hidden={value !== index}>
      {value === index && <Box sx={{ py: 3 }}>{children}</Box>}
    </div>
  );
}

interface MetricCardProps {
  title: string;
  value: number | string;
  icon: React.ReactNode;
  color?: 'primary' | 'secondary' | 'success' | 'warning' | 'error' | 'info';
  trend?: number;
  subtitle?: string;
  onClick?: () => void;
}

const MetricCard: React.FC<MetricCardProps> = ({ 
  title, 
  value, 
  icon, 
  color = 'primary',
  trend,
  subtitle,
  onClick 
}) => {
  return (
    <Card 
      sx={{ 
        height: '100%',
        cursor: onClick ? 'pointer' : 'default',
        '&:hover': onClick ? { boxShadow: 4 } : {},
        border: `1px solid`,
        borderColor: `${color}.main`,
        borderTop: `4px solid`,
        borderTopColor: `${color}.main`
      }}
      onClick={onClick}
    >
      <CardContent>
        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 1 }}>
          <Avatar sx={{ bgcolor: `${color}.main`, width: 40, height: 40 }}>
            {icon}
          </Avatar>
          {trend !== undefined && (
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
              {trend > 0 ? (
                <TrendingUpIcon color="success" fontSize="small" />
              ) : trend < 0 ? (
                <TrendingDownIcon color="error" fontSize="small" />
              ) : null}
              <Typography variant="caption" color={trend > 0 ? 'success.main' : trend < 0 ? 'error.main' : 'text.secondary'}>
                {trend > 0 ? '+' : ''}{trend.toFixed(1)}%
              </Typography>
            </Box>
          )}
        </Box>
        <Typography variant="h4" color={`${color}.main`} gutterBottom>
          {value}
        </Typography>
        <Typography variant="body2" color="text.secondary">
          {title}
        </Typography>
        {subtitle && (
          <Typography variant="caption" color="text.secondary">
            {subtitle}
          </Typography>
        )}
      </CardContent>
    </Card>
  );
};

export const SecurityDashboard: React.FC = () => {
  const [tabValue, setTabValue] = useState(0);
  
  // Hooks for dashboard data
  const { dashboard, metrics, realTimeMetrics, trends, isLoading: isDashboardLoading, refetch } = useSecurityDashboard();
  const { alerts, alertMetrics, priorityAlerts } = useSecurityAlerts();
  const { globalStatus, overview } = useSystemMonitoring();
  const { userMetrics } = useUsers();
  const { roleMetrics } = useRoles();
  const { stats: auditStats } = useAuditStats();

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'healthy': return 'success';
      case 'warning': return 'warning';
      case 'critical': return 'error';
      default: return 'info';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'healthy': return <CheckCircleIcon />;
      case 'warning': return <WarningIcon />;
      case 'critical': return <ErrorIcon />;
      default: return <InfoIcon />;
    }
  };

  return (
    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      {/* Header */}
      <Paper sx={{ p: 3, mb: 3 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
          <Box>
            <Typography variant="h4" component="h1" gutterBottom>
              Centre de sécurité
            </Typography>
            <Typography variant="body1" color="text.secondary">
              Surveillance et administration de la sécurité de la plateforme
            </Typography>
          </Box>
          <Box sx={{ display: 'flex', gap: 1, alignItems: 'center' }}>
            {/* Global Status */}
            <Chip
              icon={getStatusIcon(globalStatus)}
              label={`Statut: ${globalStatus}`}
              color={getStatusColor(globalStatus) as any}
              variant="outlined"
            />
            <Tooltip title="Actualiser les données">
              <IconButton onClick={refetch} disabled={isDashboardLoading}>
                <RefreshIcon />
              </IconButton>
            </Tooltip>
          </Box>
        </Box>

        {/* Quick Metrics */}
        <Grid container spacing={2}>
          <Grid item xs={12} md={3}>
            <MetricCard
              title="Utilisateurs actifs"
              value={userMetrics.active}
              icon={<PeopleIcon />}
              color="primary"
              trend={trends?.loginAttempts}
              subtitle={`Total: ${userMetrics.total}`}
              onClick={() => setTabValue(1)}
            />
          </Grid>
          <Grid item xs={12} md={3}>
            <MetricCard
              title="Alertes critiques"
              value={alertMetrics.criticalUnresolved}
              icon={<WarningIcon />}
              color={alertMetrics.criticalUnresolved > 0 ? 'error' : 'success'}
              subtitle={`Total alertes: ${alertMetrics.total}`}
              onClick={() => setTabValue(4)}
            />
          </Grid>
          <Grid item xs={12} md={3}>
            <MetricCard
              title="Sessions actives"
              value={overview.activeSessions}
              icon={<DevicesIcon />}
              color="info"
              subtitle="Connexions en cours"
            />
          </Grid>
          <Grid item xs={12} md={3}>
            <MetricCard
              title="Menaces bloquées"
              value={overview.threatsBlocked}
              icon={<ShieldIcon />}
              color="success"
              trend={trends?.threatsDetected}
              subtitle="Dernières 24h"
            />
          </Grid>
        </Grid>

        {/* System Health Progress */}
        {realTimeMetrics && (
          <Box sx={{ mt: 3 }}>
            <Typography variant="h6" gutterBottom>
              Santé du système
            </Typography>
            <Grid container spacing={2}>
              <Grid item xs={12} md={4}>
                <Box>
                  <Typography variant="body2" color="text.secondary">Charge système</Typography>
                  <LinearProgress 
                    variant="determinate" 
                    value={realTimeMetrics.system_load} 
                    color={realTimeMetrics.system_load > 80 ? 'error' : realTimeMetrics.system_load > 60 ? 'warning' : 'success'}
                    sx={{ height: 8, borderRadius: 4 }}
                  />
                  <Typography variant="caption">{realTimeMetrics.system_load}%</Typography>
                </Box>
              </Grid>
              <Grid item xs={12} md={4}>
                <Box>
                  <Typography variant="body2" color="text.secondary">Temps de réponse</Typography>
                  <LinearProgress 
                    variant="determinate" 
                    value={Math.min(realTimeMetrics.response_time / 10, 100)} 
                    color={realTimeMetrics.response_time > 500 ? 'error' : realTimeMetrics.response_time > 200 ? 'warning' : 'success'}
                    sx={{ height: 8, borderRadius: 4 }}
                  />
                  <Typography variant="caption">{realTimeMetrics.response_time}ms</Typography>
                </Box>
              </Grid>
              <Grid item xs={12} md={4}>
                <Box>
                  <Typography variant="body2" color="text.secondary">Menaces actuelles</Typography>
                  <LinearProgress 
                    variant="determinate" 
                    value={Math.min(realTimeMetrics.current_threats * 10, 100)} 
                    color={realTimeMetrics.current_threats > 5 ? 'error' : realTimeMetrics.current_threats > 2 ? 'warning' : 'success'}
                    sx={{ height: 8, borderRadius: 4 }}
                  />
                  <Typography variant="caption">{realTimeMetrics.current_threats} actives</Typography>
                </Box>
              </Grid>
            </Grid>
          </Box>
        )}
      </Paper>

      {/* Main Navigation Tabs */}
      <Paper sx={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <Tabs value={tabValue} onChange={(_, newValue) => setTabValue(newValue)}>
            <Tab 
              label="Tableau de bord" 
              icon={<AssessmentIcon />} 
              iconPosition="start"
            />
            <Tab 
              label="Utilisateurs" 
              icon={<PeopleIcon />} 
              iconPosition="start"
            />
            <Tab 
              label="Rôles" 
              icon={<RolesIcon />} 
              iconPosition="start"
            />
            <Tab 
              label="MFA" 
              icon={<SecurityIcon />} 
              iconPosition="start"
            />
            <Tab 
              label="Audit" 
              icon={<AuditIcon />} 
              iconPosition="start"
            />
          </Tabs>
        </Box>

        {/* Tab Panels */}
        <Box sx={{ flex: 1, overflow: 'hidden' }}>
          {/* Dashboard Overview */}
          <TabPanel value={tabValue} index={0}>
            <Grid container spacing={3}>
              {/* Recent Alerts */}
              <Grid item xs={12} md={6}>
                <Card sx={{ height: 400 }}>
                  <CardContent>
                    <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <WarningIcon color="warning" />
                      Alertes récentes
                    </Typography>
                    {priorityAlerts.length > 0 ? (
                      <List sx={{ maxHeight: 300, overflow: 'auto' }}>
                        {priorityAlerts.slice(0, 5).map((alert) => (
                          <ListItem key={alert.alert_id} divider>
                            <ListItemIcon>
                              <Avatar 
                                sx={{ 
                                  bgcolor: alert.severity === 'CRITICAL' ? 'error.main' : 'warning.main',
                                  width: 32,
                                  height: 32
                                }}
                              >
                                {getStatusIcon(alert.severity.toLowerCase())}
                              </Avatar>
                            </ListItemIcon>
                            <ListItemText
                              primary={alert.title}
                              secondary={
                                <Box>
                                  <Typography variant="caption" color="text.secondary">
                                    {new Date(alert.created_at).toLocaleString('fr-FR')}
                                  </Typography>
                                  <br />
                                  <Chip label={alert.type} size="small" variant="outlined" />
                                </Box>
                              }
                            />
                          </ListItem>
                        ))}
                      </List>
                    ) : (
                      <Alert severity="success">
                        Aucune alerte critique en cours
                      </Alert>
                    )}
                  </CardContent>
                  <CardActions>
                    <Button size="small" onClick={() => setTabValue(4)}>
                      Voir toutes les alertes
                    </Button>
                  </CardActions>
                </Card>
              </Grid>

              {/* Security Statistics */}
              <Grid item xs={12} md={6}>
                <Card sx={{ height: 400 }}>
                  <CardContent>
                    <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <AssessmentIcon color="primary" />
                      Statistiques de sécurité
                    </Typography>
                    <Grid container spacing={2}>
                      <Grid item xs={6}>
                        <Box sx={{ textAlign: 'center', p: 2, border: '1px solid', borderColor: 'divider', borderRadius: 1 }}>
                          <Typography variant="h5" color="primary">
                            {((userMetrics.withMFA / userMetrics.total) * 100).toFixed(0)}%
                          </Typography>
                          <Typography variant="body2" color="text.secondary">
                            Utilisateurs avec MFA
                          </Typography>
                        </Box>
                      </Grid>
                      <Grid item xs={6}>
                        <Box sx={{ textAlign: 'center', p: 2, border: '1px solid', borderColor: 'divider', borderRadius: 1 }}>
                          <Typography variant="h5" color="success.main">
                            {auditStats?.successRate || 0}%
                          </Typography>
                          <Typography variant="body2" color="text.secondary">
                            Taux de succès
                          </Typography>
                        </Box>
                      </Grid>
                      <Grid item xs={6}>
                        <Box sx={{ textAlign: 'center', p: 2, border: '1px solid', borderColor: 'divider', borderRadius: 1 }}>
                          <Typography variant="h5" color="warning.main">
                            {roleMetrics.system}
                          </Typography>
                          <Typography variant="body2" color="text.secondary">
                            Rôles système
                          </Typography>
                        </Box>
                      </Grid>
                      <Grid item xs={6}>
                        <Box sx={{ textAlign: 'center', p: 2, border: '1px solid', borderColor: 'divider', borderRadius: 1 }}>
                          <Typography variant="h5" color="info.main">
                            {userMetrics.recentLogins}
                          </Typography>
                          <Typography variant="body2" color="text.secondary">
                            Connexions 24h
                          </Typography>
                        </Box>
                      </Grid>
                    </Grid>
                  </CardContent>
                </Card>
              </Grid>
            </Grid>
          </TabPanel>

          {/* User Management */}
          <TabPanel value={tabValue} index={1}>
            <UserManagement />
          </TabPanel>

          {/* Role Management */}
          <TabPanel value={tabValue} index={2}>
            <RoleManagement />
          </TabPanel>

          {/* MFA Interface */}
          <TabPanel value={tabValue} index={3}>
            <MFAInterface />
          </TabPanel>

          {/* Audit Visualization */}
          <TabPanel value={tabValue} index={4}>
            <AuditVisualization />
          </TabPanel>
        </Box>
      </Paper>
    </Box>
  );
};

export default SecurityDashboard;