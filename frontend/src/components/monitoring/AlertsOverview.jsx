/**
 * AlertsOverview - Widget gestion alertes avec actions
 * US-004: Interface alertes avec acquittement et résolution
 */

import React, { useState } from 'react';
import {
  Card,
  CardContent,
  CardHeader,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  ListItemSecondaryAction,
  IconButton,
  Chip,
  Typography,
  Box,
  Button,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Badge,
  Tooltip,
  CircularProgress
} from '@mui/material';
import {
  Warning,
  Error,
  Info,
  CheckCircle,
  Refresh,
  FilterList,
  Check,
  Close,
  MoreVert,
  Notifications,
  NotificationsActive
} from '@mui/icons-material';
import { useMonitoring } from '../../contexts/MonitoringContext';

const AlertsOverview = () => {
  const {
    alerts,
    loading,
    acknowledgeAlert,
    resolveAlert,
    queries
  } = useMonitoring();

  const [actionDialog, setActionDialog] = useState(null);
  const [comment, setComment] = useState('');
  const [filterSeverity, setFilterSeverity] = useState('all');
  const [filterStatus, setFilterStatus] = useState('active');

  // Données alertes
  const alertsData = alerts || {};
  const activeAlerts = alertsData.filtered_active_alerts || [];
  const summary = alertsData.summary || {};

  // Icônes par sévérité
  const getSeverityIcon = (severity) => {
    switch (severity) {
      case 'emergency':
      case 'critical':
        return <Error color="error" />;
      case 'error':
        return <Warning color="error" />;
      case 'warning':
        return <Warning color="warning" />;
      case 'info':
        return <Info color="info" />;
      default:
        return <Info color="disabled" />;
    }
  };

  // Couleurs par sévérité
  const getSeverityColor = (severity) => {
    switch (severity) {
      case 'emergency': return 'error';
      case 'critical': return 'error';
      case 'error': return 'warning';
      case 'warning': return 'warning';
      case 'info': return 'info';
      default: return 'default';
    }
  };

  // Formatage timestamp
  const formatTime = (timestamp) => {
    const date = new Date(timestamp);
    const now = new Date();
    const diff = now - date;
    
    if (diff < 60000) return 'À l\'instant';
    if (diff < 3600000) return `${Math.floor(diff / 60000)}min`;
    if (diff < 86400000) return `${Math.floor(diff / 3600000)}h`;
    return date.toLocaleDateString();
  };

  // Gestion actions alertes
  const handleAcknowledge = async (alertId) => {
    try {
      await acknowledgeAlert(alertId, comment);
      setActionDialog(null);
      setComment('');
    } catch (error) {
      console.error('Erreur acquittement:', error);
    }
  };

  const handleResolve = async (alertId) => {
    try {
      await resolveAlert(alertId, comment);
      setActionDialog(null);
      setComment('');
    } catch (error) {
      console.error('Erreur résolution:', error);
    }
  };

  // Filtrage alertes
  const filteredAlerts = activeAlerts.filter(alert => {
    if (filterSeverity !== 'all' && alert.severity !== filterSeverity) {
      return false;
    }
    if (filterStatus !== 'all' && alert.status !== filterStatus) {
      return false;
    }
    return true;
  });

  return (
    <>
      <Card sx={{ height: '100%' }}>
        <CardHeader
          title={
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <Badge badgeContent={summary.unacknowledged_critical || 0} color="error">
                <NotificationsActive />
              </Badge>
              Alertes Actives
            </Box>
          }
          action={
            <Tooltip title="Actualiser">
              <IconButton 
                onClick={() => queries.alerts.refetch()}
                disabled={loading}
              >
                <Refresh />
              </IconButton>
            </Tooltip>
          }
          subheader={`${summary.total_active_alerts || 0} alerte(s) active(s)`}
        />

        <CardContent sx={{ pt: 0 }}>
          {/* Résumé par sévérité */}
          <Box sx={{ mb: 2, display: 'flex', gap: 1, flexWrap: 'wrap' }}>
            {summary.critical_alerts > 0 && (
              <Chip
                icon={<Error />}
                label={`${summary.critical_alerts} Critique(s)`}
                color="error"
                size="small"
              />
            )}
            {summary.warning_alerts > 0 && (
              <Chip
                icon={<Warning />}
                label={`${summary.warning_alerts} Warning(s)`}
                color="warning"
                size="small"
              />
            )}
            {summary.info_alerts > 0 && (
              <Chip
                icon={<Info />}
                label={`${summary.info_alerts} Info(s)`}
                color="info"
                size="small"
              />
            )}
          </Box>

          {/* Filtres */}
          <Box sx={{ mb: 2, display: 'flex', gap: 1 }}>
            <FormControl size="small" sx={{ minWidth: 100 }}>
              <InputLabel>Sévérité</InputLabel>
              <Select
                value={filterSeverity}
                onChange={(e) => setFilterSeverity(e.target.value)}
                label="Sévérité"
              >
                <MenuItem value="all">Toutes</MenuItem>
                <MenuItem value="emergency">Emergency</MenuItem>
                <MenuItem value="critical">Critical</MenuItem>
                <MenuItem value="error">Error</MenuItem>
                <MenuItem value="warning">Warning</MenuItem>
                <MenuItem value="info">Info</MenuItem>
              </Select>
            </FormControl>

            <FormControl size="small" sx={{ minWidth: 100 }}>
              <InputLabel>Statut</InputLabel>
              <Select
                value={filterStatus}
                onChange={(e) => setFilterStatus(e.target.value)}
                label="Statut"
              >
                <MenuItem value="all">Tous</MenuItem>
                <MenuItem value="active">Actives</MenuItem>
                <MenuItem value="acknowledged">Acquittées</MenuItem>
              </Select>
            </FormControl>
          </Box>

          {/* Liste alertes */}
          {loading && filteredAlerts.length === 0 ? (
            <Box sx={{ display: 'flex', justifyContent: 'center', p: 2 }}>
              <CircularProgress size={24} />
            </Box>
          ) : filteredAlerts.length === 0 ? (
            <Box sx={{ textAlign: 'center', p: 2 }}>
              <CheckCircle color="success" sx={{ fontSize: 48, mb: 1 }} />
              <Typography color="textSecondary">
                Aucune alerte active
              </Typography>
            </Box>
          ) : (
            <List dense>
              {filteredAlerts.slice(0, 10).map((alert) => (
                <ListItem
                  key={alert.id}
                  sx={{
                    border: 1,
                    borderColor: alert.severity === 'critical' || alert.severity === 'emergency' 
                      ? 'error.main' : 'divider',
                    borderRadius: 1,
                    mb: 1,
                    bgcolor: alert.status === 'acknowledged' ? 'action.hover' : 'background.paper'
                  }}
                >
                  <ListItemIcon>
                    {getSeverityIcon(alert.severity)}
                  </ListItemIcon>
                  
                  <ListItemText
                    primary={
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        <Typography variant="subtitle2" noWrap>
                          {alert.title}
                        </Typography>
                        <Chip
                          label={alert.severity}
                          size="small"
                          color={getSeverityColor(alert.severity)}
                          variant="outlined"
                        />
                      </Box>
                    }
                    secondary={
                      <Box>
                        <Typography variant="body2" noWrap>
                          {alert.description}
                        </Typography>
                        <Typography variant="caption" color="textSecondary">
                          {formatTime(alert.created_at)} • Valeur: {alert.current_value?.toFixed(1)}
                        </Typography>
                      </Box>
                    }
                  />
                  
                  <ListItemSecondaryAction>
                    <Box sx={{ display: 'flex', gap: 0.5 }}>
                      {alert.status === 'active' && (
                        <>
                          <Tooltip title="Acquitter">
                            <IconButton
                              size="small"
                              onClick={() => setActionDialog({ type: 'acknowledge', alert })}
                              color="primary"
                            >
                              <Check />
                            </IconButton>
                          </Tooltip>
                          <Tooltip title="Résoudre">
                            <IconButton
                              size="small"
                              onClick={() => setActionDialog({ type: 'resolve', alert })}
                              color="success"
                            >
                              <Close />
                            </IconButton>
                          </Tooltip>
                        </>
                      )}
                      {alert.status === 'acknowledged' && (
                        <Chip
                          label="Acquittée"
                          size="small"
                          color="primary"
                          variant="outlined"
                        />
                      )}
                    </Box>
                  </ListItemSecondaryAction>
                </ListItem>
              ))}
            </List>
          )}

          {/* Lien vers page alertes complète */}
          {filteredAlerts.length > 10 && (
            <Button
              fullWidth
              variant="outlined"
              sx={{ mt: 2 }}
              onClick={() => {/* Navigation vers page alertes */}}
            >
              Voir toutes les alertes ({filteredAlerts.length})
            </Button>
          )}
        </CardContent>
      </Card>

      {/* Dialog actions alertes */}
      <Dialog
        open={Boolean(actionDialog)}
        onClose={() => setActionDialog(null)}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle>
          {actionDialog?.type === 'acknowledge' ? 'Acquitter l\'alerte' : 'Résoudre l\'alerte'}
        </DialogTitle>
        
        <DialogContent>
          {actionDialog && (
            <Box>
              <Typography variant="subtitle1" gutterBottom>
                {actionDialog.alert.title}
              </Typography>
              <Typography variant="body2" color="textSecondary" gutterBottom>
                {actionDialog.alert.description}
              </Typography>
              
              <TextField
                fullWidth
                multiline
                rows={3}
                label="Commentaire (optionnel)"
                value={comment}
                onChange={(e) => setComment(e.target.value)}
                margin="normal"
                placeholder={
                  actionDialog.type === 'acknowledge' 
                    ? "Expliquez les actions prises..."
                    : "Décrivez la résolution..."
                }
              />
            </Box>
          )}
        </DialogContent>
        
        <DialogActions>
          <Button onClick={() => setActionDialog(null)}>
            Annuler
          </Button>
          <Button
            variant="contained"
            color={actionDialog?.type === 'acknowledge' ? 'primary' : 'success'}
            onClick={() => {
              if (actionDialog?.type === 'acknowledge') {
                handleAcknowledge(actionDialog.alert.id);
              } else {
                handleResolve(actionDialog.alert.id);
              }
            }}
          >
            {actionDialog?.type === 'acknowledge' ? 'Acquitter' : 'Résoudre'}
          </Button>
        </DialogActions>
      </Dialog>
    </>
  );
};

export default AlertsOverview;