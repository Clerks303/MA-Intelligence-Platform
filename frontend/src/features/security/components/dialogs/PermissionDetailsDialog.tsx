/**
 * Permission Details Dialog - M&A Intelligence Platform
 * Sprint 4 - Dialog détaillé d'affichage des permissions d'un rôle
 */

import React, { useMemo } from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  Box,
  Typography,
  Chip,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Alert,
  Divider
} from '@mui/material';
import {
  VpnKey as PermissionIcon,
  ExpandMore as ExpandMoreIcon,
  Security as SecurityIcon,
  Close as CloseIcon,
  Info as InfoIcon
} from '@mui/icons-material';
import { Role, Permission, ResourceType, ActionType, PermissionScope } from '../../types';

interface PermissionDetailsDialogProps {
  open: boolean;
  onClose: () => void;
  role: Role | null;
}

const getScopeColor = (scope: PermissionScope) => {
  switch (scope) {
    case 'own': return 'default';
    case 'department': return 'info';
    case 'organization': return 'warning';
    case 'system': return 'error';
    case 'global': return 'error';
    default: return 'default';
  }
};

const getActionColor = (action: ActionType) => {
  switch (action) {
    case 'read': return 'info';
    case 'create': return 'success';
    case 'update': return 'warning';
    case 'delete': return 'error';
    case 'export': return 'secondary';
    case 'import': return 'secondary';
    default: return 'default';
  }
};

export const PermissionDetailsDialog: React.FC<PermissionDetailsDialogProps> = ({
  open,
  onClose,
  role
}) => {
  const permissionsByResource = useMemo(() => {
    if (!role?.permissions) return {};
    
    return role.permissions.reduce((acc, permission) => {
      if (!acc[permission.resource]) {
        acc[permission.resource] = [];
      }
      acc[permission.resource].push(permission);
      return acc;
    }, {} as Record<ResourceType, Permission[]>);
  }, [role?.permissions]);

  const permissionStats = useMemo(() => {
    if (!role?.permissions) return { total: 0, byScope: {}, byAction: {}, byResource: {} };
    
    const stats = {
      total: role.permissions.length,
      byScope: {} as Record<PermissionScope, number>,
      byAction: {} as Record<ActionType, number>,
      byResource: {} as Record<ResourceType, number>,
    };
    
    role.permissions.forEach(permission => {
      stats.byScope[permission.scope] = (stats.byScope[permission.scope] || 0) + 1;
      stats.byAction[permission.action] = (stats.byAction[permission.action] || 0) + 1;
      stats.byResource[permission.resource] = (stats.byResource[permission.resource] || 0) + 1;
    });
    
    return stats;
  }, [role?.permissions]);

  if (!role) return null;

  return (
    <Dialog 
      open={open} 
      onClose={onClose} 
      maxWidth="lg" 
      fullWidth
      PaperProps={{
        sx: { height: '80vh' }
      }}
    >
      <DialogTitle>
        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <SecurityIcon color="primary" />
            <Box>
              <Typography variant="h6">
                Permissions du rôle : {role.name}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                {role.description}
              </Typography>
            </Box>
          </Box>
          <Button onClick={onClose} startIcon={<CloseIcon />}>
            Fermer
          </Button>
        </Box>
      </DialogTitle>

      <DialogContent>
        {/* Role Info */}
        <Box sx={{ mb: 3 }}>
          <Box sx={{ display: 'flex', gap: 1, mb: 2 }}>
            {role.is_system_role && (
              <Chip label="Rôle système" color="primary" />
            )}
            <Chip 
              label={`${permissionStats.total} permission(s)`} 
              icon={<PermissionIcon />} 
            />
            <Chip 
              label={`${role.users_count || 0} utilisateur(s)`} 
              variant="outlined" 
            />
          </Box>
        </Box>

        {role.permissions.length === 0 ? (
          <Alert severity="info" icon={<InfoIcon />}>
            <Typography variant="body1">
              Ce rôle n'a aucune permission assignée.
            </Typography>
          </Alert>
        ) : (
          <>
            {/* Statistics */}
            <Paper sx={{ p: 2, mb: 3 }}>
              <Typography variant="h6" gutterBottom>
                Statistiques des permissions
              </Typography>
              <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap' }}>
                <Box>
                  <Typography variant="subtitle2" color="text.secondary">
                    Par portée
                  </Typography>
                  <Box sx={{ display: 'flex', gap: 0.5, flexWrap: 'wrap' }}>
                    {Object.entries(permissionStats.byScope).map(([scope, count]) => (
                      <Chip
                        key={scope}
                        label={`${scope}: ${count}`}
                        size="small"
                        color={getScopeColor(scope as PermissionScope) as any}
                        variant="outlined"
                      />
                    ))}
                  </Box>
                </Box>
                <Box>
                  <Typography variant="subtitle2" color="text.secondary">
                    Par action
                  </Typography>
                  <Box sx={{ display: 'flex', gap: 0.5, flexWrap: 'wrap' }}>
                    {Object.entries(permissionStats.byAction).map(([action, count]) => (
                      <Chip
                        key={action}
                        label={`${action}: ${count}`}
                        size="small"
                        color={getActionColor(action as ActionType) as any}
                        variant="outlined"
                      />
                    ))}
                  </Box>
                </Box>
              </Box>
            </Paper>

            {/* Permissions by Resource */}
            <Typography variant="h6" gutterBottom>
              Permissions par ressource
            </Typography>
            
            {Object.entries(permissionsByResource).map(([resource, resourcePermissions]) => (
              <Accordion key={resource} defaultExpanded={Object.keys(permissionsByResource).length <= 3}>
                <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, width: '100%' }}>
                    <Typography variant="subtitle1" sx={{ textTransform: 'capitalize', flex: 1 }}>
                      {resource}
                    </Typography>
                    <Chip 
                      label={`${resourcePermissions.length} permission(s)`}
                      size="small"
                      color="primary"
                      variant="outlined"
                    />
                  </Box>
                </AccordionSummary>
                <AccordionDetails>
                  <TableContainer>
                    <Table size="small">
                      <TableHead>
                        <TableRow>
                          <TableCell>Permission</TableCell>
                          <TableCell>Action</TableCell>
                          <TableCell>Portée</TableCell>
                          <TableCell>Description</TableCell>
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        {resourcePermissions.map((permission) => (
                          <TableRow key={permission.id}>
                            <TableCell>
                              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                <PermissionIcon fontSize="small" color="action" />
                                <Typography variant="body2">
                                  {permission.name || `${permission.resource}:${permission.action}`}
                                </Typography>
                              </Box>
                            </TableCell>
                            <TableCell>
                              <Chip
                                label={permission.action}
                                size="small"
                                color={getActionColor(permission.action) as any}
                              />
                            </TableCell>
                            <TableCell>
                              <Chip
                                label={permission.scope}
                                size="small"
                                color={getScopeColor(permission.scope) as any}
                                variant="outlined"
                              />
                            </TableCell>
                            <TableCell>
                              <Typography variant="body2" color="text.secondary">
                                {permission.description || 'Aucune description'}
                              </Typography>
                            </TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </TableContainer>
                </AccordionDetails>
              </Accordion>
            ))}

            {/* Detailed Permissions Table */}
            <Box sx={{ mt: 3 }}>
              <Typography variant="h6" gutterBottom>
                Liste complète des permissions
              </Typography>
              <TableContainer component={Paper}>
                <Table>
                  <TableHead>
                    <TableRow>
                      <TableCell>Ressource</TableCell>
                      <TableCell>Action</TableCell>
                      <TableCell>Portée</TableCell>
                      <TableCell>Description</TableCell>
                      <TableCell>Conditions</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {role.permissions.map((permission) => (
                      <TableRow key={permission.id} hover>
                        <TableCell>
                          <Chip
                            label={permission.resource}
                            size="small"
                            variant="outlined"
                            sx={{ textTransform: 'capitalize' }}
                          />
                        </TableCell>
                        <TableCell>
                          <Chip
                            label={permission.action}
                            size="small"
                            color={getActionColor(permission.action) as any}
                          />
                        </TableCell>
                        <TableCell>
                          <Chip
                            label={permission.scope}
                            size="small"
                            color={getScopeColor(permission.scope) as any}
                            variant="outlined"
                          />
                        </TableCell>
                        <TableCell>
                          <Typography variant="body2">
                            {permission.description || 'Aucune description'}
                          </Typography>
                        </TableCell>
                        <TableCell>
                          {permission.conditions && permission.conditions.length > 0 ? (
                            <Box sx={{ display: 'flex', gap: 0.5, flexWrap: 'wrap' }}>
                              {permission.conditions.map((condition, index) => (
                                <Chip
                                  key={index}
                                  label={`${condition.field} ${condition.operator} ${condition.value}`}
                                  size="small"
                                  variant="outlined"
                                  color="secondary"
                                />
                              ))}
                            </Box>
                          ) : (
                            <Typography variant="body2" color="text.secondary">
                              Aucune condition
                            </Typography>
                          )}
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            </Box>
          </>
        )}
      </DialogContent>

      <DialogActions sx={{ p: 3 }}>
        <Button onClick={onClose} variant="outlined">
          Fermer
        </Button>
      </DialogActions>
    </Dialog>
  );
};

export default PermissionDetailsDialog;