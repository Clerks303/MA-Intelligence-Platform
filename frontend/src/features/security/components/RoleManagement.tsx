/**
 * Role Management Component - M&A Intelligence Platform
 * Sprint 4 - Interface complète de gestion des rôles et permissions RBAC
 */

import React, { useState, useMemo } from 'react';
import {
  Box,
  Paper,
  Typography,
  Button,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Chip,
  IconButton,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Alert,
  Card,
  CardContent,
  CardActions,
  Grid,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Divider,
  Collapse,
  Tooltip,
  Switch,
  FormControlLabel,
  Menu,
  ListItemButton
} from '@mui/material';
import {
  Add as AddIcon,
  Edit as EditIcon,
  Delete as DeleteIcon,
  Security as SecurityIcon,
  AdminPanelSettings as AdminIcon,
  Group as GroupIcon,
  ExpandMore as ExpandMoreIcon,
  ExpandLess as ExpandLessIcon,
  VpnKey as PermissionIcon,
  Person as PersonIcon,
  Refresh as RefreshIcon,
  MoreVert as MoreVertIcon,
  Assignment as AssignmentIcon,
  FilterList as FilterIcon,
  Search as SearchIcon
} from '@mui/icons-material';
import { useRoles, usePermissions } from '../hooks/useRBAC';
import { Role, Permission, CreateRoleRequest, UpdateRoleRequest } from '../types';
import { RoleFormDialog } from './dialogs/RoleFormDialog';
import { PermissionDetailsDialog } from './dialogs/PermissionDetailsDialog';
import { ConfirmDialog } from './dialogs/ConfirmDialog';

interface RoleCardProps {
  role: Role;
  onEdit: (role: Role) => void;
  onDelete: (role: Role) => void;
  onViewPermissions: (role: Role) => void;
}

const RoleCard: React.FC<RoleCardProps> = ({ role, onEdit, onDelete, onViewPermissions }) => {
  const [expanded, setExpanded] = useState(false);
  const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null);

  const handleMenuOpen = (event: React.MouseEvent<HTMLElement>) => {
    setAnchorEl(event.currentTarget);
  };

  const handleMenuClose = () => {
    setAnchorEl(null);
  };

  return (
    <Card 
      sx={{ 
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
        border: role.is_system_role ? '2px solid' : '1px solid',
        borderColor: role.is_system_role ? 'primary.main' : 'divider'
      }}
    >
      <CardContent sx={{ flex: 1 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 2 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, flex: 1 }}>
            {role.is_system_role ? (
              <AdminIcon color="primary" />
            ) : (
              <GroupIcon color="action" />
            )}
            <Box>
              <Typography variant="h6" component="div">
                {role.name}
              </Typography>
              {role.is_system_role && (
                <Chip label="Rôle système" size="small" color="primary" />
              )}
            </Box>
          </Box>
          <IconButton size="small" onClick={handleMenuOpen}>
            <MoreVertIcon />
          </IconButton>
        </Box>
        
        <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
          {role.description}
        </Typography>

        <Box sx={{ display: 'flex', gap: 1, mb: 2, flexWrap: 'wrap' }}>
          <Chip
            icon={<PersonIcon />}
            label={`${role.users_count || 0} utilisateur(s)`}
            size="small"
            variant="outlined"
          />
          <Chip
            icon={<PermissionIcon />}
            label={`${role.permissions.length} permission(s)`}
            size="small"
            variant="outlined"
            onClick={() => setExpanded(!expanded)}
            clickable
          />
        </Box>

        <Collapse in={expanded} timeout="auto" unmountOnExit>
          <Divider sx={{ mb: 1 }} />
          <Typography variant="subtitle2" gutterBottom>
            Permissions :
          </Typography>
          <Box sx={{ maxHeight: 150, overflow: 'auto' }}>
            {role.permissions.length > 0 ? (
              <List dense>
                {role.permissions.map((permission) => (
                  <ListItem key={permission.id} sx={{ py: 0.5 }}>
                    <ListItemIcon sx={{ minWidth: 30 }}>
                      <PermissionIcon fontSize="small" />
                    </ListItemIcon>
                    <ListItemText
                      primary={
                        <Typography variant="body2">
                          {`${permission.resource}:${permission.action}`}
                        </Typography>
                      }
                      secondary={
                        <Typography variant="caption" color="text.secondary">
                          {permission.scope}
                        </Typography>
                      }
                    />
                  </ListItem>
                ))}
              </List>
            ) : (
              <Typography variant="body2" color="text.secondary" sx={{ p: 1 }}>
                Aucune permission assignée
              </Typography>
            )}
          </Box>
        </Collapse>
      </CardContent>

      <CardActions>
        <Button
          size="small"
          startIcon={<PermissionIcon />}
          onClick={() => onViewPermissions(role)}
        >
          Permissions
        </Button>
        <Button
          size="small"
          startIcon={expanded ? <ExpandLessIcon /> : <ExpandMoreIcon />}
          onClick={() => setExpanded(!expanded)}
        >
          {expanded ? 'Reduire' : 'Détails'}
        </Button>
      </CardActions>

      <Menu
        anchorEl={anchorEl}
        open={Boolean(anchorEl)}
        onClose={handleMenuClose}
      >
        <MenuItem onClick={() => { onEdit(role); handleMenuClose(); }}>
          <ListItemIcon><EditIcon /></ListItemIcon>
          <ListItemText>Modifier</ListItemText>
        </MenuItem>
        {!role.is_system_role && (
          <MenuItem onClick={() => { onDelete(role); handleMenuClose(); }}>
            <ListItemIcon><DeleteIcon color="error" /></ListItemIcon>
            <ListItemText>Supprimer</ListItemText>
          </MenuItem>
        )}
      </Menu>
    </Card>
  );
};

export const RoleManagement: React.FC = () => {
  // State management
  const [searchTerm, setSearchTerm] = useState('');
  const [systemRoleFilter, setSystemRoleFilter] = useState<boolean | undefined>(undefined);
  const [selectedRole, setSelectedRole] = useState<Role | null>(null);
  const [dialogOpen, setDialogOpen] = useState(false);
  const [permissionsOpen, setPermissionsOpen] = useState(false);
  const [confirmOpen, setConfirmOpen] = useState(false);
  const [actionType, setActionType] = useState<'create' | 'edit' | 'delete'>('create');
  
  // Hooks
  const {
    roles,
    roleMetrics,
    filters,
    updateFilters,
    clearFilters,
    isLoading,
    isCreating,
    isUpdating,
    isDeleting,
    createRole,
    updateRole,
    deleteRole,
    refetch
  } = useRoles();

  const { permissions } = usePermissions();

  // Apply local filters
  React.useEffect(() => {
    updateFilters({
      search: searchTerm || undefined,
      is_system_role: systemRoleFilter,
    });
  }, [searchTerm, systemRoleFilter, updateFilters]);

  // Filter roles based on search
  const filteredRoles = useMemo(() => {
    return roles.filter(role => {
      const matchesSearch = !searchTerm || 
        role.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
        role.description.toLowerCase().includes(searchTerm.toLowerCase());
      
      const matchesFilter = systemRoleFilter === undefined || 
        role.is_system_role === systemRoleFilter;
      
      return matchesSearch && matchesFilter;
    });
  }, [roles, searchTerm, systemRoleFilter]);

  // Event handlers
  const handleCreateRole = () => {
    setActionType('create');
    setSelectedRole(null);
    setDialogOpen(true);
  };

  const handleEditRole = (role: Role) => {
    setActionType('edit');
    setSelectedRole(role);
    setDialogOpen(true);
  };

  const handleDeleteRole = (role: Role) => {
    setActionType('delete');
    setSelectedRole(role);
    setConfirmOpen(true);
  };

  const handleViewPermissions = (role: Role) => {
    setSelectedRole(role);
    setPermissionsOpen(true);
  };

  const handleFormSubmit = (data: CreateRoleRequest | UpdateRoleRequest) => {
    if (actionType === 'create') {
      createRole(data as CreateRoleRequest);
    } else if (actionType === 'edit' && selectedRole) {
      updateRole(selectedRole.id, data as UpdateRoleRequest);
    }
    setDialogOpen(false);
  };

  const handleConfirmDelete = () => {
    if (selectedRole) {
      deleteRole(selectedRole.id);
    }
    setConfirmOpen(false);
  };

  const activeFiltersCount = useMemo(() => {
    let count = 0;
    if (searchTerm) count++;
    if (systemRoleFilter !== undefined) count++;
    return count;
  }, [searchTerm, systemRoleFilter]);

  return (
    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      {/* Header */}
      <Paper sx={{ p: 3, mb: 3 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
          <Box>
            <Typography variant="h4" component="h1" gutterBottom>
              Gestion des rôles
            </Typography>
            <Typography variant="body1" color="text.secondary">
              Administration des rôles et permissions RBAC
            </Typography>
          </Box>
          <Box sx={{ display: 'flex', gap: 1 }}>
            <Button
              variant="outlined"
              startIcon={<RefreshIcon />}
              onClick={refetch}
              disabled={isLoading}
            >
              Actualiser
            </Button>
            <Button
              variant="contained"
              startIcon={<AddIcon />}
              onClick={handleCreateRole}
            >
              Nouveau rôle
            </Button>
          </Box>
        </Box>

        {/* Metrics Cards */}
        <Grid container spacing={2}>
          <Grid item xs={12} md={3}>
            <Card>
              <CardContent sx={{ textAlign: 'center', py: 2 }}>
                <Typography variant="h4" color="primary">
                  {roleMetrics.total}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Total rôles
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} md={3}>
            <Card>
              <CardContent sx={{ textAlign: 'center', py: 2 }}>
                <Typography variant="h4" color="warning.main">
                  {roleMetrics.system}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Rôles système
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} md={3}>
            <Card>
              <CardContent sx={{ textAlign: 'center', py: 2 }}>
                <Typography variant="h4" color="success.main">
                  {roleMetrics.custom}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Rôles personnalisés
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} md={3}>
            <Card>
              <CardContent sx={{ textAlign: 'center', py: 2 }}>
                <Typography variant="h4" color="info.main">
                  {roleMetrics.withUsers}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Avec utilisateurs
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      </Paper>

      {/* Filters */}
      <Paper sx={{ p: 2, mb: 2 }}>
        <Box sx={{ display: 'flex', gap: 2, alignItems: 'center', flexWrap: 'wrap' }}>
          <TextField
            size="small"
            placeholder="Rechercher rôle..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            InputProps={{
              startAdornment: <SearchIcon color="action" sx={{ mr: 1 }} />,
            }}
            sx={{ minWidth: 250 }}
          />
          
          <FormControl size="small" sx={{ minWidth: 150 }}>
            <InputLabel>Type de rôle</InputLabel>
            <Select
              value={systemRoleFilter === undefined ? '' : systemRoleFilter.toString()}
              onChange={(e) => {
                const value = e.target.value;
                setSystemRoleFilter(value === '' ? undefined : value === 'true');
              }}
            >
              <MenuItem value="">Tous</MenuItem>
              <MenuItem value="true">Rôles système</MenuItem>
              <MenuItem value="false">Rôles personnalisés</MenuItem>
            </Select>
          </FormControl>

          {activeFiltersCount > 0 && (
            <Button
              size="small"
              startIcon={<FilterIcon />}
              onClick={() => {
                setSearchTerm('');
                setSystemRoleFilter(undefined);
                clearFilters();
              }}
            >
              Effacer filtres ({activeFiltersCount})
            </Button>
          )}
        </Box>
      </Paper>

      {/* Roles Grid */}
      <Box sx={{ flex: 1, overflow: 'auto' }}>
        {isLoading ? (
          <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: 200 }}>
            <Typography>Chargement des rôles...</Typography>
          </Box>
        ) : filteredRoles.length > 0 ? (
          <Grid container spacing={2}>
            {filteredRoles.map((role) => (
              <Grid item xs={12} md={6} lg={4} key={role.id}>
                <RoleCard
                  role={role}
                  onEdit={handleEditRole}
                  onDelete={handleDeleteRole}
                  onViewPermissions={handleViewPermissions}
                />
              </Grid>
            ))}
          </Grid>
        ) : (
          <Paper sx={{ p: 4, textAlign: 'center' }}>
            <GroupIcon sx={{ fontSize: 48, color: 'text.secondary', mb: 2 }} />
            <Typography variant="h6" color="text.secondary" gutterBottom>
              Aucun rôle trouvé
            </Typography>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
              {searchTerm || systemRoleFilter !== undefined
                ? 'Aucun rôle ne correspond aux critères de recherche.'
                : 'Aucun rôle n\'a été créé.'
              }
            </Typography>
            <Button
              variant="contained"
              startIcon={<AddIcon />}
              onClick={handleCreateRole}
            >
              Créer le premier rôle
            </Button>
          </Paper>
        )}
      </Box>

      {/* Dialogs */}
      <RoleFormDialog
        open={dialogOpen}
        onClose={() => setDialogOpen(false)}
        onSubmit={handleFormSubmit}
        role={actionType === 'edit' ? selectedRole : undefined}
        permissions={permissions}
        loading={isCreating || isUpdating}
      />

      <PermissionDetailsDialog
        open={permissionsOpen}
        onClose={() => setPermissionsOpen(false)}
        role={selectedRole}
      />

      <ConfirmDialog
        open={confirmOpen}
        onClose={() => setConfirmOpen(false)}
        onConfirm={handleConfirmDelete}
        title="Supprimer le rôle"
        message={`Êtes-vous sûr de vouloir supprimer le rôle "${selectedRole?.name}" ? Cette action supprimera également toutes les assignations de ce rôle aux utilisateurs.`}
        severity="error"
        loading={isDeleting}
      />
    </Box>
  );
};

export default RoleManagement;