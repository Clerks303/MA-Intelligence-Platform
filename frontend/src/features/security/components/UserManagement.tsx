/**
 * User Management Component - M&A Intelligence Platform
 * Sprint 4 - Interface complète de gestion des utilisateurs RBAC
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
  CircularProgress,
  Tooltip,
  Card,
  CardContent,
  Grid,
  Switch,
  FormControlLabel,
  Tabs,
  Tab,
  Divider,
  Menu,
  ListItemIcon,
  ListItemText
} from '@mui/material';
import {
  Add as AddIcon,
  Edit as EditIcon,
  Delete as DeleteIcon,
  Security as SecurityIcon,
  Block as BlockIcon,
  CheckCircle as CheckCircleIcon,
  Warning as WarningIcon,
  MoreVert as MoreVertIcon,
  Search as SearchIcon,
  FilterList as FilterIcon,
  Download as DownloadIcon,
  Refresh as RefreshIcon,
  VpnKey as VpnKeyIcon,
  Shield as ShieldIcon,
  Person as PersonIcon,
  AdminPanelSettings as AdminIcon,
  Group as GroupIcon
} from '@mui/icons-material';
import { DataGrid, GridColDef, GridToolbar } from '@mui/x-data-grid';
import { useUsers } from '../hooks/useRBAC';
import { User, CreateUserRequest, UpdateUserRequest, SecurityLevel } from '../types';
import { UserFormDialog } from './dialogs/UserFormDialog';
import { UserDetailsDialog } from './dialogs/UserDetailsDialog';
import { ConfirmDialog } from './dialogs/ConfirmDialog';

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

export const UserManagement: React.FC = () => {
  // State management
  const [tabValue, setTabValue] = useState(0);
  const [searchTerm, setSearchTerm] = useState('');
  const [roleFilter, setRoleFilter] = useState<string[]>([]);
  const [statusFilter, setStatusFilter] = useState<boolean | undefined>(undefined);
  const [mfaFilter, setMfaFilter] = useState<boolean | undefined>(undefined);
  const [selectedUser, setSelectedUser] = useState<User | null>(null);
  const [dialogOpen, setDialogOpen] = useState(false);
  const [detailsOpen, setDetailsOpen] = useState(false);
  const [confirmOpen, setConfirmOpen] = useState(false);
  const [actionType, setActionType] = useState<'create' | 'edit' | 'delete' | 'toggle'>('create');
  const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null);
  
  // Hooks
  const {
    users,
    pagination,
    userMetrics,
    filters,
    updateFilters,
    clearFilters,
    page,
    perPage,
    goToPage,
    setPerPage,
    isLoading,
    isCreating,
    isUpdating,
    isDeleting,
    isTogglingStatus,
    createUser,
    updateUser,
    deleteUser,
    toggleUserStatus,
    resetUserPassword,
    revokeUserSessions,
    refetch
  } = useUsers();

  // Apply local filters
  React.useEffect(() => {
    updateFilters({
      search: searchTerm || undefined,
      roles: roleFilter.length > 0 ? roleFilter : undefined,
      is_active: statusFilter,
      mfa_enabled: mfaFilter,
    });
  }, [searchTerm, roleFilter, statusFilter, mfaFilter, updateFilters]);

  // DataGrid columns
  const columns: GridColDef[] = [
    {
      field: 'avatar',
      headerName: '',
      width: 60,
      renderCell: (params) => (
        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          <PersonIcon color="action" />
        </Box>
      ),
      sortable: false,
      filterable: false,
    },
    {
      field: 'username',
      headerName: 'Utilisateur',
      width: 180,
      renderCell: (params) => (
        <Box>
          <Typography variant="body2" fontWeight="medium">
            {params.row.username}
          </Typography>
          <Typography variant="caption" color="text.secondary">
            {params.row.email}
          </Typography>
        </Box>
      ),
    },
    {
      field: 'roles',
      headerName: 'Rôles',
      width: 200,
      renderCell: (params) => (
        <Box sx={{ display: 'flex', gap: 0.5, flexWrap: 'wrap' }}>
          {params.row.roles?.map((role: any) => (
            <Chip
              key={role.id}
              label={role.name}
              size="small"
              color={role.is_system_role ? "primary" : "default"}
              icon={role.is_system_role ? <AdminIcon /> : <GroupIcon />}
            />
          )) || (
            <Chip label="Aucun rôle" size="small" variant="outlined" />
          )}
        </Box>
      ),
    },
    {
      field: 'is_active',
      headerName: 'Statut',
      width: 120,
      renderCell: (params) => (
        <Chip
          label={params.row.is_active ? 'Actif' : 'Inactif'}
          color={params.row.is_active ? 'success' : 'error'}
          size="small"
          icon={params.row.is_active ? <CheckCircleIcon /> : <BlockIcon />}
        />
      ),
    },
    {
      field: 'mfa_enabled',
      headerName: 'MFA',
      width: 100,
      renderCell: (params) => (
        <Chip
          label={params.row.mfa_enabled ? 'Activé' : 'Désactivé'}
          color={params.row.mfa_enabled ? 'success' : 'warning'}
          size="small"
          icon={<SecurityIcon />}
        />
      ),
    },
    {
      field: 'security_level',
      headerName: 'Niveau de sécurité',
      width: 140,
      renderCell: (params) => {
        const level = params.row.security_level || 'MEDIUM';
        const colors = {
          LOW: 'info',
          MEDIUM: 'warning',
          HIGH: 'error',
          CRITICAL: 'error'
        } as const;
        return (
          <Chip
            label={level}
            color={colors[level as SecurityLevel]}
            size="small"
            icon={<ShieldIcon />}
          />
        );
      },
    },
    {
      field: 'last_login',
      headerName: 'Dernière connexion',
      width: 160,
      renderCell: (params) => (
        <Typography variant="body2" color="text.secondary">
          {params.row.last_login 
            ? new Date(params.row.last_login).toLocaleDateString('fr-FR')
            : 'Jamais'
          }
        </Typography>
      ),
    },
    {
      field: 'actions',
      headerName: 'Actions',
      width: 120,
      renderCell: (params) => (
        <Box>
          <Tooltip title="Voir détails">
            <IconButton
              size="small"
              onClick={() => handleViewDetails(params.row)}
            >
              <PersonIcon />
            </IconButton>
          </Tooltip>
          <Tooltip title="Plus d'actions">
            <IconButton
              size="small"
              onClick={(e) => handleMenuOpen(e, params.row)}
            >
              <MoreVertIcon />
            </IconButton>
          </Tooltip>
        </Box>
      ),
      sortable: false,
      filterable: false,
    },
  ];

  // Event handlers
  const handleViewDetails = (user: User) => {
    setSelectedUser(user);
    setDetailsOpen(true);
  };

  const handleMenuOpen = (event: React.MouseEvent<HTMLElement>, user: User) => {
    setAnchorEl(event.currentTarget);
    setSelectedUser(user);
  };

  const handleMenuClose = () => {
    setAnchorEl(null);
    setSelectedUser(null);
  };

  const handleCreateUser = () => {
    setActionType('create');
    setSelectedUser(null);
    setDialogOpen(true);
  };

  const handleEditUser = (user: User) => {
    setActionType('edit');
    setSelectedUser(user);
    setDialogOpen(true);
    handleMenuClose();
  };

  const handleDeleteUser = (user: User) => {
    setActionType('delete');
    setSelectedUser(user);
    setConfirmOpen(true);
    handleMenuClose();
  };

  const handleToggleStatus = (user: User) => {
    setActionType('toggle');
    setSelectedUser(user);
    setConfirmOpen(true);
    handleMenuClose();
  };

  const handleFormSubmit = (data: CreateUserRequest | UpdateUserRequest) => {
    if (actionType === 'create') {
      createUser(data as CreateUserRequest);
    } else if (actionType === 'edit' && selectedUser) {
      updateUser(selectedUser.id, data as UpdateUserRequest);
    }
    setDialogOpen(false);
  };

  const handleConfirmAction = () => {
    if (!selectedUser) return;
    
    switch (actionType) {
      case 'delete':
        deleteUser(selectedUser.id);
        break;
      case 'toggle':
        toggleUserStatus(selectedUser.id);
        break;
    }
    setConfirmOpen(false);
  };

  const handleResetPassword = (user: User) => {
    resetUserPassword(user.id);
    handleMenuClose();
  };

  const handleRevokeSessions = (user: User) => {
    revokeUserSessions(user.id);
    handleMenuClose();
  };

  // Filter summary
  const activeFiltersCount = useMemo(() => {
    let count = 0;
    if (searchTerm) count++;
    if (roleFilter.length > 0) count++;
    if (statusFilter !== undefined) count++;
    if (mfaFilter !== undefined) count++;
    return count;
  }, [searchTerm, roleFilter, statusFilter, mfaFilter]);

  return (
    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      {/* Header */}
      <Paper sx={{ p: 3, mb: 3 }}>
        <Box sx={{ display: 'flex', justifyContent: 'between', alignItems: 'center', mb: 2 }}>
          <Box>
            <Typography variant="h4" component="h1" gutterBottom>
              Gestion des utilisateurs
            </Typography>
            <Typography variant="body1" color="text.secondary">
              Administration des comptes utilisateurs et des permissions
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
              onClick={handleCreateUser}
            >
              Nouvel utilisateur
            </Button>
          </Box>
        </Box>

        {/* Metrics Cards */}
        <Grid container spacing={2}>
          <Grid item xs={12} md={2.4}>
            <Card>
              <CardContent sx={{ textAlign: 'center', py: 2 }}>
                <Typography variant="h4" color="primary">
                  {userMetrics.total}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Total utilisateurs
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} md={2.4}>
            <Card>
              <CardContent sx={{ textAlign: 'center', py: 2 }}>
                <Typography variant="h4" color="success.main">
                  {userMetrics.active}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Actifs
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} md={2.4}>
            <Card>
              <CardContent sx={{ textAlign: 'center', py: 2 }}>
                <Typography variant="h4" color="warning.main">
                  {userMetrics.withMFA}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Avec MFA
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} md={2.4}>
            <Card>
              <CardContent sx={{ textAlign: 'center', py: 2 }}>
                <Typography variant="h4" color="error.main">
                  {userMetrics.admins}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Administrateurs
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} md={2.4}>
            <Card>
              <CardContent sx={{ textAlign: 'center', py: 2 }}>
                <Typography variant="h4" color="info.main">
                  {userMetrics.recentLogins}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Connexions 24h
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
            placeholder="Rechercher utilisateur..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            InputProps={{
              startAdornment: <SearchIcon color="action" sx={{ mr: 1 }} />,
            }}
            sx={{ minWidth: 250 }}
          />
          
          <FormControl size="small" sx={{ minWidth: 120 }}>
            <InputLabel>Statut</InputLabel>
            <Select
              value={statusFilter === undefined ? '' : statusFilter.toString()}
              onChange={(e) => {
                const value = e.target.value;
                setStatusFilter(value === '' ? undefined : value === 'true');
              }}
            >
              <MenuItem value="">Tous</MenuItem>
              <MenuItem value="true">Actifs</MenuItem>
              <MenuItem value="false">Inactifs</MenuItem>
            </Select>
          </FormControl>

          <FormControl size="small" sx={{ minWidth: 120 }}>
            <InputLabel>MFA</InputLabel>
            <Select
              value={mfaFilter === undefined ? '' : mfaFilter.toString()}
              onChange={(e) => {
                const value = e.target.value;
                setMfaFilter(value === '' ? undefined : value === 'true');
              }}
            >
              <MenuItem value="">Tous</MenuItem>
              <MenuItem value="true">Activé</MenuItem>
              <MenuItem value="false">Désactivé</MenuItem>
            </Select>
          </FormControl>

          {activeFiltersCount > 0 && (
            <Button
              size="small"
              startIcon={<FilterIcon />}
              onClick={() => {
                setSearchTerm('');
                setRoleFilter([]);
                setStatusFilter(undefined);
                setMfaFilter(undefined);
                clearFilters();
              }}
            >
              Effacer filtres ({activeFiltersCount})
            </Button>
          )}
        </Box>
      </Paper>

      {/* Data Grid */}
      <Paper sx={{ flex: 1, overflow: 'hidden' }}>
        <DataGrid
          rows={users}
          columns={columns}
          loading={isLoading}
          pageSize={perPage}
          onPageSizeChange={setPerPage}
          rowsPerPageOptions={[10, 20, 50, 100]}
          page={page - 1}
          onPageChange={(newPage) => goToPage(newPage + 1)}
          rowCount={pagination?.total || 0}
          paginationMode="server"
          components={{ Toolbar: GridToolbar }}
          componentsProps={{
            toolbar: {
              showQuickFilter: true,
              quickFilterProps: { debounceMs: 500 },
            },
          }}
          disableSelectionOnClick
          sx={{
            '& .MuiDataGrid-row:hover': {
              backgroundColor: 'action.hover',
            },
          }}
        />
      </Paper>

      {/* Actions Menu */}
      <Menu
        anchorEl={anchorEl}
        open={Boolean(anchorEl)}
        onClose={handleMenuClose}
      >
        <MenuItem onClick={() => selectedUser && handleEditUser(selectedUser)}>
          <ListItemIcon><EditIcon /></ListItemIcon>
          <ListItemText>Modifier</ListItemText>
        </MenuItem>
        <MenuItem onClick={() => selectedUser && handleToggleStatus(selectedUser)}>
          <ListItemIcon>
            {selectedUser?.is_active ? <BlockIcon /> : <CheckCircleIcon />}
          </ListItemIcon>
          <ListItemText>
            {selectedUser?.is_active ? 'Désactiver' : 'Activer'}
          </ListItemText>
        </MenuItem>
        <MenuItem onClick={() => selectedUser && handleResetPassword(selectedUser)}>
          <ListItemIcon><VpnKeyIcon /></ListItemIcon>
          <ListItemText>Réinitialiser mot de passe</ListItemText>
        </MenuItem>
        <MenuItem onClick={() => selectedUser && handleRevokeSessions(selectedUser)}>
          <ListItemIcon><SecurityIcon /></ListItemIcon>
          <ListItemText>Révoquer sessions</ListItemText>
        </MenuItem>
        <Divider />
        <MenuItem 
          onClick={() => selectedUser && handleDeleteUser(selectedUser)}
          sx={{ color: 'error.main' }}
        >
          <ListItemIcon><DeleteIcon color="error" /></ListItemIcon>
          <ListItemText>Supprimer</ListItemText>
        </MenuItem>
      </Menu>

      {/* Dialogs */}
      <UserFormDialog
        open={dialogOpen}
        onClose={() => setDialogOpen(false)}
        onSubmit={handleFormSubmit}
        user={actionType === 'edit' ? selectedUser : undefined}
        loading={isCreating || isUpdating}
      />

      <UserDetailsDialog
        open={detailsOpen}
        onClose={() => setDetailsOpen(false)}
        user={selectedUser}
      />

      <ConfirmDialog
        open={confirmOpen}
        onClose={() => setConfirmOpen(false)}
        onConfirm={handleConfirmAction}
        title={
          actionType === 'delete' 
            ? 'Supprimer l\'utilisateur' 
            : actionType === 'toggle'
            ? (selectedUser?.is_active ? 'Désactiver l\'utilisateur' : 'Activer l\'utilisateur')
            : 'Confirmer l\'action'
        }
        message={
          actionType === 'delete'
            ? `Êtes-vous sûr de vouloir supprimer l'utilisateur ${selectedUser?.username} ? Cette action est irréversible.`
            : actionType === 'toggle'
            ? `Êtes-vous sûr de vouloir ${selectedUser?.is_active ? 'désactiver' : 'activer'} l'utilisateur ${selectedUser?.username} ?`
            : 'Voulez-vous vraiment effectuer cette action ?'
        }
        severity={actionType === 'delete' ? 'error' : 'warning'}
        loading={isDeleting || isTogglingStatus}
      />
    </Box>
  );
};

export default UserManagement;