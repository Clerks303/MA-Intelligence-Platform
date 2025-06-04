/**
 * Role Form Dialog - M&A Intelligence Platform
 * Sprint 4 - Formulaire de création/modification de rôle
 */

import React, { useState, useEffect, useMemo } from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Chip,
  Box,
  Typography,
  Grid,
  Alert,
  Divider,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Checkbox,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  FormControlLabel,
  OutlinedInput,
  Autocomplete
} from '@mui/material';
import {
  Security as SecurityIcon,
  VpnKey as PermissionIcon,
  ExpandMore as ExpandMoreIcon,
  Group as GroupIcon,
  AdminPanelSettings as AdminIcon,
  Person as PersonIcon
} from '@mui/icons-material';
import { Role, Permission, CreateRoleRequest, UpdateRoleRequest, ResourceType, ActionType, PermissionScope } from '../../types';

interface RoleFormDialogProps {
  open: boolean;
  onClose: () => void;
  onSubmit: (data: CreateRoleRequest | UpdateRoleRequest) => void;
  role?: Role | null;
  permissions: Permission[];
  loading?: boolean;
}

interface FormData {
  name: string;
  description: string;
  permissions: string[];
  parent_role_id?: string;
  metadata: {
    color?: string;
    icon?: string;
    priority?: number;
  };
}

const initialFormData: FormData = {
  name: '',
  description: '',
  permissions: [],
  parent_role_id: undefined,
  metadata: {
    color: '#1976d2',
    icon: 'group',
    priority: 0,
  },
};

const ROLE_COLORS = [
  '#1976d2', // Blue
  '#388e3c', // Green
  '#f57c00', // Orange
  '#d32f2f', // Red
  '#7b1fa2', // Purple
  '#303f9f', // Indigo
  '#1976d2', // Light Blue
  '#00796b', // Teal
];

const ROLE_ICONS = [
  { value: 'group', label: 'Groupe', icon: <GroupIcon /> },
  { value: 'admin', label: 'Admin', icon: <AdminIcon /> },
  { value: 'security', label: 'Sécurité', icon: <SecurityIcon /> },
  { value: 'person', label: 'Personne', icon: <PersonIcon /> },
];

export const RoleFormDialog: React.FC<RoleFormDialogProps> = ({
  open,
  onClose,
  onSubmit,
  role,
  permissions,
  loading = false
}) => {
  const [formData, setFormData] = useState<FormData>(initialFormData);
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [expandedPanel, setExpandedPanel] = useState<string | false>('basic');

  const isEdit = !!role;

  // Load role data for editing
  useEffect(() => {
    if (role && open) {
      setFormData({
        name: role.name,
        description: role.description,
        permissions: role.permissions.map(p => p.id),
        parent_role_id: role.parent_role?.id,
        metadata: {
          color: role.color || '#1976d2',
          icon: role.icon || 'group',
          priority: role.priority || 0,
        },
      });
    } else if (!role && open) {
      setFormData(initialFormData);
    }
  }, [role, open]);

  // Reset form when dialog closes
  useEffect(() => {
    if (!open) {
      setFormData(initialFormData);
      setErrors({});
      setExpandedPanel('basic');
    }
  }, [open]);

  // Group permissions by resource
  const permissionsByResource = useMemo(() => {
    return permissions.reduce((acc, permission) => {
      if (!acc[permission.resource]) {
        acc[permission.resource] = [];
      }
      acc[permission.resource].push(permission);
      return acc;
    }, {} as Record<ResourceType, Permission[]>);
  }, [permissions]);

  const selectedPermissions = permissions.filter(p => formData.permissions.includes(p.id));

  const handleChange = (field: string, value: any) => {
    if (field.includes('.')) {
      const [parent, child] = field.split('.');
      setFormData(prev => ({
        ...prev,
        [parent]: {
          ...prev[parent as keyof FormData],
          [child]: value
        }
      }));
    } else {
      setFormData(prev => ({ ...prev, [field]: value }));
    }

    // Clear error when field is modified
    if (errors[field]) {
      setErrors(prev => ({ ...prev, [field]: '' }));
    }
  };

  const handlePermissionToggle = (permissionId: string) => {
    setFormData(prev => ({
      ...prev,
      permissions: prev.permissions.includes(permissionId)
        ? prev.permissions.filter(id => id !== permissionId)
        : [...prev.permissions, permissionId]
    }));
  };

  const handleResourceToggle = (resource: ResourceType, isChecked: boolean) => {
    const resourcePermissions = permissionsByResource[resource] || [];
    const permissionIds = resourcePermissions.map(p => p.id);
    
    setFormData(prev => {
      if (isChecked) {
        // Add all permissions for this resource
        const newPermissions = [...new Set([...prev.permissions, ...permissionIds])];
        return { ...prev, permissions: newPermissions };
      } else {
        // Remove all permissions for this resource
        const newPermissions = prev.permissions.filter(id => !permissionIds.includes(id));
        return { ...prev, permissions: newPermissions };
      }
    });
  };

  const validateForm = (): boolean => {
    const newErrors: Record<string, string> = {};

    if (!formData.name.trim()) {
      newErrors.name = 'Le nom du rôle est requis';
    } else if (formData.name.length < 3) {
      newErrors.name = 'Le nom doit contenir au moins 3 caractères';
    }

    if (!formData.description.trim()) {
      newErrors.description = 'La description est requise';
    }

    if (formData.permissions.length === 0) {
      newErrors.permissions = 'Au moins une permission doit être sélectionnée';
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = () => {
    if (!validateForm()) return;

    const submitData = {
      name: formData.name,
      description: formData.description,
      permissions: formData.permissions,
      parent_role_id: formData.parent_role_id || undefined,
      metadata: formData.metadata,
    };

    onSubmit(submitData);
  };

  const isResourceFullySelected = (resource: ResourceType) => {
    const resourcePermissions = permissionsByResource[resource] || [];
    return resourcePermissions.length > 0 && 
           resourcePermissions.every(p => formData.permissions.includes(p.id));
  };

  const isResourcePartiallySelected = (resource: ResourceType) => {
    const resourcePermissions = permissionsByResource[resource] || [];
    return resourcePermissions.some(p => formData.permissions.includes(p.id)) &&
           !resourcePermissions.every(p => formData.permissions.includes(p.id));
  };

  return (
    <Dialog open={open} onClose={onClose} maxWidth="md" fullWidth>
      <DialogTitle>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <SecurityIcon />
          {isEdit ? 'Modifier le rôle' : 'Nouveau rôle'}
        </Box>
      </DialogTitle>

      <DialogContent>
        <Box sx={{ mt: 2 }}>
          {/* Basic Information */}
          <Accordion 
            expanded={expandedPanel === 'basic'} 
            onChange={(_, isExpanded) => setExpandedPanel(isExpanded ? 'basic' : false)}
          >
            <AccordionSummary expandIcon={<ExpandMoreIcon />}>
              <Typography variant="h6">Informations de base</Typography>
            </AccordionSummary>
            <AccordionDetails>
              <Grid container spacing={2}>
                <Grid item xs={12} md={6}>
                  <TextField
                    fullWidth
                    label="Nom du rôle *"
                    value={formData.name}
                    onChange={(e) => handleChange('name', e.target.value)}
                    error={!!errors.name}
                    helperText={errors.name}
                    disabled={loading}
                  />
                </Grid>
                <Grid item xs={12} md={6}>
                  <FormControl fullWidth>
                    <InputLabel>Rôle parent</InputLabel>
                    <Select
                      value={formData.parent_role_id || ''}
                      onChange={(e) => handleChange('parent_role_id', e.target.value || undefined)}
                      disabled={loading}
                    >
                      <MenuItem value="">Aucun parent</MenuItem>
                      {/* TODO: Add available parent roles */}
                    </Select>
                  </FormControl>
                </Grid>
                <Grid item xs={12}>
                  <TextField
                    fullWidth
                    label="Description *"
                    multiline
                    rows={3}
                    value={formData.description}
                    onChange={(e) => handleChange('description', e.target.value)}
                    error={!!errors.description}
                    helperText={errors.description}
                    disabled={loading}
                  />
                </Grid>
              </Grid>
            </AccordionDetails>
          </Accordion>

          {/* Permissions */}
          <Accordion 
            expanded={expandedPanel === 'permissions'} 
            onChange={(_, isExpanded) => setExpandedPanel(isExpanded ? 'permissions' : false)}
          >
            <AccordionSummary expandIcon={<ExpandMoreIcon />}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, flex: 1 }}>
                <Typography variant="h6">Permissions</Typography>
                <Chip 
                  label={`${selectedPermissions.length} sélectionnée(s)`}
                  size="small"
                  color={selectedPermissions.length > 0 ? 'primary' : 'default'}
                />
              </Box>
            </AccordionSummary>
            <AccordionDetails>
              {errors.permissions && (
                <Alert severity="error" sx={{ mb: 2 }}>
                  {errors.permissions}
                </Alert>
              )}
              
              <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                Sélectionnez les permissions à attribuer à ce rôle. Les permissions sont groupées par ressource.
              </Typography>

              {Object.entries(permissionsByResource).map(([resource, resourcePermissions]) => (
                <Accordion key={resource} sx={{ mb: 1 }}>
                  <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                    <FormControlLabel
                      control={
                        <Checkbox
                          checked={isResourceFullySelected(resource as ResourceType)}
                          indeterminate={isResourcePartiallySelected(resource as ResourceType)}
                          onChange={(e) => handleResourceToggle(resource as ResourceType, e.target.checked)}
                          onClick={(e) => e.stopPropagation()}
                        />
                      }
                      label={
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                          <Typography variant="subtitle1" sx={{ textTransform: 'capitalize' }}>
                            {resource}
                          </Typography>
                          <Chip 
                            label={`${resourcePermissions.filter(p => formData.permissions.includes(p.id)).length}/${resourcePermissions.length}`}
                            size="small"
                            variant="outlined"
                          />
                        </Box>
                      }
                      onClick={(e) => e.stopPropagation()}
                    />
                  </AccordionSummary>
                  <AccordionDetails>
                    <List dense>
                      {resourcePermissions.map((permission) => (
                        <ListItem key={permission.id} sx={{ py: 0 }}>
                          <ListItemIcon sx={{ minWidth: 40 }}>
                            <Checkbox
                              checked={formData.permissions.includes(permission.id)}
                              onChange={() => handlePermissionToggle(permission.id)}
                            />
                          </ListItemIcon>
                          <ListItemText
                            primary={
                              <Typography variant="body2">
                                {permission.name || `${permission.action} ${permission.resource}`}
                              </Typography>
                            }
                            secondary={
                              <Box>
                                <Typography variant="caption" color="text.secondary">
                                  {permission.description}
                                </Typography>
                                <Box sx={{ display: 'flex', gap: 0.5, mt: 0.5 }}>
                                  <Chip label={permission.action} size="small" variant="outlined" />
                                  <Chip label={permission.scope} size="small" variant="outlined" />
                                </Box>
                              </Box>
                            }
                          />
                        </ListItem>
                      ))}
                    </List>
                  </AccordionDetails>
                </Accordion>
              ))}
            </AccordionDetails>
          </Accordion>

          {/* Appearance */}
          <Accordion 
            expanded={expandedPanel === 'appearance'} 
            onChange={(_, isExpanded) => setExpandedPanel(isExpanded ? 'appearance' : false)}
          >
            <AccordionSummary expandIcon={<ExpandMoreIcon />}>
              <Typography variant="h6">Apparence</Typography>
            </AccordionSummary>
            <AccordionDetails>
              <Grid container spacing={2}>
                <Grid item xs={12} md={6}>
                  <Typography variant="subtitle2" gutterBottom>
                    Couleur
                  </Typography>
                  <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
                    {ROLE_COLORS.map((color) => (
                      <Box
                        key={color}
                        sx={{
                          width: 32,
                          height: 32,
                          backgroundColor: color,
                          borderRadius: '50%',
                          cursor: 'pointer',
                          border: formData.metadata.color === color ? '3px solid #000' : '2px solid #ccc',
                        }}
                        onClick={() => handleChange('metadata.color', color)}
                      />
                    ))}
                  </Box>
                </Grid>
                <Grid item xs={12} md={6}>
                  <FormControl fullWidth>
                    <InputLabel>Icône</InputLabel>
                    <Select
                      value={formData.metadata.icon || 'group'}
                      onChange={(e) => handleChange('metadata.icon', e.target.value)}
                      disabled={loading}
                    >
                      {ROLE_ICONS.map((icon) => (
                        <MenuItem key={icon.value} value={icon.value}>
                          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                            {icon.icon}
                            {icon.label}
                          </Box>
                        </MenuItem>
                      ))}
                    </Select>
                  </FormControl>
                </Grid>
                <Grid item xs={12} md={6}>
                  <TextField
                    fullWidth
                    label="Priorité"
                    type="number"
                    value={formData.metadata.priority || 0}
                    onChange={(e) => handleChange('metadata.priority', parseInt(e.target.value) || 0)}
                    disabled={loading}
                    helperText="Plus le nombre est élevé, plus la priorité est haute"
                  />
                </Grid>
              </Grid>
            </AccordionDetails>
          </Accordion>

          {/* Preview */}
          {selectedPermissions.length > 0 && (
            <Box sx={{ mt: 2 }}>
              <Typography variant="h6" gutterBottom>
                Aperçu des permissions sélectionnées
              </Typography>
              <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5, maxHeight: 150, overflow: 'auto', p: 1, border: '1px solid #ccc', borderRadius: 1 }}>
                {selectedPermissions.map((permission) => (
                  <Chip
                    key={permission.id}
                    label={`${permission.resource}:${permission.action}`}
                    size="small"
                    variant="outlined"
                    onDelete={() => handlePermissionToggle(permission.id)}
                  />
                ))}
              </Box>
            </Box>
          )}
        </Box>
      </DialogContent>

      <DialogActions sx={{ p: 3, gap: 1 }}>
        <Button onClick={onClose} disabled={loading}>
          Annuler
        </Button>
        <Button 
          variant="contained" 
          onClick={handleSubmit}
          disabled={loading}
        >
          {loading ? 'Enregistrement...' : (isEdit ? 'Modifier' : 'Créer')}
        </Button>
      </DialogActions>
    </Dialog>
  );
};

export default RoleFormDialog;