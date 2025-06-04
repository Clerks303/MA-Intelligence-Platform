/**
 * User Form Dialog - M&A Intelligence Platform
 * Sprint 4 - Formulaire de création/modification d'utilisateur
 */

import React, { useState, useEffect } from 'react';
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
  FormControlLabel,
  Switch,
  Box,
  Typography,
  Chip,
  OutlinedInput,
  Grid,
  Alert,
  IconButton,
  InputAdornment,
  Tooltip,
  Divider
} from '@mui/material';
import {
  Visibility,
  VisibilityOff,
  Info as InfoIcon,
  Security as SecurityIcon,
  Person as PersonIcon,
  Email as EmailIcon,
  VpnKey as VpnKeyIcon
} from '@mui/icons-material';
import { User, CreateUserRequest, UpdateUserRequest, Role, SecurityLevel } from '../../types';
import { useRoles } from '../../hooks/useRBAC';

interface UserFormDialogProps {
  open: boolean;
  onClose: () => void;
  onSubmit: (data: CreateUserRequest | UpdateUserRequest) => void;
  user?: User | null;
  loading?: boolean;
}

interface FormData {
  username: string;
  email: string;
  password: string;
  confirm_password: string;
  roles: string[];
  is_active: boolean;
  send_invitation_email: boolean;
  profile: {
    first_name: string;
    last_name: string;
    phone: string;
    department: string;
    job_title: string;
    timezone: string;
    language: string;
    security_notifications_enabled: boolean;
  };
}

const initialFormData: FormData = {
  username: '',
  email: '',
  password: '',
  confirm_password: '',
  roles: [],
  is_active: true,
  send_invitation_email: true,
  profile: {
    first_name: '',
    last_name: '',
    phone: '',
    department: '',
    job_title: '',
    timezone: 'Europe/Paris',
    language: 'fr',
    security_notifications_enabled: true,
  },
};

export const UserFormDialog: React.FC<UserFormDialogProps> = ({
  open,
  onClose,
  onSubmit,
  user,
  loading = false
}) => {
  const [formData, setFormData] = useState<FormData>(initialFormData);
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [activeTab, setActiveTab] = useState(0);

  const { roles } = useRoles();
  const isEdit = !!user;

  // Load user data for editing
  useEffect(() => {
    if (user && open) {
      setFormData({
        username: user.username,
        email: user.email,
        password: '',
        confirm_password: '',
        roles: user.roles?.map(r => r.id) || [],
        is_active: user.is_active,
        send_invitation_email: false,
        profile: {
          first_name: user.profile?.first_name || '',
          last_name: user.profile?.last_name || '',
          phone: user.profile?.phone || '',
          department: user.profile?.department || '',
          job_title: user.profile?.job_title || '',
          timezone: user.profile?.timezone || 'Europe/Paris',
          language: user.profile?.language || 'fr',
          security_notifications_enabled: user.profile?.security_notifications_enabled ?? true,
        },
      });
    } else if (!user && open) {
      setFormData(initialFormData);
    }
  }, [user, open]);

  // Reset form when dialog closes
  useEffect(() => {
    if (!open) {
      setFormData(initialFormData);
      setErrors({});
      setActiveTab(0);
    }
  }, [open]);

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

  const validateForm = (): boolean => {
    const newErrors: Record<string, string> = {};

    // Required fields
    if (!formData.username.trim()) {
      newErrors.username = 'Le nom d\'utilisateur est requis';
    } else if (formData.username.length < 3) {
      newErrors.username = 'Le nom d\'utilisateur doit contenir au moins 3 caractères';
    }

    if (!formData.email.trim()) {
      newErrors.email = 'L\'email est requis';
    } else if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(formData.email)) {
      newErrors.email = 'Format d\'email invalide';
    }

    // Password validation for new users
    if (!isEdit) {
      if (!formData.password) {
        newErrors.password = 'Le mot de passe est requis';
      } else if (formData.password.length < 8) {
        newErrors.password = 'Le mot de passe doit contenir au moins 8 caractères';
      }

      if (formData.password !== formData.confirm_password) {
        newErrors.confirm_password = 'Les mots de passe ne correspondent pas';
      }
    }

    // Password validation for edit if password is provided
    if (isEdit && formData.password) {
      if (formData.password.length < 8) {
        newErrors.password = 'Le mot de passe doit contenir au moins 8 caractères';
      }
      if (formData.password !== formData.confirm_password) {
        newErrors.confirm_password = 'Les mots de passe ne correspondent pas';
      }
    }

    // Roles validation
    if (formData.roles.length === 0) {
      newErrors.roles = 'Au moins un rôle doit être assigné';
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = () => {
    if (!validateForm()) return;

    const submitData = isEdit ? {
      username: formData.username,
      email: formData.email,
      is_active: formData.is_active,
      roles: formData.roles,
      profile: formData.profile,
      ...(formData.password && { password: formData.password })
    } as UpdateUserRequest : {
      username: formData.username,
      email: formData.email,
      password: formData.password,
      confirm_password: formData.confirm_password,
      roles: formData.roles,
      send_invitation_email: formData.send_invitation_email,
      profile: formData.profile,
    } as CreateUserRequest;

    onSubmit(submitData);
  };

  const selectedRoles = roles.filter(role => formData.roles.includes(role.id));

  return (
    <Dialog open={open} onClose={onClose} maxWidth="md" fullWidth>
      <DialogTitle>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <PersonIcon />
          {isEdit ? 'Modifier l\'utilisateur' : 'Nouvel utilisateur'}
        </Box>
      </DialogTitle>

      <DialogContent>
        <Box sx={{ mt: 2 }}>
          {/* Basic Information */}
          <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <PersonIcon color="primary" />
            Informations de base
          </Typography>
          
          <Grid container spacing={2}>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Nom d'utilisateur *"
                value={formData.username}
                onChange={(e) => handleChange('username', e.target.value)}
                error={!!errors.username}
                helperText={errors.username}
                disabled={loading}
                InputProps={{
                  startAdornment: (
                    <InputAdornment position="start">
                      <PersonIcon color="action" />
                    </InputAdornment>
                  ),
                }}
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Email *"
                type="email"
                value={formData.email}
                onChange={(e) => handleChange('email', e.target.value)}
                error={!!errors.email}
                helperText={errors.email}
                disabled={loading}
                InputProps={{
                  startAdornment: (
                    <InputAdornment position="start">
                      <EmailIcon color="action" />
                    </InputAdornment>
                  ),
                }}
              />
            </Grid>
            
            {/* Password fields */}
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label={isEdit ? 'Nouveau mot de passe' : 'Mot de passe *'}
                type={showPassword ? 'text' : 'password'}
                value={formData.password}
                onChange={(e) => handleChange('password', e.target.value)}
                error={!!errors.password}
                helperText={errors.password || (isEdit ? 'Laisser vide pour ne pas changer' : '')}
                disabled={loading}
                InputProps={{
                  startAdornment: (
                    <InputAdornment position="start">
                      <VpnKeyIcon color="action" />
                    </InputAdornment>
                  ),
                  endAdornment: (
                    <InputAdornment position="end">
                      <IconButton
                        onClick={() => setShowPassword(!showPassword)}
                        edge="end"
                      >
                        {showPassword ? <VisibilityOff /> : <Visibility />}
                      </IconButton>
                    </InputAdornment>
                  ),
                }}
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label={isEdit ? 'Confirmer nouveau mot de passe' : 'Confirmer mot de passe *'}
                type={showConfirmPassword ? 'text' : 'password'}
                value={formData.confirm_password}
                onChange={(e) => handleChange('confirm_password', e.target.value)}
                error={!!errors.confirm_password}
                helperText={errors.confirm_password}
                disabled={loading}
                InputProps={{
                  startAdornment: (
                    <InputAdornment position="start">
                      <VpnKeyIcon color="action" />
                    </InputAdornment>
                  ),
                  endAdornment: (
                    <InputAdornment position="end">
                      <IconButton
                        onClick={() => setShowConfirmPassword(!showConfirmPassword)}
                        edge="end"
                      >
                        {showConfirmPassword ? <VisibilityOff /> : <Visibility />}
                      </IconButton>
                    </InputAdornment>
                  ),
                }}
              />
            </Grid>
          </Grid>

          <Divider sx={{ my: 3 }} />

          {/* Roles and Permissions */}
          <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <SecurityIcon color="primary" />
            Rôles et permissions
          </Typography>
          
          <FormControl fullWidth error={!!errors.roles} sx={{ mb: 2 }}>
            <InputLabel>Rôles *</InputLabel>
            <Select
              multiple
              value={formData.roles}
              onChange={(e) => handleChange('roles', e.target.value)}
              input={<OutlinedInput label="Rôles *" />}
              renderValue={(selected) => (
                <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                  {selectedRoles.map((role) => (
                    <Chip
                      key={role.id}
                      label={role.name}
                      size="small"
                      color={role.is_system_role ? "primary" : "default"}
                    />
                  ))}
                </Box>
              )}
              disabled={loading}
            >
              {roles.map((role) => (
                <MenuItem key={role.id} value={role.id}>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, width: '100%' }}>
                    <SecurityIcon 
                      color={role.is_system_role ? "primary" : "action"} 
                      fontSize="small" 
                    />
                    <Box sx={{ flex: 1 }}>
                      <Typography variant="body2">{role.name}</Typography>
                      <Typography variant="caption" color="text.secondary">
                        {role.description}
                      </Typography>
                    </Box>
                    {role.is_system_role && (
                      <Chip label="Système" size="small" color="primary" />
                    )}
                  </Box>
                </MenuItem>
              ))}
            </Select>
            {errors.roles && (
              <Typography variant="caption" color="error" sx={{ mt: 0.5, ml: 1 }}>
                {errors.roles}
              </Typography>
            )}
          </FormControl>

          <Divider sx={{ my: 3 }} />

          {/* Profile Information */}
          <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <PersonIcon color="primary" />
            Informations de profil
          </Typography>
          
          <Grid container spacing={2}>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Prénom"
                value={formData.profile.first_name}
                onChange={(e) => handleChange('profile.first_name', e.target.value)}
                disabled={loading}
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Nom"
                value={formData.profile.last_name}
                onChange={(e) => handleChange('profile.last_name', e.target.value)}
                disabled={loading}
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Téléphone"
                value={formData.profile.phone}
                onChange={(e) => handleChange('profile.phone', e.target.value)}
                disabled={loading}
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Département"
                value={formData.profile.department}
                onChange={(e) => handleChange('profile.department', e.target.value)}
                disabled={loading}
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Poste"
                value={formData.profile.job_title}
                onChange={(e) => handleChange('profile.job_title', e.target.value)}
                disabled={loading}
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <FormControl fullWidth>
                <InputLabel>Fuseau horaire</InputLabel>
                <Select
                  value={formData.profile.timezone}
                  onChange={(e) => handleChange('profile.timezone', e.target.value)}
                  disabled={loading}
                >
                  <MenuItem value="Europe/Paris">Europe/Paris</MenuItem>
                  <MenuItem value="Europe/London">Europe/London</MenuItem>
                  <MenuItem value="America/New_York">America/New_York</MenuItem>
                  <MenuItem value="Asia/Tokyo">Asia/Tokyo</MenuItem>
                </Select>
              </FormControl>
            </Grid>
          </Grid>

          <Divider sx={{ my: 3 }} />

          {/* Settings */}
          <Typography variant="h6" gutterBottom>
            Paramètres
          </Typography>
          
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
            <FormControlLabel
              control={
                <Switch
                  checked={formData.is_active}
                  onChange={(e) => handleChange('is_active', e.target.checked)}
                  disabled={loading}
                />
              }
              label="Compte actif"
            />
            
            <FormControlLabel
              control={
                <Switch
                  checked={formData.profile.security_notifications_enabled}
                  onChange={(e) => handleChange('profile.security_notifications_enabled', e.target.checked)}
                  disabled={loading}
                />
              }
              label="Notifications de sécurité"
            />
            
            {!isEdit && (
              <FormControlLabel
                control={
                  <Switch
                    checked={formData.send_invitation_email}
                    onChange={(e) => handleChange('send_invitation_email', e.target.checked)}
                    disabled={loading}
                  />
                }
                label="Envoyer un email d'invitation"
              />
            )}
          </Box>

          {!isEdit && (
            <Alert severity="info" sx={{ mt: 2 }}>
              <Typography variant="body2">
                Si l'email d'invitation est activé, l'utilisateur recevra un lien pour configurer son mot de passe.
              </Typography>
            </Alert>
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

export default UserFormDialog;