/**
 * User Details Dialog - M&A Intelligence Platform
 * Sprint 4 - Dialog détaillé d'affichage des informations utilisateur
 */

import React, { useState } from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  Box,
  Typography,
  Chip,
  Grid,
  Card,
  CardContent,
  Divider,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  IconButton,
  Tooltip,
  Avatar,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Tabs,
  Tab,
  Alert
} from '@mui/material';
import {
  Person as PersonIcon,
  Email as EmailIcon,
  Phone as PhoneIcon,
  Business as BusinessIcon,
  Schedule as ScheduleIcon,
  Security as SecurityIcon,
  Shield as ShieldIcon,
  VpnKey as VpnKeyIcon,
  DeviceHub as DeviceIcon,
  History as HistoryIcon,
  Warning as WarningIcon,
  CheckCircle as CheckCircleIcon,
  Block as BlockIcon,
  Close as CloseIcon,
  Edit as EditIcon,
  Refresh as RefreshIcon
} from '@mui/icons-material';
import { User, UserSession } from '../../types';
import { useSessions } from '../../hooks/useSecurityAuth';
import { useAuditEvents } from '../../hooks/useAudit';

interface UserDetailsDialogProps {
  open: boolean;
  onClose: () => void;
  user: User | null;
}

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel({ children, value, index }: TabPanelProps) {
  return (
    <div role="tabpanel" hidden={value !== index}>
      {value === index && <Box sx={{ py: 2 }}>{children}</Box>}
    </div>
  );
}

export const UserDetailsDialog: React.FC<UserDetailsDialogProps> = ({
  open,
  onClose,
  user
}) => {
  const [tabValue, setTabValue] = useState(0);
  
  // Hooks for user sessions and activity
  const { sessions, isLoading: isLoadingSessions } = useSessions(user?.id);
  const { 
    events: auditEvents, 
    isLoading: isLoadingAudit 
  } = useAuditEvents(
    user ? { users: [user.id] } : undefined,
    1,
    20
  );

  if (!user) return null;

  const formatDate = (dateString: string | undefined) => {
    if (!dateString) return 'N/A';
    return new Date(dateString).toLocaleString('fr-FR');
  };

  const getSecurityLevelColor = (level: string) => {
    switch (level) {
      case 'LOW': return 'info';
      case 'MEDIUM': return 'warning';
      case 'HIGH': return 'error';
      case 'CRITICAL': return 'error';
      default: return 'default';
    }
  };

  const getStatusColor = (isActive: boolean) => {
    return isActive ? 'success' : 'error';
  };

  return (
    <Dialog 
      open={open} 
      onClose={onClose} 
      maxWidth="lg" 
      fullWidth
      PaperProps={{
        sx: { height: '90vh' }
      }}
    >
      <DialogTitle>
        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
            <Avatar sx={{ bgcolor: 'primary.main' }}>
              <PersonIcon />
            </Avatar>
            <Box>
              <Typography variant="h6">
                {user.profile?.first_name && user.profile?.last_name
                  ? `${user.profile.first_name} ${user.profile.last_name}`
                  : user.username
                }
              </Typography>
              <Typography variant="body2" color="text.secondary">
                {user.email}
              </Typography>
            </Box>
          </Box>
          <IconButton onClick={onClose}>
            <CloseIcon />
          </IconButton>
        </Box>
      </DialogTitle>

      <DialogContent>
        <Box sx={{ borderBottom: 1, borderColor: 'divider', mb: 2 }}>
          <Tabs value={tabValue} onChange={(_, newValue) => setTabValue(newValue)}>
            <Tab label="Informations" icon={<PersonIcon />} />
            <Tab label="Sécurité" icon={<SecurityIcon />} />
            <Tab label="Sessions" icon={<DeviceIcon />} />
            <Tab label="Activité" icon={<HistoryIcon />} />
          </Tabs>
        </Box>

        {/* Tab 1: Basic Information */}
        <TabPanel value={tabValue} index={0}>
          <Grid container spacing={3}>
            {/* Profile Info */}
            <Grid item xs={12} md={6}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <PersonIcon color="primary" />
                    Informations personnelles
                  </Typography>
                  <List dense>
                    <ListItem>
                      <ListItemIcon><PersonIcon /></ListItemIcon>
                      <ListItemText 
                        primary="Nom complet"
                        secondary={`${user.profile?.first_name || ''} ${user.profile?.last_name || ''}`.trim() || 'Non renseigné'}
                      />
                    </ListItem>
                    <ListItem>
                      <ListItemIcon><EmailIcon /></ListItemIcon>
                      <ListItemText 
                        primary="Email"
                        secondary={user.email}
                      />
                    </ListItem>
                    <ListItem>
                      <ListItemIcon><PhoneIcon /></ListItemIcon>
                      <ListItemText 
                        primary="Téléphone"
                        secondary={user.profile?.phone || 'Non renseigné'}
                      />
                    </ListItem>
                    <ListItem>
                      <ListItemIcon><BusinessIcon /></ListItemIcon>
                      <ListItemText 
                        primary="Département"
                        secondary={user.profile?.department || 'Non renseigné'}
                      />
                    </ListItem>
                    <ListItem>
                      <ListItemIcon><BusinessIcon /></ListItemIcon>
                      <ListItemText 
                        primary="Poste"
                        secondary={user.profile?.job_title || 'Non renseigné'}
                      />
                    </ListItem>
                  </List>
                </CardContent>
              </Card>
            </Grid>

            {/* Account Status */}
            <Grid item xs={12} md={6}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <ShieldIcon color="primary" />
                    Statut du compte
                  </Typography>
                  <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <Typography variant="body2">Statut</Typography>
                      <Chip
                        label={user.is_active ? 'Actif' : 'Inactif'}
                        color={getStatusColor(user.is_active) as any}
                        icon={user.is_active ? <CheckCircleIcon /> : <BlockIcon />}
                      />
                    </Box>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <Typography variant="body2">Administrateur</Typography>
                      <Chip
                        label={user.is_superuser ? 'Oui' : 'Non'}
                        color={user.is_superuser ? 'warning' : 'default'}
                        icon={<SecurityIcon />}
                      />
                    </Box>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <Typography variant="body2">MFA</Typography>
                      <Chip
                        label={user.mfa_enabled ? 'Activé' : 'Désactivé'}
                        color={user.mfa_enabled ? 'success' : 'warning'}
                        icon={<VpnKeyIcon />}
                      />
                    </Box>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <Typography variant="body2">Niveau de sécurité</Typography>
                      <Chip
                        label={user.security_level || 'MEDIUM'}
                        color={getSecurityLevelColor(user.security_level || 'MEDIUM') as any}
                        icon={<ShieldIcon />}
                      />
                    </Box>
                  </Box>
                </CardContent>
              </Card>
            </Grid>

            {/* Dates */}
            <Grid item xs={12}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <ScheduleIcon color="primary" />
                    Informations temporelles
                  </Typography>
                  <Grid container spacing={2}>
                    <Grid item xs={12} md={3}>
                      <Typography variant="body2" color="text.secondary">Créé le</Typography>
                      <Typography variant="body1">{formatDate(user.created_at)}</Typography>
                    </Grid>
                    <Grid item xs={12} md={3}>
                      <Typography variant="body2" color="text.secondary">Modifié le</Typography>
                      <Typography variant="body1">{formatDate(user.updated_at)}</Typography>
                    </Grid>
                    <Grid item xs={12} md={3}>
                      <Typography variant="body2" color="text.secondary">Dernière connexion</Typography>
                      <Typography variant="body1">{formatDate(user.last_login)}</Typography>
                    </Grid>
                    <Grid item xs={12} md={3}>
                      <Typography variant="body2" color="text.secondary">Fuseau horaire</Typography>
                      <Typography variant="body1">{user.profile?.timezone || 'Non défini'}</Typography>
                    </Grid>
                  </Grid>
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        </TabPanel>

        {/* Tab 2: Security */}
        <TabPanel value={tabValue} index={1}>
          <Grid container spacing={3}>
            {/* Roles */}
            <Grid item xs={12} md={6}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <SecurityIcon color="primary" />
                    Rôles assignés
                  </Typography>
                  {user.roles && user.roles.length > 0 ? (
                    <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                      {user.roles.map((role) => (
                        <Chip
                          key={role.id}
                          label={role.name}
                          color={role.is_system_role ? "primary" : "default"}
                          icon={<SecurityIcon />}
                          onClick={() => {}} // TODO: Show role details
                        />
                      ))}
                    </Box>
                  ) : (
                    <Alert severity="warning">
                      Aucun rôle assigné
                    </Alert>
                  )}
                </CardContent>
              </Card>
            </Grid>

            {/* Security Settings */}
            <Grid item xs={12} md={6}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <ShieldIcon color="primary" />
                    Paramètres de sécurité
                  </Typography>
                  <List dense>
                    <ListItem>
                      <ListItemText 
                        primary="Notifications de sécurité"
                        secondary={user.profile?.security_notifications_enabled ? 'Activées' : 'Désactivées'}
                      />
                      <Chip
                        size="small"
                        color={user.profile?.security_notifications_enabled ? 'success' : 'default'}
                        icon={user.profile?.security_notifications_enabled ? <CheckCircleIcon /> : <BlockIcon />}
                      />
                    </ListItem>
                    <ListItem>
                      <ListItemText 
                        primary="MFA obligatoire"
                        secondary={user.mfa_setup_complete ? 'Configuré' : 'Non configuré'}
                      />
                      <Chip
                        size="small"
                        color={user.mfa_setup_complete ? 'success' : 'warning'}
                        icon={<VpnKeyIcon />}
                      />
                    </ListItem>
                    <ListItem>
                      <ListItemText 
                        primary="Tentatives de connexion échouées"
                        secondary={`${user.failed_login_attempts || 0} tentatives`}
                      />
                      {(user.failed_login_attempts || 0) > 0 && (
                        <Chip
                          size="small"
                          color="warning"
                          icon={<WarningIcon />}
                        />
                      )}
                    </ListItem>
                    {user.locked_until && (
                      <ListItem>
                        <ListItemText 
                          primary="Compte verrouillé jusqu'au"
                          secondary={formatDate(user.locked_until)}
                        />
                        <Chip
                          size="small"
                          color="error"
                          icon={<BlockIcon />}
                        />
                      </ListItem>
                    )}
                  </List>
                </CardContent>
              </Card>
            </Grid>

            {/* Permissions Preview */}
            <Grid item xs={12}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <VpnKeyIcon color="primary" />
                    Permissions (aperçu)
                  </Typography>
                  {user.permissions && user.permissions.length > 0 ? (
                    <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, maxHeight: 200, overflow: 'auto' }}>
                      {user.permissions.map((permission) => (
                        <Chip
                          key={permission.id}
                          label={`${permission.resource}:${permission.action}`}
                          size="small"
                          variant="outlined"
                        />
                      ))}
                    </Box>
                  ) : (
                    <Typography variant="body2" color="text.secondary">
                      Permissions héritées des rôles
                    </Typography>
                  )}
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        </TabPanel>

        {/* Tab 3: Sessions */}
        <TabPanel value={tabValue} index={2}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                <Typography variant="h6" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <DeviceIcon color="primary" />
                  Sessions actives
                </Typography>
                <IconButton onClick={() => {}} disabled={isLoadingSessions}>
                  <RefreshIcon />
                </IconButton>
              </Box>
              
              {isLoadingSessions ? (
                <Typography>Chargement des sessions...</Typography>
              ) : sessions.length > 0 ? (
                <TableContainer>
                  <Table>
                    <TableHead>
                      <TableRow>
                        <TableCell>Device</TableCell>
                        <TableCell>IP Address</TableCell>
                        <TableCell>Location</TableCell>
                        <TableCell>Created</TableCell>
                        <TableCell>Last Activity</TableCell>
                        <TableCell>Status</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {sessions.map((session) => (
                        <TableRow key={session.session_id}>
                          <TableCell>
                            <Typography variant="body2">
                              {session.user_agent || 'Unknown'}
                            </Typography>
                          </TableCell>
                          <TableCell>{session.ip_address}</TableCell>
                          <TableCell>
                            {session.location 
                              ? `${session.location.city}, ${session.location.country}`
                              : 'Unknown'
                            }
                          </TableCell>
                          <TableCell>{formatDate(session.created_at)}</TableCell>
                          <TableCell>{formatDate(session.last_activity)}</TableCell>
                          <TableCell>
                            <Box sx={{ display: 'flex', gap: 1 }}>
                              {session.is_current && (
                                <Chip label="Courante" color="success" size="small" />
                              )}
                              {session.is_suspicious && (
                                <Chip label="Suspecte" color="warning" size="small" />
                              )}
                              <Chip 
                                label={session.security_level} 
                                color={getSecurityLevelColor(session.security_level) as any} 
                                size="small" 
                              />
                            </Box>
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              ) : (
                <Alert severity="info">
                  Aucune session active
                </Alert>
              )}
            </CardContent>
          </Card>
        </TabPanel>

        {/* Tab 4: Activity */}
        <TabPanel value={tabValue} index={3}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <HistoryIcon color="primary" />
                Activité récente
              </Typography>
              
              {isLoadingAudit ? (
                <Typography>Chargement de l'activité...</Typography>
              ) : auditEvents.length > 0 ? (
                <TableContainer>
                  <Table>
                    <TableHead>
                      <TableRow>
                        <TableCell>Date</TableCell>
                        <TableCell>Action</TableCell>
                        <TableCell>Ressource</TableCell>
                        <TableCell>Résultat</TableCell>
                        <TableCell>IP</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {auditEvents.map((event) => (
                        <TableRow key={event.event_id}>
                          <TableCell>{formatDate(event.timestamp)}</TableCell>
                          <TableCell>
                            <Box>
                              <Typography variant="body2">{event.action}</Typography>
                              <Typography variant="caption" color="text.secondary">
                                {event.event_type}
                              </Typography>
                            </Box>
                          </TableCell>
                          <TableCell>{event.resource_type}</TableCell>
                          <TableCell>
                            <Chip
                              label={event.success ? 'Succès' : 'Échec'}
                              color={event.success ? 'success' : 'error'}
                              size="small"
                            />
                          </TableCell>
                          <TableCell>{event.ip_address}</TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              ) : (
                <Alert severity="info">
                  Aucune activité récente
                </Alert>
              )}
            </CardContent>
          </Card>
        </TabPanel>
      </DialogContent>

      <DialogActions sx={{ p: 3 }}>
        <Button onClick={onClose}>
          Fermer
        </Button>
      </DialogActions>
    </Dialog>
  );
};

export default UserDetailsDialog;