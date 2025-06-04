// pages/Settings.js
import React, { useState } from 'react';
import {
  Box,
  Paper,
  Typography,
  TextField,
  Button,
  Grid,
  Alert,
  Divider,
  Switch,
  FormControlLabel,
  Tab,
  Tabs,
  List,
  ListItem,
  ListItemText,
  ListItemSecondaryAction,
  IconButton,
  Chip,
} from '@mui/material';
import {
  Save,
  Key,
  Storage,  // Remplace Database qui n'existe pas
  Notifications,
  Security,
  Delete,
  Add,
} from '@mui/icons-material';
import { useFormik } from 'formik';
import * as yup from 'yup';

function TabPanel({ children, value, index, ...other }) {
  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`settings-tabpanel-${index}`}
      aria-labelledby={`settings-tab-${index}`}
      {...other}
    >
      {value === index && <Box sx={{ py: 3 }}>{children}</Box>}
    </div>
  );
}

export default function Settings() {
  const [tabValue, setTabValue] = useState(0);
  const [saveSuccess, setSaveSuccess] = useState(false);
  const [apiKeys] = useState([
    { id: 1, name: 'Pappers API', key: '••••••••••••••••', active: true },
    { id: 2, name: 'OpenAI API', key: '••••••••••••••••', active: false },
  ]);

  const generalForm = useFormik({
    initialValues: {
      companyName: 'Cabinet M&A Arthur',
      adminEmail: 'admin@cabinet-arthur.com',
      maxScrapingThreads: 5,
      autoEnrichment: true,
      emailNotifications: true,
    },
    validationSchema: yup.object({
      companyName: yup.string().required('Nom requis'),
      adminEmail: yup.string().email('Email invalide').required('Email requis'),
      maxScrapingThreads: yup.number().min(1).max(10).required('Requis'),
    }),
    onSubmit: async (values) => {
      try {
        // TODO: API call to save settings
        setSaveSuccess(true);
        setTimeout(() => setSaveSuccess(false), 3000);
      } catch (error) {
        console.error('Error saving settings:', error);
      }
    },
  });

  const databaseForm = useFormik({
    initialValues: {
      supabaseUrl: process.env.REACT_APP_SUPABASE_URL || '',
      supabaseKey: '',
      backupEnabled: true,
      backupFrequency: 'daily',
    },
    onSubmit: async (values) => {
      try {
        // TODO: API call to save database settings
        setSaveSuccess(true);
        setTimeout(() => setSaveSuccess(false), 3000);
      } catch (error) {
        console.error('Error saving database settings:', error);
      }
    },
  });

  const handleTabChange = (event, newValue) => {
    setTabValue(newValue);
  };

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Paramètres
      </Typography>

      {saveSuccess && (
        <Alert severity="success" sx={{ mb: 2 }}>
          Paramètres sauvegardés avec succès
        </Alert>
      )}

      <Paper sx={{ mb: 3 }}>
        <Tabs value={tabValue} onChange={handleTabChange}>
          <Tab label="Général" icon={<Settings />} iconPosition="start" />
          <Tab label="API & Intégrations" icon={<Key />} iconPosition="start" />
          <Tab label="Base de données" icon={<Storage />} iconPosition="start" />
          <Tab label="Notifications" icon={<Notifications />} iconPosition="start" />
          <Tab label="Sécurité" icon={<Security />} iconPosition="start" />
        </Tabs>
      </Paper>

      {/* General Settings */}
      <TabPanel value={tabValue} index={0}>
        <Paper sx={{ p: 3 }}>
          <form onSubmit={generalForm.handleSubmit}>
            <Grid container spacing={3}>
              <Grid item xs={12} md={6}>
                <TextField
                  fullWidth
                  label="Nom de l'entreprise"
                  name="companyName"
                  value={generalForm.values.companyName}
                  onChange={generalForm.handleChange}
                  error={generalForm.touched.companyName && Boolean(generalForm.errors.companyName)}
                  helperText={generalForm.touched.companyName && generalForm.errors.companyName}
                />
              </Grid>
              <Grid item xs={12} md={6}>
                <TextField
                  fullWidth
                  label="Email administrateur"
                  name="adminEmail"
                  type="email"
                  value={generalForm.values.adminEmail}
                  onChange={generalForm.handleChange}
                  error={generalForm.touched.adminEmail && Boolean(generalForm.errors.adminEmail)}
                  helperText={generalForm.touched.adminEmail && generalForm.errors.adminEmail}
                />
              </Grid>
              <Grid item xs={12} md={6}>
                <TextField
                  fullWidth
                  label="Threads de scraping max"
                  name="maxScrapingThreads"
                  type="number"
                  value={generalForm.values.maxScrapingThreads}
                  onChange={generalForm.handleChange}
                  error={generalForm.touched.maxScrapingThreads && Boolean(generalForm.errors.maxScrapingThreads)}
                  helperText={generalForm.touched.maxScrapingThreads && generalForm.errors.maxScrapingThreads}
                />
              </Grid>
              <Grid item xs={12}>
                <FormControlLabel
                  control={
                    <Switch
                      checked={generalForm.values.autoEnrichment}
                      onChange={generalForm.handleChange}
                      name="autoEnrichment"
                    />
                  }
                  label="Enrichissement automatique des nouvelles entreprises"
                />
              </Grid>
              <Grid item xs={12}>
                <FormControlLabel
                  control={
                    <Switch
                      checked={generalForm.values.emailNotifications}
                      onChange={generalForm.handleChange}
                      name="emailNotifications"
                    />
                  }
                  label="Notifications par email"
                />
              </Grid>
              <Grid item xs={12}>
                <Button type="submit" variant="contained" startIcon={<Save />}>
                  Sauvegarder
                </Button>
              </Grid>
            </Grid>
          </form>
        </Paper>
      </TabPanel>

      {/* API Keys */}
      <TabPanel value={tabValue} index={1}>
        <Paper sx={{ p: 3 }}>
          <Typography variant="h6" gutterBottom>
            Clés API
          </Typography>
          <List>
            {apiKeys.map((apiKey) => (
              <ListItem key={apiKey.id} divider>
                <ListItemText
                  primary={apiKey.name}
                  secondary={apiKey.key}
                />
                <ListItemSecondaryAction>
                  <Chip
                    label={apiKey.active ? 'Active' : 'Inactive'}
                    color={apiKey.active ? 'success' : 'default'}
                    size="small"
                    sx={{ mr: 1 }}
                  />
                  <IconButton edge="end" aria-label="delete">
                    <Delete />
                  </IconButton>
                </ListItemSecondaryAction>
              </ListItem>
            ))}
          </List>
          <Box sx={{ mt: 2 }}>
            <Button startIcon={<Add />} variant="outlined">
              Ajouter une clé API
            </Button>
          </Box>

          <Divider sx={{ my: 3 }} />

          <Typography variant="h6" gutterBottom>
            Webhooks
          </Typography>
          <Typography variant="body2" color="text.secondary" paragraph>
            Configurez des webhooks pour recevoir des notifications en temps réel
          </Typography>
          <TextField
            fullWidth
            label="URL du webhook"
            placeholder="https://votre-serveur.com/webhook"
            sx={{ mb: 2 }}
          />
          <FormControlLabel
            control={<Switch />}
            label="Activer les webhooks"
          />
        </Paper>
      </TabPanel>

      {/* Database Settings */}
      <TabPanel value={tabValue} index={2}>
        <Paper sx={{ p: 3 }}>
          <form onSubmit={databaseForm.handleSubmit}>
            <Grid container spacing={3}>
              <Grid item xs={12}>
                <TextField
                  fullWidth
                  label="Supabase URL"
                  name="supabaseUrl"
                  value={databaseForm.values.supabaseUrl}
                  onChange={databaseForm.handleChange}
                  helperText="URL de votre instance Supabase"
                />
              </Grid>
              <Grid item xs={12}>
                <TextField
                  fullWidth
                  label="Supabase Key"
                  name="supabaseKey"
                  type="password"
                  value={databaseForm.values.supabaseKey}
                  onChange={databaseForm.handleChange}
                  helperText="Clé d'API Supabase (anon key)"
                />
              </Grid>
              <Grid item xs={12}>
                <FormControlLabel
                  control={
                    <Switch
                      checked={databaseForm.values.backupEnabled}
                      onChange={databaseForm.handleChange}
                      name="backupEnabled"
                    />
                  }
                  label="Sauvegardes automatiques"
                />
              </Grid>
              <Grid item xs={12} md={6}>
                <TextField
                  fullWidth
                  select
                  label="Fréquence des sauvegardes"
                  name="backupFrequency"
                  value={databaseForm.values.backupFrequency}
                  onChange={databaseForm.handleChange}
                  SelectProps={{ native: true }}
                  disabled={!databaseForm.values.backupEnabled}
                >
                  <option value="hourly">Toutes les heures</option>
                  <option value="daily">Quotidienne</option>
                  <option value="weekly">Hebdomadaire</option>
                </TextField>
              </Grid>
              <Grid item xs={12}>
                <Button type="submit" variant="contained" startIcon={<Save />}>
                  Sauvegarder
                </Button>
              </Grid>
            </Grid>
          </form>

          <Divider sx={{ my: 3 }} />

          <Typography variant="h6" gutterBottom>
            État de la base de données
          </Typography>
          <List>
            <ListItem>
              <ListItemText primary="Nombre total d'entreprises" secondary="12,456" />
            </ListItem>
            <ListItem>
              <ListItemText primary="Dernière sauvegarde" secondary="25/05/2025 14:30" />
            </ListItem>
            <ListItem>
              <ListItemText primary="Taille de la base" secondary="2.3 GB" />
            </ListItem>
          </List>
        </Paper>
      </TabPanel>

      {/* Notifications */}
      <TabPanel value={tabValue} index={3}>
        <Paper sx={{ p: 3 }}>
          <Typography variant="h6" gutterBottom>
            Préférences de notification
          </Typography>
          <List>
            <ListItem>
              <ListItemText
                primary="Nouveau scraping terminé"
                secondary="Recevoir une notification quand un scraping est terminé"
              />
              <ListItemSecondaryAction>
                <Switch defaultChecked />
              </ListItemSecondaryAction>
            </ListItem>
            <ListItem>
              <ListItemText
                primary="Nouvelles entreprises à fort potentiel"
                secondary="Alertes pour les entreprises avec un score > 80"
              />
              <ListItemSecondaryAction>
                <Switch defaultChecked />
              </ListItemSecondaryAction>
            </ListItem>
            <ListItem>
              <ListItemText
                primary="Erreurs de scraping"
                secondary="Notification en cas d'erreur lors du scraping"
              />
              <ListItemSecondaryAction>
                <Switch defaultChecked />
              </ListItemSecondaryAction>
            </ListItem>
            <ListItem>
              <ListItemText
                primary="Rapport hebdomadaire"
                secondary="Résumé des activités de la semaine"
              />
              <ListItemSecondaryAction>
                <Switch />
              </ListItemSecondaryAction>
            </ListItem>
            <ListItem>
              <ListItemText
                primary="Mises à jour système"
                secondary="Informations sur les nouvelles fonctionnalités"
              />
              <ListItemSecondaryAction>
                <Switch defaultChecked />
              </ListItemSecondaryAction>
            </ListItem>
          </List>
        </Paper>
      </TabPanel>

      {/* Security */}
      <TabPanel value={tabValue} index={4}>
        <Paper sx={{ p: 3 }}>
          <Typography variant="h6" gutterBottom>
            Sécurité du compte
          </Typography>
          <Grid container spacing={3}>
            <Grid item xs={12}>
              <Typography variant="subtitle2" gutterBottom>
                Changer le mot de passe
              </Typography>
              <Grid container spacing={2}>
                <Grid item xs={12} md={4}>
                  <TextField
                    fullWidth
                    label="Mot de passe actuel"
                    type="password"
                  />
                </Grid>
                <Grid item xs={12} md={4}>
                  <TextField
                    fullWidth
                    label="Nouveau mot de passe"
                    type="password"
                  />
                </Grid>
                <Grid item xs={12} md={4}>
                  <TextField
                    fullWidth
                    label="Confirmer le mot de passe"
                    type="password"
                  />
                </Grid>
              </Grid>
            </Grid>
            <Grid item xs={12}>
              <Button variant="outlined">
                Changer le mot de passe
              </Button>
            </Grid>
          </Grid>

          <Divider sx={{ my: 3 }} />

          <Typography variant="h6" gutterBottom>
            Sessions actives
          </Typography>
          <List>
            <ListItem>
              <ListItemText
                primary="Session actuelle"
                secondary="Chrome sur Windows - 192.168.1.1"
              />
              <ListItemSecondaryAction>
                <Chip label="Actuelle" color="primary" size="small" />
              </ListItemSecondaryAction>
            </ListItem>
            <ListItem>
              <ListItemText
                primary="Safari sur macOS"
                secondary="Dernière activité il y a 2 jours - 192.168.1.2"
              />
              <ListItemSecondaryAction>
                <Button size="small" color="error">
                  Déconnecter
                </Button>
              </ListItemSecondaryAction>
            </ListItem>
          </List>

          <Divider sx={{ my: 3 }} />

          <Typography variant="h6" gutterBottom>
            Authentification à deux facteurs
          </Typography>
          <Typography variant="body2" color="text.secondary" paragraph>
            Ajoutez une couche de sécurité supplémentaire à votre compte
          </Typography>
          <Button variant="outlined">
            Configurer 2FA
          </Button>
        </Paper>
      </TabPanel>
    </Box>
  );
}