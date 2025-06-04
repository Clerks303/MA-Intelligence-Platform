/**
 * MFA Interface - M&A Intelligence Platform
 * Sprint 4 - Interface Multi-Factor Authentication
 */

import React from 'react';
import {
  Box,
  Paper,
  Typography,
  Alert
} from '@mui/material';
import {
  Security as SecurityIcon
} from '@mui/icons-material';

export const MFAInterface: React.FC = () => {
  return (
    <Box sx={{ p: 3 }}>
      <Paper sx={{ p: 3 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 3 }}>
          <SecurityIcon color="primary" />
          <Typography variant="h5">
            Interface MFA (Multi-Factor Authentication)
          </Typography>
        </Box>
        
        <Alert severity="info">
          <Typography variant="body1">
            L'interface MFA est en cours de développement. Cette section permettra de :
          </Typography>
          <ul>
            <li>Configurer l'authentification à deux facteurs</li>
            <li>Gérer les codes de sauvegarde</li>
            <li>Configurer les méthodes d'authentification (TOTP, SMS, etc.)</li>
            <li>Administrer les appareils de confiance</li>
          </ul>
        </Alert>
      </Paper>
    </Box>
  );
};

export default MFAInterface;