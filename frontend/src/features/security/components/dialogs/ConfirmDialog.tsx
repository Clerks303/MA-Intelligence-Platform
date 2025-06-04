/**
 * Confirm Dialog - M&A Intelligence Platform
 * Sprint 4 - Dialog de confirmation pour actions critiques
 */

import React from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  DialogContentText,
  Button,
  Box,
  Typography,
  Alert,
  CircularProgress
} from '@mui/material';
import {
  Warning as WarningIcon,
  Error as ErrorIcon,
  Info as InfoIcon,
  CheckCircle as SuccessIcon
} from '@mui/icons-material';

interface ConfirmDialogProps {
  open: boolean;
  onClose: () => void;
  onConfirm: () => void;
  title: string;
  message: string;
  severity?: 'error' | 'warning' | 'info' | 'success';
  confirmText?: string;
  cancelText?: string;
  loading?: boolean;
}

const getSeverityIcon = (severity: string) => {
  switch (severity) {
    case 'error': return <ErrorIcon color="error" />;
    case 'warning': return <WarningIcon color="warning" />;
    case 'info': return <InfoIcon color="info" />;
    case 'success': return <SuccessIcon color="success" />;
    default: return <WarningIcon color="warning" />;
  }
};

const getSeverityColor = (severity: string) => {
  switch (severity) {
    case 'error': return 'error';
    case 'warning': return 'warning';
    case 'info': return 'info';
    case 'success': return 'success';
    default: return 'warning';
  }
};

export const ConfirmDialog: React.FC<ConfirmDialogProps> = ({
  open,
  onClose,
  onConfirm,
  title,
  message,
  severity = 'warning',
  confirmText = 'Confirmer',
  cancelText = 'Annuler',
  loading = false
}) => {
  return (
    <Dialog
      open={open}
      onClose={onClose}
      maxWidth="sm"
      fullWidth
      PaperProps={{
        sx: {
          borderTop: `4px solid`,
          borderTopColor: `${getSeverityColor(severity)}.main`
        }
      }}
    >
      <DialogTitle>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          {getSeverityIcon(severity)}
          <Typography variant="h6">
            {title}
          </Typography>
        </Box>
      </DialogTitle>
      
      <DialogContent>
        <Alert 
          severity={severity as any} 
          sx={{ mb: 2 }}
          icon={false}
        >
          <DialogContentText component="div">
            {message}
          </DialogContentText>
        </Alert>
        
        {severity === 'error' && (
          <Alert severity="warning" sx={{ mt: 2 }}>
            <Typography variant="body2">
              <strong>Attention :</strong> Cette action est irréversible. Assurez-vous de bien comprendre les conséquences avant de continuer.
            </Typography>
          </Alert>
        )}
      </DialogContent>
      
      <DialogActions sx={{ p: 3, gap: 1 }}>
        <Button 
          onClick={onClose} 
          disabled={loading}
          color="inherit"
        >
          {cancelText}
        </Button>
        <Button 
          onClick={onConfirm}
          disabled={loading}
          variant="contained"
          color={getSeverityColor(severity) as any}
          startIcon={loading ? <CircularProgress size={16} color="inherit" /> : undefined}
        >
          {loading ? 'Traitement...' : confirmText}
        </Button>
      </DialogActions>
    </Dialog>
  );
};

export default ConfirmDialog;