/**
 * RealTimeIndicator - Indicateur temps réel avec statut connexion
 * US-004: Widget statut temps réel avec animations
 */

import React, { useState, useEffect } from 'react';
import {
  Box,
  Chip,
  Typography,
  Tooltip,
  Zoom,
  keyframes
} from '@mui/material';
import {
  Circle,
  Wifi,
  WifiOff,
  Sync,
  SyncDisabled
} from '@mui/icons-material';

// Animation pulse pour l'indicateur actif
const pulse = keyframes`
  0% {
    opacity: 1;
    transform: scale(1);
  }
  50% {
    opacity: 0.7;
    transform: scale(1.1);
  }
  100% {
    opacity: 1;
    transform: scale(1);
  }
`;

const RealTimeIndicator = ({ enabled, lastUpdated, error }) => {
  const [secondsSinceUpdate, setSecondsSinceUpdate] = useState(0);

  // Calculer temps depuis dernière mise à jour
  useEffect(() => {
    if (!lastUpdated) return;

    const interval = setInterval(() => {
      const now = new Date();
      const last = new Date(lastUpdated);
      const diffSeconds = Math.floor((now - last) / 1000);
      setSecondsSinceUpdate(diffSeconds);
    }, 1000);

    return () => clearInterval(interval);
  }, [lastUpdated]);

  // Déterminer l'état de connexion
  const getConnectionStatus = () => {
    if (error) return 'error';
    if (!enabled) return 'disabled';
    if (secondsSinceUpdate > 120) return 'stale'; // Pas de mise à jour depuis 2min
    if (secondsSinceUpdate > 60) return 'warning'; // Pas de mise à jour depuis 1min
    return 'connected';
  };

  const status = getConnectionStatus();

  // Configuration par état
  const statusConfig = {
    connected: {
      color: 'success',
      icon: <Wifi />,
      label: 'Temps Réel',
      description: `Dernière mise à jour: ${formatUpdateTime(secondsSinceUpdate)}`,
      animate: true
    },
    warning: {
      color: 'warning',
      icon: <Wifi />,
      label: 'Connexion Lente',
      description: `Dernière mise à jour: ${formatUpdateTime(secondsSinceUpdate)}`,
      animate: false
    },
    stale: {
      color: 'error',
      icon: <WifiOff />,
      label: 'Connexion Perdue',
      description: `Dernière mise à jour: ${formatUpdateTime(secondsSinceUpdate)}`,
      animate: false
    },
    disabled: {
      color: 'default',
      icon: <SyncDisabled />,
      label: 'Mode Manuel',
      description: 'Temps réel désactivé',
      animate: false
    },
    error: {
      color: 'error',
      icon: <WifiOff />,
      label: 'Erreur',
      description: error || 'Erreur de connexion',
      animate: false
    }
  };

  const config = statusConfig[status];

  function formatUpdateTime(seconds) {
    if (seconds < 60) return `${seconds}s`;
    if (seconds < 3600) return `${Math.floor(seconds / 60)}min`;
    return `${Math.floor(seconds / 3600)}h`;
  }

  return (
    <Tooltip 
      title={
        <Box>
          <Typography variant="body2">{config.description}</Typography>
          {lastUpdated && (
            <Typography variant="caption" color="textSecondary">
              {new Date(lastUpdated).toLocaleTimeString()}
            </Typography>
          )}
        </Box>
      }
      arrow
    >
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
        {/* Indicateur visuel animé */}
        <Box
          sx={{
            position: 'relative',
            display: 'flex',
            alignItems: 'center'
          }}
        >
          <Circle
            sx={{
              fontSize: 12,
              color: `${config.color}.main`,
              animation: config.animate ? `${pulse} 2s ease-in-out infinite` : 'none'
            }}
          />
          
          {/* Cercle externe pour animation */}
          {config.animate && (
            <Circle
              sx={{
                fontSize: 16,
                color: `${config.color}.main`,
                position: 'absolute',
                opacity: 0.3,
                animation: `${pulse} 2s ease-in-out infinite reverse`
              }}
            />
          )}
        </Box>

        {/* Chip avec statut */}
        <Chip
          icon={config.icon}
          label={config.label}
          color={config.color}
          variant="outlined"
          size="small"
          sx={{
            '& .MuiChip-icon': {
              fontSize: 16
            }
          }}
        />

        {/* Temps depuis dernière mise à jour */}
        {enabled && lastUpdated && (
          <Typography 
            variant="caption" 
            color="textSecondary"
            sx={{ 
              minWidth: 40,
              textAlign: 'right',
              fontFamily: 'monospace'
            }}
          >
            {formatUpdateTime(secondsSinceUpdate)}
          </Typography>
        )}
      </Box>
    </Tooltip>
  );
};

export default RealTimeIndicator;