/**
 * Monitoring Page - Page principale dashboard monitoring
 * US-004: Page complète monitoring avec tous les widgets
 */

import React from 'react';
import { Box, Container } from '@mui/material';
import { MonitoringProvider } from '../contexts/MonitoringContext';
import MonitoringDashboard from '../components/monitoring/MonitoringDashboard';

const Monitoring = () => {
  return (
    <MonitoringProvider>
      <Container maxWidth={false} sx={{ py: 2 }}>
        <MonitoringDashboard />
      </Container>
    </MonitoringProvider>
  );
};

export default Monitoring;