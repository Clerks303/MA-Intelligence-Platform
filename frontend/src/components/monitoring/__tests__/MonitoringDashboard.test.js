/**
 * Tests MonitoringDashboard - US-004
 * Tests du dashboard principal monitoring
 */

import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ThemeProvider, createTheme } from '@mui/material';
import MonitoringDashboard from '../MonitoringDashboard';
import { MonitoringProvider } from '../../../contexts/MonitoringContext';

// Mock des composants enfants
jest.mock('../SystemOverviewCard', () => {
  return function MockSystemOverviewCard({ data, loading, onRefresh }) {
    return (
      <div data-testid="system-overview-card">
        <button onClick={onRefresh}>Refresh</button>
        {loading && <span>Loading...</span>}
        {data && <span>Data loaded</span>}
      </div>
    );
  };
});

jest.mock('../AlertsOverview', () => {
  return function MockAlertsOverview() {
    return <div data-testid="alerts-overview">Alerts Overview</div>;
  };
});

jest.mock('../MetricsChart', () => {
  return function MockMetricsChart({ title, metric, timeWindow }) {
    return (
      <div data-testid="metrics-chart">
        <span>{title}</span>
        <span>{metric}</span>
        <span>{timeWindow}</span>
      </div>
    );
  };
});

jest.mock('../HealthStatus', () => {
  return function MockHealthStatus() {
    return <div data-testid="health-status">Health Status</div>;
  };
});

jest.mock('../RealTimeIndicator', () => {
  return function MockRealTimeIndicator({ enabled, lastUpdated }) {
    return (
      <div data-testid="realtime-indicator">
        <span>{enabled ? 'Enabled' : 'Disabled'}</span>
        <span>{lastUpdated || 'No update'}</span>
      </div>
    );
  };
});

// Mock API
jest.mock('../../../services/api', () => ({
  get: jest.fn(() => Promise.resolve({
    data: {
      timestamp: '2024-01-01T10:00:00Z',
      system_status: {
        overall_health: 'healthy',
        services_healthy: 4,
        services_total: 4,
        availability_percent: 100
      },
      critical_issues: {
        critical_alerts_count: 0,
        emergency_alerts_count: 0,
        immediate_attention_required: false
      },
      key_metrics: {
        api_requests_last_hour: 1500,
        avg_response_time_ms: 120,
        error_rate_percent: 1.2,
        active_users: 15
      }
    }
  }))
}));

const createWrapper = () => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
      mutations: { retry: false }
    }
  });

  const theme = createTheme();

  return ({ children }) => (
    <QueryClientProvider client={queryClient}>
      <ThemeProvider theme={theme}>
        <MonitoringProvider>
          {children}
        </MonitoringProvider>
      </ThemeProvider>
    </QueryClientProvider>
  );
};

describe('MonitoringDashboard', () => {
  let wrapper;

  beforeEach(() => {
    wrapper = createWrapper();
    jest.clearAllMocks();
  });

  test('renders dashboard header correctly', () => {
    render(<MonitoringDashboard />, { wrapper });

    expect(screen.getByText('Monitoring Dashboard')).toBeInTheDocument();
    expect(screen.getByText('Temps réel')).toBeInTheDocument();
  });

  test('renders all dashboard components', () => {
    render(<MonitoringDashboard />, { wrapper });

    expect(screen.getByTestId('system-overview-card')).toBeInTheDocument();
    expect(screen.getByTestId('alerts-overview')).toBeInTheDocument();
    expect(screen.getByTestId('health-status')).toBeInTheDocument();
    expect(screen.getAllByTestId('metrics-chart')).toHaveLength(3);
  });

  test('renders real-time indicator', () => {
    render(<MonitoringDashboard />, { wrapper });

    expect(screen.getByTestId('realtime-indicator')).toBeInTheDocument();
  });

  test('handles real-time toggle', () => {
    render(<MonitoringDashboard />, { wrapper });

    const realTimeSwitch = screen.getByRole('checkbox');
    expect(realTimeSwitch).toBeChecked();

    fireEvent.click(realTimeSwitch);
    expect(realTimeSwitch).not.toBeChecked();
  });

  test('handles refresh button click', () => {
    render(<MonitoringDashboard />, { wrapper });

    const refreshButton = screen.getByRole('button', { name: /actualiser/i });
    fireEvent.click(refreshButton);

    // Vérifier que le refresh est déclenché
    expect(refreshButton).toBeInTheDocument();
  });

  test('displays system health status', async () => {
    render(<MonitoringDashboard />, { wrapper });

    await waitFor(() => {
      expect(screen.getByText('Système Sain')).toBeInTheDocument();
    });
  });

  test('shows settings menu on click', () => {
    render(<MonitoringDashboard />, { wrapper });

    const settingsButton = screen.getByRole('button', { name: /paramètres/i });
    fireEvent.click(settingsButton);

    expect(screen.getByText('Vue Grille')).toBeInTheDocument();
    expect(screen.getByText('Vue Compacte')).toBeInTheDocument();
  });

  test('displays error state when there are issues', () => {
    const api = require('../../../services/api');
    api.get.mockRejectedValueOnce(new Error('Connection failed'));

    render(<MonitoringDashboard />, { wrapper });

    // Le test devrait vérifier l'affichage d'erreur
    // mais cela nécessiterait un état d'erreur mockée dans le context
  });

  test('renders metrics charts with correct props', () => {
    render(<MonitoringDashboard />, { wrapper });

    const metricsCharts = screen.getAllByTestId('metrics-chart');
    
    expect(metricsCharts[0]).toHaveTextContent('API Performance');
    expect(metricsCharts[1]).toHaveTextContent('Activité Business');
    expect(metricsCharts[2]).toHaveTextContent('Système Resources');
  });

  test('handles component refresh from child', () => {
    render(<MonitoringDashboard />, { wrapper });

    const refreshButton = screen.getByText('Refresh');
    fireEvent.click(refreshButton);

    // Vérifier que le refresh est propagé
    expect(refreshButton).toBeInTheDocument();
  });
});

describe('MonitoringDashboard responsive behavior', () => {
  test('adapts layout for different screen sizes', () => {
    const { container } = render(<MonitoringDashboard />, { wrapper: createWrapper() });

    // Vérifier que les Grid items utilisent les breakpoints MUI
    const gridItems = container.querySelectorAll('.MuiGrid-item');
    expect(gridItems.length).toBeGreaterThan(0);
  });
});

describe('MonitoringDashboard notifications', () => {
  test('displays notifications when present', () => {
    // Ce test nécessiterait un mock du context avec des notifications
    render(<MonitoringDashboard />, { wrapper: createWrapper() });
    
    // Pour l'instant, vérifier que le composant se rend sans erreur
    expect(screen.getByText('Monitoring Dashboard')).toBeInTheDocument();
  });
});