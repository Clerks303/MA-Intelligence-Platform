/**
 * Tests MonitoringContext - US-004
 * Tests du context provider monitoring
 */

import React from 'react';
import { render, renderHook, waitFor, act } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { MonitoringProvider, useMonitoring } from '../../../contexts/MonitoringContext';

// Mock API
jest.mock('../../../services/api', () => ({
  get: jest.fn(() => Promise.resolve({
    data: {
      timestamp: '2024-01-01T10:00:00Z',
      system_status: {
        overall_health: 'healthy',
        services_healthy: 3,
        services_total: 4,
        availability_percent: 95.5
      },
      key_metrics: {
        api_requests_last_hour: 1250,
        avg_response_time_ms: 145,
        error_rate_percent: 2.1,
        active_users: 12
      }
    }
  })),
  post: jest.fn(() => Promise.resolve({
    data: {
      timestamp: '2024-01-01T10:00:00Z',
      alert_id: 'test_alert_123',
      status: 'acknowledged'
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

  return ({ children }) => (
    <QueryClientProvider client={queryClient}>
      <MonitoringProvider>
        {children}
      </MonitoringProvider>
    </QueryClientProvider>
  );
};

describe('MonitoringContext', () => {
  let wrapper;

  beforeEach(() => {
    wrapper = createWrapper();
    jest.clearAllMocks();
  });

  test('provides monitoring context values', () => {
    const { result } = renderHook(() => useMonitoring(), { wrapper });

    expect(result.current).toHaveProperty('overview');
    expect(result.current).toHaveProperty('alerts');
    expect(result.current).toHaveProperty('metrics');
    expect(result.current).toHaveProperty('health');
    expect(result.current).toHaveProperty('loading');
    expect(result.current).toHaveProperty('realTimeEnabled');
    expect(result.current).toHaveProperty('toggleRealTime');
    expect(result.current).toHaveProperty('refreshData');
    expect(result.current).toHaveProperty('acknowledgeAlert');
    expect(result.current).toHaveProperty('resolveAlert');
  });

  test('initializes with default state', () => {
    const { result } = renderHook(() => useMonitoring(), { wrapper });

    expect(result.current.realTimeEnabled).toBe(true);
    expect(result.current.refreshInterval).toBe(30000);
    expect(result.current.notifications).toEqual([]);
    expect(result.current.overview).toBeNull();
  });

  test('toggles real-time monitoring', () => {
    const { result } = renderHook(() => useMonitoring(), { wrapper });

    act(() => {
      result.current.toggleRealTime();
    });

    expect(result.current.realTimeEnabled).toBe(false);

    act(() => {
      result.current.toggleRealTime();
    });

    expect(result.current.realTimeEnabled).toBe(true);
  });

  test('acknowledges alert successfully', async () => {
    const { result } = renderHook(() => useMonitoring(), { wrapper });

    await act(async () => {
      const response = await result.current.acknowledgeAlert('test_alert_123', 'Test comment');
      expect(response.alert_id).toBe('test_alert_123');
      expect(response.status).toBe('acknowledged');
    });
  });

  test('adds and removes notifications', () => {
    const { result } = renderHook(() => useMonitoring(), { wrapper });

    act(() => {
      result.current.addNotification({
        type: 'success',
        title: 'Test notification',
        message: 'Test message'
      });
    });

    expect(result.current.notifications).toHaveLength(1);
    expect(result.current.notifications[0]).toMatchObject({
      type: 'success',
      title: 'Test notification',
      message: 'Test message'
    });

    const notificationId = result.current.notifications[0].id;

    act(() => {
      result.current.removeNotification(notificationId);
    });

    expect(result.current.notifications).toHaveLength(0);
  });

  test('handles connection status correctly', async () => {
    const { result } = renderHook(() => useMonitoring(), { wrapper });

    await waitFor(() => {
      expect(result.current.isConnected).toBe(true);
    });

    expect(result.current.systemHealthy).toBe(true);
  });

  test('refreshes data on demand', async () => {
    const { result } = renderHook(() => useMonitoring(), { wrapper });

    await act(async () => {
      await result.current.refreshData();
    });

    // Vérifier que les données ont été rechargées
    expect(result.current.loading).toBe(false);
  });

  test('throws error when used outside provider', () => {
    const consoleSpy = jest.spyOn(console, 'error').mockImplementation(() => {});
    
    expect(() => {
      renderHook(() => useMonitoring());
    }).toThrow('useMonitoring must be used within a MonitoringProvider');

    consoleSpy.mockRestore();
  });
});

describe('MonitoringProvider data flow', () => {
  test('updates overview data from API', async () => {
    const wrapper = createWrapper();
    const { result } = renderHook(() => useMonitoring(), { wrapper });

    await waitFor(() => {
      expect(result.current.overview).toBeTruthy();
    });

    expect(result.current.overview.system_status.overall_health).toBe('healthy');
    expect(result.current.overview.key_metrics.api_requests_last_hour).toBe(1250);
  });

  test('handles API errors gracefully', async () => {
    const api = require('../../../services/api');
    api.get.mockRejectedValueOnce(new Error('Network error'));

    const wrapper = createWrapper();
    const { result } = renderHook(() => useMonitoring(), { wrapper });

    await waitFor(() => {
      expect(result.current.error).toBeTruthy();
    });

    expect(result.current.isConnected).toBe(false);
  });
});