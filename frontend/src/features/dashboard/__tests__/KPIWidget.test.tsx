/**
 * Tests unitaires pour KPIWidget
 * Sprint 2 - Dashboard Tests
 */

import React from 'react';
import { render, screen } from '@testing-library/react';
import '@testing-library/jest-dom';
import { KPIWidget, KPIGrid } from '../components/KPIWidget';
import { DashboardKPI } from '../types';

const mockKPI: DashboardKPI = {
  id: 'test-kpi',
  title: 'Test KPI',
  value: 1234,
  change: 12.5,
  changeType: 'increase',
  icon: 'building',
  color: 'blue',
  format: 'number',
};

const mockKPILoading: DashboardKPI = {
  ...mockKPI,
  loading: true,
};

const mockKPIError: DashboardKPI = {
  ...mockKPI,
  error: 'Test error message',
};

describe('KPIWidget', () => {
  it('renders KPI data correctly', () => {
    render(<KPIWidget kpi={mockKPI} />);
    
    expect(screen.getByText('Test KPI')).toBeInTheDocument();
    expect(screen.getByText('1,234')).toBeInTheDocument();
    expect(screen.getByText('12.5% ce mois')).toBeInTheDocument();
  });

  it('shows loading state', () => {
    render(<KPIWidget kpi={mockKPILoading} />);
    
    expect(screen.getByText('Test KPI')).toBeInTheDocument();
    // Check for loading skeleton
    expect(document.querySelector('.animate-pulse')).toBeInTheDocument();
  });

  it('shows error state', () => {
    render(<KPIWidget kpi={mockKPIError} />);
    
    expect(screen.getByText('Test KPI')).toBeInTheDocument();
    expect(screen.getByText('Erreur de chargement')).toBeInTheDocument();
  });

  it('formats currency correctly', () => {
    const currencyKPI: DashboardKPI = {
      ...mockKPI,
      value: 2450000,
      format: 'currency',
    };
    
    render(<KPIWidget kpi={currencyKPI} />);
    expect(screen.getByText('2 450 000 €')).toBeInTheDocument();
  });

  it('formats percentage correctly', () => {
    const percentageKPI: DashboardKPI = {
      ...mockKPI,
      value: 85.4,
      format: 'percentage',
    };
    
    render(<KPIWidget kpi={percentageKPI} />);
    expect(screen.getByText('85.4%')).toBeInTheDocument();
  });
});

describe('KPIGrid', () => {
  const mockKPIs: DashboardKPI[] = [
    { ...mockKPI, id: 'kpi1', title: 'KPI 1' },
    { ...mockKPI, id: 'kpi2', title: 'KPI 2' },
    { ...mockKPI, id: 'kpi3', title: 'KPI 3' },
  ];

  it('renders multiple KPIs', () => {
    render(<KPIGrid kpis={mockKPIs} />);
    
    expect(screen.getByText('KPI 1')).toBeInTheDocument();
    expect(screen.getByText('KPI 2')).toBeInTheDocument();
    expect(screen.getByText('KPI 3')).toBeInTheDocument();
  });

  it('renders empty grid when no KPIs', () => {
    render(<KPIGrid kpis={[]} />);
    
    // Grid should still render but be empty
    expect(document.querySelector('.grid')).toBeInTheDocument();
  });
});