/**
 * Widget Chart pour Dashboard - M&A Intelligence Platform
 * Sprint 2 - Graphiques interactifs avec Recharts
 */

import React, { useMemo } from 'react';
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import { Card, CardContent, CardHeader, CardTitle } from '../../../components/ui/card';
import { cn } from '../../../lib/utils';
import { DashboardChart } from '../types';
// Remplacement des fonctions date-fns par des utilitaires simples
const formatDate = (dateString: string, formatType: string) => {
  try {
    const date = new Date(dateString);
    if (formatType === 'dd MMM yyyy') {
      return date.toLocaleDateString('fr-FR', { day: '2-digit', month: 'short', year: 'numeric' });
    }
    if (formatType === 'dd/MM') {
      return date.toLocaleDateString('fr-FR', { day: '2-digit', month: '2-digit' });
    }
    return date.toLocaleDateString('fr-FR');
  } catch {
    return dateString;
  }
};

interface ChartWidgetProps {
  chart: DashboardChart;
  className?: string;
}

// Palette de couleurs M&A Intelligence
const CHART_COLORS = {
  primary: ['#2563eb', '#3b82f6', '#60a5fa', '#93c5fd'],
  success: ['#22c55e', '#4ade80', '#86efac', '#bbf7d0'],
  warning: ['#f59e0b', '#fbbf24', '#fcd34d', '#fde68a'],
  danger: ['#ef4444', '#f87171', '#fca5a5', '#fecaca'],
  mixed: ['#2563eb', '#22c55e', '#f59e0b', '#ef4444', '#8b5cf6', '#06b6d4'],
};

// Formatage des tooltips
const CustomTooltip = ({ active, payload, label, chart }: any) => {
  if (active && payload && payload.length) {
    return (
      <div className="bg-white p-3 border border-ma-slate-200 rounded-lg shadow-lg">
        <p className="text-sm font-medium text-ma-slate-900">
          {chart.config.xAxis === 'date' && label 
            ? formatDate(label, 'dd MMM yyyy')
            : label
          }
        </p>
        {payload.map((entry: any, index: number) => (
          <p key={index} className="text-sm" style={{ color: entry.color }}>
            {`${entry.name}: ${entry.value?.toLocaleString('fr-FR')}`}
          </p>
        ))}
      </div>
    );
  }
  return null;
};

// Formatage des axes
const formatXAxisTick = (tickItem: string, chart: DashboardChart) => {
  if (chart.config.xAxis === 'date') {
    try {
      return formatDate(tickItem, 'dd/MM');
    } catch {
      return tickItem;
    }
  }
  return tickItem;
};

const formatYAxisTick = (tickItem: number) => {
  if (tickItem >= 1000000) {
    return `${(tickItem / 1000000).toFixed(1)}M`;
  }
  if (tickItem >= 1000) {
    return `${(tickItem / 1000).toFixed(1)}k`;
  }
  return tickItem.toLocaleString('fr-FR');
};

export const ChartWidget: React.FC<ChartWidgetProps> = ({ chart, className }) => {
  const colors = useMemo(() => {
    if (chart.config.colors && chart.config.colors.length > 0) {
      return chart.config.colors;
    }
    return CHART_COLORS.primary;
  }, [chart.config.colors]);

  if (chart.loading) {
    return (
      <Card className={cn("h-96", className)}>
        <CardHeader>
          <div className="animate-pulse">
            <div className="h-6 bg-ma-slate-200 rounded w-32"></div>
          </div>
        </CardHeader>
        <CardContent>
          <div className="animate-pulse h-64 bg-ma-slate-200 rounded"></div>
        </CardContent>
      </Card>
    );
  }

  if (chart.error) {
    return (
      <Card className={cn("h-96", className)}>
        <CardHeader>
          <CardTitle className="text-ma-red-600">{chart.title}</CardTitle>
        </CardHeader>
        <CardContent className="flex items-center justify-center h-64">
          <div className="text-center">
            <svg className="w-12 h-12 text-ma-red-400 mx-auto mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <p className="text-ma-slate-600">Erreur de chargement du graphique</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  if (!chart.data || chart.data.length === 0) {
    return (
      <Card className={cn("h-96", className)}>
        <CardHeader>
          <CardTitle>{chart.title}</CardTitle>
        </CardHeader>
        <CardContent className="flex items-center justify-center h-64">
          <div className="text-center">
            <svg className="w-12 h-12 text-ma-slate-400 mx-auto mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
            </svg>
            <p className="text-ma-slate-600">Aucune donnée disponible</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  const renderChart = () => {
    const commonProps = {
      data: chart.data,
      margin: { top: 5, right: 30, left: 20, bottom: 5 },
    };

    switch (chart.type) {
      case 'line':
        return (
          <LineChart {...commonProps}>
            {chart.config.showGrid && <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />}
            <XAxis 
              dataKey={chart.config.xAxis || 'date'}
              tickFormatter={(tick) => formatXAxisTick(tick, chart)}
              stroke="#64748b"
              fontSize={12}
            />
            <YAxis 
              tickFormatter={formatYAxisTick}
              stroke="#64748b"
              fontSize={12}
            />
            <Tooltip content={<CustomTooltip chart={chart} />} />
            {chart.config.showLegend && <Legend />}
            <Line
              type="monotone"
              dataKey={chart.config.yAxis || 'value'}
              stroke={colors[0]}
              strokeWidth={2}
              dot={{ fill: colors[0], strokeWidth: 2, r: 4 }}
              activeDot={{ r: 6, stroke: colors[0], strokeWidth: 2 }}
            />
          </LineChart>
        );

      case 'area':
        return (
          <AreaChart {...commonProps}>
            {chart.config.showGrid && <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />}
            <XAxis 
              dataKey={chart.config.xAxis || 'date'}
              tickFormatter={(tick) => formatXAxisTick(tick, chart)}
              stroke="#64748b"
              fontSize={12}
            />
            <YAxis 
              tickFormatter={formatYAxisTick}
              stroke="#64748b"
              fontSize={12}
            />
            <Tooltip content={<CustomTooltip chart={chart} />} />
            {chart.config.showLegend && <Legend />}
            <Area
              type="monotone"
              dataKey={chart.config.yAxis || 'value'}
              stroke={colors[0]}
              fill={colors[0]}
              fillOpacity={0.2}
            />
          </AreaChart>
        );

      case 'bar':
        return (
          <BarChart {...commonProps}>
            {chart.config.showGrid && <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />}
            <XAxis 
              dataKey={chart.config.xAxis || 'date'}
              tickFormatter={(tick) => formatXAxisTick(tick, chart)}
              stroke="#64748b"
              fontSize={12}
            />
            <YAxis 
              tickFormatter={formatYAxisTick}
              stroke="#64748b"
              fontSize={12}
            />
            <Tooltip content={<CustomTooltip chart={chart} />} />
            {chart.config.showLegend && <Legend />}
            <Bar
              dataKey={chart.config.yAxis || 'value'}
              fill={colors[0]}
              radius={[2, 2, 0, 0]}
            />
          </BarChart>
        );

      case 'pie':
      case 'donut':
        return (
          <PieChart>
            <Pie
              data={chart.data}
              cx="50%"
              cy="50%"
              outerRadius={chart.type === 'donut' ? 100 : 120}
              innerRadius={chart.type === 'donut' ? 60 : 0}
              paddingAngle={2}
              dataKey="value"
              label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
            >
              {chart.data.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={colors[index % colors.length]} />
              ))}
            </Pie>
            <Tooltip content={<CustomTooltip chart={chart} />} />
          </PieChart>
        );

      default:
        return <div className="flex items-center justify-center h-64 text-ma-slate-500">Type de graphique non supporté</div>;
    }
  };

  return (
    <Card className={cn("", className)}>
      <CardHeader className="pb-2">
        <CardTitle className="text-lg font-semibold text-ma-slate-900">
          {chart.title}
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div style={{ height: chart.config.height || 300 }}>
          <ResponsiveContainer width="100%" height="100%">
            {renderChart()}
          </ResponsiveContainer>
        </div>
      </CardContent>
    </Card>
  );
};

// Composant pour une grille de graphiques
interface ChartGridProps {
  charts: DashboardChart[];
  className?: string;
}

export const ChartGrid: React.FC<ChartGridProps> = ({ charts, className }) => {
  return (
    <div className={cn("grid grid-cols-1 lg:grid-cols-2 gap-6", className)}>
      {charts.map((chart) => (
        <ChartWidget key={chart.id} chart={chart} />
      ))}
    </div>
  );
};