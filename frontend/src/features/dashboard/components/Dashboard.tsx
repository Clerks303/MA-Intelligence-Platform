/**
 * Dashboard Principal - M&A Intelligence Platform
 * Sprint 2 - Dashboard central avec tous les widgets
 */

import React, { useState, useMemo } from 'react';
import { Card, CardContent } from '../../../components/ui/card';
import { Button } from '../../../components/ui/button';
import { cn } from '../../../lib/utils';
import { useDashboardData } from '../hooks/useDashboardData';
import type { DashboardFilters, DashboardLayout as DashboardLayoutType } from '../types';
import { KPIGrid } from './KPIWidget';
import { ChartGrid } from './ChartWidget';
import { AlertsWidget } from './AlertsWidget';
import { SLAWidget, DataQualityWidget } from './SLAWidget';
import { DashboardLayout, ColumnsLayout } from './DashboardLayout';
import { mockDashboardData } from '../services/dashboardService';

interface DashboardProps {
  className?: string;
}

// Composant pour les filtres du dashboard
const DashboardFilters: React.FC<{
  filters: DashboardFilters;
  onChange: (filters: DashboardFilters) => void;
}> = ({ filters, onChange }) => {
  const presets = [
    { key: 'today', label: 'Aujourd\'hui' },
    { key: 'week', label: '7 jours' },
    { key: 'month', label: '30 jours' },
    { key: 'quarter', label: '3 mois' },
  ];

  const handlePresetChange = (preset: string) => {
    const now = new Date();
    let start = new Date();

    switch (preset) {
      case 'today':
        start.setHours(0, 0, 0, 0);
        break;
      case 'week':
        start.setDate(now.getDate() - 7);
        break;
      case 'month':
        start.setDate(now.getDate() - 30);
        break;
      case 'quarter':
        start.setMonth(now.getMonth() - 3);
        break;
      default:
        return;
    }

    onChange({
      ...filters,
      dateRange: {
        start,
        end: now,
        preset: preset as any,
      },
    });
  };

  return (
    <Card className="mb-6">
      <CardContent className="py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className="text-sm font-medium text-ma-slate-600">Période:</span>
            <div className="flex gap-1">
              {presets.map((preset) => (
                <Button
                  key={preset.key}
                  variant={filters.dateRange.preset === preset.key ? "ma" : "ghost"}
                  size="sm"
                  onClick={() => handlePresetChange(preset.key)}
                >
                  {preset.label}
                </Button>
              ))}
            </div>
          </div>

          <div className="flex items-center gap-2 text-sm text-ma-slate-600">
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <span>
              {filters.dateRange.start.toLocaleDateString('fr-FR')} - {filters.dateRange.end.toLocaleDateString('fr-FR')}
            </span>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

// Composant pour widget d'activité récente
const ActivityWidget: React.FC<{ activity: any[] }> = ({ activity }) => {
  if (activity.length === 0) {
    return (
      <Card>
        <CardContent className="py-8">
          <div className="text-center">
            <svg className="w-12 h-12 text-ma-slate-400 mx-auto mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <p className="text-ma-slate-600">Aucune activité récente</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardContent className="p-6">
        <h3 className="text-lg font-semibold text-ma-slate-900 mb-4">
          Activité Récente
        </h3>
        <div className="space-y-4">
          {activity.slice(0, 5).map((item) => (
            <div key={item.id} className="flex items-start gap-3">
              <div className="flex-shrink-0 w-8 h-8 bg-ma-blue-100 rounded-full flex items-center justify-center">
                <svg className="w-4 h-4 text-ma-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                </svg>
              </div>
              <div className="flex-1 min-w-0">
                <p className="text-sm font-medium text-ma-slate-900">
                  {item.title}
                </p>
                <p className="text-sm text-ma-slate-600">
                  {item.description}
                </p>
                <p className="text-xs text-ma-slate-500 mt-1">
                  {new Date(item.timestamp).toLocaleString('fr-FR')}
                </p>
              </div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
};

export const Dashboard: React.FC<DashboardProps> = ({ className }) => {
  // État local pour les filtres
  const [filters, setFilters] = useState<DashboardFilters>({
    dateRange: {
      start: new Date(Date.now() - 30 * 24 * 60 * 60 * 1000), // 30 jours
      end: new Date(),
      preset: 'month',
    },
  });

  // État pour le mode édition du layout
  const [isEditingLayout, setIsEditingLayout] = useState(false);

  // Layout par défaut
  const [dashboardLayout, setDashboardLayout] = useState<DashboardLayoutType>({
    id: 'default',
    name: 'Dashboard Principal',
    isDefault: true,
    createdAt: new Date(),
    updatedAt: new Date(),
    widgets: [
      {
        id: 'kpis',
        type: 'kpi',
        title: 'Indicateurs Clés',
        gridPosition: { x: 0, y: 0, w: 12, h: 4 },
        config: {},
        visible: true,
      },
      {
        id: 'charts',
        type: 'chart',
        title: 'Graphiques',
        gridPosition: { x: 0, y: 1, w: 8, h: 6 },
        config: {},
        visible: true,
      },
      {
        id: 'alerts',
        type: 'alert',
        title: 'Alertes',
        gridPosition: { x: 8, y: 1, w: 4, h: 6 },
        config: {},
        visible: true,
      },
      {
        id: 'sla',
        type: 'sla',
        title: 'SLA',
        gridPosition: { x: 0, y: 2, w: 6, h: 4 },
        config: {},
        visible: true,
      },
      {
        id: 'quality',
        type: 'quality',
        title: 'Qualité',
        gridPosition: { x: 6, y: 2, w: 6, h: 4 },
        config: {},
        visible: true,
      },
      {
        id: 'activity',
        type: 'activity',
        title: 'Activité',
        gridPosition: { x: 8, y: 3, w: 4, h: 6 },
        config: {},
        visible: true,
      },
    ],
  });

  // Récupération des données (utilise mock en cas d'erreur API)
  const {
    kpis,
    charts,
    alerts,
    slaIndicators,
    dataQuality,
    activity,
    summary,
    metrics,
    isLoading,
    error,
    refreshDashboard,
  } = useDashboardData(filters);

  // Utiliser les données mock si erreur ou pas de données
  const safeData = useMemo(() => {
    if (error || (!isLoading && kpis.length === 0)) {
      return mockDashboardData;
    }
    return {
      kpis,
      charts,
      alerts,
      slaIndicators,
      dataQuality,
      activity,
      summary,
    };
  }, [error, isLoading, kpis, charts, alerts, slaIndicators, dataQuality, activity, summary]);

  // État de chargement global
  if (isLoading && kpis.length === 0) {
    return (
      <div className={cn("space-y-6", className)}>
        {/* Skeleton loading */}
        <div className="animate-pulse">
          <div className="h-8 bg-ma-slate-200 rounded w-64 mb-6"></div>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
            {Array.from({ length: 4 }).map((_, i) => (
              <div key={i} className="h-32 bg-ma-slate-200 rounded"></div>
            ))}
          </div>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="h-96 bg-ma-slate-200 rounded"></div>
            <div className="h-96 bg-ma-slate-200 rounded"></div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className={cn("space-y-6", className)}>
      {/* Filtres */}
      <DashboardFilters filters={filters} onChange={setFilters} />

      {/* Layout principal avec drag & drop */}
      <DashboardLayout
        layout={dashboardLayout}
        onLayoutChange={setDashboardLayout}
        isEditing={isEditingLayout}
        onToggleEdit={() => setIsEditingLayout(!isEditingLayout)}
      >
        {/* KPIs */}
        <KPIGrid kpis={safeData.kpis} />

        {/* Layout en colonnes pour le contenu principal */}
        <ColumnsLayout
          leftColumn={
            <>
              {/* Graphiques */}
              {safeData.charts.length > 0 && (
                <ChartGrid charts={safeData.charts} />
              )}
              
              {/* SLA et Qualité en grille */}
              <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
                <SLAWidget indicators={safeData.slaIndicators} />
                <DataQualityWidget metrics={safeData.dataQuality} />
              </div>
            </>
          }
          rightColumn={
            <>
              {/* Alertes */}
              <AlertsWidget
                alerts={safeData.alerts}
                onDismissAlert={(id) => console.log('Dismiss alert:', id)}
                onViewAlert={(id) => console.log('View alert:', id)}
              />
              
              {/* Activité récente */}
              <ActivityWidget activity={safeData.activity} />
            </>
          }
        />
      </DashboardLayout>

      {/* Footer avec métriques rapides */}
      <Card className="bg-ma-slate-50 border-ma-slate-200">
        <CardContent className="py-4">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-center">
            <div>
              <div className="text-2xl font-bold text-ma-slate-900">
                {safeData.summary.totalCompanies.toLocaleString('fr-FR')}
              </div>
              <div className="text-sm text-ma-slate-600">Entreprises</div>
            </div>
            <div>
              <div className="text-2xl font-bold text-ma-blue-600">
                {safeData.summary.activeScrapings}
              </div>
              <div className="text-sm text-ma-slate-600">Scrapings Actifs</div>
            </div>
            <div>
              <div className="text-2xl font-bold text-ma-green-600">
                {safeData.summary.completedToday}
              </div>
              <div className="text-sm text-ma-slate-600">Complétés Aujourd'hui</div>
            </div>
            <div>
              <div className="text-2xl font-bold text-ma-red-600">
                {safeData.summary.pendingActions}
              </div>
              <div className="text-sm text-ma-slate-600">Actions Pendantes</div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};