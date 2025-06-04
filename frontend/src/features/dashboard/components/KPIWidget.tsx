/**
 * Widget KPI pour Dashboard - M&A Intelligence Platform
 * Sprint 2 - Composants Dashboard
 */

import React from 'react';
import { Card, CardContent } from '../../../components/ui/card';
import { cn } from '../../../lib/utils';
import { DashboardKPI } from '../types';

interface KPIWidgetProps {
  kpi: DashboardKPI;
  className?: string;
}

const formatValue = (value: number | string, format: DashboardKPI['format']) => {
  if (typeof value === 'string') return value;
  
  switch (format) {
    case 'currency':
      return new Intl.NumberFormat('fr-FR', {
        style: 'currency',
        currency: 'EUR',
        minimumFractionDigits: 0,
        maximumFractionDigits: 0,
      }).format(value);
    
    case 'percentage':
      return `${value.toFixed(1)}%`;
    
    case 'number':
      return new Intl.NumberFormat('fr-FR').format(value);
    
    default:
      return value.toString();
  }
};

const getIcon = (iconName: string) => {
  const iconMap: Record<string, React.ReactNode> = {
    building: (
      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 21V5a2 2 0 00-2-2H7a2 2 0 00-2 2v16m14 0h2m-2 0h-5m-9 0H3m2 0h5M9 7h1m-1 4h1m4-4h1m-1 4h1m-5 10v-5a1 1 0 011-1h2a1 1 0 011 1v5m-4 0h4" />
      </svg>
    ),
    activity: (
      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
      </svg>
    ),
    'trending-up': (
      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
      </svg>
    ),
    euro: (
      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14.828 14.828a4 4 0 01-5.656 0M9 10h1m4 0h1m-6 4h1m4 0h1m2-7a9 9 0 11-18 0 9 9 0 0118 0z" />
      </svg>
    ),
    users: (
      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4.354a4 4 0 110 5.292M15 21H3v-1a6 6 0 0112 0v1zm0 0h6v-1a6 6 0 00-9-5.197m13.5-9a2.5 2.5 0 11-5 0 2.5 2.5 0 015 0z" />
      </svg>
    ),
    target: (
      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.746 0 3.332.477 4.5 1.253v13C19.832 18.477 18.246 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
      </svg>
    ),
  };
  
  return iconMap[iconName] || iconMap['trending-up'];
};

const getColorClasses = (color: DashboardKPI['color']) => {
  const colorMap = {
    blue: {
      icon: 'text-ma-blue-600 bg-ma-blue-100',
      change: {
        increase: 'text-ma-blue-600',
        decrease: 'text-ma-blue-600',
        neutral: 'text-ma-slate-500',
      },
    },
    green: {
      icon: 'text-ma-green-600 bg-ma-green-100',
      change: {
        increase: 'text-ma-green-600',
        decrease: 'text-ma-red-600',
        neutral: 'text-ma-slate-500',
      },
    },
    red: {
      icon: 'text-ma-red-600 bg-ma-red-100',
      change: {
        increase: 'text-ma-red-600',
        decrease: 'text-ma-green-600',
        neutral: 'text-ma-slate-500',
      },
    },
    yellow: {
      icon: 'text-yellow-600 bg-yellow-100',
      change: {
        increase: 'text-ma-green-600',
        decrease: 'text-ma-red-600',
        neutral: 'text-ma-slate-500',
      },
    },
    slate: {
      icon: 'text-ma-slate-600 bg-ma-slate-100',
      change: {
        increase: 'text-ma-green-600',
        decrease: 'text-ma-red-600',
        neutral: 'text-ma-slate-500',
      },
    },
  };
  
  return colorMap[color] || colorMap.slate;
};

const getChangeIcon = (changeType: DashboardKPI['changeType']) => {
  switch (changeType) {
    case 'increase':
      return (
        <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
        </svg>
      );
    case 'decrease':
      return (
        <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 17h8m0 0V9m0 8l-8-8-4 4-6-6" />
        </svg>
      );
    default:
      return (
        <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20 12H4" />
        </svg>
      );
  }
};

export const KPIWidget: React.FC<KPIWidgetProps> = ({ kpi, className }) => {
  const colors = getColorClasses(kpi.color);
  
  if (kpi.loading) {
    return (
      <Card className={cn("relative overflow-hidden", className)}>
        <CardContent className="p-6">
          <div className="animate-pulse">
            <div className="flex items-center justify-between space-y-0 pb-2">
              <div className="h-4 bg-ma-slate-200 rounded w-24"></div>
              <div className="h-8 w-8 bg-ma-slate-200 rounded"></div>
            </div>
            <div className="space-y-2">
              <div className="h-8 bg-ma-slate-200 rounded w-20"></div>
              <div className="h-3 bg-ma-slate-200 rounded w-16"></div>
            </div>
          </div>
        </CardContent>
      </Card>
    );
  }

  if (kpi.error) {
    return (
      <Card className={cn("relative overflow-hidden border-ma-red-200", className)}>
        <CardContent className="p-6">
          <div className="flex items-center justify-between space-y-0 pb-2">
            <h3 className="text-sm font-medium text-ma-slate-600">{kpi.title}</h3>
            <div className="h-8 w-8 rounded-full bg-ma-red-100 flex items-center justify-center">
              <svg className="w-4 h-4 text-ma-red-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            </div>
          </div>
          <div>
            <p className="text-xs text-ma-red-600">Erreur de chargement</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className={cn("relative overflow-hidden hover:shadow-md transition-shadow", className)}>
      <CardContent className="p-6">
        <div className="flex items-center justify-between space-y-0 pb-2">
          <h3 className="text-sm font-medium text-ma-slate-600">{kpi.title}</h3>
          <div className={cn("h-8 w-8 rounded-full flex items-center justify-center", colors.icon)}>
            {getIcon(kpi.icon)}
          </div>
        </div>
        <div>
          <div className="text-2xl font-bold text-ma-slate-900">
            {formatValue(kpi.value, kpi.format)}
          </div>
          {kpi.change !== 0 && (
            <div className={cn(
              "flex items-center text-xs font-medium mt-1",
              colors.change[kpi.changeType]
            )}>
              {getChangeIcon(kpi.changeType)}
              <span className="ml-1">
                {Math.abs(kpi.change)}% ce mois
              </span>
            </div>
          )}
        </div>
      </CardContent>
      
      {/* Subtle gradient overlay for visual appeal */}
      <div className="absolute inset-0 bg-gradient-to-br from-transparent to-white/5 pointer-events-none" />
    </Card>
  );
};

// Composant pour une grille de KPIs
interface KPIGridProps {
  kpis: DashboardKPI[];
  className?: string;
}

export const KPIGrid: React.FC<KPIGridProps> = ({ kpis, className }) => {
  return (
    <div className={cn(
      "grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4",
      className
    )}>
      {kpis.map((kpi) => (
        <KPIWidget key={kpi.id} kpi={kpi} />
      ))}
    </div>
  );
};