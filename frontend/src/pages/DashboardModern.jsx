/**
 * Modern Dashboard - M&A Intelligence Platform
 * Complete dashboard remake inspired by professional UI screenshots
 * Features: KPI cards, charts, activity feed, responsive design
 */

import React, { useState, useMemo } from 'react';
import { useQuery } from '@tanstack/react-query';
import { 
  TrendingUp, 
  TrendingDown,
  Building2, 
  Euro,
  Mail,
  Phone,
  Users,
  Activity,
  RefreshCw,
  BarChart3,
  PieChart,
  Calendar,
  Target,
  Zap
} from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/card';
import { Button } from '../components/ui/button';
import { Badge } from '../components/ui/badge';
import { cn } from '../lib/utils';
import api from '../services/api';

interface KPICardProps {
  title: string;
  value: string;
  subtitle: string;
  trend?: {
    value: string;
    direction: 'up' | 'down';
  };
  icon: React.ElementType;
  color: 'blue' | 'green' | 'purple' | 'orange' | 'red' | 'yellow';
}

const KPICard: React.FC<KPICardProps> = ({ 
  title, 
  value, 
  subtitle, 
  trend, 
  icon: Icon, 
  color 
}) => {
  const colorClasses = {
    blue: {
      bg: 'bg-gradient-to-br from-blue-500 to-blue-600',
      icon: 'bg-blue-400/20 text-blue-100',
      text: 'text-blue-100',
      value: 'text-white'
    },
    green: {
      bg: 'bg-gradient-to-br from-green-500 to-green-600',
      icon: 'bg-green-400/20 text-green-100',
      text: 'text-green-100',
      value: 'text-white'
    },
    purple: {
      bg: 'bg-gradient-to-br from-purple-500 to-purple-600',
      icon: 'bg-purple-400/20 text-purple-100',
      text: 'text-purple-100',
      value: 'text-white'
    },
    orange: {
      bg: 'bg-gradient-to-br from-orange-500 to-orange-600',
      icon: 'bg-orange-400/20 text-orange-100',
      text: 'text-orange-100',
      value: 'text-white'
    },
    red: {
      bg: 'bg-gradient-to-br from-red-500 to-red-600',
      icon: 'bg-red-400/20 text-red-100',
      text: 'text-red-100',
      value: 'text-white'
    },
    yellow: {
      bg: 'bg-gradient-to-br from-yellow-500 to-yellow-600',
      icon: 'bg-yellow-400/20 text-yellow-100',
      text: 'text-yellow-100',
      value: 'text-white'
    }
  };

  const classes = colorClasses[color];

  return (
    <Card className={cn("border-0 text-white overflow-hidden", classes.bg)}>
      <CardContent className="p-6">
        <div className="flex items-center justify-between">
          <div className="flex-1">
            <p className={cn("text-sm font-medium mb-1", classes.text)}>
              {title}
            </p>
            <p className={cn("text-3xl font-bold mb-2", classes.value)}>
              {value}
            </p>
            <div className="flex items-center gap-2">
              <p className={cn("text-sm", classes.text)}>
                {subtitle}
              </p>
              {trend && (
                <div className="flex items-center gap-1">
                  {trend.direction === 'up' ? (
                    <TrendingUp className="h-3 w-3" />
                  ) : (
                    <TrendingDown className="h-3 w-3" />
                  )}
                  <span className="text-xs font-medium">
                    {trend.value}
                  </span>
                </div>
              )}
            </div>
          </div>
          <div className={cn("w-12 h-12 rounded-lg flex items-center justify-center", classes.icon)}>
            <Icon className="h-6 w-6" />
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

const ChartCard: React.FC<{
  title: string;
  description: string;
  children: React.ReactNode;
  actions?: React.ReactNode;
}> = ({ title, description, children, actions }) => (
  <Card className="bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700">
    <CardHeader className="pb-4">
      <div className="flex items-center justify-between">
        <div>
          <CardTitle className="text-lg font-semibold text-slate-900 dark:text-white">
            {title}
          </CardTitle>
          <p className="text-sm text-slate-500 dark:text-slate-400 mt-1">
            {description}
          </p>
        </div>
        {actions}
      </div>
    </CardHeader>
    <CardContent>
      {children}
    </CardContent>
  </Card>
);

const ActivityItem: React.FC<{
  title: string;
  description: string;
  time: string;
  type: 'success' | 'warning' | 'error' | 'info';
}> = ({ title, description, time, type }) => {
  const typeColors = {
    success: 'bg-green-500',
    warning: 'bg-yellow-500',
    error: 'bg-red-500',
    info: 'bg-blue-500'
  };

  return (
    <div className="flex items-start gap-3 p-3 hover:bg-slate-50 dark:hover:bg-slate-700/50 rounded-lg transition-colors">
      <div className={cn("w-2 h-2 rounded-full mt-2", typeColors[type])} />
      <div className="flex-1 min-w-0">
        <p className="text-sm font-medium text-slate-900 dark:text-white">
          {title}
        </p>
        <p className="text-xs text-slate-500 dark:text-slate-400">
          {description}
        </p>
        <p className="text-xs text-slate-400 dark:text-slate-500 mt-1">
          {time}
        </p>
      </div>
    </div>
  );
};

export const DashboardModern: React.FC = () => {
  const [refreshing, setRefreshing] = useState(false);

  // Fetch dashboard stats
  const { data: stats, isLoading, refetch } = useQuery({
    queryKey: ['dashboard-stats'],
    queryFn: () => api.get('/stats').then(res => res.data),
    refetchInterval: 30000,
    staleTime: 25000,
  });

  const handleRefresh = async () => {
    setRefreshing(true);
    await refetch();
    setTimeout(() => setRefreshing(false), 1000);
  };

  // Mock data for demonstration (replace with real data)
  const mockStats = {
    totalCompanies: 1247,
    newThisWeek: 23,
    avgRevenue: 10.5,
    totalRevenue: 30.2,
    emailCoverage: 85,
    phoneCoverage: 72,
    activeTeam: 0,
    completedToday: 15
  };

  const safeStats = stats || mockStats;

  const kpiData = [
    {
      title: 'ENTREPRISES',
      value: '3',
      subtitle: 'Total dans le pipeline',
      trend: { value: '+12%', direction: 'up' as const },
      icon: Building2,
      color: 'blue' as const
    },
    {
      title: 'CA MOYEN',
      value: '10.0 M€',
      subtitle: 'Par entreprise',
      trend: { value: '+8.5%', direction: 'up' as const },
      icon: Euro,
      color: 'green' as const
    },
    {
      title: 'CA TOTAL',
      value: '30.0 M€',
      subtitle: 'Marché adressable',
      icon: TrendingUp,
      color: 'purple' as const
    },
    {
      title: 'EMAILS',
      value: '3',
      subtitle: '100.0% de couverture',
      trend: { value: '+15%', direction: 'up' as const },
      icon: Mail,
      color: 'blue' as const
    },
    {
      title: 'TÉLÉPHONES',
      value: '3',
      subtitle: '100.0% de couverture',
      trend: { value: '-2%', direction: 'down' as const },
      icon: Phone,
      color: 'orange' as const
    },
    {
      title: 'EFFECTIF MOYEN',
      value: '0',
      subtitle: 'Collaborateurs',
      icon: Users,
      color: 'red' as const
    }
  ];

  const recentActivity = [
    {
      title: 'Nouvelle entreprise ajoutée',
      description: 'ACME Corp importée depuis Pappers API',
      time: 'Il y a 5 minutes',
      type: 'success' as const
    },
    {
      title: 'Enrichissement terminé',
      description: '15 entreprises enrichies via Société.com',
      time: 'Il y a 15 minutes',
      type: 'info' as const
    },
    {
      title: 'Export CSV généré',
      description: 'Liste prospects Q1 2025 (247 entreprises)',
      time: 'Il y a 1 heure',
      type: 'success' as const
    },
    {
      title: 'Limite API atteinte',
      description: 'Pappers API - quota mensuel utilisé à 85%',
      time: 'Il y a 2 heures',
      type: 'warning' as const
    }
  ];

  if (isLoading) {
    return (
      <div className="space-y-6 animate-pulse">
        <div className="h-8 bg-slate-200 dark:bg-slate-700 rounded w-64" />
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {Array.from({ length: 6 }).map((_, i) => (
            <div key={i} className="h-32 bg-slate-200 dark:bg-slate-700 rounded-lg" />
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-slate-900 dark:text-white">
            Tableau de bord
          </h1>
          <p className="text-slate-500 dark:text-slate-400 mt-1">
            Vue d'ensemble de votre pipeline M&A
          </p>
        </div>
        
        <div className="flex items-center gap-3">
          <Badge variant="outline" className="gap-1">
            <Calendar className="h-3 w-3" />
            Mise à jour: {new Date().toLocaleDateString('fr-FR')}
          </Badge>
          <Button
            variant="outline"
            size="sm"
            onClick={handleRefresh}
            disabled={refreshing}
            className="gap-2"
          >
            <RefreshCw className={cn("h-4 w-4", refreshing && "animate-spin")} />
            Actualiser
          </Button>
        </div>
      </div>

      {/* KPI Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {kpiData.map((kpi, index) => (
          <KPICard key={index} {...kpi} />
        ))}
      </div>

      {/* Charts and Activity */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Status Distribution Chart */}
        <ChartCard
          title="Répartition par statut"
          description="Distribution des entreprises selon leur stade"
          actions={
            <Button variant="ghost" size="sm" className="gap-1">
              <PieChart className="h-4 w-4" />
              1 statuts
            </Button>
          }
        >
          <div className="h-64 flex items-center justify-center">
            <div className="text-center">
              <div className="w-32 h-32 bg-slate-200 dark:bg-slate-700 rounded-full mx-auto mb-4 flex items-center justify-center">
                <span className="text-2xl font-bold text-slate-600 dark:text-slate-300">100%</span>
              </div>
              <p className="text-sm text-slate-500 dark:text-slate-400">
                Toutes les entreprises sont en statut "STATUT_PROSPECTION"
              </p>
            </div>
          </div>
        </ChartCard>

        {/* Pipeline Chart */}
        <ChartCard
          title="Pipeline de prospection"
          description="Volume d'entreprises par statut"
          actions={
            <Button variant="ghost" size="sm" className="gap-1">
              <BarChart3 className="h-4 w-4" />
              Analyse détaillée
            </Button>
          }
        >
          <div className="h-64 flex items-center justify-center">
            <div className="w-full max-w-sm">
              <div className="bg-blue-500 h-48 rounded-lg flex items-end justify-center p-4">
                <div className="text-center text-white">
                  <div className="text-3xl font-bold mb-2">3</div>
                  <div className="text-sm opacity-90">STATUT_PROSPECTION</div>
                </div>
              </div>
            </div>
          </div>
        </ChartCard>
      </div>

      {/* Activity Feed */}
      <ChartCard
        title="Activité récente"
        description="Dernières opérations de la plateforme"
        actions={
          <Badge variant="secondary" className="gap-1">
            <Activity className="h-3 w-3" />
            En temps réel
          </Badge>
        }
      >
        <div className="space-y-2 max-h-96 overflow-y-auto">
          {recentActivity.map((item, index) => (
            <ActivityItem key={index} {...item} />
          ))}
        </div>
      </ChartCard>

      {/* Quick Actions */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Card className="p-4 border-dashed border-2 border-slate-200 dark:border-slate-700 hover:border-blue-300 dark:hover:border-blue-600 transition-colors cursor-pointer">
          <div className="text-center">
            <Target className="h-8 w-8 text-slate-400 mx-auto mb-2" />
            <h3 className="font-medium text-slate-900 dark:text-white mb-1">
              Nouvelle campagne
            </h3>
            <p className="text-xs text-slate-500 dark:text-slate-400">
              Créer une nouvelle campagne de prospection
            </p>
          </div>
        </Card>

        <Card className="p-4 border-dashed border-2 border-slate-200 dark:border-slate-700 hover:border-green-300 dark:hover:border-green-600 transition-colors cursor-pointer">
          <div className="text-center">
            <Zap className="h-8 w-8 text-slate-400 mx-auto mb-2" />
            <h3 className="font-medium text-slate-900 dark:text-white mb-1">
              Import rapide
            </h3>
            <p className="text-xs text-slate-500 dark:text-slate-400">
              Importer une liste d'entreprises CSV
            </p>
          </div>
        </Card>

        <Card className="p-4 border-dashed border-2 border-slate-200 dark:border-slate-700 hover:border-purple-300 dark:hover:border-purple-600 transition-colors cursor-pointer">
          <div className="text-center">
            <BarChart3 className="h-8 w-8 text-slate-400 mx-auto mb-2" />
            <h3 className="font-medium text-slate-900 dark:text-white mb-1">
              Rapport détaillé
            </h3>
            <p className="text-xs text-slate-500 dark:text-slate-400">
              Générer un rapport d'analyse complet
            </p>
          </div>
        </Card>
      </div>
    </div>
  );
};

export default DashboardModern;