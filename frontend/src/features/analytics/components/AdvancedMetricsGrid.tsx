/**
 * Grille de Métriques Avancées - M&A Intelligence Platform
 * Sprint 6 - Affichage sophistiqué des KPIs avec interactions
 */

import React, { useState, useMemo } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { 
  TrendingUp, TrendingDown, Minus, AlertTriangle, 
  CheckCircle, Clock, DollarSign, Users, Activity,
  BarChart3, Zap, Target, Award, ArrowUpRight,
  ArrowDownRight, Maximize2, MoreHorizontal,
  RefreshCw, Filter, Settings
} from 'lucide-react';
import { AnalyticsMetric, METRIC_CATEGORIES } from '../types';
import { motion, AnimatePresence } from 'framer-motion';
import { cn } from '../../../lib/utils';

interface AdvancedMetricsGridProps {
  metrics: AnalyticsMetric[];
  className?: string;
  layout?: 'grid' | 'compact' | 'detailed';
  showTrends?: boolean;
  showThresholds?: boolean;
  onMetricClick?: (metric: AnalyticsMetric) => void;
  onMetricDrill?: (metric: AnalyticsMetric) => void;
}

// Composant pour un métrique individuel
const MetricCard: React.FC<{
  metric: AnalyticsMetric;
  layout: 'grid' | 'compact' | 'detailed';
  showTrends: boolean;
  showThresholds: boolean;
  onClick?: () => void;
  onDrill?: () => void;
}> = ({ metric, layout, showTrends, showThresholds, onClick, onDrill }) => {
  
  // Formatage de la valeur selon l'unité
  const formatValue = (value: number, unit: string) => {
    switch (unit) {
      case 'currency':
        return new Intl.NumberFormat('fr-FR', {
          style: 'currency',
          currency: 'EUR',
          minimumFractionDigits: 0,
          maximumFractionDigits: 0
        }).format(value);
      case 'percentage':
        return `${value.toFixed(1)}%`;
      case 'duration':
        return `${value.toLocaleString()}ms`;
      default:
        return value.toLocaleString('fr-FR');
    }
  };

  // Icône selon la catégorie
  const getCategoryIcon = (category: string) => {
    switch (category) {
      case 'business': return <BarChart3 className="h-4 w-4" />;
      case 'financial': return <DollarSign className="h-4 w-4" />;
      case 'user': return <Users className="h-4 w-4" />;
      case 'technical': return <Zap className="h-4 w-4" />;
      default: return <Activity className="h-4 w-4" />;
    }
  };

  // Icône et couleur de tendance
  const getTrendInfo = (trend: string, value: number) => {
    switch (trend) {
      case 'up':
        return {
          icon: <TrendingUp className="h-3 w-3" />,
          color: 'text-green-600',
          bgColor: 'bg-green-100',
          sign: '+'
        };
      case 'down':
        return {
          icon: <TrendingDown className="h-3 w-3" />,
          color: 'text-red-600',
          bgColor: 'bg-red-100',
          sign: ''
        };
      default:
        return {
          icon: <Minus className="h-3 w-3" />,
          color: 'text-gray-600',
          bgColor: 'bg-gray-100',
          sign: ''
        };
    }
  };

  // Statut basé sur les seuils
  const getThresholdStatus = () => {
    if (!metric.threshold) return { status: 'normal', color: 'bg-green-500' };
    
    if (metric.value <= metric.threshold.critical) {
      return { status: 'critical', color: 'bg-red-500' };
    } else if (metric.value <= metric.threshold.warning) {
      return { status: 'warning', color: 'bg-yellow-500' };
    }
    return { status: 'good', color: 'bg-green-500' };
  };

  const trendInfo = getTrendInfo(metric.trend, metric.trendValue);
  const thresholdStatus = getThresholdStatus();
  const categoryInfo = METRIC_CATEGORIES[metric.category as keyof typeof METRIC_CATEGORIES];

  // Calcul du pourcentage de seuil pour la barre de progression
  const getThresholdProgress = () => {
    if (!metric.threshold) return 100;
    const max = Math.max(metric.threshold.warning, metric.threshold.critical, metric.value) * 1.2;
    return (metric.value / max) * 100;
  };

  if (layout === 'compact') {
    return (
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3 }}
        className="flex items-center justify-between p-3 border rounded-lg hover:bg-muted/50 cursor-pointer"
        onClick={onClick}
      >
        <div className="flex items-center gap-3">
          <div className={cn("p-2 rounded-lg", categoryInfo?.color ? `bg-${categoryInfo.color}-100` : 'bg-gray-100')}>
            {getCategoryIcon(metric.category)}
          </div>
          <div>
            <p className="font-medium text-sm">{metric.name}</p>
            <p className="text-xl font-bold">{formatValue(metric.value, metric.unit)}</p>
          </div>
        </div>
        
        {showTrends && (
          <div className={cn("flex items-center gap-1 px-2 py-1 rounded-full text-xs", trendInfo.bgColor, trendInfo.color)}>
            {trendInfo.icon}
            <span>{trendInfo.sign}{Math.abs(metric.trendValue).toFixed(1)}%</span>
          </div>
        )}
      </motion.div>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.3 }}
    >
      <Card 
        className={cn(
          "cursor-pointer transition-all duration-200 hover:shadow-md hover:scale-[1.02]",
          metric.priority === 'high' && "ring-2 ring-primary/20"
        )}
        onClick={onClick}
      >
        <CardHeader className={cn(
          "pb-2",
          layout === 'detailed' ? "pb-3" : ""
        )}>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <div className={cn(
                "p-2 rounded-lg", 
                categoryInfo?.color ? `bg-${categoryInfo.color}-100` : 'bg-gray-100'
              )}>
                {getCategoryIcon(metric.category)}
              </div>
              
              {layout === 'detailed' && (
                <div>
                  <Badge variant="outline" className="text-xs">
                    {categoryInfo?.label || metric.category}
                  </Badge>
                </div>
              )}
            </div>
            
            <div className="flex items-center gap-1">
              {metric.priority === 'high' && (
                <AlertTriangle className="h-4 w-4 text-orange-500" />
              )}
              
              {showThresholds && metric.threshold && (
                <div className={cn("w-2 h-2 rounded-full", thresholdStatus.color)} />
              )}
              
              <Button variant="ghost" size="sm" className="h-6 w-6 p-0" onClick={(e) => {
                e.stopPropagation();
                onDrill?.();
              }}>
                <MoreHorizontal className="h-3 w-3" />
              </Button>
            </div>
          </div>
          
          <CardTitle className="text-sm font-medium text-muted-foreground">
            {metric.name}
          </CardTitle>
        </CardHeader>

        <CardContent className="pt-0">
          <div className="space-y-3">
            {/* Valeur principale */}
            <div className="flex items-baseline justify-between">
              <span className="text-2xl font-bold">
                {formatValue(metric.value, metric.unit)}
              </span>
              
              {showTrends && (
                <div className={cn(
                  "flex items-center gap-1 px-2 py-1 rounded-full text-xs font-medium",
                  trendInfo.bgColor,
                  trendInfo.color
                )}>
                  {trendInfo.icon}
                  <span>{trendInfo.sign}{Math.abs(metric.trendValue).toFixed(1)}%</span>
                </div>
              )}
            </div>

            {/* Comparaison avec valeur précédente */}
            {metric.previousValue && layout === 'detailed' && (
              <div className="text-xs text-muted-foreground">
                Précédent: {formatValue(metric.previousValue, metric.unit)}
                <span className={cn("ml-2", trendInfo.color)}>
                  ({trendInfo.sign}{Math.abs(metric.value - metric.previousValue).toLocaleString()})
                </span>
              </div>
            )}

            {/* Barre de progression pour les seuils */}
            {showThresholds && metric.threshold && layout === 'detailed' && (
              <div className="space-y-1">
                <div className="flex justify-between text-xs text-muted-foreground">
                  <span>Performance</span>
                  <span>{thresholdStatus.status}</span>
                </div>
                <Progress 
                  value={getThresholdProgress()} 
                  className="h-2"
                />
                <div className="flex justify-between text-xs text-muted-foreground">
                  <span>Critique: {formatValue(metric.threshold.critical, metric.unit)}</span>
                  <span>Alerte: {formatValue(metric.threshold.warning, metric.unit)}</span>
                </div>
              </div>
            )}

            {/* Dernière mise à jour */}
            {layout === 'detailed' && (
              <div className="flex items-center gap-1 text-xs text-muted-foreground">
                <Clock className="h-3 w-3" />
                <span>Mis à jour {new Date(metric.lastUpdated).toLocaleString('fr-FR')}</span>
              </div>
            )}
          </div>
        </CardContent>
      </Card>
    </motion.div>
  );
};

// Composant principal
const AdvancedMetricsGrid: React.FC<AdvancedMetricsGridProps> = ({
  metrics,
  className = "",
  layout = 'grid',
  showTrends = true,
  showThresholds = true,
  onMetricClick,
  onMetricDrill
}) => {
  const [filter, setFilter] = useState<string>('all');
  const [sortBy, setSortBy] = useState<'name' | 'value' | 'trend' | 'priority'>('priority');
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('desc');

  // Filtrage et tri des métriques
  const processedMetrics = useMemo(() => {
    let filtered = metrics;

    // Filtrage par catégorie
    if (filter !== 'all') {
      filtered = filtered.filter(metric => metric.category === filter);
    }

    // Tri
    filtered.sort((a, b) => {
      let compareValue = 0;
      
      switch (sortBy) {
        case 'name':
          compareValue = a.name.localeCompare(b.name);
          break;
        case 'value':
          compareValue = a.value - b.value;
          break;
        case 'trend':
          compareValue = a.trendValue - b.trendValue;
          break;
        case 'priority':
          const priorityOrder = { high: 3, medium: 2, low: 1 };
          compareValue = priorityOrder[a.priority] - priorityOrder[b.priority];
          break;
      }

      return sortOrder === 'asc' ? compareValue : -compareValue;
    });

    return filtered;
  }, [metrics, filter, sortBy, sortOrder]);

  // Statistiques rapides
  const stats = useMemo(() => {
    const total = metrics.length;
    const trending_up = metrics.filter(m => m.trend === 'up').length;
    const critical = metrics.filter(m => 
      m.threshold && m.value <= m.threshold.critical
    ).length;
    const high_priority = metrics.filter(m => m.priority === 'high').length;

    return { total, trending_up, critical, high_priority };
  }, [metrics]);

  // Classes CSS selon le layout
  const getGridClasses = () => {
    switch (layout) {
      case 'compact':
        return "space-y-2";
      case 'detailed':
        return "grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6";
      default:
        return "grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4";
    }
  };

  return (
    <div className={className}>
      {/* Header avec filtres et stats */}
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-4">
          <h2 className="text-xl font-semibold">Métriques Analytiques</h2>
          
          {/* Stats rapides */}
          <div className="flex items-center gap-4 text-sm text-muted-foreground">
            <span className="flex items-center gap-1">
              <Target className="h-4 w-4" />
              {stats.total} métriques
            </span>
            <span className="flex items-center gap-1">
              <TrendingUp className="h-4 w-4 text-green-600" />
              {stats.trending_up} en hausse
            </span>
            {stats.critical > 0 && (
              <span className="flex items-center gap-1 text-red-600">
                <AlertTriangle className="h-4 w-4" />
                {stats.critical} critiques
              </span>
            )}
          </div>
        </div>

        {/* Contrôles */}
        <div className="flex items-center gap-2">
          {/* Filtre par catégorie */}
          <select
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            className="px-3 py-1 text-sm border rounded-lg bg-background"
          >
            <option value="all">Toutes catégories</option>
            {Object.entries(METRIC_CATEGORIES).map(([key, category]) => (
              <option key={key} value={key}>{category.label}</option>
            ))}
          </select>

          {/* Tri */}
          <select
            value={sortBy}
            onChange={(e) => setSortBy(e.target.value as any)}
            className="px-3 py-1 text-sm border rounded-lg bg-background"
          >
            <option value="priority">Priorité</option>
            <option value="name">Nom</option>
            <option value="value">Valeur</option>
            <option value="trend">Tendance</option>
          </select>

          <Button
            variant="ghost"
            size="sm"
            onClick={() => setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc')}
            className="p-2"
          >
            {sortOrder === 'asc' ? <ArrowUpRight className="h-4 w-4" /> : <ArrowDownRight className="h-4 w-4" />}
          </Button>
        </div>
      </div>

      {/* Grille de métriques */}
      <AnimatePresence mode="wait">
        <motion.div
          key={`${filter}-${sortBy}-${sortOrder}`}
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.2 }}
          className={getGridClasses()}
        >
          {processedMetrics.map((metric) => (
            <MetricCard
              key={metric.id}
              metric={metric}
              layout={layout}
              showTrends={showTrends}
              showThresholds={showThresholds}
              onClick={() => onMetricClick?.(metric)}
              onDrill={() => onMetricDrill?.(metric)}
            />
          ))}
        </motion.div>
      </AnimatePresence>

      {/* Message si aucune métrique */}
      {processedMetrics.length === 0 && (
        <div className="text-center py-12 text-muted-foreground">
          <BarChart3 className="h-12 w-12 mx-auto mb-4 opacity-50" />
          <p>Aucune métrique trouvée pour les filtres sélectionnés</p>
        </div>
      )}
    </div>
  );
};

export default AdvancedMetricsGrid;