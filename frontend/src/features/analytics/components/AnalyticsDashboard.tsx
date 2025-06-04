/**
 * Dashboard Analytics Principal - M&A Intelligence Platform
 * Sprint 6 - Interface complète avec visualisations et interactions avancées
 */

import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { 
  BarChart3, TrendingUp, Users, DollarSign, Activity,
  RefreshCw, Download, Settings, Maximize2, Eye,
  Brain, Target, Zap, Clock, Filter, AlertTriangle,
  PlayCircle, PauseCircle, SkipBack
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

import AdvancedMetricsGrid from './AdvancedMetricsGrid';
import InteractiveChart from './InteractiveChart';
import { useAdvancedAnalytics, useRealTimeMetrics } from '../hooks/useAdvancedAnalytics';
import { AnalyticsMetric, ChartData, AnalyticsInsight } from '../types';

// Composant pour les insights en temps réel
const RealTimeInsights: React.FC<{ insights: AnalyticsInsight[] }> = ({ insights }) => {
  const [currentIndex, setCurrentIndex] = useState(0);

  useEffect(() => {
    if (insights.length > 1) {
      const interval = setInterval(() => {
        setCurrentIndex((prev) => (prev + 1) % insights.length);
      }, 5000); // Change toutes les 5 secondes

      return () => clearInterval(interval);
    }
  }, [insights.length]);

  if (!insights || insights.length === 0) {
    return (
      <Card>
        <CardContent className="py-8 text-center">
          <Brain className="h-12 w-12 mx-auto mb-4 text-muted-foreground opacity-50" />
          <p className="text-muted-foreground">Aucun insight disponible</p>
        </CardContent>
      </Card>
    );
  }

  const currentInsight = insights[currentIndex];

  const getInsightIcon = (type: string) => {
    switch (type) {
      case 'trend': return <TrendingUp className="h-5 w-5" />;
      case 'anomaly': return <AlertTriangle className="h-5 w-5" />;
      case 'prediction': return <Brain className="h-5 w-5" />;
      case 'recommendation': return <Target className="h-5 w-5" />;
      default: return <Activity className="h-5 w-5" />;
    }
  };

  const getInsightColor = (severity: string) => {
    switch (severity) {
      case 'critical': return 'border-red-500 bg-red-50';
      case 'warning': return 'border-yellow-500 bg-yellow-50';
      case 'success': return 'border-green-500 bg-green-50';
      default: return 'border-blue-500 bg-blue-50';
    }
  };

  return (
    <Card className={`${getInsightColor(currentInsight.severity)} border-2`}>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            {getInsightIcon(currentInsight.type)}
            <CardTitle className="text-lg">Insight IA</CardTitle>
            <Badge variant="outline" className="">
              {Math.round(currentInsight.confidence * 100)}% confiance
            </Badge>
          </div>
          
          {insights.length > 1 && (
            <div className="flex items-center gap-1">
              {insights.map((_, index) => (
                <div
                  key={index}
                  className={`w-2 h-2 rounded-full ${
                    index === currentIndex ? 'bg-primary' : 'bg-muted'
                  }`}
                />
              ))}
            </div>
          )}
        </div>
      </CardHeader>
      
      <CardContent>
        <AnimatePresence mode="wait">
          <motion.div
            key={currentIndex}
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -20 }}
            transition={{ duration: 0.3 }}
          >
            <h3 className="font-semibold mb-2">{currentInsight.title}</h3>
            <p className="text-sm text-muted-foreground mb-3">
              {currentInsight.description}
            </p>
            
            {currentInsight.metadata && (
              <div className="grid grid-cols-2 gap-4 text-xs">
                {Object.entries(currentInsight.metadata).map(([key, value]) => (
                  <div key={key} className="flex justify-between">
                    <span className="capitalize">{key.replace('_', ' ')}:</span>
                    <span className="font-medium">{String(value)}</span>
                  </div>
                ))}
              </div>
            )}
          </motion.div>
        </AnimatePresence>
      </CardContent>
    </Card>
  );
};

// Composant pour les métriques temps réel
const RealTimeStatus: React.FC = () => {
  const { metrics, isConnected } = useRealTimeMetrics(3000);

  if (!metrics) {
    return (
      <Card>
        <CardContent className="py-4">
          <div className="flex items-center gap-2 text-muted-foreground">
            <div className="w-2 h-2 bg-gray-400 rounded-full" />
            <span className="text-sm">Connexion en cours...</span>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="text-sm">Statut Temps Réel</CardTitle>
          <div className="flex items-center gap-2">
            <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-500' : 'bg-red-500'} animate-pulse`} />
            <span className="text-xs text-muted-foreground">
              {isConnected ? 'Connecté' : 'Déconnecté'}
            </span>
          </div>
        </div>
      </CardHeader>
      
      <CardContent className="space-y-3">
        <div className="grid grid-cols-2 gap-3 text-sm">
          <div className="flex justify-between">
            <span className="text-muted-foreground">Utilisateurs:</span>
            <span className="font-medium">{metrics.activeUsers}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-muted-foreground">Req/sec:</span>
            <span className="font-medium">{metrics.requestsPerSecond}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-muted-foreground">Latence:</span>
            <span className="font-medium">{metrics.responseTime}ms</span>
          </div>
          <div className="flex justify-between">
            <span className="text-muted-foreground">Erreurs:</span>
            <span className={`font-medium ${metrics.errorRate > 5 ? 'text-red-600' : 'text-green-600'}`}>
              {metrics.errorRate.toFixed(1)}%
            </span>
          </div>
        </div>
        
        <div className="pt-2 border-t text-xs text-muted-foreground">
          Dernière mise à jour: {new Date(metrics.lastUpdated).toLocaleTimeString('fr-FR')}
        </div>
      </CardContent>
    </Card>
  );
};

// Composant principal
const AnalyticsDashboard: React.FC = () => {
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [selectedMetric, setSelectedMetric] = useState<AnalyticsMetric | null>(null);
  const [fullscreenChart, setFullscreenChart] = useState<ChartData | null>(null);
  const [activeTab, setActiveTab] = useState('overview');

  // Hook principal pour analytics
  const {
    metrics,
    charts,
    insights,
    businessMetrics,
    isLoading,
    error,
    refreshData,
    exportData,
    filters,
    updateFilter,
    resetFilters
  } = useAdvancedAnalytics([], {
    enableRealTime: true,
    refreshInterval: 30000,
    autoRefresh: true
  });

  // Gestion du rafraîchissement
  const handleRefresh = async () => {
    setIsRefreshing(true);
    try {
      await refreshData();
    } catch (err) {
      console.error('Erreur lors du rafraîchissement:', err);
    } finally {
      setIsRefreshing(false);
    }
  };

  // Export des données
  const handleExport = async (format: 'csv' | 'json' | 'excel') => {
    try {
      await exportData(format);
    } catch (err) {
      console.error('Erreur lors de l\'export:', err);
    }
  };

  if (isLoading) {
    return (
      <div className="p-6 space-y-6">
        <div className="animate-pulse">
          <div className="h-8 bg-muted rounded w-64 mb-6"></div>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
            {Array.from({ length: 4 }).map((_, i) => (
              <div key={i} className="h-32 bg-muted rounded"></div>
            ))}
          </div>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="h-96 bg-muted rounded"></div>
            <div className="h-96 bg-muted rounded"></div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="p-6 space-y-6">
      {/* Header avec actions */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="flex items-center justify-between"
      >
        <div>
          <h1 className="text-3xl font-bold flex items-center gap-2">
            <BarChart3 className="h-8 w-8" />
            Analytics Avancé
          </h1>
          <p className="text-muted-foreground mt-1">
            Tableau de bord intelligent avec IA et insights temps réel
          </p>
        </div>

        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={handleRefresh}
            disabled={isRefreshing}
          >
            <RefreshCw className={`h-4 w-4 mr-2 ${isRefreshing ? 'animate-spin' : ''}`} />
            Actualiser
          </Button>
          
          <Button
            variant="outline"
            size="sm"
            onClick={() => handleExport('excel')}
          >
            <Download className="h-4 w-4 mr-2" />
            Exporter
          </Button>
          
          <Button variant="outline" size="sm">
            <Settings className="h-4 w-4 mr-2" />
            Configuration
          </Button>
        </div>
      </motion.div>

      {/* Statut temps réel et insights */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.5, delay: 0.1 }}
        >
          <RealTimeStatus />
        </motion.div>
        
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.2 }}
          className="lg:col-span-2"
        >
          <RealTimeInsights insights={insights} />
        </motion.div>
      </div>

      {/* Tabs principal */}
      <Tabs defaultValue="overview" value={activeTab} onValueChange={setActiveTab} className="space-y-6">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="overview" className="flex items-center gap-2">
            <Eye className="h-4 w-4" />
            Vue d'ensemble
          </TabsTrigger>
          <TabsTrigger value="metrics" className="flex items-center gap-2">
            <Target className="h-4 w-4" />
            Métriques
          </TabsTrigger>
          <TabsTrigger value="charts" className="flex items-center gap-2">
            <BarChart3 className="h-4 w-4" />
            Graphiques
          </TabsTrigger>
          <TabsTrigger value="business" className="flex items-center gap-2">
            <DollarSign className="h-4 w-4" />
            Business
          </TabsTrigger>
        </TabsList>

        {/* Vue d'ensemble */}
        <TabsContent value="overview" className="space-y-6">
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.5 }}
          >
            <AdvancedMetricsGrid
              metrics={metrics.slice(0, 8)} // Top 8 métriques
              layout="grid"
              showTrends={true}
              showThresholds={true}
              onMetricClick={setSelectedMetric}
              onMetricDrill={(metric) => console.log('Drill down:', metric)}
            />
          </motion.div>

          {/* Graphiques principaux */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {charts.slice(0, 2).map((chart, index) => (
              <motion.div
                key={chart.id}
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.5, delay: index * 0.1 }}
              >
                <InteractiveChart
                  data={chart}
                  showControls={true}
                  showInsights={true}
                  onFullscreen={() => setFullscreenChart(chart)}
                />
              </motion.div>
            ))}
          </div>
        </TabsContent>

        {/* Métriques détaillées */}
        <TabsContent value="metrics" className="space-y-6">
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.5 }}
          >
            <AdvancedMetricsGrid
              metrics={metrics}
              layout="detailed"
              showTrends={true}
              showThresholds={true}
              onMetricClick={setSelectedMetric}
              onMetricDrill={(metric) => console.log('Drill down:', metric)}
            />
          </motion.div>
        </TabsContent>

        {/* Graphiques interactifs */}
        <TabsContent value="charts" className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {charts.map((chart, index) => (
              <motion.div
                key={chart.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: index * 0.1 }}
              >
                <InteractiveChart
                  data={chart}
                  showControls={true}
                  showInsights={true}
                  onFullscreen={() => setFullscreenChart(chart)}
                />
              </motion.div>
            ))}
          </div>
        </TabsContent>

        {/* Métriques business */}
        <TabsContent value="business" className="space-y-6">
          {businessMetrics && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 0.5 }}
              className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6"
            >
              {/* Revenue */}
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm flex items-center gap-2">
                    <DollarSign className="h-4 w-4" />
                    Revenus
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-2">
                  <div className="text-2xl font-bold">
                    {businessMetrics.revenue.total.toLocaleString('fr-FR', {
                      style: 'currency',
                      currency: 'EUR'
                    })}
                  </div>
                  <div className="flex items-center gap-1 text-sm">
                    <TrendingUp className="h-3 w-3 text-green-600" />
                    <span className="text-green-600">+{businessMetrics.revenue.growth}%</span>
                    <span className="text-muted-foreground">vs mois dernier</span>
                  </div>
                </CardContent>
              </Card>

              {/* Customers */}
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm flex items-center gap-2">
                    <Users className="h-4 w-4" />
                    Clients
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-2">
                  <div className="text-2xl font-bold">
                    {businessMetrics.customer.total.toLocaleString()}
                  </div>
                  <div className="text-sm text-muted-foreground">
                    {businessMetrics.customer.active.toLocaleString()} actifs ({businessMetrics.customer.retention}% rétention)
                  </div>
                </CardContent>
              </Card>

              {/* Operational */}
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm flex items-center gap-2">
                    <Zap className="h-4 w-4" />
                    Opérationnel
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-2">
                  <div className="text-2xl font-bold">
                    {businessMetrics.operational.efficiency}%
                  </div>
                  <div className="text-sm text-muted-foreground">
                    Efficacité globale
                  </div>
                </CardContent>
              </Card>

              {/* Market */}
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm flex items-center gap-2">
                    <Target className="h-4 w-4" />
                    Marché
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-2">
                  <div className="text-2xl font-bold">
                    {businessMetrics.market.share}%
                  </div>
                  <div className="flex items-center gap-1 text-sm">
                    <TrendingUp className="h-3 w-3 text-green-600" />
                    <span className="text-green-600">+{businessMetrics.market.growth}%</span>
                    <span className="text-muted-foreground">croissance</span>
                  </div>
                </CardContent>
              </Card>
            </motion.div>
          )}
        </TabsContent>
      </Tabs>

      {/* Modale plein écran pour graphiques */}
      <AnimatePresence>
        {fullscreenChart && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black/50 z-50 flex items-center justify-center p-4"
            onClick={() => setFullscreenChart(null)}
          >
            <motion.div
              initial={{ scale: 0.9 }}
              animate={{ scale: 1 }}
              exit={{ scale: 0.9 }}
              className="bg-background rounded-lg shadow-xl max-w-6xl w-full max-h-full overflow-auto"
              onClick={(e) => e.stopPropagation()}
            >
              <div className="p-6">
                <div className="flex items-center justify-between mb-4">
                  <h2 className="text-xl font-semibold">{fullscreenChart.title}</h2>
                  <Button
                    variant="ghost"
                    onClick={() => setFullscreenChart(null)}
                  >
                    ✕
                  </Button>
                </div>
                <InteractiveChart
                  data={fullscreenChart}
                  height={600}
                  showControls={true}
                  showInsights={true}
                />
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default AnalyticsDashboard;