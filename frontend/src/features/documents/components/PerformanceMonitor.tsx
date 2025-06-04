/**
 * Moniteur de Performance - M&A Intelligence Platform
 * Sprint 3 - Surveillance performance en temps réel
 */

import React, { useState, useEffect, useCallback } from 'react';
import { Card, CardHeader, CardContent, CardTitle } from '../../../components/ui/card';
import { Button } from '../../../components/ui/button';
import { Badge } from '../../../components/ui/badge';
import { Progress } from '../../../components/ui/progress';
import { 
  Monitor, 
  Zap, 
  Database, 
  Clock, 
  TrendingUp,
  AlertTriangle,
  CheckCircle,
  RefreshCw,
  Eye,
  EyeOff
} from 'lucide-react';
import { cn } from '../../../lib/utils';

import { performanceMonitor, performanceCache, PERFORMANCE_CONFIG } from '../utils/performance';

interface PerformanceMetrics {
  pageLoad: number;
  domReady: number;
  memoryUsed: number;
  memoryTotal: number;
  memoryLimit: number;
  cache: {
    size: number;
    maxSize: number;
    usage: number;
  };
  renderTime?: number;
  bundleSize?: number;
  chunksLoaded?: string[];
}

interface PerformanceMonitorProps {
  enabled?: boolean;
  compact?: boolean;
  showDetails?: boolean;
  onMetricsUpdate?: (metrics: PerformanceMetrics) => void;
}

interface MetricCardProps {
  title: string;
  value: string | number;
  unit?: string;
  status: 'good' | 'warning' | 'critical';
  icon: React.ReactNode;
  description?: string;
}

// Composant de métrique individuelle
const MetricCard: React.FC<MetricCardProps> = ({
  title,
  value,
  unit = '',
  status,
  icon,
  description,
}) => {
  const statusColors = {
    good: 'border-green-200 bg-green-50 text-green-800',
    warning: 'border-yellow-200 bg-yellow-50 text-yellow-800',
    critical: 'border-red-200 bg-red-50 text-red-800',
  };

  const statusIcons = {
    good: <CheckCircle className="h-4 w-4 text-green-600" />,
    warning: <AlertTriangle className="h-4 w-4 text-yellow-600" />,
    critical: <AlertTriangle className="h-4 w-4 text-red-600" />,
  };

  return (
    <Card className={cn('relative', statusColors[status])}>
      <CardContent className="p-4">
        <div className="flex items-start justify-between">
          <div className="flex items-center gap-2">
            {icon}
            <div>
              <h4 className="text-sm font-medium">{title}</h4>
              <div className="flex items-baseline gap-1">
                <span className="text-2xl font-bold">
                  {typeof value === 'number' ? Math.round(value) : value}
                </span>
                {unit && <span className="text-sm opacity-75">{unit}</span>}
              </div>
              {description && (
                <p className="text-xs opacity-75 mt-1">{description}</p>
              )}
            </div>
          </div>
          <div className="absolute top-2 right-2">
            {statusIcons[status]}
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

// Composant principal
export const PerformanceMonitor: React.FC<PerformanceMonitorProps> = ({
  enabled = true,
  compact = false,
  showDetails = false,
  onMetricsUpdate,
}) => {
  const [metrics, setMetrics] = useState<PerformanceMetrics | null>(null);
  const [isVisible, setIsVisible] = useState(showDetails);
  const [isCollecting, setIsCollecting] = useState(false);
  const [history, setHistory] = useState<PerformanceMetrics[]>([]);

  // Collecte des métriques
  const collectMetrics = useCallback(() => {
    if (!enabled) return;

    setIsCollecting(true);
    performanceMonitor.start('metrics-collection');

    const metrics = performanceMonitor.getMetrics();
    if (metrics) {
      // Ajout de métriques spécifiques au module documents
      const documentMetrics: PerformanceMetrics = {
        ...metrics,
        renderTime: performance.now(),
        bundleSize: getBundleSize(),
        chunksLoaded: getLoadedChunks(),
      };

      setMetrics(documentMetrics);
      setHistory(prev => [...prev.slice(-19), documentMetrics]); // Garder 20 dernières métriques
      onMetricsUpdate?.(documentMetrics);
    }

    performanceMonitor.end('metrics-collection');
    setIsCollecting(false);
  }, [enabled, onMetricsUpdate]);

  // Obtenir la taille du bundle
  const getBundleSize = useCallback((): number => {
    if (typeof window === 'undefined' || !window.performance) return 0;
    
    const resources = window.performance.getEntriesByType('resource') as PerformanceResourceTiming[];
    return resources
      .filter(resource => resource.name.includes('documents'))
      .reduce((total, resource) => total + (resource.transferSize || 0), 0);
  }, []);

  // Obtenir les chunks chargés
  const getLoadedChunks = useCallback((): string[] => {
    if (typeof window === 'undefined') return [];
    
    // Détecter les chunks chargés via les scripts
    const scripts = Array.from(document.querySelectorAll('script[src]'));
    return scripts
      .map(script => script.getAttribute('src') || '')
      .filter(src => src.includes('documents'))
      .map(src => src.split('/').pop() || '');
  }, []);

  // Déterminer le statut d'une métrique
  const getMetricStatus = useCallback((value: number, thresholds: { warning: number; critical: number }): 'good' | 'warning' | 'critical' => {
    if (value >= thresholds.critical) return 'critical';
    if (value >= thresholds.warning) return 'warning';
    return 'good';
  }, []);

  // Collecte automatique
  useEffect(() => {
    if (!enabled) return;

    // Collecte initiale
    setTimeout(collectMetrics, 1000);

    // Collecte périodique
    const interval = setInterval(collectMetrics, 10000); // Toutes les 10 secondes
    return () => clearInterval(interval);
  }, [enabled, collectMetrics]);

  // Observer les changements de performance
  useEffect(() => {
    if (!enabled || typeof window === 'undefined') return;

    const observer = new PerformanceObserver((list) => {
      const entries = list.getEntries();
      entries.forEach(entry => {
        if (entry.entryType === 'measure' && entry.name.includes('documents')) {
          console.log(`Performance [${entry.name}]: ${entry.duration.toFixed(2)}ms`);
        }
      });
    });

    try {
      observer.observe({ entryTypes: ['measure', 'navigation'] });
      return () => observer.disconnect();
    } catch (error) {
      console.warn('PerformanceObserver not supported:', error);
    }
  }, [enabled]);

  if (!enabled || !metrics) return null;

  // Version compacte
  if (compact) {
    const overallStatus = getMetricStatus(
      metrics.pageLoad + metrics.memoryUsed / 1000000,
      { warning: 2000, critical: 5000 }
    );

    return (
      <div className="flex items-center gap-2">
        <Badge 
          variant={overallStatus === 'good' ? 'default' : 'destructive'}
          className="flex items-center gap-1"
        >
          <Monitor className="h-3 w-3" />
          Performance
        </Badge>
        {isCollecting && (
          <RefreshCw className="h-3 w-3 animate-spin text-blue-500" />
        )}
      </div>
    );
  }

  // Version détaillée
  return (
    <Card className="mb-4">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2 text-base">
            <Monitor className="h-5 w-5" />
            Performance Monitor
            {isCollecting && (
              <RefreshCw className="h-4 w-4 animate-spin text-blue-500" />
            )}
          </CardTitle>
          <div className="flex items-center gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={collectMetrics}
              disabled={isCollecting}
            >
              <RefreshCw className={cn("h-4 w-4", isCollecting && "animate-spin")} />
            </Button>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setIsVisible(!isVisible)}
            >
              {isVisible ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
            </Button>
          </div>
        </div>
      </CardHeader>

      {isVisible && (
        <CardContent className="space-y-4">
          {/* Métriques principales */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <MetricCard
              title="Temps de chargement"
              value={metrics.pageLoad}
              unit="ms"
              status={getMetricStatus(metrics.pageLoad, { warning: 2000, critical: 5000 })}
              icon={<Clock className="h-4 w-4" />}
              description="Temps total de chargement"
            />

            <MetricCard
              title="Mémoire utilisée"
              value={Math.round(metrics.memoryUsed / 1024 / 1024)}
              unit="MB"
              status={getMetricStatus(
                metrics.memoryUsed / metrics.memoryLimit, 
                { warning: 0.7, critical: 0.9 }
              )}
              icon={<Database className="h-4 w-4" />}
              description={`${Math.round((metrics.memoryUsed / metrics.memoryLimit) * 100)}% utilisé`}
            />

            <MetricCard
              title="Cache"
              value={metrics.cache.size}
              unit={`/${metrics.cache.maxSize}`}
              status={getMetricStatus(metrics.cache.usage, { warning: 70, critical: 90 })}
              icon={<Zap className="h-4 w-4" />}
              description={`${Math.round(metrics.cache.usage)}% utilisé`}
            />

            <MetricCard
              title="Bundle"
              value={Math.round((metrics.bundleSize || 0) / 1024)}
              unit="KB"
              status={getMetricStatus(
                (metrics.bundleSize || 0) / 1024, 
                { warning: 300, critical: 500 }
              )}
              icon={<TrendingUp className="h-4 w-4" />}
              description={`${metrics.chunksLoaded?.length || 0} chunks`}
            />
          </div>

          {/* Barre de progression mémoire */}
          <div className="space-y-2">
            <div className="flex items-center justify-between text-sm">
              <span>Utilisation mémoire</span>
              <span>{Math.round((metrics.memoryUsed / metrics.memoryLimit) * 100)}%</span>
            </div>
            <Progress 
              value={(metrics.memoryUsed / metrics.memoryLimit) * 100} 
              className="h-2"
            />
          </div>

          {/* Historique (graphique simple) */}
          {history.length > 1 && (
            <div className="space-y-2">
              <h4 className="text-sm font-medium">Tendance mémoire (20 derniers points)</h4>
              <div className="flex items-end gap-1 h-12">
                {history.map((point, index) => {
                  const height = (point.memoryUsed / point.memoryLimit) * 100;
                  const status = getMetricStatus(height, { warning: 70, critical: 90 });
                  
                  return (
                    <div
                      key={index}
                      className={cn(
                        "flex-1 rounded-t transition-all",
                        status === 'good' && "bg-green-500",
                        status === 'warning' && "bg-yellow-500",
                        status === 'critical' && "bg-red-500"
                      )}
                      style={{ height: `${Math.max(height / 2, 2)}%` }}
                      title={`${Math.round(height)}% - ${new Date(point.renderTime || 0).toLocaleTimeString()}`}
                    />
                  );
                })}
              </div>
            </div>
          )}

          {/* Actions */}
          <div className="flex items-center gap-2 pt-2 border-t">
            <Button
              variant="outline"
              size="sm"
              onClick={() => performanceCache.clear()}
            >
              Vider le cache
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={() => {
                if (typeof window !== 'undefined' && 'gc' in window) {
                  // @ts-ignore
                  window.gc?.();
                }
              }}
            >
              Nettoyer mémoire
            </Button>
            <div className="flex-1" />
            <span className="text-xs text-gray-500">
              Module Documents v3.0 - Performance optimisée
            </span>
          </div>
        </CardContent>
      )}
    </Card>
  );
};

export default PerformanceMonitor;