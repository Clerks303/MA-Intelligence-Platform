/**
 * Composant Graphique Interactif - M&A Intelligence Platform
 * Sprint 6 - Visualisations avancées avec interactions utilisateur
 */

import React, { useState, useMemo, useCallback } from 'react';
import {
  LineChart, Line, AreaChart, Area, BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  Brush, ReferenceLine, ReferenceDot, ReferenceArea,
  ScatterChart, Scatter, ComposedChart
} from 'recharts';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { 
  TrendingUp, TrendingDown, Maximize2, Download, Settings, 
  Play, Pause, RotateCcw, ZoomIn, ZoomOut, Filter,
  Info, AlertTriangle, CheckCircle, Clock
} from 'lucide-react';
import { ChartData, ChartConfig, AnalyticsInsight, CHART_COLORS } from '../types';
import { motion, AnimatePresence } from 'framer-motion';

interface InteractiveChartProps {
  data: ChartData;
  className?: string;
  height?: number;
  showControls?: boolean;
  showInsights?: boolean;
  onExport?: (format: 'png' | 'svg' | 'pdf') => void;
  onFullscreen?: () => void;
}

// Composant pour les insights du graphique
const ChartInsights: React.FC<{ insights: AnalyticsInsight[] }> = ({ insights }) => {
  if (!insights || insights.length === 0) return null;

  const getInsightIcon = (type: string) => {
    switch (type) {
      case 'trend': return <TrendingUp className="h-4 w-4" />;
      case 'anomaly': return <AlertTriangle className="h-4 w-4" />;
      case 'prediction': return <Clock className="h-4 w-4" />;
      default: return <Info className="h-4 w-4" />;
    }
  };

  const getInsightColor = (severity: string) => {
    switch (severity) {
      case 'critical': return 'destructive';
      case 'warning': return 'warning';
      case 'success': return 'success';
      default: return 'secondary';
    }
  };

  return (
    <div className="mt-4 space-y-2">
      <h4 className="text-sm font-medium text-muted-foreground">Insights Analytiques</h4>
      {insights.map((insight) => (
        <motion.div
          key={insight.id}
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="flex items-start gap-2 p-2 rounded-lg bg-muted/50"
        >
          <div className="flex-shrink-0 mt-0.5">
            {getInsightIcon(insight.type)}
          </div>
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 mb-1">
              <span className="text-sm font-medium">{insight.title}</span>
              <Badge variant={getInsightColor(insight.severity) as any} className="text-xs">
                {Math.round(insight.confidence * 100)}%
              </Badge>
            </div>
            <p className="text-xs text-muted-foreground">{insight.description}</p>
          </div>
        </motion.div>
      ))}
    </div>
  );
};

// Tooltip personnalisé pour les graphiques
const CustomTooltip = ({ active, payload, label, config }: any) => {
  if (!active || !payload || !payload.length) return null;

  return (
    <div className="bg-background border border-border rounded-lg shadow-lg p-3 min-w-[150px]">
      <p className="font-medium text-sm mb-2">{label}</p>
      {payload.map((entry: any, index: number) => (
        <div key={index} className="flex items-center gap-2 text-xs">
          <div 
            className="w-3 h-3 rounded-full"
            style={{ backgroundColor: entry.color }}
          />
          <span className="text-muted-foreground">{entry.name}:</span>
          <span className="font-medium">{entry.value?.toLocaleString()}</span>
        </div>
      ))}
    </div>
  );
};

const InteractiveChart: React.FC<InteractiveChartProps> = ({
  data,
  className = "",
  height = 400,
  showControls = true,
  showInsights = true,
  onExport,
  onFullscreen
}) => {
  // État local pour les contrôles
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentRange, setCurrentRange] = useState<[number, number] | null>(null);
  const [showGrid, setShowGrid] = useState(true);
  const [showLegend, setShowLegend] = useState(true);
  const [animationSpeed, setAnimationSpeed] = useState(1000);

  // Configuration du graphique avec defaults
  const config: ChartConfig = useMemo(() => ({
    interactive: true,
    zoom: false,
    brush: false,
    tooltip: { enabled: true },
    legend: { enabled: true, position: 'top' },
    animation: { enabled: true, duration: 1000, easing: 'ease-in-out' },
    ...data.config
  }), [data.config]);

  // Données formatées selon le type de graphique
  const formattedData = useMemo(() => {
    if (data.type === 'treemap' || data.type === 'pie') {
      return data.data;
    }
    
    // Pour les graphiques temporels, s'assurer que les données sont triées
    return data.data.sort((a: any, b: any) => {
      if (a.timestamp && b.timestamp) {
        return new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime();
      }
      return 0;
    });
  }, [data]);

  // Animation des données en temps réel
  const [displayData, setDisplayData] = useState(formattedData);
  
  React.useEffect(() => {
    if (isPlaying && Array.isArray(formattedData)) {
      let index = 0;
      const interval = setInterval(() => {
        if (index < formattedData.length) {
          setDisplayData(formattedData.slice(0, index + 1));
          index++;
        } else {
          setIsPlaying(false);
        }
      }, animationSpeed / formattedData.length);

      return () => clearInterval(interval);
    } else {
      setDisplayData(formattedData);
    }
  }, [isPlaying, formattedData, animationSpeed]);

  // Gestionnaires d'événements
  const handlePlayPause = useCallback(() => {
    setIsPlaying(prev => !prev);
  }, []);

  const handleReset = useCallback(() => {
    setIsPlaying(false);
    setDisplayData(formattedData);
    setCurrentRange(null);
  }, [formattedData]);

  const handleExport = useCallback((format: 'png' | 'svg' | 'pdf') => {
    if (onExport) {
      onExport(format);
    } else {
      // Implémentation par défaut pour export
      console.log(`Exporting chart as ${format}`);
    }
  }, [onExport]);

  // Rendu du graphique selon le type
  const renderChart = () => {
    const commonProps = {
      data: displayData,
      margin: { top: 5, right: 30, left: 20, bottom: 5 }
    };

    switch (data.type) {
      case 'line':
        return (
          <LineChart {...commonProps}>
            {showGrid && <CartesianGrid strokeDasharray="3 3" className="opacity-30" />}
            <XAxis 
              dataKey={config.xAxis?.type === 'datetime' ? 'timestamp' : 'label'}
              tickFormatter={config.xAxis?.type === 'datetime' ? 
                (value) => new Date(value).toLocaleDateString('fr-FR') : 
                undefined
              }
            />
            <YAxis 
              domain={config.yAxis?.domain}
              tickFormatter={(value) => value.toLocaleString()}
            />
            <Tooltip content={<CustomTooltip config={config} />} />
            {showLegend && <Legend />}
            {config.brush && <Brush dataKey="timestamp" height={30} />}
            <Line 
              type="monotone" 
              dataKey="value" 
              stroke={config.colors?.[0] || CHART_COLORS[0]}
              strokeWidth={2}
              dot={{ fill: config.colors?.[0] || CHART_COLORS[0], strokeWidth: 2, r: 4 }}
              activeDot={{ r: 6, stroke: config.colors?.[0] || CHART_COLORS[0], strokeWidth: 2 }}
              animationDuration={config.animation?.duration}
            />
          </LineChart>
        );

      case 'area':
        return (
          <AreaChart {...commonProps}>
            {showGrid && <CartesianGrid strokeDasharray="3 3" className="opacity-30" />}
            <XAxis 
              dataKey={config.xAxis?.type === 'datetime' ? 'timestamp' : 'label'}
              tickFormatter={config.xAxis?.type === 'datetime' ? 
                (value) => new Date(value).toLocaleDateString('fr-FR') : 
                undefined
              }
            />
            <YAxis />
            <Tooltip content={<CustomTooltip config={config} />} />
            {showLegend && <Legend />}
            <Area 
              type="monotone" 
              dataKey="value" 
              stroke={config.colors?.[0] || CHART_COLORS[0]}
              fill={config.colors?.[0] || CHART_COLORS[0]}
              fillOpacity={0.3}
              animationDuration={config.animation?.duration}
            />
          </AreaChart>
        );

      case 'bar':
        return (
          <BarChart {...commonProps}>
            {showGrid && <CartesianGrid strokeDasharray="3 3" className="opacity-30" />}
            <XAxis dataKey="label" />
            <YAxis />
            <Tooltip content={<CustomTooltip config={config} />} />
            {showLegend && <Legend />}
            <Bar 
              dataKey="value" 
              fill={config.colors?.[0] || CHART_COLORS[0]}
              animationDuration={config.animation?.duration}
            >
              {displayData.map((entry: any, index: number) => (
                <Cell key={`cell-${index}`} fill={entry.color || config.colors?.[index % (config.colors?.length || CHART_COLORS.length)] || CHART_COLORS[index % CHART_COLORS.length]} />
              ))}
            </Bar>
          </BarChart>
        );

      case 'pie':
        return (
          <PieChart {...commonProps}>
            <Pie
              data={displayData}
              cx="50%"
              cy="50%"
              outerRadius={Math.min(height / 3, 120)}
              fill="#8884d8"
              dataKey="value"
              label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
              animationDuration={config.animation?.duration}
            >
              {displayData.map((entry: any, index: number) => (
                <Cell key={`cell-${index}`} fill={config.colors?.[index % (config.colors?.length || CHART_COLORS.length)] || CHART_COLORS[index % CHART_COLORS.length]} />
              ))}
            </Pie>
            <Tooltip content={<CustomTooltip config={config} />} />
          </PieChart>
        );

      case 'scatter':
        return (
          <ScatterChart {...commonProps}>
            {showGrid && <CartesianGrid strokeDasharray="3 3" className="opacity-30" />}
            <XAxis dataKey="x" type="number" />
            <YAxis dataKey="y" type="number" />
            <Tooltip content={<CustomTooltip config={config} />} />
            <Scatter 
              data={displayData} 
              fill={config.colors?.[0] || CHART_COLORS[0]}
              animationDuration={config.animation?.duration}
            />
          </ScatterChart>
        );

      default:
        return (
          <div className="flex items-center justify-center h-full text-muted-foreground">
            Type de graphique non supporté: {data.type}
          </div>
        );
    }
  };

  return (
    <Card className={className}>
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="text-lg">{data.title}</CardTitle>
          
          {showControls && (
            <div className="flex items-center gap-1">
              {/* Contrôles d'animation */}
              <Button
                variant="ghost"
                size="sm"
                onClick={handlePlayPause}
                className="h-8 w-8 p-0"
              >
                {isPlaying ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
              </Button>
              
              <Button
                variant="ghost"
                size="sm"
                onClick={handleReset}
                className="h-8 w-8 p-0"
              >
                <RotateCcw className="h-4 w-4" />
              </Button>

              {/* Contrôles d'affichage */}
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setShowGrid(!showGrid)}
                className="h-8 w-8 p-0"
              >
                <Filter className="h-4 w-4" />
              </Button>

              {/* Export */}
              <Button
                variant="ghost"
                size="sm"
                onClick={() => handleExport('png')}
                className="h-8 w-8 p-0"
              >
                <Download className="h-4 w-4" />
              </Button>

              {/* Plein écran */}
              {onFullscreen && (
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={onFullscreen}
                  className="h-8 w-8 p-0"
                >
                  <Maximize2 className="h-4 w-4" />
                </Button>
              )}
            </div>
          )}
        </div>
      </CardHeader>

      <CardContent>
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.5 }}
          style={{ height }}
        >
          <ResponsiveContainer width="100%" height="100%">
            {renderChart()}
          </ResponsiveContainer>
        </motion.div>

        {/* Insights */}
        <AnimatePresence>
          {showInsights && data.insights && data.insights.length > 0 && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              transition={{ duration: 0.3 }}
            >
              <ChartInsights insights={data.insights} />
            </motion.div>
          )}
        </AnimatePresence>

        {/* Méta-informations */}
        <div className="flex items-center justify-between mt-4 pt-4 border-t text-xs text-muted-foreground">
          <div className="flex items-center gap-4">
            <span>Points de données: {Array.isArray(displayData) ? displayData.length : 'N/A'}</span>
            {isPlaying && (
              <span className="flex items-center gap-1">
                <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
                Animation en cours
              </span>
            )}
          </div>
          
          <div className="flex items-center gap-2">
            {config.interactive && <Badge variant="outline">Interactif</Badge>}
            {config.animation?.enabled && <Badge variant="outline">Animé</Badge>}
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

export default InteractiveChart;