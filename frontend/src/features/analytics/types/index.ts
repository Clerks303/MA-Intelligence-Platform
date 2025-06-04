/**
 * Types pour le module Analytics Avancé - M&A Intelligence Platform
 * Sprint 6 - Visualisations interactives et analytics sophistiqués
 */

export interface AnalyticsMetric {
  id: string;
  name: string;
  value: number;
  previousValue?: number;
  unit: 'number' | 'currency' | 'percentage' | 'duration';
  trend: 'up' | 'down' | 'stable';
  trendValue: number;
  category: 'business' | 'technical' | 'user' | 'financial';
  priority: 'high' | 'medium' | 'low';
  threshold?: {
    warning: number;
    critical: number;
  };
  lastUpdated: string;
}

export interface TimeSeriesData {
  timestamp: string;
  value: number;
  label?: string;
  metadata?: Record<string, any>;
}

export interface ChartData {
  id: string;
  title: string;
  type: 'line' | 'bar' | 'area' | 'pie' | 'scatter' | 'heatmap' | 'funnel' | 'treemap';
  data: TimeSeriesData[] | any[];
  config: ChartConfig;
  insights?: AnalyticsInsight[];
}

export interface ChartConfig {
  xAxis?: {
    label: string;
    type: 'datetime' | 'category' | 'linear';
    format?: string;
  };
  yAxis?: {
    label: string;
    type: 'linear' | 'logarithmic';
    format?: string;
    domain?: [number, number];
  };
  colors?: string[];
  interactive?: boolean;
  zoom?: boolean;
  brush?: boolean;
  tooltip?: {
    enabled: boolean;
    format?: string;
  };
  legend?: {
    enabled: boolean;
    position: 'top' | 'bottom' | 'left' | 'right';
  };
  animation?: {
    enabled: boolean;
    duration: number;
    easing?: string;
  };
}

export interface AnalyticsInsight {
  id: string;
  type: 'trend' | 'anomaly' | 'correlation' | 'prediction' | 'recommendation';
  title: string;
  description: string;
  confidence: number; // 0-1
  severity: 'info' | 'warning' | 'critical' | 'success';
  metadata?: Record<string, any>;
  timestamp: string;
}

export interface AnalyticsDashboard {
  id: string;
  name: string;
  description: string;
  sections: AnalyticsSection[];
  filters: AnalyticsFilter[];
  refreshInterval?: number;
  isPublic: boolean;
  owner: string;
  createdAt: string;
  updatedAt: string;
}

export interface AnalyticsSection {
  id: string;
  title: string;
  description?: string;
  layout: 'grid' | 'flow' | 'tabs' | 'accordion';
  widgets: AnalyticsWidget[];
  collapsible?: boolean;
  defaultExpanded?: boolean;
}

export interface AnalyticsWidget {
  id: string;
  type: 'metric' | 'chart' | 'table' | 'heatmap' | 'gauge' | 'scorecard' | 'timeline';
  title: string;
  description?: string;
  data: any;
  config: WidgetConfig;
  position: {
    x: number;
    y: number;
    w: number;
    h: number;
  };
  refreshRate?: number;
  visible: boolean;
}

export interface WidgetConfig {
  showHeader?: boolean;
  showFooter?: boolean;
  allowFullscreen?: boolean;
  allowExport?: boolean;
  customStyles?: Record<string, string>;
  interactionMode?: 'view' | 'edit' | 'explore';
}

export interface AnalyticsFilter {
  id: string;
  name: string;
  type: 'date' | 'select' | 'multiselect' | 'range' | 'search' | 'toggle';
  options?: FilterOption[];
  value: any;
  required?: boolean;
  visible: boolean;
}

export interface FilterOption {
  label: string;
  value: any;
  metadata?: Record<string, any>;
}

// Advanced Analytics Types

export interface PredictiveModel {
  id: string;
  name: string;
  type: 'regression' | 'classification' | 'clustering' | 'forecasting';
  status: 'training' | 'ready' | 'error' | 'deprecated';
  accuracy: number;
  lastTrained: string;
  features: ModelFeature[];
  predictions: PredictionResult[];
}

export interface ModelFeature {
  name: string;
  importance: number;
  type: 'numerical' | 'categorical' | 'boolean' | 'text';
  description?: string;
}

export interface PredictionResult {
  id: string;
  input: Record<string, any>;
  output: any;
  confidence: number;
  explanation?: string;
  timestamp: string;
}

export interface SegmentationAnalysis {
  id: string;
  name: string;
  method: 'kmeans' | 'hierarchical' | 'dbscan' | 'custom';
  segments: DataSegment[];
  features: string[];
  quality: {
    silhouetteScore: number;
    inertia: number;
    calinski: number;
  };
  insights: AnalyticsInsight[];
}

export interface DataSegment {
  id: string;
  name: string;
  size: number;
  characteristics: Record<string, any>;
  centroid: Record<string, number>;
  color: string;
}

export interface CorrelationMatrix {
  variables: string[];
  matrix: number[][];
  significant: boolean[][];
  pValues: number[][];
}

export interface AnomalyDetection {
  id: string;
  method: 'isolation_forest' | 'one_class_svm' | 'local_outlier' | 'statistical';
  anomalies: Anomaly[];
  threshold: number;
  score: number;
}

export interface Anomaly {
  id: string;
  timestamp: string;
  value: number;
  score: number; // Anomaly score
  features: Record<string, any>;
  explanation?: string;
}

// Real-time Analytics

export interface RealTimeMetrics {
  activeUsers: number;
  requestsPerSecond: number;
  responseTime: number;
  errorRate: number;
  dataProcessingRate: number;
  queueSize: number;
  lastUpdated: string;
}

export interface EventStream {
  id: string;
  type: string;
  timestamp: string;
  data: Record<string, any>;
  source: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
}

// Business Intelligence Types

export interface BusinessMetrics {
  revenue: RevenueMetrics;
  customer: CustomerMetrics;
  operational: OperationalMetrics;
  market: MarketMetrics;
}

export interface RevenueMetrics {
  total: number;
  growth: number;
  recurring: number;
  churn: number;
  arpu: number; // Average Revenue Per User
  ltv: number;  // Lifetime Value
}

export interface CustomerMetrics {
  total: number;
  new: number;
  active: number;
  retention: number;
  satisfaction: number;
  segments: CustomerSegment[];
}

export interface CustomerSegment {
  name: string;
  count: number;
  value: number;
  growth: number;
}

export interface OperationalMetrics {
  efficiency: number;
  quality: number;
  utilization: number;
  throughput: number;
  bottlenecks: Bottleneck[];
}

export interface Bottleneck {
  process: string;
  severity: number;
  impact: string;
  suggestion: string;
}

export interface MarketMetrics {
  share: number;
  growth: number;
  competition: CompetitorAnalysis[];
  trends: MarketTrend[];
}

export interface CompetitorAnalysis {
  name: string;
  share: number;
  growth: number;
  strengths: string[];
  weaknesses: string[];
}

export interface MarketTrend {
  category: string;
  direction: 'up' | 'down' | 'stable';
  strength: number;
  timeframe: string;
}

// Export and Reporting

export interface AnalyticsReport {
  id: string;
  title: string;
  description: string;
  type: 'pdf' | 'excel' | 'powerpoint' | 'html' | 'json';
  sections: ReportSection[];
  schedule?: ScheduleConfig;
  recipients?: string[];
  template: string;
  generatedAt: string;
}

export interface ReportSection {
  id: string;
  title: string;
  type: 'summary' | 'chart' | 'table' | 'text' | 'metrics';
  content: any;
  options: ReportSectionOptions;
}

export interface ReportSectionOptions {
  showTitle: boolean;
  pageBreak: boolean;
  styling: Record<string, any>;
}

export interface ScheduleConfig {
  enabled: boolean;
  frequency: 'daily' | 'weekly' | 'monthly' | 'quarterly';
  time: string;
  timezone: string;
  nextRun: string;
}

// Hooks and Service Types

export interface UseAnalyticsReturn {
  // Data
  metrics: AnalyticsMetric[];
  charts: ChartData[];
  insights: AnalyticsInsight[];
  businessMetrics: any;
  realTimeMetrics: RealTimeMetrics;
  
  // State
  isLoading: boolean;
  error: string | null;
  lastUpdated: string | null;
  
  // Actions
  refreshData: () => Promise<void>;
  exportData: (format: 'csv' | 'json' | 'excel') => Promise<void>;
  createAlert: (metric: string, condition: any) => Promise<void>;
  
  // Filters
  filters: AnalyticsFilter[];
  updateFilter: (id: string, value: any) => void;
  resetFilters: () => void;
}

export interface AnalyticsConfig {
  apiUrl: string;
  refreshInterval: number;
  maxDataPoints: number;
  enableRealTime: boolean;
  defaultFilters: AnalyticsFilter[];
  chartDefaults: Partial<ChartConfig>;
}

// Constants

export const CHART_COLORS = [
  '#3B82F6', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6',
  '#06B6D4', '#84CC16', '#F97316', '#EC4899', '#6366F1'
];

export const METRIC_CATEGORIES = {
  business: { label: 'Business', color: '#3B82F6' },
  technical: { label: 'Technique', color: '#10B981' },
  user: { label: 'Utilisateur', color: '#F59E0B' },
  financial: { label: 'Financier', color: '#EF4444' }
};

export const INSIGHT_TYPES = {
  trend: { label: 'Tendance', icon: 'TrendingUp' },
  anomaly: { label: 'Anomalie', icon: 'AlertTriangle' },
  correlation: { label: 'Corrélation', icon: 'GitBranch' },
  prediction: { label: 'Prédiction', icon: 'Crystal' },
  recommendation: { label: 'Recommandation', icon: 'Lightbulb' }
};

export default {
  AnalyticsMetric,
  ChartData,
  AnalyticsDashboard,
  AnalyticsWidget,
  PredictiveModel,
  BusinessMetrics,
  AnalyticsReport,
  CHART_COLORS,
  METRIC_CATEGORIES,
  INSIGHT_TYPES
};