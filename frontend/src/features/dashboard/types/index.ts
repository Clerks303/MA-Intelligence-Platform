/**
 * Types pour le module Dashboard - M&A Intelligence Platform
 * Sprint 2 - Dashboard Central
 */

export interface DashboardKPI {
  id: string;
  title: string;
  value: number | string;
  change: number;
  changeType: 'increase' | 'decrease' | 'neutral';
  icon: string;
  color: 'blue' | 'green' | 'red' | 'yellow' | 'slate';
  format: 'number' | 'currency' | 'percentage' | 'text';
  loading?: boolean;
  error?: string;
}

export interface ChartDataPoint {
  date: string;
  value: number;
  label?: string;
  category?: string;
}

export interface DashboardChart {
  id: string;
  title: string;
  type: 'line' | 'bar' | 'area' | 'pie' | 'donut';
  data: ChartDataPoint[];
  config: {
    xAxis?: string;
    yAxis?: string;
    colors?: string[];
    height?: number;
    showGrid?: boolean;
    showLegend?: boolean;
  };
  loading?: boolean;
  error?: string;
}

export interface AlertLevel {
  id: string;
  level: 'critical' | 'warning' | 'info' | 'success';
  message: string;
  timestamp: Date;
  action?: {
    label: string;
    href: string;
  };
  dismissed?: boolean;
}

export interface SLAIndicator {
  id: string;
  name: string;
  target: number;
  current: number;
  unit: string;
  status: 'excellent' | 'good' | 'warning' | 'critical';
  trend: 'up' | 'down' | 'stable';
  lastUpdate: Date;
}

export interface DataQualityMetric {
  id: string;
  source: string;
  completeness: number; // 0-100
  accuracy: number; // 0-100
  freshness: number; // hours since last update
  consistency: number; // 0-100
  overall: number; // 0-100
  issues: Array<{
    type: 'missing' | 'invalid' | 'outdated' | 'duplicate';
    count: number;
    severity: 'high' | 'medium' | 'low';
  }>;
}

export interface DashboardWidget {
  id: string;
  type: 'kpi' | 'chart' | 'alert' | 'sla' | 'quality' | 'activity';
  title: string;
  gridPosition: {
    x: number;
    y: number;
    w: number; // width in grid units
    h: number; // height in grid units
  };
  config: Record<string, any>;
  visible: boolean;
  refreshInterval?: number; // in seconds
}

export interface DashboardLayout {
  id: string;
  name: string;
  widgets: DashboardWidget[];
  isDefault: boolean;
  createdAt: Date;
  updatedAt: Date;
}

export interface ActivityItem {
  id: string;
  type: 'scraping' | 'company_added' | 'status_change' | 'export' | 'alert';
  title: string;
  description: string;
  timestamp: Date;
  user?: {
    name: string;
    avatar?: string;
  };
  metadata?: Record<string, any>;
}

export interface DashboardFilters {
  dateRange: {
    start: Date;
    end: Date;
    preset?: 'today' | 'week' | 'month' | 'quarter' | 'year' | 'custom';
  };
  companies?: string[]; // Company IDs
  status?: string[];
  sources?: string[];
}

export interface DashboardState {
  layout: DashboardLayout;
  filters: DashboardFilters;
  refreshing: boolean;
  lastUpdate: Date;
  autoRefresh: boolean;
  autoRefreshInterval: number;
}

// API Response Types
export interface DashboardDataResponse {
  kpis: DashboardKPI[];
  charts: DashboardChart[];
  alerts: AlertLevel[];
  slaIndicators: SLAIndicator[];
  dataQuality: DataQualityMetric[];
  activity: ActivityItem[];
  summary: {
    totalCompanies: number;
    activeScrapings: number;
    completedToday: number;
    pendingActions: number;
  };
}