/**
 * Service API pour le Dashboard - M&A Intelligence Platform
 * Sprint 2 - Connexion Backend FastAPI
 */

import { 
  DashboardDataResponse, 
  DashboardFilters, 
  DashboardKPI, 
  DashboardChart,
  AlertLevel,
  SLAIndicator,
  DataQualityMetric,
  ActivityItem,
  ChartDataPoint
} from '../types';

// Base API URL (from environment)
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000/api/v1';

// API Client avec gestion d'erreurs
class DashboardApiClient {
  private async fetchWithAuth(endpoint: string, options: RequestInit = {}) {
    const token = localStorage.getItem('auth_token');
    
    const response = await fetch(`${API_BASE_URL}${endpoint}`, {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        'Authorization': token ? `Bearer ${token}` : '',
        ...options.headers,
      },
    });

    if (!response.ok) {
      throw new Error(`API Error: ${response.status} ${response.statusText}`);
    }

    return response.json();
  }

  // Formater les filtres pour l'API
  private formatFilters(filters: DashboardFilters) {
    const params = new URLSearchParams();
    
    if (filters.dateRange?.start) {
      params.append('start_date', filters.dateRange.start.toISOString().split('T')[0]);
    }
    if (filters.dateRange?.end) {
      params.append('end_date', filters.dateRange.end.toISOString().split('T')[0]);
    }
    if (filters.companies?.length) {
      params.append('companies', filters.companies.join(','));
    }
    if (filters.status?.length) {
      params.append('status', filters.status.join(','));
    }
    if (filters.sources?.length) {
      params.append('sources', filters.sources.join(','));
    }

    return params.toString();
  }

  // Récupérer toutes les données du dashboard
  async getDashboardData(filters: DashboardFilters): Promise<DashboardDataResponse> {
    const queryParams = this.formatFilters(filters);
    const data = await this.fetchWithAuth(`/stats/dashboard?${queryParams}`);
    
    return {
      kpis: this.transformKPIs(data.kpis || []),
      charts: this.transformCharts(data.charts || []),
      alerts: this.transformAlerts(data.alerts || []),
      slaIndicators: this.transformSLAIndicators(data.sla_indicators || []),
      dataQuality: this.transformDataQuality(data.data_quality || []),
      activity: this.transformActivity(data.activity || []),
      summary: data.summary || {
        totalCompanies: 0,
        activeScrapings: 0,
        completedToday: 0,
        pendingActions: 0,
      },
    };
  }

  // KPIs seulement (pour rafraîchissement rapide)
  async getKPIs(filters: DashboardFilters): Promise<DashboardKPI[]> {
    const queryParams = this.formatFilters(filters);
    const data = await this.fetchWithAuth(`/stats/kpis?${queryParams}`);
    return this.transformKPIs(data.kpis || []);
  }

  // Alertes en temps réel
  async getAlerts(): Promise<AlertLevel[]> {
    const data = await this.fetchWithAuth('/stats/alerts');
    return this.transformAlerts(data.alerts || []);
  }

  // Indicateurs SLA
  async getSLAIndicators(): Promise<SLAIndicator[]> {
    const data = await this.fetchWithAuth('/stats/sla');
    return this.transformSLAIndicators(data.sla_indicators || []);
  }

  // Qualité des données
  async getDataQuality(): Promise<DataQualityMetric[]> {
    const data = await this.fetchWithAuth('/stats/data-quality');
    return this.transformDataQuality(data.data_quality || []);
  }

  // Activité récente
  async getActivity(page: number = 1, limit: number = 20): Promise<ActivityItem[]> {
    const data = await this.fetchWithAuth(`/stats/activity?page=${page}&limit=${limit}`);
    return this.transformActivity(data.activity || []);
  }

  // Transformers pour adapter les données backend au frontend
  private transformKPIs(backendKPIs: any[]): DashboardKPI[] {
    return backendKPIs.map(kpi => ({
      id: kpi.id || kpi.key,
      title: kpi.title || kpi.name,
      value: kpi.value,
      change: kpi.change || 0,
      changeType: kpi.change > 0 ? 'increase' : kpi.change < 0 ? 'decrease' : 'neutral',
      icon: kpi.icon || 'trending-up',
      color: this.mapKPIColor(kpi.type),
      format: kpi.format || 'number',
    }));
  }

  private transformCharts(backendCharts: any[]): DashboardChart[] {
    return backendCharts.map(chart => ({
      id: chart.id,
      title: chart.title,
      type: chart.type || 'line',
      data: chart.data?.map((point: any) => ({
        date: point.date || point.x,
        value: point.value || point.y,
        label: point.label,
        category: point.category,
      })) || [],
      config: {
        xAxis: chart.config?.x_axis || 'date',
        yAxis: chart.config?.y_axis || 'value',
        colors: chart.config?.colors || ['#2563eb'],
        height: chart.config?.height || 300,
        showGrid: chart.config?.show_grid !== false,
        showLegend: chart.config?.show_legend !== false,
      },
    }));
  }

  private transformAlerts(backendAlerts: any[]): AlertLevel[] {
    return backendAlerts.map(alert => ({
      id: alert.id,
      level: alert.level || 'info',
      message: alert.message,
      timestamp: new Date(alert.timestamp || alert.created_at),
      action: alert.action ? {
        label: alert.action.label,
        href: alert.action.href || alert.action.url,
      } : undefined,
      dismissed: alert.dismissed || false,
    }));
  }

  private transformSLAIndicators(backendSLA: any[]): SLAIndicator[] {
    return backendSLA.map(sla => ({
      id: sla.id,
      name: sla.name,
      target: sla.target,
      current: sla.current,
      unit: sla.unit || '',
      status: this.mapSLAStatus(sla.current, sla.target),
      trend: sla.trend || 'stable',
      lastUpdate: new Date(sla.last_update || sla.updated_at),
    }));
  }

  private transformDataQuality(backendQuality: any[]): DataQualityMetric[] {
    return backendQuality.map(quality => ({
      id: quality.id,
      source: quality.source,
      completeness: quality.completeness || 0,
      accuracy: quality.accuracy || 0,
      freshness: quality.freshness || 0,
      consistency: quality.consistency || 0,
      overall: quality.overall || 0,
      issues: quality.issues?.map((issue: any) => ({
        type: issue.type,
        count: issue.count,
        severity: issue.severity,
      })) || [],
    }));
  }

  private transformActivity(backendActivity: any[]): ActivityItem[] {
    return backendActivity.map(activity => ({
      id: activity.id,
      type: activity.type,
      title: activity.title,
      description: activity.description,
      timestamp: new Date(activity.timestamp || activity.created_at),
      user: activity.user ? {
        name: activity.user.name || activity.user.username,
        avatar: activity.user.avatar,
      } : undefined,
      metadata: activity.metadata,
    }));
  }

  // Helpers
  private mapKPIColor(type: string): DashboardKPI['color'] {
    switch (type) {
      case 'revenue':
      case 'profit':
      case 'success':
        return 'green';
      case 'cost':
      case 'error':
      case 'failed':
        return 'red';
      case 'warning':
      case 'pending':
        return 'yellow';
      case 'primary':
      case 'total':
        return 'blue';
      default:
        return 'slate';
    }
  }

  private mapSLAStatus(current: number, target: number): SLAIndicator['status'] {
    const ratio = current / target;
    if (ratio >= 0.95) return 'excellent';
    if (ratio >= 0.85) return 'good';
    if (ratio >= 0.70) return 'warning';
    return 'critical';
  }
}

// Instance singleton du service
export const dashboardService = new DashboardApiClient();

// Mock Data pour développement (si pas d'API disponible)
export const mockDashboardData: DashboardDataResponse = {
  kpis: [
    {
      id: 'total-companies',
      title: 'Total Entreprises',
      value: 1247,
      change: 12,
      changeType: 'increase',
      icon: 'building',
      color: 'blue',
      format: 'number',
    },
    {
      id: 'active-scrapings',
      title: 'Scrapings Actifs',
      value: 23,
      change: 5,
      changeType: 'increase',
      icon: 'activity',
      color: 'green',
      format: 'number',
    },
    {
      id: 'conversion-rate',
      title: 'Taux Conversion',
      value: 8.4,
      change: -2.1,
      changeType: 'decrease',
      icon: 'trending-up',
      color: 'yellow',
      format: 'percentage',
    },
    {
      id: 'revenue-potential',
      title: 'CA Potentiel',
      value: 2450000,
      change: 18.5,
      changeType: 'increase',
      icon: 'euro',
      color: 'green',
      format: 'currency',
    },
  ],
  charts: [
    {
      id: 'companies-trend',
      title: 'Évolution Entreprises',
      type: 'line',
      data: Array.from({ length: 30 }, (_, i) => ({
        date: new Date(Date.now() - (29 - i) * 24 * 60 * 60 * 1000).toISOString().split('T')[0],
        value: Math.floor(Math.random() * 50) + 20 + i,
      })),
      config: {
        height: 300,
        colors: ['#2563eb'],
        showGrid: true,
        showLegend: false,
      },
    },
  ],
  alerts: [
    {
      id: 'alert-1',
      level: 'critical',
      message: 'Échec du scraping Société.com depuis 2h',
      timestamp: new Date(Date.now() - 2 * 60 * 60 * 1000),
      action: {
        label: 'Voir détails',
        href: '/scraping/alerts/1',
      },
    },
    {
      id: 'alert-2',
      level: 'warning',
      message: 'Quota API Pappers à 85%',
      timestamp: new Date(Date.now() - 30 * 60 * 1000),
    },
  ],
  slaIndicators: [
    {
      id: 'uptime',
      name: 'Uptime Scraping',
      target: 99.5,
      current: 98.7,
      unit: '%',
      status: 'good',
      trend: 'down',
      lastUpdate: new Date(),
    },
  ],
  dataQuality: [
    {
      id: 'pappers',
      source: 'Pappers API',
      completeness: 95,
      accuracy: 98,
      freshness: 2,
      consistency: 92,
      overall: 94,
      issues: [
        { type: 'missing', count: 12, severity: 'medium' },
        { type: 'outdated', count: 3, severity: 'low' },
      ],
    },
  ],
  activity: [
    {
      id: 'activity-1',
      type: 'company_added',
      title: 'Nouvelle entreprise ajoutée',
      description: 'ACME Corp a été ajoutée au portefeuille',
      timestamp: new Date(Date.now() - 15 * 60 * 1000),
      user: { name: 'Admin' },
    },
  ],
  summary: {
    totalCompanies: 1247,
    activeScrapings: 23,
    completedToday: 156,
    pendingActions: 8,
  },
};