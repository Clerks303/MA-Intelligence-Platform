/**
 * Service Analytics Avancé - M&A Intelligence Platform
 * Sprint 6 - Service pour récupération et traitement des données analytics
 */

import { 
  AnalyticsMetric, 
  ChartData, 
  AnalyticsInsight, 
  BusinessMetrics,
  RealTimeMetrics,
  PredictiveModel,
  SegmentationAnalysis,
  CorrelationMatrix,
  AnomalyDetection,
  AnalyticsReport,
  AnalyticsFilter,
  CHART_COLORS 
} from '../types';

// Configuration du service
const CONFIG = {
  API_BASE_URL: process.env.REACT_APP_API_URL || 'http://localhost:8000/api/v1',
  REFRESH_INTERVAL: 30000, // 30 secondes
  MAX_DATA_POINTS: 1000,
  CACHE_DURATION: 5 * 60 * 1000, // 5 minutes
};

// Cache simple pour optimiser les requêtes
const cache = new Map<string, { data: any; timestamp: number }>();

const getCachedData = <T>(key: string): T | null => {
  const cached = cache.get(key);
  if (cached && Date.now() - cached.timestamp < CONFIG.CACHE_DURATION) {
    return cached.data;
  }
  return null;
};

const setCachedData = <T>(key: string, data: T): void => {
  cache.set(key, { data, timestamp: Date.now() });
};

// Générateur de données mock sophistiquées
export const generateMockAnalyticsData = () => {
  const now = new Date();
  const days = 30;
  
  // Métriques principales
  const metrics: AnalyticsMetric[] = [
    {
      id: 'companies_total',
      name: 'Total Entreprises',
      value: 15420,
      previousValue: 14280,
      unit: 'number',
      trend: 'up',
      trendValue: 8.0,
      category: 'business',
      priority: 'high',
      threshold: { warning: 10000, critical: 5000 },
      lastUpdated: now.toISOString()
    },
    {
      id: 'revenue_monthly',
      name: 'Chiffre d\'Affaires Moyen',
      value: 2450000,
      previousValue: 2280000,
      unit: 'currency',
      trend: 'up',
      trendValue: 7.5,
      category: 'financial',
      priority: 'high',
      lastUpdated: now.toISOString()
    },
    {
      id: 'scraping_success_rate',
      name: 'Taux de Succès Scraping',
      value: 94.5,
      previousValue: 92.1,
      unit: 'percentage',
      trend: 'up',
      trendValue: 2.4,
      category: 'technical',
      priority: 'medium',
      threshold: { warning: 90, critical: 80 },
      lastUpdated: now.toISOString()
    },
    {
      id: 'data_quality_score',
      name: 'Score Qualité Données',
      value: 88.7,
      previousValue: 85.3,
      unit: 'percentage',
      trend: 'up',
      trendValue: 3.4,
      category: 'technical',
      priority: 'high',
      threshold: { warning: 85, critical: 75 },
      lastUpdated: now.toISOString()
    },
    {
      id: 'processing_time',
      name: 'Temps de Traitement Moyen',
      value: 1247,
      previousValue: 1580,
      unit: 'duration',
      trend: 'down',
      trendValue: -21.1,
      category: 'technical',
      priority: 'medium',
      lastUpdated: now.toISOString()
    },
    {
      id: 'user_engagement',
      name: 'Engagement Utilisateur',
      value: 76.3,
      previousValue: 71.8,
      unit: 'percentage',
      trend: 'up',
      trendValue: 4.5,
      category: 'user',
      priority: 'medium',
      lastUpdated: now.toISOString()
    }
  ];

  // Données de séries temporelles
  const generateTimeSeriesData = (baseValue: number, volatility: number = 0.1) => {
    return Array.from({ length: days }, (_, i) => {
      const date = new Date(now.getTime() - (days - i) * 24 * 60 * 60 * 1000);
      const trend = i * 0.02; // Tendance légèrement positive
      const noise = (Math.random() - 0.5) * volatility * baseValue;
      const seasonal = Math.sin((i / days) * 4 * Math.PI) * 0.05 * baseValue;
      
      return {
        timestamp: date.toISOString(),
        value: Math.max(0, baseValue + trend * baseValue + noise + seasonal),
        label: date.toLocaleDateString('fr-FR')
      };
    });
  };

  // Charts avec différents types de visualisations
  const charts: ChartData[] = [
    {
      id: 'companies_growth',
      title: 'Évolution du nombre d\'entreprises',
      type: 'area',
      data: generateTimeSeriesData(500, 0.15),
      config: {
        xAxis: { label: 'Date', type: 'datetime' },
        yAxis: { label: 'Nombre d\'entreprises', type: 'linear' },
        colors: [CHART_COLORS[0]],
        interactive: true,
        zoom: true,
        animation: { enabled: true, duration: 1000, easing: 'easeInOutCubic' }
      },
      insights: [
        {
          id: 'insight_1',
          type: 'trend',
          title: 'Croissance constante',
          description: 'Le nombre d\'entreprises augmente de façon constante avec une accélération récente.',
          confidence: 0.89,
          severity: 'success',
          timestamp: now.toISOString()
        }
      ]
    },
    {
      id: 'revenue_distribution',
      title: 'Distribution du Chiffre d\'Affaires',
      type: 'bar',
      data: [
        { label: '< 100K€', value: 3420, color: CHART_COLORS[0] },
        { label: '100K-500K€', value: 5680, color: CHART_COLORS[1] },
        { label: '500K-1M€', value: 4120, color: CHART_COLORS[2] },
        { label: '1M-5M€', value: 1890, color: CHART_COLORS[3] },
        { label: '> 5M€', value: 310, color: CHART_COLORS[4] }
      ],
      config: {
        xAxis: { label: 'Tranches de CA', type: 'category' },
        yAxis: { label: 'Nombre d\'entreprises', type: 'linear' },
        interactive: true,
        tooltip: { enabled: true, format: '{label}: {value} entreprises' }
      }
    },
    {
      id: 'performance_heatmap',
      title: 'Carte de Performance par Région',
      type: 'heatmap',
      data: [
        { x: 'Île-de-France', y: 'Scraping', value: 95.2 },
        { x: 'Île-de-France', y: 'Qualité', value: 88.7 },
        { x: 'Île-de-France', y: 'Vitesse', value: 92.1 },
        { x: 'Rhône-Alpes', y: 'Scraping', value: 93.8 },
        { x: 'Rhône-Alpes', y: 'Qualité', value: 85.3 },
        { x: 'Rhône-Alpes', y: 'Vitesse', value: 89.4 },
        { x: 'PACA', y: 'Scraping', value: 91.2 },
        { x: 'PACA', y: 'Qualité', value: 86.9 },
        { x: 'PACA', y: 'Vitesse', value: 87.6 },
        { x: 'Nord', y: 'Scraping', value: 89.7 },
        { x: 'Nord', y: 'Qualité', value: 82.1 },
        { x: 'Nord', y: 'Vitesse', value: 85.3 }
      ],
      config: {
        colors: ['#FEF3C7', '#FCD34D', '#F59E0B', '#D97706', '#92400E'],
        interactive: true,
        tooltip: { enabled: true, format: '{x} - {y}: {value}%' }
      }
    },
    {
      id: 'sector_analysis',
      title: 'Analyse par Secteur d\'Activité',
      type: 'treemap',
      data: [
        { name: 'Services', value: 4850, children: [
          { name: 'Conseil', value: 2100 },
          { name: 'IT', value: 1650 },
          { name: 'Finance', value: 1100 }
        ]},
        { name: 'Commerce', value: 3200, children: [
          { name: 'Retail', value: 1800 },
          { name: 'Grossiste', value: 900 },
          { name: 'E-commerce', value: 500 }
        ]},
        { name: 'Industrie', value: 2800, children: [
          { name: 'Manufacturing', value: 1500 },
          { name: 'Energie', value: 800 },
          { name: 'Chimie', value: 500 }
        ]},
        { name: 'Construction', value: 1900 },
        { name: 'Agriculture', value: 850 }
      ],
      config: {
        colors: CHART_COLORS,
        interactive: true,
        tooltip: { enabled: true }
      }
    },
    {
      id: 'predictive_analysis',
      title: 'Prédiction de Croissance (6 mois)',
      type: 'line',
      data: [
        ...generateTimeSeriesData(15420, 0.05).slice(-7), // Données historiques
        ...Array.from({ length: 6 }, (_, i) => { // Prédictions
          const date = new Date(now.getTime() + (i + 1) * 30 * 24 * 60 * 60 * 1000);
          return {
            timestamp: date.toISOString(),
            value: 15420 + (i + 1) * 380 + (Math.random() - 0.5) * 200,
            label: date.toLocaleDateString('fr-FR'),
            predicted: true
          };
        })
      ],
      config: {
        xAxis: { label: 'Date', type: 'datetime' },
        yAxis: { label: 'Nombre d\'entreprises', type: 'linear' },
        colors: [CHART_COLORS[0], CHART_COLORS[1]],
        interactive: true,
        brush: true,
        animation: { enabled: true, duration: 1500 }
      },
      insights: [
        {
          id: 'prediction_insight',
          type: 'prediction',
          title: 'Croissance prédite de 15%',
          description: 'Le modèle prédit une croissance de 15% sur les 6 prochains mois avec une confiance de 87%.',
          confidence: 0.87,
          severity: 'info',
          timestamp: now.toISOString()
        }
      ]
    }
  ];

  // Insights analytiques
  const insights: AnalyticsInsight[] = [
    {
      id: 'insight_anomaly_1',
      type: 'anomaly',
      title: 'Pic de trafic détecté',
      description: 'Un pic inhabituel d\'activité de scraping a été détecté le 28/05, 300% au-dessus de la normale.',
      confidence: 0.94,
      severity: 'warning',
      metadata: { 
        peakValue: 1500, 
        normalValue: 375, 
        duration: '2h15min',
        possibleCause: 'Campagne marketing'
      },
      timestamp: new Date(now.getTime() - 2 * 24 * 60 * 60 * 1000).toISOString()
    },
    {
      id: 'insight_correlation_1',
      type: 'correlation',
      title: 'Corrélation qualité-performance',
      description: 'Une forte corrélation (r=0.78) entre la qualité des données et le temps de traitement a été identifiée.',
      confidence: 0.85,
      severity: 'info',
      metadata: { 
        correlation: 0.78, 
        variables: ['data_quality', 'processing_time'],
        significance: 0.001
      },
      timestamp: new Date(now.getTime() - 12 * 60 * 60 * 1000).toISOString()
    },
    {
      id: 'insight_recommendation_1',
      type: 'recommendation',
      title: 'Optimisation suggérée',
      description: 'Implémenter un cache Redis pourrait réduire le temps de traitement de 35% selon nos simulations.',
      confidence: 0.79,
      severity: 'success',
      metadata: { 
        estimatedImprovement: 0.35,
        implementationCost: 'medium',
        timeline: '2-3 semaines'
      },
      timestamp: new Date(now.getTime() - 6 * 60 * 60 * 1000).toISOString()
    },
    {
      id: 'insight_trend_1',
      type: 'trend',
      title: 'Tendance saisonnière identifiée',
      description: 'Les données montrent une augmentation cyclique de 20% tous les trimestres.',
      confidence: 0.91,
      severity: 'info',
      metadata: { 
        cycle: 'quarterly',
        amplitude: 0.20,
        nextPeak: new Date(now.getTime() + 45 * 24 * 60 * 60 * 1000).toISOString()
      },
      timestamp: new Date(now.getTime() - 1 * 60 * 60 * 1000).toISOString()
    }
  ];

  // Métriques temps réel
  const realTimeMetrics: RealTimeMetrics = {
    activeUsers: Math.floor(Math.random() * 20) + 5,
    requestsPerSecond: Math.floor(Math.random() * 50) + 10,
    responseTime: Math.floor(Math.random() * 200) + 50,
    errorRate: Math.random() * 2,
    dataProcessingRate: Math.floor(Math.random() * 1000) + 200,
    queueSize: Math.floor(Math.random() * 100),
    lastUpdated: now.toISOString()
  };

  // Métriques business
  const businessMetrics: BusinessMetrics = {
    revenue: {
      total: 2450000,
      growth: 7.5,
      recurring: 1850000,
      churn: 3.2,
      arpu: 158.75,
      ltv: 4750.00
    },
    customer: {
      total: 15420,
      new: 342,
      active: 12890,
      retention: 87.3,
      satisfaction: 4.2,
      segments: [
        { name: 'Enterprise', count: 156, value: 850000, growth: 12.3 },
        { name: 'SMB', count: 1240, value: 620000, growth: 8.7 },
        { name: 'Startup', count: 2890, value: 280000, growth: 15.2 },
        { name: 'Freelance', count: 11134, value: 700000, growth: 5.1 }
      ]
    },
    operational: {
      efficiency: 89.2,
      quality: 88.7,
      utilization: 76.4,
      throughput: 1247,
      bottlenecks: [
        {
          process: 'Data Validation',
          severity: 0.7,
          impact: 'Ralentit le processing de 15%',
          suggestion: 'Paralléliser les validations'
        },
        {
          process: 'API Rate Limits',
          severity: 0.4,
          impact: 'Limite le débit de scraping',
          suggestion: 'Négocier des quotas plus élevés'
        }
      ]
    },
    market: {
      share: 12.8,
      growth: 18.5,
      competition: [
        { name: 'Competitor A', share: 23.1, growth: 8.2, strengths: ['Brand', 'Pricing'], weaknesses: ['Tech', 'Support'] },
        { name: 'Competitor B', share: 19.7, growth: 12.1, strengths: ['Features', 'UX'], weaknesses: ['Scale', 'Performance'] },
        { name: 'Competitor C', share: 15.9, growth: 6.8, strengths: ['Enterprise', 'Security'], weaknesses: ['Innovation', 'Agility'] }
      ],
      trends: [
        { category: 'AI Integration', direction: 'up', strength: 0.85, timeframe: '12 mois' },
        { category: 'Data Privacy', direction: 'up', strength: 0.92, timeframe: '6 mois' },
        { category: 'Real-time Processing', direction: 'up', strength: 0.76, timeframe: '18 mois' }
      ]
    }
  };

  return {
    metrics,
    charts,
    insights,
    realTimeMetrics,
    businessMetrics
  };
};

// Service Analytics API
export class AnalyticsService {
  private static instance: AnalyticsService;
  
  public static getInstance(): AnalyticsService {
    if (!AnalyticsService.instance) {
      AnalyticsService.instance = new AnalyticsService();
    }
    return AnalyticsService.instance;
  }

  // Récupération des métriques principales
  async getMetrics(filters?: AnalyticsFilter[]): Promise<AnalyticsMetric[]> {
    const cacheKey = `metrics_${JSON.stringify(filters)}`;
    const cached = getCachedData<AnalyticsMetric[]>(cacheKey);
    
    if (cached) {
      return cached;
    }

    try {
      // En production, appel API réel
      // const response = await fetch(`${CONFIG.API_BASE_URL}/analytics/metrics`, {
      //   method: 'POST',
      //   headers: { 'Content-Type': 'application/json' },
      //   body: JSON.stringify({ filters })
      // });
      // const data = await response.json();

      // Pour la démo, utiliser les données mock
      const mockData = generateMockAnalyticsData();
      setCachedData(cacheKey, mockData.metrics);
      return mockData.metrics;
    } catch (error) {
      console.error('Error fetching analytics metrics:', error);
      return generateMockAnalyticsData().metrics;
    }
  }

  // Récupération des graphiques
  async getCharts(filters?: AnalyticsFilter[]): Promise<ChartData[]> {
    const cacheKey = `charts_${JSON.stringify(filters)}`;
    const cached = getCachedData<ChartData[]>(cacheKey);
    
    if (cached) {
      return cached;
    }

    try {
      const mockData = generateMockAnalyticsData();
      setCachedData(cacheKey, mockData.charts);
      return mockData.charts;
    } catch (error) {
      console.error('Error fetching analytics charts:', error);
      return generateMockAnalyticsData().charts;
    }
  }

  // Récupération des insights
  async getInsights(): Promise<AnalyticsInsight[]> {
    const cacheKey = 'insights';
    const cached = getCachedData<AnalyticsInsight[]>(cacheKey);
    
    if (cached) {
      return cached;
    }

    try {
      const mockData = generateMockAnalyticsData();
      setCachedData(cacheKey, mockData.insights);
      return mockData.insights;
    } catch (error) {
      console.error('Error fetching analytics insights:', error);
      return generateMockAnalyticsData().insights;
    }
  }

  // Métriques temps réel
  async getRealTimeMetrics(): Promise<RealTimeMetrics> {
    try {
      // Pas de cache pour les données temps réel
      const mockData = generateMockAnalyticsData();
      return mockData.realTimeMetrics;
    } catch (error) {
      console.error('Error fetching real-time metrics:', error);
      return generateMockAnalyticsData().realTimeMetrics;
    }
  }

  // Métriques business
  async getBusinessMetrics(): Promise<BusinessMetrics> {
    const cacheKey = 'business_metrics';
    const cached = getCachedData<BusinessMetrics>(cacheKey);
    
    if (cached) {
      return cached;
    }

    try {
      const mockData = generateMockAnalyticsData();
      setCachedData(cacheKey, mockData.businessMetrics);
      return mockData.businessMetrics;
    } catch (error) {
      console.error('Error fetching business metrics:', error);
      return generateMockAnalyticsData().businessMetrics;
    }
  }

  // Export de données
  async exportData(format: 'csv' | 'json' | 'excel', data: any): Promise<Blob> {
    try {
      let content: string;
      let mimeType: string;

      switch (format) {
        case 'csv':
          content = this.convertToCSV(data);
          mimeType = 'text/csv';
          break;
        case 'json':
          content = JSON.stringify(data, null, 2);
          mimeType = 'application/json';
          break;
        case 'excel':
          // Pour l'Excel, on simule avec du CSV
          content = this.convertToCSV(data);
          mimeType = 'application/vnd.ms-excel';
          break;
        default:
          throw new Error(`Format non supporté: ${format}`);
      }

      return new Blob([content], { type: mimeType });
    } catch (error) {
      console.error('Error exporting data:', error);
      throw error;
    }
  }

  // Utilitaire pour convertir en CSV
  private convertToCSV(data: any[]): string {
    if (!data || data.length === 0) return '';

    const headers = Object.keys(data[0]).join(',');
    const rows = data.map(row => 
      Object.values(row).map(value => 
        typeof value === 'string' ? `"${value}"` : value
      ).join(',')
    );

    return [headers, ...rows].join('\n');
  }

  // Nettoyage du cache
  clearCache(): void {
    cache.clear();
  }
}

export const analyticsService = AnalyticsService.getInstance();
export default analyticsService;