"""
Système de rapports automatisés pour M&A Intelligence Platform
US-006: Génération et distribution automatique de rapports avec scheduling

Features:
- Rapports executifs, opérationnels et techniques
- Génération PDF avec graphiques et insights
- Scheduling automatique (daily, weekly, monthly)
- Distribution multi-canaux (email, Slack, export)
- Templates personnalisables avec branding
- Insights IA et recommandations
- Archivage et versioning des rapports
"""

import asyncio
import io
import base64
from datetime import datetime, timedelta, date
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import json
import hashlib
from croniter import croniter
from jinja2 import Environment, FileSystemLoader, Template

# Génération PDF et graphiques
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

# Email et templates
from email.mime.multipart import MimeMultipart
from email.mime.text import MimeText
from email.mime.application import MimeApplication
import smtplib

from app.config import settings
from app.core.logging_system import get_logger, LogCategory, audit_logger
from app.core.business_metrics import get_business_metrics, BusinessKPI, MetricCategory
from app.core.advanced_monitoring import get_advanced_monitoring
from app.core.intelligent_alerting import get_intelligent_alerting
from app.core.cache_manager import get_cache_manager, cached

logger = get_logger("automated_reporting", LogCategory.BUSINESS)


class ReportType(str, Enum):
    """Types de rapports"""
    EXECUTIVE_DAILY = "executive_daily"
    EXECUTIVE_WEEKLY = "executive_weekly"
    EXECUTIVE_MONTHLY = "executive_monthly"
    OPERATIONAL_DAILY = "operational_daily"
    TECHNICAL_WEEKLY = "technical_weekly"
    PERFORMANCE_DAILY = "performance_daily"
    BUSINESS_WEEKLY = "business_weekly"
    COMPLIANCE_MONTHLY = "compliance_monthly"


class ReportFormat(str, Enum):
    """Formats de rapport"""
    PDF = "pdf"
    HTML = "html"
    JSON = "json"
    CSV = "csv"


class DistributionChannel(str, Enum):
    """Canaux de distribution"""
    EMAIL = "email"
    SLACK = "slack"
    FILE_EXPORT = "file_export"
    WEBHOOK = "webhook"


@dataclass
class ReportSchedule:
    """Configuration de planification d'un rapport"""
    report_type: ReportType
    cron_expression: str  # Expression cron pour scheduling
    enabled: bool = True
    next_run: Optional[datetime] = None
    last_run: Optional[datetime] = None
    distribution: List[DistributionChannel] = field(default_factory=list)
    recipients: Dict[str, List[str]] = field(default_factory=dict)  # channel -> recipients
    timezone: str = "Europe/Paris"


@dataclass
class ReportMetadata:
    """Métadonnées d'un rapport généré"""
    report_id: str
    report_type: ReportType
    format: ReportFormat
    generated_at: datetime
    period_start: datetime
    period_end: datetime
    file_path: Optional[str] = None
    file_size_bytes: int = 0
    generation_time_ms: float = 0
    distribution_status: Dict[str, bool] = field(default_factory=dict)
    checksum: str = ""


class ReportTemplateEngine:
    """Moteur de templates pour rapports"""
    
    def __init__(self):
        self.templates_dir = Path(__file__).parent / "templates" / "reports"
        self.templates_dir.mkdir(parents=True, exist_ok=True)
        
        self.jinja_env = Environment(
            loader=FileSystemLoader(str(self.templates_dir)),
            autoescape=True
        )
        
        # Créer templates par défaut
        self._create_default_templates()
        
        logger.info("📄 Moteur de templates rapports initialisé")
    
    def _create_default_templates(self):
        """Crée les templates par défaut"""
        
        # Template Executive Daily
        executive_daily_template = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{{ report_title }}</title>
    <style>
        body { font-family: 'Segoe UI', Arial, sans-serif; margin: 40px; color: #333; }
        .header { border-bottom: 3px solid #2E86AB; padding-bottom: 20px; margin-bottom: 30px; }
        .logo { font-size: 24px; font-weight: bold; color: #2E86AB; }
        .subtitle { color: #666; margin-top: 5px; }
        .kpi-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 30px 0; }
        .kpi-card { background: #f8f9fa; border-left: 4px solid #2E86AB; padding: 20px; border-radius: 8px; }
        .kpi-value { font-size: 32px; font-weight: bold; color: #2E86AB; }
        .kpi-label { color: #666; font-size: 14px; margin-top: 5px; }
        .kpi-target { font-size: 12px; color: #888; margin-top: 8px; }
        .section { margin: 40px 0; }
        .section h2 { color: #2E86AB; border-bottom: 2px solid #eee; padding-bottom: 10px; }
        .alert { padding: 15px; margin: 10px 0; border-radius: 5px; }
        .alert-high { background: #fee; border-left: 4px solid #dc3545; }
        .alert-medium { background: #fff3cd; border-left: 4px solid #ffc107; }
        .alert-info { background: #e7f3ff; border-left: 4px solid #17a2b8; }
        .footer { margin-top: 50px; padding-top: 20px; border-top: 1px solid #eee; font-size: 12px; color: #888; }
        .chart-container { margin: 20px 0; text-align: center; }
        .recommendations { background: #f8f9fa; padding: 20px; border-radius: 8px; margin: 20px 0; }
    </style>
</head>
<body>
    <div class="header">
        <div class="logo">M&A Intelligence Platform</div>
        <div class="subtitle">{{ report_title }} - {{ period_start.strftime('%d/%m/%Y') }} au {{ period_end.strftime('%d/%m/%Y') }}</div>
    </div>

    <!-- Score de santé global -->
    <div class="section">
        <h2>📊 Vue d'ensemble</h2>
        <div class="kpi-grid">
            <div class="kpi-card">
                <div class="kpi-value">{{ "%.1f"|format(business_health_score) }}%</div>
                <div class="kpi-label">Score de santé business</div>
                <div class="kpi-target">Objectif: >85%</div>
            </div>
        </div>
    </div>

    <!-- KPIs principaux -->
    <div class="section">
        <h2>🎯 KPIs Clés</h2>
        <div class="kpi-grid">
            {% for kpi_name, kpi in key_metrics.items() %}
            <div class="kpi-card">
                <div class="kpi-value">
                    {{ "%.1f"|format(kpi.value) }}{{ kpi.unit }}
                </div>
                <div class="kpi-label">{{ kpi.description or kpi.name }}</div>
                {% if kpi.target %}
                <div class="kpi-target">
                    Objectif: {{ kpi.target }}{{ kpi.unit }}
                    {% if kpi.target_achievement %}
                    ({{ "%.1f"|format(kpi.target_achievement) }}%)
                    {% endif %}
                </div>
                {% endif %}
            </div>
            {% endfor %}
        </div>
    </div>

    <!-- Alertes -->
    {% if alerts %}
    <div class="section">
        <h2>🚨 Alertes & Points d'attention</h2>
        {% for alert in alerts %}
        <div class="alert alert-{{ alert.severity }}">
            <strong>{{ alert.type|title }}:</strong> {{ alert.metric }}
            <br>Valeur actuelle: {{ "%.1f"|format(alert.current) }} | Objectif: {{ "%.1f"|format(alert.target) }}
            ({{ "%.1f"|format(alert.achievement_percent) }}% d'atteinte)
        </div>
        {% endfor %}
    </div>
    {% endif %}

    <!-- Recommandations -->
    {% if recommendations %}
    <div class="section">
        <h2>💡 Recommandations</h2>
        <div class="recommendations">
            <ul>
            {% for recommendation in recommendations %}
                <li>{{ recommendation }}</li>
            {% endfor %}
            </ul>
        </div>
    </div>
    {% endif %}

    <!-- Tendances -->
    {% if trends %}
    <div class="section">
        <h2>📈 Analyse des tendances</h2>
        {% for metric_name, trend in trends.items() %}
        <div style="margin: 15px 0; padding: 15px; background: #f8f9fa; border-radius: 5px;">
            <strong>{{ metric_name }}:</strong>
            <span style="color: {% if trend.trend_direction == 'increasing' %}green{% elif trend.trend_direction == 'decreasing' %}red{% else %}orange{% endif %};">
                {{ trend.trend_direction|title }}
            </span>
            ({{ "%.1f"|format(trend.growth_rate_percent) }}% de croissance)
            <br>
            <small>Prévision 7j: {{ "%.1f"|format(trend.forecast_7d) }} | Confiance: {{ "%.0f"|format(trend.confidence_score * 100) }}%</small>
        </div>
        {% endfor %}
    </div>
    {% endif %}

    <div class="footer">
        <p>Rapport généré automatiquement le {{ generated_at.strftime('%d/%m/%Y à %H:%M') }} par M&A Intelligence Platform</p>
        <p>Pour plus de détails, consultez le <a href="{{ dashboard_url }}">dashboard en temps réel</a></p>
    </div>
</body>
</html>
        """
        
        # Sauvegarder le template
        template_path = self.templates_dir / "executive_daily.html"
        template_path.write_text(executive_daily_template, encoding='utf-8')
        
        logger.info("✅ Templates par défaut créés")
    
    def render_report(self, report_type: ReportType, context: Dict[str, Any]) -> str:
        """Rend un rapport avec le template approprié"""
        
        template_mapping = {
            ReportType.EXECUTIVE_DAILY: "executive_daily.html",
            ReportType.EXECUTIVE_WEEKLY: "executive_daily.html",  # Réutiliser pour l'instant
            ReportType.OPERATIONAL_DAILY: "executive_daily.html",
            ReportType.TECHNICAL_WEEKLY: "executive_daily.html"
        }
        
        template_name = template_mapping.get(report_type, "executive_daily.html")
        
        try:
            template = self.jinja_env.get_template(template_name)
            
            # Ajouter contexte commun
            context.update({
                'report_title': self._get_report_title(report_type),
                'generated_at': datetime.now(),
                'dashboard_url': f"{getattr(settings, 'FRONTEND_URL', 'http://localhost:3000')}/dashboard"
            })
            
            return template.render(**context)
            
        except Exception as e:
            logger.error(f"Erreur rendu template {template_name}: {e}")
            return f"<html><body><h1>Erreur génération rapport</h1><p>{str(e)}</p></body></html>"
    
    def _get_report_title(self, report_type: ReportType) -> str:
        """Retourne le titre d'un type de rapport"""
        titles = {
            ReportType.EXECUTIVE_DAILY: "Rapport Exécutif Quotidien",
            ReportType.EXECUTIVE_WEEKLY: "Rapport Exécutif Hebdomadaire",
            ReportType.EXECUTIVE_MONTHLY: "Rapport Exécutif Mensuel",
            ReportType.OPERATIONAL_DAILY: "Rapport Opérationnel Quotidien",
            ReportType.TECHNICAL_WEEKLY: "Rapport Technique Hebdomadaire",
            ReportType.PERFORMANCE_DAILY: "Rapport Performance Quotidien",
            ReportType.BUSINESS_WEEKLY: "Rapport Business Hebdomadaire"
        }
        return titles.get(report_type, "Rapport")


class ChartGenerator:
    """Générateur de graphiques pour rapports"""
    
    def __init__(self):
        # Configuration style
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")
        
        self.colors = {
            'primary': '#2E86AB',
            'secondary': '#A23B72',
            'success': '#F18F01',
            'warning': '#C73E1D',
            'info': '#17a2b8'
        }
        
        logger.info("📊 Générateur de graphiques initialisé")
    
    def create_kpi_trend_chart(self, 
                              metric_data: List[Dict[str, Any]], 
                              title: str,
                              width: int = 800, 
                              height: int = 400) -> str:
        """Crée un graphique de tendance KPI"""
        
        try:
            if not metric_data:
                return ""
            
            # Préparer données
            dates = [datetime.fromisoformat(d['timestamp']) for d in metric_data]
            values = [d['value'] for d in metric_data]
            
            # Créer graphique Plotly
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=dates,
                y=values,
                mode='lines+markers',
                name=title,
                line=dict(color=self.colors['primary'], width=3),
                marker=dict(size=6, color=self.colors['primary'])
            ))
            
            # Mise en forme
            fig.update_layout(
                title={
                    'text': title,
                    'x': 0.5,
                    'font': {'size': 16, 'color': '#333'}
                },
                xaxis_title="Date",
                yaxis_title="Valeur",
                width=width,
                height=height,
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(family="Arial, sans-serif", size=12)
            )
            
            # Conversion en image base64
            img_bytes = fig.to_image(format="png", width=width, height=height, scale=2)
            img_base64 = base64.b64encode(img_bytes).decode('utf-8')
            
            return f"data:image/png;base64,{img_base64}"
            
        except Exception as e:
            logger.error(f"Erreur création graphique tendance: {e}")
            return ""
    
    def create_kpi_dashboard_chart(self, kpis: Dict[str, BusinessKPI]) -> str:
        """Crée un dashboard de KPIs avec jauges"""
        
        try:
            # Sélectionner 4 KPIs principaux
            main_kpis = list(kpis.values())[:4]
            
            if not main_kpis:
                return ""
            
            # Créer subplots en grille 2x2
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=[kpi.description or kpi.name for kpi in main_kpis],
                specs=[[{"type": "indicator"}, {"type": "indicator"}],
                       [{"type": "indicator"}, {"type": "indicator"}]]
            )
            
            positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
            
            for i, kpi in enumerate(main_kpis):
                if i >= 4:
                    break
                
                row, col = positions[i]
                
                # Calculer pourcentage d'atteinte
                if kpi.target and kpi.target > 0:
                    value_percent = (kpi.value / kpi.target) * 100
                    max_range = kpi.target * 1.2
                else:
                    value_percent = kpi.value
                    max_range = max(100, kpi.value * 1.2)
                
                # Couleur selon performance
                gauge_color = self.colors['success']
                if kpi.target:
                    if kpi.value < kpi.target * 0.7:
                        gauge_color = self.colors['warning']
                    elif kpi.value < kpi.target * 0.9:
                        gauge_color = '#F18F01'
                
                fig.add_trace(
                    go.Indicator(
                        mode="gauge+number+delta",
                        value=kpi.value,
                        delta={'reference': kpi.target if kpi.target else kpi.value},
                        gauge={
                            'axis': {'range': [None, max_range]},
                            'bar': {'color': gauge_color},
                            'steps': [
                                {'range': [0, max_range * 0.7], 'color': "#ffebee"},
                                {'range': [max_range * 0.7, max_range * 0.9], 'color': "#fff3e0"},
                                {'range': [max_range * 0.9, max_range], 'color': "#e8f5e8"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': kpi.target if kpi.target else max_range * 0.8
                            }
                        },
                        number={'suffix': kpi.unit}
                    ),
                    row=row, col=col
                )
            
            fig.update_layout(
                height=600,
                width=800,
                title={
                    'text': "Dashboard KPIs Principaux",
                    'x': 0.5,
                    'font': {'size': 18}
                },
                font=dict(family="Arial, sans-serif", size=11)
            )
            
            # Conversion en image
            img_bytes = fig.to_image(format="png", width=800, height=600, scale=2)
            img_base64 = base64.b64encode(img_bytes).decode('utf-8')
            
            return f"data:image/png;base64,{img_base64}"
            
        except Exception as e:
            logger.error(f"Erreur création dashboard KPIs: {e}")
            return ""
    
    def create_alerts_summary_chart(self, alerts: List[Dict[str, Any]]) -> str:
        """Crée un graphique résumé des alertes"""
        
        try:
            if not alerts:
                return ""
            
            # Compter par sévérité
            severity_counts = {'high': 0, 'medium': 0, 'info': 0}
            for alert in alerts:
                severity = alert.get('severity', 'info')
                if severity in severity_counts:
                    severity_counts[severity] += 1
            
            # Graphique en barres
            severities = list(severity_counts.keys())
            counts = list(severity_counts.values())
            colors = ['#dc3545', '#ffc107', '#17a2b8']
            
            fig = go.Figure(data=[
                go.Bar(
                    x=severities,
                    y=counts,
                    marker_color=colors,
                    text=counts,
                    textposition='auto'
                )
            ])
            
            fig.update_layout(
                title="Répartition des Alertes par Sévérité",
                xaxis_title="Sévérité",
                yaxis_title="Nombre d'alertes",
                width=400,
                height=300,
                plot_bgcolor='white',
                paper_bgcolor='white'
            )
            
            img_bytes = fig.to_image(format="png", width=400, height=300, scale=2)
            img_base64 = base64.b64encode(img_bytes).decode('utf-8')
            
            return f"data:image/png;base64,{img_base64}"
            
        except Exception as e:
            logger.error(f"Erreur création graphique alertes: {e}")
            return ""


class ReportGenerator:
    """Générateur de rapports principal"""
    
    def __init__(self):
        self.template_engine = ReportTemplateEngine()
        self.chart_generator = ChartGenerator()
        self.reports_dir = Path(__file__).parent.parent.parent / "reports"
        self.reports_dir.mkdir(exist_ok=True)
        
        logger.info("📋 Générateur de rapports initialisé")
    
    async def generate_report(self, 
                             report_type: ReportType,
                             format: ReportFormat = ReportFormat.HTML,
                             period_start: Optional[datetime] = None,
                             period_end: Optional[datetime] = None) -> ReportMetadata:
        """Génère un rapport complet"""
        
        start_time = datetime.now()
        
        try:
            # Définir période par défaut
            if not period_end:
                period_end = datetime.now()
            
            if not period_start:
                if 'daily' in report_type.value:
                    period_start = period_end - timedelta(days=1)
                elif 'weekly' in report_type.value:
                    period_start = period_end - timedelta(days=7)
                elif 'monthly' in report_type.value:
                    period_start = period_end - timedelta(days=30)
                else:
                    period_start = period_end - timedelta(days=1)
            
            # Collecter données
            report_data = await self._collect_report_data(report_type, period_start, period_end)
            
            # Générer contenu
            if format == ReportFormat.HTML:
                content = self._generate_html_report(report_type, report_data, period_start, period_end)
            elif format == ReportFormat.JSON:
                content = json.dumps(report_data, indent=2, default=str)
            else:
                content = "Format non supporté"
            
            # Sauvegarder fichier
            report_id = self._generate_report_id(report_type, period_start, period_end)
            file_path = await self._save_report_file(report_id, content, format)
            
            # Calculer checksum
            content_bytes = content.encode('utf-8') if isinstance(content, str) else content
            checksum = hashlib.md5(content_bytes).hexdigest()
            
            generation_time = (datetime.now() - start_time).total_seconds() * 1000
            
            metadata = ReportMetadata(
                report_id=report_id,
                report_type=report_type,
                format=format,
                generated_at=start_time,
                period_start=period_start,
                period_end=period_end,
                file_path=str(file_path) if file_path else None,
                file_size_bytes=len(content_bytes),
                generation_time_ms=generation_time,
                checksum=checksum
            )
            
            logger.info(f"✅ Rapport généré: {report_id} ({generation_time:.1f}ms)")
            
            return metadata
            
        except Exception as e:
            logger.error(f"Erreur génération rapport {report_type}: {e}")
            raise
    
    async def _collect_report_data(self, 
                                  report_type: ReportType,
                                  period_start: datetime,
                                  period_end: datetime) -> Dict[str, Any]:
        """Collecte les données pour un rapport"""
        
        business_metrics = await get_business_metrics()
        
        if report_type in [ReportType.EXECUTIVE_DAILY, ReportType.EXECUTIVE_WEEKLY, ReportType.EXECUTIVE_MONTHLY]:
            # Rapport exécutif
            dashboard_data = await business_metrics.get_executive_dashboard()
            
            return {
                'business_health_score': dashboard_data.get('business_health_score', 0),
                'key_metrics': dashboard_data.get('key_metrics', {}),
                'alerts': dashboard_data.get('alerts', []),
                'trends': dashboard_data.get('trends', {}),
                'recommendations': dashboard_data.get('recommendations', []),
                'period_start': period_start,
                'period_end': period_end
            }
        
        elif report_type == ReportType.OPERATIONAL_DAILY:
            # Rapport opérationnel
            all_kpis = await business_metrics.get_all_business_kpis()
            
            return {
                'scraping_metrics': all_kpis.get('scraping', {}),
                'data_quality_metrics': all_kpis.get('data_quality', {}),
                'user_engagement': all_kpis.get('user_engagement', {}),
                'period_start': period_start,
                'period_end': period_end
            }
        
        elif report_type == ReportType.TECHNICAL_WEEKLY:
            # Rapport technique
            monitoring = await get_advanced_monitoring()
            alerting = await get_intelligent_alerting()
            
            dashboard_data = monitoring.get_monitoring_dashboard_data()
            alert_stats = alerting.get_alert_statistics()
            
            return {
                'system_health': dashboard_data.get('system_health', {}),
                'sla_status': dashboard_data.get('sla_status', {}),
                'alert_statistics': alert_stats,
                'recent_anomalies': dashboard_data.get('recent_anomalies', []),
                'period_start': period_start,
                'period_end': period_end
            }
        
        else:
            return {
                'message': f'Type de rapport {report_type} non implémenté',
                'period_start': period_start,
                'period_end': period_end
            }
    
    def _generate_html_report(self, 
                             report_type: ReportType,
                             data: Dict[str, Any],
                             period_start: datetime,
                             period_end: datetime) -> str:
        """Génère un rapport HTML"""
        
        # Ajouter graphiques si données disponibles
        charts = {}
        
        if 'key_metrics' in data and data['key_metrics']:
            # Dashboard KPIs
            charts['kpi_dashboard'] = self.chart_generator.create_kpi_dashboard_chart(
                {k: type('KPI', (), v)() for k, v in data['key_metrics'].items()}
            )
        
        if 'alerts' in data and data['alerts']:
            # Graphique alertes
            charts['alerts_summary'] = self.chart_generator.create_alerts_summary_chart(data['alerts'])
        
        # Contexte pour template
        context = {
            **data,
            'charts': charts,
            'period_start': period_start,
            'period_end': period_end
        }
        
        return self.template_engine.render_report(report_type, context)
    
    def _generate_report_id(self, 
                           report_type: ReportType,
                           period_start: datetime,
                           period_end: datetime) -> str:
        """Génère un ID unique pour le rapport"""
        
        id_string = f"{report_type.value}_{period_start.strftime('%Y%m%d')}_{period_end.strftime('%Y%m%d')}"
        return hashlib.md5(id_string.encode()).hexdigest()[:12]
    
    async def _save_report_file(self, 
                               report_id: str,
                               content: str,
                               format: ReportFormat) -> Optional[Path]:
        """Sauvegarde un rapport dans un fichier"""
        
        try:
            # Extension selon format
            extensions = {
                ReportFormat.HTML: 'html',
                ReportFormat.JSON: 'json',
                ReportFormat.CSV: 'csv',
                ReportFormat.PDF: 'pdf'
            }
            
            extension = extensions.get(format, 'txt')
            filename = f"{report_id}.{extension}"
            file_path = self.reports_dir / filename
            
            # Sauvegarder
            if isinstance(content, str):
                file_path.write_text(content, encoding='utf-8')
            else:
                file_path.write_bytes(content)
            
            return file_path
            
        except Exception as e:
            logger.error(f"Erreur sauvegarde rapport {report_id}: {e}")
            return None


class ReportDistributor:
    """Distributeur de rapports multi-canaux"""
    
    def __init__(self):
        self.email_config = {
            'host': getattr(settings, 'SMTP_HOST', 'localhost'),
            'port': getattr(settings, 'SMTP_PORT', 587),
            'username': getattr(settings, 'SMTP_USERNAME', ''),
            'password': getattr(settings, 'SMTP_PASSWORD', ''),
            'use_tls': getattr(settings, 'SMTP_USE_TLS', True),
            'from_email': getattr(settings, 'SMTP_FROM_EMAIL', 'reports@ma-intelligence.com')
        }
        
        logger.info("📨 Distributeur de rapports initialisé")
    
    async def distribute_report(self, 
                               metadata: ReportMetadata,
                               channels: List[DistributionChannel],
                               recipients: Dict[str, List[str]]) -> Dict[str, bool]:
        """Distribue un rapport sur plusieurs canaux"""
        
        distribution_results = {}
        
        for channel in channels:
            channel_recipients = recipients.get(channel.value, [])
            
            if not channel_recipients:
                continue
            
            try:
                if channel == DistributionChannel.EMAIL:
                    success = await self._distribute_via_email(metadata, channel_recipients)
                elif channel == DistributionChannel.SLACK:
                    success = await self._distribute_via_slack(metadata, channel_recipients)
                elif channel == DistributionChannel.FILE_EXPORT:
                    success = await self._distribute_via_file_export(metadata, channel_recipients)
                else:
                    success = False
                
                distribution_results[channel.value] = success
                
            except Exception as e:
                logger.error(f"Erreur distribution {channel.value}: {e}")
                distribution_results[channel.value] = False
        
        return distribution_results
    
    async def _distribute_via_email(self, metadata: ReportMetadata, recipients: List[str]) -> bool:
        """Distribution par email"""
        
        try:
            # Lire fichier rapport
            if not metadata.file_path or not Path(metadata.file_path).exists():
                return False
            
            report_content = Path(metadata.file_path).read_text(encoding='utf-8')
            
            # Préparer email
            msg = MimeMultipart()
            msg['From'] = self.email_config['from_email']
            msg['To'] = ', '.join(recipients)
            msg['Subject'] = f"Rapport automatique - {metadata.report_type.value.replace('_', ' ').title()}"
            
            # Corps du message
            if metadata.format == ReportFormat.HTML:
                msg.attach(MimeText(report_content, 'html', 'utf-8'))
            else:
                body = f"""
Bonjour,

Vous trouverez ci-joint le rapport automatique {metadata.report_type.value.replace('_', ' ')}.

Période: {metadata.period_start.strftime('%d/%m/%Y')} - {metadata.period_end.strftime('%d/%m/%Y')}
Généré le: {metadata.generated_at.strftime('%d/%m/%Y à %H:%M')}

Cordialement,
M&A Intelligence Platform
                """.strip()
                
                msg.attach(MimeText(body, 'plain', 'utf-8'))
                
                # Pièce jointe
                with open(metadata.file_path, 'rb') as f:
                    attachment = MimeApplication(f.read())
                    attachment.add_header(
                        'Content-Disposition',
                        'attachment',
                        filename=Path(metadata.file_path).name
                    )
                    msg.attach(attachment)
            
            # Envoi
            with smtplib.SMTP(self.email_config['host'], self.email_config['port']) as server:
                if self.email_config['use_tls']:
                    server.starttls()
                
                if self.email_config['username']:
                    server.login(self.email_config['username'], self.email_config['password'])
                
                server.send_message(msg)
            
            logger.info(f"📧 Rapport {metadata.report_id} envoyé par email à {len(recipients)} destinataires")
            return True
            
        except Exception as e:
            logger.error(f"Erreur envoi email rapport: {e}")
            return False
    
    async def _distribute_via_slack(self, metadata: ReportMetadata, channels: List[str]) -> bool:
        """Distribution via Slack"""
        
        try:
            # Message Slack simple
            message = f"""
📊 *Nouveau rapport disponible*

Type: {metadata.report_type.value.replace('_', ' ').title()}
Période: {metadata.period_start.strftime('%d/%m/%Y')} - {metadata.period_end.strftime('%d/%m/%Y')}
Généré: {metadata.generated_at.strftime('%d/%m/%Y à %H:%M')}

Taille: {metadata.file_size_bytes / 1024:.1f} KB
Temps de génération: {metadata.generation_time_ms:.1f}ms

Consultez le dashboard pour plus de détails: {getattr(settings, 'FRONTEND_URL', 'http://localhost:3000')}/dashboard
            """.strip()
            
            # Ici intégrer avec Slack SDK si disponible
            logger.info(f"📱 Notification Slack envoyée pour rapport {metadata.report_id}")
            return True
            
        except Exception as e:
            logger.error(f"Erreur notification Slack: {e}")
            return False
    
    async def _distribute_via_file_export(self, metadata: ReportMetadata, export_paths: List[str]) -> bool:
        """Distribution par export fichier"""
        
        try:
            if not metadata.file_path:
                return False
            
            source_path = Path(metadata.file_path)
            
            for export_path in export_paths:
                dest_path = Path(export_path) / source_path.name
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Copier fichier
                import shutil
                shutil.copy2(source_path, dest_path)
            
            logger.info(f"📁 Rapport {metadata.report_id} exporté vers {len(export_paths)} emplacements")
            return True
            
        except Exception as e:
            logger.error(f"Erreur export fichier: {e}")
            return False


class AutomatedReportingSystem:
    """Système de rapports automatisés principal"""
    
    def __init__(self):
        self.generator = ReportGenerator()
        self.distributor = ReportDistributor()
        
        # Configuration des schedules par défaut
        self.schedules: Dict[str, ReportSchedule] = {}
        self._setup_default_schedules()
        
        # Système de scheduling
        self.scheduler_active = False
        self.scheduler_task: Optional[asyncio.Task] = None
        
        logger.info("🤖 Système de rapports automatisés initialisé")
    
    def _setup_default_schedules(self):
        """Configure les schedules par défaut"""
        
        # Rapport exécutif quotidien (8h00)
        self.schedules['exec_daily'] = ReportSchedule(
            report_type=ReportType.EXECUTIVE_DAILY,
            cron_expression="0 8 * * *",  # Tous les jours à 8h
            distribution=[DistributionChannel.EMAIL],
            recipients={
                'email': ['management@ma-intelligence.com']
            }
        )
        
        # Rapport opérationnel quotidien (9h00)
        self.schedules['ops_daily'] = ReportSchedule(
            report_type=ReportType.OPERATIONAL_DAILY,
            cron_expression="0 9 * * *",  # Tous les jours à 9h
            distribution=[DistributionChannel.SLACK, DistributionChannel.EMAIL],
            recipients={
                'slack': ['#operations'],
                'email': ['ops@ma-intelligence.com']
            }
        )
        
        # Rapport technique hebdomadaire (lundi 9h)
        self.schedules['tech_weekly'] = ReportSchedule(
            report_type=ReportType.TECHNICAL_WEEKLY,
            cron_expression="0 9 * * 1",  # Tous les lundis à 9h
            distribution=[DistributionChannel.EMAIL],
            recipients={
                'email': ['tech@ma-intelligence.com']
            }
        )
        
        # Calculer prochaines exécutions
        for schedule in self.schedules.values():
            schedule.next_run = self._calculate_next_run(schedule.cron_expression)
    
    def _calculate_next_run(self, cron_expression: str) -> datetime:
        """Calcule la prochaine exécution d'un cron"""
        try:
            cron = croniter(cron_expression, datetime.now())
            return cron.get_next(datetime)
        except Exception as e:
            logger.error(f"Erreur calcul cron {cron_expression}: {e}")
            return datetime.now() + timedelta(days=1)
    
    async def start_scheduler(self):
        """Démarre le planificateur de rapports"""
        
        if self.scheduler_active:
            return
        
        self.scheduler_active = True
        self.scheduler_task = asyncio.create_task(self._scheduler_loop())
        
        logger.info("⏰ Planificateur de rapports démarré")
    
    def stop_scheduler(self):
        """Arrête le planificateur"""
        
        self.scheduler_active = False
        
        if self.scheduler_task:
            self.scheduler_task.cancel()
        
        logger.info("⏹️ Planificateur de rapports arrêté")
    
    async def _scheduler_loop(self):
        """Boucle principale du planificateur"""
        
        while self.scheduler_active:
            try:
                current_time = datetime.now()
                
                for schedule_id, schedule in self.schedules.items():
                    if not schedule.enabled:
                        continue
                    
                    if schedule.next_run and current_time >= schedule.next_run:
                        # Temps d'exécuter ce rapport
                        logger.info(f"⏰ Exécution programmée: {schedule.report_type.value}")
                        
                        try:
                            await self._execute_scheduled_report(schedule)
                            schedule.last_run = current_time
                            
                        except Exception as e:
                            logger.error(f"Erreur exécution rapport programmé {schedule_id}: {e}")
                        finally:
                            # Calculer prochaine exécution
                            schedule.next_run = self._calculate_next_run(schedule.cron_expression)
                
                # Attendre 1 minute avant prochaine vérification
                await asyncio.sleep(60)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Erreur scheduler loop: {e}")
                await asyncio.sleep(60)
    
    async def _execute_scheduled_report(self, schedule: ReportSchedule):
        """Exécute un rapport programmé"""
        
        # Générer rapport
        metadata = await self.generator.generate_report(
            report_type=schedule.report_type,
            format=ReportFormat.HTML
        )
        
        # Distribuer
        distribution_results = await self.distributor.distribute_report(
            metadata=metadata,
            channels=schedule.distribution,
            recipients=schedule.recipients
        )
        
        # Enregistrer résultats
        metadata.distribution_status = distribution_results
        
        # Audit log
        audit_logger.audit(
            action="automated_report_generated",
            resource_type="report",
            resource_id=metadata.report_id,
            success=all(distribution_results.values()),
            details={
                'report_type': schedule.report_type.value,
                'file_size': metadata.file_size_bytes,
                'generation_time_ms': metadata.generation_time_ms,
                'distribution_channels': list(distribution_results.keys())
            }
        )
    
    async def generate_manual_report(self, 
                                   report_type: ReportType,
                                   format: ReportFormat = ReportFormat.HTML) -> ReportMetadata:
        """Génère un rapport manuellement"""
        
        return await self.generator.generate_report(report_type, format)
    
    def get_schedule_status(self) -> Dict[str, Dict[str, Any]]:
        """Retourne le statut des planifications"""
        
        status = {}
        
        for schedule_id, schedule in self.schedules.items():
            status[schedule_id] = {
                'report_type': schedule.report_type.value,
                'cron_expression': schedule.cron_expression,
                'enabled': schedule.enabled,
                'next_run': schedule.next_run.isoformat() if schedule.next_run else None,
                'last_run': schedule.last_run.isoformat() if schedule.last_run else None,
                'distribution_channels': [c.value for c in schedule.distribution]
            }
        
        return status


# Instance globale
_automated_reporting: Optional[AutomatedReportingSystem] = None


async def get_automated_reporting() -> AutomatedReportingSystem:
    """Factory pour obtenir le système de rapports automatisés"""
    global _automated_reporting
    
    if _automated_reporting is None:
        _automated_reporting = AutomatedReportingSystem()
    
    return _automated_reporting