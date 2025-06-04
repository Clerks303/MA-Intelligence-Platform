"""
Jobs Celery pour la génération de rapports et exports
US-009: Génération de rapports complexes en arrière-plan

Ce module contient:
- Génération de rapports PDF/Excel
- Exports de données volumineuses
- Rapports d'analyse avec graphiques
- Envoi automatique par email
- Compression et archivage
- Rapports programmés
"""

import asyncio
import time
import os
import tempfile
import zipfile
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
import json
from pathlib import Path
import base64
from io import BytesIO

# Import conditionnel pour génération PDF/Excel
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    REPORT_LIBS_AVAILABLE = True
except ImportError:
    REPORT_LIBS_AVAILABLE = False
    logger.warning("Librairies de génération de rapports non disponibles")

from app.core.background_jobs import (
    celery_task, JobPriority, JobType, get_background_job_manager
)
from app.core.logging_system import get_logger, LogCategory
from app.core.cache_manager import get_cache_manager

logger = get_logger("background_reports", LogCategory.REPORTS)


@dataclass
class ReportConfig:
    """Configuration de génération de rapport"""
    output_format: str = "pdf"  # pdf, excel, json, csv
    include_charts: bool = True
    include_summary: bool = True
    include_details: bool = True
    max_items_per_page: int = 50
    chart_style: str = "professional"
    compress_output: bool = False
    send_email: bool = False
    email_recipients: List[str] = field(default_factory=list)
    
    
@dataclass 
class ReportSection:
    """Section d'un rapport"""
    title: str
    content_type: str  # text, table, chart, summary
    data: Any
    order: int = 0
    include_in_summary: bool = True


@celery_task(priority=JobPriority.NORMAL)
async def generate_company_analysis_report(
    company_ids: List[str],
    report_type: str = "detailed",
    config: Optional[Dict[str, Any]] = None,
    job_id: Optional[str] = None,
    user_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Génère un rapport d'analyse d'entreprises
    
    Args:
        company_ids: Liste des SIREN à analyser
        report_type: Type de rapport (summary, detailed, comparison)
        config: Configuration du rapport
        job_id: ID du job
        user_id: ID utilisateur pour email
    """
    start_time = time.time()
    
    if not REPORT_LIBS_AVAILABLE:
        return {
            'success': False,
            'error': 'Librairies de génération de rapports non installées',
            'timestamp': datetime.now().isoformat()
        }
    
    # Configuration
    report_config = ReportConfig()
    if config:
        for key, value in config.items():
            if hasattr(report_config, key):
                setattr(report_config, key, value)
    
    logger.info(f"📊 Génération rapport entreprises: {len(company_ids)} companies ({report_type})")
    
    try:
        # Collecter données des entreprises
        companies_data = await _collect_companies_data(company_ids)
        
        if not companies_data:
            return {
                'success': False,
                'error': 'Aucune donnée d\'entreprise trouvée',
                'timestamp': datetime.now().isoformat()
            }
        
        # Analyser les données
        analysis_results = await _analyze_companies_for_report(companies_data)
        
        # Construire sections du rapport
        report_sections = []
        
        # 1. Résumé exécutif
        if report_config.include_summary:
            summary_section = await _create_executive_summary(analysis_results, companies_data)
            report_sections.append(summary_section)
        
        # 2. Analyse financière
        financial_section = await _create_financial_analysis(analysis_results, report_config)
        report_sections.append(financial_section)
        
        # 3. Analyse sectorielle
        sector_section = await _create_sector_analysis(analysis_results, report_config)
        report_sections.append(sector_section)
        
        # 4. Scoring et recommandations
        scoring_section = await _create_scoring_analysis(companies_data, report_config)
        report_sections.append(scoring_section)
        
        # 5. Détails par entreprise
        if report_config.include_details and report_type == "detailed":
            details_section = await _create_company_details(companies_data, report_config)
            report_sections.append(details_section)
        
        # Générer le rapport selon le format
        output_path = None
        if report_config.output_format.lower() == "pdf":
            output_path = await _generate_pdf_report(report_sections, report_config, job_id)
        elif report_config.output_format.lower() == "excel":
            output_path = await _generate_excel_report(report_sections, report_config, job_id)
        elif report_config.output_format.lower() == "json":
            output_path = await _generate_json_report(report_sections, report_config, job_id)
        else:
            output_path = await _generate_csv_report(companies_data, report_config, job_id)
        
        # Compression si demandée
        if report_config.compress_output and output_path:
            output_path = await _compress_report(output_path)
        
        # Envoi email si configuré
        email_sent = False
        if report_config.send_email and report_config.email_recipients and output_path:
            email_sent = await _send_report_email(
                output_path, 
                report_config.email_recipients,
                f"Rapport d'analyse - {len(company_ids)} entreprises",
                user_id
            )
        
        execution_time = time.time() - start_time
        
        # Métadonnées du rapport
        report_metadata = {
            'report_id': job_id,
            'report_type': report_type,
            'companies_analyzed': len(companies_data),
            'sections_count': len(report_sections),
            'generation_time': execution_time,
            'output_format': report_config.output_format,
            'file_path': output_path,
            'file_size_mb': os.path.getsize(output_path) / (1024*1024) if output_path and os.path.exists(output_path) else 0,
            'email_sent': email_sent,
            'generated_at': datetime.now().isoformat(),
            'generated_by': user_id
        }
        
        # Cache des métadonnées
        cache_manager = await get_cache_manager()
        await cache_manager.set(
            'reports',
            f"metadata_{job_id}",
            report_metadata,
            ttl_seconds=86400 * 7  # 7 jours
        )
        
        logger.info(f"✅ Rapport généré: {output_path} ({report_metadata['file_size_mb']:.1f}MB) "
                   f"en {execution_time:.1f}s")
        
        return {
            'success': True,
            'job_id': job_id,
            'report_metadata': report_metadata,
            'analysis_summary': {
                'total_companies': len(companies_data),
                'avg_revenue': analysis_results.get('avg_revenue', 0),
                'top_sector': analysis_results.get('top_sector', 'N/A'),
                'high_score_companies': analysis_results.get('high_score_count', 0)
            },
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Erreur génération rapport: {e}")
        return {
            'success': False,
            'job_id': job_id,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }


@celery_task(priority=JobPriority.LOW)
async def generate_market_analysis_report(
    timeframe: str = "12_months",
    sectors: List[str] = None,
    config: Optional[Dict[str, Any]] = None,
    job_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Génère un rapport d'analyse de marché
    
    Args:
        timeframe: Période d'analyse (3_months, 6_months, 12_months, 24_months)
        sectors: Secteurs à analyser
        config: Configuration du rapport
        job_id: ID du job
    """
    start_time = time.time()
    
    logger.info(f"📈 Génération rapport marché: {timeframe}")
    
    try:
        # Générer données de marché simulées (remplacer par vraies données)
        market_data = await _generate_market_data(timeframe, sectors)
        
        # Analyses prédictives
        from app.services.predictive_analytics import get_predictive_analytics_engine, TimeHorizon
        analytics_engine = await get_predictive_analytics_engine()
        
        # Mapping timeframe
        horizon_map = {
            "3_months": TimeHorizon.SHORT_TERM,
            "6_months": TimeHorizon.MEDIUM_TERM, 
            "12_months": TimeHorizon.MEDIUM_TERM,
            "24_months": TimeHorizon.LONG_TERM
        }
        horizon = horizon_map.get(timeframe, TimeHorizon.MEDIUM_TERM)
        
        # Prédictions M&A
        ma_predictions = await analytics_engine.predict_ma_activity(horizon)
        
        # Prédictions sectorielles
        sector_predictions = {}
        if sectors:
            sector_predictions = await analytics_engine.predict_sector_performance(sectors, horizon)
        
        # Prévisions marché
        market_forecast = await analytics_engine.generate_market_forecast(horizon)
        
        # Construction du rapport marché
        report_sections = []
        
        # 1. Vue d'ensemble marché
        overview_section = ReportSection(
            title="Vue d'ensemble du marché M&A",
            content_type="summary",
            data={
                'timeframe': timeframe,
                'total_transactions': market_data.get('total_transactions', 0),
                'total_volume': market_data.get('total_volume', 0),
                'avg_valuation': market_data.get('avg_valuation', 0),
                'growth_rate': market_data.get('growth_rate', 0)
            },
            order=1
        )
        report_sections.append(overview_section)
        
        # 2. Tendances temporelles
        trends_section = ReportSection(
            title="Évolution des tendances",
            content_type="chart",
            data={
                'predictions': ma_predictions.predictions,
                'confidence': ma_predictions.confidence_score,
                'trend_direction': 'hausse' if ma_predictions.predictions[-1] > ma_predictions.predictions[0] else 'baisse'
            },
            order=2
        )
        report_sections.append(trends_section)
        
        # 3. Analyse sectorielle
        if sector_predictions:
            sector_section = ReportSection(
                title="Performance sectorielle",
                content_type="table",
                data=sector_predictions,
                order=3
            )
            report_sections.append(sector_section)
        
        # 4. Prévisions
        forecast_section = ReportSection(
            title="Prévisions de marché",
            content_type="summary",
            data={
                'forecast_horizon': horizon.value,
                'predicted_activity': market_forecast.overall_activity,
                'key_sectors': list(market_forecast.sector_predictions.keys())[:5],
                'market_sentiment': 'positif'  # Simplification
            },
            order=4
        )
        report_sections.append(forecast_section)
        
        # Générer rapport
        output_path = await _generate_pdf_report(
            report_sections, 
            ReportConfig(output_format="pdf", include_charts=True),
            job_id,
            title="Rapport d'Analyse de Marché M&A"
        )
        
        execution_time = time.time() - start_time
        
        return {
            'success': True,
            'job_id': job_id,
            'report_path': output_path,
            'analysis_period': timeframe,
            'sections_generated': len(report_sections),
            'execution_time': execution_time,
            'market_insights': {
                'predicted_growth': ma_predictions.predictions[-1] - ma_predictions.predictions[0] if ma_predictions.predictions else 0,
                'confidence_score': ma_predictions.confidence_score,
                'sectors_analyzed': len(sectors) if sectors else 0
            },
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Erreur rapport marché: {e}")
        return {
            'success': False,
            'job_id': job_id,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }


@celery_task(priority=JobPriority.LOW)
async def export_companies_data(
    filters: Dict[str, Any] = None,
    export_format: str = "excel",
    include_analytics: bool = True,
    config: Optional[Dict[str, Any]] = None,
    job_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Export massif de données d'entreprises
    
    Args:
        filters: Filtres à appliquer
        export_format: Format d'export (excel, csv, json)
        include_analytics: Inclure analyses et scores
        config: Configuration export
        job_id: ID du job
    """
    start_time = time.time()
    
    logger.info(f"📤 Export données entreprises: format {export_format}")
    
    try:
        # Récupérer données filtrées (simulation)
        companies_data = await _get_filtered_companies_data(filters or {})
        
        if not companies_data:
            return {
                'success': False,
                'error': 'Aucune donnée trouvée avec les filtres spécifiés',
                'timestamp': datetime.now().isoformat()
            }
        
        # Enrichir avec analytics si demandé
        if include_analytics:
            companies_data = await _enrich_companies_with_analytics(companies_data)
        
        # Génération selon format
        output_path = None
        
        if export_format.lower() == "excel":
            output_path = await _export_to_excel(companies_data, job_id, config)
        elif export_format.lower() == "csv":
            output_path = await _export_to_csv(companies_data, job_id, config)
        elif export_format.lower() == "json":
            output_path = await _export_to_json(companies_data, job_id, config)
        else:
            raise ValueError(f"Format d'export non supporté: {export_format}")
        
        # Statistiques export
        file_size_mb = os.path.getsize(output_path) / (1024*1024) if output_path and os.path.exists(output_path) else 0
        execution_time = time.time() - start_time
        
        return {
            'success': True,
            'job_id': job_id,
            'export_path': output_path,
            'export_format': export_format,
            'companies_exported': len(companies_data),
            'file_size_mb': file_size_mb,
            'execution_time': execution_time,
            'includes_analytics': include_analytics,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Erreur export données: {e}")
        return {
            'success': False,
            'job_id': job_id,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }


# Fonctions utilitaires internes

async def _collect_companies_data(company_ids: List[str]) -> List[Dict[str, Any]]:
    """Collecte les données des entreprises"""
    # Simulation de collection de données
    companies_data = []
    
    for siren in company_ids:
        # Ici on récupèrerait les vraies données depuis la DB
        company_data = {
            'siren': siren,
            'nom': f'Entreprise {siren[-3:]} SAS',
            'chiffre_affaires': np.random.lognormal(15, 1),
            'resultat_net': np.random.normal(500000, 200000),
            'effectifs': np.random.lognormal(3, 1),
            'secteur': np.random.choice(['tech', 'healthcare', 'finance', 'industry']),
            'region': np.random.choice(['IDF', 'PACA', 'AURA', 'HDF']),
            'score_ai': np.random.uniform(30, 95),
            'date_creation': f'20{np.random.randint(10, 23)}-{np.random.randint(1, 13):02d}-01'
        }
        companies_data.append(company_data)
    
    return companies_data


async def _analyze_companies_for_report(companies_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyse les données pour le rapport"""
    df = pd.DataFrame(companies_data)
    
    analysis = {
        'total_companies': len(companies_data),
        'avg_revenue': df['chiffre_affaires'].mean(),
        'median_revenue': df['chiffre_affaires'].median(),
        'total_revenue': df['chiffre_affaires'].sum(),
        'avg_employees': df['effectifs'].mean(),
        'top_sector': df['secteur'].mode().iloc[0] if len(df) > 0 else 'N/A',
        'sector_distribution': df['secteur'].value_counts().to_dict(),
        'region_distribution': df['region'].value_counts().to_dict(),
        'high_score_count': len(df[df['score_ai'] >= 80]),
        'avg_score': df['score_ai'].mean(),
        'score_distribution': {
            'excellent': len(df[df['score_ai'] >= 90]),
            'good': len(df[(df['score_ai'] >= 70) & (df['score_ai'] < 90)]),
            'average': len(df[(df['score_ai'] >= 50) & (df['score_ai'] < 70)]),
            'low': len(df[df['score_ai'] < 50])
        }
    }
    
    return analysis


async def _create_executive_summary(analysis: Dict[str, Any], companies_data: List[Dict[str, Any]]) -> ReportSection:
    """Crée le résumé exécutif"""
    summary_text = f"""
    Analyse de {analysis['total_companies']} entreprises:
    
    • Chiffre d'affaires moyen: {analysis['avg_revenue']:,.0f} €
    • Secteur principal: {analysis['top_sector']}
    • Score IA moyen: {analysis['avg_score']:.1f}/100
    • Entreprises à fort potentiel: {analysis['high_score_count']} ({analysis['high_score_count']/analysis['total_companies']*100:.1f}%)
    
    Recommandation: Focus sur les {analysis['score_distribution']['excellent']} entreprises avec score excellent.
    """
    
    return ReportSection(
        title="Résumé Exécutif",
        content_type="text",
        data=summary_text.strip(),
        order=1
    )


async def _create_financial_analysis(analysis: Dict[str, Any], config: ReportConfig) -> ReportSection:
    """Crée l'analyse financière"""
    financial_data = {
        'revenue_stats': {
            'mean': analysis['avg_revenue'],
            'median': analysis['median_revenue'],
            'total': analysis['total_revenue']
        },
        'employee_stats': {
            'average': analysis['avg_employees']
        },
        'growth_potential': 'Élevé' if analysis['avg_score'] > 70 else 'Modéré'
    }
    
    return ReportSection(
        title="Analyse Financière",
        content_type="table",
        data=financial_data,
        order=2
    )


async def _create_sector_analysis(analysis: Dict[str, Any], config: ReportConfig) -> ReportSection:
    """Crée l'analyse sectorielle"""
    return ReportSection(
        title="Répartition Sectorielle",
        content_type="chart",
        data=analysis['sector_distribution'],
        order=3
    )


async def _create_scoring_analysis(companies_data: List[Dict[str, Any]], config: ReportConfig) -> ReportSection:
    """Crée l'analyse des scores IA"""
    df = pd.DataFrame(companies_data)
    
    scoring_data = {
        'score_distribution': df['score_ai'].describe().to_dict(),
        'top_companies': df.nlargest(10, 'score_ai')[['siren', 'nom', 'score_ai']].to_dict('records'),
        'recommendations': [
            'Prioriser les entreprises avec score > 80',
            'Analyser en détail le top 10',
            'Surveiller les entreprises score 60-80'
        ]
    }
    
    return ReportSection(
        title="Analyse des Scores IA",
        content_type="table",
        data=scoring_data,
        order=4
    )


async def _create_company_details(companies_data: List[Dict[str, Any]], config: ReportConfig) -> ReportSection:
    """Crée les détails par entreprise"""
    # Limiter aux entreprises les mieux scorées
    df = pd.DataFrame(companies_data)
    top_companies = df.nlargest(min(20, len(df)), 'score_ai')
    
    return ReportSection(
        title="Détail des Entreprises (Top 20)",
        content_type="table",
        data=top_companies.to_dict('records'),
        order=5
    )


async def _generate_pdf_report(
    sections: List[ReportSection], 
    config: ReportConfig, 
    job_id: str,
    title: str = "Rapport d'Analyse"
) -> str:
    """Génère un rapport PDF"""
    
    # Créer fichier temporaire
    output_dir = Path(tempfile.gettempdir()) / "reports"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"report_{job_id}_{int(time.time())}.pdf"
    
    # Créer document PDF
    doc = SimpleDocTemplate(str(output_path), pagesize=A4)
    story = []
    styles = getSampleStyleSheet()
    
    # Titre
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=18,
        alignment=1,  # Center
        spaceAfter=30
    )
    story.append(Paragraph(title, title_style))
    story.append(Spacer(1, 12))
    
    # Générer sections
    for section in sorted(sections, key=lambda x: x.order):
        # Titre de section
        story.append(Paragraph(section.title, styles['Heading2']))
        story.append(Spacer(1, 12))
        
        # Contenu selon type
        if section.content_type == "text":
            story.append(Paragraph(str(section.data), styles['Normal']))
            
        elif section.content_type == "table":
            if isinstance(section.data, dict):
                # Convertir dict en table
                table_data = [['Métrique', 'Valeur']]
                for key, value in section.data.items():
                    if isinstance(value, dict):
                        for subkey, subvalue in value.items():
                            table_data.append([f"{key} - {subkey}", str(subvalue)])
                    else:
                        table_data.append([key, str(value)])
                
                table = Table(table_data)
                table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 14),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                story.append(table)
            
        elif section.content_type == "chart" and config.include_charts:
            # Générer graphique simple (placeholder)
            story.append(Paragraph("📊 Graphique généré", styles['Normal']))
            
        elif section.content_type == "summary":
            story.append(Paragraph(str(section.data), styles['Normal']))
        
        story.append(Spacer(1, 20))
    
    # Footer
    story.append(Spacer(1, 30))
    footer_text = f"Généré le {datetime.now().strftime('%d/%m/%Y à %H:%M')} par M&A Intelligence Platform"
    story.append(Paragraph(footer_text, styles['Normal']))
    
    # Construire PDF
    doc.build(story)
    
    logger.info(f"📄 Rapport PDF généré: {output_path}")
    return str(output_path)


async def _generate_excel_report(sections: List[ReportSection], config: ReportConfig, job_id: str) -> str:
    """Génère un rapport Excel"""
    
    output_dir = Path(tempfile.gettempdir()) / "reports"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"report_{job_id}_{int(time.time())}.xlsx"
    
    with pd.ExcelWriter(str(output_path), engine='openpyxl') as writer:
        # Feuille de résumé
        summary_data = []
        for section in sections:
            summary_data.append({
                'Section': section.title,
                'Type': section.content_type,
                'Ordre': section.order
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Résumé', index=False)
        
        # Feuilles par section
        for section in sections:
            sheet_name = section.title[:30]  # Limiter nom feuille
            
            if section.content_type == "table" and isinstance(section.data, dict):
                if 'top_companies' in section.data:
                    df = pd.DataFrame(section.data['top_companies'])
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
                else:
                    # Convertir dict en DataFrame
                    df = pd.DataFrame(list(section.data.items()), columns=['Métrique', 'Valeur'])
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
            
            elif section.content_type == "chart":
                df = pd.DataFrame(list(section.data.items()), columns=['Catégorie', 'Valeur'])
                df.to_excel(writer, sheet_name=sheet_name, index=False)
    
    logger.info(f"📊 Rapport Excel généré: {output_path}")
    return str(output_path)


async def _generate_json_report(sections: List[ReportSection], config: ReportConfig, job_id: str) -> str:
    """Génère un rapport JSON"""
    
    output_dir = Path(tempfile.gettempdir()) / "reports"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"report_{job_id}_{int(time.time())}.json"
    
    report_data = {
        'metadata': {
            'generated_at': datetime.now().isoformat(),
            'job_id': job_id,
            'format': 'json',
            'sections_count': len(sections)
        },
        'sections': []
    }
    
    for section in sections:
        section_data = {
            'title': section.title,
            'content_type': section.content_type,
            'order': section.order,
            'data': section.data
        }
        report_data['sections'].append(section_data)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False, default=str)
    
    logger.info(f"📄 Rapport JSON généré: {output_path}")
    return str(output_path)


async def _generate_csv_report(companies_data: List[Dict[str, Any]], config: ReportConfig, job_id: str) -> str:
    """Génère un rapport CSV simple"""
    
    output_dir = Path(tempfile.gettempdir()) / "reports"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"companies_{job_id}_{int(time.time())}.csv"
    
    df = pd.DataFrame(companies_data)
    df.to_csv(output_path, index=False, encoding='utf-8')
    
    logger.info(f"📄 Rapport CSV généré: {output_path}")
    return str(output_path)


async def _compress_report(file_path: str) -> str:
    """Compresse un rapport"""
    
    zip_path = file_path + ".zip"
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        zip_file.write(file_path, os.path.basename(file_path))
    
    # Supprimer fichier original
    os.remove(file_path)
    
    logger.info(f"🗜️ Rapport compressé: {zip_path}")
    return zip_path


async def _send_report_email(file_path: str, recipients: List[str], subject: str, user_id: str) -> bool:
    """Envoie le rapport par email (simulation)"""
    try:
        # Simulation envoi email
        file_size_mb = os.path.getsize(file_path) / (1024*1024)
        
        logger.info(f"📧 Email simulé envoyé à {len(recipients)} destinataires:")
        logger.info(f"   Sujet: {subject}")
        logger.info(f"   Fichier: {os.path.basename(file_path)} ({file_size_mb:.1f}MB)")
        logger.info(f"   Expéditeur: {user_id}")
        
        return True
        
    except Exception as e:
        logger.error(f"Erreur envoi email: {e}")
        return False


async def _generate_market_data(timeframe: str, sectors: List[str] = None) -> Dict[str, Any]:
    """Génère des données de marché simulées"""
    
    # Simulation de données
    base_transactions = 1000
    multiplier = {'3_months': 0.25, '6_months': 0.5, '12_months': 1.0, '24_months': 2.0}
    
    return {
        'total_transactions': int(base_transactions * multiplier.get(timeframe, 1.0)),
        'total_volume': np.random.lognormal(18, 1.5),  # En euros
        'avg_valuation': np.random.lognormal(16, 1),
        'growth_rate': np.random.uniform(-0.1, 0.3),
        'sectors_analyzed': sectors or ['tech', 'healthcare', 'finance']
    }


async def _get_filtered_companies_data(filters: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Récupère données filtrées (simulation)"""
    
    # Simulation de 1000 entreprises
    companies = []
    for i in range(1000):
        company = {
            'siren': f'12345{i:04d}',
            'nom': f'Entreprise {i+1} SAS',
            'chiffre_affaires': np.random.lognormal(15, 1.5),
            'resultat_net': np.random.normal(100000, 500000),
            'effectifs': np.random.lognormal(3, 1),
            'secteur': np.random.choice(['tech', 'healthcare', 'finance', 'industry', 'services']),
            'region': np.random.choice(['IDF', 'PACA', 'AURA', 'HDF', 'NAQ']),
            'score_ai': np.random.uniform(20, 95),
            'created_at': datetime.now() - timedelta(days=np.random.randint(30, 3650))
        }
        companies.append(company)
    
    # Appliquer filtres basiques
    if filters.get('min_ca'):
        companies = [c for c in companies if c['chiffre_affaires'] >= filters['min_ca']]
    
    if filters.get('secteur'):
        companies = [c for c in companies if c['secteur'] == filters['secteur']]
    
    if filters.get('min_score'):
        companies = [c for c in companies if c['score_ai'] >= filters['min_score']]
    
    return companies


async def _enrich_companies_with_analytics(companies_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Enrichit les données avec analyses ML"""
    
    enriched_data = []
    
    for company in companies_data:
        # Ajouter analyses simulées
        enriched_company = company.copy()
        enriched_company.update({
            'growth_prediction': np.random.uniform(-0.2, 0.4),
            'risk_score': np.random.uniform(0.1, 0.9),
            'market_position': np.random.choice(['leader', 'challenger', 'follower']),
            'recommendation': np.random.choice(['buy', 'hold', 'watch', 'avoid']),
            'confidence_level': np.random.uniform(0.6, 0.95)
        })
        enriched_data.append(enriched_company)
    
    return enriched_data


async def _export_to_excel(companies_data: List[Dict[str, Any]], job_id: str, config: Dict[str, Any]) -> str:
    """Export Excel avec mise en forme"""
    
    output_dir = Path(tempfile.gettempdir()) / "exports"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"export_{job_id}_{int(time.time())}.xlsx"
    
    df = pd.DataFrame(companies_data)
    
    with pd.ExcelWriter(str(output_path), engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Entreprises', index=False)
        
        # Feuille statistiques
        stats = {
            'Total entreprises': len(df),
            'CA moyen': df['chiffre_affaires'].mean() if 'chiffre_affaires' in df.columns else 0,
            'Score moyen': df['score_ai'].mean() if 'score_ai' in df.columns else 0,
        }
        stats_df = pd.DataFrame(list(stats.items()), columns=['Métrique', 'Valeur'])
        stats_df.to_excel(writer, sheet_name='Statistiques', index=False)
    
    return str(output_path)


async def _export_to_csv(companies_data: List[Dict[str, Any]], job_id: str, config: Dict[str, Any]) -> str:
    """Export CSV"""
    
    output_dir = Path(tempfile.gettempdir()) / "exports"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"export_{job_id}_{int(time.time())}.csv"
    
    df = pd.DataFrame(companies_data)
    df.to_csv(output_path, index=False, encoding='utf-8', sep=';')
    
    return str(output_path)


async def _export_to_json(companies_data: List[Dict[str, Any]], job_id: str, config: Dict[str, Any]) -> str:
    """Export JSON"""
    
    output_dir = Path(tempfile.gettempdir()) / "exports"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"export_{job_id}_{int(time.time())}.json"
    
    export_data = {
        'metadata': {
            'exported_at': datetime.now().isoformat(),
            'job_id': job_id,
            'total_companies': len(companies_data)
        },
        'companies': companies_data
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)
    
    return str(output_path)


# Fonctions utilitaires pour l'API

async def submit_report_generation_job(
    report_type: str,
    parameters: Dict[str, Any],
    user_id: Optional[str] = None,
    priority: JobPriority = JobPriority.NORMAL
) -> str:
    """Soumet un job de génération de rapport"""
    
    job_manager = await get_background_job_manager()
    
    # Sélectionner tâche selon type
    if report_type == "company_analysis":
        task_name = 'generate_company_analysis_report'
        estimated_duration = len(parameters.get('company_ids', [])) * 0.1
    elif report_type == "market_analysis":
        task_name = 'generate_market_analysis_report'
        estimated_duration = 300  # 5 minutes
    elif report_type == "data_export":
        task_name = 'export_companies_data'
        estimated_duration = 120  # 2 minutes
    else:
        raise ValueError(f"Type de rapport non supporté: {report_type}")
    
    job_id = await job_manager.submit_job(
        task_name=task_name,
        args=[],
        kwargs=parameters,
        priority=priority,
        job_type=JobType.REPORT_GENERATION,
        estimated_duration=estimated_duration,
        user_id=user_id,
        context={
            'report_type': report_type,
            'parameters': parameters
        }
    )
    
    logger.info(f"📤 Job rapport soumis: {job_id} ({report_type})")
    return job_id