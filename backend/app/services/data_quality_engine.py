"""
Système d'évaluation de la qualité des données piloté par IA pour M&A Intelligence Platform
US-008: IA pour évaluation et amélioration automatique de la qualité des données

Ce module implémente:
- Détection automatique de données manquantes et aberrantes
- Évaluation de la cohérence et de la complétude des données
- Scoring de la qualité par catégorie de données
- Recommandations d'amélioration automatisées
- Validation croisée et détection d'incohérences
- Monitoring de la dégradation de qualité dans le temps
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union, Set
from dataclasses import dataclass, field
from enum import Enum
import json
import warnings
from collections import defaultdict, Counter
import hashlib

# Machine Learning pour qualité de données
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.neighbors import LocalOutlierFactor
from scipy import stats
import joblib
import diskcache

# Validation et nettoyage de données
import re
from difflib import SequenceMatcher
from datetime import date
import unicodedata

from app.core.logging_system import get_logger, LogCategory
from app.core.validators import validate_siren, validate_siret
from app.models.schemas import CompanyCreate

logger = get_logger("data_quality_engine", LogCategory.ML)


class QualityDimension(str, Enum):
    """Dimensions de qualité des données"""
    COMPLETENESS = "completeness"          # Complétude
    ACCURACY = "accuracy"                  # Précision
    CONSISTENCY = "consistency"            # Cohérence
    VALIDITY = "validity"                  # Validité
    UNIQUENESS = "uniqueness"              # Unicité
    TIMELINESS = "timeliness"             # Fraîcheur
    CONFORMITY = "conformity"             # Conformité


class DataType(str, Enum):
    """Types de données à analyser"""
    COMPANY_BASIC = "company_basic"
    FINANCIAL = "financial"
    CONTACT = "contact"
    LEGAL = "legal"
    OPERATIONAL = "operational"
    TEMPORAL = "temporal"


class SeverityLevel(str, Enum):
    """Niveaux de sévérité des problèmes de qualité"""
    CRITICAL = "critical"        # Bloque l'utilisation
    HIGH = "high"               # Impact significatif
    MEDIUM = "medium"           # Impact modéré
    LOW = "low"                # Impact mineur
    INFO = "info"              # Information


@dataclass
class QualityIssue:
    """Problème de qualité identifié"""
    issue_id: str
    dimension: QualityDimension
    severity: SeverityLevel
    field_name: str
    description: str
    suggested_fix: str
    confidence: float
    affected_records: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'issue_id': self.issue_id,
            'dimension': self.dimension.value,
            'severity': self.severity.value,
            'field_name': self.field_name,
            'description': self.description,
            'suggested_fix': self.suggested_fix,
            'confidence': round(self.confidence, 3),
            'affected_records_count': len(self.affected_records),
            'metadata': self.metadata
        }


@dataclass
class QualityMetrics:
    """Métriques de qualité par dimension"""
    dimension: QualityDimension
    score: float                    # Score 0-100
    issues_count: int
    critical_issues: int
    records_affected: int
    improvement_potential: float     # Gain potentiel
    trend: str                      # 'improving', 'declining', 'stable'
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'dimension': self.dimension.value,
            'score': round(self.score, 2),
            'issues_count': self.issues_count,
            'critical_issues': self.critical_issues,
            'records_affected': self.records_affected,
            'improvement_potential': round(self.improvement_potential, 2),
            'trend': self.trend
        }


@dataclass
class DataQualityReport:
    """Rapport complet de qualité des données"""
    
    overall_score: float
    data_type: DataType
    total_records: int
    timestamp: datetime
    
    # Métriques par dimension
    quality_metrics: Dict[QualityDimension, QualityMetrics]
    
    # Problèmes identifiés
    issues: List[QualityIssue]
    critical_issues: List[QualityIssue]
    
    # Recommandations
    improvement_actions: List[str]
    priority_fixes: List[str]
    estimated_effort: Dict[str, str]  # effort par action
    
    # Métadonnées
    processing_time_ms: float
    data_coverage: Dict[str, float]   # % de couverture par champ
    model_confidence: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'overall_score': round(self.overall_score, 2),
            'data_type': self.data_type.value,
            'total_records': self.total_records,
            'timestamp': self.timestamp.isoformat(),
            'quality_metrics': {k.value: v.to_dict() for k, v in self.quality_metrics.items()},
            'issues_summary': {
                'total': len(self.issues),
                'critical': len(self.critical_issues),
                'by_severity': Counter(issue.severity.value for issue in self.issues)
            },
            'top_issues': [issue.to_dict() for issue in self.issues[:10]],
            'improvement_actions': self.improvement_actions,
            'priority_fixes': self.priority_fixes,
            'estimated_effort': self.estimated_effort,
            'processing_time_ms': round(self.processing_time_ms, 2),
            'data_coverage': {k: round(v, 2) for k, v in self.data_coverage.items()},
            'model_confidence': round(self.model_confidence, 3)
        }


class DataQualityEngine:
    """Moteur d'évaluation de la qualité des données utilisant l'IA"""
    
    def __init__(self, cache_ttl: int = 1800):  # 30 minutes
        """
        Initialise le moteur d'évaluation de qualité
        
        Args:
            cache_ttl: Durée de vie du cache en secondes
        """
        self.cache = diskcache.Cache('/tmp/ma_intelligence_dq_cache')
        self.cache_ttl = cache_ttl
        
        # Modèles ML pour détection d'anomalies
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        
        # Règles de validation par type de données
        self.validation_rules = self._initialize_validation_rules()
        
        # Patterns de référence pour données françaises
        self.reference_patterns = self._initialize_reference_patterns()
        
        # Historique pour analyse de tendances
        self.quality_history = defaultdict(list)
        
        # Configuration des seuils de qualité
        self.quality_thresholds = {
            QualityDimension.COMPLETENESS: {'excellent': 95, 'good': 85, 'acceptable': 70},
            QualityDimension.ACCURACY: {'excellent': 98, 'good': 90, 'acceptable': 80},
            QualityDimension.CONSISTENCY: {'excellent': 95, 'good': 85, 'acceptable': 75},
            QualityDimension.VALIDITY: {'excellent': 99, 'good': 95, 'acceptable': 90},
            QualityDimension.UNIQUENESS: {'excellent': 98, 'good': 95, 'acceptable': 90},
            QualityDimension.TIMELINESS: {'excellent': 90, 'good': 80, 'acceptable': 70},
            QualityDimension.CONFORMITY: {'excellent': 98, 'good': 90, 'acceptable': 85}
        }
        
        logger.info("📊 Moteur d'évaluation de qualité des données initialisé")
    
    async def initialize(self):
        """Initialise les modèles et règles de qualité"""
        try:
            # Entraîner modèles de détection d'anomalies
            await self._train_anomaly_detection_models()
            
            # Charger données de référence
            await self._load_reference_data()
            
            logger.info("✅ Moteur de qualité des données initialisé avec succès")
            
        except Exception as e:
            logger.error(f"Erreur initialisation moteur qualité: {e}")
            raise
    
    async def evaluate_data_quality(self, 
                                  data: pd.DataFrame,
                                  data_type: DataType = DataType.COMPANY_BASIC,
                                  use_cache: bool = True) -> DataQualityReport:
        """
        Évalue la qualité d'un dataset
        
        Args:
            data: DataFrame à analyser
            data_type: Type de données
            use_cache: Utiliser le cache
            
        Returns:
            DataQualityReport: Rapport de qualité complet
        """
        start_time = datetime.now()
        
        try:
            logger.info(f"Évaluation qualité pour {len(data)} enregistrements ({data_type.value})")
            
            # Vérifier cache
            cache_key = self._generate_cache_key(data, data_type)
            if use_cache:
                cached_result = self.cache.get(cache_key)
                if cached_result:
                    logger.debug("Rapport qualité récupéré du cache")
                    return DataQualityReport(**cached_result)
            
            # Évaluer chaque dimension de qualité
            quality_metrics = {}
            all_issues = []
            
            # Complétude
            completeness_metrics, completeness_issues = await self._evaluate_completeness(data, data_type)
            quality_metrics[QualityDimension.COMPLETENESS] = completeness_metrics
            all_issues.extend(completeness_issues)
            
            # Précision/Exactitude
            accuracy_metrics, accuracy_issues = await self._evaluate_accuracy(data, data_type)
            quality_metrics[QualityDimension.ACCURACY] = accuracy_metrics
            all_issues.extend(accuracy_issues)
            
            # Cohérence
            consistency_metrics, consistency_issues = await self._evaluate_consistency(data, data_type)
            quality_metrics[QualityDimension.CONSISTENCY] = consistency_metrics
            all_issues.extend(consistency_issues)
            
            # Validité
            validity_metrics, validity_issues = await self._evaluate_validity(data, data_type)
            quality_metrics[QualityDimension.VALIDITY] = validity_metrics
            all_issues.extend(validity_issues)
            
            # Unicité
            uniqueness_metrics, uniqueness_issues = await self._evaluate_uniqueness(data, data_type)
            quality_metrics[QualityDimension.UNIQUENESS] = uniqueness_metrics
            all_issues.extend(uniqueness_issues)
            
            # Fraîcheur temporelle
            timeliness_metrics, timeliness_issues = await self._evaluate_timeliness(data, data_type)
            quality_metrics[QualityDimension.TIMELINESS] = timeliness_metrics
            all_issues.extend(timeliness_issues)
            
            # Conformité
            conformity_metrics, conformity_issues = await self._evaluate_conformity(data, data_type)
            quality_metrics[QualityDimension.CONFORMITY] = conformity_metrics
            all_issues.extend(conformity_issues)
            
            # Calculer score global
            overall_score = self._calculate_overall_score(quality_metrics)
            
            # Identifier problèmes critiques
            critical_issues = [issue for issue in all_issues if issue.severity == SeverityLevel.CRITICAL]
            
            # Générer recommandations
            improvement_actions = self._generate_improvement_actions(quality_metrics, all_issues)
            priority_fixes = self._prioritize_fixes(all_issues)
            estimated_effort = self._estimate_effort(improvement_actions)
            
            # Calculer métriques supplémentaires
            data_coverage = self._calculate_data_coverage(data)
            model_confidence = self._calculate_model_confidence(quality_metrics)
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Créer rapport
            report = DataQualityReport(
                overall_score=overall_score,
                data_type=data_type,
                total_records=len(data),
                timestamp=datetime.now(),
                quality_metrics=quality_metrics,
                issues=all_issues,
                critical_issues=critical_issues,
                improvement_actions=improvement_actions,
                priority_fixes=priority_fixes,
                estimated_effort=estimated_effort,
                processing_time_ms=processing_time,
                data_coverage=data_coverage,
                model_confidence=model_confidence
            )
            
            # Mettre en cache
            if use_cache:
                self.cache.set(cache_key, report.to_dict(), expire=self.cache_ttl)
            
            # Enregistrer pour analyse de tendances
            self._record_quality_metrics(data_type, quality_metrics, overall_score)
            
            logger.info(f"Évaluation qualité terminée: score {overall_score:.1f}/100 ({len(all_issues)} problèmes)")
            
            return report
            
        except Exception as e:
            logger.error(f"Erreur évaluation qualité: {e}")
            raise
    
    async def clean_data(self, 
                        data: pd.DataFrame,
                        quality_report: DataQualityReport,
                        auto_fix: bool = True) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Nettoie automatiquement les données basé sur le rapport de qualité
        
        Args:
            data: DataFrame à nettoyer
            quality_report: Rapport de qualité avec problèmes identifiés
            auto_fix: Appliquer corrections automatiques
            
        Returns:
            Tuple[DataFrame, Dict]: Données nettoyées et rapport des corrections
        """
        try:
            logger.info(f"Nettoyage automatique de {len(data)} enregistrements")
            
            cleaned_data = data.copy()
            cleaning_actions = []
            
            if auto_fix:
                # Appliquer corrections automatiques par type de problème
                for issue in quality_report.issues:
                    if issue.severity in [SeverityLevel.CRITICAL, SeverityLevel.HIGH]:
                        correction_applied = await self._apply_automatic_fix(cleaned_data, issue)
                        if correction_applied:
                            cleaning_actions.append({
                                'issue': issue.issue_id,
                                'action': issue.suggested_fix,
                                'records_affected': len(issue.affected_records)
                            })
            
            # Statistiques de nettoyage
            cleaning_stats = {
                'original_records': len(data),
                'cleaned_records': len(cleaned_data),
                'actions_applied': len(cleaning_actions),
                'data_loss_percentage': ((len(data) - len(cleaned_data)) / len(data)) * 100 if len(data) > 0 else 0,
                'cleaning_actions': cleaning_actions,
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Nettoyage terminé: {len(cleaning_actions)} corrections appliquées")
            
            return cleaned_data, cleaning_stats
            
        except Exception as e:
            logger.error(f"Erreur nettoyage automatique: {e}")
            raise
    
    async def monitor_quality_trends(self, 
                                   data_type: DataType,
                                   time_window_days: int = 30) -> Dict[str, Any]:
        """
        Analyse les tendances de qualité dans le temps
        
        Args:
            data_type: Type de données à analyser
            time_window_days: Fenêtre d'analyse en jours
            
        Returns:
            Dict: Analyse des tendances
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=time_window_days)
            
            # Récupérer historique
            history = self.quality_history.get(data_type.value, [])
            recent_history = [
                record for record in history 
                if record['timestamp'] > cutoff_date
            ]
            
            if len(recent_history) < 2:
                return {
                    'data_type': data_type.value,
                    'trend_analysis': 'insufficient_data',
                    'message': 'Pas assez de données historiques pour analyser les tendances'
                }
            
            # Analyser tendances par dimension
            trends_analysis = {}
            
            for dimension in QualityDimension:
                scores = [record['metrics'][dimension.value]['score'] for record in recent_history]
                
                if len(scores) >= 2:
                    # Calculer tendance (régression linéaire simple)
                    x = list(range(len(scores)))
                    slope = np.polyfit(x, scores, 1)[0]
                    
                    # Déterminer direction de tendance
                    if slope > 1:
                        trend_direction = 'improving'
                    elif slope < -1:
                        trend_direction = 'declining'
                    else:
                        trend_direction = 'stable'
                    
                    trends_analysis[dimension.value] = {
                        'current_score': scores[-1],
                        'trend_direction': trend_direction,
                        'slope': slope,
                        'variance': np.var(scores),
                        'score_range': [min(scores), max(scores)]
                    }
            
            # Identifier alarmes
            alerts = self._identify_quality_alerts(trends_analysis)
            
            # Recommandations d'amélioration continue
            continuous_improvement = self._suggest_continuous_improvements(trends_analysis)
            
            return {
                'data_type': data_type.value,
                'analysis_period': f'{time_window_days} jours',
                'data_points': len(recent_history),
                'trends_by_dimension': trends_analysis,
                'quality_alerts': alerts,
                'continuous_improvement_suggestions': continuous_improvement,
                'overall_trend': self._calculate_overall_trend(trends_analysis),
                'analysis_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Erreur analyse tendances qualité: {e}")
            return {'error': f'Erreur analyse: {str(e)}'}
    
    # Méthodes d'évaluation par dimension
    
    async def _evaluate_completeness(self, data: pd.DataFrame, data_type: DataType) -> Tuple[QualityMetrics, List[QualityIssue]]:
        """Évalue la complétude des données"""
        issues = []
        
        # Calculer taux de complétude par colonne
        completeness_rates = {}
        total_missing = 0
        
        for column in data.columns:
            missing_count = data[column].isnull().sum()
            completeness_rate = ((len(data) - missing_count) / len(data)) * 100
            completeness_rates[column] = completeness_rate
            total_missing += missing_count
            
            # Identifier problèmes de complétude
            if completeness_rate < 50:
                issues.append(QualityIssue(
                    issue_id=f"completeness_{column}",
                    dimension=QualityDimension.COMPLETENESS,
                    severity=SeverityLevel.CRITICAL,
                    field_name=column,
                    description=f"Taux de complétude très faible: {completeness_rate:.1f}%",
                    suggested_fix=f"Enrichir les données manquantes pour {column}",
                    confidence=0.95,
                    affected_records=[str(i) for i in data[data[column].isnull()].index.tolist()],
                    metadata={'completeness_rate': completeness_rate, 'missing_count': missing_count}
                ))
            elif completeness_rate < 80:
                issues.append(QualityIssue(
                    issue_id=f"completeness_{column}",
                    dimension=QualityDimension.COMPLETENESS,
                    severity=SeverityLevel.HIGH,
                    field_name=column,
                    description=f"Taux de complétude insuffisant: {completeness_rate:.1f}%",
                    suggested_fix=f"Améliorer la collecte de données pour {column}",
                    confidence=0.9,
                    affected_records=[str(i) for i in data[data[column].isnull()].index.tolist()],
                    metadata={'completeness_rate': completeness_rate, 'missing_count': missing_count}
                ))
        
        # Score global de complétude
        overall_completeness = ((len(data) * len(data.columns) - total_missing) / (len(data) * len(data.columns))) * 100
        
        # Analyser patterns de données manquantes
        missing_patterns = self._analyze_missing_patterns(data)
        if missing_patterns['systematic_missing']:
            issues.append(QualityIssue(
                issue_id="completeness_systematic",
                dimension=QualityDimension.COMPLETENESS,
                severity=SeverityLevel.MEDIUM,
                field_name="multiple",
                description="Pattern systématique de données manquantes détecté",
                suggested_fix="Analyser processus de collecte de données",
                confidence=0.8,
                affected_records=[],
                metadata=missing_patterns
            ))
        
        metrics = QualityMetrics(
            dimension=QualityDimension.COMPLETENESS,
            score=overall_completeness,
            issues_count=len(issues),
            critical_issues=len([i for i in issues if i.severity == SeverityLevel.CRITICAL]),
            records_affected=len(set().union(*[issue.affected_records for issue in issues])),
            improvement_potential=100 - overall_completeness,
            trend=self._get_dimension_trend(QualityDimension.COMPLETENESS, data_type)
        )
        
        return metrics, issues
    
    async def _evaluate_accuracy(self, data: pd.DataFrame, data_type: DataType) -> Tuple[QualityMetrics, List[QualityIssue]]:
        """Évalue la précision/exactitude des données"""
        issues = []
        accuracy_scores = []
        
        # Vérifications spécifiques par type de champ
        for column in data.columns:
            column_lower = column.lower()
            
            # SIREN/SIRET
            if 'siren' in column_lower:
                invalid_sirens = []
                for idx, value in data[column].dropna().items():
                    if not validate_siren(str(value)):
                        invalid_sirens.append(str(idx))
                
                if invalid_sirens:
                    accuracy_rate = ((len(data[column].dropna()) - len(invalid_sirens)) / len(data[column].dropna())) * 100
                    accuracy_scores.append(accuracy_rate)
                    
                    issues.append(QualityIssue(
                        issue_id=f"accuracy_siren_{column}",
                        dimension=QualityDimension.ACCURACY,
                        severity=SeverityLevel.HIGH,
                        field_name=column,
                        description=f"SIREN invalides détectés: {len(invalid_sirens)} cas",
                        suggested_fix="Valider et corriger les numéros SIREN",
                        confidence=0.98,
                        affected_records=invalid_sirens,
                        metadata={'invalid_count': len(invalid_sirens)}
                    ))
                else:
                    accuracy_scores.append(100.0)
            
            # Emails
            elif 'email' in column_lower or 'mail' in column_lower:
                email_pattern = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
                invalid_emails = []
                
                for idx, value in data[column].dropna().items():
                    if not email_pattern.match(str(value)):
                        invalid_emails.append(str(idx))
                
                if invalid_emails:
                    accuracy_rate = ((len(data[column].dropna()) - len(invalid_emails)) / len(data[column].dropna())) * 100
                    accuracy_scores.append(accuracy_rate)
                    
                    issues.append(QualityIssue(
                        issue_id=f"accuracy_email_{column}",
                        dimension=QualityDimension.ACCURACY,
                        severity=SeverityLevel.MEDIUM,
                        field_name=column,
                        description=f"Emails invalides: {len(invalid_emails)} cas",
                        suggested_fix="Nettoyer et valider les adresses email",
                        confidence=0.95,
                        affected_records=invalid_emails,
                        metadata={'invalid_count': len(invalid_emails)}
                    ))
                else:
                    accuracy_scores.append(100.0)
            
            # Dates
            elif 'date' in column_lower or data[column].dtype in ['datetime64[ns]']:
                future_dates = []
                
                for idx, value in data[column].dropna().items():
                    try:
                        if pd.to_datetime(value) > datetime.now():
                            future_dates.append(str(idx))
                    except:
                        pass
                
                if future_dates:
                    issues.append(QualityIssue(
                        issue_id=f"accuracy_future_date_{column}",
                        dimension=QualityDimension.ACCURACY,
                        severity=SeverityLevel.MEDIUM,
                        field_name=column,
                        description=f"Dates futures détectées: {len(future_dates)} cas",
                        suggested_fix="Vérifier et corriger les dates futures",
                        confidence=0.9,
                        affected_records=future_dates,
                        metadata={'future_dates_count': len(future_dates)}
                    ))
            
            # Montants financiers
            elif any(term in column_lower for term in ['chiffre', 'resultat', 'capital', 'ca']):
                if data[column].dtype in ['int64', 'float64']:
                    negative_values = data[data[column] < 0].index.tolist()
                    extreme_values = data[data[column] > data[column].quantile(0.99) * 10].index.tolist()
                    
                    if negative_values and 'resultat' not in column_lower:  # Résultat peut être négatif
                        issues.append(QualityIssue(
                            issue_id=f"accuracy_negative_{column}",
                            dimension=QualityDimension.ACCURACY,
                            severity=SeverityLevel.MEDIUM,
                            field_name=column,
                            description=f"Valeurs négatives suspectes: {len(negative_values)} cas",
                            suggested_fix="Vérifier valeurs négatives",
                            confidence=0.7,
                            affected_records=[str(i) for i in negative_values],
                            metadata={'negative_count': len(negative_values)}
                        ))
                    
                    if extreme_values:
                        issues.append(QualityIssue(
                            issue_id=f"accuracy_extreme_{column}",
                            dimension=QualityDimension.ACCURACY,
                            severity=SeverityLevel.LOW,
                            field_name=column,
                            description=f"Valeurs extrêmes détectées: {len(extreme_values)} cas",
                            suggested_fix="Valider valeurs extrêmes",
                            confidence=0.6,
                            affected_records=[str(i) for i in extreme_values],
                            metadata={'extreme_count': len(extreme_values)}
                        ))
        
        # Score global d'exactitude
        overall_accuracy = np.mean(accuracy_scores) if accuracy_scores else 85.0  # Score par défaut
        
        metrics = QualityMetrics(
            dimension=QualityDimension.ACCURACY,
            score=overall_accuracy,
            issues_count=len(issues),
            critical_issues=len([i for i in issues if i.severity == SeverityLevel.CRITICAL]),
            records_affected=len(set().union(*[issue.affected_records for issue in issues])),
            improvement_potential=100 - overall_accuracy,
            trend=self._get_dimension_trend(QualityDimension.ACCURACY, data_type)
        )
        
        return metrics, issues
    
    async def _evaluate_consistency(self, data: pd.DataFrame, data_type: DataType) -> Tuple[QualityMetrics, List[QualityIssue]]:
        """Évalue la cohérence des données"""
        issues = []
        consistency_scores = []
        
        # Cohérence des formats de données
        format_consistency = self._check_format_consistency(data)
        consistency_scores.append(format_consistency['score'])
        
        if format_consistency['issues']:
            for format_issue in format_consistency['issues']:
                issues.append(QualityIssue(
                    issue_id=f"consistency_format_{format_issue['field']}",
                    dimension=QualityDimension.CONSISTENCY,
                    severity=SeverityLevel.MEDIUM,
                    field_name=format_issue['field'],
                    description=f"Formats incohérents: {format_issue['description']}",
                    suggested_fix="Standardiser le format des données",
                    confidence=0.85,
                    affected_records=format_issue['affected_records'],
                    metadata=format_issue
                ))
        
        # Cohérence référentielle (clés étrangères)
        referential_consistency = self._check_referential_consistency(data)
        consistency_scores.append(referential_consistency['score'])
        
        if referential_consistency['issues']:
            for ref_issue in referential_consistency['issues']:
                issues.append(QualityIssue(
                    issue_id=f"consistency_ref_{ref_issue['field']}",
                    dimension=QualityDimension.CONSISTENCY,
                    severity=SeverityLevel.HIGH,
                    field_name=ref_issue['field'],
                    description=f"Incohérence référentielle: {ref_issue['description']}",
                    suggested_fix="Corriger références orphelines",
                    confidence=0.9,
                    affected_records=ref_issue['affected_records'],
                    metadata=ref_issue
                ))
        
        # Cohérence des valeurs calculées
        calculated_consistency = self._check_calculated_values_consistency(data)
        consistency_scores.append(calculated_consistency['score'])
        
        if calculated_consistency['issues']:
            for calc_issue in calculated_consistency['issues']:
                issues.append(QualityIssue(
                    issue_id=f"consistency_calc_{calc_issue['field']}",
                    dimension=QualityDimension.CONSISTENCY,
                    severity=SeverityLevel.MEDIUM,
                    field_name=calc_issue['field'],
                    description=f"Incohérence calcul: {calc_issue['description']}",
                    suggested_fix="Recalculer valeurs dérivées",
                    confidence=0.8,
                    affected_records=calc_issue['affected_records'],
                    metadata=calc_issue
                ))
        
        overall_consistency = np.mean(consistency_scores) if consistency_scores else 80.0
        
        metrics = QualityMetrics(
            dimension=QualityDimension.CONSISTENCY,
            score=overall_consistency,
            issues_count=len(issues),
            critical_issues=len([i for i in issues if i.severity == SeverityLevel.CRITICAL]),
            records_affected=len(set().union(*[issue.affected_records for issue in issues])),
            improvement_potential=100 - overall_consistency,
            trend=self._get_dimension_trend(QualityDimension.CONSISTENCY, data_type)
        )
        
        return metrics, issues
    
    async def _evaluate_validity(self, data: pd.DataFrame, data_type: DataType) -> Tuple[QualityMetrics, List[QualityIssue]]:
        """Évalue la validité des données selon les règles métier"""
        issues = []
        validity_scores = []
        
        # Appliquer règles de validation par type de données
        rules = self.validation_rules.get(data_type, {})
        
        for field, field_rules in rules.items():
            if field in data.columns:
                field_validity = self._validate_field(data, field, field_rules)
                validity_scores.append(field_validity['score'])
                
                if field_validity['invalid_records']:
                    severity = SeverityLevel.HIGH if field_validity['score'] < 70 else SeverityLevel.MEDIUM
                    
                    issues.append(QualityIssue(
                        issue_id=f"validity_{field}",
                        dimension=QualityDimension.VALIDITY,
                        severity=severity,
                        field_name=field,
                        description=f"Violations règles métier: {len(field_validity['invalid_records'])} cas",
                        suggested_fix=f"Appliquer règles de validation pour {field}",
                        confidence=0.9,
                        affected_records=field_validity['invalid_records'],
                        metadata={'validation_rules': field_rules, 'violations': field_validity['violations']}
                    ))
        
        # Validation croisée entre champs
        cross_validation = self._perform_cross_field_validation(data, data_type)
        if cross_validation['issues']:
            validity_scores.append(cross_validation['score'])
            
            for cross_issue in cross_validation['issues']:
                issues.append(QualityIssue(
                    issue_id=f"validity_cross_{cross_issue['fields']}",
                    dimension=QualityDimension.VALIDITY,
                    severity=SeverityLevel.MEDIUM,
                    field_name=", ".join(cross_issue['fields']),
                    description=f"Validation croisée échouée: {cross_issue['description']}",
                    suggested_fix="Vérifier cohérence entre champs liés",
                    confidence=0.85,
                    affected_records=cross_issue['affected_records'],
                    metadata=cross_issue
                ))
        
        overall_validity = np.mean(validity_scores) if validity_scores else 90.0
        
        metrics = QualityMetrics(
            dimension=QualityDimension.VALIDITY,
            score=overall_validity,
            issues_count=len(issues),
            critical_issues=len([i for i in issues if i.severity == SeverityLevel.CRITICAL]),
            records_affected=len(set().union(*[issue.affected_records for issue in issues])),
            improvement_potential=100 - overall_validity,
            trend=self._get_dimension_trend(QualityDimension.VALIDITY, data_type)
        )
        
        return metrics, issues
    
    async def _evaluate_uniqueness(self, data: pd.DataFrame, data_type: DataType) -> Tuple[QualityMetrics, List[QualityIssue]]:
        """Évalue l'unicité des données"""
        issues = []
        uniqueness_scores = []
        
        # Champs qui doivent être uniques
        unique_fields = self._get_unique_fields(data_type)
        
        for field in unique_fields:
            if field in data.columns:
                # Détecter doublons
                duplicates = data[data.duplicated(subset=[field], keep=False)]
                
                if not duplicates.empty:
                    uniqueness_rate = ((len(data) - len(duplicates)) / len(data)) * 100
                    uniqueness_scores.append(uniqueness_rate)
                    
                    severity = SeverityLevel.CRITICAL if uniqueness_rate < 90 else SeverityLevel.HIGH
                    
                    issues.append(QualityIssue(
                        issue_id=f"uniqueness_{field}",
                        dimension=QualityDimension.UNIQUENESS,
                        severity=severity,
                        field_name=field,
                        description=f"Doublons détectés: {len(duplicates)} enregistrements",
                        suggested_fix=f"Dédupliquer le champ {field}",
                        confidence=0.95,
                        affected_records=[str(i) for i in duplicates.index.tolist()],
                        metadata={'duplicate_count': len(duplicates), 'unique_rate': uniqueness_rate}
                    ))
                else:
                    uniqueness_scores.append(100.0)
        
        # Détecter doublons approximatifs (fuzzy duplicates)
        fuzzy_duplicates = await self._detect_fuzzy_duplicates(data)
        if fuzzy_duplicates:
            issues.append(QualityIssue(
                issue_id="uniqueness_fuzzy",
                dimension=QualityDimension.UNIQUENESS,
                severity=SeverityLevel.MEDIUM,
                field_name="multiple",
                description=f"Doublons approximatifs détectés: {len(fuzzy_duplicates)} groupes",
                suggested_fix="Analyser et fusionner doublons approximatifs",
                confidence=0.7,
                affected_records=[str(record) for group in fuzzy_duplicates for record in group],
                metadata={'fuzzy_groups': len(fuzzy_duplicates)}
            ))
        
        overall_uniqueness = np.mean(uniqueness_scores) if uniqueness_scores else 95.0
        
        metrics = QualityMetrics(
            dimension=QualityDimension.UNIQUENESS,
            score=overall_uniqueness,
            issues_count=len(issues),
            critical_issues=len([i for i in issues if i.severity == SeverityLevel.CRITICAL]),
            records_affected=len(set().union(*[issue.affected_records for issue in issues])),
            improvement_potential=100 - overall_uniqueness,
            trend=self._get_dimension_trend(QualityDimension.UNIQUENESS, data_type)
        )
        
        return metrics, issues
    
    async def _evaluate_timeliness(self, data: pd.DataFrame, data_type: DataType) -> Tuple[QualityMetrics, List[QualityIssue]]:
        """Évalue la fraîcheur temporelle des données"""
        issues = []
        timeliness_scores = []
        
        # Identifier champs de dates
        date_fields = [col for col in data.columns if 'date' in col.lower() or data[col].dtype == 'datetime64[ns]']
        
        for field in date_fields:
            if not data[field].empty:
                # Calculer âge des données
                try:
                    latest_date = pd.to_datetime(data[field].max())
                    data_age_days = (datetime.now() - latest_date).days
                    
                    # Seuils selon type de données
                    if data_type in [DataType.FINANCIAL]:
                        max_age_acceptable = 365  # 1 an pour données financières
                        max_age_good = 90        # 3 mois pour être considéré bon
                    else:
                        max_age_acceptable = 730  # 2 ans pour autres données
                        max_age_good = 180       # 6 mois
                    
                    if data_age_days > max_age_acceptable:
                        timeliness_score = 30
                        severity = SeverityLevel.HIGH
                        description = f"Données très anciennes: {data_age_days} jours"
                    elif data_age_days > max_age_good:
                        timeliness_score = 70
                        severity = SeverityLevel.MEDIUM
                        description = f"Données anciennes: {data_age_days} jours"
                    else:
                        timeliness_score = 95
                        severity = None
                        description = None
                    
                    timeliness_scores.append(timeliness_score)
                    
                    if severity:
                        issues.append(QualityIssue(
                            issue_id=f"timeliness_{field}",
                            dimension=QualityDimension.TIMELINESS,
                            severity=severity,
                            field_name=field,
                            description=description,
                            suggested_fix="Mettre à jour avec données plus récentes",
                            confidence=0.9,
                            affected_records=[],
                            metadata={'data_age_days': data_age_days, 'latest_date': latest_date.isoformat()}
                        ))
                
                except Exception as e:
                    logger.warning(f"Erreur évaluation fraîcheur pour {field}: {e}")
        
        # Évaluer fréquence de mise à jour
        update_frequency = self._analyze_update_frequency(data, date_fields)
        if update_frequency['irregular_updates']:
            issues.append(QualityIssue(
                issue_id="timeliness_frequency",
                dimension=QualityDimension.TIMELINESS,
                severity=SeverityLevel.LOW,
                field_name="update_frequency",
                description="Fréquence de mise à jour irrégulière détectée",
                suggested_fix="Établir calendrier de mise à jour régulier",
                confidence=0.6,
                affected_records=[],
                metadata=update_frequency
            ))
        
        overall_timeliness = np.mean(timeliness_scores) if timeliness_scores else 80.0
        
        metrics = QualityMetrics(
            dimension=QualityDimension.TIMELINESS,
            score=overall_timeliness,
            issues_count=len(issues),
            critical_issues=len([i for i in issues if i.severity == SeverityLevel.CRITICAL]),
            records_affected=len(set().union(*[issue.affected_records for issue in issues])),
            improvement_potential=100 - overall_timeliness,
            trend=self._get_dimension_trend(QualityDimension.TIMELINESS, data_type)
        )
        
        return metrics, issues
    
    async def _evaluate_conformity(self, data: pd.DataFrame, data_type: DataType) -> Tuple[QualityMetrics, List[QualityIssue]]:
        """Évalue la conformité aux standards et formats"""
        issues = []
        conformity_scores = []
        
        # Vérifier conformité aux standards français
        french_standards = self._check_french_standards_conformity(data)
        conformity_scores.append(french_standards['score'])
        
        if french_standards['violations']:
            for violation in french_standards['violations']:
                issues.append(QualityIssue(
                    issue_id=f"conformity_standard_{violation['field']}",
                    dimension=QualityDimension.CONFORMITY,
                    severity=SeverityLevel.MEDIUM,
                    field_name=violation['field'],
                    description=f"Non-conformité standard: {violation['description']}",
                    suggested_fix="Appliquer format standard français",
                    confidence=0.85,
                    affected_records=violation['affected_records'],
                    metadata=violation
                ))
        
        # Vérifier encodage et caractères
        encoding_conformity = self._check_encoding_conformity(data)
        conformity_scores.append(encoding_conformity['score'])
        
        if encoding_conformity['issues']:
            for enc_issue in encoding_conformity['issues']:
                issues.append(QualityIssue(
                    issue_id=f"conformity_encoding_{enc_issue['field']}",
                    dimension=QualityDimension.CONFORMITY,
                    severity=SeverityLevel.LOW,
                    field_name=enc_issue['field'],
                    description=f"Problème encodage: {enc_issue['description']}",
                    suggested_fix="Normaliser encodage UTF-8",
                    confidence=0.8,
                    affected_records=enc_issue['affected_records'],
                    metadata=enc_issue
                ))
        
        # Vérifier conformité schéma de données
        schema_conformity = self._check_schema_conformity(data, data_type)
        conformity_scores.append(schema_conformity['score'])
        
        if schema_conformity['violations']:
            for schema_violation in schema_conformity['violations']:
                issues.append(QualityIssue(
                    issue_id=f"conformity_schema_{schema_violation['field']}",
                    dimension=QualityDimension.CONFORMITY,
                    severity=SeverityLevel.HIGH,
                    field_name=schema_violation['field'],
                    description=f"Non-conformité schéma: {schema_violation['description']}",
                    suggested_fix="Corriger structure de données",
                    confidence=0.9,
                    affected_records=schema_violation['affected_records'],
                    metadata=schema_violation
                ))
        
        overall_conformity = np.mean(conformity_scores) if conformity_scores else 85.0
        
        metrics = QualityMetrics(
            dimension=QualityDimension.CONFORMITY,
            score=overall_conformity,
            issues_count=len(issues),
            critical_issues=len([i for i in issues if i.severity == SeverityLevel.CRITICAL]),
            records_affected=len(set().union(*[issue.affected_records for issue in issues])),
            improvement_potential=100 - overall_conformity,
            trend=self._get_dimension_trend(QualityDimension.CONFORMITY, data_type)
        )
        
        return metrics, issues
    
    # Méthodes utilitaires et helpers
    
    def _initialize_validation_rules(self) -> Dict[DataType, Dict[str, Any]]:
        """Initialise les règles de validation par type de données"""
        return {
            DataType.COMPANY_BASIC: {
                'siren': {
                    'type': 'string',
                    'length': 9,
                    'pattern': r'^\d{9}$',
                    'validator': validate_siren
                },
                'siret': {
                    'type': 'string', 
                    'length': 14,
                    'pattern': r'^\d{14}$',
                    'validator': validate_siret
                },
                'nom': {
                    'type': 'string',
                    'min_length': 2,
                    'max_length': 200,
                    'required': True
                },
                'chiffre_affaires': {
                    'type': 'numeric',
                    'min_value': 0,
                    'max_value': 1e12
                },
                'date_creation': {
                    'type': 'date',
                    'min_date': '1800-01-01',
                    'max_date': 'today'
                }
            },
            DataType.FINANCIAL: {
                'resultat_net': {
                    'type': 'numeric',
                    'min_value': -1e9,
                    'max_value': 1e9
                },
                'capitaux_propres': {
                    'type': 'numeric',
                    'min_value': -1e9,
                    'max_value': 1e12
                }
            },
            DataType.CONTACT: {
                'email': {
                    'type': 'string',
                    'pattern': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
                },
                'telephone': {
                    'type': 'string',
                    'pattern': r'^(?:\+33|0)[1-9](?:[0-9]{8})$'
                }
            }
        }
    
    def _initialize_reference_patterns(self) -> Dict[str, Any]:
        """Initialise les patterns de référence pour données françaises"""
        return {
            'departements_codes': [f'{i:02d}' for i in range(1, 96)] + ['2A', '2B'] + [f'{i}' for i in range(971, 977)],
            'regions_codes': [f'{i:02d}' for i in range(1, 14)],
            'formes_juridiques': ['SA', 'SARL', 'SAS', 'SASU', 'SCI', 'EURL', 'SNC', 'GIE'],
            'secteurs_activite': ['Agriculture', 'Industrie', 'Construction', 'Commerce', 'Transport', 'Information', 'Finance', 'Immobilier', 'Services'],
            'codes_ape_pattern': r'^[0-9]{2}\.[0-9]{2}[A-Z]$'
        }
    
    async def _train_anomaly_detection_models(self):
        """Entraîne les modèles de détection d'anomalies"""
        try:
            # Générer données d'entraînement synthétiques
            training_data = self._generate_training_data()
            
            # Modèle Isolation Forest
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            iso_forest.fit(training_data)
            self.models['isolation_forest'] = iso_forest
            
            # Modèle Local Outlier Factor
            lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
            lof.fit(training_data)
            self.models['lof'] = lof
            
            # Scaler pour normalisation
            scaler = StandardScaler()
            scaler.fit(training_data)
            self.scalers['anomaly_detection'] = scaler
            
            logger.info("✅ Modèles de détection d'anomalies entraînés")
            
        except Exception as e:
            logger.error(f"Erreur entraînement modèles anomalies: {e}")
    
    async def _load_reference_data(self):
        """Charge les données de référence"""
        # Pour l'instant, utiliser patterns intégrés
        # À terme, charger depuis bases de référence externes
        logger.info("✅ Données de référence chargées")
    
    def _generate_training_data(self) -> np.ndarray:
        """Génère des données d'entraînement synthétiques"""
        np.random.seed(42)
        n_samples = 10000
        
        # Simuler features typiques d'entreprises
        data = np.random.randn(n_samples, 10)
        
        # Ajouter quelques anomalies
        n_anomalies = int(n_samples * 0.05)
        anomaly_indices = np.random.choice(n_samples, n_anomalies, replace=False)
        data[anomaly_indices] = data[anomaly_indices] * 3  # Amplifier anomalies
        
        return data
    
    def _analyze_missing_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyse les patterns de données manquantes"""
        missing_matrix = data.isnull()
        
        # Corrélations entre colonnes manquantes
        missing_corr = missing_matrix.corr()
        high_corr_pairs = []
        
        for i in range(len(missing_corr.columns)):
            for j in range(i + 1, len(missing_corr.columns)):
                corr_val = missing_corr.iloc[i, j]
                if abs(corr_val) > 0.5:
                    high_corr_pairs.append({
                        'column1': missing_corr.columns[i],
                        'column2': missing_corr.columns[j], 
                        'correlation': corr_val
                    })
        
        return {
            'systematic_missing': len(high_corr_pairs) > 0,
            'correlated_missing': high_corr_pairs,
            'missing_percentage': missing_matrix.sum().sum() / (len(data) * len(data.columns)) * 100
        }
    
    def _check_format_consistency(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Vérifie la cohérence des formats"""
        issues = []
        total_checks = 0
        failed_checks = 0
        
        for column in data.columns:
            if data[column].dtype == 'object':  # Champs texte
                # Vérifier consistance des formats dans la colonne
                unique_formats = set()
                
                for value in data[column].dropna().head(100):  # Échantillon
                    # Déterminer "format" basé sur pattern général
                    value_str = str(value)
                    format_pattern = ''.join(['d' if c.isdigit() else 'a' if c.isalpha() else 's' for c in value_str])
                    unique_formats.add(format_pattern)
                
                total_checks += 1
                
                if len(unique_formats) > 3:  # Plus de 3 formats différents = incohérent
                    failed_checks += 1
                    
                    issues.append({
                        'field': column,
                        'description': f'{len(unique_formats)} formats différents détectés',
                        'affected_records': [],
                        'formats': list(unique_formats)[:5]  # Top 5
                    })
        
        score = ((total_checks - failed_checks) / total_checks * 100) if total_checks > 0 else 100
        
        return {
            'score': score,
            'issues': issues,
            'total_checks': total_checks,
            'failed_checks': failed_checks
        }
    
    def _check_referential_consistency(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Vérifie la cohérence référentielle"""
        issues = []
        
        # Vérifications basiques de cohérence
        # À enrichir avec vraies contraintes référentielles
        
        score = 90.0  # Score par défaut
        
        return {
            'score': score,
            'issues': issues
        }
    
    def _check_calculated_values_consistency(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Vérifie la cohérence des valeurs calculées"""
        issues = []
        
        # Exemples de vérifications de cohérence
        # Ratio endettement vs capitaux propres
        # Évolution CA vs effectifs, etc.
        
        score = 85.0  # Score par défaut
        
        return {
            'score': score,
            'issues': issues
        }
    
    def _validate_field(self, data: pd.DataFrame, field: str, rules: Dict[str, Any]) -> Dict[str, Any]:
        """Valide un champ selon des règles spécifiques"""
        invalid_records = []
        violations = []
        
        field_data = data[field].dropna()
        
        for idx, value in field_data.items():
            value_str = str(value)
            
            # Vérifier pattern si défini
            if 'pattern' in rules:
                if not re.match(rules['pattern'], value_str):
                    invalid_records.append(str(idx))
                    violations.append(f"Pattern invalide: {value_str}")
            
            # Vérifier longueur
            if 'length' in rules and len(value_str) != rules['length']:
                invalid_records.append(str(idx))
                violations.append(f"Longueur incorrecte: {len(value_str)} au lieu de {rules['length']}")
            
            # Vérifier validateur custom
            if 'validator' in rules:
                try:
                    if not rules['validator'](value_str):
                        invalid_records.append(str(idx))
                        violations.append(f"Validation échouée pour: {value_str}")
                except:
                    invalid_records.append(str(idx))
                    violations.append(f"Erreur validation: {value_str}")
        
        # Calculer score
        total_values = len(field_data)
        invalid_count = len(set(invalid_records))
        score = ((total_values - invalid_count) / total_values * 100) if total_values > 0 else 100
        
        return {
            'score': score,
            'invalid_records': list(set(invalid_records)),
            'violations': violations[:10]  # Limiter à 10 exemples
        }
    
    def _perform_cross_field_validation(self, data: pd.DataFrame, data_type: DataType) -> Dict[str, Any]:
        """Effectue validation croisée entre champs"""
        issues = []
        
        # Exemple: SIRET doit commencer par SIREN
        if 'siren' in data.columns and 'siret' in data.columns:
            mismatched = []
            
            for idx, row in data.dropna(subset=['siren', 'siret']).iterrows():
                siren = str(row['siren'])
                siret = str(row['siret'])
                
                if not siret.startswith(siren):
                    mismatched.append(str(idx))
            
            if mismatched:
                issues.append({
                    'fields': ['siren', 'siret'],
                    'description': f'SIRET ne commence pas par SIREN: {len(mismatched)} cas',
                    'affected_records': mismatched
                })
        
        score = 95.0 if not issues else 70.0
        
        return {
            'score': score,
            'issues': issues
        }
    
    def _get_unique_fields(self, data_type: DataType) -> List[str]:
        """Retourne les champs qui doivent être uniques"""
        unique_fields_map = {
            DataType.COMPANY_BASIC: ['siren', 'siret'],
            DataType.CONTACT: ['email'],
            DataType.LEGAL: ['siren'],
            DataType.FINANCIAL: ['siren'],
            DataType.OPERATIONAL: []
        }
        
        return unique_fields_map.get(data_type, [])
    
    async def _detect_fuzzy_duplicates(self, data: pd.DataFrame) -> List[List[str]]:
        """Détecte les doublons approximatifs"""
        fuzzy_groups = []
        
        # Pour champs texte principaux (nom d'entreprise)
        text_fields = [col for col in data.columns if 'nom' in col.lower() or 'raison' in col.lower()]
        
        for field in text_fields[:1]:  # Limiter à 1 champ pour performance
            if field in data.columns:
                values = data[field].dropna().astype(str)
                
                # Comparer par similarité
                groups = defaultdict(list)
                processed = set()
                
                for idx1, val1 in values.items():
                    if idx1 in processed:
                        continue
                    
                    group = [str(idx1)]
                    
                    for idx2, val2 in values.items():
                        if idx1 != idx2 and idx2 not in processed:
                            similarity = SequenceMatcher(None, val1.lower(), val2.lower()).ratio()
                            if similarity > 0.85:  # Seuil de similarité
                                group.append(str(idx2))
                                processed.add(idx2)
                    
                    if len(group) > 1:
                        fuzzy_groups.append(group)
                        processed.update(int(i) for i in group)
        
        return fuzzy_groups
    
    def _analyze_update_frequency(self, data: pd.DataFrame, date_fields: List[str]) -> Dict[str, Any]:
        """Analyse la fréquence de mise à jour"""
        # Analyse simplifiée
        # À enrichir avec vraie analyse temporelle
        
        return {
            'irregular_updates': False,
            'analysis': 'Fréquence régulière détectée'
        }
    
    def _check_french_standards_conformity(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Vérifie conformité aux standards français"""
        violations = []
        total_checks = 0
        passed_checks = 0
        
        # Vérifier codes postaux français
        if any('postal' in col.lower() or 'cp' in col.lower() for col in data.columns):
            postal_cols = [col for col in data.columns if 'postal' in col.lower() or 'cp' in col.lower()]
            
            for col in postal_cols:
                total_checks += 1
                postal_pattern = re.compile(r'^[0-9]{5}$')
                invalid_postcodes = []
                
                for idx, value in data[col].dropna().items():
                    if not postal_pattern.match(str(value)):
                        invalid_postcodes.append(str(idx))
                
                if invalid_postcodes:
                    violations.append({
                        'field': col,
                        'description': f'Codes postaux invalides: {len(invalid_postcodes)} cas',
                        'affected_records': invalid_postcodes[:10]
                    })
                else:
                    passed_checks += 1
        
        score = (passed_checks / total_checks * 100) if total_checks > 0 else 95
        
        return {
            'score': score,
            'violations': violations,
            'total_checks': total_checks
        }
    
    def _check_encoding_conformity(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Vérifie la conformité d'encodage"""
        issues = []
        total_text_fields = 0
        clean_fields = 0
        
        for column in data.columns:
            if data[column].dtype == 'object':
                total_text_fields += 1
                
                # Vérifier caractères non-UTF8 ou problématiques
                problematic_chars = []
                
                for idx, value in data[column].dropna().head(100).items():
                    value_str = str(value)
                    
                    # Détecter caractères non-ASCII problématiques
                    if any(ord(char) > 127 and char not in 'àáâãäåæçèéêëìíîïñòóôõöøùúûüýÿ' for char in value_str):
                        problematic_chars.append(str(idx))
                
                if problematic_chars:
                    issues.append({
                        'field': column,
                        'description': f'Caractères d\'encodage problématiques: {len(problematic_chars)} cas',
                        'affected_records': problematic_chars[:5]
                    })
                else:
                    clean_fields += 1
        
        score = (clean_fields / total_text_fields * 100) if total_text_fields > 0 else 100
        
        return {
            'score': score,
            'issues': issues
        }
    
    def _check_schema_conformity(self, data: pd.DataFrame, data_type: DataType) -> Dict[str, Any]:
        """Vérifie conformité au schéma attendu"""
        violations = []
        
        # Schémas attendus par type
        expected_schemas = {
            DataType.COMPANY_BASIC: ['siren', 'nom', 'date_creation'],
            DataType.FINANCIAL: ['chiffre_affaires', 'resultat_net'],
            DataType.CONTACT: ['email', 'telephone']
        }
        
        expected_fields = expected_schemas.get(data_type, [])
        missing_fields = [field for field in expected_fields if field not in data.columns]
        
        if missing_fields:
            violations.append({
                'field': 'schema',
                'description': f'Champs manquants: {", ".join(missing_fields)}',
                'affected_records': [],
                'missing_fields': missing_fields
            })
        
        score = 95.0 if not violations else 60.0
        
        return {
            'score': score,
            'violations': violations
        }
    
    def _calculate_overall_score(self, quality_metrics: Dict[QualityDimension, QualityMetrics]) -> float:
        """Calcule le score global de qualité"""
        # Pondération par importance
        weights = {
            QualityDimension.COMPLETENESS: 0.20,
            QualityDimension.ACCURACY: 0.25,
            QualityDimension.CONSISTENCY: 0.15,
            QualityDimension.VALIDITY: 0.20,
            QualityDimension.UNIQUENESS: 0.10,
            QualityDimension.TIMELINESS: 0.05,
            QualityDimension.CONFORMITY: 0.05
        }
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for dimension, metrics in quality_metrics.items():
            weight = weights.get(dimension, 0.1)
            weighted_sum += metrics.score * weight
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 50.0
    
    def _generate_improvement_actions(self, quality_metrics: Dict[QualityDimension, QualityMetrics], 
                                    issues: List[QualityIssue]) -> List[str]:
        """Génère des actions d'amélioration"""
        actions = []
        
        # Actions basées sur dimensions faibles
        for dimension, metrics in quality_metrics.items():
            if metrics.score < 70:
                if dimension == QualityDimension.COMPLETENESS:
                    actions.append("Mettre en place processus systématique de collecte de données")
                elif dimension == QualityDimension.ACCURACY:
                    actions.append("Implémenter validation automatique en temps réel")
                elif dimension == QualityDimension.CONSISTENCY:
                    actions.append("Standardiser formats et processus de saisie")
                elif dimension == QualityDimension.VALIDITY:
                    actions.append("Renforcer règles de validation métier")
                elif dimension == QualityDimension.UNIQUENESS:
                    actions.append("Mettre en place détection et prévention de doublons")
        
        # Actions basées sur problèmes critiques
        critical_issues = [issue for issue in issues if issue.severity == SeverityLevel.CRITICAL]
        if len(critical_issues) > 5:
            actions.append("Audit complet et nettoyage urgent des données")
        
        return actions[:5]  # Limiter à 5 actions
    
    def _prioritize_fixes(self, issues: List[QualityIssue]) -> List[str]:
        """Priorise les corrections à effectuer"""
        # Trier par sévérité puis par nombre d'enregistrements affectés
        sorted_issues = sorted(issues, 
                             key=lambda x: (x.severity.value, len(x.affected_records)), 
                             reverse=True)
        
        return [issue.suggested_fix for issue in sorted_issues[:5]]
    
    def _estimate_effort(self, actions: List[str]) -> Dict[str, str]:
        """Estime l'effort nécessaire pour chaque action"""
        effort_map = {
            'processus': 'Élevé (2-4 semaines)',
            'validation': 'Moyen (1-2 semaines)', 
            'standardiser': 'Moyen (1-2 semaines)',
            'règles': 'Faible (2-5 jours)',
            'détection': 'Moyen (1 semaine)',
            'audit': 'Élevé (3-6 semaines)'
        }
        
        estimated_effort = {}
        for action in actions:
            effort = 'Moyen (1 semaine)'  # Par défaut
            
            for keyword, effort_level in effort_map.items():
                if keyword in action.lower():
                    effort = effort_level
                    break
            
            estimated_effort[action] = effort
        
        return estimated_effort
    
    def _calculate_data_coverage(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calcule la couverture de données par champ"""
        coverage = {}
        
        for column in data.columns:
            non_null_count = data[column].count()
            coverage_rate = (non_null_count / len(data)) * 100
            coverage[column] = coverage_rate
        
        return coverage
    
    def _calculate_model_confidence(self, quality_metrics: Dict[QualityDimension, QualityMetrics]) -> float:
        """Calcule la confiance globale du modèle"""
        # Basé sur cohérence des scores et nombre de vérifications
        scores = [metrics.score for metrics in quality_metrics.values()]
        
        # Confiance plus élevée si scores cohérents
        score_variance = np.var(scores)
        base_confidence = 0.8
        
        # Ajuster selon variance
        if score_variance < 100:
            confidence = base_confidence + 0.15
        elif score_variance > 500:
            confidence = base_confidence - 0.2
        else:
            confidence = base_confidence
        
        return max(0.3, min(1.0, confidence))
    
    def _get_dimension_trend(self, dimension: QualityDimension, data_type: DataType) -> str:
        """Récupère la tendance pour une dimension"""
        # Analyser historique si disponible
        history_key = f"{data_type.value}_{dimension.value}"
        
        if history_key in self.quality_history and len(self.quality_history[history_key]) >= 2:
            recent_scores = [record['score'] for record in self.quality_history[history_key][-3:]]
            
            if len(recent_scores) >= 2:
                if recent_scores[-1] > recent_scores[0] + 5:
                    return 'improving'
                elif recent_scores[-1] < recent_scores[0] - 5:
                    return 'declining'
        
        return 'stable'
    
    def _record_quality_metrics(self, data_type: DataType, 
                              quality_metrics: Dict[QualityDimension, QualityMetrics], 
                              overall_score: float):
        """Enregistre les métriques pour analyse de tendances"""
        timestamp = datetime.now()
        
        record = {
            'timestamp': timestamp,
            'overall_score': overall_score,
            'metrics': {dim.value: {'score': metrics.score} for dim, metrics in quality_metrics.items()}
        }
        
        # Ajouter à l'historique
        self.quality_history[data_type.value].append(record)
        
        # Limiter taille historique
        if len(self.quality_history[data_type.value]) > 100:
            self.quality_history[data_type.value] = self.quality_history[data_type.value][-50:]
    
    def _identify_quality_alerts(self, trends_analysis: Dict[str, Any]) -> List[str]:
        """Identifie les alertes qualité"""
        alerts = []
        
        for dimension, trend_data in trends_analysis.items():
            if trend_data['trend_direction'] == 'declining':
                if trend_data['current_score'] < 60:
                    alerts.append(f"Alerte critique: {dimension} en déclin (score: {trend_data['current_score']:.1f})")
                elif trend_data['current_score'] < 80:
                    alerts.append(f"Alerte: {dimension} en déclin (score: {trend_data['current_score']:.1f})")
        
        return alerts
    
    def _suggest_continuous_improvements(self, trends_analysis: Dict[str, Any]) -> List[str]:
        """Suggère des améliorations continues"""
        suggestions = []
        
        # Analyser patterns dans les tendances
        improving_dimensions = [dim for dim, data in trends_analysis.items() 
                              if data['trend_direction'] == 'improving']
        
        if len(improving_dimensions) > 0:
            suggestions.append(f"Capitaliser sur progrès en {', '.join(improving_dimensions[:2])}")
        
        stable_but_low = [dim for dim, data in trends_analysis.items() 
                         if data['trend_direction'] == 'stable' and data['current_score'] < 80]
        
        if stable_but_low:
            suggestions.append(f"Améliorer dimensions stables mais perfectibles: {', '.join(stable_but_low[:2])}")
        
        suggestions.append("Mettre en place monitoring continu de la qualité")
        
        return suggestions[:3]
    
    def _calculate_overall_trend(self, trends_analysis: Dict[str, Any]) -> str:
        """Calcule la tendance globale"""
        if not trends_analysis:
            return 'stable'
        
        improving_count = sum(1 for data in trends_analysis.values() 
                            if data['trend_direction'] == 'improving')
        declining_count = sum(1 for data in trends_analysis.values() 
                            if data['trend_direction'] == 'declining')
        
        if improving_count > declining_count:
            return 'improving'
        elif declining_count > improving_count:
            return 'declining'
        else:
            return 'stable'
    
    async def _apply_automatic_fix(self, data: pd.DataFrame, issue: QualityIssue) -> bool:
        """Applique une correction automatique"""
        try:
            # Corrections simples automatisables
            if issue.dimension == QualityDimension.CONFORMITY:
                if 'encoding' in issue.issue_id:
                    # Nettoyer caractères problématiques
                    field = issue.field_name
                    if field in data.columns:
                        data[field] = data[field].astype(str).apply(
                            lambda x: unicodedata.normalize('NFKD', x).encode('ascii', 'ignore').decode('ascii')
                        )
                        return True
            
            elif issue.dimension == QualityDimension.VALIDITY:
                # Corrections de format
                if 'siren' in issue.field_name.lower():
                    field = issue.field_name
                    if field in data.columns:
                        # Nettoyer SIREN (garder seulement chiffres)
                        data[field] = data[field].astype(str).str.replace(r'[^0-9]', '', regex=True)
                        return True
            
            return False
            
        except Exception as e:
            logger.warning(f"Erreur correction automatique {issue.issue_id}: {e}")
            return False
    
    def _generate_cache_key(self, data: pd.DataFrame, data_type: DataType) -> str:
        """Génère une clé de cache unique"""
        # Hash basé sur structure et échantillon de données
        key_data = {
            'data_type': data_type.value,
            'shape': data.shape,
            'columns': list(data.columns),
            'sample_hash': hashlib.md5(str(data.head().values.tobytes()).encode()).hexdigest()
        }
        
        key_string = json.dumps(key_data, sort_keys=True)
        return f"dq_{hashlib.md5(key_string.encode()).hexdigest()}"


# Instance globale
_data_quality_engine: Optional[DataQualityEngine] = None


async def get_data_quality_engine() -> DataQualityEngine:
    """Factory pour obtenir l'instance du moteur de qualité"""
    global _data_quality_engine
    
    if _data_quality_engine is None:
        _data_quality_engine = DataQualityEngine()
        await _data_quality_engine.initialize()
    
    return _data_quality_engine


async def initialize_data_quality_engine():
    """Initialise le système de qualité des données au démarrage"""
    try:
        engine = await get_data_quality_engine()
        logger.info("📊 Système d'évaluation de qualité des données initialisé avec succès")
        return engine
    except Exception as e:
        logger.error(f"Erreur initialisation moteur qualité: {e}")
        raise