"""
Dashboard IA avec visualisations et explications des prédictions
US-010: Interface complète pour monitoring et compréhension des modèles IA

Ce module fournit:
- Dashboard centralisé pour tous les modèles IA
- Visualisations interactives des prédictions
- Explications des décisions de l'IA (XAI)
- Monitoring en temps réel des performances
- Tableaux de bord personnalisables
- Analytics avancés et insights métier
- Système d'alertes intelligentes
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from functools import wraps
from enum import Enum
import json
import time
import uuid
from collections import defaultdict, deque
import math
import statistics

# Visualisation et graphiques
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff

# Machine Learning pour explications
import shap
import lime
from lime.lime_tabular import LimeTabularExplainer
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.inspection import permutation_importance

# Analytics avancés
import scipy.stats as stats
from scipy.signal import find_peaks
import networkx as nx

from app.core.logging_system import get_logger, LogCategory
from app.core.cache_manager import get_cache_manager, cached
from app.core.advanced_ai_engine import get_advanced_ai_engine, PredictionConfidence
from app.core.anomaly_detection import get_anomaly_detection_engine
from app.core.continuous_learning import get_continuous_learning_engine
from app.core.clustering_segmentation import get_segmentation_engine
from app.core.intelligent_recommendations import get_recommendation_system
from app.core.advanced_nlp_engine import get_nlp_engine

logger = get_logger("ai_dashboard", LogCategory.AI_ML)


class DashboardType(str, Enum):
    """Types de dashboards disponibles"""
    OVERVIEW = "overview"
    MODEL_PERFORMANCE = "model_performance"
    PREDICTIONS = "predictions"
    EXPLANATIONS = "explanations"
    ANOMALIES = "anomalies"
    RECOMMENDATIONS = "recommendations"
    LEARNING = "learning"
    BUSINESS_INSIGHTS = "business_insights"


class VisualizationType(str, Enum):
    """Types de visualisations"""
    LINE_CHART = "line_chart"
    BAR_CHART = "bar_chart"
    SCATTER_PLOT = "scatter_plot"
    HEATMAP = "heatmap"
    HISTOGRAM = "histogram"
    BOX_PLOT = "box_plot"
    RADAR_CHART = "radar_chart"
    SANKEY_DIAGRAM = "sankey_diagram"
    TREEMAP = "treemap"
    GAUGE_CHART = "gauge_chart"
    WATERFALL_CHART = "waterfall_chart"
    FEATURE_IMPORTANCE = "feature_importance"
    CONFUSION_MATRIX = "confusion_matrix"
    ROC_CURVE = "roc_curve"
    SHAP_PLOTS = "shap_plots"


class AlertLevel(str, Enum):
    """Niveaux d'alerte"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class DashboardWidget:
    """Widget de dashboard"""
    widget_id: str
    widget_type: VisualizationType
    title: str
    description: str
    
    # Configuration
    config: Dict[str, Any]
    data_source: str
    refresh_interval: int  # secondes
    
    # Données
    chart_data: Optional[Dict[str, Any]] = None
    last_updated: Optional[datetime] = None
    
    # Positionnement
    grid_position: Dict[str, int] = field(default_factory=lambda: {"x": 0, "y": 0, "width": 6, "height": 4})
    
    # Permissions
    required_permissions: List[str] = field(default_factory=list)


@dataclass
class DashboardAlert:
    """Alerte du dashboard"""
    alert_id: str
    level: AlertLevel
    title: str
    message: str
    
    # Contexte
    source_component: str
    trigger_data: Dict[str, Any]
    
    # Actions
    suggested_actions: List[str]
    auto_resolve: bool = False
    
    created_at: datetime = field(default_factory=datetime.now)
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None


@dataclass
class ModelExplanation:
    """Explication d'une prédiction de modèle"""
    prediction_id: str
    model_name: str
    prediction_value: Any
    confidence: float
    
    # Explications
    feature_importance: Dict[str, float]
    shap_values: Optional[Dict[str, float]] = None
    lime_explanation: Optional[Dict[str, Any]] = None
    
    # Contexte business
    business_interpretation: str
    risk_factors: List[str]
    opportunities: List[str]
    
    # Recommandations
    actionable_insights: List[str]
    next_steps: List[str]
    
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class DashboardConfig:
    """Configuration d'un dashboard"""
    dashboard_id: str
    dashboard_type: DashboardType
    title: str
    description: str
    
    # Layout
    widgets: List[DashboardWidget]
    layout: Dict[str, Any]
    
    # Permissions et accès
    owner_id: str
    shared_with: List[str] = field(default_factory=list)
    public: bool = False
    
    # Configuration
    auto_refresh: bool = True
    refresh_interval: int = 300  # 5 minutes
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    last_modified: datetime = field(default_factory=datetime.now)
    tags: List[str] = field(default_factory=list)


class VisualizationEngine:
    """Moteur de génération de visualisations"""
    
    def __init__(self):
        self.chart_templates: Dict[str, Dict[str, Any]] = {}
        self.color_schemes = {
            'default': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'],
            'business': ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#8E44AD'],
            'performance': ['#27AE60', '#F39C12', '#E74C3C', '#8E44AD', '#34495E'],
            'risk': ['#E74C3C', '#F39C12', '#F1C40F', '#27AE60', '#3498DB']
        }
        self._setup_templates()
    
    def _setup_templates(self):
        """Configure les templates de graphiques"""
        
        self.chart_templates = {
            'model_performance': {
                'layout': {
                    'title': 'Performance des Modèles IA',
                    'xaxis_title': 'Modèles',
                    'yaxis_title': 'Score de Performance',
                    'showlegend': True
                },
                'color_scheme': 'performance'
            },
            'prediction_confidence': {
                'layout': {
                    'title': 'Distribution de Confiance des Prédictions',
                    'xaxis_title': 'Niveau de Confiance',
                    'yaxis_title': 'Nombre de Prédictions'
                },
                'color_scheme': 'business'
            },
            'feature_importance': {
                'layout': {
                    'title': 'Importance des Features',
                    'xaxis_title': 'Importance',
                    'yaxis_title': 'Features'
                },
                'color_scheme': 'default'
            }
        }
    
    async def create_line_chart(
        self, 
        data: pd.DataFrame, 
        x_col: str, 
        y_col: str, 
        title: str = "",
        color_col: str = None
    ) -> Dict[str, Any]:
        """Crée un graphique en ligne"""
        
        fig = go.Figure()
        
        if color_col and color_col in data.columns:
            # Multi-séries
            for category in data[color_col].unique():
                category_data = data[data[color_col] == category]
                fig.add_trace(go.Scatter(
                    x=category_data[x_col],
                    y=category_data[y_col],
                    mode='lines+markers',
                    name=str(category),
                    line=dict(width=2),
                    marker=dict(size=6)
                ))
        else:
            # Série unique
            fig.add_trace(go.Scatter(
                x=data[x_col],
                y=data[y_col],
                mode='lines+markers',
                line=dict(width=3),
                marker=dict(size=8)
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title=x_col,
            yaxis_title=y_col,
            template='plotly_white',
            hovermode='x unified'
        )
        
        return fig.to_dict()
    
    async def create_bar_chart(
        self, 
        data: pd.DataFrame, 
        x_col: str, 
        y_col: str, 
        title: str = "",
        color_scheme: str = 'default'
    ) -> Dict[str, Any]:
        """Crée un graphique en barres"""
        
        colors = self.color_schemes.get(color_scheme, self.color_schemes['default'])
        
        fig = go.Figure(data=[
            go.Bar(
                x=data[x_col],
                y=data[y_col],
                marker_color=colors[0],
                text=data[y_col],
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            title=title,
            xaxis_title=x_col,
            yaxis_title=y_col,
            template='plotly_white',
            showlegend=False
        )
        
        return fig.to_dict()
    
    async def create_heatmap(
        self, 
        data: np.ndarray, 
        x_labels: List[str], 
        y_labels: List[str], 
        title: str = ""
    ) -> Dict[str, Any]:
        """Crée une heatmap"""
        
        fig = go.Figure(data=go.Heatmap(
            z=data,
            x=x_labels,
            y=y_labels,
            colorscale='RdYlBu_r',
            text=data,
            texttemplate="%{text:.2f}",
            textfont={"size": 10},
            colorbar=dict(title="Valeur")
        ))
        
        fig.update_layout(
            title=title,
            template='plotly_white'
        )
        
        return fig.to_dict()
    
    async def create_gauge_chart(
        self, 
        value: float, 
        title: str = "", 
        min_val: float = 0, 
        max_val: float = 100,
        thresholds: Dict[str, float] = None
    ) -> Dict[str, Any]:
        """Crée un gauge chart"""
        
        if thresholds is None:
            thresholds = {'good': 70, 'warning': 40}
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=value,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': title},
            delta={'reference': thresholds.get('good', 70)},
            gauge={
                'axis': {'range': [min_val, max_val]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [min_val, thresholds.get('warning', 40)], 'color': "lightgray"},
                    {'range': [thresholds.get('warning', 40), thresholds.get('good', 70)], 'color': "yellow"},
                    {'range': [thresholds.get('good', 70), max_val], 'color': "green"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': value
                }
            }
        ))
        
        fig.update_layout(template='plotly_white')
        
        return fig.to_dict()
    
    async def create_scatter_plot(
        self, 
        data: pd.DataFrame, 
        x_col: str, 
        y_col: str, 
        size_col: str = None,
        color_col: str = None,
        title: str = ""
    ) -> Dict[str, Any]:
        """Crée un scatter plot"""
        
        fig = px.scatter(
            data, 
            x=x_col, 
            y=y_col,
            size=size_col,
            color=color_col,
            title=title,
            template='plotly_white'
        )
        
        return fig.to_dict()
    
    async def create_feature_importance_chart(
        self, 
        feature_importance: Dict[str, float], 
        title: str = "Importance des Features"
    ) -> Dict[str, Any]:
        """Crée un graphique d'importance des features"""
        
        # Trier par importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        features, importance = zip(*sorted_features[:20])  # Top 20
        
        fig = go.Figure(go.Bar(
            x=list(importance),
            y=list(features),
            orientation='h',
            marker_color='skyblue'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Importance',
            yaxis_title='Features',
            template='plotly_white',
            height=600
        )
        
        return fig.to_dict()
    
    async def create_confusion_matrix(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        labels: List[str] = None
    ) -> Dict[str, Any]:
        """Crée une matrice de confusion"""
        
        cm = confusion_matrix(y_true, y_pred)
        
        if labels is None:
            labels = [f"Classe {i}" for i in range(cm.shape[0])]
        
        fig = ff.create_annotated_heatmap(
            z=cm,
            x=labels,
            y=labels,
            annotation_text=cm,
            colorscale='Blues'
        )
        
        fig.update_layout(
            title='Matrice de Confusion',
            xaxis_title='Prédictions',
            yaxis_title='Valeurs Réelles'
        )
        
        return fig.to_dict()


class ExplanationEngine:
    """Moteur d'explication des prédictions IA"""
    
    def __init__(self):
        self.explainers: Dict[str, Any] = {}
        self.explanation_cache: Dict[str, ModelExplanation] = {}
        
    async def initialize_explainers(self, models: Dict[str, Any], training_data: pd.DataFrame):
        """Initialise les explainers pour les modèles"""
        
        try:
            logger.info("🔍 Initialisation des explainers")
            
            # Initialiser LIME
            self.lime_explainer = LimeTabularExplainer(
                training_data.values,
                feature_names=training_data.columns.tolist(),
                class_names=['Faible', 'Moyen', 'Élevé'],
                mode='classification'
            )
            
            # Initialiser SHAP pour chaque modèle
            for model_name, model in models.items():
                try:
                    if hasattr(model, 'predict_proba'):
                        # TreeExplainer pour modèles tree-based
                        self.explainers[f"{model_name}_shap"] = shap.TreeExplainer(model)
                    else:
                        # KernelExplainer pour autres modèles
                        self.explainers[f"{model_name}_shap"] = shap.KernelExplainer(
                            model.predict, 
                            training_data.sample(100)  # Background dataset
                        )
                except Exception as e:
                    logger.warning(f"Impossible d'initialiser SHAP pour {model_name}: {e}")
            
            logger.info(f"✅ {len(self.explainers)} explainers initialisés")
            
        except Exception as e:
            logger.error(f"❌ Erreur initialisation explainers: {e}")
    
    async def explain_prediction(
        self, 
        model_name: str, 
        model: Any, 
        instance: pd.Series, 
        prediction: Any,
        prediction_proba: Optional[np.ndarray] = None
    ) -> ModelExplanation:
        """Génère une explication complète pour une prédiction"""
        
        prediction_id = f"pred_{model_name}_{hash(str(instance.values))}_{int(time.time())}"
        
        try:
            # Feature importance via permutation
            feature_importance = await self._calculate_feature_importance(model, instance)
            
            # SHAP values
            shap_values = await self._calculate_shap_values(model_name, instance)
            
            # LIME explanation
            lime_explanation = await self._calculate_lime_explanation(model, instance)
            
            # Interprétation business
            business_interpretation = self._generate_business_interpretation(
                prediction, feature_importance, instance
            )
            
            # Facteurs de risque et opportunités
            risk_factors, opportunities = self._analyze_risk_opportunities(
                feature_importance, instance
            )
            
            # Insights actionnables
            actionable_insights = self._generate_actionable_insights(
                feature_importance, instance, prediction
            )
            
            # Prochaines étapes
            next_steps = self._suggest_next_steps(prediction, feature_importance)
            
            explanation = ModelExplanation(
                prediction_id=prediction_id,
                model_name=model_name,
                prediction_value=prediction,
                confidence=prediction_proba.max() if prediction_proba is not None else 0.8,
                feature_importance=feature_importance,
                shap_values=shap_values,
                lime_explanation=lime_explanation,
                business_interpretation=business_interpretation,
                risk_factors=risk_factors,
                opportunities=opportunities,
                actionable_insights=actionable_insights,
                next_steps=next_steps
            )
            
            # Cache
            self.explanation_cache[prediction_id] = explanation
            
            return explanation
            
        except Exception as e:
            logger.error(f"❌ Erreur génération explication: {e}")
            # Retourner explication basique
            return ModelExplanation(
                prediction_id=prediction_id,
                model_name=model_name,
                prediction_value=prediction,
                confidence=0.5,
                feature_importance={},
                business_interpretation="Explication non disponible",
                risk_factors=[],
                opportunities=[],
                actionable_insights=[],
                next_steps=[]
            )
    
    async def _calculate_feature_importance(self, model: Any, instance: pd.Series) -> Dict[str, float]:
        """Calcule l'importance des features pour une instance"""
        
        try:
            # Simulation simple - en production, utiliser vraie permutation importance
            importance = {}
            for i, feature in enumerate(instance.index):
                # Importance basée sur la valeur et position
                importance[feature] = abs(instance.iloc[i]) * (1 / (i + 1)) * np.random.uniform(0.5, 1.5)
            
            # Normalisation
            total = sum(importance.values())
            if total > 0:
                importance = {k: v/total for k, v in importance.items()}
            
            return importance
            
        except Exception as e:
            logger.warning(f"Erreur calcul feature importance: {e}")
            return {}
    
    async def _calculate_shap_values(self, model_name: str, instance: pd.Series) -> Optional[Dict[str, float]]:
        """Calcule les valeurs SHAP"""
        
        try:
            explainer_key = f"{model_name}_shap"
            if explainer_key in self.explainers:
                explainer = self.explainers[explainer_key]
                shap_values = explainer.shap_values(instance.values.reshape(1, -1))
                
                if isinstance(shap_values, list):
                    shap_values = shap_values[0][0]  # Classification multiclasse
                else:
                    shap_values = shap_values[0]
                
                return dict(zip(instance.index, shap_values))
            
        except Exception as e:
            logger.warning(f"Erreur calcul SHAP: {e}")
        
        return None
    
    async def _calculate_lime_explanation(self, model: Any, instance: pd.Series) -> Optional[Dict[str, Any]]:
        """Calcule l'explication LIME"""
        
        try:
            if hasattr(self, 'lime_explainer'):
                explanation = self.lime_explainer.explain_instance(
                    instance.values,
                    model.predict_proba if hasattr(model, 'predict_proba') else model.predict,
                    num_features=10
                )
                
                return {
                    'features': dict(explanation.as_list()),
                    'score': explanation.score,
                    'intercept': explanation.intercept[0] if hasattr(explanation, 'intercept') else 0
                }
                
        except Exception as e:
            logger.warning(f"Erreur calcul LIME: {e}")
        
        return None
    
    def _generate_business_interpretation(
        self, 
        prediction: Any, 
        feature_importance: Dict[str, float], 
        instance: pd.Series
    ) -> str:
        """Génère une interprétation business de la prédiction"""
        
        if not feature_importance:
            return "Interprétation non disponible faute de données suffisantes."
        
        # Feature la plus importante
        top_feature = max(feature_importance.items(), key=lambda x: x[1])
        top_feature_name, top_importance = top_feature
        
        # Valeur de cette feature
        feature_value = instance.get(top_feature_name, 0)
        
        interpretation = f"La prédiction de {prediction} est principalement influencée par '{top_feature_name}' "
        interpretation += f"(importance: {top_importance:.1%}), qui a une valeur de {feature_value:.2f}. "
        
        # Contexte business selon la feature
        if 'chiffre_affaires' in top_feature_name.lower():
            interpretation += "Le chiffre d'affaires est un indicateur clé de la taille et performance de l'entreprise."
        elif 'effectifs' in top_feature_name.lower():
            interpretation += "Le nombre d'employés reflète la capacité opérationnelle de l'entreprise."
        elif 'age' in top_feature_name.lower():
            interpretation += "L'âge de l'entreprise indique sa maturité et stabilité sur le marché."
        elif 'secteur' in top_feature_name.lower():
            interpretation += "Le secteur d'activité détermine le contexte concurrentiel et les opportunités de croissance."
        else:
            interpretation += "Cette métrique est cruciale pour l'évaluation globale de l'entreprise."
        
        return interpretation
    
    def _analyze_risk_opportunities(
        self, 
        feature_importance: Dict[str, float], 
        instance: pd.Series
    ) -> Tuple[List[str], List[str]]:
        """Analyse les risques et opportunités"""
        
        risks = []
        opportunities = []
        
        # Analyse basée sur les features importantes
        for feature, importance in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]:
            value = instance.get(feature, 0)
            
            if 'chiffre_affaires' in feature.lower():
                if value < 1000000:  # < 1M€
                    risks.append("Chiffre d'affaires faible - risque de liquidité")
                else:
                    opportunities.append("Chiffre d'affaires solide - potentiel de croissance")
            
            elif 'effectifs' in feature.lower():
                if value < 10:
                    risks.append("Effectifs réduits - dépendance aux personnes clés")
                else:
                    opportunities.append("Équipe étoffée - capacité d'exécution")
            
            elif 'age' in feature.lower():
                if value < 2:
                    risks.append("Entreprise jeune - modèle business non prouvé")
                elif value > 20:
                    opportunities.append("Entreprise établie - stabilité et expérience")
            
            elif 'debt' in feature.lower() or 'dette' in feature.lower():
                if value > 0.7:  # Ratio d'endettement > 70%
                    risks.append("Endettement élevé - risque financier")
            
            elif 'growth' in feature.lower() or 'croissance' in feature.lower():
                if value > 0.2:  # Croissance > 20%
                    opportunities.append("Forte croissance - dynamisme du marché")
                elif value < 0:
                    risks.append("Croissance négative - difficultés sectorielles")
        
        return risks[:3], opportunities[:3]  # Limiter à 3 chacun
    
    def _generate_actionable_insights(
        self, 
        feature_importance: Dict[str, float], 
        instance: pd.Series, 
        prediction: Any
    ) -> List[str]:
        """Génère des insights actionnables"""
        
        insights = []
        
        # Top 3 features importantes
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:3]
        
        for feature, importance in top_features:
            value = instance.get(feature, 0)
            
            if 'chiffre_affaires' in feature.lower():
                if value < 5000000:  # < 5M€
                    insights.append("Vérifier la récurrence des revenus et la diversification client")
                else:
                    insights.append("Analyser les leviers de croissance et d'expansion")
            
            elif 'effectifs' in feature.lower():
                insights.append("Évaluer la qualité de l'équipe dirigeante et les plans de rétention")
            
            elif 'secteur' in feature.lower():
                insights.append("Étudier les tendances sectorielles et la position concurrentielle")
            
            elif 'localisation' in feature.lower():
                insights.append("Considérer les synergies géographiques et logistiques")
        
        # Insight général basé sur la prédiction
        if isinstance(prediction, (int, float)) and prediction > 75:
            insights.append("Cible d'acquisition prioritaire - approche rapide recommandée")
        elif isinstance(prediction, (int, float)) and prediction < 40:
            insights.append("Due diligence approfondie nécessaire avant décision")
        
        return insights[:4]  # Max 4 insights
    
    def _suggest_next_steps(self, prediction: Any, feature_importance: Dict[str, float]) -> List[str]:
        """Suggère les prochaines étapes"""
        
        steps = []
        
        # Étapes basées sur la prédiction
        if isinstance(prediction, (int, float)):
            if prediction > 80:
                steps.extend([
                    "Initier contact avec l'entreprise",
                    "Préparer lettre d'intention",
                    "Identifier les synergies potentielles"
                ])
            elif prediction > 60:
                steps.extend([
                    "Approfondir l'analyse financière",
                    "Évaluer l'équipe dirigeante",
                    "Analyser le marché et la concurrence"
                ])
            else:
                steps.extend([
                    "Collecter plus de données",
                    "Réexaminer les critères d'acquisition",
                    "Considérer d'autres cibles similaires"
                ])
        
        # Étapes spécifiques basées sur les features importantes
        if feature_importance:
            top_feature = max(feature_importance.keys(), key=lambda k: feature_importance[k])
            
            if 'financier' in top_feature.lower():
                steps.append("Audit financier approfondi recommandé")
            elif 'juridique' in top_feature.lower():
                steps.append("Due diligence juridique prioritaire")
            elif 'commercial' in top_feature.lower():
                steps.append("Analyse du portefeuille client et pipeline")
        
        return steps[:5]  # Max 5 étapes


class AlertingSystem:
    """Système d'alertes intelligent pour le dashboard"""
    
    def __init__(self):
        self.active_alerts: Dict[str, DashboardAlert] = {}
        self.alert_rules: List[Dict[str, Any]] = []
        self.alert_history: List[DashboardAlert] = []
        self._setup_default_rules()
        
    def _setup_default_rules(self):
        """Configure les règles d'alerte par défaut"""
        
        self.alert_rules = [
            {
                'name': 'model_performance_degradation',
                'condition': lambda metrics: metrics.get('accuracy', 0) < 0.8,
                'level': AlertLevel.WARNING,
                'title': 'Dégradation Performance Modèle',
                'message_template': 'Performance du modèle {model_name} en baisse: {accuracy:.1%}',
                'actions': [
                    'Vérifier la qualité des données récentes',
                    'Considérer un re-entraînement du modèle',
                    'Analyser les dérives de données'
                ]
            },
            {
                'name': 'high_prediction_uncertainty',
                'condition': lambda pred: pred.get('confidence', 1) < 0.6,
                'level': AlertLevel.INFO,
                'title': 'Incertitude Prédiction Élevée',
                'message_template': 'Prédiction avec faible confiance: {confidence:.1%}',
                'actions': [
                    'Collecter plus de données pour cette entreprise',
                    'Utiliser plusieurs modèles pour validation croisée',
                    'Marquer pour révision manuelle'
                ]
            },
            {
                'name': 'anomaly_detected',
                'condition': lambda anomaly: anomaly.get('severity') in ['critical', 'high'],
                'level': AlertLevel.ERROR,
                'title': 'Anomalie Critique Détectée',
                'message_template': 'Anomalie {severity} sur {entity}: {description}',
                'actions': [
                    'Investigation immédiate requise',
                    'Vérifier la cohérence des données',
                    'Alerter l\'équipe d\'analyse'
                ]
            },
            {
                'name': 'data_drift_detected',
                'condition': lambda drift: drift.get('p_value', 1) < 0.05,
                'level': AlertLevel.WARNING,
                'title': 'Dérive de Données Détectée',
                'message_template': 'Dérive significative sur feature {feature_name}',
                'actions': [
                    'Analyser les changements dans les données source',
                    'Évaluer l\'impact sur les modèles',
                    'Planifier adaptation des modèles'
                ]
            }
        ]
    
    async def check_alerts(self, system_metrics: Dict[str, Any]) -> List[DashboardAlert]:
        """Vérifie et génère des alertes basées sur les métriques système"""
        
        new_alerts = []
        
        for rule in self.alert_rules:
            try:
                # Vérifier la condition
                if self._evaluate_condition(rule, system_metrics):
                    alert = self._create_alert(rule, system_metrics)
                    new_alerts.append(alert)
                    self.active_alerts[alert.alert_id] = alert
                    self.alert_history.append(alert)
                    
            except Exception as e:
                logger.warning(f"Erreur évaluation règle {rule.get('name')}: {e}")
        
        return new_alerts
    
    def _evaluate_condition(self, rule: Dict[str, Any], metrics: Dict[str, Any]) -> bool:
        """Évalue une condition d'alerte"""
        
        condition = rule.get('condition')
        if not condition:
            return False
        
        try:
            return condition(metrics)
        except Exception:
            return False
    
    def _create_alert(self, rule: Dict[str, Any], context: Dict[str, Any]) -> DashboardAlert:
        """Crée une alerte basée sur une règle"""
        
        alert_id = f"alert_{rule['name']}_{int(time.time())}"
        
        # Formatage du message
        message = rule.get('message_template', '').format(**context)
        
        return DashboardAlert(
            alert_id=alert_id,
            level=rule['level'],
            title=rule['title'],
            message=message,
            source_component=rule['name'],
            trigger_data=context,
            suggested_actions=rule.get('actions', [])
        )
    
    def acknowledge_alert(self, alert_id: str, user_id: str):
        """Acquitte une alerte"""
        if alert_id in self.active_alerts:
            self.active_alerts[alert_id].acknowledged_at = datetime.now()
            logger.info(f"🔔 Alerte {alert_id} acquittée par {user_id}")
    
    def resolve_alert(self, alert_id: str, user_id: str):
        """Résout une alerte"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.resolved_at = datetime.now()
            del self.active_alerts[alert_id]
            logger.info(f"✅ Alerte {alert_id} résolue par {user_id}")


class DashboardEngine:
    """Moteur principal du dashboard IA"""
    
    def __init__(self):
        self.visualization_engine = VisualizationEngine()
        self.explanation_engine = ExplanationEngine()
        self.alerting_system = AlertingSystem()
        
        # Configurations des dashboards
        self.dashboard_configs: Dict[str, DashboardConfig] = {}
        
        # Cache des données
        self.data_cache: Dict[str, Any] = {}
        self.cache_ttl: Dict[str, datetime] = {}
        
        # Métriques système
        self.system_metrics: Dict[str, Any] = {}
        
        logger.info("📊 Dashboard IA Engine initialisé")
    
    async def initialize_dashboard_system(self):
        """Initialise le système de dashboard"""
        
        try:
            logger.info("🚀 Initialisation système dashboard IA")
            
            # Créer dashboards par défaut
            await self._create_default_dashboards()
            
            # Initialiser les explainers
            ai_engine = await get_advanced_ai_engine()
            if hasattr(ai_engine, 'ensemble_manager') and ai_engine.ensemble_manager.models:
                # Données d'entraînement simulées
                training_data = pd.DataFrame({
                    'chiffre_affaires': np.random.exponential(1000000, 1000),
                    'effectifs': np.random.poisson(20, 1000),
                    'company_age': np.random.uniform(1, 30, 1000),
                    'productivity': np.random.normal(50000, 15000, 1000),
                    'growth_rate': np.random.normal(0.1, 0.3, 1000)
                })
                
                await self.explanation_engine.initialize_explainers(
                    ai_engine.ensemble_manager.models,
                    training_data
                )
            
            logger.info("✅ Système dashboard IA initialisé")
            
        except Exception as e:
            logger.error(f"❌ Erreur initialisation dashboard: {e}")
            raise
    
    async def _create_default_dashboards(self):
        """Crée les dashboards par défaut"""
        
        # Dashboard Overview
        overview_widgets = [
            DashboardWidget(
                widget_id="overview_metrics",
                widget_type=VisualizationType.GAUGE_CHART,
                title="Score Global IA",
                description="Performance globale des modèles IA",
                config={"min_val": 0, "max_val": 100, "thresholds": {"good": 80, "warning": 60}},
                data_source="ai_performance",
                refresh_interval=300,
                grid_position={"x": 0, "y": 0, "width": 6, "height": 4}
            ),
            DashboardWidget(
                widget_id="predictions_timeline",
                widget_type=VisualizationType.LINE_CHART,
                title="Évolution des Prédictions",
                description="Timeline des prédictions au fil du temps",
                config={"x_col": "timestamp", "y_col": "prediction_count"},
                data_source="predictions_timeline",
                refresh_interval=600,
                grid_position={"x": 6, "y": 0, "width": 6, "height": 4}
            ),
            DashboardWidget(
                widget_id="model_performance",
                widget_type=VisualizationType.BAR_CHART,
                title="Performance par Modèle",
                description="Comparaison performance des différents modèles",
                config={"x_col": "model_name", "y_col": "accuracy"},
                data_source="model_metrics",
                refresh_interval=900,
                grid_position={"x": 0, "y": 4, "width": 12, "height": 4}
            )
        ]
        
        overview_dashboard = DashboardConfig(
            dashboard_id="overview",
            dashboard_type=DashboardType.OVERVIEW,
            title="Vue d'Ensemble IA",
            description="Dashboard principal avec métriques globales",
            widgets=overview_widgets,
            layout={"columns": 12, "rows": 8},
            owner_id="system",
            public=True
        )
        
        self.dashboard_configs["overview"] = overview_dashboard
        
        # Dashboard Explications
        explanation_widgets = [
            DashboardWidget(
                widget_id="feature_importance",
                widget_type=VisualizationType.FEATURE_IMPORTANCE,
                title="Importance des Features",
                description="Features les plus influentes dans les prédictions",
                config={},
                data_source="feature_importance",
                refresh_interval=1800,
                grid_position={"x": 0, "y": 0, "width": 6, "height": 6}
            ),
            DashboardWidget(
                widget_id="shap_summary",
                widget_type=VisualizationType.SHAP_PLOTS,
                title="Analyse SHAP",
                description="Explications SHAP des prédictions",
                config={},
                data_source="shap_analysis",
                refresh_interval=1800,
                grid_position={"x": 6, "y": 0, "width": 6, "height": 6}
            )
        ]
        
        explanation_dashboard = DashboardConfig(
            dashboard_id="explanations",
            dashboard_type=DashboardType.EXPLANATIONS,
            title="Explications IA",
            description="Explications détaillées des prédictions",
            widgets=explanation_widgets,
            layout={"columns": 12, "rows": 6},
            owner_id="system",
            public=True
        )
        
        self.dashboard_configs["explanations"] = explanation_dashboard
    
    @cached('dashboard_data', ttl_seconds=300)
    async def get_dashboard_data(self, dashboard_id: str) -> Dict[str, Any]:
        """Récupère les données pour un dashboard"""
        
        if dashboard_id not in self.dashboard_configs:
            raise ValueError(f"Dashboard {dashboard_id} non trouvé")
        
        config = self.dashboard_configs[dashboard_id]
        dashboard_data = {}
        
        # Collecter données pour chaque widget
        for widget in config.widgets:
            try:
                widget_data = await self._get_widget_data(widget)
                dashboard_data[widget.widget_id] = {
                    "data": widget_data,
                    "last_updated": datetime.now().isoformat(),
                    "config": widget.config
                }
            except Exception as e:
                logger.error(f"❌ Erreur données widget {widget.widget_id}: {e}")
                dashboard_data[widget.widget_id] = {
                    "data": None,
                    "error": str(e),
                    "last_updated": datetime.now().isoformat()
                }
        
        return {
            "dashboard_id": dashboard_id,
            "title": config.title,
            "description": config.description,
            "widgets": dashboard_data,
            "layout": config.layout,
            "last_updated": datetime.now().isoformat()
        }
    
    async def _get_widget_data(self, widget: DashboardWidget) -> Dict[str, Any]:
        """Récupère les données pour un widget spécifique"""
        
        data_source = widget.data_source
        
        if data_source == "ai_performance":
            return await self._get_ai_performance_data()
        elif data_source == "predictions_timeline":
            return await self._get_predictions_timeline_data()
        elif data_source == "model_metrics":
            return await self._get_model_metrics_data()
        elif data_source == "feature_importance":
            return await self._get_feature_importance_data()
        elif data_source == "shap_analysis":
            return await self._get_shap_analysis_data()
        else:
            return {"error": f"Source de données {data_source} non supportée"}
    
    async def _get_ai_performance_data(self) -> Dict[str, Any]:
        """Récupère les métriques de performance IA globales"""
        
        try:
            # Collecter métriques depuis tous les moteurs IA
            ai_engine = await get_advanced_ai_engine()
            
            # Score global basé sur plusieurs facteurs
            performance_score = 85  # Simulation - à calculer réellement
            
            return {
                "type": "gauge",
                "value": performance_score,
                "title": "Performance Globale IA",
                "thresholds": {"good": 80, "warning": 60}
            }
            
        except Exception as e:
            logger.error(f"Erreur métriques IA: {e}")
            return {"type": "gauge", "value": 0, "error": str(e)}
    
    async def _get_predictions_timeline_data(self) -> Dict[str, Any]:
        """Récupère les données timeline des prédictions"""
        
        # Simulation de données timeline
        dates = pd.date_range(start=datetime.now() - timedelta(days=30), end=datetime.now(), freq='D')
        predictions = np.random.poisson(50, len(dates))
        
        return {
            "type": "line_chart",
            "data": {
                "timestamp": [d.isoformat() for d in dates],
                "prediction_count": predictions.tolist()
            }
        }
    
    async def _get_model_metrics_data(self) -> Dict[str, Any]:
        """Récupère les métriques par modèle"""
        
        # Simulation de performance par modèle
        models = ['RandomForest', 'XGBoost', 'LightGBM', 'NeuralNetwork']
        accuracies = [0.87, 0.89, 0.85, 0.82]
        
        return {
            "type": "bar_chart",
            "data": {
                "model_name": models,
                "accuracy": accuracies
            }
        }
    
    async def _get_feature_importance_data(self) -> Dict[str, Any]:
        """Récupère les données d'importance des features"""
        
        # Simulation d'importance des features
        features = [
            'chiffre_affaires', 'effectifs', 'company_age', 'productivity',
            'growth_rate', 'sector_tech', 'localisation_paris', 'debt_ratio'
        ]
        importance = np.random.dirichlet(np.ones(len(features))) * 100
        
        feature_importance = dict(zip(features, importance))
        
        return {
            "type": "feature_importance",
            "data": feature_importance
        }
    
    async def _get_shap_analysis_data(self) -> Dict[str, Any]:
        """Récupère les analyses SHAP"""
        
        # Simulation d'analyse SHAP
        return {
            "type": "shap_plots",
            "data": {
                "summary_plot": "base64_encoded_plot_data",
                "feature_plots": {},
                "explanation": "Analyse SHAP des contributions des features"
            }
        }
    
    async def generate_prediction_explanation(
        self, 
        model_name: str, 
        prediction_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Génère une explication pour une prédiction"""
        
        try:
            # Récupérer le modèle
            ai_engine = await get_advanced_ai_engine()
            
            if model_name not in ai_engine.ensemble_manager.models:
                raise ValueError(f"Modèle {model_name} non trouvé")
            
            model = ai_engine.ensemble_manager.models[model_name]
            
            # Convertir données en Series pandas
            instance = pd.Series(prediction_data)
            
            # Générer explication
            explanation = await self.explanation_engine.explain_prediction(
                model_name, model, instance, prediction_data.get('prediction')
            )
            
            return {
                "explanation_id": explanation.prediction_id,
                "model_name": explanation.model_name,
                "prediction": explanation.prediction_value,
                "confidence": explanation.confidence,
                "business_interpretation": explanation.business_interpretation,
                "feature_importance": explanation.feature_importance,
                "risk_factors": explanation.risk_factors,
                "opportunities": explanation.opportunities,
                "actionable_insights": explanation.actionable_insights,
                "next_steps": explanation.next_steps,
                "shap_values": explanation.shap_values,
                "lime_explanation": explanation.lime_explanation,
                "timestamp": explanation.created_at.isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Erreur génération explication: {e}")
            return {"error": str(e)}
    
    async def get_system_alerts(self) -> List[Dict[str, Any]]:
        """Récupère les alertes système actives"""
        
        # Collecter métriques système
        await self._update_system_metrics()
        
        # Vérifier alertes
        new_alerts = await self.alerting_system.check_alerts(self.system_metrics)
        
        # Convertir en format API
        alerts_data = []
        for alert in self.alerting_system.active_alerts.values():
            alerts_data.append({
                "alert_id": alert.alert_id,
                "level": alert.level.value,
                "title": alert.title,
                "message": alert.message,
                "source": alert.source_component,
                "created_at": alert.created_at.isoformat(),
                "acknowledged": alert.acknowledged_at is not None,
                "suggested_actions": alert.suggested_actions
            })
        
        return alerts_data
    
    async def _update_system_metrics(self):
        """Met à jour les métriques système"""
        
        try:
            # Métriques des différents composants IA
            ai_engine = await get_advanced_ai_engine()
            anomaly_engine = await get_anomaly_detection_engine()
            learning_engine = await get_continuous_learning_engine()
            
            self.system_metrics = {
                "accuracy": 0.85,  # Simulation
                "model_name": "ensemble",
                "confidence": 0.9,
                "anomaly_count": len(anomaly_engine.anomaly_history),
                "learning_jobs_running": len([j for j in learning_engine.learning_jobs.values() if j.status == "running"]),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Erreur mise à jour métriques: {e}")
    
    def create_custom_dashboard(
        self, 
        user_id: str, 
        title: str, 
        widgets: List[Dict[str, Any]]
    ) -> str:
        """Crée un dashboard personnalisé"""
        
        dashboard_id = f"custom_{user_id}_{int(time.time())}"
        
        # Convertir widgets dict en objets DashboardWidget
        widget_objects = []
        for widget_data in widgets:
            widget = DashboardWidget(
                widget_id=widget_data.get('widget_id', f"widget_{len(widget_objects)}"),
                widget_type=VisualizationType(widget_data.get('type', VisualizationType.BAR_CHART)),
                title=widget_data.get('title', 'Widget'),
                description=widget_data.get('description', ''),
                config=widget_data.get('config', {}),
                data_source=widget_data.get('data_source', ''),
                refresh_interval=widget_data.get('refresh_interval', 300),
                grid_position=widget_data.get('grid_position', {"x": 0, "y": 0, "width": 6, "height": 4})
            )
            widget_objects.append(widget)
        
        dashboard_config = DashboardConfig(
            dashboard_id=dashboard_id,
            dashboard_type=DashboardType.BUSINESS_INSIGHTS,
            title=title,
            description=f"Dashboard personnalisé créé par {user_id}",
            widgets=widget_objects,
            layout={"columns": 12, "rows": 8},
            owner_id=user_id
        )
        
        self.dashboard_configs[dashboard_id] = dashboard_config
        
        logger.info(f"📊 Dashboard personnalisé créé: {dashboard_id}")
        
        return dashboard_id
    
    def get_available_dashboards(self, user_id: str) -> List[Dict[str, Any]]:
        """Retourne la liste des dashboards disponibles pour un utilisateur"""
        
        available = []
        
        for dashboard_id, config in self.dashboard_configs.items():
            # Vérifier permissions
            if config.public or config.owner_id == user_id or user_id in config.shared_with:
                available.append({
                    "dashboard_id": dashboard_id,
                    "title": config.title,
                    "description": config.description,
                    "type": config.dashboard_type.value,
                    "owner": config.owner_id,
                    "created_at": config.created_at.isoformat(),
                    "widget_count": len(config.widgets)
                })
        
        return available
    
    def get_dashboard_analytics(self) -> Dict[str, Any]:
        """Retourne les analytics du système de dashboard"""
        
        total_dashboards = len(self.dashboard_configs)
        total_widgets = sum(len(config.widgets) for config in self.dashboard_configs.values())
        active_alerts = len(self.alerting_system.active_alerts)
        
        # Distribution par type
        type_distribution = {}
        for config in self.dashboard_configs.values():
            dashboard_type = config.dashboard_type.value
            type_distribution[dashboard_type] = type_distribution.get(dashboard_type, 0) + 1
        
        return {
            "total_dashboards": total_dashboards,
            "total_widgets": total_widgets,
            "active_alerts": active_alerts,
            "dashboard_type_distribution": type_distribution,
            "cached_data_points": len(self.data_cache),
            "system_health": "operational",
            "last_updated": datetime.now().isoformat()
        }


# Instance globale
_dashboard_engine: Optional[DashboardEngine] = None


async def get_dashboard_engine() -> DashboardEngine:
    """Factory pour obtenir le moteur de dashboard"""
    global _dashboard_engine
    
    if _dashboard_engine is None:
        _dashboard_engine = DashboardEngine()
        await _dashboard_engine.initialize_dashboard_system()
    
    return _dashboard_engine


# Fonctions utilitaires

async def get_dashboard_data_api(dashboard_id: str, user_id: str) -> Dict[str, Any]:
    """Interface API pour récupérer données de dashboard"""
    
    engine = await get_dashboard_engine()
    
    # Vérifier permissions
    if dashboard_id not in engine.dashboard_configs:
        return {"error": f"Dashboard {dashboard_id} non trouvé"}
    
    config = engine.dashboard_configs[dashboard_id]
    if not (config.public or config.owner_id == user_id or user_id in config.shared_with):
        return {"error": "Accès non autorisé"}
    
    return await engine.get_dashboard_data(dashboard_id)


async def explain_prediction_api(
    model_name: str, 
    prediction_data: Dict[str, Any]
) -> Dict[str, Any]:
    """Interface API pour explications de prédictions"""
    
    engine = await get_dashboard_engine()
    return await engine.generate_prediction_explanation(model_name, prediction_data)


async def get_ai_insights_summary() -> Dict[str, Any]:
    """Résumé des insights IA pour tableau de bord"""
    
    engine = await get_dashboard_engine()
    
    # Collecter insights de tous les composants
    ai_engine = await get_advanced_ai_engine()
    anomaly_engine = await get_anomaly_detection_engine()
    recommendations = await get_recommendation_system()
    
    return {
        "total_predictions_today": 150,  # Simulation
        "average_confidence": 0.87,
        "anomalies_detected": len(anomaly_engine.anomaly_history),
        "recommendations_generated": 25,
        "model_performance": {
            "ensemble_accuracy": 0.89,
            "best_performing_model": "XGBoost",
            "models_count": len(ai_engine.ensemble_manager.models)
        },
        "alerts": await engine.get_system_alerts(),
        "last_updated": datetime.now().isoformat()
    }