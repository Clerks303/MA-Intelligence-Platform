from pydantic import BaseModel, EmailStr
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum
from uuid import UUID

class StatusEnum(str, Enum):
    PROSPECT = "prospect"
    CONTACT = "contact"
    QUALIFICATION = "qualification"
    NEGOCIATION = "negociation"
    CLIENT = "client"
    PERDU = "perdu"

class CompanyBase(BaseModel):
    siren: str
    nom_entreprise: str
    forme_juridique: Optional[str] = None
    date_creation: Optional[datetime] = None
    adresse: Optional[str] = None
    ville: Optional[str] = None
    code_postal: Optional[str] = None
    email: Optional[EmailStr] = None
    telephone: Optional[str] = None
    numero_tva: Optional[str] = None
    chiffre_affaires: Optional[float] = None
    resultat: Optional[float] = None
    effectif: Optional[int] = None
    capital_social: Optional[float] = None
    code_naf: Optional[str] = None
    libelle_code_naf: Optional[str] = None
    dirigeant_principal: Optional[str] = None
    statut: StatusEnum = StatusEnum.PROSPECT
    score_prospection: Optional[float] = None
    description: Optional[str] = None

class CompanyCreate(CompanyBase):
    pass

class CompanyUpdate(BaseModel):
    nom_entreprise: Optional[str] = None
    email: Optional[EmailStr] = None
    telephone: Optional[str] = None
    adresse: Optional[str] = None
    ville: Optional[str] = None
    code_postal: Optional[str] = None
    dirigeant_principal: Optional[str] = None
    chiffre_affaires: Optional[float] = None
    statut: Optional[StatusEnum] = None
    effectif: Optional[int] = None
    capital_social: Optional[float] = None
    description: Optional[str] = None

class Company(CompanyBase):
    id: UUID
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True

class CompanyDetail(Company):
    dirigeants_json: Optional[Dict[str, Any]] = None
    score_details: Optional[Dict[str, Any]] = None
    activity_logs: Optional[List[Dict[str, Any]]] = None
    details_complets: Optional[Dict[str, Any]] = None

class ScrapingStatus(BaseModel):
    is_running: bool
    progress: int
    message: str
    error: Optional[str] = None
    new_companies: int = 0
    skipped_companies: int = 0
    source: Optional[str] = None

class Stats(BaseModel):
    total: int
    ca_moyen: float
    ca_total: float
    effectif_moyen: float
    avec_email: int
    avec_telephone: int
    taux_email: float
    taux_telephone: float
    par_statut: Dict[str, int]

class FilterParams(BaseModel):
    ca_min: Optional[float] = None
    effectif_min: Optional[int] = None
    ville: Optional[str] = None
    statut: Optional[StatusEnum] = None
    search: Optional[str] = None

class Token(BaseModel):
    access_token: str
    token_type: str

class UserLogin(BaseModel):
    username: str
    password: str

class InfogreffeRequest(BaseModel):
    siren_list: Optional[List[str]] = None
    max_companies: Optional[int] = 1000


# ========================================
# SCHEMAS POUR DASHBOARD IA - US-010
# ========================================

class DashboardTypeEnum(str, Enum):
    OVERVIEW = "overview"
    MODEL_PERFORMANCE = "model_performance"
    PREDICTIONS = "predictions"
    EXPLANATIONS = "explanations"
    ANOMALIES = "anomalies"
    RECOMMENDATIONS = "recommendations"
    LEARNING = "learning"
    BUSINESS_INSIGHTS = "business_insights"


class VisualizationTypeEnum(str, Enum):
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


class AlertLevelEnum(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class DashboardWidget(BaseModel):
    widget_id: str
    widget_type: VisualizationTypeEnum
    title: str
    description: str
    config: Dict[str, Any]
    data_source: str
    refresh_interval: int
    grid_position: Dict[str, int]
    required_permissions: Optional[List[str]] = []


class DashboardAlert(BaseModel):
    alert_id: str
    level: AlertLevelEnum
    title: str
    message: str
    source_component: str
    trigger_data: Dict[str, Any]
    suggested_actions: List[str]
    auto_resolve: bool = False
    created_at: datetime
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None


class DashboardConfig(BaseModel):
    dashboard_id: str
    dashboard_type: DashboardTypeEnum
    title: str
    description: str
    widgets: List[DashboardWidget]
    layout: Dict[str, Any]
    owner_id: str
    shared_with: List[str] = []
    public: bool = False
    auto_refresh: bool = True
    refresh_interval: int = 300
    created_at: datetime
    last_modified: datetime
    tags: List[str] = []


class PredictionExplanation(BaseModel):
    explanation_id: str
    model_name: str
    prediction: Any
    confidence: float
    business_interpretation: str
    feature_importance: Dict[str, float]
    risk_factors: List[str]
    opportunities: List[str]
    actionable_insights: List[str]
    next_steps: List[str]
    shap_values: Optional[Dict[str, float]] = None
    lime_explanation: Optional[Dict[str, Any]] = None
    timestamp: datetime


class CreateDashboardRequest(BaseModel):
    title: str
    description: Optional[str] = ""
    dashboard_type: DashboardTypeEnum = DashboardTypeEnum.BUSINESS_INSIGHTS
    widgets: List[Dict[str, Any]]
    layout: Optional[Dict[str, Any]] = {"columns": 12, "rows": 8}
    public: bool = False
    auto_refresh: bool = True
    refresh_interval: int = 300
    tags: List[str] = []


class ExplainPredictionRequest(BaseModel):
    model_name: str
    prediction_data: Dict[str, Any]
    include_shap: bool = True
    include_lime: bool = True
    include_business_context: bool = True


class DashboardSummary(BaseModel):
    dashboard_id: str
    title: str
    description: str
    type: DashboardTypeEnum
    owner: str
    created_at: datetime
    widget_count: int
    public: bool


class AIInsightsSummary(BaseModel):
    total_predictions_today: int
    average_confidence: float
    anomalies_detected: int
    recommendations_generated: int
    model_performance: Dict[str, Any]
    alerts: List[DashboardAlert]
    last_updated: datetime


class ModelPerformanceMetrics(BaseModel):
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    training_date: Optional[datetime] = None
    last_prediction: Optional[datetime] = None
    prediction_count: int = 0


class FeatureImportanceData(BaseModel):
    feature_name: str
    importance: float
    description: Optional[str] = None
    category: Optional[str] = None


class SystemHealthStatus(BaseModel):
    overall_status: str
    dashboard_engine: str
    visualization_engine: str
    explanation_engine: str
    alerting_system: str
    total_dashboards: int
    active_alerts: int
    cache_size: int
    last_check: datetime


class AlertAction(BaseModel):
    action_type: str  # acknowledge, resolve, escalate
    user_id: str
    timestamp: datetime
    comment: Optional[str] = None