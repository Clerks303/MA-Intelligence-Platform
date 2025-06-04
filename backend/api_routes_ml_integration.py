"""
API Routes Integration - ML Scoring
Modification des routes pour récupérer les scores pré-calculés
M&A Intelligence Platform
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from datetime import datetime

from app.core.deps import get_current_active_user, get_db
from app.models.user import User
from shared.database.supabase_client import create_supabase_client
from scheduler.tasks import trigger_manual_scoring, get_task_status

router = APIRouter()

# ============================================================================
# NOUVELLES ROUTES POUR LES SCORES ML
# ============================================================================

@router.get("/companies/{company_id}/scores")
async def get_company_scores(
    company_id: int,
    current_user: User = Depends(get_current_active_user)
):
    """
    Récupère les scores ML pré-calculés pour une entreprise
    """
    supabase = create_supabase_client()
    
    try:
        # Récupération des scores
        result = await supabase.table('ml_scores')\
            .select('*')\
            .eq('company_id', company_id)\
            .order('calculated_at', desc=True)\
            .limit(1)\
            .execute()
        
        if not result.data:
            raise HTTPException(
                status_code=404, 
                detail="Aucun score trouvé pour cette entreprise"
            )
        
        score_data = result.data[0]
        
        return {
            "company_id": company_id,
            "scores": {
                "score_ma": score_data.get('score_ma'),
                "score_croissance": score_data.get('score_croissance'), 
                "score_stabilite": score_data.get('score_stabilite'),
                "score_composite": score_data.get('score_composite')
            },
            "metadata": {
                "confidence": score_data.get('confidence'),
                "model_version": score_data.get('model_version'),
                "calculated_at": score_data.get('calculated_at'),
                "features_used": score_data.get('features_used', [])
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur récupération scores: {str(e)}")

@router.get("/companies/scores/batch")
async def get_batch_scores(
    company_ids: List[int],
    current_user: User = Depends(get_current_active_user)
):
    """
    Récupère les scores pour plusieurs entreprises
    """
    supabase = create_supabase_client()
    
    try:
        result = await supabase.table('ml_scores')\
            .select('*')\
            .in_('company_id', company_ids)\
            .execute()
        
        # Organiser par company_id
        scores_by_company = {}
        for score in result.data:
            company_id = score['company_id']
            if company_id not in scores_by_company:
                scores_by_company[company_id] = []
            scores_by_company[company_id].append(score)
        
        # Prendre le score le plus récent pour chaque entreprise
        latest_scores = {}
        for company_id, scores in scores_by_company.items():
            latest_score = max(scores, key=lambda x: x['calculated_at'])
            latest_scores[company_id] = {
                "score_ma": latest_score.get('score_ma'),
                "score_croissance": latest_score.get('score_croissance'),
                "score_stabilite": latest_score.get('score_stabilite'),
                "score_composite": latest_score.get('score_composite'),
                "calculated_at": latest_score.get('calculated_at')
            }
        
        return {
            "scores": latest_scores,
            "total_companies": len(company_ids),
            "companies_with_scores": len(latest_scores)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur récupération scores batch: {str(e)}")

# ============================================================================
# ROUTES DE GESTION DU SCORING ML
# ============================================================================

@router.post("/ml/trigger-scoring")
async def trigger_ml_scoring(
    company_ids: Optional[List[int]] = None,
    current_user: User = Depends(get_current_active_user)
):
    """
    Déclenche le calcul ML manuellement
    """
    # Vérifier les permissions (admin uniquement)
    if not current_user.is_superuser:
        raise HTTPException(
            status_code=403, 
            detail="Permission insuffisante pour déclencher le scoring ML"
        )
    
    try:
        # Déclencher la tâche Celery
        task = trigger_manual_scoring.delay(
            company_ids=company_ids,
            user_id=current_user.id
        )
        
        return {
            "task_id": task.id,
            "status": "started",
            "message": "Tâche de scoring ML initiée",
            "companies_count": len(company_ids) if company_ids else "all",
            "triggered_by": current_user.email,
            "triggered_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur déclenchement scoring: {str(e)}")

@router.get("/ml/scoring-status/{task_id}")
async def get_scoring_status(
    task_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """
    Vérifie le statut d'une tâche ML
    """
    try:
        status_info = get_task_status.delay(task_id)
        result = status_info.get(timeout=5)
        
        return {
            "task_id": task_id,
            "status": result.get('status'),
            "result": result.get('result'),
            "info": result.get('info'),
            "checked_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur vérification statut: {str(e)}")

@router.get("/ml/models/info")
async def get_models_info(
    current_user: User = Depends(get_current_active_user)
):
    """
    Récupère les informations sur les modèles ML en production
    """
    supabase = create_supabase_client()
    
    try:
        # Récupérer les informations des modèles
        result = await supabase.table('ml_models')\
            .select('*')\
            .order('created_at', desc=True)\
            .execute()
        
        models_info = {}
        for model in result.data:
            models_info[model['name']] = {
                "version": model['version'],
                "metrics": model.get('metrics', {}),
                "created_at": model['created_at'],
                "status": "active" if model.get('is_active') else "inactive"
            }
        
        return {
            "models": models_info,
            "total_models": len(models_info),
            "last_updated": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur récupération infos modèles: {str(e)}")

# ============================================================================
# ROUTE MODIFIÉE POUR L'ENDPOINT COMPANIES EXISTANT
# ============================================================================

@router.get("/companies/{company_id}")
async def get_company_with_scores(
    company_id: int,
    include_scores: bool = True,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Récupère une entreprise avec ses scores ML (route modifiée)
    """
    supabase = create_supabase_client()
    
    try:
        # Récupération des données de l'entreprise
        company_result = await supabase.table('companies')\
            .select('*')\
            .eq('id', company_id)\
            .single()\
            .execute()
        
        if not company_result.data:
            raise HTTPException(status_code=404, detail="Entreprise non trouvée")
        
        company_data = company_result.data
        
        # Ajouter les scores si demandés
        if include_scores:
            try:
                scores_result = await supabase.table('ml_scores')\
                    .select('*')\
                    .eq('company_id', company_id)\
                    .order('calculated_at', desc=True)\
                    .limit(1)\
                    .execute()
                
                if scores_result.data:
                    score_data = scores_result.data[0]
                    company_data['ml_scores'] = {
                        "score_ma": score_data.get('score_ma'),
                        "score_croissance": score_data.get('score_croissance'),
                        "score_stabilite": score_data.get('score_stabilite'),
                        "score_composite": score_data.get('score_composite'),
                        "confidence": score_data.get('confidence'),
                        "calculated_at": score_data.get('calculated_at')
                    }
                else:
                    company_data['ml_scores'] = None
                    
            except Exception as e:
                # Erreur non bloquante pour les scores
                company_data['ml_scores'] = {"error": f"Erreur récupération scores: {str(e)}"}
        
        return company_data
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur récupération entreprise: {str(e)}")

# ============================================================================
# ROUTE DE STATISTIQUES ML
# ============================================================================

@router.get("/ml/statistics")
async def get_ml_statistics(
    current_user: User = Depends(get_current_active_user)
):
    """
    Récupère les statistiques sur le scoring ML
    """
    supabase = create_supabase_client()
    
    try:
        # Statistiques générales
        total_companies = await supabase.table('companies').select('id', count='exact').execute()
        total_scored = await supabase.table('ml_scores').select('company_id', count='exact').execute()
        
        # Scores récents (dernière semaine)
        recent_scores = await supabase.table('ml_scores')\
            .select('*')\
            .gte('calculated_at', (datetime.utcnow() - timedelta(days=7)).isoformat())\
            .execute()
        
        # Distribution des scores
        all_scores = await supabase.table('ml_scores')\
            .select('score_composite')\
            .not_.is_('score_composite', 'null')\
            .execute()
        
        score_values = [s['score_composite'] for s in all_scores.data]
        
        distribution = {
            "excellent": len([s for s in score_values if s >= 80]),
            "bon": len([s for s in score_values if 60 <= s < 80]),
            "moyen": len([s for s in score_values if 40 <= s < 60]),
            "faible": len([s for s in score_values if s < 40])
        }
        
        return {
            "overview": {
                "total_companies": total_companies.count,
                "companies_scored": total_scored.count,
                "coverage_percentage": round((total_scored.count / total_companies.count) * 100, 2) if total_companies.count > 0 else 0
            },
            "recent_activity": {
                "scores_last_week": len(recent_scores.data),
                "last_scoring_date": max([s['calculated_at'] for s in recent_scores.data]) if recent_scores.data else None
            },
            "score_distribution": distribution,
            "generated_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur récupération statistiques: {str(e)}")


# ============================================================================
# EXEMPLE D'INTÉGRATION DANS app/api/routes/companies.py
# ============================================================================

"""
Pour intégrer ces nouvelles fonctionnalités dans votre API existante:

1. Ajoutez ces routes dans app/api/routes/companies.py
2. Modifiez la route GET /companies/{company_id} existante pour inclure les scores
3. Ajoutez un nouveau router ml.py dans app/api/routes/ml.py
4. Importez et incluez les routers dans app/main.py

Exemple d'intégration dans main.py:

from app.api.routes import companies, ml

app.include_router(companies.router, prefix="/api/v1/companies", tags=["companies"])
app.include_router(ml.router, prefix="/api/v1/ml", tags=["ml"])

"""