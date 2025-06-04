from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import func, distinct
from app.models.schemas import Stats, FilterParams
from app.core.database import get_db
from app.core.dependencies import get_current_active_user
from app.models.user import User
from app.models.company import Company as CompanyModel
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/", response_model=Stats)
async def get_stats(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get overall statistics"""
    try:
        # Get all companies
        companies = db.query(CompanyModel).all()
        
        if not companies:
            return Stats(
                total=0, ca_moyen=0, ca_total=0, effectif_moyen=0,
                avec_email=0, avec_telephone=0, taux_email=0, taux_telephone=0,
                par_statut={}
            )
        
        total = len(companies)
        
        # Calculate stats
        ca_values = [c.chiffre_affaires for c in companies if c.chiffre_affaires is not None]
        effectif_values = [c.effectif for c in companies if c.effectif is not None]
        
        ca_total = sum(ca_values) if ca_values else 0
        ca_moyen = ca_total / len(ca_values) if ca_values else 0
        effectif_moyen = sum(effectif_values) / len(effectif_values) if effectif_values else 0
        
        avec_email = len([c for c in companies if c.email])
        avec_telephone = len([c for c in companies if c.telephone])
        
        # Count by status
        par_statut = {}
        for company in companies:
            statut = str(company.statut) if company.statut else 'unknown'
            par_statut[statut] = par_statut.get(statut, 0) + 1
        
        stats = {
            'total': total,
            'ca_moyen': ca_moyen,
            'ca_total': ca_total,
            'effectif_moyen': effectif_moyen,
            'avec_email': avec_email,
            'avec_telephone': avec_telephone,
            'taux_email': (avec_email / total * 100) if total > 0 else 0,
            'taux_telephone': (avec_telephone / total * 100) if total > 0 else 0,
            'par_statut': par_statut
        }
        
        return Stats(**stats)
    except Exception as e:
        logger.error(f"Error calculating stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/filtered", response_model=Stats)
async def get_filtered_stats(
    filters: FilterParams,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get statistics for filtered data"""
    try:
        # Build query with filters
        query = db.query(CompanyModel)
        
        if filters.ca_min:
            query = query.filter(CompanyModel.chiffre_affaires >= filters.ca_min)
        if filters.effectif_min:
            query = query.filter(CompanyModel.effectif >= filters.effectif_min)
        if filters.ville:
            query = query.filter(CompanyModel.adresse.ilike(f'%{filters.ville}%'))
        if filters.statut:
            query = query.filter(CompanyModel.statut == filters.statut)
        if filters.search:
            from sqlalchemy import or_
            query = query.filter(
                or_(
                    CompanyModel.nom_entreprise.ilike(f'%{filters.search}%'),
                    CompanyModel.siren.ilike(f'%{filters.search}%')
                )
            )
        
        companies = query.all()
        
        if not companies:
            return Stats(
                total=0, ca_moyen=0, ca_total=0, effectif_moyen=0,
                avec_email=0, avec_telephone=0, taux_email=0, taux_telephone=0,
                par_statut={}
            )
        
        total = len(companies)
        
        # Same calculations as above
        ca_values = [c.chiffre_affaires for c in companies if c.chiffre_affaires is not None]
        effectif_values = [c.effectif for c in companies if c.effectif is not None]
        
        ca_total = sum(ca_values) if ca_values else 0
        ca_moyen = ca_total / len(ca_values) if ca_values else 0
        effectif_moyen = sum(effectif_values) / len(effectif_values) if effectif_values else 0
        
        avec_email = len([c for c in companies if c.email])
        avec_telephone = len([c for c in companies if c.telephone])
        
        # Count by status
        par_statut = {}
        for company in companies:
            statut = str(company.statut) if company.statut else 'unknown'
            par_statut[statut] = par_statut.get(statut, 0) + 1
        
        stats = {
            'total': total,
            'ca_moyen': ca_moyen,
            'ca_total': ca_total,
            'effectif_moyen': effectif_moyen,
            'avec_email': avec_email,
            'avec_telephone': avec_telephone,
            'taux_email': (avec_email / total * 100) if total > 0 else 0,
            'taux_telephone': (avec_telephone / total * 100) if total > 0 else 0,
            'par_statut': par_statut
        }
        
        return Stats(**stats)
    except Exception as e:
        logger.error(f"Error calculating filtered stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/cities")
async def get_cities(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get list of unique cities"""
    try:
        # Get distinct cities from the ville column
        cities_result = db.query(distinct(CompanyModel.ville)).filter(
            CompanyModel.ville.isnot(None),
            CompanyModel.ville != ''
        ).all()
        
        cities = [city[0] for city in cities_result if city[0]]
        cities.sort()
        
        return {"cities": cities}
    except Exception as e:
        logger.error(f"Error fetching cities: {e}")
        raise HTTPException(status_code=500, detail=str(e))