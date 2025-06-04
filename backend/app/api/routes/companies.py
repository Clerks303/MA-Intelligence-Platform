from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, Form, status
from fastapi.responses import StreamingResponse
from typing import List, Optional
from sqlalchemy.orm import Session
from sqlalchemy import or_
from app.models.schemas import Company, CompanyCreate, CompanyUpdate, CompanyDetail, FilterParams
from app.core.database import get_db
from app.core.dependencies import get_current_active_user
from app.models.user import User
from app.models.company import Company as CompanyModel
from app.services.data_processing import process_csv_file
import logging
import io
import csv

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/", response_model=Company)
async def create_company(
    company: CompanyCreate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Create a new company manually"""
    try:
        # Validation SIREN unique
        existing_company = db.query(CompanyModel).filter(CompanyModel.siren == company.siren).first()
        if existing_company:
            raise HTTPException(status_code=400, detail="SIREN already exists")
        
        # Préparation des données
        company_data = company.model_dump()
        company_data['score_prospection'] = 50  # Score par défaut
        
        # Création du modèle
        new_company = CompanyModel(**company_data)
        
        # Insertion
        db.add(new_company)
        db.commit()
        db.refresh(new_company)
        
        logger.info(f"Company created manually: {company.siren} by {current_user.username}")
        return new_company
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error creating company: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/", response_model=List[Company])
async def get_companies(
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get all companies with pagination"""
    try:
        companies = db.query(CompanyModel).offset(skip).limit(limit).all()
        return companies
    except Exception as e:
        logger.error(f"Error fetching companies: {e}")
        raise HTTPException(status_code=500, detail=str(e))




@router.post("/filter", response_model=List[Company])
async def filter_companies(
    filters: FilterParams,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Filter companies based on criteria"""
    try:
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
            query = query.filter(
                or_(
                    CompanyModel.nom_entreprise.ilike(f'%{filters.search}%'),
                    CompanyModel.siren.ilike(f'%{filters.search}%')
                )
            )
        
        companies = query.order_by(CompanyModel.score_prospection.desc()).all()
        return companies
    except Exception as e:
        logger.error(f"Error filtering companies: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{siren}", response_model=CompanyDetail)
async def get_company(
    siren: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get company details by SIREN"""
    try:
        # Get company
        company = db.query(CompanyModel).filter(CompanyModel.siren == siren).first()
        if not company:
            raise HTTPException(status_code=404, detail="Company not found")
        
        # Convert to dict for CompanyDetail
        company_dict = {
            "id": company.id,
            "siren": company.siren,
            "siret": company.siret,
            "nom_entreprise": company.nom_entreprise,
            "forme_juridique": company.forme_juridique,
            "date_creation": company.date_creation,
            "adresse": company.adresse,
            "ville": company.ville,
            "code_postal": company.code_postal,
            "email": company.email,
            "telephone": company.telephone,
            "numero_tva": company.numero_tva,
            "chiffre_affaires": company.chiffre_affaires,
            "resultat": company.resultat,
            "effectif": company.effectif,
            "capital_social": company.capital_social,
            "code_naf": company.code_naf,
            "libelle_code_naf": company.libelle_code_naf,
            "dirigeant_principal": company.dirigeant_principal,
            "dirigeants_json": company.dirigeants_json,
            "statut": company.statut,
            "score_prospection": company.score_prospection,
            "score_details": company.score_details,
            "description": company.description,
            "details_complets": company.details_complets,
            "created_at": company.created_at,
            "updated_at": company.updated_at,
            "activity_logs": []  # TODO: Implement activity logs if needed
        }
        
        return company_dict
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching company {siren}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/{siren}", response_model=Company)
async def update_company(
    siren: str,
    company_update: CompanyUpdate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Update company information"""
    try:
        company = db.query(CompanyModel).filter(CompanyModel.siren == siren).first()
        if not company:
            raise HTTPException(status_code=404, detail="Company not found")
        
        update_data = company_update.model_dump(exclude_unset=True)
        
        for field, value in update_data.items():
            setattr(company, field, value)
        
        db.commit()
        db.refresh(company)
        
        return company
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error updating company {siren}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{siren}")
async def delete_company(
    siren: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Delete a company"""
    try:
        company = db.query(CompanyModel).filter(CompanyModel.siren == siren).first()
        if not company:
            raise HTTPException(status_code=404, detail="Company not found")
        
        db.delete(company)
        db.commit()
        
        return {"success": True, "message": "Company deleted"}
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error deleting company {siren}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/upload")
async def upload_csv(
    file: UploadFile = File(...),
    update_existing: bool = False,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Upload and process CSV file"""
    try:
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Only CSV files are allowed")
        
        result = await process_csv_file(file, db, update_existing)
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing CSV: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/export")
async def export_companies(
    filters: FilterParams,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Export filtered companies as CSV"""
    try:
        # Utiliser la même logique que filter_companies
        query = db.query(CompanyModel)
        
        if filters.ca_min:
            query = query.filter(CompanyModel.chiffre_affaires >= filters.ca_min)
        if filters.effectif_min:
            query = query.filter(CompanyModel.effectif >= filters.effectif_min)
        if filters.ville and filters.ville != 'all':
            query = query.filter(CompanyModel.adresse.ilike(f'%{filters.ville}%'))
        if filters.statut and filters.statut != 'all':
            query = query.filter(CompanyModel.statut == filters.statut)
        if filters.search:
            query = query.filter(
                or_(
                    CompanyModel.nom_entreprise.ilike(f'%{filters.search}%'),
                    CompanyModel.siren.ilike(f'%{filters.search}%')
                )
            )
        
        companies = query.order_by(CompanyModel.score_prospection.desc()).all()
        
        # Créer le CSV en mémoire
        output = io.StringIO()
        fieldnames = [
            'siren', 'nom_entreprise', 'dirigeant_principal', 'adresse', 'ville', 
            'code_postal', 'email', 'telephone', 'chiffre_affaires', 'effectif', 
            'statut', 'score_prospection', 'date_creation'
        ]
        writer = csv.DictWriter(output, fieldnames=fieldnames, delimiter=';')
        writer.writeheader()
        
        for company in companies:
            # Nettoyer les données pour l'export
            cleaned_company = {}
            for field in fieldnames:
                value = getattr(company, field, '')
                if value is None:
                    value = ''
                cleaned_company[field] = value
            writer.writerow(cleaned_company)
        
        # Retourner le CSV comme réponse streaming
        output.seek(0)
        
        headers = {
            'Content-Disposition': f'attachment; filename="entreprises_export_{len(companies)}_companies.csv"'
        }
        
        logger.info(f"CSV export requested by {current_user.username}: {len(companies)} companies")
        
        return StreamingResponse(
            io.BytesIO(output.getvalue().encode('utf-8-sig')),
            media_type='text/csv',
            headers=headers
        )
        
    except Exception as e:
        logger.error(f"Error exporting companies: {e}")
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")