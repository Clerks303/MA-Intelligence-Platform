from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from sqlalchemy.orm import Session
from app.models.schemas import ScrapingStatus, InfogreffeRequest
from app.scrapers import pappers, societe, infogreffe
from app.core.database import get_db
from app.core.dependencies import get_current_active_user
from app.models.user import User
from app.models.company import Company as CompanyModel
from app.models.schemas import CompanyCreate, CompanyUpdate
import asyncio
import logging
from typing import Optional

router = APIRouter()
logger = logging.getLogger(__name__)

# Global status tracking
scraping_status = {
    'pappers': ScrapingStatus(is_running=False, progress=0, message='Ready', source='pappers'),
    'societe': ScrapingStatus(is_running=False, progress=0, message='Ready', source='societe'),
    'infogreffe': ScrapingStatus(is_running=False, progress=0, message='Ready', source='infogreffe')
}

@router.get("/status", response_model=ScrapingStatus)
async def get_scraping_status(
    current_user: User = Depends(get_current_active_user)
):
    """Get overall scraping status"""
    # Return combined status
    is_running = any(status.is_running for status in scraping_status.values())
    total_progress = sum(status.progress for status in scraping_status.values()) // 3
    
    return ScrapingStatus(
        is_running=is_running,
        progress=total_progress,
        message="Scraping in progress" if is_running else "Ready",
        new_companies=sum(status.new_companies for status in scraping_status.values()),
        skipped_companies=sum(status.skipped_companies for status in scraping_status.values())
    )

@router.get("/status/{source}", response_model=ScrapingStatus)
async def get_source_status(
    source: str,
    current_user: User = Depends(get_current_active_user)
):
    """Get status for specific scraping source"""
    if source not in scraping_status:
        raise HTTPException(status_code=404, detail=f"Source {source} not found")
    
    return scraping_status[source]

@router.post("/pappers")
async def start_pappers_scraping(
    background_tasks: BackgroundTasks,
    search_terms: Optional[str] = "cabinet comptable",
    max_results: Optional[int] = 100,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Start Pappers API scraping"""
    if scraping_status['pappers'].is_running:
        raise HTTPException(status_code=400, detail="Pappers scraping already running")
    
    background_tasks.add_task(run_pappers_scraping, search_terms, max_results, db, current_user.username)
    
    return {"message": "Pappers scraping started", "search_terms": search_terms, "max_results": max_results}

@router.post("/societe")
async def start_societe_scraping(
    background_tasks: BackgroundTasks,
    search_terms: Optional[str] = "expert comptable",
    max_pages: Optional[int] = 5,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Start Société.com scraping"""
    if scraping_status['societe'].is_running:
        raise HTTPException(status_code=400, detail="Société.com scraping already running")
    
    background_tasks.add_task(run_societe_scraping, search_terms, max_pages, db, current_user.username)
    
    return {"message": "Société.com scraping started", "search_terms": search_terms, "max_pages": max_pages}

@router.post("/infogreffe")
async def start_infogreffe_enrichment(
    request: InfogreffeRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Start Infogreffe enrichment for existing companies"""
    if scraping_status['infogreffe'].is_running:
        raise HTTPException(status_code=400, detail="Infogreffe enrichment already running")
    
    # If no SIREN list provided, get all companies without complete data
    siren_list = request.siren_list
    if not siren_list:
        max_companies = request.max_companies or 1000
        companies = db.query(CompanyModel).filter(
            CompanyModel.details_complets.is_(None)
        ).limit(max_companies).all()
        siren_list = [c.siren for c in companies if c.siren]
    
    if not siren_list:
        raise HTTPException(status_code=400, detail="No companies to enrich")
    
    background_tasks.add_task(run_infogreffe_enrichment, siren_list, db, current_user.username)
    
    return {"message": "Infogreffe enrichment started", "companies_count": len(siren_list)}

# Background task functions
async def run_pappers_scraping(search_terms: str, max_results: int, db: Session, user_info: str):
    """Run Pappers API scraping in background"""
    global scraping_status
    try:
        scraping_status['pappers'] = ScrapingStatus(
            is_running=True,
            progress=0,
            message="Initialisation du scraping Pappers...",
            source='pappers'
        )
        
        # Use Pappers scraper
        results = await pappers.search_companies(search_terms, max_results)
        
        new_count = 0
        skipped_count = 0
        
        for i, company_data in enumerate(results):
            # Update progress
            progress = int((i + 1) / len(results) * 100)
            scraping_status['pappers'].progress = progress
            scraping_status['pappers'].message = f'Processing company {i+1}/{len(results)}'
            
            try:
                # Check if company exists
                existing = db.query(CompanyModel).filter(CompanyModel.siren == company_data['siren']).first()
                if existing:
                    skipped_count += 1
                    continue
                
                # Create company
                company_create = CompanyCreate(**company_data)
                new_company = CompanyModel(**company_create.model_dump())
                new_company.score_prospection = 50
                
                db.add(new_company)
                db.commit()
                new_count += 1
                
            except Exception as e:
                logger.error(f"Error processing company {company_data.get('siren', 'unknown')}: {e}")
                db.rollback()
                skipped_count += 1
        
        scraping_status['pappers'] = ScrapingStatus(
            is_running=False,
            progress=100,
            message=f'Completed: {new_count} new, {skipped_count} skipped',
            new_companies=new_count,
            skipped_companies=skipped_count,
            source='pappers'
        )
        
    except Exception as e:
        logger.error(f"Pappers scraping failed: {e}")
        scraping_status['pappers'] = ScrapingStatus(
            is_running=False,
            progress=0,
            message='Failed',
            error=str(e),
            source='pappers'
        )

async def run_societe_scraping(search_terms: str, max_pages: int, db: Session, user_info: str):
    """Run Société.com scraping in background"""
    global scraping_status
    try:
        scraping_status['societe'] = ScrapingStatus(
            is_running=True,
            progress=0,
            message="Initialisation du scraping Société.com...",
            source='societe'
        )
        
        # Use Société scraper
        results = await societe.scrape_companies(search_terms, max_pages)
        
        new_count = 0
        skipped_count = 0
        
        for i, company_data in enumerate(results):
            # Update progress
            progress = int((i + 1) / len(results) * 100)
            scraping_status['societe'].progress = progress
            scraping_status['societe'].message = f'Processing company {i+1}/{len(results)}'
            
            try:
                # Check if company exists
                existing = db.query(CompanyModel).filter(CompanyModel.siren == company_data['siren']).first()
                if existing:
                    skipped_count += 1
                    continue
                
                # Create company
                company_create = CompanyCreate(**company_data)
                new_company = CompanyModel(**company_create.model_dump())
                new_company.score_prospection = 50
                
                db.add(new_company)
                db.commit()
                new_count += 1
                
            except Exception as e:
                logger.error(f"Error processing company {company_data.get('siren', 'unknown')}: {e}")
                db.rollback()
                skipped_count += 1
        
        scraping_status['societe'] = ScrapingStatus(
            is_running=False,
            progress=100,
            message=f'Completed: {new_count} new, {skipped_count} skipped',
            new_companies=new_count,
            skipped_companies=skipped_count,
            source='societe'
        )
        
    except Exception as e:
        logger.error(f"Société.com scraping failed: {e}")
        scraping_status['societe'] = ScrapingStatus(
            is_running=False,
            progress=0,
            message='Failed',
            error=str(e),
            source='societe'
        )

async def run_infogreffe_enrichment(siren_list: list, db: Session, user_info: str):
    """Run Infogreffe enrichment in background"""
    global scraping_status
    try:
        scraping_status['infogreffe'] = ScrapingStatus(
            is_running=True,
            progress=0,
            message="Initialisation de l'enrichissement Infogreffe...",
            source='infogreffe'
        )
        
        enriched_count = 0
        skipped_count = 0
        
        for i, siren in enumerate(siren_list):
            # Update progress
            progress = int((i + 1) / len(siren_list) * 100)
            scraping_status['infogreffe'].progress = progress
            scraping_status['infogreffe'].message = f'Enriching company {i+1}/{len(siren_list)}'
            
            try:
                # Get enrichment data from Infogreffe
                enrichment_data = await infogreffe.get_company_details(siren)
                
                if not enrichment_data:
                    skipped_count += 1
                    continue
                
                # Update company with enrichment data
                company = db.query(CompanyModel).filter(CompanyModel.siren == siren).first()
                if company:
                    company.details_complets = enrichment_data
                    if 'dirigeants' in enrichment_data:
                        company.dirigeants_json = enrichment_data['dirigeants']
                    if 'forme_juridique' in enrichment_data:
                        company.forme_juridique = enrichment_data['forme_juridique']
                    if 'capital_social' in enrichment_data:
                        company.capital_social = enrichment_data['capital_social']
                    
                    db.commit()
                    enriched_count += 1
                else:
                    skipped_count += 1
                
            except Exception as e:
                logger.error(f"Error enriching company {siren}: {e}")
                db.rollback()
                skipped_count += 1
        
        scraping_status['infogreffe'] = ScrapingStatus(
            is_running=False,
            progress=100,
            message=f'Completed: {enriched_count} enriched, {skipped_count} skipped',
            new_companies=enriched_count,
            skipped_companies=skipped_count,
            source='infogreffe'
        )
        
    except Exception as e:
        logger.error(f"Infogreffe enrichment failed: {e}")
        scraping_status['infogreffe'] = ScrapingStatus(
            is_running=False,
            progress=0,
            message='Failed',
            error=str(e),
            source='infogreffe'
        )