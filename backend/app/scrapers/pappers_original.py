import asyncio
import aiohttp
import logging
from typing import Dict, List, Optional, Set
from datetime import datetime
import os
import json

logger = logging.getLogger(__name__)

class PappersAPIClient:
    """Client asynchrone pour l'API Pappers"""
    
    BASE_URL = "https://api.pappers.fr/v2"
    CODES_NAF = ['6920Z']
    DEPARTEMENTS_IDF = ['75', '77', '78', '91', '92', '93', '94', '95']
    
    def __init__(self, db_client):
        self.api_key = os.environ.get('PAPPERS_API_KEY', '')
        self.db = db_client
        self.session = None
        self.existing_sirens = set()
        self.new_companies_count = 0
        self.skipped_companies_count = 0
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        await self._load_existing_sirens()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def _load_existing_sirens(self) -> Set[str]:
        """Charge les SIREN existants depuis Supabase"""
        try:
            response = self.db.table('cabinets_comptables').select('siren').execute()
            self.existing_sirens = set(str(company['siren']) for company in response.data)
            logger.info(f"Chargé {len(self.existing_sirens)} SIREN existants")
        except Exception as e:
            logger.error(f"Erreur chargement SIREN: {e}")
            
    async def search_companies(self, **params) -> Dict:
        """Recherche asynchrone des entreprises"""
        endpoint = f"{self.BASE_URL}/recherche"
        
        default_params = {
            'api_token': self.api_key,
            'par_page': 100,
            'precision': 'standard'
        }
        
        all_params = {**default_params, **params}
        
        try:
            async with self.session.get(endpoint, params=all_params) as response:
                response.raise_for_status()
                return await response.json()
        except Exception as e:
            logger.error(f"Erreur API Pappers: {e}")
            raise
    
    async def get_company_details(self, siren: str) -> Dict:
        """Récupère les détails d'une entreprise"""
        endpoint = f"{self.BASE_URL}/entreprise"
        params = {
            'api_token': self.api_key,
            'siren': siren
        }
        
        try:
            async with self.session.get(endpoint, params=params) as response:
                response.raise_for_status()
                return await response.json()
        except Exception as e:
            logger.error(f"Erreur détails SIREN {siren}: {e}")
            return {}
    
    async def process_company(self, company_data: Dict) -> Optional[Dict]:
        """Traite et sauvegarde une entreprise"""
        siren = str(company_data.get('siren', ''))
        
        # Vérifier si déjà en base
        if siren in self.existing_sirens:
            self.skipped_companies_count += 1
            return None
        
        # Vérifier le CA
        ca = company_data.get('chiffre_affaires', 0)
        if ca and (ca < 3000000 or ca > 50000000):
            return None
        
        # Formater pour la base
        clean_data = self._format_company_data(company_data)
        
        # Sauvegarder
        try:
            response = self.db.table('cabinets_comptables').insert(clean_data).execute()
            self.new_companies_count += 1
            logger.info(f"Nouvelle entreprise: {clean_data['nom_entreprise']}")
            return clean_data
        except Exception as e:
            logger.error(f"Erreur sauvegarde: {e}")
            return None
    
    def _format_company_data(self, data: Dict) -> Dict:
        """Formate les données pour Supabase"""
        return {
            'siren': str(data.get('siren', '')),
            'siret_siege': data.get('siret_siege', ''),
            'nom_entreprise': data.get('nom_entreprise', ''),
            'forme_juridique': data.get('forme_juridique', ''),
            'date_creation': data.get('date_creation'),
            'adresse': self._format_address(data),
            'email': data.get('email', ''),
            'telephone': data.get('telephone', ''),
            'numero_tva': data.get('numero_tva_intracommunautaire', ''),
            'chiffre_affaires': data.get('chiffre_affaires'),
            'resultat': data.get('resultat'),
            'effectif': data.get('effectif'),
            'capital_social': data.get('capital'),
            'code_naf': data.get('code_naf', ''),
            'libelle_code_naf': data.get('libelle_code_naf', ''),
            'dirigeant_principal': self._get_dirigeant(data),
            'statut': 'à contacter',
            'lien_pappers': f"https://www.pappers.fr/entreprise/{data.get('siren', '')}",
            'last_scraped_at': datetime.now().isoformat()
        }
    
    def _format_address(self, company: Dict) -> str:
        """Formate l'adresse complète"""
        parts = []
        if company.get('adresse_ligne_1'):
            parts.append(company['adresse_ligne_1'])
        if company.get('code_postal') and company.get('ville'):
            parts.append(f"{company['code_postal']} {company['ville']}")
        return ', '.join(parts)
    
    def _get_dirigeant(self, company: Dict) -> str:
        """Extrait le dirigeant principal"""
        if 'representants' in company and company['representants']:
            rep = company['representants'][0]
            nom = f"{rep.get('prenom', '')} {rep.get('nom', '')}".strip()
            qualite = rep.get('qualite', '')
            return f"{nom} ({qualite})" if qualite else nom
        return ''
    
    async def run_full_scraping(self, status_tracker):
        """Lance le scraping complet"""
        async with self:
            for code_naf in self.CODES_NAF:
                for dept in self.DEPARTEMENTS_IDF:
                    status_tracker.message = f"Scraping {code_naf} - Département {dept}"
                    logger.info(status_tracker.message)
                    
                    page = 1
                    has_more = True
                    
                    while has_more:
                        try:
                            # Recherche
                            response = await self.search_companies(
                                code_naf=code_naf,
                                departement=dept,
                                page=page,
                                entreprise_cessee=False,
                                chiffre_affaires_min=3000000
                            )
                            
                            if 'resultats' in response:
                                companies = response['resultats']
                                
                                # Traiter chaque entreprise
                                for company in companies:
                                    # Récupérer les détails
                                    siren = company.get('siren')
                                    if siren:
                                        details = await self.get_company_details(siren)
                                        if details:
                                            company.update(details)
                                    
                                    await self.process_company(company)
                                    
                                    # Mettre à jour le statut
                                    status_tracker.new_companies = self.new_companies_count
                                    status_tracker.skipped_companies = self.skipped_companies_count
                                    progress = (self.DEPARTEMENTS_IDF.index(dept) / len(self.DEPARTEMENTS_IDF)) * 100
                                    status_tracker.progress = int(progress)
                                
                                # Pagination
                                total = response.get('total', 0)
                                per_page = response.get('par_page', 100)
                                has_more = (page * per_page) < total
                                page += 1
                                
                                # Pause pour respecter les limites API
                                await asyncio.sleep(0.5)
                            else:
                                has_more = False
                                
                        except Exception as e:
                            logger.error(f"Erreur scraping: {e}")
                            if "quota" in str(e).lower():
                                status_tracker.error = "Quota API atteint"
                                return
                            has_more = False
            
            status_tracker.message = f"Terminé: {self.new_companies_count} nouvelles entreprises"
            status_tracker.progress = 100


# Standalone functions for API routes compatibility
async def search_companies(query: str = None, max_results: int = 100, **kwargs) -> List[Dict]:
    """Standalone function for searching companies via Pappers API"""
    from app.db.supabase import get_supabase_client
    
    try:
        db_client = get_supabase_client()
        
        async with PappersAPIClient(db_client) as client:
            search_params = {
                'q': query or 'cabinet comptable',
                'par_page': min(max_results, 100),
                'precision': 'standard',
                'entreprise_cessee': False
            }
            
            if 'department' in kwargs:
                search_params['departement'] = kwargs['department']
            if 'code_naf' in kwargs:
                search_params['code_naf'] = kwargs['code_naf']
            
            response = await client.search_companies(**search_params)
            
            if 'resultats' in response:
                companies = []
                for company_data in response['resultats'][:max_results]:
                    siren = company_data.get('siren')
                    if siren:
                        details = await client.get_company_details(siren)
                        if details:
                            company_data.update(details)
                    
                    formatted_company = client._format_company_data(company_data)
                    companies.append(formatted_company)
                
                return companies
            
            return []
            
    except Exception as e:
        logger.error(f"Error in search_companies: {e}")
        return []