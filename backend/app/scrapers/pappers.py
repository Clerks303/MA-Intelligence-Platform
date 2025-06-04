"""
Scraper Pappers unifié avec cache intégré
Remplace pappers.py et cached_pappers.py
"""

import asyncio
import aiohttp
import logging
import os
from typing import Dict, List, Optional, Set
from datetime import datetime

from .base_scraper import RateLimitedScraper, normalize_siren, is_valid_siren

logger = logging.getLogger(__name__)

class PappersClient(RateLimitedScraper):
    """Client unifié pour l'API Pappers avec cache et rate limiting"""
    
    BASE_URL = "https://api.pappers.fr/v2"
    CODES_NAF = ['6920Z']  # Cabinets comptables
    DEPARTEMENTS_IDF = ['75', '77', '78', '91', '92', '93', '94', '95']
    
    def __init__(self, db_client=None):
        # Cache 24h pour données légales stables, rate limit 100 req/min
        super().__init__(db_client, cache_ttl=86400, rate_limit=100, rate_window=60)
        self.api_key = os.environ.get('PAPPERS_API_KEY', '')
        self.existing_sirens = set()
        
    async def __aenter__(self):
        await super().__aenter__()
        await self._load_existing_sirens()
        return self
    
    async def _load_existing_sirens(self):
        """Charge les SIRENs existants depuis la base"""
        if self.db:
            try:
                # Simuler le chargement des SIRENs existants
                # À remplacer par une vraie requête DB
                self.existing_sirens = set()
                logger.info(f"Loaded {len(self.existing_sirens)} existing SIRENs")
            except Exception as e:
                logger.error(f"Error loading existing SIRENs: {e}")
    
    async def _make_request(self, endpoint: str, params: Dict) -> Optional[Dict]:
        """Effectue une requête vers l'API Pappers"""
        if not self.api_key:
            logger.warning("No Pappers API key configured, using mock data")
            return self._get_mock_data(params.get('siren', ''))
        
        url = f"{self.BASE_URL}/{endpoint}"
        params['api_token'] = self.api_key
        
        try:
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return data
                else:
                    logger.error(f"Pappers API error: {response.status}")
                    return None
        except Exception as e:
            logger.error(f"Request error: {e}")
            return None
    
    def _get_mock_data(self, siren: str) -> Dict:
        """Données mock pour développement"""
        return {
            'siren': siren,
            'denomination': f'Cabinet Comptable {siren[-3:]}',
            'code_naf': '6920Z',
            'libelle_code_naf': 'Activités comptables',
            'adresse': f'123 rue de la Comptabilité, 75001 Paris',
            'date_creation': '2020-01-01',
            'dirigeants': [
                {
                    'nom': 'Martin',
                    'prenom': 'Jean',
                    'fonction': 'Président'
                }
            ],
            'capital': 10000,
            'chiffre_affaires': 500000,
            'effectif': 5,
            'derniere_mise_a_jour': datetime.now().isoformat()
        }
    
    async def get_company_details(self, siren: str) -> Optional[Dict]:
        """Récupère les détails d'une entreprise par SIREN"""
        siren = normalize_siren(siren)
        if not is_valid_siren(siren):
            logger.error(f"Invalid SIREN: {siren}")
            return None
        
        async def fetch_details():
            return await self._rate_limited_request(
                self._make_request, 
                'entreprise',
                {'siren': siren}
            )
        
        return await self._cached_request(siren, fetch_details)
    
    async def search_companies(self, **criteria) -> List[Dict]:
        """Recherche d'entreprises par critères"""
        params = {
            'code_naf': ','.join(self.CODES_NAF),
            'departement': ','.join(self.DEPARTEMENTS_IDF),
            'precision': 'standard',
            'par_page': criteria.get('limit', 100),
            **criteria
        }
        
        # Utiliser un hash des paramètres comme clé de cache
        from .base_scraper import hash_query
        cache_key = f"search_{hash_query(params)}"
        
        async def fetch_search():
            return await self._rate_limited_request(
                self._make_request,
                'recherche',
                params
            )
        
        result = await self._cached_request(cache_key, fetch_search)
        return result.get('resultats', []) if result else []
    
    async def enrich_company_data(self, basic_data: Dict) -> Dict:
        """Enrichit les données de base d'une entreprise"""
        siren = basic_data.get('siren')
        if not siren:
            return basic_data
        
        detailed_data = await self.get_company_details(siren)
        if detailed_data:
            # Merger les données
            enriched = {**basic_data, **detailed_data}
            enriched['source'] = 'pappers'
            enriched['last_updated'] = datetime.now().isoformat()
            return enriched
        
        return basic_data
    
    async def bulk_search(self, sirens: List[str]) -> List[Dict]:
        """Recherche en lot pour plusieurs SIRENs"""
        results = []
        
        # Traitement par lots pour éviter la surcharge
        batch_size = 10
        for i in range(0, len(sirens), batch_size):
            batch = sirens[i:i + batch_size]
            
            # Créer les tâches asynchrones pour ce lot
            tasks = [self.get_company_details(siren) for siren in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filtrer les résultats valides
            for result in batch_results:
                if isinstance(result, dict) and result:
                    results.append(result)
            
            # Pause entre les lots pour respecter le rate limiting
            if i + batch_size < len(sirens):
                await asyncio.sleep(1)
        
        return results
    
    async def get_company_financials(self, siren: str) -> Optional[Dict]:
        """Récupère les données financières d'une entreprise"""
        company_data = await self.get_company_details(siren)
        if not company_data:
            return None
        
        return {
            'siren': siren,
            'chiffre_affaires': company_data.get('chiffre_affaires'),
            'resultat': company_data.get('resultat'),
            'effectif': company_data.get('effectif'),
            'capital': company_data.get('capital'),
            'date_derniers_comptes': company_data.get('date_derniers_comptes'),
            'source': 'pappers_financials'
        }
    
    async def verify_company_exists(self, siren: str) -> bool:
        """Vérifie l'existence d'une entreprise"""
        details = await self.get_company_details(siren)
        return details is not None
    
    def is_cabinet_comptable(self, company_data: Dict) -> bool:
        """Vérifie si une entreprise est un cabinet comptable"""
        code_naf = company_data.get('code_naf', '')
        libelle = company_data.get('libelle_code_naf', '').lower()
        denomination = company_data.get('denomination', '').lower()
        
        return (
            code_naf in self.CODES_NAF or
            'comptab' in libelle or
            'expert' in libelle or
            'comptab' in denomination or
            'expert' in denomination
        )

# Export de la classe principale
__all__ = ['PappersClient']