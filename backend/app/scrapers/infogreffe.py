"""
Scraper Infogreffe unifié avec cache intégré
Remplace infogreffe.py avec cache et rate limiting de base_scraper
"""

import asyncio
import aiohttp
import logging
from typing import Dict, List, Optional
from datetime import datetime
import os

from .base_scraper import RateLimitedScraper, normalize_siren, is_valid_siren

logger = logging.getLogger(__name__)

class InfogreffeClient(RateLimitedScraper):
    """Client unifié pour l'API Infogreffe avec cache et rate limiting"""
    
    BASE_URL = "https://opendata-rncs.infogreffe.fr/api/v1"
    
    def __init__(self, db_client=None):
        # Cache 48h pour données officielles stables, rate limit 60 req/min
        super().__init__(db_client, cache_ttl=172800, rate_limit=60, rate_window=60)
        self.api_key = os.environ.get('INFOGREFFE_API_KEY', '')
    
    async def __aenter__(self):
        await super().__aenter__()
        return self
    
    async def search_companies(self, **criteria) -> List[Dict]:
        """Recherche d'entreprises par critères"""
        # Infogreffe ne supporte pas la recherche générale, 
        # on retourne une liste vide pour la compatibilité
        logger.info("Infogreffe: search_companies non supporté, utiliser get_company_details")
        return []
    
    async def get_company_details(self, siren: str) -> Optional[Dict]:
        """Récupère les détails d'une entreprise par SIREN"""
        siren = normalize_siren(siren)
        if not is_valid_siren(siren):
            logger.error(f"Invalid SIREN: {siren}")
            return None
        
        async def fetch_details():
            return await self._rate_limited_request(
                self._api_get_company_details,
                siren
            )
        
        return await self._cached_request(siren, fetch_details)
    
    async def _api_get_company_details(self, siren: str) -> Optional[Dict]:
        """Effectue l'appel API vers Infogreffe"""
        if not siren or len(siren) != 9:
            logger.warning(f"SIREN invalide: {siren}")
            return None
            
        endpoint = f"{self.BASE_URL}/entreprises"
        
        headers = {
            'Accept': 'application/json',
            'User-Agent': 'M&A-Intelligence-Platform/2.0'
        }
        
        params = {
            'siren': siren,
            'format': 'json'
        }
        
        # Ajouter la clé API si disponible
        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'
        else:
            logger.warning("Pas de clé API Infogreffe configurée, utilisation des données mock")
            return self._get_mock_data(siren)
        
        try:
            async with self.session.get(endpoint, params=params, headers=headers) as response:
                logger.info(f"Infogreffe API call for SIREN {siren}: {response.status}")
                
                if response.status == 200:
                    data = await response.json()
                    return self._format_infogreffe_data(data, siren)
                elif response.status == 404:
                    logger.info(f"Entreprise non trouvée: {siren}")
                    return None
                elif response.status == 400:
                    error_text = await response.text()
                    logger.error(f"Erreur 400 pour SIREN {siren}: {error_text}")
                    return None
                elif response.status == 401:
                    logger.error("Erreur d'authentification Infogreffe API")
                    return None
                elif response.status == 429:
                    logger.warning("Rate limit atteint pour Infogreffe API")
                    await asyncio.sleep(60)  # Attendre 1 minute
                    return None
                else:
                    error_text = await response.text()
                    logger.error(f"Erreur Infogreffe API {response.status}: {error_text}")
                    return None
                    
        except aiohttp.ClientError as e:
            logger.error(f"Erreur réseau Infogreffe pour SIREN {siren}: {e}")
            return None
        except Exception as e:
            logger.error(f"Erreur inattendue Infogreffe pour SIREN {siren}: {e}")
            return None
    
    def _get_mock_data(self, siren: str) -> Dict:
        """Données mock pour développement sans clé API"""
        return {
            'siren': siren,
            'denomination': f'Société Mock {siren[-3:]}',
            'formeJuridique': 'SARL',
            'dateCreation': '2020-01-01',
            'capital': 10000,
            'adresse': '123 rue du Registre',
            'codePostal': '75001',
            'ville': 'Paris',
            'activitePrincipale': 'Activités comptables',
            'codeActivitePrincipale': '6920Z',
            'dirigeants': [
                {
                    'denomination': 'DUPONT Jean',
                    'qualite': 'Gérant',
                    'datePriseFonction': '2020-01-01'
                }
            ],
            '_mock': True,
            'last_enriched_at': datetime.now().isoformat(),
            'source': 'infogreffe_mock'
        }
    
    def _format_infogreffe_data(self, data: Dict, siren: str) -> Dict:
        """Formate les données Infogreffe pour notre base"""
        try:
            # Structure des données Infogreffe peut varier
            if isinstance(data, list) and data:
                company_data = data[0]
            elif isinstance(data, dict):
                company_data = data
            else:
                logger.warning(f"Format inattendu des données Infogreffe pour {siren}")
                return {}
            
            formatted = {
                'siren': siren,
                'last_enriched_at': datetime.now().isoformat(),
                'source': 'infogreffe'
            }
            
            # Mapping des champs Infogreffe
            field_mapping = {
                'denomination': 'nom_entreprise',
                'formeJuridique': 'forme_juridique',
                'dateCreation': 'date_creation',
                'capital': 'capital_social',
                'adresse': 'adresse',
                'codePostal': 'code_postal',
                'ville': 'ville',
                'activitePrincipale': 'libelle_code_naf',
                'codeActivitePrincipale': 'code_naf'
            }
            
            for infogreffe_field, our_field in field_mapping.items():
                if infogreffe_field in company_data:
                    value = company_data[infogreffe_field]
                    if value:
                        formatted[our_field] = value
            
            # Formater l'adresse complète
            if company_data.get('adresse') and company_data.get('codePostal') and company_data.get('ville'):
                adresse_parts = [
                    company_data['adresse'],
                    f"{company_data['codePostal']} {company_data['ville']}"
                ]
                formatted['adresse'] = ', '.join(adresse_parts)
            
            # Traitement spécifique pour les dirigeants
            if 'dirigeants' in company_data:
                dirigeants_list = []
                for dirigeant in company_data['dirigeants'][:5]:  # Limite à 5
                    if isinstance(dirigeant, dict):
                        dirigeant_info = {
                            'nom_complet': dirigeant.get('denomination', ''),
                            'qualite': dirigeant.get('qualite', ''),
                            'date_prise_fonction': dirigeant.get('datePriseFonction', '')
                        }
                        dirigeants_list.append(dirigeant_info)
                
                if dirigeants_list:
                    formatted['dirigeants_json'] = dirigeants_list
                    formatted['dirigeant_principal'] = f"{dirigeants_list[0]['nom_complet']} ({dirigeants_list[0]['qualite']})"
            
            # Données brutes pour référence
            formatted['details_complets'] = company_data
            
            return formatted
            
        except Exception as e:
            logger.error(f"Erreur formatage données Infogreffe pour {siren}: {e}")
            return {}
    
    async def enrich_company_data(self, basic_data: Dict) -> Dict:
        """Enrichit les données de base d'une entreprise"""
        siren = basic_data.get('siren')
        if not siren:
            return basic_data
        
        detailed_data = await self.get_company_details(siren)
        if detailed_data:
            # Merger les données
            enriched = {**basic_data, **detailed_data}
            enriched['source'] = 'infogreffe'
            enriched['last_updated'] = datetime.now().isoformat()
            return enriched
        
        return basic_data
    
    async def bulk_enrich(self, sirens: List[str]) -> List[Dict]:
        """Enrichissement en lot pour plusieurs SIRENs"""
        results = []
        
        # Traitement par lots pour éviter la surcharge
        batch_size = 5
        for i in range(0, len(sirens), batch_size):
            batch = sirens[i:i + batch_size]
            
            # Créer les tâches asynchrones pour ce lot
            tasks = [self.get_company_details(siren) for siren in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filtrer les résultats valides
            for j, result in enumerate(batch_results):
                if isinstance(result, dict) and result:
                    result['siren'] = batch[j]  # S'assurer que le SIREN est présent
                    results.append(result)
                elif isinstance(result, Exception):
                    logger.error(f"Erreur pour SIREN {batch[j]}: {result}")
            
            # Pause entre les lots pour respecter le rate limiting
            if i + batch_size < len(sirens):
                await asyncio.sleep(2)
        
        return results
    
    async def verify_company_exists(self, siren: str) -> bool:
        """Vérifie l'existence d'une entreprise"""
        details = await self.get_company_details(siren)
        return details is not None and bool(details.get('siren'))
    
    async def get_company_legal_status(self, siren: str) -> Optional[Dict]:
        """Récupère le statut légal d'une entreprise"""
        company_data = await self.get_company_details(siren)
        if not company_data:
            return None
        
        return {
            'siren': siren,
            'forme_juridique': company_data.get('forme_juridique'),
            'date_creation': company_data.get('date_creation'),
            'capital_social': company_data.get('capital_social'),
            'statut_legal': 'active',  # Approximation, Infogreffe ne donne pas toujours le statut exact
            'source': 'infogreffe'
        }

# Export de la classe principale
__all__ = ['InfogreffeClient']