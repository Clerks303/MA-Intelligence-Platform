import asyncio
import aiohttp
import logging
from typing import Dict, List, Optional
from datetime import datetime
import os

logger = logging.getLogger(__name__)


class InfogreffeAPIClient:
    """Client asynchrone pour l'API Infogreffe"""
    
    BASE_URL = "https://opendata-rncs.infogreffe.fr/api/v1"
    
    def __init__(self, db_client=None):
        self.db = db_client
        self.session = None
        self.api_key = os.environ.get('INFOGREFFE_API_KEY', '')
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def get_company_details(self, siren: str) -> Optional[Dict]:
        """Récupère les détails d'une entreprise depuis Infogreffe"""

        if not siren or len(siren) != 9:
            logger.warning(f"SIREN invalide: {siren}")
            return None
            
        endpoint = f"{self.BASE_URL}/entreprises"
        
        headers = {
            'Accept': 'application/json',
            'User-Agent': 'M&A-Intelligence-Platform/1.0'
        }
        
        params = {
            'siren': siren,
            'format': 'json'
        }
        
        # Ajouter la cl� API si disponible
        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'
        
        try:
            async with self.session.get(endpoint, params=params, headers=headers) as response:
                logger.info(f"Infogreffe API call for SIREN {siren}: {response.status}")
                
                if response.status == 200:
                    data = await response.json()
                    return self._format_infogreffe_data(data, siren)
                elif response.status == 404:
                    logger.info(f"Entreprise non trouv�e: {siren}")
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
            logger.error(f"Erreur r�seau Infogreffe pour SIREN {siren}: {e}")
            return None
        except Exception as e:
            logger.error(f"Erreur inattendue Infogreffe pour SIREN {siren}: {e}")
            return None
    
    def _format_infogreffe_data(self, data: Dict, siren: str) -> Dict:
        """Formate les donn�es Infogreffe pour notre base"""
        try:
            # Structure des donn�es Infogreffe peut varier
            if isinstance(data, list) and data:
                company_data = data[0]
            elif isinstance(data, dict):
                company_data = data
            else:
                logger.warning(f"Format inattendu des donn�es Infogreffe pour {siren}")
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
            
            # Traitement sp�cifique pour les dirigeants
            if 'dirigeants' in company_data:
                dirigeants_list = []
                for dirigeant in company_data['dirigeants'][:5]:  # Limite � 5
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
            
            # Donn�es brutes pour r�f�rence
            formatted['details_complets'] = company_data
            
            return formatted
            
        except Exception as e:
            logger.error(f"Erreur formatage donn�es Infogreffe pour {siren}: {e}")
            return {}


# Fonction standalone pour compatibilit� avec les routes
async def get_company_details(siren: str) -> Optional[Dict]:
    """Fonction standalone pour r�cup�rer les d�tails d'une entreprise"""
    try:
        async with InfogreffeAPIClient() as client:
            return await client.get_company_details(siren)
    except Exception as e:
        logger.error(f"Erreur get_company_details pour {siren}: {e}")
        return None


async def enrich_companies(siren_list: List[str], **kwargs) -> List[Dict]:
    """Enrichit une liste d'entreprises avec les donn�es Infogreffe"""
    enriched_companies = []
    
    try:
        async with InfogreffeAPIClient() as client:
            for siren in siren_list:
                if siren:
                    details = await client.get_company_details(siren)
                    if details:
                        enriched_companies.append(details)
                    
                    # Respecter les limites de taux
                    await asyncio.sleep(0.5)
        
        logger.info(f"Enrichissement Infogreffe termin�: {len(enriched_companies)} entreprises enrichies")
        return enriched_companies
        
    except Exception as e:
        logger.error(f"Erreur enrichissement Infogreffe: {e}")
        return enriched_companies