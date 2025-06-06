"""
Scraper Société.com unifié avec cache intégré
Remplace societe.py avec cache et rate limiting de base_scraper
"""

import asyncio
import random
import re
import logging
from typing import Dict, List, Optional, Set
from datetime import datetime
from playwright.async_api import async_playwright
from urllib.parse import quote, urljoin

from .base_scraper import RateLimitedScraper, normalize_siren, is_valid_siren

logger = logging.getLogger(__name__)

class SocieteClient(RateLimitedScraper):
    """Client unifié pour Société.com avec cache et rate limiting"""
    
    BASE_URL = "https://www.societe.com"
    SEARCH_URL = "https://www.societe.com/cgi-bin/search"
    
    USER_AGENTS = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:122.0) Gecko/20100101 Firefox/122.0'
    ]
    
    def __init__(self, db_client=None):
        # Cache 12h pour données extraites, rate limit 30 req/min (plus strict que API)
        super().__init__(db_client, cache_ttl=43200, rate_limit=30, rate_window=60)
        self.browser = None
        self.context = None
        self.page = None
        self.existing_sirens = set()
        self.new_companies_count = 0
        self.skipped_companies_count = 0
    
    async def __aenter__(self):
        await super().__aenter__()
        await self._setup_browser()
        await self._load_existing_sirens()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.browser:
            await self.browser.close()
        await super().__aexit__(exc_type, exc_val, exc_tb)
    
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
    
    async def _setup_browser(self):
        """Configure le navigateur avec anti-détection"""
        playwright = await async_playwright().start()
        
        browser_args = [
            '--disable-blink-features=AutomationControlled',
            '--disable-features=IsolateOrigins,site-per-process',
            '--no-sandbox',
            '--disable-dev-shm-usage',
            '--window-size=1920,1080'
        ]
        
        self.browser = await playwright.chromium.launch(
            headless=True,
            args=browser_args
        )
        
        # Contexte avec fingerprint aléatoire
        user_agent = random.choice(self.USER_AGENTS)
        self.context = await self.browser.new_context(
            user_agent=user_agent,
            viewport={'width': 1920, 'height': 1080},
            locale='fr-FR',
            timezone_id='Europe/Paris'
        )
        
        # Scripts anti-détection
        await self.context.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
            window.chrome = { runtime: {} };
            Object.defineProperty(navigator, 'plugins', { get: () => [1, 2, 3, 4, 5] });
            Object.defineProperty(navigator, 'languages', { get: () => ['fr-FR', 'fr', 'en'] });
        """)
        
        self.page = await self.context.new_page()
    
    async def _random_delay(self, min_seconds: float = 0.5, max_seconds: float = 2.0):
        """Délai aléatoire pour anti-détection"""
        await asyncio.sleep(random.uniform(min_seconds, max_seconds))
    
    async def search_companies(self, department: str = None, page_num: int = 1, **criteria) -> List[Dict]:
        """Recherche d'entreprises par critères"""
        
        # Créer une clé de cache basée sur les paramètres
        from .base_scraper import hash_query
        search_params = {
            'department': department,
            'page': page_num,
            'naf': '6920Z',
            **criteria
        }
        cache_key = f"search_{hash_query(search_params)}"
        
        async def fetch_search():
            return await self._rate_limited_request(
                self._scrape_search_page,
                department, page_num, **criteria
            )
        
        result = await self._cached_request(cache_key, fetch_search)
        return result if result else []
    
    async def _scrape_search_page(self, department: str, page_num: int, **criteria) -> List[Dict]:
        """Effectue le scraping d'une page de recherche"""
        companies = []
        
        try:
            # Construction URL
            params = {
                'champs': department or '75',
                'naf': '6920Z',
                'page': str(page_num)
            }
            query_string = '&'.join([f"{k}={quote(v)}" for k, v in params.items()])
            search_url = f"{self.SEARCH_URL}?{query_string}"
            
            logger.info(f"Société.com: recherche département {department}, page {page_num}")
            
            # Navigation
            await self.page.goto(search_url, wait_until='networkidle')
            await self._random_delay(1, 3)
            
            # Vérifier captcha
            if await self.page.locator('div.g-recaptcha').count() > 0:
                logger.warning("Captcha détecté")
                return companies
            
            # Extraction des liens
            await self.page.wait_for_selector('div#result-list', timeout=10000)
            company_links = await self.page.locator('div#result-list a.txt-no-wrap').all()
            
            for link in company_links:
                try:
                    href = await link.get_attribute('href')
                    if href and '/societe/' in href:
                        # Extraire SIREN
                        siren_match = re.search(r'/societe/[^/]+/(\d{9})', href)
                        if siren_match:
                            siren = siren_match.group(1)
                            
                            if siren in self.existing_sirens:
                                self.skipped_companies_count += 1
                                continue
                            
                            company_info = {
                                'siren': siren,
                                'url': urljoin(self.BASE_URL, href),
                                'nom_entreprise': await link.inner_text(),
                                'source': 'societe_com'
                            }
                            companies.append(company_info)
                            
                except Exception as e:
                    logger.error(f"Erreur extraction lien: {e}")
            
            return companies
            
        except Exception as e:
            logger.error(f"Erreur recherche Société.com: {e}")
            return companies
    
    async def get_company_details(self, siren: str) -> Optional[Dict]:
        """Récupère les détails d'une entreprise par SIREN ou URL"""
        siren = normalize_siren(siren)
        if not is_valid_siren(siren):
            logger.error(f"Invalid SIREN: {siren}")
            return None
        
        async def fetch_details():
            return await self._rate_limited_request(
                self._scrape_company_details,
                siren
            )
        
        return await self._cached_request(siren, fetch_details)
    
    async def _scrape_company_details(self, siren: str) -> Optional[Dict]:
        """Scrape les détails d'une entreprise depuis Société.com"""
        try:
            # Construire l'URL de recherche pour le SIREN
            search_url = f"{self.SEARCH_URL}?champs={siren}"
            
            logger.info(f"Société.com: scraping détails pour SIREN {siren}")
            
            await self._random_delay(2, 5)
            await self.page.goto(search_url, wait_until='networkidle')
            
            # Vérifier captcha
            if await self.page.locator('div.g-recaptcha').count() > 0:
                logger.warning("Captcha détecté")
                return None
            
            # Trouver le lien vers la fiche entreprise
            company_link = await self.page.locator(f'a[href*="/societe/"][href*="{siren}"]').first
            if await company_link.count() == 0:
                logger.warning(f"Entreprise non trouvée: {siren}")
                return None
            
            # Aller sur la page de détails
            await company_link.click()
            await self.page.wait_for_load_state('networkidle')
            
            # Extraction des données
            data = {
                'siren': siren,
                'lien_societe_com': self.page.url,
                'statut': 'à contacter',
                'source': 'societe_com',
                'last_scraped_at': datetime.now().isoformat()
            }
            
            # Nom de l'entreprise
            nom_element = await self._safe_get_text('h1.company-title')
            if nom_element:
                data['nom_entreprise'] = nom_element
            
            # Sélecteurs pour les données
            selectors = {
                'forme_juridique': 'td:has-text("Forme juridique") + td',
                'siret_siege': 'td:has-text("SIRET (siège)") + td',
                'numero_tva': 'td:has-text("TVA") + td',
                'code_naf': 'td:has-text("Activité") + td span.NAF',
                'libelle_code_naf': 'td:has-text("Activité") + td'
            }
            
            for field, selector in selectors.items():
                data[field] = await self._safe_get_text(selector)
            
            # Capital social
            capital_text = await self._safe_get_text('td:has-text("Capital social") + td')
            if capital_text:
                match = re.search(r'([\d\s]+)', capital_text.replace(' ', ''))
                if match:
                    try:
                        data['capital_social'] = int(match.group(1))
                    except ValueError:
                        pass
            
            # Date création
            date_text = await self._safe_get_text('td:has-text("Date création entreprise") + td')
            if date_text:
                match = re.search(r'(\d{2})-(\d{2})-(\d{4})', date_text)
                if match:
                    data['date_creation'] = f"{match.group(3)}-{match.group(2)}-{match.group(1)}"
            
            # CA et résultat
            await self._extract_financial_data(data)
            
            # Dirigeants
            await self._extract_dirigeants(data)
            
            # Adresse
            await self._extract_address(data)
            
            return data
                
        except Exception as e:
            logger.error(f"Erreur scraping détails SIREN {siren}: {e}")
            return None
    
    async def _safe_get_text(self, selector: str) -> Optional[str]:
        """Récupère le texte de manière sécurisée"""
        try:
            element = self.page.locator(selector).first
            if await element.count() > 0:
                text = await element.inner_text()
                return text.strip() if text else None
        except:
            pass
        return None
    
    async def _extract_financial_data(self, data: Dict):
        """Extrait les données financières"""
        # CA
        ca_elements = await self.page.locator('text=/Chiffre d\'affaires/').all()
        for elem in ca_elements:
            try:
                parent = await elem.locator('..').inner_text()
                match = re.search(r'([\d\s]+)(?:\s*€|EUR)', parent.replace(' ', ''))
                if match:
                    try:
                        data['chiffre_affaires'] = int(match.group(1))
                        break
                    except ValueError:
                        pass
            except:
                continue
        
        # Résultat
        res_elements = await self.page.locator('text=/Résultat net/').all()
        for elem in res_elements:
            try:
                parent = await elem.locator('..').inner_text()
                match = re.search(r'(-?[\d\s]+)(?:\s*€|EUR)', parent.replace(' ', ''))
                if match:
                    try:
                        data['resultat'] = int(match.group(1))
                        break
                    except ValueError:
                        pass
            except:
                continue
    
    async def _extract_dirigeants(self, data: Dict):
        """Extrait les dirigeants"""
        dirigeants = []
        dirigeant_elements = await self.page.locator('div.dirigeant').all()
        
        for elem in dirigeant_elements[:5]:  # Max 5 dirigeants
            try:
                nom = await self._safe_get_text_from_element(elem, 'a.nom')
                fonction = await self._safe_get_text_from_element(elem, 'span.fonction')
                
                if nom:
                    dirigeants.append({
                        'nom_complet': nom,
                        'qualite': fonction or 'Dirigeant'
                    })
            except:
                continue
        
        if dirigeants:
            data['dirigeants_json'] = dirigeants
            data['dirigeant_principal'] = f"{dirigeants[0]['nom_complet']} ({dirigeants[0]['qualite']})"
    
    async def _extract_address(self, data: Dict):
        """Extrait l'adresse complète"""
        try:
            adresse_parts = []
            
            # Adresse ligne 1
            adresse1 = await self._safe_get_text('td:has-text("Adresse") + td')
            if adresse1:
                adresse_parts.append(adresse1)
            
            # Code postal et ville
            cp_ville = await self._safe_get_text('td:has-text("Code postal") + td, td:has-text("Ville") + td')
            if cp_ville:
                adresse_parts.append(cp_ville)
            
            if adresse_parts:
                data['adresse'] = ', '.join(adresse_parts)
                
        except Exception as e:
            logger.debug(f"Erreur extraction adresse: {e}")
    
    async def _safe_get_text_from_element(self, parent, selector: str) -> Optional[str]:
        """Récupère le texte depuis un élément parent"""
        try:
            element = parent.locator(selector).first
            if await element.count() > 0:
                return await element.inner_text()
        except:
            pass
        return None
    
    async def enrich_company_data(self, basic_data: Dict) -> Dict:
        """Enrichit les données de base d'une entreprise"""
        siren = basic_data.get('siren')
        if not siren:
            return basic_data
        
        detailed_data = await self.get_company_details(siren)
        if detailed_data:
            # Merger les données
            enriched = {**basic_data, **detailed_data}
            enriched['source'] = 'societe_com'
            enriched['last_updated'] = datetime.now().isoformat()
            return enriched
        
        return basic_data
    
    async def bulk_search(self, departments: List[str], max_pages: int = 5) -> List[Dict]:
        """Recherche en lot pour plusieurs départements"""
        results = []
        
        for dept in departments:
            logger.info(f"Société.com: scraping département {dept}")
            
            page_num = 1
            has_next = True
            
            while has_next and page_num <= max_pages:
                try:
                    companies = await self.search_companies(dept, page_num)
                    
                    if not companies:
                        has_next = False
                        break
                    
                    # Enrichir chaque entreprise
                    for company_info in companies:
                        try:
                            detailed_company = await self.get_company_details(company_info['siren'])
                            if detailed_company:
                                # Fusionner les données de recherche et de détail
                                enriched = {**company_info, **detailed_company}
                                results.append(enriched)
                                
                        except Exception as e:
                            logger.error(f"Erreur enrichissement {company_info.get('siren')}: {e}")
                    
                    page_num += 1
                    
                    # Délai anti-détection
                    await self._random_delay(3, 8)
                    
                except Exception as e:
                    logger.error(f"Erreur page {page_num} département {dept}: {e}")
                    has_next = False
        
        return results
    
    def clean_data_for_db(self, data: Dict) -> Dict:
        """Nettoie les données pour la base"""
        clean_data = {}
        numeric_fields = ['chiffre_affaires', 'resultat', 'capital_social', 'effectif']
        
        for key, value in data.items():
            if key in numeric_fields:
                clean_data[key] = value if isinstance(value, (int, float)) else None
            elif key == 'dirigeants_json' and isinstance(value, list):
                clean_data[key] = value
            elif value is None or value == '':
                clean_data[key] = None
            else:
                clean_data[key] = str(value)
        
        return clean_data

# Export de la classe principale
__all__ = ['SocieteClient']