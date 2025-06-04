import asyncio
import random
import re
import logging
from typing import Dict, List, Optional, Set
from datetime import datetime
from playwright.async_api import async_playwright
from urllib.parse import quote, urljoin

logger = logging.getLogger(__name__)

class SocieteScraper:
    """Scraper asynchrone pour Société.com avec Playwright"""
    
    BASE_URL = "https://www.societe.com"
    SEARCH_URL = "https://www.societe.com/cgi-bin/search"
    
    USER_AGENTS = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:122.0) Gecko/20100101 Firefox/122.0'
    ]
    
    def __init__(self, db_client):
        self.db = db_client
        self.browser = None
        self.context = None
        self.page = None
        self.existing_sirens = set()
        self.new_companies_count = 0
        self.skipped_companies_count = 0
        
    async def __aenter__(self):
        await self._setup_browser()
        await self._load_existing_sirens()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.browser:
            await self.browser.close()
    
    async def _load_existing_sirens(self):
        """Charge les SIREN existants"""
        try:
            response = self.db.table('cabinets_comptables').select('siren').execute()
            self.existing_sirens = set(str(company['siren']) for company in response.data)
            logger.info(f"Chargé {len(self.existing_sirens)} SIREN existants")
        except Exception as e:
            logger.error(f"Erreur chargement SIREN: {e}")
    
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
        """Délai aléatoire"""
        await asyncio.sleep(random.uniform(min_seconds, max_seconds))
    
    async def search_companies(self, department: str, page_num: int = 1) -> tuple[List[Dict], bool]:
        """Recherche les entreprises par département"""
        companies = []
        
        try:
            # Construction URL
            params = {
                'champs': department,
                'naf': '6920Z',
                'page': str(page_num)
            }
            query_string = '&'.join([f"{k}={quote(v)}" for k, v in params.items()])
            search_url = f"{self.SEARCH_URL}?{query_string}"
            
            logger.info(f"Recherche département {department}, page {page_num}")
            
            # Navigation
            await self.page.goto(search_url, wait_until='networkidle')
            await self._random_delay(1, 3)
            
            # Vérifier captcha
            if await self.page.locator('div.g-recaptcha').count() > 0:
                logger.warning("Captcha détecté")
                return companies, False
            
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
                                'nom_entreprise': await link.inner_text()
                            }
                            companies.append(company_info)
                            
                except Exception as e:
                    logger.error(f"Erreur extraction lien: {e}")
            
            # Page suivante ?
            has_next = await self.page.locator('a:has-text("Suivant")').count() > 0
            
            return companies, has_next
            
        except Exception as e:
            logger.error(f"Erreur recherche: {e}")
            return companies, False
    
    async def scrape_company_details(self, company_info: Dict) -> Optional[Dict]:
        """Récupère les détails d'une entreprise"""
        try:
            url = company_info['url']
            logger.info(f"Scraping {company_info['nom_entreprise']}")
            
            await self._random_delay(2, 5)
            await self.page.goto(url, wait_until='networkidle')
            
            # Vérifier captcha
            if await self.page.locator('div.g-recaptcha').count() > 0:
                return None
            
            # Extraction des données
            data = {
                'siren': company_info['siren'],
                'nom_entreprise': company_info['nom_entreprise'],
                'lien_societe_com': url,
                'statut': 'à contacter',
                'last_scraped_at': datetime.now().isoformat()
            }
            
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
                    data['capital_social'] = int(match.group(1))
            
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
            
            # Vérifier CA
            ca = data.get('chiffre_affaires', 0)
            if ca and (ca < 3000000 or ca > 50000000):
                return None
            
            # Sauvegarder
            try:
                clean_data = self._clean_data_for_db(data)
                self.db.table('cabinets_comptables').insert(clean_data).execute()
                self.new_companies_count += 1
                return clean_data
            except Exception as e:
                logger.error(f"Erreur sauvegarde: {e}")
                return None
                
        except Exception as e:
            logger.error(f"Erreur scraping détails: {e}")
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
                    data['chiffre_affaires'] = int(match.group(1))
                    break
            except:
                continue
        
        # Résultat
        res_elements = await self.page.locator('text=/Résultat net/').all()
        for elem in res_elements:
            try:
                parent = await elem.locator('..').inner_text()
                match = re.search(r'(-?[\d\s]+)(?:\s*€|EUR)', parent.replace(' ', ''))
                if match:
                    data['resultat'] = int(match.group(1))
                    break
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
    
    async def _safe_get_text_from_element(self, parent, selector: str) -> Optional[str]:
        """Récupère le texte depuis un élément parent"""
        try:
            element = parent.locator(selector).first
            if await element.count() > 0:
                return await element.inner_text()
        except:
            pass
        return None
    
    def _clean_data_for_db(self, data: Dict) -> Dict:
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
    
    async def run_full_scraping(self, status_tracker):
        """Lance le scraping complet"""
        departments = ['75', '77', '78', '91', '92', '93', '94', '95']
        
        async with self:
            for i, dept in enumerate(departments):
                status_tracker.message = f"Scraping Société.com - Département {dept}"
                logger.info(status_tracker.message)
                
                page_num = 1
                has_next = True
                
                while has_next and page_num <= 5:  # Limite pages
                    companies, has_next = await self.search_companies(dept, page_num)
                    
                    for company in companies:
                        await self.scrape_company_details(company)
                        
                        # Mettre à jour statut
                        status_tracker.new_companies = self.new_companies_count
                        status_tracker.skipped_companies = self.skipped_companies_count
                        
                        # Pause anti-détection
                        if self.new_companies_count % 10 == 0:
                            await self._random_delay(30, 60)
                    
                    page_num += 1
                    status_tracker.progress = int((i + 1) / len(departments) * 100)
                
            status_tracker.message = f"Terminé: {self.new_companies_count} nouvelles entreprises"
            status_tracker.progress = 100


# Standalone functions for API routes compatibility
async def scrape_companies(query: str = None, max_pages: int = 5, **kwargs) -> List[Dict]:
    """Standalone function for scraping companies from Société.com"""
    from app.db.supabase import get_supabase_client
    
    try:
        db_client = get_supabase_client()
        
        async with SocieteScraper(db_client) as scraper:
            all_companies = []
            departments = kwargs.get('departments', ['75', '77', '78', '91', '92', '93', '94', '95'])
            
            for dept in departments:
                logger.info(f"Scraping Société.com - Département {dept}")
                
                page_num = 1
                has_next = True
                
                while has_next and page_num <= max_pages:
                    companies, has_next = await scraper.search_companies(dept, page_num)
                    
                    for company_info in companies:
                        detailed_company = await scraper.scrape_company_details(company_info)
                        if detailed_company:
                            all_companies.append(detailed_company)
                    
                    page_num += 1
                    
                    # Anti-detection delay
                    await scraper._random_delay(2, 5)
            
            logger.info(f"Société.com scraping completed: {len(all_companies)} companies")
            return all_companies
            
    except Exception as e:
        logger.error(f"Error in scrape_companies: {e}")
        return []