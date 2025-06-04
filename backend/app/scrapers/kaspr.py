"""
Scraper Kaspr unifié avec cache intégré
Remplace kaspr.py avec cache et rate limiting de base_scraper
"""

import asyncio
import random
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import os
from dataclasses import dataclass
from enum import Enum

from .base_scraper import RateLimitedScraper, normalize_siren, is_valid_siren

logger = logging.getLogger(__name__)

class KasprContactType(Enum):
    """Types de contacts Kaspr"""
    CEO = "CEO"
    CFO = "CFO"
    CTO = "CTO"
    FOUNDER = "Founder"
    PRESIDENT = "President"
    DIRECTOR = "Director"
    MANAGER = "Manager"
    VP = "VP"
    OWNER = "Owner"

@dataclass
class KasprContact:
    """Structure d'un contact Kaspr"""
    first_name: str = ""
    last_name: str = ""
    full_name: str = ""
    job_title: str = ""
    seniority_level: str = ""
    department: str = ""
    
    # Coordonnées
    email: Optional[str] = None
    phone: Optional[str] = None
    mobile_phone: Optional[str] = None
    linkedin_url: Optional[str] = None
    
    # Métadonnées
    confidence_score: float = 0.0
    last_verified: Optional[str] = None
    source: str = "kaspr"
    
    # Données complémentaires
    company_name: str = ""
    company_domain: str = ""
    location: str = ""
    experience_years: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire pour JSON"""
        return {
            'first_name': self.first_name,
            'last_name': self.last_name,
            'full_name': self.full_name or f"{self.first_name} {self.last_name}".strip(),
            'job_title': self.job_title,
            'seniority_level': self.seniority_level,
            'department': self.department,
            'email': self.email,
            'phone': self.phone,
            'mobile_phone': self.mobile_phone,
            'linkedin_url': self.linkedin_url,
            'confidence_score': self.confidence_score,
            'last_verified': self.last_verified,
            'source': self.source,
            'company_name': self.company_name,
            'company_domain': self.company_domain,
            'location': self.location,
            'experience_years': self.experience_years
        }

class KasprClient(RateLimitedScraper):
    """Client unifié pour l'API Kaspr avec cache et rate limiting"""
    
    BASE_URL = "https://api.kaspr.io/api/v1"
    SEARCH_ENDPOINT = "/search/people"
    COMPANY_ENDPOINT = "/search/companies"
    CONTACT_ENDPOINT = "/contacts"
    VERIFY_ENDPOINT = "/verify/email"
    
    def __init__(self, db_client=None, use_mock: bool = None):
        # Cache 6h pour contacts (données moins stables), rate limit 50 req/min
        super().__init__(db_client, cache_ttl=21600, rate_limit=50, rate_window=60)
        self.api_key = os.environ.get('KASPR_API_KEY', '')
        
        # Détermination auto du mode mock
        if use_mock is None:
            self.use_mock = not bool(self.api_key)
        else:
            self.use_mock = use_mock
        
        self.max_contacts_per_company = 5
        self.contact_stats = {
            'companies_processed': 0,
            'contacts_found': 0,
            'contacts_with_email': 0,
            'contacts_with_phone': 0,
            'api_calls': 0,
            'errors': 0
        }
        
        if self.use_mock:
            logger.info("Mode MOCK Kaspr activé (clé API manquante)")
        else:
            logger.info("Mode API Kaspr réel activé")
    
    async def __aenter__(self):
        await super().__aenter__()
        if not self.use_mock and self.session:
            # Ajouter les headers d'authentification
            self.session._default_headers.update({
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json',
                'User-Agent': 'MA-Intelligence-Platform/2.0'
            })
        return self
    
    async def search_companies(self, **criteria) -> List[Dict]:
        """Recherche d'entreprises par critères"""
        # Kaspr se concentre sur les contacts, pas la recherche d'entreprises
        logger.info("Kaspr: search_companies limité, utiliser get_company_contacts")
        return []
    
    async def get_company_details(self, siren: str) -> Optional[Dict]:
        """Récupère les contacts dirigeants d'une entreprise par SIREN"""
        siren = normalize_siren(siren)
        if not is_valid_siren(siren):
            logger.error(f"Invalid SIREN: {siren}")
            return None
        
        # Pour Kaspr, on retourne les contacts formatés
        contacts = await self.get_company_contacts(siren, f"Entreprise {siren}")
        
        if contacts:
            return {
                'siren': siren,
                'kaspr_contacts': [contact.to_dict() for contact in contacts],
                'kaspr_enriched_at': datetime.now().isoformat(),
                'source': 'kaspr',
                'contacts_count': len(contacts)
            }
        
        return None
    
    async def get_company_contacts(
        self, 
        siren: str, 
        company_name: str,
        domain: Optional[str] = None,
        max_contacts: Optional[int] = None
    ) -> List[KasprContact]:
        """Récupère les contacts dirigeants d'une entreprise"""
        
        if self.use_mock:
            return await self._get_mock_contacts(siren, company_name)
        
        max_contacts = max_contacts or self.max_contacts_per_company
        
        # Utiliser le cache pour éviter les appels répétés
        cache_key = f"contacts_{siren}_{hash(company_name)}"
        
        async def fetch_contacts():
            return await self._rate_limited_request(
                self._api_get_company_contacts,
                siren, company_name, domain, max_contacts
            )
        
        contacts_data = await self._cached_request(cache_key, fetch_contacts)
        
        # Convertir en objets KasprContact
        if contacts_data:
            return [self._dict_to_contact(contact_dict) for contact_dict in contacts_data]
        
        return []
    
    async def _api_get_company_contacts(
        self, 
        siren: str, 
        company_name: str,
        domain: Optional[str] = None,
        max_contacts: int = 5
    ) -> List[Dict]:
        """Appel API réel pour récupérer les contacts"""
        
        logger.info(f"Kaspr API: recherche contacts pour {company_name} ({siren})")
        
        try:
            # 1. Recherche de l'entreprise
            company_data = await self._search_company(company_name, domain)
            if not company_data:
                logger.warning(f"Entreprise non trouvée dans Kaspr: {company_name}")
                return []
            
            # 2. Recherche des contacts dirigeants
            contacts = await self._search_company_people(
                company_data, 
                max_contacts=max_contacts
            )
            
            # 3. Enrichissement des contacts trouvés
            enriched_contacts = []
            for contact_data in contacts:
                try:
                    enriched_contact = await self._enrich_contact(contact_data)
                    if enriched_contact:
                        enriched_contacts.append(enriched_contact.to_dict())
                        
                        # Respect du rate limiting
                        await asyncio.sleep(1.2)  # 50 req/min
                        
                except Exception as e:
                    logger.error(f"Erreur enrichissement contact: {e}")
                    self.contact_stats['errors'] += 1
            
            # 4. Mise à jour des statistiques
            self.contact_stats['companies_processed'] += 1
            self.contact_stats['contacts_found'] += len(enriched_contacts)
            self.contact_stats['contacts_with_email'] += sum(1 for c in enriched_contacts if c.get('email'))
            self.contact_stats['contacts_with_phone'] += sum(1 for c in enriched_contacts if c.get('phone') or c.get('mobile_phone'))
            
            logger.info(f"Kaspr: {len(enriched_contacts)} contacts trouvés pour {company_name}")
            return enriched_contacts
            
        except Exception as e:
            logger.error(f"Erreur Kaspr API pour {company_name}: {e}")
            self.contact_stats['errors'] += 1
            return []
    
    async def _search_company(self, company_name: str, domain: Optional[str] = None) -> Optional[Dict]:
        """Recherche une entreprise dans Kaspr"""
        
        search_params = {
            'name': company_name,
            'country': 'FR',
            'limit': 1
        }
        
        if domain:
            search_params['domain'] = domain
        
        try:
            url = f"{self.BASE_URL}{self.COMPANY_ENDPOINT}"
            self.contact_stats['api_calls'] += 1
            
            async with self.session.get(url, params=search_params) as response:
                if response.status == 200:
                    data = await response.json()
                    companies = data.get('data', [])
                    
                    if companies:
                        return companies[0]
                        
                elif response.status == 404:
                    logger.info(f"Entreprise non trouvée dans Kaspr: {company_name}")
                    return None
                    
                elif response.status == 429:
                    logger.warning("Rate limit Kaspr atteint")
                    await asyncio.sleep(60)
                    return None
                    
                else:
                    error_text = await response.text()
                    logger.error(f"Erreur API Kaspr {response.status}: {error_text}")
                    return None
                    
        except Exception as e:
            logger.error(f"Erreur recherche entreprise Kaspr: {e}")
            return None
    
    async def _search_company_people(self, company_data: Dict, max_contacts: int = 5) -> List[Dict]:
        """Recherche les dirigeants d'une entreprise"""
        
        company_id = company_data.get('id')
        if not company_id:
            return []
        
        search_params = {
            'company_id': company_id,
            'seniority_levels': ['director', 'vp', 'c_level', 'owner'],
            'limit': max_contacts * 2,
            'verified_emails': True
        }
        
        try:
            url = f"{self.BASE_URL}{self.SEARCH_ENDPOINT}"
            self.contact_stats['api_calls'] += 1
            
            async with self.session.get(url, params=search_params) as response:
                if response.status == 200:
                    data = await response.json()
                    people = data.get('data', [])
                    
                    # Filtrer et prioriser les dirigeants
                    filtered_people = []
                    for person in people:
                        job_title = person.get('job_title', '').lower()
                        
                        # Vérifier si c'est un poste de direction
                        if self._is_executive_title(job_title):
                            priority_score = self._calculate_contact_priority(person)
                            person['_priority_score'] = priority_score
                            filtered_people.append(person)
                    
                    # Trier par priorité et limiter
                    filtered_people.sort(key=lambda x: x.get('_priority_score', 0), reverse=True)
                    return filtered_people[:max_contacts]
                    
                else:
                    error_text = await response.text()
                    logger.error(f"Erreur recherche dirigeants Kaspr {response.status}: {error_text}")
                    return []
                    
        except Exception as e:
            logger.error(f"Erreur recherche dirigeants: {e}")
            return []
    
    def _is_executive_title(self, job_title: str) -> bool:
        """Vérifie si le titre correspond à un dirigeant"""
        executive_keywords = [
            'ceo', 'pdg', 'président', 'president', 'directeur', 'director', 
            'gérant', 'dg', 'cfo', 'daf', 'associé', 'partner', 'fondateur'
        ]
        return any(keyword in job_title for keyword in executive_keywords)
    
    def _calculate_contact_priority(self, person_data: Dict) -> float:
        """Calcule un score de priorité pour un contact (0-100)"""
        score = 0
        
        job_title = person_data.get('job_title', '').lower()
        seniority = person_data.get('seniority_level', '').lower()
        
        # Score basé sur le titre
        if any(title in job_title for title in ['ceo', 'pdg', 'président', 'president']):
            score += 50
        elif any(title in job_title for title in ['directeur', 'director', 'gérant', 'dg']):
            score += 40
        elif any(title in job_title for title in ['cfo', 'daf', 'financier']):
            score += 35
        elif any(title in job_title for title in ['associé', 'partner', 'fondateur']):
            score += 45
        
        # Score basé sur le niveau de séniorité
        seniority_scores = {
            'c_level': 25,
            'vp': 20,
            'director': 15,
            'manager': 10,
            'owner': 30
        }
        score += seniority_scores.get(seniority, 5)
        
        # Bonus pour email vérifié
        if person_data.get('verified_email'):
            score += 15
        
        # Bonus pour téléphone disponible
        if person_data.get('phone_number'):
            score += 10
        
        # Bonus pour LinkedIn
        if person_data.get('linkedin_url'):
            score += 5
        
        return min(100, score)
    
    async def _enrich_contact(self, contact_data: Dict) -> Optional[KasprContact]:
        """Enrichit un contact avec toutes les données disponibles"""
        
        try:
            contact = KasprContact()
            
            # Informations de base
            contact.first_name = contact_data.get('first_name', '')
            contact.last_name = contact_data.get('last_name', '')
            contact.full_name = f"{contact.first_name} {contact.last_name}".strip()
            contact.job_title = contact_data.get('job_title', '')
            contact.seniority_level = contact_data.get('seniority_level', '')
            contact.department = contact_data.get('department', '')
            
            # Coordonnées
            contact.email = contact_data.get('email')
            contact.phone = contact_data.get('phone_number')
            contact.mobile_phone = contact_data.get('mobile_phone')
            contact.linkedin_url = contact_data.get('linkedin_url')
            
            # Score de confiance
            if contact.email:
                email_verified = await self._verify_email(contact.email)
                contact.confidence_score = 0.9 if email_verified else 0.7
            else:
                contact.confidence_score = 0.5
            
            # Métadonnées
            contact.last_verified = datetime.now().isoformat()
            contact.company_name = contact_data.get('company_name', '')
            contact.company_domain = contact_data.get('company_domain', '')
            contact.location = contact_data.get('location', '')
            
            if contact_data.get('years_of_experience'):
                contact.experience_years = contact_data['years_of_experience']
            
            return contact
            
        except Exception as e:
            logger.error(f"Erreur enrichissement contact: {e}")
            return None
    
    async def _verify_email(self, email: str) -> bool:
        """Vérifie la validité d'un email via l'API Kaspr"""
        
        try:
            url = f"{self.BASE_URL}{self.VERIFY_ENDPOINT}"
            payload = {'email': email}
            self.contact_stats['api_calls'] += 1
            
            async with self.session.post(url, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('is_valid', False)
                else:
                    return True  # Bénéfice du doute
                    
        except Exception as e:
            logger.warning(f"Erreur vérification email {email}: {e}")
            return True
    
    # ========================================
    # SYSTÈME DE MOCK POUR DÉVELOPPEMENT
    # ========================================
    
    async def _get_mock_contacts(self, siren: str, company_name: str) -> List[KasprContact]:
        """Génère des contacts mock réalistes pour le développement"""
        
        # Simuler un délai réseau
        await asyncio.sleep(random.uniform(0.5, 2.0))
        
        logger.info(f"🎭 Mode MOCK Kaspr: Génération contacts pour {company_name}")
        
        # Templates de dirigeants français
        mock_templates = [
            {
                'first_name': 'Jean',
                'last_name': 'Martin',
                'job_title': 'Président Directeur Général',
                'seniority_level': 'c_level',
                'department': 'executive',
                'has_email': 0.9,
                'has_phone': 0.7,
                'has_linkedin': 0.8
            },
            {
                'first_name': 'Marie',
                'last_name': 'Dubois',
                'job_title': 'Directrice Générale',
                'seniority_level': 'c_level', 
                'department': 'executive',
                'has_email': 0.85,
                'has_phone': 0.6,
                'has_linkedin': 0.9
            },
            {
                'first_name': 'Pierre',
                'last_name': 'Bernard',
                'job_title': 'Directeur Administratif et Financier',
                'seniority_level': 'director',
                'department': 'finance',
                'has_email': 0.8,
                'has_phone': 0.5,
                'has_linkedin': 0.7
            }
        ]
        
        # Nombre de contacts basé sur la taille de l'entreprise
        siren_hash = hash(siren) % 100
        if siren_hash < 30:
            num_contacts = random.randint(1, 2)
        elif siren_hash < 70:
            num_contacts = random.randint(2, 3)
        else:
            num_contacts = random.randint(3, 5)
        
        contacts = []
        used_templates = random.sample(mock_templates, min(num_contacts, len(mock_templates)))
        
        for template in used_templates:
            contact = KasprContact()
            
            # Informations de base
            contact.first_name = template['first_name']
            contact.last_name = template['last_name']
            contact.full_name = f"{contact.first_name} {contact.last_name}"
            contact.job_title = template['job_title']
            contact.seniority_level = template['seniority_level']
            contact.department = template['department']
            
            # Génération d'email
            if random.random() < template['has_email']:
                company_domain = self._generate_company_domain(company_name)
                email_format = random.choice([
                    f"{contact.first_name.lower()}.{contact.last_name.lower()}@{company_domain}",
                    f"{contact.first_name[0].lower()}{contact.last_name.lower()}@{company_domain}"
                ])
                contact.email = email_format
            
            # Génération de téléphone
            if random.random() < template['has_phone']:
                contact.phone = self._generate_french_phone()
                if random.random() < 0.3:
                    contact.mobile_phone = self._generate_french_mobile()
            
            # Génération LinkedIn
            if random.random() < template['has_linkedin']:
                linkedin_name = f"{contact.first_name.lower()}-{contact.last_name.lower()}"
                contact.linkedin_url = f"https://www.linkedin.com/in/{linkedin_name}-{random.randint(100, 999)}"
            
            # Métadonnées
            contact.confidence_score = random.uniform(0.7, 0.95)
            contact.last_verified = datetime.now().isoformat()
            contact.company_name = company_name
            contact.company_domain = self._generate_company_domain(company_name)
            contact.location = random.choice(['Paris', 'Lyon', 'Marseille', 'Toulouse', 'Nantes'])
            contact.experience_years = random.randint(5, 25)
            
            contacts.append(contact)
        
        # Mise à jour des stats mock
        self.contact_stats['companies_processed'] += 1
        self.contact_stats['contacts_found'] += len(contacts)
        self.contact_stats['contacts_with_email'] += sum(1 for c in contacts if c.email)
        self.contact_stats['contacts_with_phone'] += sum(1 for c in contacts if c.phone or c.mobile_phone)
        
        logger.info(f"🎭 Mock Kaspr: {len(contacts)} contacts générés pour {company_name}")
        return contacts
    
    def _generate_company_domain(self, company_name: str) -> str:
        """Génère un domaine plausible pour une entreprise"""
        clean_name = company_name.lower()
        clean_name = clean_name.replace('cabinet', '').replace('expertise', '').replace('comptable', '')
        clean_name = clean_name.replace('&', '').replace(' et ', '').replace(' ', '').replace('-', '')
        
        words = [w for w in clean_name.split() if len(w) > 2]
        if words:
            domain_base = ''.join(words[:2])[:12]
        else:
            domain_base = 'cabinet' + str(random.randint(100, 999))
        
        extensions = ['.fr', '.com', '.net', '-expertise.fr', '-conseil.fr']
        return f"{domain_base}{random.choice(extensions)}"
    
    def _generate_french_phone(self) -> str:
        """Génère un numéro de téléphone français réaliste"""
        prefixes = ['01', '02', '03', '04', '05']
        prefix = random.choice(prefixes)
        number = ''.join([str(random.randint(0, 9)) for _ in range(8)])
        return f"{prefix} {number[:2]} {number[2:4]} {number[4:6]} {number[6:8]}"
    
    def _generate_french_mobile(self) -> str:
        """Génère un numéro de mobile français réaliste"""
        prefix = random.choice(['06', '07'])
        number = ''.join([str(random.randint(0, 9)) for _ in range(8)])
        return f"{prefix} {number[:2]} {number[2:4]} {number[4:6]} {number[6:8]}"
    
    def _dict_to_contact(self, contact_dict: Dict) -> KasprContact:
        """Convertit un dictionnaire en objet KasprContact"""
        contact = KasprContact()
        for key, value in contact_dict.items():
            if hasattr(contact, key):
                setattr(contact, key, value)
        return contact
    
    async def enrich_company_data(self, basic_data: Dict) -> Dict:
        """Enrichit les données de base d'une entreprise avec les contacts"""
        siren = basic_data.get('siren')
        company_name = basic_data.get('nom_entreprise', f"Entreprise {siren}")
        
        if not siren:
            return basic_data
        
        contacts = await self.get_company_contacts(siren, company_name)
        if contacts:
            enriched = basic_data.copy()
            enriched['kaspr_contacts'] = [contact.to_dict() for contact in contacts]
            enriched['kaspr_enriched_at'] = datetime.now().isoformat()
            enriched['source'] = 'kaspr'
            return enriched
        
        return basic_data
    
    def get_contact_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques des contacts"""
        return {
            **self.contact_stats,
            'mode': 'MOCK' if self.use_mock else 'API',
            'api_key_configured': bool(self.api_key),
            'success_rate': (self.contact_stats['contacts_found'] / max(self.contact_stats['companies_processed'], 1)) * 100,
            'email_coverage': (self.contact_stats['contacts_with_email'] / max(self.contact_stats['contacts_found'], 1)) * 100,
            'phone_coverage': (self.contact_stats['contacts_with_phone'] / max(self.contact_stats['contacts_found'], 1)) * 100
        }

# Export de la classe principale
__all__ = ['KasprClient', 'KasprContact', 'KasprContactType']