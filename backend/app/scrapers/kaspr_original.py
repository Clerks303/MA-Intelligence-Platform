"""
Module Kaspr API Client pour l'enrichissement de contacts dirigeants.

Ce module permet de récupérer les coordonnées des dirigeants d'entreprise
via l'API Kaspr (emails, téléphones, LinkedIn) et de les formater pour 
insertion dans le modèle CompanyContact.
"""

import asyncio
import aiohttp
import logging
import random
import json
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import os
from dataclasses import dataclass, field
from enum import Enum

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


class KasprAPIClient:
    """
    Client asynchrone pour l'API Kaspr d'enrichissement de contacts.
    
    Kaspr permet de récupérer les coordonnées professionnelles des dirigeants
    d'entreprise avec un fort taux de précision et de vérification.
    """
    
    # Endpoints API Kaspr
    BASE_URL = "https://api.kaspr.io/api/v1"
    SEARCH_ENDPOINT = "/search/people"
    COMPANY_ENDPOINT = "/search/companies"
    CONTACT_ENDPOINT = "/contacts"
    VERIFY_ENDPOINT = "/verify/email"
    
    def __init__(self, db_client=None, use_mock: bool = None):
        """
        Initialise le client Kaspr
        
        Args:
            db_client: Client de base de données
            use_mock: Force l'utilisation du mock (None = auto-détection)
        """
        self.db = db_client
        self.api_key = os.environ.get('KASPR_API_KEY', '')
        self.session = None
        
        # Détermination auto du mode mock
        if use_mock is None:
            self.use_mock = not bool(self.api_key)
        else:
            self.use_mock = use_mock
        
        # Configuration des requêtes
        self.rate_limit_delay = 1.0  # Délai entre requêtes (secondes)
        self.max_contacts_per_company = 5  # Limite de contacts par entreprise
        self.timeout = 30  # Timeout des requêtes
        
        # Statistiques
        self.stats = {
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
        """Context manager pour gestion des ressources async"""
        if not self.use_mock:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout),
                headers={
                    'Authorization': f'Bearer {self.api_key}',
                    'Content-Type': 'application/json',
                    'User-Agent': 'MA-Intelligence-Platform/2.0'
                }
            )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Nettoyage des ressources"""
        if self.session:
            await self.session.close()
    
    async def get_company_contacts(
        self, 
        siren: str, 
        company_name: str,
        domain: Optional[str] = None,
        max_contacts: Optional[int] = None
    ) -> List[KasprContact]:
        """
        Récupère les contacts dirigeants d'une entreprise.
        
        Args:
            siren: Numéro SIREN de l'entreprise
            company_name: Nom de l'entreprise
            domain: Domaine web de l'entreprise (optionnel)
            max_contacts: Nombre max de contacts (défaut: 5)
            
        Returns:
            Liste des contacts enrichis
        """
        
        if self.use_mock:
            return await self._get_mock_contacts(siren, company_name)
        
        max_contacts = max_contacts or self.max_contacts_per_company
        
        logger.info(f"Recherche contacts Kaspr pour {company_name} ({siren})")
        
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
                        enriched_contacts.append(enriched_contact)
                        
                        # Respect du rate limiting
                        await asyncio.sleep(self.rate_limit_delay)
                        
                except Exception as e:
                    logger.error(f"Erreur enrichissement contact: {e}")
                    self.stats['errors'] += 1
            
            # 4. Mise à jour des statistiques
            self.stats['companies_processed'] += 1
            self.stats['contacts_found'] += len(enriched_contacts)
            self.stats['contacts_with_email'] += sum(1 for c in enriched_contacts if c.email)
            self.stats['contacts_with_phone'] += sum(1 for c in enriched_contacts if c.phone or c.mobile_phone)
            
            logger.info(f"Kaspr: {len(enriched_contacts)} contacts trouvés pour {company_name}")
            return enriched_contacts
            
        except Exception as e:
            logger.error(f"Erreur Kaspr pour {company_name}: {e}")
            self.stats['errors'] += 1
            return []
    
    async def _search_company(self, company_name: str, domain: Optional[str] = None) -> Optional[Dict]:
        """Recherche une entreprise dans Kaspr"""
        
        search_params = {
            'name': company_name,
            'country': 'FR',  # France uniquement
            'limit': 1
        }
        
        if domain:
            search_params['domain'] = domain
        
        try:
            url = f"{self.BASE_URL}{self.COMPANY_ENDPOINT}"
            self.stats['api_calls'] += 1
            
            async with self.session.get(url, params=search_params) as response:
                if response.status == 200:
                    data = await response.json()
                    companies = data.get('data', [])
                    
                    if companies:
                        return companies[0]  # Premier résultat
                        
                elif response.status == 404:
                    logger.info(f"Entreprise non trouvée dans Kaspr: {company_name}")
                    return None
                    
                elif response.status == 429:
                    logger.warning("Rate limit Kaspr atteint")
                    await asyncio.sleep(60)  # Attendre 1 minute
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
        
        # Cibler les postes de direction
        target_titles = [
            'CEO', 'PDG', 'Président', 'President',
            'Directeur', 'Director', 'DG', 'Gérant',
            'CFO', 'DAF', 'Directeur Financier',
            'Associé', 'Partner', 'Fondateur', 'Founder'
        ]
        
        search_params = {
            'company_id': company_id,
            'seniority_levels': ['director', 'vp', 'c_level', 'owner'],
            'limit': max_contacts * 2,  # Chercher plus pour filtrer ensuite
            'verified_emails': True  # Privilégier les emails vérifiés
        }
        
        try:
            url = f"{self.BASE_URL}{self.SEARCH_ENDPOINT}"
            self.stats['api_calls'] += 1
            
            async with self.session.get(url, params=search_params) as response:
                if response.status == 200:
                    data = await response.json()
                    people = data.get('data', [])
                    
                    # Filtrer et prioriser les dirigeants
                    filtered_people = []
                    for person in people:
                        job_title = person.get('job_title', '').lower()
                        
                        # Vérifier si c'est un poste de direction
                        is_executive = any(title.lower() in job_title for title in target_titles)
                        
                        if is_executive:
                            # Calculer un score de priorité
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
            # Construction du contact de base
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
            
            # Vérification email si disponible
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
            
            # Calcul expérience (approximatif)
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
            self.stats['api_calls'] += 1
            
            async with self.session.post(url, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('is_valid', False)
                else:
                    # En cas d'erreur, on considère l'email comme potentiellement valide
                    return True
                    
        except Exception as e:
            logger.warning(f"Erreur vérification email {email}: {e}")
            return True  # Bénéfice du doute
    
    # ========================================
    # SYSTÈME DE MOCK POUR DÉVELOPPEMENT
    # ========================================
    
    async def _get_mock_contacts(self, siren: str, company_name: str) -> List[KasprContact]:
        """Génère des contacts mock réalistes pour le développement"""
        
        # Simuler un délai réseau
        await asyncio.sleep(random.uniform(0.5, 2.0))
        
        logger.info(f"🎭 Mode MOCK: Génération contacts pour {company_name}")
        
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
            },
            {
                'first_name': 'Sophie',
                'last_name': 'Leroy',
                'job_title': 'Associée Gérante',
                'seniority_level': 'owner',
                'department': 'executive',
                'has_email': 0.95,
                'has_phone': 0.8,
                'has_linkedin': 0.6
            },
            {
                'first_name': 'Thomas',
                'last_name': 'Moreau',
                'job_title': 'Directeur des Opérations',
                'seniority_level': 'director',
                'department': 'operations',
                'has_email': 0.75,
                'has_phone': 0.4,
                'has_linkedin': 0.8
            }
        ]
        
        # Nombre de contacts basé sur la taille de l'entreprise (approximée par SIREN)
        siren_hash = hash(siren) % 100
        if siren_hash < 30:
            num_contacts = random.randint(1, 2)  # Petite entreprise
        elif siren_hash < 70:
            num_contacts = random.randint(2, 3)  # Moyenne entreprise
        else:
            num_contacts = random.randint(3, 5)  # Grande entreprise
        
        contacts = []
        used_templates = random.sample(mock_templates, min(num_contacts, len(mock_templates)))
        
        for i, template in enumerate(used_templates):
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
                    f"{contact.first_name[0].lower()}{contact.last_name.lower()}@{company_domain}",
                    f"{contact.first_name.lower()}{contact.last_name[0].lower()}@{company_domain}"
                ])
                contact.email = email_format
            
            # Génération de téléphone
            if random.random() < template['has_phone']:
                contact.phone = self._generate_french_phone()
                if random.random() < 0.3:  # 30% ont aussi un mobile
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
        self.stats['companies_processed'] += 1
        self.stats['contacts_found'] += len(contacts)
        self.stats['contacts_with_email'] += sum(1 for c in contacts if c.email)
        self.stats['contacts_with_phone'] += sum(1 for c in contacts if c.phone or c.mobile_phone)
        
        logger.info(f"🎭 Mock: {len(contacts)} contacts générés pour {company_name}")
        return contacts
    
    def _generate_company_domain(self, company_name: str) -> str:
        """Génère un domaine plausible pour une entreprise"""
        # Nettoyer le nom
        clean_name = company_name.lower()
        clean_name = clean_name.replace('cabinet', '').replace('expertise', '').replace('comptable', '')
        clean_name = clean_name.replace('&', '').replace(' et ', '').replace(' ', '').replace('-', '')
        
        # Prendre les premiers mots significatifs
        words = [w for w in clean_name.split() if len(w) > 2]
        if words:
            domain_base = ''.join(words[:2])[:12]  # Max 12 caractères
        else:
            domain_base = 'cabinet' + str(random.randint(100, 999))
        
        # Extensions françaises communes
        extensions = ['.fr', '.com', '.net', '-expertise.fr', '-conseil.fr']
        return f"{domain_base}{random.choice(extensions)}"
    
    def _generate_french_phone(self) -> str:
        """Génère un numéro de téléphone français réaliste"""
        # Formats téléphone français
        prefixes = ['01', '02', '03', '04', '05']  # Fixes
        prefix = random.choice(prefixes)
        number = ''.join([str(random.randint(0, 9)) for _ in range(8)])
        return f"{prefix} {number[:2]} {number[2:4]} {number[4:6]} {number[6:8]}"
    
    def _generate_french_mobile(self) -> str:
        """Génère un numéro de mobile français réaliste"""
        # Mobiles français commencent par 06 ou 07
        prefix = random.choice(['06', '07'])
        number = ''.join([str(random.randint(0, 9)) for _ in range(8)])
        return f"{prefix} {number[:2]} {number[2:4]} {number[4:6]} {number[6:8]}"
    
    # ========================================
    # FORMATAGE POUR BASE DE DONNÉES
    # ========================================
    
    def format_contacts_for_db(self, contacts: List[KasprContact], company_id: str) -> List[Dict[str, Any]]:
        """
        Formate les contacts Kaspr pour insertion dans CompanyContact.
        
        Args:
            contacts: Liste des contacts Kaspr
            company_id: UUID de l'entreprise
            
        Returns:
            Liste de dictionnaires prêts pour CompanyContact
        """
        
        formatted_contacts = []
        
        for contact in contacts:
            # Détermination du type de contact
            contact_type = self._determine_contact_type(contact.job_title, contact.seniority_level)
            
            # Formatage pour le modèle CompanyContact
            formatted_contact = {
                'company_id': company_id,
                'nom_complet': contact.full_name,
                'prenom': contact.first_name,
                'nom': contact.last_name,
                'type_contact': contact_type,
                'poste': contact.job_title,
                'qualite': self._extract_legal_quality(contact.job_title),
                'est_dirigeant': self._is_executive(contact.job_title, contact.seniority_level),
                'email_professionnel': contact.email,
                'telephone_direct': contact.phone,
                'telephone_mobile': contact.mobile_phone,
                'linkedin_url': contact.linkedin_url,
                'source': 'kaspr',
                'confidence_score': contact.confidence_score,
                'derniere_verification': contact.last_verified,
                'statut_email': 'VERIFIED' if contact.email and contact.confidence_score > 0.8 else 'UNKNOWN',
                'statut_telephone': 'VERIFIED' if contact.phone else 'UNKNOWN',
                'experience_precedente': f"{contact.experience_years} ans d'expérience" if contact.experience_years else None,
                'reseaux_sociaux': {'linkedin': contact.linkedin_url} if contact.linkedin_url else None
            }
            
            formatted_contacts.append(formatted_contact)
        
        return formatted_contacts
    
    def _determine_contact_type(self, job_title: str, seniority_level: str) -> str:
        """Détermine le type de contact selon CompanyContact.ContactTypeEnum"""
        job_lower = job_title.lower()
        
        if any(keyword in job_lower for keyword in ['pdg', 'ceo', 'président', 'directeur général', 'gérant']):
            return 'dirigeant'
        elif any(keyword in job_lower for keyword in ['daf', 'cfo', 'financier', 'comptab']):
            return 'comptable'
        elif any(keyword in job_lower for keyword in ['commercial', 'vente', 'marketing']):
            return 'commercial'
        elif any(keyword in job_lower for keyword in ['rh', 'ressources humaines', 'hr']):
            return 'rh'
        else:
            return 'autre'
    
    def _extract_legal_quality(self, job_title: str) -> str:
        """Extrait la qualité juridique du titre"""
        # Mapping des titres vers qualités juridiques
        quality_mapping = {
            'pdg': 'PDG',
            'président directeur général': 'PDG',
            'président': 'Président',
            'directeur général': 'Directeur Général',
            'gérant': 'Gérant',
            'associé gérant': 'Associé Gérant',
            'directeur': 'Directeur',
            'daf': 'DAF',
            'cfo': 'CFO'
        }
        
        job_lower = job_title.lower()
        for key, quality in quality_mapping.items():
            if key in job_lower:
                return quality
        
        return job_title  # Retour par défaut
    
    def _is_executive(self, job_title: str, seniority_level: str) -> bool:
        """Détermine si c'est un dirigeant"""
        executive_keywords = ['pdg', 'ceo', 'président', 'directeur général', 'gérant', 'associé']
        executive_levels = ['c_level', 'owner']
        
        job_lower = job_title.lower()
        
        return (
            any(keyword in job_lower for keyword in executive_keywords) or
            seniority_level in executive_levels
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques d'utilisation"""
        return {
            **self.stats,
            'mode': 'MOCK' if self.use_mock else 'API',
            'api_key_configured': bool(self.api_key),
            'success_rate': (self.stats['contacts_found'] / max(self.stats['companies_processed'], 1)) * 100,
            'email_coverage': (self.stats['contacts_with_email'] / max(self.stats['contacts_found'], 1)) * 100,
            'phone_coverage': (self.stats['contacts_with_phone'] / max(self.stats['contacts_found'], 1)) * 100
        }


# Fonctions standalone pour compatibilité

async def get_company_contacts(
    siren: str, 
    company_name: str, 
    domain: Optional[str] = None,
    use_mock: bool = None
) -> List[Dict[str, Any]]:
    """
    Fonction standalone pour récupérer les contacts d'une entreprise.
    
    Returns:
        Liste de contacts formatés pour CompanyContact
    """
    
    try:
        async with KasprAPIClient(use_mock=use_mock) as kaspr_client:
            # Récupération des contacts
            contacts = await kaspr_client.get_company_contacts(siren, company_name, domain)
            
            # Formatage pour la base (sans company_id pour l'instant)
            formatted_contacts = []
            for contact in contacts:
                formatted = {
                    'nom_complet': contact.full_name,
                    'prenom': contact.first_name,
                    'nom': contact.last_name,
                    'poste': contact.job_title,
                    'email_professionnel': contact.email,
                    'telephone_direct': contact.phone,
                    'telephone_mobile': contact.mobile_phone,
                    'linkedin_url': contact.linkedin_url,
                    'source': 'kaspr',
                    'confidence_score': contact.confidence_score,
                    'est_dirigeant': kaspr_client._is_executive(contact.job_title, contact.seniority_level),
                    'type_contact': kaspr_client._determine_contact_type(contact.job_title, contact.seniority_level)
                }
                formatted_contacts.append(formatted)
            
            return formatted_contacts
            
    except Exception as e:
        logger.error(f"Erreur récupération contacts pour {company_name}: {e}")
        return []


async def enrich_companies_contacts(
    companies: List[Dict[str, Any]], 
    max_concurrent: int = 3
) -> List[Dict[str, Any]]:
    """
    Enrichit une liste d'entreprises avec leurs contacts dirigeants.
    
    Args:
        companies: Liste d'entreprises avec siren et nom_entreprise
        max_concurrent: Nombre max d'enrichissements simultanés
        
    Returns:
        Liste d'entreprises enrichies avec contacts
    """
    
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def enrich_single_company(company: Dict[str, Any]) -> Dict[str, Any]:
        async with semaphore:
            try:
                siren = company.get('siren')
                nom = company.get('nom_entreprise')
                
                if not siren or not nom:
                    return company
                
                contacts = await get_company_contacts(siren, nom)
                company['kaspr_contacts'] = contacts
                company['kaspr_enriched_at'] = datetime.now().isoformat()
                
                return company
                
            except Exception as e:
                logger.error(f"Erreur enrichissement {company.get('nom_entreprise', 'Unknown')}: {e}")
                company['kaspr_contacts'] = []
                return company
    
    # Exécution en parallèle
    enriched_companies = await asyncio.gather(
        *[enrich_single_company(company) for company in companies],
        return_exceptions=True
    )
    
    # Filtrage des exceptions
    results = []
    for result in enriched_companies:
        if isinstance(result, Exception):
            logger.error(f"Exception dans enrichissement parallèle: {result}")
        else:
            results.append(result)
    
    return results