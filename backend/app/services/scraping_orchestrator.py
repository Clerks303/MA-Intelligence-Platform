import asyncio
import logging
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime, timedelta
from enum import Enum
import json

from app.scrapers.pappers import PappersAPIClient
from app.scrapers.infogreffe import InfogreffeAPIClient  
from app.scrapers.societe import SocieteScraper
from app.scrapers.kaspr import KasprAPIClient
from app.core.exceptions import ScrapingError, ValidationError

logger = logging.getLogger(__name__)


class EnrichmentSource(Enum):
    """Sources d'enrichissement disponibles"""
    PAPPERS = "pappers"
    INFOGREFFE = "infogreffe" 
    SOCIETE = "societe"
    KASPR = "kaspr"


class EnrichmentStatus(Enum):
    """Statuts d'enrichissement"""
    PENDING = "pending"
    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"
    SKIPPED = "skipped"


class ScrapingOrchestrator:
    """
    Orchestrateur central pour l'enrichissement multi-sources des entreprises.
    
    Gère l'appel séquentiel des scrapers avec gestion d'erreurs robuste,
    hooks pour scoring et export, et suivi détaillé du processus.
    """
    
    def __init__(self, db_client, config: Optional[Dict] = None):
        self.db = db_client
        self.config = config or self._get_default_config()
        
        # Initialisation des clients (lazy loading)
        self._pappers_client = None
        self._infogreffe_client = None
        self._societe_client = None
        self._kaspr_client = None
        
        # Hooks pour extensions futures
        self._scoring_hooks: List[Callable] = []
        self._export_hooks: List[Callable] = []
        self._validation_hooks: List[Callable] = []
        
        # Statistiques d'exécution
        self.stats = {
            'companies_processed': 0,
            'total_success': 0,
            'partial_success': 0,
            'total_failed': 0,
            'source_stats': {source.value: {'success': 0, 'failed': 0} for source in EnrichmentSource}
        }
    
    async def __aenter__(self):
        """Context manager pour gestion des ressources async"""
        await self._initialize_clients()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Nettoyage des ressources"""
        await self._cleanup_clients()
    
    def _get_default_config(self) -> Dict:
        """Configuration par défaut de l'orchestrateur"""
        return {
            'sources_enabled': {
                EnrichmentSource.PAPPERS: True,
                EnrichmentSource.INFOGREFFE: True,
                EnrichmentSource.SOCIETE: True,
                EnrichmentSource.KASPR: False  # Désactivé par défaut jusqu'à implémentation
            },
            'source_order': [
                EnrichmentSource.PAPPERS,     # Base de données la plus fiable en premier
                EnrichmentSource.INFOGREFFE,  # Enrichissement officiel
                EnrichmentSource.SOCIETE,     # Données financières détaillées
                EnrichmentSource.KASPR        # Contacts (dernière étape)
            ],
            'retry_policy': {
                'max_retries': 3,
                'retry_delay': 2,  # secondes
                'exponential_backoff': True
            },
            'timeouts': {
                EnrichmentSource.PAPPERS: 30,
                EnrichmentSource.INFOGREFFE: 45,
                EnrichmentSource.SOCIETE: 120,  # Plus long car scraping web
                EnrichmentSource.KASPR: 60
            },
            'validation': {
                'required_fields': ['siren', 'nom_entreprise'],
                'ca_min': 3_000_000,  # 3M€
                'ca_max': 50_000_000,  # 50M€
                'effectif_min': 30
            }
        }
    
    async def _initialize_clients(self):
        """Initialisation des clients de scraping"""
        try:
            if self.config['sources_enabled'][EnrichmentSource.PAPPERS]:
                self._pappers_client = PappersAPIClient(self.db)
                await self._pappers_client.__aenter__()
                
            if self.config['sources_enabled'][EnrichmentSource.INFOGREFFE]:
                self._infogreffe_client = InfogreffeAPIClient(self.db)
                await self._infogreffe_client.__aenter__()
                
            if self.config['sources_enabled'][EnrichmentSource.SOCIETE]:
                self._societe_client = SocieteScraper(self.db)
                await self._societe_client.__aenter__()
                
            if self.config['sources_enabled'][EnrichmentSource.KASPR]:
                self._kaspr_client = KasprAPIClient(self.db)
                await self._kaspr_client.__aenter__()
                
            logger.info("Tous les clients de scraping initialisés avec succès")
            
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation des clients: {e}")
            await self._cleanup_clients()
            raise ScrapingError(f"Impossible d'initialiser les clients: {e}")
    
    async def _cleanup_clients(self):
        """Nettoyage des ressources des clients"""
        clients = [
            self._pappers_client,
            self._infogreffe_client, 
            self._societe_client,
            self._kaspr_client
        ]
        
        for client in clients:
            if client:
                try:
                    await client.__aexit__(None, None, None)
                except Exception as e:
                    logger.warning(f"Erreur lors du nettoyage d'un client: {e}")
    
    async def enrich_company_full(
        self, 
        siren: str, 
        force_refresh: bool = False,
        sources_override: Optional[List[EnrichmentSource]] = None
    ) -> Dict[str, Any]:
        """
        Enrichissement complet d'une entreprise via toutes les sources disponibles.
        
        Args:
            siren: Numéro SIREN de l'entreprise (9 chiffres)
            force_refresh: Force le re-scraping même si données récentes
            sources_override: Liste des sources à utiliser (override config)
            
        Returns:
            Dict contenant les données enrichies et le statut d'enrichissement
            
        Raises:
            ValidationError: Si le SIREN est invalide
            ScrapingError: Si aucune source n'a réussi l'enrichissement
        """
        
        # Validation du SIREN
        if not self._validate_siren(siren):
            raise ValidationError(f"SIREN invalide: {siren}")
        
        logger.info(f"Début enrichissement complet pour SIREN: {siren}")
        
        # Vérification des données existantes
        existing_data = await self._get_existing_company_data(siren)
        if existing_data and not force_refresh and not self._needs_refresh(existing_data):
            logger.info(f"Données récentes trouvées pour {siren}, enrichissement skippé")
            return {
                'siren': siren,
                'status': EnrichmentStatus.SKIPPED,
                'data': existing_data,
                'enrichment_details': existing_data.get('enrichment_status', {}),
                'message': 'Données récentes disponibles'
            }
        
        # Préparation du résultat d'enrichissement
        enrichment_result = {
            'siren': siren,
            'status': EnrichmentStatus.PENDING,
            'data': existing_data or {},
            'enrichment_details': {},
            'errors': [],
            'started_at': datetime.now().isoformat(),
            'completed_at': None
        }
        
        # Détermination des sources à utiliser
        sources_to_use = sources_override or [
            source for source in self.config['source_order'] 
            if self.config['sources_enabled'][source]
        ]
        
        successful_sources = 0
        total_sources = len(sources_to_use)
        
        # Enrichissement séquentiel par source
        for source in sources_to_use:
            try:
                logger.info(f"Enrichissement {source.value} pour SIREN {siren}")
                
                source_data = await self._enrich_from_source(siren, source, enrichment_result['data'])
                
                if source_data:
                    # Fusion des données
                    enrichment_result['data'].update(source_data)
                    enrichment_result['enrichment_details'][source.value] = {
                        'status': 'success',
                        'timestamp': datetime.now().isoformat(),
                        'fields_added': list(source_data.keys())
                    }
                    successful_sources += 1
                    self.stats['source_stats'][source.value]['success'] += 1
                    logger.info(f"✅ {source.value} réussi pour {siren}")
                else:
                    enrichment_result['enrichment_details'][source.value] = {
                        'status': 'no_data',
                        'timestamp': datetime.now().isoformat(),
                        'message': 'Aucune donnée retournée'
                    }
                    logger.warning(f"⚠️ {source.value} n'a retourné aucune donnée pour {siren}")
                
            except Exception as e:
                error_msg = f"Erreur {source.value}: {str(e)}"
                enrichment_result['errors'].append(error_msg)
                enrichment_result['enrichment_details'][source.value] = {
                    'status': 'failed',
                    'timestamp': datetime.now().isoformat(),
                    'error': str(e)
                }
                self.stats['source_stats'][source.value]['failed'] += 1
                logger.error(f"❌ {error_msg}")
        
        # Détermination du statut final
        if successful_sources == 0:
            enrichment_result['status'] = EnrichmentStatus.FAILED
            self.stats['total_failed'] += 1
            raise ScrapingError(f"Aucune source n'a réussi l'enrichissement pour {siren}")
        elif successful_sources == total_sources:
            enrichment_result['status'] = EnrichmentStatus.SUCCESS
            self.stats['total_success'] += 1
        else:
            enrichment_result['status'] = EnrichmentStatus.PARTIAL
            self.stats['partial_success'] += 1
        
        # Validation des données enrichies
        if not await self._validate_enriched_data(enrichment_result['data']):
            enrichment_result['errors'].append("Validation des données enrichies échouée")
            logger.warning(f"Données enrichies invalides pour {siren}")
        
        # Hooks de post-traitement
        await self._run_validation_hooks(enrichment_result)
        await self._run_scoring_hooks(enrichment_result)
        await self._run_export_hooks(enrichment_result)
        
        # Sauvegarde en base
        enrichment_result['completed_at'] = datetime.now().isoformat()
        await self._save_enriched_data(enrichment_result)
        
        self.stats['companies_processed'] += 1
        logger.info(f"Enrichissement terminé pour {siren}: {enrichment_result['status'].value}")
        
        return enrichment_result
    
    async def _enrich_from_source(
        self, 
        siren: str, 
        source: EnrichmentSource, 
        existing_data: Dict
    ) -> Optional[Dict]:
        """
        Enrichissement depuis une source spécifique avec gestion d'erreurs et retry.
        """
        timeout = self.config['timeouts'][source]
        max_retries = self.config['retry_policy']['max_retries']
        retry_delay = self.config['retry_policy']['retry_delay']
        
        for attempt in range(max_retries + 1):
            try:
                # Timeout global pour la source
                enriched_data = await asyncio.wait_for(
                    self._call_source_method(siren, source, existing_data),
                    timeout=timeout
                )
                
                if enriched_data:
                    # Nettoyage et validation des données
                    cleaned_data = self._clean_source_data(enriched_data, source)
                    return cleaned_data
                
                return None
                
            except asyncio.TimeoutError:
                logger.warning(f"Timeout {source.value} pour {siren} (tentative {attempt + 1})")
                if attempt < max_retries:
                    await asyncio.sleep(retry_delay * (2 ** attempt if self.config['retry_policy']['exponential_backoff'] else 1))
                    continue
                raise ScrapingError(f"Timeout définitif pour {source.value}")
                
            except Exception as e:
                logger.error(f"Erreur {source.value} pour {siren} (tentative {attempt + 1}): {e}")
                if attempt < max_retries:
                    await asyncio.sleep(retry_delay * (2 ** attempt if self.config['retry_policy']['exponential_backoff'] else 1))
                    continue
                raise
        
        return None
    
    async def _call_source_method(
        self, 
        siren: str, 
        source: EnrichmentSource, 
        existing_data: Dict
    ) -> Optional[Dict]:
        """Appel de la méthode spécifique à chaque source"""
        
        if source == EnrichmentSource.PAPPERS and self._pappers_client:
            return await self._pappers_client.get_company_details(siren)
            
        elif source == EnrichmentSource.INFOGREFFE and self._infogreffe_client:
            return await self._infogreffe_client.get_company_details(siren)
            
        elif source == EnrichmentSource.SOCIETE and self._societe_client:
            # Pour Société.com, on a besoin du nom de l'entreprise
            company_name = existing_data.get('nom_entreprise', '')
            if not company_name:
                logger.warning(f"Nom d'entreprise manquant pour scraping Société.com: {siren}")
                return None
            
            company_info = {
                'siren': siren,
                'nom_entreprise': company_name,
                'url': f"https://www.societe.com/societe/{company_name.replace(' ', '-').lower()}/{siren}.html"
            }
            return await self._societe_client.scrape_company_details(company_info)
            
        elif source == EnrichmentSource.KASPR and self._kaspr_client:
            company_name = existing_data.get('nom_entreprise', '')
            if not company_name:
                logger.warning(f"Nom d'entreprise manquant pour enrichissement Kaspr: {siren}")
                return None
            
            # Récupération des contacts dirigeants
            contacts = await self._kaspr_client.get_company_contacts(siren, company_name)
            
            if contacts:
                # Formatage des contacts pour la réponse
                formatted_contacts = []
                for contact in contacts:
                    contact_data = {
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
                        'est_dirigeant': self._kaspr_client._is_executive(contact.job_title, contact.seniority_level),
                        'type_contact': self._kaspr_client._determine_contact_type(contact.job_title, contact.seniority_level)
                    }
                    formatted_contacts.append(contact_data)
                
                return {
                    'kaspr_contacts': formatted_contacts,
                    'kaspr_contacts_count': len(formatted_contacts),
                    'kaspr_contacts_with_email': sum(1 for c in contacts if c.email),
                    'kaspr_contacts_with_phone': sum(1 for c in contacts if c.phone or c.mobile_phone),
                    'kaspr_enriched_at': datetime.now().isoformat()
                }
            
            return None
            
        else:
            logger.warning(f"Source non supportée ou client non initialisé: {source}")
            return None
    
    def _clean_source_data(self, data: Dict, source: EnrichmentSource) -> Dict:
        """Nettoyage et standardisation des données par source"""
        if not data:
            return {}
        
        # Ajout de métadonnées sur la source
        cleaned = data.copy()
        cleaned['_source'] = source.value
        cleaned['_enriched_at'] = datetime.now().isoformat()
        
        # Nettoyage spécifique par source
        if source == EnrichmentSource.PAPPERS:
            # Les données Pappers sont déjà bien formatées
            pass
            
        elif source == EnrichmentSource.INFOGREFFE:
            # Conversion des dates au format ISO
            if 'date_creation' in cleaned and cleaned['date_creation']:
                try:
                    # Format attendu: DD/MM/YYYY -> YYYY-MM-DD
                    date_parts = cleaned['date_creation'].split('/')
                    if len(date_parts) == 3:
                        cleaned['date_creation'] = f"{date_parts[2]}-{date_parts[1]}-{date_parts[0]}"
                except:
                    pass
                    
        elif source == EnrichmentSource.SOCIETE:
            # Nettoyage des données numériques
            for field in ['chiffre_affaires', 'resultat', 'capital_social']:
                if field in cleaned and isinstance(cleaned[field], str):
                    try:
                        cleaned[field] = float(cleaned[field].replace(' ', '').replace(',', '.'))
                    except:
                        cleaned[field] = None
        
        return cleaned
    
    async def _validate_enriched_data(self, data: Dict) -> bool:
        """Validation des données enrichies selon les critères M&A"""
        validation_config = self.config['validation']
        
        # Champs requis
        for field in validation_config['required_fields']:
            if not data.get(field):
                logger.warning(f"Champ requis manquant: {field}")
                return False
        
        # Validation CA
        ca = data.get('chiffre_affaires')
        if ca and isinstance(ca, (int, float)):
            if ca < validation_config['ca_min'] or ca > validation_config['ca_max']:
                logger.info(f"CA hors critères M&A: {ca}")
                return False
        
        # Validation effectif
        effectif = data.get('effectif')
        if effectif and isinstance(effectif, (int, float)):
            if effectif < validation_config['effectif_min']:
                logger.info(f"Effectif trop faible: {effectif}")
                return False
        
        return True
    
    # Hooks pour extensions futures
    def add_scoring_hook(self, hook_func: Callable):
        """Ajoute un hook de scoring"""
        self._scoring_hooks.append(hook_func)
    
    def add_export_hook(self, hook_func: Callable):
        """Ajoute un hook d'export"""
        self._export_hooks.append(hook_func)
    
    def add_validation_hook(self, hook_func: Callable):
        """Ajoute un hook de validation"""
        self._validation_hooks.append(hook_func)
    
    async def _run_validation_hooks(self, enrichment_result: Dict):
        """Exécute les hooks de validation"""
        for hook in self._validation_hooks:
            try:
                await hook(enrichment_result)
            except Exception as e:
                logger.warning(f"Erreur dans hook de validation: {e}")
    
    async def _run_scoring_hooks(self, enrichment_result: Dict):
        """Exécute les hooks de scoring"""
        for hook in self._scoring_hooks:
            try:
                result = await hook(enrichment_result)
                if result:
                    # Mettre à jour les données avec le score
                    if isinstance(result, dict):
                        enrichment_result['data'].update(result)
                    logger.info(f"Hook de scoring exécuté avec succès")
            except Exception as e:
                logger.warning(f"Erreur dans hook de scoring: {e}")
        
        # Hook de scoring M&A par défaut si aucun hook personnalisé
        if not self._scoring_hooks:
            try:
                await self._default_ma_scoring(enrichment_result)
            except Exception as e:
                logger.warning(f"Erreur scoring M&A par défaut: {e}")
    
    async def _default_ma_scoring(self, enrichment_result: Dict):
        """Scoring M&A par défaut utilisant le module ma_scoring"""
        try:
            from app.services.ma_scoring import calculate_company_ma_score
            
            company_data = enrichment_result['data']
            
            # Ne calculer que si on a les données minimales
            if company_data.get('siren') and company_data.get('nom_entreprise'):
                logger.info(f"Calcul score M&A pour {company_data.get('nom_entreprise')}")
                
                score_result = await calculate_company_ma_score(company_data)
                
                # Mise à jour des données avec le score
                enrichment_result['data'].update({
                    'ma_score': score_result.final_score,
                    'ma_score_details': {
                        'component_scores': score_result.component_scores,
                        'component_weights': score_result.component_weights,
                        'weighted_scores': score_result.weighted_scores,
                        'data_quality_penalty': score_result.data_quality_penalty,
                        'warnings': score_result.warnings,
                        'calculation_timestamp': score_result.calculation_timestamp
                    },
                    'potentiel_acquisition': score_result.final_score >= 70,
                    'potentiel_cession': score_result.final_score >= 60,
                    'priorite_contact': self._get_priority_from_score(score_result.final_score),
                    'last_scored_at': datetime.now().isoformat()
                })
                
                logger.info(f"Score M&A calculé: {score_result.final_score:.1f}/100")
                
        except Exception as e:
            logger.error(f"Erreur calcul score M&A: {e}")
    
    def _get_priority_from_score(self, score: float) -> str:
        """Détermine la priorité de contact basée sur le score M&A"""
        if score >= 80:
            return "HIGH"
        elif score >= 60:
            return "MEDIUM"
        else:
            return "LOW"
    
    async def _run_export_hooks(self, enrichment_result: Dict):
        """Exécute les hooks d'export"""
        for hook in self._export_hooks:
            try:
                # Hook d'export - sera branché plus tard
                # await hook(enrichment_result)
                logger.debug("Hook d'export appelé (placeholder)")
            except Exception as e:
                logger.warning(f"Erreur dans hook d'export: {e}")
    
    # Méthodes utilitaires
    def _validate_siren(self, siren: str) -> bool:
        """Validation du format SIREN"""
        if not siren or not isinstance(siren, str):
            return False
        
        # Nettoyage
        siren_clean = siren.replace(' ', '').replace('-', '')
        
        # Vérification longueur et format numérique
        if len(siren_clean) != 9 or not siren_clean.isdigit():
            return False
        
        # Algorithme de Luhn pour SIREN
        total = 0
        for i, digit in enumerate(siren_clean):
            n = int(digit)
            if i % 2 == 1:  # Positions paires (indexées à partir de 0)
                n *= 2
                if n > 9:
                    n = n // 10 + n % 10
            total += n
        
        return total % 10 == 0
    
    async def _get_existing_company_data(self, siren: str) -> Optional[Dict]:
        """Récupère les données existantes d'une entreprise"""
        try:
            # Adapter selon votre ORM (SQLAlchemy vs Supabase)
            if hasattr(self.db, 'table'):  # Supabase
                response = self.db.table('cabinets_comptables').select('*').eq('siren', siren).execute()
                return response.data[0] if response.data else None
            else:
                # TODO: Adapter pour SQLAlchemy
                # session = self.db
                # company = session.query(Company).filter(Company.siren == siren).first()
                # return company.to_dict() if company else None
                pass
        except Exception as e:
            logger.error(f"Erreur récupération données existantes pour {siren}: {e}")
        
        return None
    
    def _needs_refresh(self, existing_data: Dict, max_age_days: int = 30) -> bool:
        """Détermine si les données ont besoin d'être rafraîchies"""
        last_enriched = existing_data.get('last_scraped_at')
        if not last_enriched:
            return True
        
        try:
            last_date = datetime.fromisoformat(last_enriched.replace('Z', '+00:00'))
            age = datetime.now() - last_date.replace(tzinfo=None)
            return age.days > max_age_days
        except:
            return True
    
    async def _save_enriched_data(self, enrichment_result: Dict):
        """Sauvegarde les données enrichies"""
        try:
            siren = enrichment_result['siren']
            data = enrichment_result['data']
            
            # Adapter selon votre ORM
            if hasattr(self.db, 'table'):  # Supabase
                # Upsert des données
                self.db.table('cabinets_comptables').upsert(data, on_conflict='siren').execute()
            else:
                # TODO: Adapter pour SQLAlchemy
                pass
                
            logger.info(f"Données sauvegardées pour {siren}")
            
        except Exception as e:
            logger.error(f"Erreur sauvegarde pour {enrichment_result['siren']}: {e}")
    
    async def enrich_companies_batch(
        self, 
        siren_list: List[str], 
        max_concurrent: int = 3,
        progress_callback: Optional[Callable] = None
    ) -> List[Dict]:
        """
        Enrichissement par lot avec concurrence limitée.
        
        Args:
            siren_list: Liste des SIREN à enrichir
            max_concurrent: Nombre max d'enrichissements simultanés
            progress_callback: Fonction de callback pour le suivi de progression
            
        Returns:
            Liste des résultats d'enrichissement
        """
        
        semaphore = asyncio.Semaphore(max_concurrent)
        results = []
        
        async def enrich_with_semaphore(siren: str) -> Dict:
            async with semaphore:
                try:
                    result = await self.enrich_company_full(siren)
                    if progress_callback:
                        await progress_callback(siren, result, len(results) + 1, len(siren_list))
                    return result
                except Exception as e:
                    error_result = {
                        'siren': siren,
                        'status': EnrichmentStatus.FAILED,
                        'error': str(e)
                    }
                    if progress_callback:
                        await progress_callback(siren, error_result, len(results) + 1, len(siren_list))
                    return error_result
        
        # Exécution en parallèle avec semaphore
        tasks = [enrich_with_semaphore(siren) for siren in siren_list]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Gestion des exceptions
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                final_results.append({
                    'siren': siren_list[i],
                    'status': EnrichmentStatus.FAILED,
                    'error': str(result)
                })
            else:
                final_results.append(result)
        
        return final_results
    
    def get_stats(self) -> Dict:
        """Retourne les statistiques d'exécution"""
        return {
            **self.stats,
            'success_rate': (self.stats['total_success'] / max(self.stats['companies_processed'], 1)) * 100,
            'partial_rate': (self.stats['partial_success'] / max(self.stats['companies_processed'], 1)) * 100,
            'failure_rate': (self.stats['total_failed'] / max(self.stats['companies_processed'], 1)) * 100
        }


# Fonction standalone pour compatibilité avec les routes API
async def enrich_company(
    siren: str, 
    db_client,
    force_refresh: bool = False,
    sources: Optional[List[str]] = None
) -> Dict:
    """
    Fonction standalone pour enrichir une entreprise.
    Compatible avec les routes API existantes.
    """
    try:
        # Conversion des sources string vers enum
        source_enums = []
        if sources:
            for source in sources:
                try:
                    source_enums.append(EnrichmentSource(source.lower()))
                except ValueError:
                    logger.warning(f"Source inconnue ignorée: {source}")
        
        async with ScrapingOrchestrator(db_client) as orchestrator:
            result = await orchestrator.enrich_company_full(
                siren=siren,
                force_refresh=force_refresh,
                sources_override=source_enums if source_enums else None
            )
            return result
            
    except Exception as e:
        logger.error(f"Erreur enrichissement standalone pour {siren}: {e}")
        return {
            'siren': siren,
            'status': EnrichmentStatus.FAILED.value,
            'error': str(e)
        }


# Exemple d'utilisation avec hooks
async def example_usage():
    """Exemple d'utilisation de l'orchestrateur"""
    
    # Hook de scoring (placeholder)
    async def ma_scoring_hook(enrichment_result: Dict):
        data = enrichment_result['data']
        # Calcul du score M&A (à implémenter)
        # score = calculate_ma_score(data)
        # data['ma_score'] = score
        logger.info("Scoring M&A appliqué")
    
    # Hook d'export (placeholder) 
    async def export_hook(enrichment_result: Dict):
        # Export vers CSV/Airtable (à implémenter)
        # await export_to_csv(enrichment_result)
        logger.info("Export déclenché")
    
    # Configuration personnalisée
    config = {
        'sources_enabled': {
            EnrichmentSource.PAPPERS: True,
            EnrichmentSource.INFOGREFFE: True,
            EnrichmentSource.SOCIETE: False,  # Désactivé pour test
            EnrichmentSource.KASPR: False
        },
        'validation': {
            'ca_min': 5_000_000,  # Critères plus stricts
            'effectif_min': 50
        }
    }
    
    # Utilisation
    from app.db.supabase import get_supabase_client
    db_client = get_supabase_client()
    
    async with ScrapingOrchestrator(db_client, config) as orchestrator:
        # Ajout des hooks
        orchestrator.add_scoring_hook(ma_scoring_hook)
        orchestrator.add_export_hook(export_hook)
        
        # Enrichissement simple
        result = await orchestrator.enrich_company_full('123456789')
        print(f"Résultat: {result['status']}")
        
        # Enrichissement par lot
        siren_list = ['123456789', '987654321', '456789123']
        results = await orchestrator.enrich_companies_batch(siren_list, max_concurrent=2)
        
        # Statistiques
        stats = orchestrator.get_stats()
        print(f"Taux de succès: {stats['success_rate']:.1f}%")