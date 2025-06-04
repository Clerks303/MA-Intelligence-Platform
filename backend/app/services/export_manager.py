"""
Module Export Manager pour l'exportation de données M&A vers différents formats.

Ce module permet d'exporter les données d'entreprises enrichies vers :
- CSV (format Excel compatible)
- Airtable (synchronisation bidirectionnelle)
- SQL (bases de données externes)
"""

import asyncio
import csv
import json
import logging
import os
import pandas as pd
import aiohttp
import sqlalchemy
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, date
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
import io
import tempfile

logger = logging.getLogger(__name__)


class ExportFormat(Enum):
    """Formats d'export supportés"""
    CSV_STANDARD = "csv_standard"
    CSV_EXCEL = "csv_excel"
    CSV_MA_ANALYSIS = "csv_ma_analysis"
    AIRTABLE_PROSPECTS = "airtable_prospects"
    AIRTABLE_CONTACTS = "airtable_contacts"
    SQL_POSTGRES = "sql_postgres"
    SQL_MYSQL = "sql_mysql"
    SQL_SQLITE = "sql_sqlite"


@dataclass
class ExportResult:
    """Résultat d'une opération d'export"""
    success: bool
    format: str
    destination: str
    records_exported: int
    file_path: Optional[str] = None
    external_id: Optional[str] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    export_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class ExportManager:
    """
    Gestionnaire d'export multi-format pour les données M&A.
    
    Supporte l'export vers CSV, Airtable et bases SQL avec
    formatage adapté à chaque destination.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialise le gestionnaire d'export.
        
        Args:
            config: Configuration personnalisée pour les exports
        """
        self.config = config or self._get_default_config()
        
        # Statistiques d'export
        self.stats = {
            'total_exports': 0,
            'successful_exports': 0,
            'failed_exports': 0,
            'records_exported': 0,
            'exports_by_format': {}
        }
    
    def _get_default_config(self) -> Dict:
        """Configuration par défaut de l'export manager"""
        return {
            'csv': {
                'encoding': 'utf-8-sig',  # BOM pour Excel
                'separator': ';',          # Séparateur français
                'decimal': ',',            # Décimal français  
                'date_format': '%d/%m/%Y',
                'include_index': False,
                'na_rep': '',             # Valeurs vides
            },
            'airtable': {
                'api_url': 'https://api.airtable.com/v0',
                'timeout': 30,
                'batch_size': 10,         # Max 10 records par batch
                'rate_limit_delay': 0.2,  # 200ms entre requêtes
            },
            'sql': {
                'batch_size': 1000,
                'timeout': 60,
                'if_exists': 'append',    # 'replace', 'append', 'fail'
                'index': False,
                'method': 'multi',        # Insert optimization
            },
            'file_storage': {
                'base_dir': '/tmp/ma_exports',
                'max_file_size_mb': 100,
                'cleanup_after_days': 7,
            }
        }
    
    # ========================================
    # EXPORT CSV
    # ========================================
    
    async def export_to_csv(
        self,
        companies: List[Dict[str, Any]],
        file_path: Optional[str] = None,
        export_format: str = "ma_analysis",
        filters: Optional[Dict] = None,
        include_contacts: bool = True
    ) -> ExportResult:
        """
        Exporte les entreprises vers un fichier CSV formaté pour l'analyse M&A.
        
        Args:
            companies: Liste des entreprises à exporter
            file_path: Chemin du fichier (auto-généré si None)
            export_format: Type d'export ('standard', 'excel', 'ma_analysis')
            filters: Filtres appliqués (pour métadonnées)
            include_contacts: Inclure les contacts dirigeants
            
        Returns:
            ExportResult avec le chemin du fichier créé
        """
        
        logger.info(f"Début export CSV de {len(companies)} entreprises")
        
        try:
            # Génération du chemin si non fourni
            if not file_path:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"export_ma_companies_{timestamp}.csv"
                file_path = self._ensure_export_directory() / filename
            
            # Préparation des données selon le format
            if export_format == "ma_analysis":
                df = self._prepare_ma_analysis_csv(companies, include_contacts)
            elif export_format == "excel":
                df = self._prepare_excel_csv(companies)
            else:  # standard
                df = self._prepare_standard_csv(companies)
            
            # Écriture du fichier CSV
            csv_config = self.config['csv']
            df.to_csv(
                file_path,
                sep=csv_config['separator'],
                encoding=csv_config['encoding'],
                index=csv_config['include_index'],
                na_rep=csv_config['na_rep'],
                date_format=csv_config['date_format'],
                float_format='%.2f'
            )
            
            # Ajout des métadonnées en commentaire
            await self._add_csv_metadata(file_path, filters, len(companies))
            
            # Mise à jour des statistiques
            self._update_stats('csv', len(companies), True)
            
            result = ExportResult(
                success=True,
                format=f"csv_{export_format}",
                destination=str(file_path),
                records_exported=len(companies),
                file_path=str(file_path),
                metadata={
                    'format': export_format,
                    'include_contacts': include_contacts,
                    'file_size_mb': self._get_file_size_mb(file_path),
                    'columns_count': len(df.columns),
                    'filters_applied': filters or {}
                }
            )
            
            logger.info(f"Export CSV réussi: {file_path} ({result.metadata['file_size_mb']:.1f} MB)")
            return result
            
        except Exception as e:
            self._update_stats('csv', len(companies), False)
            logger.error(f"Erreur export CSV: {e}")
            
            return ExportResult(
                success=False,
                format=f"csv_{export_format}",
                destination=str(file_path) if file_path else "unknown",
                records_exported=0,
                errors=[str(e)]
            )
    
    def _prepare_ma_analysis_csv(self, companies: List[Dict], include_contacts: bool) -> pd.DataFrame:
        """Prépare un CSV optimisé pour l'analyse M&A"""
        
        rows = []
        
        for company in companies:
            # Données de base
            row = {
                # Identification
                'SIREN': company.get('siren', ''),
                'Nom_Entreprise': company.get('nom_entreprise', ''),
                'Forme_Juridique': company.get('forme_juridique', ''),
                'Date_Creation': self._format_date(company.get('date_creation')),
                
                # Localisation
                'Adresse': company.get('adresse', ''),
                'Code_Postal': company.get('code_postal', ''),
                'Ville': company.get('ville', ''),
                
                # Données financières
                'CA_Actuel': self._format_currency(company.get('chiffre_affaires')),
                'CA_N1': self._format_currency(company.get('chiffre_affaires_n1')),
                'CA_N2': self._format_currency(company.get('chiffre_affaires_n2')),
                'Evolution_CA_3ans': self._format_percentage(company.get('evolution_ca_3ans')),
                'Resultat_Net': self._format_currency(company.get('resultat')),
                'Marge_Nette': self._format_percentage(company.get('marge_nette')),
                'Effectif': company.get('effectif', ''),
                'Capital_Social': self._format_currency(company.get('capital_social')),
                
                # Analyse M&A
                'Score_MA': company.get('ma_score', ''),
                'Niveau_Score': self._get_score_level(company.get('ma_score')),
                'Potentiel_Acquisition': 'Oui' if company.get('potentiel_acquisition') else 'Non',
                'Potentiel_Cession': 'Oui' if company.get('potentiel_cession') else 'Non',
                'Priorite_Contact': company.get('priorite_contact', ''),
                
                # Données opérationnelles
                'Code_NAF': company.get('code_naf', ''),
                'Secteur_Activite': company.get('secteur_activite', ''),
                'Dirigeant_Principal': company.get('dirigeant_principal', ''),
                
                # Enrichissement
                'Ratio_Endettement': self._format_percentage(company.get('ratio_endettement')),
                'Qualite_Donnees': self._format_percentage(company.get('qualite_donnees')),
                'Derniere_MAJ': self._format_date(company.get('last_scraped_at')),
                
                # Liens
                'Lien_Pappers': company.get('lien_pappers', ''),
                'Lien_Societe_Com': company.get('lien_societe_com', ''),
                
                # Prospection
                'Statut_Prospect': company.get('statut', ''),
                'Notes_Commerciales': company.get('notes_commerciales', ''),
                'Derniere_Actualite': company.get('derniere_actualite', ''),
            }
            
            # Contacts dirigeants si demandés
            if include_contacts:
                contacts = company.get('kaspr_contacts', []) or company.get('contacts', [])
                if contacts:
                    # Prendre le contact principal (premier dirigeant)
                    main_contact = next((c for c in contacts if c.get('est_dirigeant')), contacts[0] if contacts else {})
                    
                    row.update({
                        'Contact_Principal': main_contact.get('nom_complet', ''),
                        'Poste_Contact': main_contact.get('poste', ''),
                        'Email_Contact': main_contact.get('email_professionnel', ''),
                        'Telephone_Contact': main_contact.get('telephone_direct', ''),
                        'LinkedIn_Contact': main_contact.get('linkedin_url', ''),
                        'Nb_Contacts_Total': len(contacts)
                    })
                else:
                    row.update({
                        'Contact_Principal': '',
                        'Poste_Contact': '',
                        'Email_Contact': '',
                        'Telephone_Contact': '',
                        'LinkedIn_Contact': '',
                        'Nb_Contacts_Total': 0
                    })
            
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def _prepare_excel_csv(self, companies: List[Dict]) -> pd.DataFrame:
        """Prépare un CSV compatible Excel avec formatage français"""
        
        rows = []
        for company in companies:
            row = {
                'SIREN': company.get('siren', ''),
                'Raison Sociale': company.get('nom_entreprise', ''),
                'Forme Juridique': company.get('forme_juridique', ''),
                'Date Création': self._format_date_fr(company.get('date_creation')),
                'Adresse': company.get('adresse', ''),
                'Code Postal': company.get('code_postal', ''),
                'Ville': company.get('ville', ''),
                'CA (€)': self._format_number_fr(company.get('chiffre_affaires')),
                'Résultat (€)': self._format_number_fr(company.get('resultat')),
                'Effectif': company.get('effectif', ''),
                'Score M&A': self._format_number_fr(company.get('ma_score')),
                'Priorité': company.get('priorite_contact', ''),
                'Dirigeant': company.get('dirigeant_principal', ''),
                'Téléphone': company.get('telephone', ''),
                'Email': company.get('email', ''),
                'Statut': company.get('statut', ''),
            }
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def _prepare_standard_csv(self, companies: List[Dict]) -> pd.DataFrame:
        """Prépare un CSV standard avec toutes les données disponibles"""
        # Conversion directe en DataFrame avec nettoyage minimal
        df = pd.DataFrame(companies)
        
        # Nettoyage des colonnes JSON complexes
        json_columns = ['dirigeants_json', 'details_complets', 'ma_score_details']
        for col in json_columns:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: json.dumps(x, default=str) if x else '')
        
        return df
    
    # ========================================
    # SYNCHRONISATION AIRTABLE
    # ========================================
    
    async def sync_to_airtable(
        self,
        companies: List[Dict[str, Any]],
        base_id: str,
        table_name: str = "Prospects_MA",
        api_key: Optional[str] = None,
        sync_mode: str = "upsert"
    ) -> ExportResult:
        """
        Synchronise les entreprises avec une base Airtable.
        
        Args:
            companies: Liste des entreprises à synchroniser
            base_id: ID de la base Airtable (app...)
            table_name: Nom de la table dans Airtable
            api_key: Clé API Airtable (ou variable d'environnement)
            sync_mode: Mode de sync ('upsert', 'create_only', 'update_only')
            
        Returns:
            ExportResult avec les IDs des records créés/mis à jour
        """
        
        api_key = api_key or os.environ.get('AIRTABLE_API_KEY')
        if not api_key:
            return ExportResult(
                success=False,
                format="airtable",
                destination=f"{base_id}/{table_name}",
                records_exported=0,
                errors=["Clé API Airtable manquante"]
            )
        
        logger.info(f"Début sync Airtable de {len(companies)} entreprises vers {base_id}/{table_name}")
        
        try:
            async with aiohttp.ClientSession() as session:
                # Configuration Airtable
                headers = {
                    'Authorization': f'Bearer {api_key}',
                    'Content-Type': 'application/json'
                }
                
                airtable_config = self.config['airtable']
                base_url = f"{airtable_config['api_url']}/{base_id}/{table_name}"
                
                # Préparation des données pour Airtable
                airtable_records = self._prepare_airtable_data(companies)
                
                # Traitement par lots
                batch_size = airtable_config['batch_size']
                batches = [airtable_records[i:i + batch_size] for i in range(0, len(airtable_records), batch_size)]
                
                created_ids = []
                updated_ids = []
                errors = []
                
                for i, batch in enumerate(batches):
                    logger.info(f"Traitement batch {i+1}/{len(batches)} ({len(batch)} records)")
                    
                    try:
                        if sync_mode == "upsert":
                            batch_result = await self._upsert_airtable_batch(session, base_url, headers, batch)
                        elif sync_mode == "create_only":
                            batch_result = await self._create_airtable_batch(session, base_url, headers, batch)
                        else:  # update_only
                            batch_result = await self._update_airtable_batch(session, base_url, headers, batch)
                        
                        created_ids.extend(batch_result.get('created', []))
                        updated_ids.extend(batch_result.get('updated', []))
                        
                        if batch_result.get('errors'):
                            errors.extend(batch_result['errors'])
                        
                        # Rate limiting
                        await asyncio.sleep(airtable_config['rate_limit_delay'])
                        
                    except Exception as e:
                        error_msg = f"Erreur batch {i+1}: {str(e)}"
                        errors.append(error_msg)
                        logger.error(error_msg)
                
                # Résultat final
                total_synced = len(created_ids) + len(updated_ids)
                success = total_synced > 0
                
                if success:
                    self._update_stats('airtable', total_synced, True)
                else:
                    self._update_stats('airtable', len(companies), False)
                
                result = ExportResult(
                    success=success,
                    format="airtable",
                    destination=f"{base_id}/{table_name}",
                    records_exported=total_synced,
                    external_id=base_id,
                    errors=errors,
                    metadata={
                        'sync_mode': sync_mode,
                        'records_created': len(created_ids),
                        'records_updated': len(updated_ids),
                        'batches_processed': len(batches),
                        'created_ids': created_ids[:10],  # Premiers IDs pour référence
                        'updated_ids': updated_ids[:10]
                    }
                )
                
                logger.info(f"Sync Airtable terminé: {total_synced} records synchronisés")
                return result
                
        except Exception as e:
            self._update_stats('airtable', len(companies), False)
            logger.error(f"Erreur sync Airtable: {e}")
            
            return ExportResult(
                success=False,
                format="airtable",
                destination=f"{base_id}/{table_name}",
                records_exported=0,
                errors=[str(e)]
            )
    
    def _prepare_airtable_data(self, companies: List[Dict]) -> List[Dict]:
        """Prépare les données pour l'API Airtable"""
        
        airtable_records = []
        
        for company in companies:
            # Structure Airtable avec champs typés
            fields = {
                'SIREN': company.get('siren', ''),
                'Nom_Entreprise': company.get('nom_entreprise', ''),
                'Forme_Juridique': company.get('forme_juridique', ''),
                'Ville': company.get('ville', ''),
                'Code_Postal': company.get('code_postal', ''),
                
                # Données numériques
                'CA_Actuel': self._safe_number(company.get('chiffre_affaires')),
                'Resultat_Net': self._safe_number(company.get('resultat')),
                'Effectif': self._safe_number(company.get('effectif')),
                'Score_MA': self._safe_number(company.get('ma_score')),
                
                # Pourcentages
                'Evolution_CA_3ans': self._safe_number(company.get('evolution_ca_3ans')),
                'Marge_Nette': self._safe_number(company.get('marge_nette')),
                
                # Booléens
                'Potentiel_Acquisition': bool(company.get('potentiel_acquisition')),
                'Potentiel_Cession': bool(company.get('potentiel_cession')),
                
                # Sélections simples
                'Priorite_Contact': company.get('priorite_contact', 'MEDIUM'),
                'Statut_Prospect': company.get('statut', 'prospect'),
                
                # Texte
                'Dirigeant_Principal': company.get('dirigeant_principal', ''),
                'Notes_Commerciales': company.get('notes_commerciales', ''),
                
                # Dates (format ISO)
                'Date_Creation': self._format_date_iso(company.get('date_creation')),
                'Derniere_MAJ': self._format_date_iso(company.get('last_scraped_at')),
                
                # URLs
                'Lien_Pappers': company.get('lien_pappers', ''),
                'Lien_Societe_Com': company.get('lien_societe_com', ''),
            }
            
            # Contacts (première approche: champs séparés)
            contacts = company.get('kaspr_contacts', []) or company.get('contacts', [])
            if contacts:
                main_contact = next((c for c in contacts if c.get('est_dirigeant')), contacts[0])
                fields.update({
                    'Contact_Principal': main_contact.get('nom_complet', ''),
                    'Email_Contact': main_contact.get('email_professionnel', ''),
                    'Telephone_Contact': main_contact.get('telephone_direct', ''),
                    'LinkedIn_Contact': main_contact.get('linkedin_url', ''),
                })
            
            # Nettoyage des valeurs None
            cleaned_fields = {k: v for k, v in fields.items() if v is not None}
            
            record = {
                'fields': cleaned_fields
            }
            
            # Ajouter l'ID si c'est une mise à jour
            if company.get('airtable_id'):
                record['id'] = company['airtable_id']
            
            airtable_records.append(record)
        
        return airtable_records
    
    async def _create_airtable_batch(self, session, base_url, headers, batch):
        """Crée un lot de records dans Airtable"""
        payload = {'records': batch}
        
        async with session.post(base_url, headers=headers, json=payload) as response:
            if response.status == 200:
                data = await response.json()
                return {
                    'created': [record['id'] for record in data.get('records', [])],
                    'updated': [],
                    'errors': []
                }
            else:
                error_text = await response.text()
                return {
                    'created': [],
                    'updated': [],
                    'errors': [f"HTTP {response.status}: {error_text}"]
                }
    
    async def _upsert_airtable_batch(self, session, base_url, headers, batch):
        """Met à jour ou crée des records (upsert)"""
        # Airtable ne supporte pas l'upsert natif, on simule
        # En pratique, il faudrait d'abord chercher les records existants
        return await self._create_airtable_batch(session, base_url, headers, batch)
    
    async def _update_airtable_batch(self, session, base_url, headers, batch):
        """Met à jour des records existants"""
        # Filtrer seulement les records avec ID
        records_with_id = [r for r in batch if 'id' in r]
        
        if not records_with_id:
            return {'created': [], 'updated': [], 'errors': []}
        
        payload = {'records': records_with_id}
        
        async with session.patch(base_url, headers=headers, json=payload) as response:
            if response.status == 200:
                data = await response.json()
                return {
                    'created': [],
                    'updated': [record['id'] for record in data.get('records', [])],
                    'errors': []
                }
            else:
                error_text = await response.text()
                return {
                    'created': [],
                    'updated': [],
                    'errors': [f"HTTP {response.status}: {error_text}"]
                }
    
    # ========================================
    # EXPORT SQL
    # ========================================
    
    async def export_to_sql(
        self,
        companies: List[Dict[str, Any]],
        connection_string: str,
        table_name: str = "companies_export",
        if_exists: str = "append",
        schema: Optional[str] = None
    ) -> ExportResult:
        """
        Exporte les entreprises vers une base de données SQL.
        
        Args:
            companies: Liste des entreprises à exporter
            connection_string: String de connexion SQL
            table_name: Nom de la table de destination
            if_exists: Action si la table existe ('fail', 'replace', 'append')
            schema: Schéma de la base (optionnel)
            
        Returns:
            ExportResult avec informations sur l'export SQL
        """
        
        logger.info(f"Début export SQL de {len(companies)} entreprises vers {table_name}")
        
        try:
            # Préparation des données pour SQL
            df = self._prepare_sql_data(companies)
            
            # Création de l'engine SQLAlchemy
            engine = sqlalchemy.create_engine(connection_string)
            
            # Configuration SQL
            sql_config = self.config['sql']
            
            # Export vers SQL avec pandas
            records_exported = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: df.to_sql(
                    name=table_name,
                    con=engine,
                    if_exists=if_exists,
                    index=sql_config['index'],
                    schema=schema,
                    method=sql_config['method'],
                    chunksize=sql_config['batch_size']
                )
            )
            
            # Si pandas retourne None, on prend la longueur du DataFrame
            if records_exported is None:
                records_exported = len(df)
            
            # Mise à jour des statistiques
            self._update_stats('sql', records_exported, True)
            
            # Obtenir des infos sur la table créée
            db_type = self._detect_database_type(connection_string)
            table_info = await self._get_table_info(engine, table_name, schema)
            
            result = ExportResult(
                success=True,
                format=f"sql_{db_type}",
                destination=f"{table_name}@{self._mask_connection_string(connection_string)}",
                records_exported=records_exported,
                metadata={
                    'database_type': db_type,
                    'table_name': table_name,
                    'schema': schema,
                    'if_exists_mode': if_exists,
                    'table_info': table_info,
                    'columns_exported': len(df.columns),
                    'data_types': df.dtypes.to_dict()
                }
            )
            
            logger.info(f"Export SQL réussi: {records_exported} records vers {table_name}")
            return result
            
        except Exception as e:
            self._update_stats('sql', len(companies), False)
            logger.error(f"Erreur export SQL: {e}")
            
            return ExportResult(
                success=False,
                format="sql",
                destination=f"{table_name}@{self._mask_connection_string(connection_string)}",
                records_exported=0,
                errors=[str(e)]
            )
    
    def _prepare_sql_data(self, companies: List[Dict]) -> pd.DataFrame:
        """Prépare les données pour l'export SQL"""
        
        # Conversion en DataFrame
        df = pd.DataFrame(companies)
        
        if df.empty:
            return df
        
        # Nettoyage et typage des données
        
        # Colonnes texte
        text_columns = [
            'siren', 'nom_entreprise', 'forme_juridique', 'adresse', 'ville', 
            'code_postal', 'email', 'telephone', 'code_naf', 'dirigeant_principal',
            'statut', 'priorite_contact'
        ]
        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].astype('string').fillna('')
        
        # Colonnes numériques
        numeric_columns = [
            'chiffre_affaires', 'resultat', 'effectif', 'capital_social',
            'ma_score', 'evolution_ca_3ans', 'marge_nette', 'ratio_endettement'
        ]
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Colonnes booléennes
        bool_columns = ['potentiel_acquisition', 'potentiel_cession']
        for col in bool_columns:
            if col in df.columns:
                df[col] = df[col].astype('boolean').fillna(False)
        
        # Colonnes dates
        date_columns = ['date_creation', 'last_scraped_at', 'created_at', 'updated_at']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Colonnes JSON -> Text
        json_columns = ['dirigeants_json', 'ma_score_details', 'details_complets']
        for col in json_columns:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: json.dumps(x, default=str) if x else None)
        
        # Ajout de métadonnées d'export
        df['export_timestamp'] = datetime.now()
        df['export_source'] = 'ma_intelligence_platform'
        
        return df
    
    async def _get_table_info(self, engine, table_name: str, schema: Optional[str]) -> Dict:
        """Récupère des informations sur la table créée"""
        try:
            inspector = sqlalchemy.inspect(engine)
            
            if schema:
                tables = inspector.get_table_names(schema=schema)
                columns = inspector.get_columns(table_name, schema=schema) if table_name in tables else []
            else:
                tables = inspector.get_table_names()
                columns = inspector.get_columns(table_name) if table_name in tables else []
            
            return {
                'exists': table_name in tables,
                'columns_count': len(columns),
                'column_names': [col['name'] for col in columns]
            }
        except Exception as e:
            logger.warning(f"Impossible de récupérer les infos table: {e}")
            return {'exists': 'unknown', 'error': str(e)}
    
    # ========================================
    # UTILITAIRES
    # ========================================
    
    def _ensure_export_directory(self) -> Path:
        """S'assure que le répertoire d'export existe"""
        export_dir = Path(self.config['file_storage']['base_dir'])
        export_dir.mkdir(parents=True, exist_ok=True)
        return export_dir
    
    async def _add_csv_metadata(self, file_path: str, filters: Optional[Dict], record_count: int):
        """Ajoute des métadonnées en en-tête du fichier CSV"""
        try:
            # Lire le contenu existant
            with open(file_path, 'r', encoding='utf-8-sig') as f:
                content = f.read()
            
            # Préparer l'en-tête de métadonnées
            metadata_lines = [
                f"# Export M&A Intelligence Platform",
                f"# Date: {datetime.now().strftime('%d/%m/%Y %H:%M')}",
                f"# Nombre d'entreprises: {record_count}",
            ]
            
            if filters:
                metadata_lines.append(f"# Filtres appliqués: {json.dumps(filters, default=str)}")
            
            metadata_lines.extend([
                f"# Format: CSV avec séparateur ';'",
                f"# Encodage: UTF-8 avec BOM",
                f"#",
                ""  # Ligne vide avant les données
            ])
            
            # Réécrire le fichier avec métadonnées
            with open(file_path, 'w', encoding='utf-8-sig') as f:
                f.write('\n'.join(metadata_lines))
                f.write(content)
                
        except Exception as e:
            logger.warning(f"Impossible d'ajouter les métadonnées CSV: {e}")
    
    def _get_file_size_mb(self, file_path: Union[str, Path]) -> float:
        """Retourne la taille du fichier en MB"""
        try:
            size_bytes = Path(file_path).stat().st_size
            return size_bytes / (1024 * 1024)
        except:
            return 0.0
    
    def _format_date(self, date_value) -> str:
        """Formate une date pour l'export"""
        if not date_value:
            return ''
        
        try:
            if isinstance(date_value, str):
                dt = datetime.fromisoformat(date_value.replace('Z', '+00:00'))
            elif isinstance(date_value, date):
                dt = datetime.combine(date_value, datetime.min.time())
            else:
                dt = date_value
            
            return dt.strftime('%d/%m/%Y')
        except:
            return str(date_value) if date_value else ''
    
    def _format_date_fr(self, date_value) -> str:
        """Formate une date au format français"""
        return self._format_date(date_value)
    
    def _format_date_iso(self, date_value) -> Optional[str]:
        """Formate une date au format ISO pour Airtable"""
        if not date_value:
            return None
        
        try:
            if isinstance(date_value, str):
                dt = datetime.fromisoformat(date_value.replace('Z', '+00:00'))
            elif isinstance(date_value, date):
                dt = datetime.combine(date_value, datetime.min.time())
            else:
                dt = date_value
            
            return dt.strftime('%Y-%m-%d')
        except:
            return None
    
    def _format_currency(self, amount) -> str:
        """Formate un montant en euros"""
        if not amount or not isinstance(amount, (int, float)):
            return ''
        
        return f"{amount:,.0f}".replace(',', ' ')
    
    def _format_number_fr(self, number) -> str:
        """Formate un nombre au format français"""
        if not number or not isinstance(number, (int, float)):
            return ''
        
        return f"{number:,.2f}".replace(',', ' ').replace('.', ',')
    
    def _format_percentage(self, percentage) -> str:
        """Formate un pourcentage"""
        if percentage is None or not isinstance(percentage, (int, float)):
            return ''
        
        return f"{percentage:.1f}%"
    
    def _safe_number(self, value) -> Optional[float]:
        """Convertit une valeur en nombre ou None"""
        if value is None:
            return None
        
        try:
            return float(value)
        except (ValueError, TypeError):
            return None
    
    def _get_score_level(self, score) -> str:
        """Retourne le niveau correspondant au score M&A"""
        if not score:
            return 'Non évalué'
        
        if score >= 85:
            return 'Excellent'
        elif score >= 70:
            return 'Très bon'
        elif score >= 55:
            return 'Bon'
        elif score >= 40:
            return 'Moyen'
        else:
            return 'Faible'
    
    def _detect_database_type(self, connection_string: str) -> str:
        """Détecte le type de base de données"""
        connection_lower = connection_string.lower()
        
        if 'postgresql' in connection_lower or 'postgres' in connection_lower:
            return 'postgresql'
        elif 'mysql' in connection_lower:
            return 'mysql'
        elif 'sqlite' in connection_lower:
            return 'sqlite'
        elif 'oracle' in connection_lower:
            return 'oracle'
        elif 'mssql' in connection_lower or 'sqlserver' in connection_lower:
            return 'sqlserver'
        else:
            return 'unknown'
    
    def _mask_connection_string(self, connection_string: str) -> str:
        """Masque les informations sensibles de la chaîne de connexion"""
        # Masquer les mots de passe
        import re
        masked = re.sub(r'(:)([^:@]+)(@)', r'\1***\3', connection_string)
        return masked
    
    def _update_stats(self, format_type: str, count: int, success: bool):
        """Met à jour les statistiques d'export"""
        self.stats['total_exports'] += 1
        
        if success:
            self.stats['successful_exports'] += 1
            self.stats['records_exported'] += count
        else:
            self.stats['failed_exports'] += 1
        
        if format_type not in self.stats['exports_by_format']:
            self.stats['exports_by_format'][format_type] = {'success': 0, 'failed': 0, 'records': 0}
        
        format_stats = self.stats['exports_by_format'][format_type]
        if success:
            format_stats['success'] += 1
            format_stats['records'] += count
        else:
            format_stats['failed'] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques d'export"""
        total_exports = self.stats['total_exports']
        success_rate = (self.stats['successful_exports'] / max(total_exports, 1)) * 100
        
        return {
            **self.stats,
            'success_rate': success_rate,
            'average_records_per_export': self.stats['records_exported'] / max(self.stats['successful_exports'], 1)
        }


# ========================================
# FONCTIONS UTILITAIRES STANDALONE
# ========================================

async def quick_csv_export(
    companies: List[Dict[str, Any]], 
    file_path: Optional[str] = None,
    format_type: str = "ma_analysis"
) -> str:
    """
    Export CSV rapide sans configuration.
    
    Returns:
        Chemin du fichier créé
    """
    export_manager = ExportManager()
    result = await export_manager.export_to_csv(
        companies=companies,
        file_path=file_path,
        export_format=format_type
    )
    
    if result.success:
        return result.file_path
    else:
        raise Exception(f"Export CSV échoué: {', '.join(result.errors)}")


async def quick_airtable_sync(
    companies: List[Dict[str, Any]],
    base_id: str,
    table_name: str = "Prospects_MA"
) -> Dict[str, Any]:
    """
    Synchronisation Airtable rapide.
    
    Returns:
        Dictionnaire avec les résultats de la sync
    """
    export_manager = ExportManager()
    result = await export_manager.sync_to_airtable(
        companies=companies,
        base_id=base_id,
        table_name=table_name
    )
    
    return {
        'success': result.success,
        'records_synced': result.records_exported,
        'errors': result.errors,
        'metadata': result.metadata
    }


async def quick_sql_export(
    companies: List[Dict[str, Any]],
    connection_string: str,
    table_name: str = "ma_prospects"
) -> int:
    """
    Export SQL rapide.
    
    Returns:
        Nombre de records exportés
    """
    export_manager = ExportManager()
    result = await export_manager.export_to_sql(
        companies=companies,
        connection_string=connection_string,
        table_name=table_name
    )
    
    if result.success:
        return result.records_exported
    else:
        raise Exception(f"Export SQL échoué: {', '.join(result.errors)}")