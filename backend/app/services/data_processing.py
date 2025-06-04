import pandas as pd
import io
import logging
from typing import Dict, List
from datetime import datetime
from fastapi import UploadFile

logger = logging.getLogger(__name__)

# Mapping des colonnes CSV vers les champs de la base
COLUMN_MAPPING = {
    'siren': 'siren',
    'siret': 'siret_siege',
    'siret_siege': 'siret_siege',
    'nom': 'nom_entreprise',
    'nom_entreprise': 'nom_entreprise',
    'denomination': 'nom_entreprise',
    'raison_sociale': 'nom_entreprise',
    'forme_juridique': 'forme_juridique',
    'date_creation': 'date_creation',
    'date_de_creation': 'date_creation',
    'adresse': 'adresse',
    'adresse_complete': 'adresse',
    'email': 'email',
    'mail': 'email',
    'telephone': 'telephone',
    'tel': 'telephone',
    'numero_tva': 'numero_tva',
    'tva': 'numero_tva',
    'ca': 'chiffre_affaires',
    'chiffre_affaires': 'chiffre_affaires',
    'chiffre_d_affaires': 'chiffre_affaires',
    'resultat': 'resultat',
    'resultat_net': 'resultat',
    'effectif': 'effectif',
    'effectifs': 'effectif',
    'capital': 'capital_social',
    'capital_social': 'capital_social',
    'code_naf': 'code_naf',
    'code_ape': 'code_naf',
    'naf': 'code_naf',
    'activite': 'libelle_code_naf',
    'libelle_naf': 'libelle_code_naf',
    'dirigeant': 'dirigeant_principal',
    'dirigeants': 'dirigeant_principal',
    'gerant': 'dirigeant_principal',
    'president': 'dirigeant_principal'
}

async def process_csv_file(file: UploadFile, db_session, update_existing: bool = False) -> Dict:
    """Traite un fichier CSV et importe les données"""
    try:
        # Lire le CSV
        contents = await file.read()
        
        # Essayer différents encodages
        for encoding in ['utf-8-sig', 'utf-8', 'latin1', 'iso-8859-1']:
            try:
                df = pd.read_csv(io.StringIO(contents.decode(encoding)))
                break
            except:
                continue
        else:
            raise ValueError("Impossible de décoder le fichier CSV")
        
        logger.info(f"CSV lu: {len(df)} lignes, colonnes: {list(df.columns)}")
        
        # Normaliser les noms de colonnes
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
        
        # Appliquer le mapping
        df.rename(columns=COLUMN_MAPPING, inplace=True)
        
        # Récupérer les SIREN existants 
        from app.models.company import Company as CompanyModel
        
        # Vérifier que le modèle est bien chargé
        logger.info(f"CompanyModel table: {CompanyModel.__tablename__}")
        
        try:
            existing_companies = db_session.query(CompanyModel.siren).all()
            existing_sirens_companies = set(str(company.siren) for company in existing_companies)
            logger.info(f"Found {len(existing_sirens_companies)} existing companies")
        except Exception as e:
            logger.error(f"Error querying companies: {e}")
            existing_sirens_companies = set()
        
        # Le modèle Company gère toutes les données d'entreprise
        existing_sirens_entreprises = existing_sirens_companies  # Alias pour compatibilité
        logger.info(f"Found {len(existing_sirens_entreprises)} existing companies")
        
        # Préparer les données (modèle Company unifié)
        companies_to_insert = []
        companies_to_update = []
        skipped = 0
        
        for _, row in df.iterrows():
            # Nettoyer et valider les données
            company_data = clean_company_data(row.to_dict())
            
            # Vérifier SIREN
            siren = str(company_data.get('siren', '')).strip()
            if not siren or siren == 'nan' or len(siren) != 9:
                logger.warning(f"SIREN invalide: {siren}")
                skipped += 1
                continue
            
            # Vérifier nom
            nom = str(company_data.get('nom_entreprise', '')).strip()
            if not nom or nom == 'nan':
                logger.warning(f"Nom manquant pour SIREN {siren}")
                skipped += 1
                continue
            
            # Ajouter le statut par défaut
            if 'statut' not in company_data:
                company_data['statut'] = 'à contacter'
            
            # Traitement modèle Company unifié
            if siren in existing_sirens_companies:
                if update_existing:
                    companies_to_update.append(company_data)
                else:
                    skipped += 1
            else:
                companies_to_insert.append(company_data)
        
        # Insertion des nouvelles entreprises
        inserted_count = 0
        if companies_to_insert:
            for i in range(0, len(companies_to_insert), 50):
                batch = companies_to_insert[i:i+50]
                try:
                    for company_data in batch:
                        new_company = CompanyModel(**company_data)
                        db_session.add(new_company)
                    db_session.commit()
                    inserted_count += len(batch)
                    logger.info(f"Batch {i//50 + 1} inséré: {len(batch)} entreprises")
                except Exception as e:
                    db_session.rollback()
                    logger.error(f"Erreur insertion batch: {e}")
        
        updated_count = 0
        if companies_to_update and update_existing:
            for company in companies_to_update:
                try:
                    siren = company['siren']
                    update_data = {k: v for k, v in company.items() if k != 'siren'}
                    existing_company = db_session.query(CompanyModel).filter(CompanyModel.siren == siren).first()
                    if existing_company:
                        for field, value in update_data.items():
                            setattr(existing_company, field, value)
                        db_session.commit()
                        updated_count += 1
                except Exception as e:
                    db_session.rollback()
                    logger.error(f"Erreur mise à jour SIREN {siren}: {e}")
        
        # Calcul des statistiques (compatibilité avec l'ancienne API)
        nouvelles_entreprises = inserted_count
        entreprises_mises_a_jour = updated_count
        entreprises_ignorees = skipped
        
        # Compter les totaux
        try:
            total_count = db_session.query(CompanyModel).count()
        except Exception as e:
            logger.error(f"Error counting companies: {e}")
            total_count = 0
            
        # Count total (alias pour compatibilité)
        total_count_entreprises = total_count
        
        return {
            'success': True,
            'total_rows': total_count,
            'new_companies': inserted_count,
            'updated_companies': updated_count,
            'skipped_companies': skipped,
            'filename': file.filename,
            'entreprises': {
                'total': total_count_entreprises,
                'nouvelles': nouvelles_entreprises,
                'mises_a_jour': entreprises_mises_a_jour,
                'ignorees': entreprises_ignorees
            }
        }
        
    except Exception as e:
        logger.error(f"Erreur traitement CSV: {e}")
        raise

# Fonction supprimée - modèle Company unifié gère toutes les données

def clean_company_data(data: Dict) -> Dict:
    """Nettoie et valide les données d'une entreprise"""
    cleaned = {}
    
    # Champs texte
    text_fields = [
        'siren', 'siret_siege', 'nom_entreprise', 'forme_juridique',
        'adresse', 'ville', 'code_postal', 'email', 'telephone', 'numero_tva', 'code_naf',
        'libelle_code_naf', 'dirigeant_principal', 'statut'
    ]
    
    for field in text_fields:
        value = data.get(field)
        if pd.notna(value) and str(value).strip() and str(value) != 'nan':
            cleaned[field] = str(value).strip()
    
    # Champs numériques
    numeric_fields = [
        'chiffre_affaires', 'resultat', 'effectif', 'capital_social'
    ]
    
    for field in numeric_fields:
        value = clean_numeric_value(data.get(field))
        if value is not None:
            cleaned[field] = value
    
    # Champs date
    date_fields = ['date_creation']
    
    for field in date_fields:
        value = clean_date_value(data.get(field))
        if value:
            cleaned[field] = value
    
    # Ajouter les timestamps
    cleaned['last_scraped_at'] = datetime.now()
    
    return cleaned

def clean_numeric_value(value):
    """Nettoie une valeur numérique"""
    if pd.isna(value) or value == '' or value == 'nan':
        return None
    
    if isinstance(value, (int, float)):
        return float(value)
    
    if isinstance(value, str):
        # Enlever espaces, symboles monétaires
        cleaned = value.strip().replace(' ', '').replace('€', '').replace(',', '.')
        try:
            return float(cleaned) if cleaned else None
        except:
            return None
    
    return None

def clean_date_value(value):
    """Nettoie une valeur date"""
    if pd.isna(value) or value == '' or value == 'nan':
        return None
    
    try:
        if isinstance(value, str):
            # Format DD/MM/YYYY
            if '/' in value:
                parts = value.split('/')
                if len(parts) == 3:
                    return f"{parts[2]}-{parts[1].zfill(2)}-{parts[0].zfill(2)}"
            # Format YYYY-MM-DD
            elif '-' in value and len(value.split('-')) == 3:
                return value
        
        # Essayer avec pandas
        parsed = pd.to_datetime(value, errors='coerce')
        if pd.notna(parsed):
            return parsed.strftime('%Y-%m-%d')
    except:
        pass
    
    return None

async def calculate_prospection_score(company_data: Dict, openai_client=None) -> Dict:
    """Calcule le score de prospection M&A"""
    # Score par défaut si pas d'OpenAI
    if not openai_client:
        return {
            "score_achat": 50,
            "score_vente": 50,
            "score_global": 50,
            "justification": "Score par défaut (OpenAI non configuré)",
            "facteurs_positifs": [],
            "facteurs_negatifs": []
        }
    
    # TODO: Implémenter le calcul avec OpenAI
    # Pour l'instant, calcul basique basé sur les données
    score_achat = 50
    score_vente = 50
    facteurs_positifs = []
    facteurs_negatifs = []
    
    # CA élevé = potentiel acheteur
    ca = company_data.get('chiffre_affaires', 0)
    if ca > 20000000:
        score_achat += 20
        facteurs_positifs.append("CA supérieur à 20M€")
    elif ca < 5000000:
        score_vente += 10
        facteurs_negatifs.append("CA inférieur à 5M€")
    
    # Effectif important = stabilité
    effectif = company_data.get('effectif', 0)
    if effectif > 50:
        score_achat += 10
        facteurs_positifs.append("Effectif important (>50)")
    elif effectif < 10:
        score_vente += 10
        facteurs_negatifs.append("Effectif réduit (<10)")
    
    # Résultat positif = santé financière
    resultat = company_data.get('resultat', 0)
    if resultat > 0:
        facteurs_positifs.append("Résultat positif")
        score_achat += 10
    else:
        facteurs_negatifs.append("Résultat négatif ou nul")
        score_vente += 15
    
    # Score global pondéré
    score_global = (score_achat * 0.4 + score_vente * 0.6)
    
    return {
        "score_achat": min(100, score_achat),
        "score_vente": min(100, score_vente),
        "score_global": score_global,
        "justification": f"Entreprise avec CA de {ca/1000000:.1f}M€ et {effectif} employés",
        "facteurs_positifs": facteurs_positifs,
        "facteurs_negatifs": facteurs_negatifs
    }