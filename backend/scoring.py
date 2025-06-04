#!/usr/bin/env python3
"""
Script de Scoring ML - M&A Intelligence Platform
Se connecte à Supabase, récupère les données, calcule les scores ML et met à jour la base

Usage:
    python scoring.py                    # Score toutes les entreprises
    python scoring.py --batch-size 500  # Taille batch personnalisée
    python scoring.py --company-ids 1,2,3  # Entreprises spécifiques
    python scoring.py --force-retrain   # Force le ré-entraînement
"""

import os
import sys
import logging
import argparse
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# Database
from supabase import create_client, Client
from sqlalchemy import create_engine, text
import psycopg2

# Configuration
from dotenv import load_dotenv
load_dotenv()

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('scoring.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class MLScoringEngine:
    """Engine principal pour le calcul des scores ML"""
    
    def __init__(self):
        self.supabase_url = os.getenv('SUPABASE_URL')
        self.supabase_key = os.getenv('SUPABASE_KEY')
        self.database_url = os.getenv('DATABASE_URL')  # Alternative PostgreSQL direct
        
        if not self.supabase_url or not self.supabase_key:
            raise ValueError("SUPABASE_URL et SUPABASE_KEY requis dans .env")
        
        # Initialisation clients
        self.supabase: Client = create_client(self.supabase_url, self.supabase_key)
        
        # Modèles ML
        self.models = {}
        self.feature_columns = []
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
        logger.info("🚀 MLScoringEngine initialisé")
    
    async def run_scoring(self, batch_size: int = 1000, company_ids: List[int] = None, force_retrain: bool = False):
        """Lance le processus complet de scoring"""
        try:
            logger.info("📊 Début du processus de scoring ML")
            
            # 1. Récupération des données
            logger.info("📥 Récupération des données des cabinets...")
            raw_data = await self.fetch_companies_data(company_ids)
            logger.info(f"✅ {len(raw_data)} entreprises récupérées")
            
            if len(raw_data) == 0:
                logger.warning("⚠️ Aucune donnée trouvée")
                return
            
            # 2. Préparation des features
            logger.info("🔧 Préparation des features...")
            features_df = await self.prepare_features(raw_data)
            logger.info(f"✅ {len(features_df.columns)} features préparées")
            
            # 3. Chargement ou entraînement des modèles
            if force_retrain or not await self.load_models():
                logger.info("🧠 Entraînement des modèles ML...")
                await self.train_models(features_df)
            else:
                logger.info("📦 Modèles chargés depuis les fichiers")
            
            # 4. Calcul des scores
            logger.info("🎯 Calcul des scores ML...")
            scores = await self.calculate_scores(features_df)
            logger.info(f"✅ {len(scores)} scores calculés")
            
            # 5. Sauvegarde en base
            logger.info("💾 Sauvegarde des scores...")
            await self.save_scores(scores)
            logger.info("✅ Scores sauvegardés avec succès")
            
            # 6. Statistiques finales
            await self.print_statistics(scores)
            
        except Exception as e:
            logger.error(f"❌ Erreur lors du scoring: {e}")
            raise
    
    async def fetch_companies_data(self, company_ids: List[int] = None) -> List[Dict]:
        """Récupère les données brutes des cabinets depuis Supabase"""
        try:
            query = self.supabase.table('companies').select('*')
            
            # Filtrer par IDs si spécifiés
            if company_ids:
                query = query.in_('id', company_ids)
            
            # Filtrer les entreprises avec données suffisantes
            query = query.not_.is_('chiffre_affaires', 'null')\
                        .not_.is_('nom_entreprise', 'null')
            
            response = query.execute()
            
            if not response.data:
                logger.warning("Aucune entreprise trouvée avec les critères")
                return []
            
            return response.data
            
        except Exception as e:
            logger.error(f"Erreur récupération données: {e}")
            raise
    
    async def prepare_features(self, raw_data: List[Dict]) -> pd.DataFrame:
        """Prépare les features pour l'entraînement/prédiction"""
        df = pd.DataFrame(raw_data)
        
        # Features de base
        features = df.copy()
        
        # Nettoyage des données
        features = await self._clean_data(features)
        
        # Feature engineering
        features = await self._engineer_features(features)
        
        # Encodage des variables catégorielles
        features = await self._encode_categorical(features)
        
        # Sélection des colonnes finales
        self.feature_columns = [
            'chiffre_affaires', 'benefice_net', 'fonds_propres', 'dette_totale',
            'effectif', 'age_entreprise', 'croissance_ca_1an', 'croissance_ca_3ans',
            'ratio_endettement', 'rentabilite_nette', 'rentabilite_brute',
            'ratio_liquidite', 'rotation_actifs', 'secteur_encoded', 'region_encoded',
            'taille_entreprise', 'stabilite_financiere'
        ]
        
        # Garder seulement les colonnes disponibles
        available_columns = [col for col in self.feature_columns if col in features.columns]
        features_final = features[['id'] + available_columns].copy()
        
        logger.info(f"Features utilisées: {available_columns}")
        return features_final
    
    async def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Nettoyage des données"""
        # Conversion des types
        numeric_columns = ['chiffre_affaires', 'benefice_net', 'fonds_propres', 'dette_totale', 'effectif']
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Gestion des valeurs manquantes
        df['chiffre_affaires'] = df['chiffre_affaires'].fillna(0)
        df['benefice_net'] = df['benefice_net'].fillna(0)
        df['effectif'] = df['effectif'].fillna(1)
        df['secteur'] = df['secteur'].fillna('Autres')
        df['region'] = df['region'].fillna('Non renseigné')
        
        return df
    
    async def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Feature engineering avancé"""
        
        # Âge de l'entreprise
        if 'date_creation' in df.columns:
            df['date_creation'] = pd.to_datetime(df['date_creation'], errors='coerce')
            df['age_entreprise'] = (datetime.now() - df['date_creation']).dt.days / 365.25
        else:
            df['age_entreprise'] = 10  # Valeur par défaut
        
        # Ratios financiers
        df['ratio_endettement'] = np.where(
            df['fonds_propres'] > 0,
            df['dette_totale'] / df['fonds_propres'],
            0
        )
        
        df['rentabilite_nette'] = np.where(
            df['chiffre_affaires'] > 0,
            (df['benefice_net'] / df['chiffre_affaires']) * 100,
            0
        )
        
        df['rentabilite_brute'] = np.where(
            df['chiffre_affaires'] > 0,
            ((df['chiffre_affaires'] - df.get('charges', 0)) / df['chiffre_affaires']) * 100,
            0
        )
        
        # Taille d'entreprise
        df['taille_entreprise'] = pd.cut(
            df['effectif'], 
            bins=[0, 10, 50, 250, float('inf')], 
            labels=[1, 2, 3, 4]
        ).astype(float)
        
        # Croissance (simulée si pas disponible)
        df['croissance_ca_1an'] = np.random.normal(5, 15, len(df))  # Simulation
        df['croissance_ca_3ans'] = np.random.normal(8, 20, len(df))  # Simulation
        
        # Stabilité financière (score composite)
        df['stabilite_financiere'] = (
            (df['age_entreprise'] / 20) * 30 +  # Âge (max 30 points)
            np.where(df['ratio_endettement'] < 0.5, 25, 0) +  # Endettement faible
            np.where(df['rentabilite_nette'] > 5, 25, 0) +  # Rentabilité positive
            np.where(df['effectif'] > 10, 20, 0)  # Taille critique
        ).clip(0, 100)
        
        return df
    
    async def _encode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encodage des variables catégorielles"""
        
        # Secteur
        if 'secteur' in df.columns:
            if 'secteur' not in self.label_encoders:
                self.label_encoders['secteur'] = LabelEncoder()
                df['secteur_encoded'] = self.label_encoders['secteur'].fit_transform(df['secteur'].astype(str))
            else:
                # Gestion des nouvelles valeurs
                known_labels = set(self.label_encoders['secteur'].classes_)
                df['secteur_temp'] = df['secteur'].astype(str)
                unknown_mask = ~df['secteur_temp'].isin(known_labels)
                df.loc[unknown_mask, 'secteur_temp'] = 'Autres'
                df['secteur_encoded'] = self.label_encoders['secteur'].transform(df['secteur_temp'])
        
        # Région
        if 'region' in df.columns:
            if 'region' not in self.label_encoders:
                self.label_encoders['region'] = LabelEncoder()
                df['region_encoded'] = self.label_encoders['region'].fit_transform(df['region'].astype(str))
            else:
                known_labels = set(self.label_encoders['region'].classes_)
                df['region_temp'] = df['region'].astype(str)
                unknown_mask = ~df['region_temp'].isin(known_labels)
                df.loc[unknown_mask, 'region_temp'] = 'Non renseigné'
                df['region_encoded'] = self.label_encoders['region'].transform(df['region_temp'])
        
        return df
    
    async def train_models(self, features_df: pd.DataFrame):
        """Entraîne les modèles ML"""
        
        # Préparation des targets (simulés pour la démo)
        X = features_df.drop(['id'], axis=1)
        
        # Targets simulés basés sur les features réelles
        y_ma = self._generate_ma_target(features_df)
        y_croissance = self._generate_croissance_target(features_df)
        y_stabilite = self._generate_stabilite_target(features_df)
        
        # Gestion des valeurs manquantes
        X = X.fillna(0)
        
        # Split train/test
        X_train, X_test, y_ma_train, y_ma_test = train_test_split(X, y_ma, test_size=0.2, random_state=42)
        _, _, y_croissance_train, y_croissance_test = train_test_split(X, y_croissance, test_size=0.2, random_state=42)
        _, _, y_stabilite_train, y_stabilite_test = train_test_split(X, y_stabilite, test_size=0.2, random_state=42)
        
        # Entraînement modèle M&A (XGBoost)
        logger.info("🔥 Entraînement modèle M&A (XGBoost)...")
        self.models['score_ma'] = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            verbosity=0
        )
        self.models['score_ma'].fit(X_train, y_ma_train)
        
        # Entraînement modèle Croissance (LightGBM)
        logger.info("📈 Entraînement modèle Croissance (LightGBM)...")
        self.models['score_croissance'] = lgb.LGBMRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            verbosity=-1
        )
        self.models['score_croissance'].fit(X_train, y_croissance_train)
        
        # Entraînement modèle Stabilité (CatBoost)
        logger.info("⚖️ Entraînement modèle Stabilité (CatBoost)...")
        self.models['score_stabilite'] = CatBoostRegressor(
            iterations=100,
            depth=6,
            learning_rate=0.1,
            random_state=42,
            verbose=False
        )
        self.models['score_stabilite'].fit(X_train, y_stabilite_train)
        
        # Évaluation des modèles
        await self._evaluate_models(X_test, y_ma_test, y_croissance_test, y_stabilite_test)
        
        # Sauvegarde des modèles
        await self.save_models()
    
    def _generate_ma_target(self, df: pd.DataFrame) -> np.ndarray:
        """Génère un target M&A réaliste basé sur les features"""
        base_score = 50
        
        # Impact du CA
        ca_impact = np.where(df['chiffre_affaires'] > 5_000_000, 20, 
                           np.where(df['chiffre_affaires'] > 1_000_000, 10, 0))
        
        # Impact rentabilité
        rent_impact = np.where(df['rentabilite_nette'] > 10, 15,
                             np.where(df['rentabilite_nette'] > 5, 8, -5))
        
        # Impact taille
        taille_impact = (df['effectif'] / 10).clip(0, 15)
        
        # Impact âge (maturité)
        age_impact = np.where(df['age_entreprise'] > 5, 10, -10)
        
        target = base_score + ca_impact + rent_impact + taille_impact + age_impact
        target += np.random.normal(0, 5, len(df))  # Bruit
        
        return target.clip(0, 100)
    
    def _generate_croissance_target(self, df: pd.DataFrame) -> np.ndarray:
        """Génère un target croissance"""
        return (df['croissance_ca_1an'] * 2 + 50 + np.random.normal(0, 10, len(df))).clip(0, 100)
    
    def _generate_stabilite_target(self, df: pd.DataFrame) -> np.ndarray:
        """Génère un target stabilité"""
        return df['stabilite_financiere'] + np.random.normal(0, 5, len(df))
    
    async def _evaluate_models(self, X_test, y_ma_test, y_croissance_test, y_stabilite_test):
        """Évalue les performances des modèles"""
        models_performance = {}
        
        for model_name, target_test in [
            ('score_ma', y_ma_test),
            ('score_croissance', y_croissance_test), 
            ('score_stabilite', y_stabilite_test)
        ]:
            y_pred = self.models[model_name].predict(X_test)
            mae = mean_absolute_error(target_test, y_pred)
            r2 = r2_score(target_test, y_pred)
            
            models_performance[model_name] = {'MAE': mae, 'R2': r2}
            logger.info(f"📊 {model_name}: MAE={mae:.2f}, R2={r2:.3f}")
        
        return models_performance
    
    async def load_models(self) -> bool:
        """Charge les modèles sauvegardés"""
        try:
            model_files = {
                'score_ma': 'xgb_ma_model.joblib',
                'score_croissance': 'lgb_croissance_model.joblib',
                'score_stabilite': 'catboost_stabilite_model.joblib'
            }
            
            for model_name, filename in model_files.items():
                if os.path.exists(filename):
                    self.models[model_name] = joblib.load(filename)
                    logger.info(f"✅ Modèle {model_name} chargé")
                else:
                    logger.warning(f"⚠️ Fichier {filename} non trouvé")
                    return False
            
            # Charger les encoders
            if os.path.exists('label_encoders.joblib'):
                self.label_encoders = joblib.load('label_encoders.joblib')
            
            return len(self.models) == 3
            
        except Exception as e:
            logger.error(f"Erreur chargement modèles: {e}")
            return False
    
    async def save_models(self):
        """Sauvegarde les modèles"""
        try:
            model_files = {
                'score_ma': 'xgb_ma_model.joblib',
                'score_croissance': 'lgb_croissance_model.joblib', 
                'score_stabilite': 'catboost_stabilite_model.joblib'
            }
            
            for model_name, filename in model_files.items():
                joblib.dump(self.models[model_name], filename)
                logger.info(f"💾 Modèle {model_name} sauvegardé: {filename}")
            
            # Sauvegarder les encoders
            joblib.dump(self.label_encoders, 'label_encoders.joblib')
            logger.info("💾 Encoders sauvegardés")
            
        except Exception as e:
            logger.error(f"Erreur sauvegarde modèles: {e}")
    
    async def calculate_scores(self, features_df: pd.DataFrame) -> List[Dict]:
        """Calcule les scores pour toutes les entreprises"""
        scores = []
        
        X = features_df.drop(['id'], axis=1).fillna(0)
        
        # Prédictions
        pred_ma = self.models['score_ma'].predict(X)
        pred_croissance = self.models['score_croissance'].predict(X)
        pred_stabilite = self.models['score_stabilite'].predict(X)
        
        for i, row in features_df.iterrows():
            # Score composite (pondéré)
            score_composite = (
                pred_ma[i] * 0.4 +
                pred_croissance[i] * 0.3 +
                pred_stabilite[i] * 0.3
            )
            
            # Calcul de la confiance
            confidence = self._calculate_confidence(row, pred_ma[i], pred_croissance[i], pred_stabilite[i])
            
            score_record = {
                'company_id': int(row['id']),
                'score_ma': float(np.clip(pred_ma[i], 0, 100)),
                'score_croissance': float(np.clip(pred_croissance[i], 0, 100)),
                'score_stabilite': float(np.clip(pred_stabilite[i], 0, 100)),
                'score_composite': float(np.clip(score_composite, 0, 100)),
                'confidence': float(confidence),
                'model_version': 'v1.0',
                'calculated_at': datetime.utcnow().isoformat(),
                'features_count': len(X.columns)
            }
            
            scores.append(score_record)
        
        return scores
    
    def _calculate_confidence(self, row, score_ma, score_croissance, score_stabilite) -> float:
        """Calcule un niveau de confiance pour les scores"""
        confidence = 85.0  # Base
        
        # Réduction si données manquantes
        critical_features = ['chiffre_affaires', 'benefice_net', 'effectif']
        missing_count = sum(1 for f in critical_features if pd.isna(row.get(f, np.nan)) or row.get(f, 0) == 0)
        confidence -= missing_count * 15
        
        # Réduction si scores très dispersés
        scores = [score_ma, score_croissance, score_stabilite]
        score_std = np.std(scores)
        if score_std > 25:
            confidence -= 10
        
        # Bonus si entreprise mature
        if row.get('age_entreprise', 0) > 10:
            confidence += 5
        
        return max(30.0, min(95.0, confidence))
    
    async def save_scores(self, scores: List[Dict]):
        """Sauvegarde les scores dans Supabase"""
        try:
            # Supprimer les anciens scores pour les mêmes entreprises
            company_ids = [score['company_id'] for score in scores]
            
            if company_ids:
                delete_response = self.supabase.table('ml_scores')\
                    .delete()\
                    .in_('company_id', company_ids)\
                    .execute()
                
                logger.info(f"🗑️ Anciens scores supprimés pour {len(company_ids)} entreprises")
            
            # Insérer les nouveaux scores par batch
            batch_size = 100
            total_inserted = 0
            
            for i in range(0, len(scores), batch_size):
                batch = scores[i:i + batch_size]
                
                response = self.supabase.table('ml_scores')\
                    .insert(batch)\
                    .execute()
                
                total_inserted += len(batch)
                logger.info(f"✅ Batch {i//batch_size + 1}: {len(batch)} scores insérés")
            
            logger.info(f"🎉 Total: {total_inserted} scores sauvegardés")
            
        except Exception as e:
            logger.error(f"Erreur sauvegarde scores: {e}")
            raise
    
    async def print_statistics(self, scores: List[Dict]):
        """Affiche les statistiques finales"""
        if not scores:
            return
        
        # Calcul des statistiques
        scores_ma = [s['score_ma'] for s in scores]
        scores_croissance = [s['score_croissance'] for s in scores]
        scores_stabilite = [s['score_stabilite'] for s in scores]
        scores_composite = [s['score_composite'] for s in scores]
        confidences = [s['confidence'] for s in scores]
        
        logger.info("📊 === STATISTIQUES FINALES ===")
        logger.info(f"🏢 Entreprises traitées: {len(scores)}")
        logger.info(f"🎯 Score M&A moyen: {np.mean(scores_ma):.1f} (σ={np.std(scores_ma):.1f})")
        logger.info(f"📈 Score Croissance moyen: {np.mean(scores_croissance):.1f} (σ={np.std(scores_croissance):.1f})")
        logger.info(f"⚖️ Score Stabilité moyen: {np.mean(scores_stabilite):.1f} (σ={np.std(scores_stabilite):.1f})")
        logger.info(f"🔮 Score Composite moyen: {np.mean(scores_composite):.1f} (σ={np.std(scores_composite):.1f})")
        logger.info(f"✅ Confiance moyenne: {np.mean(confidences):.1f}%")
        
        # Distribution des scores
        excellent = len([s for s in scores_composite if s >= 75])
        bon = len([s for s in scores_composite if 50 <= s < 75])
        moyen = len([s for s in scores_composite if 25 <= s < 50])
        faible = len([s for s in scores_composite if s < 25])
        
        logger.info("📊 Distribution des scores composites:")
        logger.info(f"   🟢 Excellent (≥75): {excellent} ({excellent/len(scores)*100:.1f}%)")
        logger.info(f"   🟡 Bon (50-74): {bon} ({bon/len(scores)*100:.1f}%)")
        logger.info(f"   🟠 Moyen (25-49): {moyen} ({moyen/len(scores)*100:.1f}%)")
        logger.info(f"   🔴 Faible (<25): {faible} ({faible/len(scores)*100:.1f}%)")

async def main():
    """Point d'entrée principal"""
    parser = argparse.ArgumentParser(description='Script de scoring ML pour M&A Intelligence')
    parser.add_argument('--batch-size', type=int, default=1000, help='Taille des batches')
    parser.add_argument('--company-ids', type=str, help='IDs des entreprises (séparés par des virgules)')
    parser.add_argument('--force-retrain', action='store_true', help='Force le ré-entraînement des modèles')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], default='INFO')
    
    args = parser.parse_args()
    
    # Configuration du niveau de log
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Parse company IDs
    company_ids = None
    if args.company_ids:
        try:
            company_ids = [int(x.strip()) for x in args.company_ids.split(',')]
            logger.info(f"🎯 Scoring spécifique pour {len(company_ids)} entreprises")
        except ValueError:
            logger.error("❌ Format d'IDs d'entreprises invalide")
            return
    
    # Lancement du scoring
    try:
        engine = MLScoringEngine()
        await engine.run_scoring(
            batch_size=args.batch_size,
            company_ids=company_ids,
            force_retrain=args.force_retrain
        )
        logger.info("🎉 Scoring terminé avec succès!")
        
    except Exception as e:
        logger.error(f"❌ Erreur fatale: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())