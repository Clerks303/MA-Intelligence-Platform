#!/usr/bin/env python3
"""
ML Scoring Service - Batch Processing
M&A Intelligence Platform

Ce script calcule les scores ML pour toutes les entreprises
et les sauvegarde dans Supabase.
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
import numpy as np
from datetime import datetime

# Ajout du path pour les imports
sys.path.append(str(Path(__file__).parent.parent))

from app.models.model_manager import ModelManager
from app.features.feature_engineer import FeatureEngineer
from app.scoring.scorer import MLScorer
from shared.database.supabase_client import create_supabase_client
from shared.config.settings import get_ml_settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MLScoringService:
    """Service principal pour le calcul des scores ML"""
    
    def __init__(self):
        self.settings = get_ml_settings()
        self.supabase = create_supabase_client()
        self.model_manager = ModelManager()
        self.feature_engineer = FeatureEngineer()
        self.scorer = MLScorer()
        
    async def run_batch_scoring(self, batch_size: int = 1000):
        """
        Lance le calcul de scores pour toutes les entreprises
        """
        logger.info("🚀 Démarrage du batch scoring ML")
        
        try:
            # 1. Charger les modèles ML
            logger.info("📦 Chargement des modèles ML...")
            models = await self.model_manager.load_production_models()
            
            # 2. Récupérer les entreprises à scorer
            logger.info("🏢 Récupération des entreprises...")
            companies = await self.get_companies_to_score()
            total_companies = len(companies)
            logger.info(f"📊 {total_companies} entreprises à traiter")
            
            # 3. Traitement par batch
            processed = 0
            for i in range(0, total_companies, batch_size):
                batch = companies[i:i + batch_size]
                logger.info(f"⚙️  Traitement batch {i//batch_size + 1} ({len(batch)} entreprises)")
                
                # Feature engineering
                features_df = await self.feature_engineer.create_features(batch)
                
                # Calcul des scores
                scores = await self.scorer.calculate_scores(features_df, models)
                
                # Sauvegarde
                await self.save_scores_batch(scores)
                
                processed += len(batch)
                logger.info(f"✅ {processed}/{total_companies} entreprises traitées")
            
            logger.info("🎉 Batch scoring terminé avec succès!")
            return {"status": "success", "processed": processed}
            
        except Exception as e:
            logger.error(f"❌ Erreur lors du batch scoring: {e}")
            raise
    
    async def get_companies_to_score(self) -> List[Dict]:
        """Récupère les entreprises nécessitant un scoring"""
        query = """
        SELECT c.*, 
               COALESCE(ms.calculated_at, '1970-01-01') as last_scored
        FROM companies c
        LEFT JOIN ml_scores ms ON c.id = ms.company_id
        WHERE c.chiffre_affaires IS NOT NULL
          AND c.secteur IS NOT NULL
        ORDER BY last_scored ASC
        """
        
        result = await self.supabase.execute_query(query)
        return result.data
    
    async def save_scores_batch(self, scores: List[Dict]):
        """Sauvegarde les scores en batch"""
        await self.supabase.upsert_batch('ml_scores', scores)

async def main():
    """Point d'entrée principal"""
    service = MLScoringService()
    
    # Arguments optionnels
    import argparse
    parser = argparse.ArgumentParser(description='ML Scoring Service')
    parser.add_argument('--batch-size', type=int, default=1000, 
                       help='Taille des batches (défaut: 1000)')
    parser.add_argument('--companies', nargs='+', type=int,
                       help='IDs spécifiques des entreprises à scorer')
    
    args = parser.parse_args()
    
    if args.companies:
        logger.info(f"📋 Scoring spécifique pour les entreprises: {args.companies}")
        # TODO: Implémenter le scoring spécifique
    else:
        await service.run_batch_scoring(batch_size=args.batch_size)

if __name__ == "__main__":
    asyncio.run(main())