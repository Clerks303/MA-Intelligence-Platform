"""
ML Scorer - Calcul des scores M&A
M&A Intelligence Platform
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any
from datetime import datetime

logger = logging.getLogger(__name__)

class MLScorer:
    """Calculateur de scores ML pour l'analyse M&A"""
    
    def __init__(self):
        self.score_ranges = {
            "score_ma": (0, 100),
            "score_croissance": (0, 100), 
            "score_stabilite": (0, 100)
        }
    
    async def calculate_scores(self, features_df: pd.DataFrame, models: Dict[str, Any]) -> List[Dict]:
        """
        Calcule les scores ML pour un batch d'entreprises
        
        Args:
            features_df: DataFrame avec les features engineerées
            models: Dictionnaire des modèles ML chargés
            
        Returns:
            Liste des scores calculés
        """
        logger.info(f"🎯 Calcul des scores pour {len(features_df)} entreprises")
        
        scores_list = []
        
        for idx, row in features_df.iterrows():
            try:
                # Calcul des scores individuels
                scores = await self._calculate_individual_scores(row, models)
                
                # Calcul du score composite
                score_composite = self._calculate_composite_score(scores)
                
                # Calcul de la confiance
                confidence = self._calculate_confidence(scores, row)
                
                # Préparation de l'objet de sortie
                score_record = {
                    "company_id": int(row['company_id']),
                    "score_ma": float(scores['score_ma']),
                    "score_croissance": float(scores['score_croissance']),
                    "score_stabilite": float(scores['score_stabilite']),
                    "score_composite": float(score_composite),
                    "confidence": float(confidence),
                    "model_version": "v1.0",
                    "calculated_at": datetime.utcnow().isoformat(),
                    "features_used": list(row.index)
                }
                
                scores_list.append(score_record)
                
            except Exception as e:
                logger.error(f"❌ Erreur calcul score pour entreprise {row.get('company_id', 'unknown')}: {e}")
                continue
        
        logger.info(f"✅ {len(scores_list)} scores calculés avec succès")
        return scores_list
    
    async def _calculate_individual_scores(self, features: pd.Series, models: Dict[str, Any]) -> Dict[str, float]:
        """Calcule les scores individuels pour une entreprise"""
        scores = {}
        
        # Préparer les features pour la prédiction
        feature_array = self._prepare_features_for_prediction(features)
        
        for score_type, model in models.items():
            try:
                if hasattr(model, 'predict'):
                    # Modèle sklearn/lightgbm/xgboost/catboost standard
                    prediction = model.predict(feature_array.reshape(1, -1))[0]
                elif hasattr(model, 'predict_proba'):
                    # Modèle de classification avec probabilité
                    prediction = model.predict_proba(feature_array.reshape(1, -1))[0][1] * 100
                else:
                    # Modèle MLflow ou custom
                    prediction = model.predict(feature_array.reshape(1, -1))[0]
                
                # Normalisation du score
                scores[score_type] = self._normalize_score(prediction, score_type)
                
            except Exception as e:
                logger.warning(f"⚠️ Erreur prédiction {score_type}: {e}")
                scores[score_type] = self._get_default_score(score_type, features)
        
        return scores
    
    def _prepare_features_for_prediction(self, features: pd.Series) -> np.ndarray:
        """Prépare les features pour la prédiction ML"""
        
        # Sélectionner uniquement les features numériques pour le modèle
        numeric_features = [
            'chiffre_affaires', 'benefice_net', 'fonds_propres',
            'dette_totale', 'effectif', 'age_entreprise',
            'croissance_ca_1an', 'croissance_ca_3ans',
            'ratio_endettement', 'rentabilite_nette',
            'secteur_encoded', 'region_encoded'
        ]
        
        feature_values = []
        
        for feature in numeric_features:
            if feature in features:
                value = features[feature]
                # Gestion des valeurs manquantes
                if pd.isna(value):
                    value = 0.0
                feature_values.append(float(value))
            else:
                # Valeur par défaut si feature manquante
                feature_values.append(0.0)
        
        return np.array(feature_values)
    
    def _normalize_score(self, raw_score: float, score_type: str) -> float:
        """Normalise un score dans la plage attendue"""
        min_score, max_score = self.score_ranges[score_type]
        
        # Clamp dans la plage
        normalized = max(min_score, min(max_score, raw_score))
        
        # Si le score brut est très différent de la plage, re-normaliser
        if raw_score < -10 or raw_score > 110:
            # Sigmoïde pour normaliser dans [0, 100]
            normalized = 100 / (1 + np.exp(-raw_score / 20))
        
        return round(normalized, 2)
    
    def _get_default_score(self, score_type: str, features: pd.Series) -> float:
        """Calcule un score par défaut basé sur des règles métier"""
        
        if score_type == "score_ma":
            # Score M&A basé sur la taille et la rentabilité
            ca = features.get('chiffre_affaires', 0)
            benefice = features.get('benefice_net', 0)
            
            if ca > 10_000_000:  # > 10M€
                base_score = 70
            elif ca > 1_000_000:  # > 1M€
                base_score = 50
            else:
                base_score = 30
            
            # Bonus rentabilité
            if benefice > 0 and ca > 0:
                rentabilite = (benefice / ca) * 100
                if rentabilite > 10:
                    base_score += 20
                elif rentabilite > 5:
                    base_score += 10
            
            return min(100, base_score)
        
        elif score_type == "score_croissance":
            # Score croissance basé sur l'évolution CA
            croissance = features.get('croissance_ca_1an', 0)
            
            if croissance > 20:
                return 80
            elif croissance > 10:
                return 65
            elif croissance > 0:
                return 50
            else:
                return 30
        
        elif score_type == "score_stabilite":
            # Score stabilité basé sur l'âge et l'endettement
            age = features.get('age_entreprise', 0)
            endettement = features.get('ratio_endettement', 0)
            
            base_score = min(80, age * 2)  # 2 points par année
            
            # Pénalité endettement
            if endettement > 0.8:
                base_score -= 30
            elif endettement > 0.5:
                base_score -= 15
            
            return max(10, base_score)
        
        return 50.0  # Score neutre par défaut
    
    def _calculate_composite_score(self, scores: Dict[str, float]) -> float:
        """Calcule un score composite pondéré"""
        weights = {
            "score_ma": 0.4,
            "score_croissance": 0.3,
            "score_stabilite": 0.3
        }
        
        composite = 0.0
        total_weight = 0.0
        
        for score_type, weight in weights.items():
            if score_type in scores:
                composite += scores[score_type] * weight
                total_weight += weight
        
        if total_weight > 0:
            composite = composite / total_weight
        else:
            composite = 50.0  # Score neutre si aucun score disponible
        
        return round(composite, 2)
    
    def _calculate_confidence(self, scores: Dict[str, float], features: pd.Series) -> float:
        """Calcule un niveau de confiance pour les scores"""
        confidence = 100.0
        
        # Réduction basée sur les données manquantes
        critical_features = ['chiffre_affaires', 'benefice_net', 'effectif']
        missing_critical = sum(1 for f in critical_features if pd.isna(features.get(f, np.nan)))
        confidence -= missing_critical * 20
        
        # Réduction basée sur la cohérence des scores
        if len(scores) >= 2:
            score_values = list(scores.values())
            score_std = np.std(score_values)
            if score_std > 30:  # Scores très dispersés
                confidence -= 15
            elif score_std > 20:
                confidence -= 10
        
        # Bonus si données récentes
        if features.get('derniere_maj', '') > '2023-01-01':
            confidence += 5
        
        return max(30.0, min(100.0, confidence))