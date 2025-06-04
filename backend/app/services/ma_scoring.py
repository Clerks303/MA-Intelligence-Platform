"""
Module de scoring M&A intelligent pour l'évaluation du potentiel d'acquisition/cession.

Ce module calcule un score sur 100 basé sur des critères financiers, opérationnels,
et stratégiques pondérés selon l'importance pour les opérations M&A.
"""

import logging
import math
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, date
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class ScoreComponent(Enum):
    """Composants du score M&A"""
    FINANCIAL_PERFORMANCE = "financial_performance"
    GROWTH_TRAJECTORY = "growth_trajectory" 
    PROFITABILITY = "profitability"
    FINANCIAL_HEALTH = "financial_health"
    OPERATIONAL_SCALE = "operational_scale"
    MARKET_POSITION = "market_position"
    MANAGEMENT_QUALITY = "management_quality"
    STRATEGIC_VALUE = "strategic_value"


@dataclass
class ScoringWeights:
    """
    Configuration des pondérations pour le calcul du score M&A.
    
    Toutes les pondérations doivent totaliser 1.0 (100%).
    Modifiez ces valeurs selon votre stratégie M&A.
    """
    
    # Performance financière (25%) - Critère principal
    financial_performance: float = 0.25
    
    # Trajectoire de croissance (20%) - Potentiel futur
    growth_trajectory: float = 0.20
    
    # Rentabilité (15%) - Qualité des résultats
    profitability: float = 0.15
    
    # Santé financière (15%) - Solidité
    financial_health: float = 0.15
    
    # Taille opérationnelle (10%) - Masse critique
    operational_scale: float = 0.10
    
    # Position marché (8%) - Avantage concurrentiel
    market_position: float = 0.08
    
    # Qualité management (4%) - Facteur humain
    management_quality: float = 0.04
    
    # Valeur stratégique (3%) - Synergies potentielles
    strategic_value: float = 0.03
    
    def __post_init__(self):
        """Validation que la somme des pondérations = 1.0"""
        total = sum([
            self.financial_performance,
            self.growth_trajectory,
            self.profitability,
            self.financial_health,
            self.operational_scale,
            self.market_position,
            self.management_quality,
            self.strategic_value
        ])
        
        if abs(total - 1.0) > 0.001:  # Tolérance pour les arrondis
            raise ValueError(f"Les pondérations doivent totaliser 1.0, actuel: {total:.3f}")


@dataclass
class ScoreBreakdown:
    """Détail du calcul de score pour transparence"""
    final_score: float
    component_scores: Dict[str, float] = field(default_factory=dict)
    component_weights: Dict[str, float] = field(default_factory=dict)
    weighted_scores: Dict[str, float] = field(default_factory=dict)
    data_quality_penalty: float = 0.0
    warnings: List[str] = field(default_factory=list)
    calculation_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class ScoringResult:
    """Résultat complet du scoring M&A"""
    final_score: float
    breakdown: ScoreBreakdown
    metadata: Dict[str, Any] = field(default_factory=dict)
    company_info: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Post-processing du résultat"""
        if 'calculated_at' not in self.metadata:
            self.metadata['calculated_at'] = datetime.now().isoformat()
    
    @property
    def grade(self) -> str:
        """Grade qualitatif basé sur le score"""
        if self.final_score >= 90:
            return "A+"
        elif self.final_score >= 80:
            return "A"
        elif self.final_score >= 70:
            return "B+"
        elif self.final_score >= 60:
            return "B"
        elif self.final_score >= 50:
            return "C+"
        elif self.final_score >= 40:
            return "C"
        else:
            return "D"


class MAScoring:
    """
    Calculateur de score M&A avec algorithme de pondération configurable.
    
    Le score final (0-100) évalue l'attractivité d'une entreprise pour des opérations
    de fusion-acquisition basé sur des critères financiers et opérationnels.
    """
    
    def __init__(self, weights: Optional[ScoringWeights] = None):
        """
        Initialise le calculateur avec des pondérations personnalisées.
        
        Args:
            weights: Pondérations personnalisées, utilise les valeurs par défaut si None
        """
        self.weights = weights or ScoringWeights()
        
        # Seuils de référence pour normalisation des scores
        self.thresholds = {
            # Chiffre d'affaires (en euros)
            'ca_excellent': 20_000_000,      # >20M€ = score max
            'ca_good': 10_000_000,           # 10-20M€ = bon score
            'ca_minimum': 3_000_000,         # 3M€ = seuil minimum M&A
            
            # Effectif
            'effectif_excellent': 100,        # >100 = score max
            'effectif_good': 50,             # 50-100 = bon score  
            'effectif_minimum': 30,          # 30 = seuil minimum
            
            # Croissance CA (%)
            'croissance_excellent': 15,      # >15% = excellente
            'croissance_good': 8,            # 8-15% = bonne
            'croissance_minimum': 3,         # 3% = minimum acceptable
            
            # Marge nette (%)
            'marge_excellent': 15,           # >15% = excellente
            'marge_good': 8,                 # 8-15% = bonne
            'marge_minimum': 3,              # 3% = minimum viable
            
            # Ratio d'endettement
            'endettement_excellent': 0.3,    # <30% = excellent
            'endettement_good': 0.5,         # 30-50% = acceptable
            'endettement_maximum': 0.8,      # >80% = risqué
            
            # Ancienneté entreprise (années)
            'anciennete_excellent': 10,      # >10 ans = maturité
            'anciennete_minimum': 3,         # 3 ans = minimum
        }
    
    def calculate_ma_score(self, company_data: Dict[str, Any]) -> ScoringResult:
        """
        Calcule le score M&A complet d'une entreprise.
        
        Args:
            company_data: Dictionnaire contenant les données de l'entreprise
            
        Returns:
            ScoringResult avec le score final, détail du calcul et métadonnées
            
        Raises:
            ValueError: Si les données minimales sont manquantes
        """
        
        logger.info(f"Calcul score M&A pour {company_data.get('nom_entreprise', 'Entreprise inconnue')}")
        
        # Validation des données minimales
        self._validate_required_data(company_data)
        
        # Calcul des scores par composant
        component_scores = {}
        warnings = []
        
        try:
            component_scores[ScoreComponent.FINANCIAL_PERFORMANCE.value] = self._score_financial_performance(company_data)
            component_scores[ScoreComponent.GROWTH_TRAJECTORY.value] = self._score_growth_trajectory(company_data)
            component_scores[ScoreComponent.PROFITABILITY.value] = self._score_profitability(company_data)
            component_scores[ScoreComponent.FINANCIAL_HEALTH.value] = self._score_financial_health(company_data)
            component_scores[ScoreComponent.OPERATIONAL_SCALE.value] = self._score_operational_scale(company_data)
            component_scores[ScoreComponent.MARKET_POSITION.value] = self._score_market_position(company_data)
            component_scores[ScoreComponent.MANAGEMENT_QUALITY.value] = self._score_management_quality(company_data)
            component_scores[ScoreComponent.STRATEGIC_VALUE.value] = self._score_strategic_value(company_data)
            
        except Exception as e:
            logger.error(f"Erreur calcul composant: {e}")
            warnings.append(f"Erreur calcul: {str(e)}")
        
        # Pondération des scores
        weighted_scores = self._apply_weights(component_scores)
        
        # Score final brut
        final_score = sum(weighted_scores.values())
        
        # Pénalité qualité des données
        data_quality_penalty = self._calculate_data_quality_penalty(company_data)
        final_score = max(0, final_score - data_quality_penalty)
        
        # Normalisation finale (0-100)
        final_score = min(100, max(0, final_score))
        
        # Construction du résultat détaillé
        breakdown = ScoreBreakdown(
            final_score=round(final_score, 2),
            component_scores=component_scores,
            component_weights=self._get_weights_dict(),
            weighted_scores=weighted_scores,
            data_quality_penalty=data_quality_penalty,
            warnings=warnings
        )
        
        # Create ScoringResult with breakdown and metadata
        result = ScoringResult(
            final_score=round(final_score, 2),
            breakdown=breakdown,
            metadata={
                'algorithm_version': '1.0',
                'calculation_method': 'weighted_components',
                'data_quality_score': 100 - data_quality_penalty
            },
            company_info={
                'siren': company_data.get('siren'),
                'nom_entreprise': company_data.get('nom_entreprise'),
                'chiffre_affaires': company_data.get('chiffre_affaires'),
                'effectif': company_data.get('effectif')
            }
        )
        
        logger.info(f"Score M&A calculé: {final_score:.1f}/100")
        return result
    
    def _validate_required_data(self, data: Dict[str, Any]) -> None:
        """Valide que les données minimales sont présentes"""
        required_fields = ['siren', 'nom_entreprise']
        missing_fields = [field for field in required_fields if not data.get(field)]
        
        if missing_fields:
            raise ValueError(f"Champs obligatoires manquants: {', '.join(missing_fields)}")
    
    def _score_financial_performance(self, data: Dict[str, Any]) -> float:
        """Score basé sur la performance financière globale (0-100)"""
        ca = data.get('chiffre_affaires', 0)
        if not ca or ca <= 0:
            return 0
        
        # Score basé sur le CA avec courbe logarithmique
        if ca >= self.thresholds['ca_excellent']:
            ca_score = 100
        elif ca >= self.thresholds['ca_good']:
            # Progression linéaire entre bon et excellent
            ratio = (ca - self.thresholds['ca_good']) / (self.thresholds['ca_excellent'] - self.thresholds['ca_good'])
            ca_score = 70 + (30 * ratio)
        elif ca >= self.thresholds['ca_minimum']:
            # Progression linéaire entre minimum et bon
            ratio = (ca - self.thresholds['ca_minimum']) / (self.thresholds['ca_good'] - self.thresholds['ca_minimum'])
            ca_score = 40 + (30 * ratio)
        else:
            # En dessous du minimum M&A
            ca_score = min(40, (ca / self.thresholds['ca_minimum']) * 40)
        
        # Bonus pour stabilité si plusieurs années de CA
        stability_bonus = 0
        ca_n1 = data.get('chiffre_affaires_n1')
        ca_n2 = data.get('chiffre_affaires_n2')
        
        if ca_n1 and ca_n2:
            # Vérifier la régularité des performances
            cas = [ca, ca_n1, ca_n2]
            min_ca, max_ca = min(cas), max(cas)
            if min_ca > 0:
                stability = 1 - (max_ca - min_ca) / max_ca
                stability_bonus = min(10, stability * 10)  # Max 10 points
        
        return min(100, ca_score + stability_bonus)
    
    def _score_growth_trajectory(self, data: Dict[str, Any]) -> float:
        """Score basé sur la trajectoire de croissance (0-100)"""
        evolution_ca = data.get('evolution_ca_3ans')
        if evolution_ca is None:
            # Calcul automatique si possible
            ca = data.get('chiffre_affaires', 0)
            ca_n2 = data.get('chiffre_affaires_n2', 0)
            
            if ca > 0 and ca_n2 > 0:
                evolution_ca = ((ca / ca_n2) ** (1/2) - 1) * 100  # TCAM sur 2 ans
            else:
                return 50  # Score neutre si pas de données
        
        # Score basé sur la croissance
        if evolution_ca >= self.thresholds['croissance_excellent']:
            growth_score = 100
        elif evolution_ca >= self.thresholds['croissance_good']:
            ratio = (evolution_ca - self.thresholds['croissance_good']) / (self.thresholds['croissance_excellent'] - self.thresholds['croissance_good'])
            growth_score = 75 + (25 * ratio)
        elif evolution_ca >= self.thresholds['croissance_minimum']:
            ratio = (evolution_ca - self.thresholds['croissance_minimum']) / (self.thresholds['croissance_good'] - self.thresholds['croissance_minimum'])
            growth_score = 50 + (25 * ratio)
        elif evolution_ca >= 0:
            # Croissance positive mais faible
            growth_score = 25 + (25 * (evolution_ca / self.thresholds['croissance_minimum']))
        else:
            # Décroissance - score faible mais pas nul
            growth_score = max(0, 25 + (evolution_ca * 2))  # Pénalité progressive
        
        # Bonus pour accélération récente
        acceleration_bonus = 0
        ca = data.get('chiffre_affaires', 0)
        ca_n1 = data.get('chiffre_affaires_n1', 0)
        
        if ca > 0 and ca_n1 > 0:
            recent_growth = ((ca / ca_n1) - 1) * 100
            if recent_growth > evolution_ca:
                acceleration_bonus = min(10, (recent_growth - evolution_ca) / 2)
        
        return min(100, growth_score + acceleration_bonus)
    
    def _score_profitability(self, data: Dict[str, Any]) -> float:
        """Score basé sur la rentabilité (0-100)"""
        ca = data.get('chiffre_affaires', 0)
        resultat = data.get('resultat', 0)
        
        if not ca or ca <= 0:
            return 0
        
        # Calcul de la marge nette
        marge_nette = data.get('marge_nette')
        if marge_nette is None and resultat is not None:
            marge_nette = (resultat / ca) * 100
        
        if marge_nette is None:
            return 30  # Score par défaut si pas de données
        
        # Score basé sur la marge
        if marge_nette >= self.thresholds['marge_excellent']:
            marge_score = 100
        elif marge_nette >= self.thresholds['marge_good']:
            ratio = (marge_nette - self.thresholds['marge_good']) / (self.thresholds['marge_excellent'] - self.thresholds['marge_good'])
            marge_score = 75 + (25 * ratio)
        elif marge_nette >= self.thresholds['marge_minimum']:
            ratio = (marge_nette - self.thresholds['marge_minimum']) / (self.thresholds['marge_good'] - self.thresholds['marge_minimum'])
            marge_score = 50 + (25 * ratio)
        elif marge_nette >= 0:
            marge_score = (marge_nette / self.thresholds['marge_minimum']) * 50
        else:
            # Perte - score très faible
            marge_score = max(0, 10 + marge_nette)  # Pénalité pour pertes
        
        # Bonus pour régularité de la rentabilité
        consistency_bonus = 0
        resultat_n1 = data.get('resultat_n1')
        ca_n1 = data.get('chiffre_affaires_n1')
        
        if resultat_n1 and ca_n1 and ca_n1 > 0:
            marge_n1 = (resultat_n1 / ca_n1) * 100
            if marge_n1 > 0 and marge_nette > 0:
                consistency_bonus = min(5, abs(marge_nette - marge_n1) / max(marge_nette, marge_n1) * 5)
        
        return min(100, marge_score + consistency_bonus)
    
    def _score_financial_health(self, data: Dict[str, Any]) -> float:
        """Score basé sur la santé financière (0-100)"""
        ratio_endettement = data.get('ratio_endettement')
        
        # Score d'endettement
        if ratio_endettement is not None:
            if ratio_endettement <= self.thresholds['endettement_excellent']:
                debt_score = 100
            elif ratio_endettement <= self.thresholds['endettement_good']:
                ratio = (ratio_endettement - self.thresholds['endettement_excellent']) / (self.thresholds['endettement_good'] - self.thresholds['endettement_excellent'])
                debt_score = 100 - (30 * ratio)
            elif ratio_endettement <= self.thresholds['endettement_maximum']:
                ratio = (ratio_endettement - self.thresholds['endettement_good']) / (self.thresholds['endettement_maximum'] - self.thresholds['endettement_good'])
                debt_score = 70 - (50 * ratio)
            else:
                debt_score = max(0, 20 - (ratio_endettement - self.thresholds['endettement_maximum']) * 100)
        else:
            debt_score = 60  # Score neutre par défaut
        
        # Score de liquidité (approximatif)
        liquidity_score = 70  # Score par défaut
        ca = data.get('chiffre_affaires', 0)
        resultat = data.get('resultat', 0)
        
        if ca > 0 and resultat is not None:
            cash_flow_ratio = resultat / ca
            if cash_flow_ratio > 0.1:  # >10% du CA en résultat
                liquidity_score = 90
            elif cash_flow_ratio > 0.05:
                liquidity_score = 80
            elif cash_flow_ratio < 0:
                liquidity_score = 30
        
        # Score de maturité financière
        maturity_score = self._score_company_maturity(data)
        
        # Moyenne pondérée
        health_score = (debt_score * 0.5) + (liquidity_score * 0.3) + (maturity_score * 0.2)
        
        return min(100, health_score)
    
    def _score_operational_scale(self, data: Dict[str, Any]) -> float:
        """Score basé sur la taille opérationnelle (0-100)"""
        effectif = data.get('effectif', 0)
        ca = data.get('chiffre_affaires', 0)
        
        # Score effectif
        if effectif >= self.thresholds['effectif_excellent']:
            effectif_score = 100
        elif effectif >= self.thresholds['effectif_good']:
            ratio = (effectif - self.thresholds['effectif_good']) / (self.thresholds['effectif_excellent'] - self.thresholds['effectif_good'])
            effectif_score = 70 + (30 * ratio)
        elif effectif >= self.thresholds['effectif_minimum']:
            ratio = (effectif - self.thresholds['effectif_minimum']) / (self.thresholds['effectif_good'] - self.thresholds['effectif_minimum'])
            effectif_score = 40 + (30 * ratio)
        else:
            effectif_score = (effectif / self.thresholds['effectif_minimum']) * 40
        
        # Score CA (déjà calculé mais simplifié ici)
        if ca >= self.thresholds['ca_excellent']:
            ca_score = 100
        elif ca >= self.thresholds['ca_minimum']:
            ratio = (ca - self.thresholds['ca_minimum']) / (self.thresholds['ca_excellent'] - self.thresholds['ca_minimum'])
            ca_score = 50 + (50 * ratio)
        else:
            ca_score = (ca / self.thresholds['ca_minimum']) * 50
        
        # Moyenne pondérée (effectif plus important que CA pour la dimension opérationnelle)
        scale_score = (effectif_score * 0.6) + (ca_score * 0.4)
        
        return min(100, scale_score)
    
    def _score_market_position(self, data: Dict[str, Any]) -> float:
        """Score basé sur la position marché (0-100)"""
        # Score géographique (zone premium)
        geo_score = 70  # Score par défaut
        code_postal = data.get('code_postal', '')
        
        if code_postal:
            if code_postal.startswith('75'):  # Paris
                geo_score = 100
            elif code_postal.startswith(('92', '93', '94')):  # Hauts-de-Seine, Seine-Saint-Denis, Val-de-Marne
                geo_score = 90
            elif code_postal.startswith(('77', '78', '91', '95')):  # Île-de-France
                geo_score = 80
            elif code_postal.startswith(('69', '13')):  # Lyon, Marseille
                geo_score = 75
        
        # Score sectoriel
        sector_score = 70  # Score par défaut
        code_naf = data.get('code_naf', '')
        
        if code_naf:
            if code_naf.startswith('692'):  # Activités comptables
                sector_score = 90
            elif code_naf.startswith('691'):  # Activités juridiques
                sector_score = 80
            elif code_naf.startswith('70'):   # Activités de conseil
                sector_score = 85
        
        # Score de spécialisation
        specialisation_score = 60
        specialisation = data.get('specialisation', '')
        nom_entreprise = data.get('nom_entreprise', '').lower()
        
        keywords_premium = ['audit', 'conseil', 'expertise', 'consulting']
        if any(keyword in nom_entreprise for keyword in keywords_premium):
            specialisation_score = 80
        
        # Moyenne pondérée
        position_score = (geo_score * 0.4) + (sector_score * 0.4) + (specialisation_score * 0.2)
        
        return min(100, position_score)
    
    def _score_management_quality(self, data: Dict[str, Any]) -> float:
        """Score basé sur la qualité du management (0-100)"""
        base_score = 60  # Score par défaut
        
        # Stabilité dirigeant
        anciennete_dirigeant = data.get('anciennete_dirigeant', 0)
        if anciennete_dirigeant >= 5:
            stability_score = 90
        elif anciennete_dirigeant >= 2:
            stability_score = 70
        else:
            stability_score = 50
        
        # Âge dirigeant (pour succession)
        age_score = 70  # Score neutre par défaut
        age_dirigeant = data.get('age_dirigeant_principal')
        if age_dirigeant:
            if 35 <= age_dirigeant <= 55:  # Âge optimal
                age_score = 90
            elif 25 <= age_dirigeant <= 65:  # Âge acceptable
                age_score = 80
            else:
                age_score = 60  # Très jeune ou proche retraite
        
        # Qualité des dirigeants (nombre et diversité)
        team_score = 60
        dirigeants_json = data.get('dirigeants_json', [])
        if isinstance(dirigeants_json, list) and len(dirigeants_json) > 1:
            team_score = 80  # Équipe de direction étoffée
        
        # Moyenne pondérée
        management_score = (stability_score * 0.5) + (age_score * 0.3) + (team_score * 0.2)
        
        return min(100, management_score)
    
    def _score_strategic_value(self, data: Dict[str, Any]) -> float:
        """Score basé sur la valeur stratégique (0-100)"""
        strategic_score = 60  # Score de base
        
        # Localisation stratégique
        code_postal = data.get('code_postal', '')
        if code_postal.startswith('75'):  # Paris = valeur stratégique élevée
            strategic_score += 20
        elif code_postal.startswith(('92', '93', '94', '77', '78', '91', '95')):
            strategic_score += 10
        
        # Taille critique pour synergies
        effectif = data.get('effectif', 0)
        ca = data.get('chiffre_affaires', 0)
        
        if effectif >= 50 and ca >= 5_000_000:  # Taille significative
            strategic_score += 15
        elif effectif >= 30 and ca >= 3_000_000:  # Taille minimum
            strategic_score += 10
        
        # Spécialisation
        nom_entreprise = data.get('nom_entreprise', '').lower()
        if any(keyword in nom_entreprise for keyword in ['audit', 'conseil', 'expertise']):
            strategic_score += 5
        
        return min(100, strategic_score)
    
    def _score_company_maturity(self, data: Dict[str, Any]) -> float:
        """Score basé sur la maturité de l'entreprise"""
        date_creation = data.get('date_creation')
        if not date_creation:
            return 60  # Score neutre
        
        try:
            if isinstance(date_creation, str):
                creation_date = datetime.fromisoformat(date_creation.replace('Z', '+00:00'))
            elif isinstance(date_creation, date):
                creation_date = datetime.combine(date_creation, datetime.min.time())
            else:
                creation_date = date_creation
            
            years_old = (datetime.now() - creation_date).days / 365.25
            
            if years_old >= self.thresholds['anciennete_excellent']:
                return 100
            elif years_old >= self.thresholds['anciennete_minimum']:
                ratio = (years_old - self.thresholds['anciennete_minimum']) / (self.thresholds['anciennete_excellent'] - self.thresholds['anciennete_minimum'])
                return 60 + (40 * ratio)
            else:
                return max(20, (years_old / self.thresholds['anciennete_minimum']) * 60)
                
        except Exception as e:
            logger.warning(f"Erreur calcul ancienneté: {e}")
            return 60
    
    def _apply_weights(self, component_scores: Dict[str, float]) -> Dict[str, float]:
        """Applique les pondérations aux scores des composants"""
        weight_mapping = {
            ScoreComponent.FINANCIAL_PERFORMANCE.value: self.weights.financial_performance,
            ScoreComponent.GROWTH_TRAJECTORY.value: self.weights.growth_trajectory,
            ScoreComponent.PROFITABILITY.value: self.weights.profitability,
            ScoreComponent.FINANCIAL_HEALTH.value: self.weights.financial_health,
            ScoreComponent.OPERATIONAL_SCALE.value: self.weights.operational_scale,
            ScoreComponent.MARKET_POSITION.value: self.weights.market_position,
            ScoreComponent.MANAGEMENT_QUALITY.value: self.weights.management_quality,
            ScoreComponent.STRATEGIC_VALUE.value: self.weights.strategic_value,
        }
        
        weighted_scores = {}
        for component, score in component_scores.items():
            weight = weight_mapping.get(component, 0)
            weighted_scores[component] = score * weight
        
        return weighted_scores
    
    def _calculate_data_quality_penalty(self, data: Dict[str, Any]) -> float:
        """Calcule une pénalité basée sur la qualité des données"""
        penalty = 0
        
        # Données financières manquantes
        financial_fields = ['chiffre_affaires', 'resultat', 'effectif']
        missing_financial = sum(1 for field in financial_fields if not data.get(field))
        penalty += missing_financial * 5  # 5 points par donnée financière manquante
        
        # Données d'enrichissement manquantes
        enrichment_fields = ['date_creation', 'forme_juridique', 'code_naf']
        missing_enrichment = sum(1 for field in enrichment_fields if not data.get(field))
        penalty += missing_enrichment * 2  # 2 points par donnée d'enrichissement manquante
        
        # Données anciennes
        last_scraped = data.get('last_scraped_at')
        if last_scraped:
            try:
                scraped_date = datetime.fromisoformat(last_scraped.replace('Z', '+00:00'))
                days_old = (datetime.now() - scraped_date).days
                if days_old > 90:  # Données de plus de 3 mois
                    penalty += min(10, days_old / 30)  # Jusqu'à 10 points
            except:
                penalty += 5  # Pénalité si date non parsable
        
        return min(20, penalty)  # Pénalité maximale de 20 points
    
    def _get_weights_dict(self) -> Dict[str, float]:
        """Retourne les pondérations sous forme de dictionnaire"""
        return {
            ScoreComponent.FINANCIAL_PERFORMANCE.value: self.weights.financial_performance,
            ScoreComponent.GROWTH_TRAJECTORY.value: self.weights.growth_trajectory,
            ScoreComponent.PROFITABILITY.value: self.weights.profitability,
            ScoreComponent.FINANCIAL_HEALTH.value: self.weights.financial_health,
            ScoreComponent.OPERATIONAL_SCALE.value: self.weights.operational_scale,
            ScoreComponent.MARKET_POSITION.value: self.weights.market_position,
            ScoreComponent.MANAGEMENT_QUALITY.value: self.weights.management_quality,
            ScoreComponent.STRATEGIC_VALUE.value: self.weights.strategic_value,
        }
    
    def get_score_interpretation(self, score: float) -> Tuple[str, str]:
        """
        Interprète un score M&A et retourne le niveau et une description.
        
        Args:
            score: Score M&A (0-100)
            
        Returns:
            Tuple (niveau, description)
        """
        if score >= 85:
            return ("EXCELLENT", "Cible M&A premium - À prioriser absolument")
        elif score >= 70:
            return ("TRÈS BON", "Cible M&A attractive - Forte recommandation")
        elif score >= 55:
            return ("BON", "Cible M&A intéressante - À étudier")
        elif score >= 40:
            return ("MOYEN", "Potentiel M&A limité - Analyse approfondie requise")
        elif score >= 25:
            return ("FAIBLE", "Peu adapté pour M&A - Risques élevés")
        else:
            return ("TRÈS FAIBLE", "Non recommandé pour M&A")


# Fonctions utilitaires pour intégration

async def calculate_company_ma_score(
    company_data: Dict[str, Any], 
    custom_weights: Optional[ScoringWeights] = None
) -> ScoreBreakdown:
    """
    Fonction async wrapper pour calcul de score M&A.
    Compatible avec l'orchestrateur de scraping.
    """
    scorer = MAScoring(weights=custom_weights)
    return scorer.calculate_ma_score(company_data)


def create_conservative_weights() -> ScoringWeights:
    """Pondérations conservatrices privilégiant la stabilité financière"""
    return ScoringWeights(
        financial_performance=0.30,  # Plus important
        growth_trajectory=0.15,      # Moins important
        profitability=0.20,          # Plus important
        financial_health=0.20,       # Plus important
        operational_scale=0.08,
        market_position=0.04,
        management_quality=0.02,
        strategic_value=0.01
    )


def create_growth_focused_weights() -> ScoringWeights:
    """Pondérations axées croissance pour acquisitions de développement"""
    return ScoringWeights(
        financial_performance=0.20,
        growth_trajectory=0.30,      # Plus important
        profitability=0.10,          # Moins important
        financial_health=0.15,
        operational_scale=0.15,      # Plus important
        market_position=0.05,
        management_quality=0.03,
        strategic_value=0.02
    )


def create_strategic_weights() -> ScoringWeights:
    """Pondérations pour acquisitions stratégiques"""
    return ScoringWeights(
        financial_performance=0.20,
        growth_trajectory=0.15,
        profitability=0.15,
        financial_health=0.15,
        operational_scale=0.10,
        market_position=0.15,       # Plus important
        management_quality=0.05,    # Plus important
        strategic_value=0.05        # Plus important
    )