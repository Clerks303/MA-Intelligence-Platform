"""
Tests unitaires pour le module de scoring M&A.

Ces tests vérifient le bon fonctionnement de l'algorithme de scoring,
la cohérence des pondérations, et la robustesse face aux données manquantes.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any

from app.services.ma_scoring import (
    MAScoring,
    ScoringWeights,
    ScoreBreakdown,
    calculate_company_ma_score,
    create_conservative_weights,
    create_growth_focused_weights,
    create_strategic_weights
)


class TestScoringWeights:
    """Tests pour la configuration des pondérations"""
    
    def test_default_weights_sum_to_one(self):
        """Les pondérations par défaut doivent totaliser 1.0"""
        weights = ScoringWeights()
        total = (
            weights.financial_performance +
            weights.growth_trajectory +
            weights.profitability +
            weights.financial_health +
            weights.operational_scale +
            weights.market_position +
            weights.management_quality +
            weights.strategic_value
        )
        assert abs(total - 1.0) < 0.001, f"Total des pondérations: {total}"
    
    def test_custom_weights_validation(self):
        """Les pondérations customisées doivent être validées"""
        # Pondérations valides
        valid_weights = ScoringWeights(
            financial_performance=0.3,
            growth_trajectory=0.2,
            profitability=0.15,
            financial_health=0.15,
            operational_scale=0.1,
            market_position=0.05,
            management_quality=0.03,
            strategic_value=0.02
        )
        assert valid_weights  # Ne doit pas lever d'erreur
        
        # Pondérations invalides (ne totalisent pas 1.0)
        with pytest.raises(ValueError, match="doivent totaliser 1.0"):
            ScoringWeights(
                financial_performance=0.5,  # Total > 1.0
                growth_trajectory=0.5,
                profitability=0.2,
                financial_health=0.1,
                operational_scale=0.1,
                market_position=0.05,
                management_quality=0.03,
                strategic_value=0.02
            )
    
    def test_preset_weights_are_valid(self):
        """Les pondérations prédéfinies doivent être valides"""
        presets = [
            create_conservative_weights(),
            create_growth_focused_weights(),
            create_strategic_weights()
        ]
        
        for weights in presets:
            # Chaque preset doit être valide (ne pas lever d'erreur)
            assert weights
            
            # Vérifier que toutes les valeurs sont positives
            weight_values = [
                weights.financial_performance,
                weights.growth_trajectory,
                weights.profitability,
                weights.financial_health,
                weights.operational_scale,
                weights.market_position,
                weights.management_quality,
                weights.strategic_value
            ]
            assert all(w >= 0 for w in weight_values), "Toutes les pondérations doivent être positives"


class TestMAScoring:
    """Tests pour la classe MAScoring"""
    
    @pytest.fixture
    def scorer(self):
        """Instance de MAScoring pour les tests"""
        return MAScoring()
    
    @pytest.fixture
    def sample_company_excellent(self) -> Dict[str, Any]:
        """Données d'une entreprise excellente pour M&A"""
        return {
            'siren': '123456789',
            'nom_entreprise': 'Cabinet Audit Excellence',
            'chiffre_affaires': 25_000_000,      # 25M€ - excellent
            'chiffre_affaires_n1': 22_000_000,   # Croissance
            'chiffre_affaires_n2': 20_000_000,   # Progression constante
            'resultat': 3_750_000,               # 15% de marge - excellent
            'resultat_n1': 3_300_000,
            'resultat_n2': 3_000_000,
            'effectif': 120,                     # > 100 - excellent
            'capital_social': 500_000,
            'evolution_ca_3ans': 11.8,           # ~12% TCAM - excellent
            'marge_nette': 15.0,                 # 15% - excellent
            'ratio_endettement': 0.25,          # 25% - excellent
            'code_postal': '75008',              # Paris - premium
            'code_naf': '6920Z',                 # Expertise comptable
            'forme_juridique': 'SAS',
            'date_creation': '2010-01-15',       # 14 ans - mature
            'dirigeant_principal': 'Jean Martin (PDG)',
            'age_dirigeant_principal': 45,       # Âge optimal
            'anciennete_dirigeant': 8,           # Expérimenté
            'dirigeants_json': [
                {'nom_complet': 'Jean Martin', 'qualite': 'PDG'},
                {'nom_complet': 'Marie Dupont', 'qualite': 'DG'}
            ],
            'last_scraped_at': datetime.now().isoformat(),
            'specialisation': 'Audit et conseil'
        }
    
    @pytest.fixture
    def sample_company_poor(self) -> Dict[str, Any]:
        """Données d'une entreprise peu attractive pour M&A"""
        return {
            'siren': '987654321',
            'nom_entreprise': 'Petit Cabinet Comptable',
            'chiffre_affaires': 1_500_000,       # < 3M€ - en dessous seuil
            'resultat': -50_000,                 # Perte
            'effectif': 15,                      # < 30 - trop petit
            'code_postal': '95000',              # Banlieue lointaine
            'code_naf': '6920Z',
            'date_creation': '2022-01-01',       # Très récent
            'ratio_endettement': 0.9,           # 90% - très risqué
            'last_scraped_at': (datetime.now() - timedelta(days=180)).isoformat()  # Données anciennes
        }
    
    def test_calculate_score_with_excellent_company(self, scorer, sample_company_excellent):
        """Test avec une entreprise excellente"""
        result = scorer.calculate_ma_score(sample_company_excellent)
        
        assert isinstance(result, ScoreBreakdown)
        assert result.final_score >= 80, f"Score attendu ≥80, obtenu: {result.final_score}"
        assert result.final_score <= 100
        
        # Vérifier que tous les composants ont été calculés
        expected_components = [
            'financial_performance',
            'growth_trajectory', 
            'profitability',
            'financial_health',
            'operational_scale',
            'market_position',
            'management_quality',
            'strategic_value'
        ]
        
        for component in expected_components:
            assert component in result.component_scores
            assert 0 <= result.component_scores[component] <= 100
    
    def test_calculate_score_with_poor_company(self, scorer, sample_company_poor):
        """Test avec une entreprise peu attractive"""
        result = scorer.calculate_ma_score(sample_company_poor)
        
        assert isinstance(result, ScoreBreakdown)
        assert result.final_score <= 40, f"Score attendu ≤40, obtenu: {result.final_score}"
        assert result.final_score >= 0
        
        # Vérifier qu'il y a une pénalité qualité
        assert result.data_quality_penalty > 0
    
    def test_missing_required_data_raises_error(self, scorer):
        """Test avec données obligatoires manquantes"""
        incomplete_data = {'chiffre_affaires': 5_000_000}  # Manque siren et nom
        
        with pytest.raises(ValueError, match="Champs obligatoires manquants"):
            scorer.calculate_ma_score(incomplete_data)
    
    def test_minimal_data_returns_score(self, scorer):
        """Test avec données minimales"""
        minimal_data = {
            'siren': '123456789',
            'nom_entreprise': 'Test Company'
        }
        
        result = scorer.calculate_ma_score(minimal_data)
        assert isinstance(result, ScoreBreakdown)
        assert 0 <= result.final_score <= 100
    
    def test_score_interpretation(self, scorer):
        """Test des interprétations de score"""
        test_cases = [
            (95, "EXCELLENT"),
            (85, "EXCELLENT"), 
            (75, "TRÈS BON"),
            (70, "TRÈS BON"),
            (60, "BON"),
            (45, "MOYEN"),
            (30, "FAIBLE"),
            (15, "TRÈS FAIBLE")
        ]
        
        for score, expected_level in test_cases:
            level, description = scorer.get_score_interpretation(score)
            assert level == expected_level, f"Score {score}: attendu {expected_level}, obtenu {level}"
            assert isinstance(description, str)
            assert len(description) > 10  # Description non vide
    
    def test_custom_weights_affect_score(self, sample_company_excellent):
        """Test que les pondérations personnalisées affectent le score"""
        # Scorer par défaut
        default_scorer = MAScoring()
        default_result = default_scorer.calculate_ma_score(sample_company_excellent)
        
        # Scorer axé croissance
        growth_scorer = MAScoring(weights=create_growth_focused_weights())
        growth_result = growth_scorer.calculate_ma_score(sample_company_excellent)
        
        # Scorer conservateur
        conservative_scorer = MAScoring(weights=create_conservative_weights())
        conservative_result = conservative_scorer.calculate_ma_score(sample_company_excellent)
        
        # Les scores doivent être différents
        scores = [default_result.final_score, growth_result.final_score, conservative_result.final_score]
        assert len(set(scores)) > 1, "Les pondérations différentes doivent donner des scores différents"
    
    def test_financial_performance_scoring(self, scorer):
        """Test spécifique du scoring de performance financière"""
        test_cases = [
            ({'chiffre_affaires': 25_000_000}, 90),    # Excellent
            ({'chiffre_affaires': 15_000_000}, 75),    # Bon
            ({'chiffre_affaires': 5_000_000}, 50),     # Correct
            ({'chiffre_affaires': 2_000_000}, 25),     # Faible
            ({'chiffre_affaires': 0}, 0),              # Nul
        ]
        
        for data, expected_min_score in test_cases:
            data.update({'siren': '123456789', 'nom_entreprise': 'Test'})
            score = scorer._score_financial_performance(data)
            assert score >= expected_min_score - 10, f"CA {data['chiffre_affaires']}: score {score}, attendu ≥{expected_min_score-10}"
    
    def test_growth_trajectory_scoring(self, scorer):
        """Test spécifique du scoring de croissance"""
        test_cases = [
            ({'evolution_ca_3ans': 20}, 90),   # Excellente croissance
            ({'evolution_ca_3ans': 10}, 75),   # Bonne croissance
            ({'evolution_ca_3ans': 5}, 60),    # Croissance correcte
            ({'evolution_ca_3ans': 0}, 25),    # Stagnation
            ({'evolution_ca_3ans': -10}, 10),  # Décroissance
        ]
        
        for data, expected_min_score in test_cases:
            score = scorer._score_growth_trajectory(data)
            assert score >= expected_min_score - 10, f"Évolution {data['evolution_ca_3ans']}%: score {score}, attendu ≥{expected_min_score-10}"
    
    def test_data_quality_penalty(self, scorer):
        """Test du calcul de pénalité qualité"""
        # Données complètes - pénalité faible
        complete_data = {
            'siren': '123456789',
            'nom_entreprise': 'Test',
            'chiffre_affaires': 5_000_000,
            'resultat': 500_000,
            'effectif': 50,
            'date_creation': '2015-01-01',
            'forme_juridique': 'SAS',
            'code_naf': '6920Z',
            'last_scraped_at': datetime.now().isoformat()
        }
        penalty_complete = scorer._calculate_data_quality_penalty(complete_data)
        
        # Données incomplètes - pénalité élevée
        incomplete_data = {
            'siren': '123456789',
            'nom_entreprise': 'Test',
            'last_scraped_at': (datetime.now() - timedelta(days=200)).isoformat()
        }
        penalty_incomplete = scorer._calculate_data_quality_penalty(incomplete_data)
        
        assert penalty_incomplete > penalty_complete
        assert penalty_complete >= 0
        assert penalty_incomplete <= 20  # Pénalité maximale


class TestAsyncScoring:
    """Tests pour les fonctions async"""
    
    @pytest.mark.asyncio
    async def test_async_scoring_function(self):
        """Test de la fonction async de scoring"""
        company_data = {
            'siren': '123456789',
            'nom_entreprise': 'Test Async Company',
            'chiffre_affaires': 8_000_000,
            'effectif': 60
        }
        
        result = await calculate_company_ma_score(company_data)
        
        assert isinstance(result, ScoreBreakdown)
        assert 0 <= result.final_score <= 100
        assert result.calculation_timestamp
    
    @pytest.mark.asyncio
    async def test_async_scoring_with_custom_weights(self):
        """Test de la fonction async avec pondérations personnalisées"""
        company_data = {
            'siren': '123456789',
            'nom_entreprise': 'Test Custom Weights',
            'chiffre_affaires': 10_000_000,
            'evolution_ca_3ans': 15.0
        }
        
        custom_weights = create_growth_focused_weights()
        result = await calculate_company_ma_score(company_data, custom_weights)
        
        assert isinstance(result, ScoreBreakdown)
        assert result.component_weights['growth_trajectory'] == 0.30  # Pondération growth-focused


class TestScoreConsistency:
    """Tests de cohérence et de robustesse"""
    
    def test_score_deterministic(self, scorer):
        """Le score doit être déterministe pour des données identiques"""
        company_data = {
            'siren': '123456789',
            'nom_entreprise': 'Test Deterministic',
            'chiffre_affaires': 7_500_000,
            'effectif': 45,
            'resultat': 750_000
        }
        
        result1 = scorer.calculate_ma_score(company_data)
        result2 = scorer.calculate_ma_score(company_data)
        
        assert result1.final_score == result2.final_score
    
    def test_score_monotonicity_ca(self, scorer):
        """Score doit croître avec le chiffre d'affaires (toutes choses égales)"""
        base_data = {
            'siren': '123456789',
            'nom_entreprise': 'Test Monotonic',
            'effectif': 50
        }
        
        cas = [3_000_000, 5_000_000, 10_000_000, 20_000_000]
        scores = []
        
        for ca in cas:
            data = base_data.copy()
            data['chiffre_affaires'] = ca
            result = scorer.calculate_ma_score(data)
            scores.append(result.final_score)
        
        # Vérifier que les scores croissent globalement
        assert scores[-1] > scores[0], f"Scores: {scores}"
    
    def test_boundary_conditions(self, scorer):
        """Test des conditions limites"""
        boundary_cases = [
            # CA exactement aux seuils
            {'chiffre_affaires': 3_000_000, 'effectif': 30},    # Seuils minimum
            {'chiffre_affaires': 10_000_000, 'effectif': 50},   # Seuils intermédiaires
            {'chiffre_affaires': 20_000_000, 'effectif': 100},  # Seuils excellents
            
            # Valeurs extrêmes
            {'chiffre_affaires': 1, 'effectif': 1},             # Très petite entreprise
            {'chiffre_affaires': 100_000_000, 'effectif': 1000}, # Très grande entreprise
        ]
        
        for data in boundary_cases:
            data.update({'siren': '123456789', 'nom_entreprise': 'Test Boundary'})
            result = scorer.calculate_ma_score(data)
            
            # Le score doit toujours être dans [0, 100]
            assert 0 <= result.final_score <= 100, f"Score hors limites pour {data}: {result.final_score}"


# Tests d'intégration
class TestScoringIntegration:
    """Tests d'intégration avec d'autres composants"""
    
    def test_scorebreakdown_serialization(self, scorer):
        """Test que ScoreBreakdown peut être sérialisé en JSON"""
        import json
        
        company_data = {
            'siren': '123456789',
            'nom_entreprise': 'Test Serialization',
            'chiffre_affaires': 6_000_000
        }
        
        result = scorer.calculate_ma_score(company_data)
        
        # Conversion en dict puis JSON
        result_dict = {
            'final_score': result.final_score,
            'component_scores': result.component_scores,
            'component_weights': result.component_weights,
            'weighted_scores': result.weighted_scores,
            'data_quality_penalty': result.data_quality_penalty,
            'warnings': result.warnings,
            'calculation_timestamp': result.calculation_timestamp
        }
        
        # Doit pouvoir être sérialisé sans erreur
        json_str = json.dumps(result_dict)
        assert isinstance(json_str, str)
        
        # Et désérialisé
        deserialized = json.loads(json_str)
        assert deserialized['final_score'] == result.final_score


if __name__ == "__main__":
    # Exécution directe des tests pour développement
    pytest.main([__file__, "-v"])