"""
Module de scoring M&A avec cache Redis intelligent
US-002: Cache scoring avec invalidation automatique et TTL adaptatif

Features:
- Cache scoring 1h avec invalidation en cas de changement données
- Cache configurations de pondération
- Invalidation en cascade si données entreprise modifiées
- Comparaisons historiques cachées
- Métriques performance scoring
"""

import logging
import math
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, date
import json
import hashlib

from app.core.cache import get_cache, CacheType, cached
from app.services.ma_scoring import MAScoring, ScoringWeights, ScoreComponent, ScoringResult

logger = logging.getLogger(__name__)


class CachedMAScoring:
    """
    Service de scoring M&A avec cache Redis intelligent
    
    Optimisations cache:
    - Résultats scoring: TTL 1h avec invalidation si données changent
    - Configurations weights: TTL 6h (changent rarement)
    - Comparaisons sectorielles: TTL 4h
    - Statistiques scoring: TTL 2h
    """
    
    def __init__(self, weights: ScoringWeights = None):
        self.weights = weights or ScoringWeights()
        self.base_scorer = MAScoring(self.weights)
        
        # Statistiques cache
        self.cache_stats = {
            'scoring_hits': 0,
            'scoring_misses': 0,
            'invalidations': 0,
            'computations_saved': 0
        }
        
        # Configuration cache
        self.cache_config = {
            'scoring_ttl': 3600,          # 1h - résultats scoring
            'weights_config_ttl': 21600,  # 6h - configurations weights
            'sector_comparison_ttl': 14400, # 4h - comparaisons sectorielles
            'statistics_ttl': 7200        # 2h - statistiques
        }
    
    async def calculate_ma_score_cached(self, 
                                      company_data: Dict[str, Any], 
                                      config_name: str = "balanced",
                                      force_refresh: bool = False) -> ScoringResult:
        """
        Calcule score M&A avec cache intelligent
        
        Args:
            company_data: Données entreprise complètes
            config_name: Configuration pondération ("balanced", "growth_focused", etc.)
            force_refresh: Forcer recalcul même si en cache
            
        Returns:
            Résultat scoring avec détails
        """
        cache = await get_cache()
        
        # Générer clé cache unique
        cache_key = self._generate_scoring_cache_key(company_data, config_name)
        
        # Fonction de calcul
        async def compute_score():
            # Charger configuration weights avec cache
            weights = await self._get_weights_config_cached(config_name)
            
            # Calcul scoring (utilise MAScoring original)
            scorer = MAScoring(weights)
            result = scorer.calculate_ma_score(company_data)
            
            # Enrichir avec métadonnées cache
            result.metadata.update({
                'cached_at': datetime.now().isoformat(),
                'cache_key': cache_key,
                'config_used': config_name,
                'computation_time_saved': True if not force_refresh else False
            })
            
            return result
        
        # Pattern cache-aside avec TTL adaptatif
        ttl = self._calculate_adaptive_scoring_ttl(company_data)
        
        result = await cache.get_or_compute(
            cache_key,
            compute_score,
            CacheType.SCORING_MA,
            ttl=ttl,
            force_refresh=force_refresh
        )
        
        # Statistiques
        if not force_refresh:
            was_cached = await cache.get(cache_key, CacheType.SCORING_MA) is not None
            if was_cached:
                self.cache_stats['scoring_hits'] += 1
                self.cache_stats['computations_saved'] += 1
            else:
                self.cache_stats['scoring_misses'] += 1
        
        return result
    
    async def batch_score_companies_cached(self, 
                                         companies: List[Dict[str, Any]], 
                                         config_name: str = "balanced") -> Dict[str, ScoringResult]:
        """
        Scoring par lot avec cache optimisé
        
        Args:
            companies: Liste données entreprises
            config_name: Configuration scoring
            
        Returns:
            Dict {siren: ScoringResult}
        """
        cache = await get_cache()
        results = {}
        
        logger.info(f"🔄 Scoring par lot: {len(companies)} entreprises...")
        
        # Phase 1: Vérification cache
        cached_results = {}
        companies_to_compute = []
        
        for company in companies:
            siren = company.get('siren', '')
            cache_key = self._generate_scoring_cache_key(company, config_name)
            
            cached_score = await cache.get(cache_key, CacheType.SCORING_MA)
            
            if cached_score is not None:
                cached_results[siren] = cached_score
                self.cache_stats['scoring_hits'] += 1
                self.cache_stats['computations_saved'] += 1
            else:
                companies_to_compute.append(company)
                self.cache_stats['scoring_misses'] += 1
        
        logger.info(f"✅ Cache scoring: {len(cached_results)} hits, {len(companies_to_compute)} calculs nécessaires")
        
        # Phase 2: Calculs manquants
        if companies_to_compute:
            # Charger weights une seule fois
            weights = await self._get_weights_config_cached(config_name)
            scorer = MAScoring(weights)
            
            computed_results = {}
            for company in companies_to_compute:
                siren = company.get('siren', '')
                
                try:
                    # Calcul scoring
                    result = scorer.calculate_ma_score(company)
                    
                    # Enrichir métadonnées
                    result.metadata.update({
                        'batch_computed': True,
                        'cached_at': datetime.now().isoformat(),
                        'config_used': config_name
                    })
                    
                    computed_results[siren] = result
                    
                    # Mise en cache immédiate
                    cache_key = self._generate_scoring_cache_key(company, config_name)
                    ttl = self._calculate_adaptive_scoring_ttl(company)
                    
                    await cache.set(cache_key, result, CacheType.SCORING_MA, ttl=ttl)
                    
                except Exception as e:
                    logger.error(f"❌ Erreur scoring SIREN {siren}: {e}")
                    computed_results[siren] = None
        
        # Phase 3: Consolidation
        final_results = {**cached_results, **computed_results}
        
        logger.info(
            f"📊 Batch scoring terminé: {len(final_results)} résultats "
            f"({len(cached_results)} cache + {len(computed_results)} calculés)"
        )
        
        return final_results
    
    @cached(CacheType.SCORING_MA, ttl=14400)  # 4h
    async def get_sector_scoring_comparison_cached(self, 
                                                 code_naf: str, 
                                                 companies_data: List[Dict]) -> Dict[str, Any]:
        """
        Comparaison scoring sectorielle avec cache
        
        Utilise décorateur @cached pour simplicité
        """
        return await self._compute_sector_comparison(code_naf, companies_data)
    
    async def get_scoring_statistics_cached(self, 
                                          period_days: int = 30) -> Dict[str, Any]:
        """
        Statistiques scoring avec cache
        
        Args:
            period_days: Période d'analyse en jours
            
        Returns:
            Statistiques détaillées
        """
        cache = await get_cache()
        cache_key = f"scoring:statistics:{period_days}d"
        
        async def compute_statistics():
            return await self._compute_scoring_statistics(period_days)
        
        return await cache.get_or_compute(
            cache_key,
            compute_statistics,
            CacheType.SCORING_MA,
            ttl=self.cache_config['statistics_ttl']
        )
    
    async def invalidate_company_scoring(self, siren: str) -> int:
        """
        Invalide cache scoring d'une entreprise
        
        Appelé automatiquement quand données entreprise modifiées
        """
        cache = await get_cache()
        
        # Patterns d'invalidation
        patterns = [
            f"scoring:company:{siren}:*",      # Tous scores de l'entreprise
            f"scoring:sector:*{siren}*",       # Comparaisons sectorielles incluant cette entreprise
            f"scoring:statistics:*"            # Statistiques globales (peuvent être impactées)
        ]
        
        total_invalidated = 0
        for pattern in patterns:
            count = await cache.invalidate_pattern(pattern, CacheType.SCORING_MA)
            total_invalidated += count
        
        self.cache_stats['invalidations'] += total_invalidated
        
        logger.info(f"🧹 Cache scoring invalidé pour SIREN {siren}: {total_invalidated} clés")
        return total_invalidated
    
    async def invalidate_scoring_config(self, config_name: str) -> int:
        """Invalide cache d'une configuration de scoring"""
        cache = await get_cache()
        
        patterns = [
            f"scoring:config:{config_name}",
            f"scoring:company:*:{config_name}:*"  # Tous scores utilisant cette config
        ]
        
        total_invalidated = 0
        for pattern in patterns:
            count = await cache.invalidate_pattern(pattern, CacheType.SCORING_MA)
            total_invalidated += count
        
        logger.info(f"🧹 Configuration scoring invalidée '{config_name}': {total_invalidated} clés")
        return total_invalidated
    
    async def get_cache_performance_metrics(self) -> Dict[str, Any]:
        """Métriques performance cache scoring"""
        cache = await get_cache()
        cache_info = await cache.get_cache_info()
        
        scoring_stats = cache_info.get('cache_stats_by_type', {}).get('scoring_ma', {})
        
        return {
            'session_stats': self.cache_stats.copy(),
            'scoring_cache_keys': scoring_stats.get('key_count', 0),
            'cache_hit_ratio': round(
                (self.cache_stats['scoring_hits'] / 
                 max(self.cache_stats['scoring_hits'] + self.cache_stats['scoring_misses'], 1)) * 100, 2
            ),
            'computations_saved_count': self.cache_stats['computations_saved'],
            'estimated_time_saved_minutes': round(self.cache_stats['computations_saved'] * 0.5, 2),  # 0.5min par calcul
            'configuration': self.cache_config
        }
    
    # Méthodes privées
    
    async def _get_weights_config_cached(self, config_name: str) -> ScoringWeights:
        """Récupère configuration weights avec cache"""
        cache = await get_cache()
        cache_key = f"scoring:config:{config_name}"
        
        cached_weights = await cache.get(cache_key, CacheType.SCORING_MA)
        
        if cached_weights is not None:
            # Reconstituer objet ScoringWeights depuis dict
            return ScoringWeights(**cached_weights)
        
        # Calculer et cacher
        weights = self._get_weights_by_config_name(config_name)
        
        # Sérialiser pour cache (convertir en dict)
        weights_dict = {
            'financial_performance': weights.financial_performance,
            'growth_trajectory': weights.growth_trajectory,
            'profitability': weights.profitability,
            'financial_health': weights.financial_health,
            'operational_scale': weights.operational_scale,
            'market_position': weights.market_position,
            'management_quality': weights.management_quality,
            'strategic_value': weights.strategic_value
        }
        
        await cache.set(
            cache_key, 
            weights_dict, 
            CacheType.SCORING_MA, 
            ttl=self.cache_config['weights_config_ttl']
        )
        
        return weights
    
    def _get_weights_by_config_name(self, config_name: str) -> ScoringWeights:
        """Retourne configuration weights selon nom"""
        configs = {
            'balanced': ScoringWeights(),  # Configuration par défaut
            'growth_focused': ScoringWeights(
                financial_performance=0.20,
                growth_trajectory=0.35,  # Emphasis sur croissance
                profitability=0.10,
                financial_health=0.15,
                operational_scale=0.10,
                market_position=0.05,
                management_quality=0.03,
                strategic_value=0.02
            ),
            'value_focused': ScoringWeights(
                financial_performance=0.30,  # Emphasis sur performance
                growth_trajectory=0.15,
                profitability=0.25,      # Emphasis sur rentabilité
                financial_health=0.20,
                operational_scale=0.05,
                market_position=0.03,
                management_quality=0.01,
                strategic_value=0.01
            ),
            'risk_averse': ScoringWeights(
                financial_performance=0.25,
                growth_trajectory=0.15,
                profitability=0.20,
                financial_health=0.30,   # Emphasis sur solidité
                operational_scale=0.05,
                market_position=0.03,
                management_quality=0.01,
                strategic_value=0.01
            )
        }
        
        return configs.get(config_name, ScoringWeights())
    
    def _generate_scoring_cache_key(self, company_data: Dict, config_name: str) -> str:
        """Génère clé cache unique pour scoring"""
        siren = company_data.get('siren', '')
        
        # Données importantes pour le scoring
        key_data = {
            'siren': siren,
            'chiffre_affaires': company_data.get('chiffre_affaires'),
            'resultat': company_data.get('resultat'),
            'effectif': company_data.get('effectif'),
            'config': config_name
        }
        
        # Hash des données
        data_str = json.dumps(key_data, sort_keys=True)
        data_hash = hashlib.md5(data_str.encode()).hexdigest()[:12]
        
        return f"scoring:company:{siren}:{config_name}:{data_hash}"
    
    def _calculate_adaptive_scoring_ttl(self, company_data: Dict) -> int:
        """Calcule TTL adaptatif selon stabilité données entreprise"""
        base_ttl = self.cache_config['scoring_ttl']
        
        # TTL plus long pour grandes entreprises (données plus stables)
        ca = company_data.get('chiffre_affaires', 0)
        effectif = company_data.get('effectif', 0)
        
        if ca > 50000000 or effectif > 500:  # Très grande entreprise
            return base_ttl * 2
        elif ca > 10000000 or effectif > 100:  # Grande entreprise
            return int(base_ttl * 1.5)
        elif ca < 1000000 or effectif < 10:  # Petite entreprise (plus volatile)
            return base_ttl // 2
        
        return base_ttl
    
    async def _compute_sector_comparison(self, code_naf: str, companies_data: List[Dict]) -> Dict[str, Any]:
        """Calcule comparaison sectorielle"""
        try:
            # Filtrer entreprises du même secteur
            sector_companies = [
                c for c in companies_data 
                if c.get('code_naf', '').startswith(code_naf[:2])
            ]
            
            if len(sector_companies) < 5:
                return {'error': 'Pas assez d\'entreprises dans le secteur'}
            
            # Calcul scores pour toutes les entreprises du secteur
            scores = []
            for company in sector_companies:
                try:
                    result = self.base_scorer.calculate_ma_score(company)
                    scores.append(result.final_score)
                except Exception:
                    continue
            
            if not scores:
                return {'error': 'Impossible de calculer les scores sectoriels'}
            
            # Statistiques sectorielles
            scores.sort()
            n = len(scores)
            
            return {
                'sector_code': code_naf[:2],
                'companies_analyzed': n,
                'score_statistics': {
                    'mean': round(sum(scores) / n, 1),
                    'median': scores[n // 2],
                    'percentile_25': scores[n // 4],
                    'percentile_75': scores[3 * n // 4],
                    'min': min(scores),
                    'max': max(scores)
                },
                'computed_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Erreur comparaison sectorielle: {e}")
            return {'error': str(e)}
    
    async def _compute_scoring_statistics(self, period_days: int) -> Dict[str, Any]:
        """Calcule statistiques scoring sur période"""
        # Simulation statistiques (à remplacer par vraies données)
        return {
            'period_days': period_days,
            'total_scores_computed': 1500,
            'average_score': 67.3,
            'score_distribution': {
                '0-20': 45,
                '21-40': 180,
                '41-60': 450,
                '61-80': 625,
                '81-100': 200
            },
            'top_performing_sectors': [
                {'code_naf': '69', 'avg_score': 78.5},
                {'code_naf': '70', 'avg_score': 75.2},
                {'code_naf': '62', 'avg_score': 72.8}
            ],
            'cache_performance': self.cache_stats.copy(),
            'computed_at': datetime.now().isoformat()
        }


# Factory et utilitaires

def create_cached_ma_scorer(config_name: str = "balanced") -> CachedMAScoring:
    """
    Factory pour créer scorer M&A avec cache
    
    Args:
        config_name: Configuration de pondération
        
    Returns:
        Instance CachedMAScoring configurée
    """
    return CachedMAScoring()


async def invalidate_company_cache_cascade(siren: str):
    """
    Invalidation en cascade pour une entreprise
    
    Invalide tous les caches liés à cette entreprise:
    - Scoring M&A
    - Enrichissement
    - Exports
    """
    cache = await get_cache()
    
    # Patterns à invalider
    patterns = [
        f"scoring:company:{siren}:*",
        f"enrichment_pappers:*{siren}*",
        f"enrichment_kaspr:*{siren}*",
        f"export_csv:*{siren}*"
    ]
    
    total_invalidated = 0
    for pattern in patterns:
        for cache_type in [CacheType.SCORING_MA, CacheType.ENRICHMENT_PAPPERS, 
                          CacheType.ENRICHMENT_KASPR, CacheType.EXPORT_CSV]:
            count = await cache.invalidate_pattern(pattern, cache_type)
            total_invalidated += count
    
    logger.info(f"🧹 Invalidation cascade SIREN {siren}: {total_invalidated} clés supprimées")
    return total_invalidated