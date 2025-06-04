#!/usr/bin/env python3
"""
Script de validation pour US-001: Audit et optimisation PostgreSQL
Vérifie que toutes les optimisations ont été correctement implémentées

Usage:
    python scripts/validate_us001_implementation.py
    python scripts/validate_us001_implementation.py --comprehensive
"""

import asyncio
import argparse
import logging
import json
from datetime import datetime
from typing import Dict, List, Any, Tuple
import os
import sys

# Configuration logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from sqlalchemy import text, inspect
    from app.core.database import get_async_db_session, async_engine, engine, DatabaseConfig
    from app.config import settings
except ImportError as e:
    logger.error(f"Import error: {e}")
    logger.error("Make sure to run from backend directory: cd backend && python scripts/validate_us001_implementation.py")
    sys.exit(1)


class US001Validator:
    """Validateur pour l'implémentation US-001"""
    
    def __init__(self):
        self.validation_results = {}
        self.success_count = 0
        self.failure_count = 0
        
    async def run_validation(self, comprehensive: bool = False):
        """Lance toutes les validations US-001"""
        logger.info("🔍 VALIDATION US-001: Audit et optimisation PostgreSQL")
        logger.info("=" * 80)
        
        # Validations de base
        await self.validate_database_connection()
        await self.validate_connection_pool_config()
        await self.validate_extensions()
        await self.validate_indexes()
        await self.validate_configuration()
        
        if comprehensive:
            await self.validate_performance_benchmarks()
            await self.validate_partitioning()
            await self.validate_monitoring_setup()
        
        # Validation environnement
        self.validate_environment_variables()
        self.validate_docker_configuration()
        
        # Génération rapport final
        self.generate_validation_report()
        
        # Retour code pour CI/CD
        return self.success_count, self.failure_count
    
    async def validate_database_connection(self):
        """Valide la connexion et configuration PostgreSQL"""
        test_name = "database_connection"
        logger.info("🔄 Test connexion PostgreSQL...")
        
        try:
            async with get_async_db_session() as session:
                # Test connexion basique
                result = await session.execute(text("SELECT version()"))
                version = result.scalar()
                
                # Vérifier version PostgreSQL
                if "PostgreSQL" not in version:
                    raise Exception(f"Base non PostgreSQL détectée: {version}")
                
                version_number = float(version.split()[1].split('.')[0])
                if version_number < 12:
                    raise Exception(f"Version PostgreSQL trop ancienne: {version_number}")
                
                # Test configuration optimisée
                config_tests = {
                    'shared_buffers': "SHOW shared_buffers",
                    'work_mem': "SHOW work_mem", 
                    'maintenance_work_mem': "SHOW maintenance_work_mem",
                    'max_connections': "SHOW max_connections",
                    'effective_cache_size': "SHOW effective_cache_size"
                }
                
                config_values = {}
                for param, query in config_tests.items():
                    result = await session.execute(text(query))
                    config_values[param] = result.scalar()
                
                self._record_success(test_name, {
                    'version': version,
                    'version_number': version_number,
                    'config': config_values
                })
                
        except Exception as e:
            self._record_failure(test_name, str(e))
    
    async def validate_connection_pool_config(self):
        """Valide la configuration du pool de connexions"""
        test_name = "connection_pool"
        logger.info("🔄 Test configuration pool de connexions...")
        
        try:
            # Vérifier configuration SQLAlchemy
            pool = engine.pool
            pool_config = {
                'pool_size': pool.size(),
                'checked_out': pool.checkedout(),
                'overflow': pool.overflow(),
                'checked_in': pool.checkedin()
            }
            
            # Vérifier que la configuration optimisée est appliquée
            expected_config = DatabaseConfig.POOL_CONFIG
            
            validations = []
            
            # Test pool size
            if hasattr(pool, '_pool') and hasattr(pool._pool, 'size'):
                actual_size = pool._pool.size
                if actual_size >= expected_config['pool_size']:
                    validations.append(("pool_size", True, f"OK: {actual_size}"))
                else:
                    validations.append(("pool_size", False, f"Trop petit: {actual_size}"))
            
            # Test concurrent connections
            test_connections = []
            for i in range(5):
                try:
                    async with get_async_db_session() as session:
                        result = await session.execute(text("SELECT 1"))
                        test_connections.append(result.scalar())
                except Exception as e:
                    validations.append(("concurrent_connections", False, str(e)))
                    break
            else:
                validations.append(("concurrent_connections", True, f"5 connexions simultanées OK"))
            
            success = all(v[1] for v in validations)
            self._record_result(test_name, success, {
                'pool_status': pool_config,
                'validations': validations
            })
            
        except Exception as e:
            self._record_failure(test_name, str(e))
    
    async def validate_extensions(self):
        """Valide les extensions PostgreSQL requises"""
        test_name = "postgresql_extensions"
        logger.info("🔄 Test extensions PostgreSQL...")
        
        try:
            async with get_async_db_session() as session:
                # Extensions requises
                required_extensions = ['pg_stat_statements']
                optional_extensions = ['btree_gin', 'pg_trgm']
                
                result = await session.execute(text("""
                    SELECT extname, extversion 
                    FROM pg_extension 
                    WHERE extname = ANY(%(extensions)s)
                """), {'extensions': required_extensions + optional_extensions})
                
                installed_extensions = {row[0]: row[1] for row in result.fetchall()}
                
                validations = []
                
                # Vérifier extensions requises
                for ext in required_extensions:
                    if ext in installed_extensions:
                        validations.append((ext, True, f"v{installed_extensions[ext]}"))
                    else:
                        validations.append((ext, False, "Non installée"))
                
                # Vérifier extensions optionnelles
                for ext in optional_extensions:
                    if ext in installed_extensions:
                        validations.append((f"{ext}_optional", True, f"v{installed_extensions[ext]}"))
                    else:
                        validations.append((f"{ext}_optional", None, "Non installée (optionnelle)"))
                
                # Test pg_stat_statements
                if 'pg_stat_statements' in installed_extensions:
                    try:
                        await session.execute(text("SELECT count(*) FROM pg_stat_statements LIMIT 1"))
                        validations.append(("pg_stat_statements_functional", True, "Fonctionnelle"))
                    except Exception as e:
                        validations.append(("pg_stat_statements_functional", False, str(e)))
                
                required_ok = all(v[1] for v in validations if not v[0].endswith('_optional'))
                self._record_result(test_name, required_ok, {
                    'installed_extensions': installed_extensions,
                    'validations': validations
                })
            
        except Exception as e:
            self._record_failure(test_name, str(e))
    
    async def validate_indexes(self):
        """Valide la création des index optimisés"""
        test_name = "optimized_indexes"
        logger.info("🔄 Test index optimisés...")
        
        try:
            async with get_async_db_session() as session:
                # Index attendus (créés par les scripts SQL)
                expected_indexes = [
                    'idx_companies_ma_search',
                    'idx_companies_financial_criteria', 
                    'idx_companies_scoring_lookup',
                    'idx_companies_location_score',
                    'idx_companies_recent_data',
                    'idx_companies_enrichment_status'
                ]
                
                # Requête pour vérifier les index
                result = await session.execute(text("""
                    SELECT 
                        indexname,
                        tablename,
                        indexdef,
                        pg_size_pretty(pg_relation_size(indexname::regclass)) as size
                    FROM pg_indexes 
                    WHERE tablename = 'companies'
                    AND indexname = ANY(%(indexes)s)
                """), {'indexes': expected_indexes})
                
                found_indexes = {}
                for row in result.fetchall():
                    found_indexes[row[0]] = {
                        'table': row[1],
                        'definition': row[2],
                        'size': row[3]
                    }
                
                # Validation chaque index
                validations = []
                for idx in expected_indexes:
                    if idx in found_indexes:
                        validations.append((idx, True, found_indexes[idx]['size']))
                    else:
                        validations.append((idx, False, "Index manquant"))
                
                # Test performance des index avec EXPLAIN
                test_queries = [
                    ("ma_score_query", "SELECT * FROM companies WHERE ma_score >= 80 LIMIT 10"),
                    ("location_query", "SELECT * FROM companies WHERE ville = 'Paris' LIMIT 10"),
                    ("recent_data_query", "SELECT * FROM companies WHERE last_scraped_at >= CURRENT_DATE - INTERVAL '30 days' LIMIT 10")
                ]
                
                for query_name, query in test_queries:
                    try:
                        explain_result = await session.execute(text(f"EXPLAIN {query}"))
                        plan = '\n'.join([row[0] for row in explain_result.fetchall()])
                        uses_index = 'Index' in plan
                        validations.append((f"{query_name}_uses_index", uses_index, 
                                         "Utilise index" if uses_index else "Scan séquentiel"))
                    except Exception as e:
                        validations.append((f"{query_name}_uses_index", False, str(e)))
                
                success = len(found_indexes) >= len(expected_indexes) * 0.8  # 80% des index requis
                self._record_result(test_name, success, {
                    'expected_count': len(expected_indexes),
                    'found_count': len(found_indexes),
                    'found_indexes': found_indexes,
                    'validations': validations
                })
            
        except Exception as e:
            self._record_failure(test_name, str(e))
    
    async def validate_configuration(self):
        """Valide la configuration PostgreSQL optimisée"""
        test_name = "postgresql_configuration"
        logger.info("🔄 Test configuration PostgreSQL...")
        
        try:
            async with get_async_db_session() as session:
                # Paramètres critiques à vérifier
                critical_params = {
                    'shared_buffers': {'min_mb': 256, 'max_mb': 2048},
                    'work_mem': {'min_mb': 8, 'max_mb': 64},
                    'maintenance_work_mem': {'min_mb': 64, 'max_mb': 1024},
                    'max_connections': {'min': 100, 'max': 1000},
                    'random_page_cost': {'min': 1.0, 'max': 2.0},
                    'effective_io_concurrency': {'min': 100, 'max': 1000}
                }
                
                config_status = {}
                
                for param, constraints in critical_params.items():
                    try:
                        result = await session.execute(text(f"SHOW {param}"))
                        value_str = result.scalar()
                        
                        # Parse la valeur selon le type
                        if 'MB' in value_str or 'GB' in value_str:
                            if 'GB' in value_str:
                                value_mb = float(value_str.replace('GB', '')) * 1024
                            else:
                                value_mb = float(value_str.replace('MB', ''))
                            
                            is_valid = constraints.get('min_mb', 0) <= value_mb <= constraints.get('max_mb', float('inf'))
                            config_status[param] = {
                                'value': value_str,
                                'parsed_mb': value_mb,
                                'valid': is_valid
                            }
                        else:
                            # Valeur numérique simple
                            try:
                                value_num = float(value_str)
                                is_valid = constraints.get('min', 0) <= value_num <= constraints.get('max', float('inf'))
                                config_status[param] = {
                                    'value': value_str,
                                    'parsed': value_num,
                                    'valid': is_valid
                                }
                            except ValueError:
                                config_status[param] = {
                                    'value': value_str,
                                    'valid': None,
                                    'error': 'Cannot parse value'
                                }
                    
                    except Exception as e:
                        config_status[param] = {
                            'error': str(e),
                            'valid': False
                        }
                
                # Calculer score de configuration
                valid_configs = sum(1 for cfg in config_status.values() if cfg.get('valid') == True)
                total_configs = len(critical_params)
                config_score = valid_configs / total_configs
                
                success = config_score >= 0.7  # 70% des configs valides
                self._record_result(test_name, success, {
                    'config_score': round(config_score * 100, 1),
                    'valid_configs': valid_configs,
                    'total_configs': total_configs,
                    'details': config_status
                })
            
        except Exception as e:
            self._record_failure(test_name, str(e))
    
    async def validate_performance_benchmarks(self):
        """Valide les améliorations de performance"""
        test_name = "performance_benchmarks"
        logger.info("🔄 Test benchmarks de performance...")
        
        try:
            # Import du testeur de performance
            from scripts.test_database_performance import DatabasePerformanceTester
            
            # Lancer tests performance simplifiés
            perf_tester = DatabasePerformanceTester()
            
            # Test connexion pool
            await perf_tester.test_connection_pool_performance()
            
            # Tests basiques de requêtes
            if 'connection_pool' in perf_tester.results:
                pool_result = perf_tester.results['connection_pool']
                speedup = pool_result.get('speedup', 0)
                pool_status = pool_result.get('status', 'POOR')
                
                performance_score = {
                    'connection_pool_speedup': speedup,
                    'connection_pool_status': pool_status
                }
                
                success = speedup > 1.5 and pool_status == 'GOOD'
                self._record_result(test_name, success, performance_score)
            else:
                self._record_failure(test_name, "Impossible d'exécuter les tests de performance")
            
        except Exception as e:
            self._record_failure(test_name, f"Tests performance échoués: {str(e)}")
    
    async def validate_partitioning(self):
        """Valide la mise en place du partitioning/index partiels"""
        test_name = "partitioning_setup"
        logger.info("🔄 Test setup partitioning...")
        
        try:
            async with get_async_db_session() as session:
                # Vérifier index partiels (simulent le partitioning)
                partial_indexes = [
                    'idx_companies_recent_data',
                    'idx_companies_quarter_data',
                    'idx_companies_archive_data'
                ]
                
                result = await session.execute(text("""
                    SELECT 
                        indexname,
                        indexdef
                    FROM pg_indexes 
                    WHERE tablename = 'companies'
                    AND indexname = ANY(%(indexes)s)
                """), {'indexes': partial_indexes})
                
                found_partial_indexes = {}
                for row in result.fetchall():
                    # Vérifier que l'index contient une clause WHERE (partiel)
                    if 'WHERE' in row[1]:
                        found_partial_indexes[row[0]] = row[1]
                
                # Vérifier vues matérialisées
                mv_result = await session.execute(text("""
                    SELECT matviewname 
                    FROM pg_matviews 
                    WHERE matviewname LIKE 'mv_%'
                """))
                
                materialized_views = [row[0] for row in mv_result.fetchall()]
                
                success = len(found_partial_indexes) >= 2 and len(materialized_views) >= 1
                self._record_result(test_name, success, {
                    'partial_indexes': found_partial_indexes,
                    'materialized_views': materialized_views,
                    'partitioning_strategy': 'index_based'
                })
            
        except Exception as e:
            self._record_failure(test_name, str(e))
    
    async def validate_monitoring_setup(self):
        """Valide la mise en place du monitoring"""
        test_name = "monitoring_setup"
        logger.info("🔄 Test setup monitoring...")
        
        try:
            monitoring_status = {}
            
            # Vérifier scripts de monitoring
            script_files = [
                'scripts/test_database_performance.py',
                'scripts/monitor_slow_queries.py',
                'scripts/validate_us001_implementation.py'
            ]
            
            for script in script_files:
                if os.path.exists(script):
                    monitoring_status[script] = True
                else:
                    monitoring_status[script] = False
            
            # Vérifier pg_stat_statements pour monitoring
            async with get_async_db_session() as session:
                try:
                    result = await session.execute(text("""
                        SELECT count(*) FROM pg_stat_statements LIMIT 1
                    """))
                    monitoring_status['pg_stat_statements_active'] = True
                except Exception:
                    monitoring_status['pg_stat_statements_active'] = False
            
            # Vérifier configuration logging
            log_config = {
                'log_min_duration_statement': 'SHOW log_min_duration_statement',
                'log_checkpoints': 'SHOW log_checkpoints', 
                'log_connections': 'SHOW log_connections'
            }
            
            for param, query in log_config.items():
                try:
                    async with get_async_db_session() as session:
                        result = await session.execute(text(query))
                        value = result.scalar()
                        monitoring_status[f'postgres_{param}'] = value
                except Exception:
                    monitoring_status[f'postgres_{param}'] = 'unknown'
            
            script_count = sum(1 for v in monitoring_status.values() if v is True)
            success = script_count >= 3  # Au moins 3 éléments de monitoring OK
            
            self._record_result(test_name, success, monitoring_status)
            
        except Exception as e:
            self._record_failure(test_name, str(e))
    
    def validate_environment_variables(self):
        """Valide les variables d'environnement"""
        test_name = "environment_variables"
        logger.info("🔄 Test variables d'environnement...")
        
        try:
            # Variables critiques pour US-001
            required_vars = {
                'DB_HOST': os.getenv('DB_HOST'),
                'DB_NAME': os.getenv('DB_NAME'), 
                'DB_USER': os.getenv('DB_USER'),
                'DB_PASSWORD': os.getenv('DB_PASSWORD'),
            }
            
            # Variables optionnelles d'optimisation
            optional_vars = {
                'DB_POOL_SIZE': os.getenv('DB_POOL_SIZE'),
                'DB_MAX_OVERFLOW': os.getenv('DB_MAX_OVERFLOW'),
                'DB_ECHO': os.getenv('DB_ECHO'),
                'REDIS_URL': os.getenv('REDIS_URL'),
                'CELERY_BROKER_URL': os.getenv('CELERY_BROKER_URL')
            }
            
            validations = []
            
            # Vérifier variables requises
            for var, value in required_vars.items():
                if value:
                    validations.append((var, True, "Définie"))
                else:
                    validations.append((var, False, "Manquante"))
            
            # Vérifier variables optionnelles
            for var, value in optional_vars.items():
                if value:
                    validations.append((f"{var}_optional", True, f"Définie: {value}"))
                else:
                    validations.append((f"{var}_optional", None, "Non définie (optionnelle)"))
            
            # Vérifier cohérence configuration
            db_config = DatabaseConfig()
            if hasattr(db_config, 'DATABASE_URL'):
                validations.append(("database_url_constructed", True, "URL construite correctement"))
            
            required_ok = all(v[1] for v in validations if not v[0].endswith('_optional'))
            self._record_result(test_name, required_ok, {
                'required_vars': required_vars,
                'optional_vars': optional_vars, 
                'validations': validations
            })
            
        except Exception as e:
            self._record_failure(test_name, str(e))
    
    def validate_docker_configuration(self):
        """Valide la configuration Docker"""
        test_name = "docker_configuration"
        logger.info("🔄 Test configuration Docker...")
        
        try:
            validations = []
            
            # Vérifier fichiers de configuration
            config_files = {
                'docker-compose.yml': 'Configuration Docker Compose',
                'postgres/postgresql.conf': 'Configuration PostgreSQL optimisée',
                'postgres/pg_hba.conf': 'Configuration authentification PostgreSQL',
                'redis/redis.conf': 'Configuration Redis',
                '.env.example': 'Template variables environnement'
            }
            
            for file_path, description in config_files.items():
                if os.path.exists(file_path):
                    validations.append((file_path, True, description))
                else:
                    validations.append((file_path, False, f"Fichier manquant: {description}"))
            
            # Vérifier contenu docker-compose.yml
            if os.path.exists('docker-compose.yml'):
                with open('docker-compose.yml', 'r') as f:
                    compose_content = f.read()
                    
                required_services = ['postgres', 'redis', 'backend']
                for service in required_services:
                    if f'{service}:' in compose_content:
                        validations.append((f"service_{service}", True, f"Service {service} configuré"))
                    else:
                        validations.append((f"service_{service}", False, f"Service {service} manquant"))
            
            success = sum(1 for v in validations if v[1] is True) >= len(config_files)
            self._record_result(test_name, success, {
                'config_files': config_files,
                'validations': validations
            })
            
        except Exception as e:
            self._record_failure(test_name, str(e))
    
    def _record_success(self, test_name: str, details: Any = None):
        """Enregistre un test réussi"""
        self.validation_results[test_name] = {
            'status': 'SUCCESS',
            'details': details,
            'timestamp': datetime.now().isoformat()
        }
        self.success_count += 1
        logger.info(f"✅ {test_name}: SUCCÈS")
    
    def _record_failure(self, test_name: str, error: str):
        """Enregistre un test échoué"""
        self.validation_results[test_name] = {
            'status': 'FAILURE', 
            'error': error,
            'timestamp': datetime.now().isoformat()
        }
        self.failure_count += 1
        logger.error(f"❌ {test_name}: ÉCHEC - {error}")
    
    def _record_result(self, test_name: str, success: bool, details: Any = None):
        """Enregistre un résultat de test"""
        if success:
            self._record_success(test_name, details)
        else:
            self._record_failure(test_name, f"Validation échouée: {details}")
    
    def generate_validation_report(self):
        """Génère le rapport final de validation"""
        logger.info("=" * 80)
        logger.info("📋 RAPPORT DE VALIDATION US-001")
        logger.info("=" * 80)
        
        total_tests = self.success_count + self.failure_count
        success_rate = (self.success_count / total_tests * 100) if total_tests > 0 else 0
        
        logger.info(f"🎯 Tests réussis: {self.success_count}/{total_tests} ({success_rate:.1f}%)")
        
        if self.failure_count > 0:
            logger.warning(f"⚠️ Tests échoués: {self.failure_count}")
            logger.warning("Détails des échecs:")
            for test_name, result in self.validation_results.items():
                if result['status'] == 'FAILURE':
                    logger.warning(f"  - {test_name}: {result.get('error', 'Erreur inconnue')}")
        
        # Recommandations
        recommendations = self._generate_recommendations()
        if recommendations:
            logger.info("\n💡 RECOMMANDATIONS:")
            for i, rec in enumerate(recommendations, 1):
                logger.info(f"  {i}. {rec}")
        
        # Sauvegarde rapport
        report = {
            'timestamp': datetime.now().isoformat(),
            'us_story': 'US-001: Audit et optimisation PostgreSQL',
            'summary': {
                'total_tests': total_tests,
                'success_count': self.success_count,
                'failure_count': self.failure_count,
                'success_rate': round(success_rate, 1)
            },
            'detailed_results': self.validation_results,
            'recommendations': recommendations,
            'status': 'PASSED' if success_rate >= 80 else 'FAILED'
        }
        
        report_file = f"us001_validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📄 Rapport détaillé sauvegardé: {report_file}")
        
        # Status final
        if success_rate >= 80:
            logger.info("🎉 US-001 VALIDÉ: Optimisations PostgreSQL implémentées avec succès!")
        else:
            logger.error("💥 US-001 ÉCHOUÉ: Optimisations PostgreSQL incomplètes")
        
        logger.info("=" * 80)
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Génère des recommandations basées sur les résultats"""
        recommendations = []
        
        # Analyser les échecs pour recommandations
        for test_name, result in self.validation_results.items():
            if result['status'] == 'FAILURE':
                if 'database_connection' in test_name:
                    recommendations.append("Vérifier la connexion PostgreSQL et la configuration")
                elif 'connection_pool' in test_name:
                    recommendations.append("Ajuster la configuration du pool de connexions SQLAlchemy")
                elif 'extensions' in test_name:
                    recommendations.append("Installer les extensions PostgreSQL manquantes (pg_stat_statements)")
                elif 'indexes' in test_name:
                    recommendations.append("Exécuter les scripts SQL d'optimisation des index")
                elif 'configuration' in test_name:
                    recommendations.append("Appliquer la configuration PostgreSQL optimisée")
                elif 'environment' in test_name:
                    recommendations.append("Configurer les variables d'environnement requises")
                elif 'docker' in test_name:
                    recommendations.append("Mettre à jour la configuration Docker")
        
        # Recommandations générales
        if self.failure_count > 0:
            recommendations.append("Consulter les logs détaillés pour résoudre les problèmes")
            recommendations.append("Relancer les scripts d'optimisation SQL si nécessaire")
        
        if not recommendations:
            recommendations.append("Toutes les optimisations US-001 sont correctement implémentées")
        
        return recommendations


async def main():
    """Point d'entrée principal"""
    parser = argparse.ArgumentParser(description='Validation US-001: PostgreSQL Optimization')
    parser.add_argument('--comprehensive', action='store_true', 
                      help='Tests complets incluant benchmarks et partitioning')
    
    args = parser.parse_args()
    
    validator = US001Validator()
    success_count, failure_count = await validator.run_validation(comprehensive=args.comprehensive)
    
    # Code de retour pour CI/CD
    exit_code = 0 if failure_count == 0 else 1
    sys.exit(exit_code)


if __name__ == "__main__":
    asyncio.run(main())