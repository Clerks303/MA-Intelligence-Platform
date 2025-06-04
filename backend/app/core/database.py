import asyncio
import logging
from typing import AsyncGenerator, Generator
from contextlib import asynccontextmanager, contextmanager

from sqlalchemy import create_engine, event, pool, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool
import asyncpg

from app.config import settings
from app.core.security import get_password_hash

logger = logging.getLogger(__name__)

# ========================================
# CONFIGURATION SQLAlchemy - PostgreSQL Local
# ========================================

class DatabaseConfig:
    """Configuration centralisée de la base de données"""
    
    # URL de connexion PostgreSQL locale
    DATABASE_URL = f"postgresql://{settings.DB_USER}:{settings.DB_PASSWORD}@{settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}"
    
    # URL de connexion async PostgreSQL
    ASYNC_DATABASE_URL = f"postgresql+asyncpg://{settings.DB_USER}:{settings.DB_PASSWORD}@{settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}"
    
    # Configuration du pool de connexions
    POOL_CONFIG = {
        'pool_size': 20,          # Nombre de connexions persistantes
        'max_overflow': 30,       # Connexions supplémentaires en pic
        'pool_timeout': 30,       # Timeout pour obtenir une connexion
        'pool_recycle': 3600,     # Recyclage des connexions (1h)
        'pool_pre_ping': True,    # Vérification des connexions
    }
    
    # Configuration des performances
    ENGINE_CONFIG = {
        'echo': settings.DB_ECHO,           # Logs SQL si DEBUG
        'echo_pool': False,                 # Logs du pool
        'future': True,                     # SQLAlchemy 2.0 style
        'connect_args': {
            'application_name': 'ma_intelligence_platform',
            'options': '-c timezone=Europe/Paris',
        }
    }


# ========================================
# MOTEURS DE BASE DE DONNÉES
# ========================================

# Moteur synchrone principal
engine = create_engine(
    DatabaseConfig.DATABASE_URL,
    **DatabaseConfig.POOL_CONFIG,
    **DatabaseConfig.ENGINE_CONFIG
)

# Moteur asynchrone pour les opérations lourdes
async_engine = create_async_engine(
    DatabaseConfig.ASYNC_DATABASE_URL,
    **DatabaseConfig.POOL_CONFIG,
    **DatabaseConfig.ENGINE_CONFIG
)

# Sessions makers
SessionLocal = sessionmaker(
    autocommit=False, 
    autoflush=False, 
    bind=engine,
    expire_on_commit=False
)

AsyncSessionLocal = async_sessionmaker(
    async_engine,
    class_=AsyncSession,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False
)

# Base déclarative
Base = declarative_base()


# ========================================
# ÉVÉNEMENTS ET OPTIMISATIONS
# ========================================

@event.listens_for(engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    """Optimisations PostgreSQL à la connexion"""
    if 'postgresql' in str(dbapi_connection):
        with dbapi_connection.cursor() as cursor:
            # Optimisations performance
            cursor.execute("SET statement_timeout = '300s'")  # 5 min max par requête
            cursor.execute("SET lock_timeout = '10s'")        # 10s max pour les locks
            cursor.execute("SET work_mem = '256MB'")          # Mémoire pour les tris
            cursor.execute("SET maintenance_work_mem = '512MB'")  # Mémoire pour maintenance
            cursor.execute("SET shared_preload_libraries = 'pg_stat_statements'")


@event.listens_for(async_engine.sync_engine, "connect")
def set_async_pragma(dbapi_connection, connection_record):
    """Optimisations pour les connexions async"""
    if 'postgresql' in str(dbapi_connection):
        with dbapi_connection.cursor() as cursor:
            cursor.execute("SET statement_timeout = '600s'")  # Plus de temps pour async
            cursor.execute("SET work_mem = '512MB'")          # Plus de mémoire pour scraping


# ========================================
# DEPENDENCY INJECTION
# ========================================

def get_db() -> Generator[Session, None, None]:
    """
    Dependency pour obtenir une session de base de données synchrone.
    Utilisé pour les opérations CRUD classiques.
    """
    db = SessionLocal()
    try:
        yield db
    except Exception as e:
        logger.error(f"Database error in sync session: {e}")
        db.rollback()
        raise
    finally:
        db.close()


async def get_async_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency pour obtenir une session de base de données asynchrone.
    Utilisé pour les opérations de scraping et enrichissement.
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception as e:
            logger.error(f"Database error in async session: {e}")
            await session.rollback()
            raise
        finally:
            await session.close()


# ========================================
# CONTEXT MANAGERS
# ========================================

@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """Context manager pour session synchrone"""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception as e:
        logger.error(f"Database transaction error: {e}")
        db.rollback()
        raise
    finally:
        db.close()


@asynccontextmanager
async def get_async_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Context manager pour session asynchrone"""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception as e:
            logger.error(f"Async database transaction error: {e}")
            await session.rollback()
            raise
        finally:
            await session.close()


# ========================================
# FONCTIONS UTILITAIRES
# ========================================

async def check_db_connection() -> bool:
    """
    Vérifie la connexion à la base de données
    Retourne True si la connexion est OK
    """
    try:
        async with get_async_db_session() as session:
            result = await session.execute(text("SELECT 1"))
            return result.scalar() == 1
    except Exception as e:
        logger.error(f"Database connection check failed: {e}")
        return False


def check_db_connection_sync() -> bool:
    """Vérification synchrone de la connexion"""
    try:
        with get_db_session() as session:
            result = session.execute(text("SELECT 1"))
            return result.scalar() == 1
    except Exception as e:
        logger.error(f"Sync database connection check failed: {e}")
        return False


async def get_db_info() -> dict:
    """Informations sur la base de données"""
    try:
        async with get_async_db_session() as session:
            queries = [
                ("version", "SELECT version()"),
                ("current_database", "SELECT current_database()"),
                ("current_user", "SELECT current_user"),
                ("encoding", "SELECT pg_encoding_to_char(encoding) FROM pg_database WHERE datname = current_database()"),
                ("timezone", "SELECT current_setting('timezone')"),
                ("connections", "SELECT count(*) FROM pg_stat_activity WHERE datname = current_database()"),
            ]
            
            info = {}
            for key, query in queries:
                try:
                    result = await session.execute(text(query))
                    info[key] = result.scalar()
                except Exception as e:
                    info[key] = f"Error: {e}"
            
            return info
    except Exception as e:
        logger.error(f"Error getting database info: {e}")
        return {"error": str(e)}


# ========================================
# INITIALISATION DE LA BASE
# ========================================

async def init_db():
    """
    Initialisation complète de la base de données
    - Création des tables
    - Création du superuser
    - Optimisations PostgreSQL
    """
    logger.info("Initialisation de la base de données...")
    
    try:
        # Import des modèles pour créer les tables
        from app.models import user, company
        from app.models.user import User
        from app.models.company import Company, CompanyContact
        
        # Vérification de la connexion
        if not await check_db_connection():
            raise Exception("Impossible de se connecter à PostgreSQL")
        
        # Création des tables avec le moteur async
        async with async_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        
        logger.info("Tables créées avec succès")
        
        # Création du superuser initial
        await create_initial_superuser()
        
        # Optimisations PostgreSQL
        await optimize_database()
        
        logger.info("Base de données initialisée avec succès")
        
    except Exception as e:
        logger.error(f"Erreur lors de l'initialisation de la base: {e}")
        raise


async def create_initial_superuser():
    """Création du superuser initial"""
    try:
        async with get_async_db_session() as session:
            from app.models.user import User
            
            # Vérifier si le superuser existe déjà
            result = await session.execute(
                text("SELECT id FROM users WHERE username = :username"),
                {"username": settings.FIRST_SUPERUSER}
            )
            
            if not result.scalar():
                # Créer le superuser
                superuser = User(
                    username=settings.FIRST_SUPERUSER,
                    email=f"{settings.FIRST_SUPERUSER}@ma-intelligence.com",
                    hashed_password=get_password_hash(settings.FIRST_SUPERUSER_PASSWORD),
                    is_superuser=True,
                    is_active=True
                )
                session.add(superuser)
                await session.commit()
                logger.info(f"Superuser '{settings.FIRST_SUPERUSER}' créé avec succès")
            else:
                logger.info(f"Superuser '{settings.FIRST_SUPERUSER}' existe déjà")
                
    except Exception as e:
        logger.error(f"Erreur lors de la création du superuser: {e}")
        raise


async def optimize_database():
    """Optimisations PostgreSQL spécifiques à l'application"""
    optimizations = [
        # Index partiels pour améliorer les performances
        """
        CREATE INDEX CONCURRENTLY IF NOT EXISTS ix_companies_ma_targets 
        ON companies (ma_score DESC, chiffre_affaires DESC) 
        WHERE ma_score >= 70 AND chiffre_affaires >= 3000000
        """,
        
        # Index pour les recherches géographiques
        """
        CREATE INDEX CONCURRENTLY IF NOT EXISTS ix_companies_geo_search 
        ON companies (code_postal, ville) 
        WHERE statut IN ('prospect', 'a_contacter')
        """,
        
        # Index pour les contacts dirigeants
        """
        CREATE INDEX CONCURRENTLY IF NOT EXISTS ix_contacts_decision_makers 
        ON company_contacts (company_id, est_dirigeant, confidence_score DESC) 
        WHERE est_dirigeant = true
        """,
        
        # Statistiques automatiques
        "ANALYZE companies",
        "ANALYZE company_contacts",
    ]
    
    try:
        async with get_async_db_session() as session:
            for optimization in optimizations:
                try:
                    await session.execute(text(optimization))
                    logger.debug(f"Optimisation appliquée: {optimization[:50]}...")
                except Exception as e:
                    logger.warning(f"Optimisation échouée: {e}")
            
            await session.commit()
            logger.info("Optimisations PostgreSQL appliquées")
            
    except Exception as e:
        logger.error(f"Erreur lors des optimisations: {e}")


# ========================================
# MAINTENANCE
# ========================================

async def cleanup_old_data():
    """Nettoyage des données anciennes"""
    try:
        async with get_async_db_session() as session:
            # Supprimer les données de scraping anciennes (> 6 mois)
            cleanup_query = text("""
                UPDATE companies 
                SET details_complets = NULL,
                    historique_modifications = NULL
                WHERE last_scraped_at < NOW() - INTERVAL '6 months'
            """)
            
            result = await session.execute(cleanup_query)
            await session.commit()
            
            logger.info(f"Nettoyage effectué: {result.rowcount} entreprises nettoyées")
            
    except Exception as e:
        logger.error(f"Erreur lors du nettoyage: {e}")


async def vacuum_analyze():
    """Maintenance PostgreSQL: VACUUM et ANALYZE"""
    try:
        # Utilisation d'une connexion directe pour VACUUM
        async with async_engine.connect() as conn:
            await conn.execute(text("VACUUM ANALYZE companies"))
            await conn.execute(text("VACUUM ANALYZE company_contacts"))
            
        logger.info("VACUUM ANALYZE effectué")
        
    except Exception as e:
        logger.error(f"Erreur VACUUM ANALYZE: {e}")


# ========================================
# HELPER POUR MIGRATION DEPUIS SUPABASE
# ========================================

async def migrate_from_supabase(supabase_client):
    """
    Migration des données depuis Supabase vers PostgreSQL local
    (si besoin de migrer les données existantes)
    """
    try:
        # Récupérer les données Supabase
        response = supabase_client.table('cabinets_comptables').select('*').execute()
        companies_data = response.data
        
        logger.info(f"Migration de {len(companies_data)} entreprises depuis Supabase")
        
        async with get_async_db_session() as session:
            from app.models.company import Company
            
            migrated_count = 0
            
            for company_data in companies_data:
                try:
                    # Adapter les données au nouveau modèle
                    adapted_data = _adapt_supabase_data(company_data)
                    
                    # Créer l'objet Company
                    company = Company(**adapted_data)
                    session.add(company)
                    migrated_count += 1
                    
                    # Commit par batch de 100
                    if migrated_count % 100 == 0:
                        await session.commit()
                        logger.info(f"Migration: {migrated_count} entreprises migrées")
                
                except Exception as e:
                    logger.error(f"Erreur migration entreprise {company_data.get('siren', 'unknown')}: {e}")
                    continue
            
            await session.commit()
            logger.info(f"Migration terminée: {migrated_count} entreprises migrées")
            
    except Exception as e:
        logger.error(f"Erreur lors de la migration Supabase: {e}")
        raise


def _adapt_supabase_data(supabase_data: dict) -> dict:
    """Adapte les données Supabase au nouveau modèle SQLAlchemy"""
    # Mapping des champs et nettoyage
    adapted = {}
    
    field_mapping = {
        'siren': 'siren',
        'siret_siege': 'siret',
        'nom_entreprise': 'nom_entreprise',
        'forme_juridique': 'forme_juridique',
        'date_creation': 'date_creation',
        'adresse': 'adresse',
        'email': 'email',
        'telephone': 'telephone',
        'numero_tva': 'numero_tva',
        'chiffre_affaires': 'chiffre_affaires',
        'resultat': 'resultat',
        'effectif': 'effectif',
        'capital_social': 'capital_social',
        'code_naf': 'code_naf',
        'libelle_code_naf': 'libelle_code_naf',
        'dirigeant_principal': 'dirigeant_principal',
        'dirigeants_json': 'dirigeants_json',
        'score_prospection': 'score_prospection',
        'score_details': 'score_details',
        'lien_pappers': 'lien_pappers',
        'lien_societe_com': 'lien_societe_com',
        'details_complets': 'details_complets',
        'last_scraped_at': 'last_scraped_at',
    }
    
    for supabase_field, sqlalchemy_field in field_mapping.items():
        if supabase_field in supabase_data:
            adapted[sqlalchemy_field] = supabase_data[supabase_field]
    
    # Valeurs par défaut pour nouveaux champs M&A
    adapted.update({
        'statut': 'prospect',
        'potentiel_acquisition': False,
        'potentiel_cession': False,
        'priorite_contact': 'MEDIUM',
        'qualite_donnees': 0.7,  # Score par défaut
    })
    
    return adapted
