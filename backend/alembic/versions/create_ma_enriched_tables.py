"""Create MA enriched tables

Revision ID: ma_enriched_001
Revises: 
Create Date: 2025-01-30 10:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
import uuid

# revision identifiers, used by Alembic.
revision = 'ma_enriched_001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create the enhanced MA tables"""
    
    # Create enum types
    op.execute("CREATE TYPE statusenum AS ENUM ('prospect', 'a_contacter', 'contacte', 'qualifie', 'propose', 'refuse', 'en_cours', 'clos_gagne', 'clos_perdu')")
    op.execute("CREATE TYPE enrichmentsourceenum AS ENUM ('pappers', 'infogreffe', 'societe', 'kaspr', 'manual')")
    op.execute("CREATE TYPE contacttypeenum AS ENUM ('dirigeant', 'comptable', 'commercial', 'rh', 'autre')")
    
    # Create companies table
    op.create_table('companies',
        # === IDENTIFIANTS ===
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column('siren', sa.String(9), unique=True, nullable=False, index=True),
        sa.Column('siret', sa.String(14), index=True),
        
        # === INFORMATIONS GÉNÉRALES ===
        sa.Column('nom_entreprise', sa.String(255), nullable=False, index=True),
        sa.Column('forme_juridique', sa.String(100)),
        sa.Column('date_creation', sa.DateTime(timezone=True)),
        
        # === COORDONNÉES ===
        sa.Column('adresse', sa.Text),
        sa.Column('ville', sa.String(100), index=True),
        sa.Column('code_postal', sa.String(10), index=True),
        sa.Column('email', sa.String(255)),
        sa.Column('telephone', sa.String(20)),
        sa.Column('numero_tva', sa.String(20)),
        sa.Column('site_web', sa.String(500)),
        
        # === DONNÉES FINANCIÈRES ===
        sa.Column('chiffre_affaires', sa.Float, index=True),
        sa.Column('chiffre_affaires_n1', sa.Float),
        sa.Column('chiffre_affaires_n2', sa.Float),
        sa.Column('resultat', sa.Float),
        sa.Column('resultat_n1', sa.Float),
        sa.Column('resultat_n2', sa.Float),
        sa.Column('effectif', sa.Integer, index=True),
        sa.Column('capital_social', sa.Float),
        
        # === DONNÉES M&A ENRICHIES ===
        sa.Column('evolution_ca_3ans', sa.Float),
        sa.Column('marge_nette', sa.Float),
        sa.Column('ratio_endettement', sa.Float),
        sa.Column('rentabilite_capitaux', sa.Float),
        sa.Column('croissance_effectif', sa.Float),
        
        # === SCORING M&A ===
        sa.Column('ma_score', sa.Float, index=True),
        sa.Column('ma_score_details', sa.JSON),
        sa.Column('potentiel_acquisition', sa.Boolean, default=False, index=True),
        sa.Column('potentiel_cession', sa.Boolean, default=False, index=True),
        sa.Column('priorite_contact', sa.String(20), index=True),
        
        # === ACTIVITÉ ===
        sa.Column('code_naf', sa.String(10), index=True),
        sa.Column('libelle_code_naf', sa.String(255)),
        sa.Column('secteur_activite', sa.String(100), index=True),
        sa.Column('specialisation', sa.Text),
        
        # === DIRECTION ===
        sa.Column('dirigeant_principal', sa.String(255)),
        sa.Column('dirigeants_json', sa.JSON),
        sa.Column('age_dirigeant_principal', sa.Integer),
        sa.Column('anciennete_dirigeant', sa.Integer),
        
        # === PROSPECTION ===
        sa.Column('statut', postgresql.ENUM('prospect', 'a_contacter', 'contacte', 'qualifie', 'propose', 'refuse', 'en_cours', 'clos_gagne', 'clos_perdu', name='statusenum'), default='prospect', index=True),
        sa.Column('score_prospection', sa.Float),
        sa.Column('score_details', sa.JSON),
        sa.Column('description', sa.Text),
        sa.Column('notes_commerciales', sa.Text),
        sa.Column('derniere_actualite', sa.Text),
        sa.Column('date_derniere_actualite', sa.DateTime(timezone=True)),
        
        # === VEILLE ET SIGNAUX ===
        sa.Column('signaux_faibles', sa.JSON),
        sa.Column('actualites_recentes', sa.JSON),
        sa.Column('mentions_presse', sa.JSON),
        sa.Column('changements_dirigeants', sa.JSON),
        
        # === ENRICHISSEMENT ===
        sa.Column('enrichment_status', sa.JSON),
        sa.Column('sources_enrichissement', sa.JSON),
        sa.Column('derniere_verification', sa.DateTime(timezone=True)),
        sa.Column('qualite_donnees', sa.Float),
        sa.Column('donnees_manquantes', sa.JSON),
        
        # === LIENS EXTERNES ===
        sa.Column('lien_pappers', sa.String(500)),
        sa.Column('lien_societe_com', sa.String(500)),
        sa.Column('lien_infogreffe', sa.String(500)),
        sa.Column('lien_linkedin', sa.String(500)),
        
        # === MÉTADONNÉES ===
        sa.Column('details_complets', sa.JSON),
        sa.Column('historique_modifications', sa.JSON),
        sa.Column('tags', sa.JSON),
        
        # === HORODATAGE ===
        sa.Column('last_scraped_at', sa.DateTime(timezone=True)),
        sa.Column('last_enriched_pappers', sa.DateTime(timezone=True)),
        sa.Column('last_enriched_infogreffe', sa.DateTime(timezone=True)),
        sa.Column('last_enriched_societe', sa.DateTime(timezone=True)),
        sa.Column('last_enriched_kaspr', sa.DateTime(timezone=True)),
        sa.Column('last_scored_at', sa.DateTime(timezone=True)),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), onupdate=sa.func.now(), server_default=sa.func.now()),
    )
    
    # Create company_contacts table
    op.create_table('company_contacts',
        # === IDENTIFIANTS ===
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column('company_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('companies.id', ondelete='CASCADE'), nullable=False),
        
        # === INFORMATIONS PERSONNELLES ===
        sa.Column('nom_complet', sa.String(255), nullable=False),
        sa.Column('prenom', sa.String(100)),
        sa.Column('nom', sa.String(100)),
        sa.Column('civilite', sa.String(10)),
        
        # === FONCTION ===
        sa.Column('type_contact', postgresql.ENUM('dirigeant', 'comptable', 'commercial', 'rh', 'autre', name='contacttypeenum'), nullable=False, index=True),
        sa.Column('poste', sa.String(255)),
        sa.Column('qualite', sa.String(100)),
        sa.Column('anciennete_poste', sa.Integer),
        sa.Column('est_dirigeant', sa.Boolean, default=False, index=True),
        sa.Column('est_actionnaire', sa.Boolean, default=False),
        sa.Column('pourcentage_participation', sa.Float),
        
        # === COORDONNÉES ===
        sa.Column('email_professionnel', sa.String(255)),
        sa.Column('email_personnel', sa.String(255)),
        sa.Column('telephone_direct', sa.String(20)),
        sa.Column('telephone_mobile', sa.String(20)),
        sa.Column('linkedin_url', sa.String(500)),
        
        # === MÉTADONNÉES D'ENRICHISSEMENT ===
        sa.Column('source', postgresql.ENUM('pappers', 'infogreffe', 'societe', 'kaspr', 'manual', name='enrichmentsourceenum'), nullable=False),
        sa.Column('confidence_score', sa.Float),
        sa.Column('derniere_verification', sa.DateTime(timezone=True)),
        sa.Column('statut_email', sa.String(20)),
        sa.Column('statut_telephone', sa.String(20)),
        
        # === DONNÉES COMPLÉMENTAIRES ===
        sa.Column('age_estime', sa.Integer),
        sa.Column('formation', sa.String(255)),
        sa.Column('experience_precedente', sa.Text),
        sa.Column('reseaux_sociaux', sa.JSON),
        
        # === PROSPECTION ===
        sa.Column('derniere_interaction', sa.DateTime(timezone=True)),
        sa.Column('type_derniere_interaction', sa.String(50)),
        sa.Column('notes_contact', sa.Text),
        sa.Column('score_accessibilite', sa.Float),
        sa.Column('preferences_contact', sa.JSON),
        
        # === HORODATAGE ===
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), onupdate=sa.func.now(), server_default=sa.func.now()),
    )
    
    # Create composite indexes for optimal performance
    op.create_index('ix_companies_ma_score_ca', 'companies', ['ma_score', 'chiffre_affaires'])
    op.create_index('ix_companies_geo', 'companies', ['code_postal', 'ville'])
    op.create_index('ix_companies_activity', 'companies', ['code_naf', 'secteur_activite'])
    op.create_index('ix_companies_prospection', 'companies', ['statut', 'priorite_contact'])
    op.create_index('ix_companies_financial', 'companies', ['chiffre_affaires', 'effectif', 'ma_score'])
    
    op.create_index('ix_contacts_company_type', 'company_contacts', ['company_id', 'type_contact'])
    op.create_index('ix_contacts_dirigeant', 'company_contacts', ['est_dirigeant', 'confidence_score'])
    op.create_index('ix_contacts_email', 'company_contacts', ['email_professionnel', 'statut_email'])
    
    # Create partial indexes for MA targets (high performance queries)
    op.execute("""
        CREATE INDEX CONCURRENTLY ix_companies_ma_targets 
        ON companies (ma_score DESC, chiffre_affaires DESC) 
        WHERE ma_score >= 70 AND chiffre_affaires >= 3000000
    """)
    
    op.execute("""
        CREATE INDEX CONCURRENTLY ix_companies_geo_search 
        ON companies (code_postal, ville) 
        WHERE statut IN ('prospect', 'a_contacter')
    """)
    
    op.execute("""
        CREATE INDEX CONCURRENTLY ix_contacts_decision_makers 
        ON company_contacts (company_id, est_dirigeant, confidence_score DESC) 
        WHERE est_dirigeant = true
    """)


def downgrade() -> None:
    """Drop the enhanced MA tables"""
    
    # Drop partial indexes first
    op.execute("DROP INDEX IF EXISTS ix_companies_ma_targets")
    op.execute("DROP INDEX IF EXISTS ix_companies_geo_search") 
    op.execute("DROP INDEX IF EXISTS ix_contacts_decision_makers")
    
    # Drop composite indexes
    op.drop_index('ix_companies_ma_score_ca', 'companies')
    op.drop_index('ix_companies_geo', 'companies')
    op.drop_index('ix_companies_activity', 'companies')
    op.drop_index('ix_companies_prospection', 'companies')
    op.drop_index('ix_companies_financial', 'companies')
    
    op.drop_index('ix_contacts_company_type', 'company_contacts')
    op.drop_index('ix_contacts_dirigeant', 'company_contacts')
    op.drop_index('ix_contacts_email', 'company_contacts')
    
    # Drop tables
    op.drop_table('company_contacts')
    op.drop_table('companies')
    
    # Drop enum types
    op.execute("DROP TYPE IF EXISTS contacttypeenum")
    op.execute("DROP TYPE IF EXISTS enrichmentsourceenum")
    op.execute("DROP TYPE IF EXISTS statusenum")