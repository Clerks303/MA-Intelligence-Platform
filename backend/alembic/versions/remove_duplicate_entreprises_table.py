"""Remove duplicate entreprises table

Revision ID: remove_entreprises_001
Revises: ma_enriched_001
Create Date: 2025-01-06 12:00:00.000000

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = 'remove_entreprises_001'
down_revision = 'ma_enriched_001'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Remove the duplicate entreprises table - all data now in unified companies table"""
    
    # Drop the entreprises table if it exists
    op.execute("""
        DROP TABLE IF EXISTS entreprises CASCADE;
    """)
    
    # Drop any related indexes if they exist
    op.execute("""
        DROP INDEX IF EXISTS ix_entreprises_siren;
        DROP INDEX IF EXISTS ix_entreprises_nom_entreprise;
        DROP INDEX IF EXISTS ix_entreprises_statut;
        DROP INDEX IF EXISTS ix_entreprises_chiffre_affaires;
        DROP INDEX IF EXISTS ix_entreprises_effectif;
    """)


def downgrade() -> None:
    """Recreate the basic entreprises table (for rollback only)"""
    
    # Create a minimal entreprises table for rollback compatibility
    op.create_table('entreprises',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('siren', sa.String(9), unique=True, nullable=False, index=True),
        sa.Column('siret', sa.String(14), index=True),
        sa.Column('nom_entreprise', sa.String(255), nullable=False, index=True),
        sa.Column('forme_juridique', sa.String(100)),
        sa.Column('date_creation', sa.DateTime(timezone=True)),
        sa.Column('adresse', sa.Text),
        sa.Column('ville', sa.String(100)),
        sa.Column('code_postal', sa.String(10)),
        sa.Column('email', sa.String(255)),
        sa.Column('telephone', sa.String(20)),
        sa.Column('numero_tva', sa.String(20)),
        sa.Column('chiffre_affaires', sa.Float),
        sa.Column('resultat', sa.Float),
        sa.Column('effectif', sa.Integer),
        sa.Column('capital_social', sa.Float),
        sa.Column('code_naf', sa.String(10)),
        sa.Column('libelle_code_naf', sa.String(255)),
        sa.Column('dirigeant_principal', sa.String(255)),
        sa.Column('statut', sa.String(50), default='à contacter'),
        sa.Column('last_scraped_at', sa.DateTime(timezone=True)),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), onupdate=sa.func.now(), server_default=sa.func.now()),
    )
    
    # Create basic indexes
    op.create_index('ix_entreprises_siren', 'entreprises', ['siren'])
    op.create_index('ix_entreprises_nom_entreprise', 'entreprises', ['nom_entreprise'])
    op.create_index('ix_entreprises_statut', 'entreprises', ['statut'])
    op.create_index('ix_entreprises_chiffre_affaires', 'entreprises', ['chiffre_affaires'])
    op.create_index('ix_entreprises_effectif', 'entreprises', ['effectif'])