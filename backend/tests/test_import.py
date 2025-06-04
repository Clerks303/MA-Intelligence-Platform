import pytest
import tempfile
import io
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.main import app
from app.core.database import Base, get_db
from app.models.company import Company
# Using unified Company model (removed duplicate Entreprise model)
from app.services.data_processing import process_csv_file
from fastapi import UploadFile

# Test database setup
SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def override_get_db():
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()

app.dependency_overrides[get_db] = override_get_db

@pytest.fixture(scope="function")
def db_session():
    """Create a fresh database for each test"""
    Base.metadata.create_all(bind=engine)
    db = TestingSessionLocal()
    yield db
    db.close()
    Base.metadata.drop_all(bind=engine)

@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture
def sample_csv_content():
    """Sample CSV content for testing"""
    return """siren,nom_entreprise,email,telephone,chiffre_affaires,effectif
123456789,Entreprise Test 1,test1@example.com,0123456789,1000000,10
987654321,Entreprise Test 2,test2@example.com,0987654321,2000000,20
555666777,Entreprise Test 3,test3@example.com,0555666777,3000000,30"""

def create_upload_file(content: str, filename: str = "test.csv"):
    """Helper to create UploadFile for testing"""
    file_obj = io.BytesIO(content.encode())
    return UploadFile(filename=filename, file=file_obj)

class TestCSVImport:
    
    def test_nouvelle_entreprise_insertion(self, db_session, sample_csv_content):
        """Test l'insertion d'une nouvelle entreprise depuis un CSV"""
        upload_file = create_upload_file(sample_csv_content)
        
        # Vérifier qu'aucune entreprise n'existe
        assert db_session.query(Company).count() == 0
        
        # Traiter le CSV
        result = pytest.mark.asyncio(process_csv_file)(upload_file, db_session, update_existing=False)
        
        # Vérifier les résultats
        assert result['success'] is True
        assert result['new_companies'] == 3
        assert result['entreprises']['nouvelles'] == 3
        assert result['entreprises']['total'] == 3
        
        # Vérifier en base
        assert db_session.query(Company).count() == 3
        
        # Vérifier une entreprise spécifique
        company = db_session.query(Company).filter(Company.siren == "123456789").first()
        assert company is not None
        assert company.nom_entreprise == "Entreprise Test 1"
        assert company.email == "test1@example.com"
        assert company.chiffre_affaires == 1000000.0
        assert company.effectif == 10

    def test_mise_a_jour_entreprise_existante(self, db_session):
        """Test la mise à jour d'une entreprise existante si update_existing=True"""
        # Créer une entreprise existante
        existing_company = Company(
            siren="123456789",
            nom_entreprise="Ancienne Entreprise",
            email="ancien@example.com",
            chiffre_affaires=500000.0
        )
        db_session.add(existing_company)
        db_session.commit()
        
        # CSV avec mise à jour
        updated_csv = """siren,nom_entreprise,email,chiffre_affaires
123456789,Nouvelle Entreprise,nouveau@example.com,1500000"""
        
        upload_file = create_upload_file(updated_csv)
        
        # Traiter avec mise à jour activée
        result = pytest.mark.asyncio(process_csv_file)(upload_file, db_session, update_existing=True)
        
        # Vérifier les résultats
        assert result['success'] is True
        assert result['entreprises']['mises_a_jour'] == 1
        assert result['entreprises']['total'] == 1
        
        # Vérifier la mise à jour en base
        updated_company = db_session.query(Company).filter(Company.siren == "123456789").first()
        assert updated_company.nom_entreprise == "Nouvelle Entreprise"
        assert updated_company.email == "nouveau@example.com"
        assert updated_company.chiffre_affaires == 1500000.0

    def test_non_ajout_doublon_sans_update(self, db_session):
        """Test le non-ajout d'un doublon si update_existing=False"""
        # Créer une entreprise existante
        existing_company = Company(
            siren="123456789",
            nom_entreprise="Entreprise Existante",
            email="existante@example.com"
        )
        db_session.add(existing_company)
        db_session.commit()
        
        # CSV avec même SIREN
        duplicate_csv = """siren,nom_entreprise,email
123456789,Tentative Doublon,doublon@example.com
987654321,Nouvelle Entreprise,nouvelle@example.com"""
        
        upload_file = create_upload_file(duplicate_csv)
        
        # Traiter sans mise à jour
        result = pytest.mark.asyncio(process_csv_file)(upload_file, db_session, update_existing=False)
        
        # Vérifier les résultats
        assert result['success'] is True
        assert result['entreprises']['nouvelles'] == 1  # Seulement la nouvelle
        assert result['entreprises']['ignorees'] >= 0   # Le doublon est ignoré
        assert result['entreprises']['total'] == 2      # 1 existante + 1 nouvelle
        
        # Vérifier que l'entreprise existante n'a pas changé
        unchanged_company = db_session.query(Company).filter(Company.siren == "123456789").first()
        assert unchanged_company.nom_entreprise == "Entreprise Existante"
        assert unchanged_company.email == "existante@example.com"
        
        # Vérifier que la nouvelle entreprise a été ajoutée
        new_company = db_session.query(Company).filter(Company.siren == "987654321").first()
        assert new_company is not None
        assert new_company.nom_entreprise == "Nouvelle Entreprise"

    def test_validation_siren_invalide(self, db_session):
        """Test la validation des SIREN invalides"""
        invalid_csv = """siren,nom_entreprise
12345,SIREN trop court
ABCDEFGHI,SIREN non numérique
,SIREN vide
123456789,SIREN valide"""
        
        upload_file = create_upload_file(invalid_csv)
        
        result = pytest.mark.asyncio(process_csv_file)(upload_file, db_session, update_existing=False)
        
        # Seul le SIREN valide doit être traité
        assert result['success'] is True
        assert result['entreprises']['nouvelles'] == 1
        assert result['skipped_companies'] == 3  # 3 SIREN invalides ignorés
        
        # Vérifier qu'une seule entreprise a été créée
        assert db_session.query(Company).count() == 1
        company = db_session.query(Company).first()
        assert company.siren == "123456789"

    def test_contrainte_unique_siren(self, db_session):
        """Test la contrainte unique sur le SIREN"""
        # Créer une première entreprise
        company1 = Company(
            siren="123456789",
            nom_entreprise="Première Entreprise"
        )
        db_session.add(company1)
        db_session.commit()
        
        # Tenter de créer une seconde entreprise avec le même SIREN
        company2 = Company(
            siren="123456789",
            nom_entreprise="Seconde Entreprise"
        )
        db_session.add(company2)
        
        # Cela doit lever une exception
        with pytest.raises(Exception):  # IntegrityError ou similaire
            db_session.commit()

    def test_nettoyage_donnees_numeriques(self, db_session):
        """Test le nettoyage des données numériques"""
        numeric_csv = """siren,nom_entreprise,chiffre_affaires,effectif
123456789,Test Numerique,"1 000 000,50",25
987654321,Test Euro,500000€,15"""
        
        upload_file = create_upload_file(numeric_csv)
        
        result = pytest.mark.asyncio(process_csv_file)(upload_file, db_session, update_existing=False)
        
        assert result['success'] is True
        assert result['entreprises']['nouvelles'] == 2
        
        # Vérifier le nettoyage des données
        company1 = db_session.query(Company).filter(Company.siren == "123456789").first()
        assert company1.chiffre_affaires == 1000000.50
        assert company1.effectif == 25
        
        company2 = db_session.query(Company).filter(Company.siren == "987654321").first()
        assert company2.chiffre_affaires == 500000.0
        assert company2.effectif == 15

if __name__ == "__main__":
    pytest.main([__file__])