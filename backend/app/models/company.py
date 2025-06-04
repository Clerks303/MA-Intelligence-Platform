from sqlalchemy import Column, String, DateTime, Float, Integer, Enum, JSON, Text, Boolean, ForeignKey, Index
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from app.core.database import Base
from app.models.schemas import StatusEnum
import uuid
from enum import Enum as PyEnum


class EnrichmentSourceEnum(PyEnum):
    """Sources d'enrichissement des données"""
    PAPPERS = "pappers"
    INFOGREFFE = "infogreffe"
    SOCIETE = "societe"
    KASPR = "kaspr"
    MANUAL = "manual"


class ContactTypeEnum(PyEnum):
    """Types de contacts"""
    DIRIGEANT = "dirigeant"
    COMPTABLE = "comptable"
    COMMERCIAL = "commercial"
    RH = "rh"
    AUTRE = "autre"


class Company(Base):
    """
    Modèle Company enrichi pour l'analyse M&A
    
    Table principale contenant toutes les données d'entreprises
    avec enrichissements multi-sources et scoring M&A
    """
    __tablename__ = "companies"

    # === IDENTIFIANTS ===
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    siren = Column(String(9), unique=True, nullable=False, index=True)
    siret = Column(String(14), index=True)
    
    # === INFORMATIONS GÉNÉRALES ===
    nom_entreprise = Column(String(255), nullable=False, index=True)
    forme_juridique = Column(String(100))
    date_creation = Column(DateTime(timezone=True))
    
    # === COORDONNÉES ===
    adresse = Column(Text)
    ville = Column(String(100), index=True)
    code_postal = Column(String(10), index=True)
    email = Column(String(255))
    telephone = Column(String(20))
    numero_tva = Column(String(20))
    site_web = Column(String(500))
    
    # === DONNÉES FINANCIÈRES ===
    chiffre_affaires = Column(Float, index=True)  # CA année courante
    chiffre_affaires_n1 = Column(Float)  # CA année n-1
    chiffre_affaires_n2 = Column(Float)  # CA année n-2
    resultat = Column(Float)  # Résultat net
    resultat_n1 = Column(Float)  # Résultat n-1
    resultat_n2 = Column(Float)  # Résultat n-2
    effectif = Column(Integer, index=True)
    capital_social = Column(Float)
    
    # === DONNÉES M&A ENRICHIES ===
    evolution_ca_3ans = Column(Float)  # Évolution CA sur 3 ans (%)
    marge_nette = Column(Float)  # Marge nette (%)
    ratio_endettement = Column(Float)  # Ratio d'endettement
    rentabilite_capitaux = Column(Float)  # ROE (%)
    croissance_effectif = Column(Float)  # Croissance effectif (%)
    
    # === SCORING M&A ===
    ma_score = Column(Float, index=True)  # Score M&A global (0-100)
    ma_score_details = Column(JSON)  # Détail du scoring par critère
    potentiel_acquisition = Column(Boolean, default=False, index=True)
    potentiel_cession = Column(Boolean, default=False, index=True)
    priorite_contact = Column(String(20), index=True)  # HIGH, MEDIUM, LOW
    
    # === ACTIVITÉ ===
    code_naf = Column(String(10), index=True)
    libelle_code_naf = Column(String(255))
    secteur_activite = Column(String(100), index=True)
    specialisation = Column(Text)  # Spécialisation métier
    
    # === DIRECTION ===
    dirigeant_principal = Column(String(255))
    dirigeants_json = Column(JSON)
    age_dirigeant_principal = Column(Integer)  # Âge pour analyse succession
    anciennete_dirigeant = Column(Integer)  # Ancienneté en années
    
    # === PROSPECTION ===
    statut = Column(Enum(StatusEnum), default=StatusEnum.PROSPECT, index=True)
    score_prospection = Column(Float)  # Score prospection classique
    score_details = Column(JSON)
    description = Column(Text)
    notes_commerciales = Column(Text)
    derniere_actualite = Column(Text)  # Dernière actualité trouvée
    date_derniere_actualite = Column(DateTime(timezone=True))
    
    # === VEILLE ET SIGNAUX ===
    signaux_faibles = Column(JSON)  # Signaux de M&A détectés
    actualites_recentes = Column(JSON)  # Actualités récentes
    mentions_presse = Column(JSON)  # Mentions dans la presse
    changements_dirigeants = Column(JSON)  # Historique changements
    
    # === ENRICHISSEMENT ===
    enrichment_status = Column(JSON)  # Statut d'enrichissement par source
    sources_enrichissement = Column(JSON)  # Sources utilisées
    derniere_verification = Column(DateTime(timezone=True))
    qualite_donnees = Column(Float)  # Score qualité des données (0-1)
    donnees_manquantes = Column(JSON)  # Liste des champs manquants
    
    # === LIENS EXTERNES ===
    lien_pappers = Column(String(500))
    lien_societe_com = Column(String(500))
    lien_infogreffe = Column(String(500))
    lien_linkedin = Column(String(500))
    
    # === MÉTADONNÉES ===
    details_complets = Column(JSON)  # Données brutes complètes
    historique_modifications = Column(JSON)  # Historique des modifs
    tags = Column(JSON)  # Tags personnalisés
    
    # === HORODATAGE ===
    last_scraped_at = Column(DateTime(timezone=True))
    last_enriched_pappers = Column(DateTime(timezone=True))
    last_enriched_infogreffe = Column(DateTime(timezone=True))
    last_enriched_societe = Column(DateTime(timezone=True))
    last_enriched_kaspr = Column(DateTime(timezone=True))
    last_scored_at = Column(DateTime(timezone=True))
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now(), server_default=func.now())
    
    # === RELATIONS ===
    contacts = relationship("CompanyContact", back_populates="company", cascade="all, delete-orphan")
    
    # === INDEX COMPOSÉS ===
    __table_args__ = (
        Index('ix_companies_ma_score_ca', 'ma_score', 'chiffre_affaires'),
        Index('ix_companies_geo', 'code_postal', 'ville'),
        Index('ix_companies_activity', 'code_naf', 'secteur_activite'),
        Index('ix_companies_prospection', 'statut', 'priorite_contact'),
        Index('ix_companies_financial', 'chiffre_affaires', 'effectif', 'ma_score'),
    )

    def __repr__(self):
        return f"<Company(siren={self.siren}, nom={self.nom_entreprise}, score_ma={self.ma_score})>"
    
    def to_dict(self) -> dict:
        """Conversion en dictionnaire pour JSON"""
        result = {}
        for column in self.__table__.columns:
            value = getattr(self, column.name)
            if isinstance(value, (datetime, PyEnum)):
                result[column.name] = str(value) if value else None
            else:
                result[column.name] = value
        return result
    
    @property
    def is_target_ma(self) -> bool:
        """Détermine si l'entreprise est une cible M&A prioritaire"""
        if not self.ma_score:
            return False
        
        criteria = (
            self.ma_score >= 70 and
            self.chiffre_affaires and self.chiffre_affaires >= 3_000_000 and
            self.effectif and self.effectif >= 30 and
            self.statut in [StatusEnum.PROSPECT, StatusEnum.A_CONTACTER]
        )
        return criteria
    
    @property
    def evolution_ca_formatted(self) -> str:
        """Évolution CA formatée pour affichage"""
        if self.evolution_ca_3ans is None:
            return "N/A"
        return f"{self.evolution_ca_3ans:+.1f}%"


class CompanyContact(Base):
    """
    Contacts enrichis des entreprises
    
    Table des contacts dirigeants et collaborateurs
    avec informations d'enrichissement multi-sources
    """
    __tablename__ = "company_contacts"

    # === IDENTIFIANTS ===
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    company_id = Column(UUID(as_uuid=True), ForeignKey('companies.id', ondelete='CASCADE'), nullable=False)
    
    # === INFORMATIONS PERSONNELLES ===
    nom_complet = Column(String(255), nullable=False)
    prenom = Column(String(100))
    nom = Column(String(100))
    civilite = Column(String(10))  # M., Mme, Dr., etc.
    
    # === FONCTION ===
    type_contact = Column(Enum(ContactTypeEnum), nullable=False, index=True)
    poste = Column(String(255))  # Intitulé du poste
    qualite = Column(String(100))  # Qualité juridique (PDG, DG, etc.)
    anciennete_poste = Column(Integer)  # Ancienneté dans le poste (années)
    est_dirigeant = Column(Boolean, default=False, index=True)
    est_actionnaire = Column(Boolean, default=False)
    pourcentage_participation = Column(Float)  # % de participation si actionnaire
    
    # === COORDONNÉES ===
    email_professionnel = Column(String(255))
    email_personnel = Column(String(255))
    telephone_direct = Column(String(20))
    telephone_mobile = Column(String(20))
    linkedin_url = Column(String(500))
    
    # === MÉTADONNÉES D'ENRICHISSEMENT ===
    source = Column(Enum(EnrichmentSourceEnum), nullable=False)
    confidence_score = Column(Float)  # Score de confiance (0-1)
    derniere_verification = Column(DateTime(timezone=True))
    statut_email = Column(String(20))  # VERIFIED, BOUNCE, UNKNOWN
    statut_telephone = Column(String(20))  # VERIFIED, INVALID, UNKNOWN
    
    # === DONNÉES COMPLÉMENTAIRES ===
    age_estime = Column(Integer)
    formation = Column(String(255))  # Formation/diplômes
    experience_precedente = Column(Text)  # Expériences précédentes
    reseaux_sociaux = Column(JSON)  # Autres réseaux sociaux
    
    # === PROSPECTION ===
    derniere_interaction = Column(DateTime(timezone=True))
    type_derniere_interaction = Column(String(50))  # EMAIL, CALL, MEETING
    notes_contact = Column(Text)
    score_accessibilite = Column(Float)  # Score facilité de contact
    preferences_contact = Column(JSON)  # Préférences de contact
    
    # === HORODATAGE ===
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now(), server_default=func.now())
    
    # === RELATIONS ===
    company = relationship("Company", back_populates="contacts")
    
    # === INDEX COMPOSÉS ===
    __table_args__ = (
        Index('ix_contacts_company_type', 'company_id', 'type_contact'),
        Index('ix_contacts_dirigeant', 'est_dirigeant', 'confidence_score'),
        Index('ix_contacts_email', 'email_professionnel', 'statut_email'),
    )

    def __repr__(self):
        return f"<CompanyContact(nom={self.nom_complet}, poste={self.poste}, source={self.source})>"
    
    def to_dict(self) -> dict:
        """Conversion en dictionnaire pour JSON"""
        result = {}
        for column in self.__table__.columns:
            value = getattr(self, column.name)
            if isinstance(value, (datetime, PyEnum)):
                result[column.name] = str(value) if value else None
            else:
                result[column.name] = value
        return result
    
    @property
    def is_decision_maker(self) -> bool:
        """Détermine si le contact est décisionnaire"""
        decision_keywords = ['pdg', 'directeur', 'président', 'gérant', 'associé']
        if self.est_dirigeant:
            return True
        if self.qualite and any(keyword in self.qualite.lower() for keyword in decision_keywords):
            return True
        return False
    
    @property
    def contact_quality_score(self) -> float:
        """Score de qualité du contact (0-100)"""
        score = 0
        
        # Email professionnel (+30)
        if self.email_professionnel and self.statut_email == 'VERIFIED':
            score += 30
        elif self.email_professionnel:
            score += 15
            
        # Téléphone direct (+25)
        if self.telephone_direct and self.statut_telephone == 'VERIFIED':
            score += 25
        elif self.telephone_direct:
            score += 10
            
        # LinkedIn (+15)
        if self.linkedin_url:
            score += 15
            
        # Fonction dirigeant (+20)
        if self.est_dirigeant or self.is_decision_maker:
            score += 20
            
        # Confidence score (+10)
        if self.confidence_score:
            score += self.confidence_score * 10
            
        return min(score, 100)