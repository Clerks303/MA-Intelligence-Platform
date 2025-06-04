"""
Modèles de données pour le SDK Python
"""

from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum


class StatusEnum(str, Enum):
    """Énumération des statuts d'entreprise"""
    PROSPECT = "prospect"
    CONTACT = "contact" 
    QUALIFICATION = "qualification"
    NEGOCIATION = "negociation"
    CLIENT = "client"
    PERDU = "perdu"


@dataclass
class Company:
    """
    Modèle de données pour une entreprise.
    
    Attributes:
        id: Identifiant unique
        siren: Numéro SIREN
        nom_entreprise: Nom de l'entreprise
        ville: Ville
        chiffre_affaires: Chiffre d'affaires en euros
        effectif: Nombre d'employés
        statut: Statut dans le pipeline
        ... (autres champs)
    """
    
    # Champs obligatoires
    id: str
    siren: str
    nom_entreprise: str
    
    # Champs optionnels
    forme_juridique: Optional[str] = None
    date_creation: Optional[datetime] = None
    adresse: Optional[str] = None
    ville: Optional[str] = None
    code_postal: Optional[str] = None
    email: Optional[str] = None
    telephone: Optional[str] = None
    numero_tva: Optional[str] = None
    chiffre_affaires: Optional[float] = None
    resultat: Optional[float] = None
    effectif: Optional[int] = None
    capital_social: Optional[float] = None
    code_naf: Optional[str] = None
    libelle_code_naf: Optional[str] = None
    dirigeant_principal: Optional[str] = None
    statut: StatusEnum = StatusEnum.PROSPECT
    score_prospection: Optional[float] = None
    description: Optional[str] = None
    
    # Métadonnées
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    # Données enrichies (optionnelles)
    dirigeants_json: Optional[Dict[str, Any]] = None
    score_details: Optional[Dict[str, Any]] = None
    activity_logs: Optional[List[Dict[str, Any]]] = None
    details_complets: Optional[Dict[str, Any]] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Company':
        """
        Crée une instance Company depuis un dictionnaire.
        
        Args:
            data: Données de l'entreprise
            
        Returns:
            Company: Instance créée
        """
        
        # Convertir les dates ISO string en datetime
        if data.get("date_creation") and isinstance(data["date_creation"], str):
            data["date_creation"] = datetime.fromisoformat(data["date_creation"].replace("Z", "+00:00"))
        
        if data.get("created_at") and isinstance(data["created_at"], str):
            data["created_at"] = datetime.fromisoformat(data["created_at"].replace("Z", "+00:00"))
            
        if data.get("updated_at") and isinstance(data["updated_at"], str):
            data["updated_at"] = datetime.fromisoformat(data["updated_at"].replace("Z", "+00:00"))
        
        # Convertir le statut en enum
        if data.get("statut"):
            data["statut"] = StatusEnum(data["statut"])
        
        # Filtrer les champs valides pour __init__
        valid_fields = {field.name for field in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        
        return cls(**filtered_data)
    
    def to_dict(self, exclude_none: bool = True) -> Dict[str, Any]:
        """
        Convertit l'instance en dictionnaire.
        
        Args:
            exclude_none: Exclure les valeurs None
            
        Returns:
            Dict: Données sérialisées
        """
        
        result = {}
        
        for field_name, field_def in self.__dataclass_fields__.items():
            value = getattr(self, field_name)
            
            if exclude_none and value is None:
                continue
            
            # Sérialiser les dates
            if isinstance(value, datetime):
                result[field_name] = value.isoformat()
            # Sérialiser les enums
            elif isinstance(value, Enum):
                result[field_name] = value.value
            else:
                result[field_name] = value
        
        return result
    
    def __str__(self) -> str:
        """Représentation string lisible."""
        return f"Company(siren={self.siren}, nom={self.nom_entreprise}, ville={self.ville})"
    
    def __repr__(self) -> str:
        """Représentation string pour debug."""
        return f"Company(id={self.id}, siren={self.siren}, nom_entreprise='{self.nom_entreprise}')"


@dataclass
class CompanyFilters:
    """
    Filtres pour la recherche d'entreprises.
    
    Example:
        >>> filters = CompanyFilters(
        ...     q="comptable",
        ...     ville="Paris",
        ...     ca_min=50000,
        ...     ca_max=1000000,
        ...     with_email=True,
        ...     sort_by="chiffre_affaires",
        ...     sort_order="desc"
        ... )
        >>> companies = client.companies.search(filters)
    """
    
    # Recherche textuelle
    q: Optional[str] = None
    
    # Filtres exacts
    siren: Optional[str] = None
    nom_entreprise: Optional[str] = None
    ville: Optional[str] = None
    code_postal: Optional[str] = None
    secteur_activite: Optional[str] = None
    
    # Filtres numériques
    ca_min: Optional[float] = None
    ca_max: Optional[float] = None
    effectif_min: Optional[int] = None
    effectif_max: Optional[int] = None
    score_min: Optional[float] = None
    
    # Filtres de dates
    date_creation_after: Optional[datetime] = None
    date_creation_before: Optional[datetime] = None
    
    # Filtres booléens
    with_email: Optional[bool] = None
    with_phone: Optional[bool] = None
    
    # Tri
    sort_by: str = "nom_entreprise"
    sort_order: str = "asc"  # asc ou desc
    
    def to_dict(self, exclude_none: bool = True) -> Dict[str, Any]:
        """
        Convertit les filtres en dictionnaire.
        
        Args:
            exclude_none: Exclure les valeurs None
            
        Returns:
            Dict: Filtres sérialisés
        """
        
        result = {}
        
        for field_name, field_def in self.__dataclass_fields__.items():
            value = getattr(self, field_name)
            
            if exclude_none and value is None:
                continue
            
            # Sérialiser les dates
            if isinstance(value, datetime):
                result[field_name] = value.isoformat()
            else:
                result[field_name] = value
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CompanyFilters':
        """Crée des filtres depuis un dictionnaire."""
        
        # Convertir les dates string en datetime
        for date_field in ["date_creation_after", "date_creation_before"]:
            if data.get(date_field) and isinstance(data[date_field], str):
                data[date_field] = datetime.fromisoformat(data[date_field].replace("Z", "+00:00"))
        
        # Filtrer les champs valides
        valid_fields = {field.name for field in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        
        return cls(**filtered_data)


@dataclass  
class APIResponse:
    """
    Réponse API standardisée.
    
    Attributes:
        success: Indique si la requête a réussi
        data: Données de la réponse
        message: Message descriptif
        timestamp: Horodatage de la réponse
        request_id: ID unique de la requête
    """
    
    success: bool
    data: Any
    message: str = ""
    timestamp: Optional[datetime] = None
    request_id: Optional[str] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'APIResponse':
        """Crée une APIResponse depuis un dictionnaire."""
        
        if data.get("timestamp") and isinstance(data["timestamp"], str):
            data["timestamp"] = datetime.fromisoformat(data["timestamp"].replace("Z", "+00:00"))
        
        return cls(**data)


@dataclass
class PaginatedResponse:
    """
    Réponse paginée standardisée.
    
    Attributes:
        data: Liste des éléments
        pagination: Métadonnées de pagination
        success: Statut de la requête
        message: Message descriptif
        timestamp: Horodatage
    """
    
    data: List[Any]
    pagination: Dict[str, Any] = field(default_factory=dict)
    success: bool = True
    message: str = ""
    timestamp: Optional[datetime] = None
    request_id: Optional[str] = None
    
    @property
    def total(self) -> int:
        """Nombre total d'éléments."""
        return self.pagination.get("total", 0)
    
    @property
    def page(self) -> int:
        """Numéro de page actuel."""
        return self.pagination.get("page", 1)
    
    @property
    def size(self) -> int:
        """Taille de la page."""
        return self.pagination.get("size", 50)
    
    @property
    def total_pages(self) -> int:
        """Nombre total de pages."""
        return self.pagination.get("total_pages", 1)
    
    @property
    def has_next(self) -> bool:
        """Indique s'il y a une page suivante."""
        return self.pagination.get("has_next", False)
    
    @property
    def has_prev(self) -> bool:
        """Indique s'il y a une page précédente."""
        return self.pagination.get("has_prev", False)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PaginatedResponse':
        """Crée une PaginatedResponse depuis un dictionnaire."""
        
        if data.get("timestamp") and isinstance(data["timestamp"], str):
            data["timestamp"] = datetime.fromisoformat(data["timestamp"].replace("Z", "+00:00"))
        
        # Convertir les éléments en Company si approprié
        if "data" in data and isinstance(data["data"], list):
            converted_data = []
            for item in data["data"]:
                if isinstance(item, dict) and "siren" in item and "nom_entreprise" in item:
                    # C'est probablement une entreprise
                    converted_data.append(Company.from_dict(item))
                else:
                    converted_data.append(item)
            data["data"] = converted_data
        
        return cls(**data)
    
    def __iter__(self):
        """Permet l'itération directe sur les données."""
        return iter(self.data)
    
    def __len__(self):
        """Retourne le nombre d'éléments dans la page actuelle."""
        return len(self.data)
    
    def __getitem__(self, index):
        """Permet l'accès par index aux données."""
        return self.data[index]


@dataclass
class ValidationError:
    """Erreur de validation détaillée."""
    
    field: str
    message: str
    code: Optional[str] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ValidationError':
        """Crée depuis un dictionnaire."""
        return cls(**data)


@dataclass
class RateLimitInfo:
    """Informations sur le rate limiting."""
    
    limit: int
    remaining: int
    reset_time: datetime
    retry_after: Optional[int] = None
    
    @classmethod
    def from_headers(cls, headers: Dict[str, str]) -> 'RateLimitInfo':
        """Crée depuis les headers HTTP."""
        
        return cls(
            limit=int(headers.get("X-RateLimit-Limit", 0)),
            remaining=int(headers.get("X-RateLimit-Remaining", 0)),
            reset_time=datetime.fromtimestamp(int(headers.get("X-RateLimit-Reset", 0))),
            retry_after=int(headers.get("Retry-After", 0)) if headers.get("Retry-After") else None
        )