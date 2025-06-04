"""
Client principal du SDK Python pour M&A Intelligence
"""

import httpx
import asyncio
from typing import Optional, Dict, Any, List, Union
from urllib.parse import urljoin
import json
from datetime import datetime

from .models import Company, CompanyFilters, APIResponse, PaginatedResponse
from .exceptions import (
    MAIntelligenceError,
    AuthenticationError,
    RateLimitError,
    NotFoundError,
    ValidationError
)


class CompaniesAPI:
    """Interface pour l'API des entreprises"""
    
    def __init__(self, client: 'MAIntelligenceClient'):
        self.client = client
    
    def list(
        self,
        page: int = 1,
        size: int = 50,
        city: Optional[str] = None,
        sector: Optional[str] = None,
        ca_min: Optional[float] = None,
        ca_max: Optional[float] = None,
        sort_by: str = "nom_entreprise",
        sort_order: str = "asc",
        **kwargs
    ) -> PaginatedResponse:
        """
        Liste les entreprises avec filtres et pagination.
        
        Args:
            page: Numéro de page (défaut: 1)
            size: Taille de page (défaut: 50, max: 1000)
            city: Filtrer par ville
            sector: Filtrer par secteur
            ca_min: Chiffre d'affaires minimum
            ca_max: Chiffre d'affaires maximum
            sort_by: Champ de tri
            sort_order: Ordre de tri (asc/desc)
            **kwargs: Autres filtres disponibles
        
        Returns:
            PaginatedResponse avec la liste des entreprises
        
        Example:
            >>> companies = client.companies.list(
            ...     city="Paris",
            ...     ca_min=100000,
            ...     page=1,
            ...     size=100
            ... )
            >>> print(f"Trouvé {companies.total} entreprises")
            >>> for company in companies.data:
            ...     print(company.nom_entreprise)
        """
        
        params = {
            "page": page,
            "size": size,
            "sort_by": sort_by,
            "sort_order": sort_order
        }
        
        if city:
            params["ville"] = city
        if sector:
            params["secteur"] = sector
        if ca_min is not None:
            params["ca_min"] = ca_min
        if ca_max is not None:
            params["ca_max"] = ca_max
        
        # Ajouter kwargs
        params.update(kwargs)
        
        response = self.client._request("GET", "/external/companies", params=params)
        return PaginatedResponse.from_dict(response)
    
    def search(
        self,
        filters: Union[CompanyFilters, Dict[str, Any]],
        page: int = 1,
        size: int = 50
    ) -> PaginatedResponse:
        """
        Recherche avancée d'entreprises.
        
        Args:
            filters: Filtres de recherche (CompanyFilters ou dict)
            page: Numéro de page
            size: Taille de page
        
        Returns:
            PaginatedResponse avec les résultats
        
        Example:
            >>> filters = CompanyFilters(
            ...     q="comptable",
            ...     ville="Paris",
            ...     ca_min=50000,
            ...     with_email=True
            ... )
            >>> results = client.companies.search(filters)
        """
        
        if isinstance(filters, CompanyFilters):
            search_data = filters.to_dict()
        else:
            search_data = filters
        
        params = {"page": page, "size": size}
        
        response = self.client._request(
            "POST", 
            "/external/companies/search",
            json=search_data,
            params=params
        )
        return PaginatedResponse.from_dict(response)
    
    def get(self, company_id: str, include_logs: bool = False) -> Company:
        """
        Récupère les détails d'une entreprise.
        
        Args:
            company_id: ID unique de l'entreprise
            include_logs: Inclure les logs d'activité
        
        Returns:
            Company: Détails complets de l'entreprise
        
        Example:
            >>> company = client.companies.get("123e4567-e89b-12d3-a456-426614174000")
            >>> print(f"{company.nom_entreprise} - SIREN: {company.siren}")
        """
        
        params = {"include_logs": include_logs} if include_logs else {}
        
        response = self.client._request(
            "GET", 
            f"/external/companies/{company_id}",
            params=params
        )
        return Company.from_dict(response["data"])
    
    def create(self, company_data: Dict[str, Any]) -> Company:
        """
        Crée une nouvelle entreprise.
        
        Args:
            company_data: Données de l'entreprise à créer
        
        Returns:
            Company: Entreprise créée
        
        Example:
            >>> new_company = client.companies.create({
            ...     "siren": "123456789",
            ...     "nom_entreprise": "SARL Exemple",
            ...     "ville": "Paris"
            ... })
        """
        
        response = self.client._request("POST", "/external/companies", json=company_data)
        return Company.from_dict(response["data"])
    
    def update(self, company_id: str, update_data: Dict[str, Any]) -> Company:
        """
        Met à jour une entreprise existante.
        
        Args:
            company_id: ID de l'entreprise
            update_data: Données à mettre à jour
        
        Returns:
            Company: Entreprise mise à jour
        """
        
        response = self.client._request(
            "PUT", 
            f"/external/companies/{company_id}",
            json=update_data
        )
        return Company.from_dict(response["data"])
    
    def delete(self, company_id: str) -> None:
        """
        Supprime une entreprise.
        
        Args:
            company_id: ID de l'entreprise à supprimer
        """
        
        self.client._request("DELETE", f"/external/companies/{company_id}")
    
    def export_csv(
        self,
        filters: Optional[Dict[str, Any]] = None,
        filename: Optional[str] = None
    ) -> bytes:
        """
        Exporte les entreprises en CSV.
        
        Args:
            filters: Filtres à appliquer (optionnel)
            filename: Nom du fichier suggéré
        
        Returns:
            bytes: Contenu du fichier CSV
        
        Example:
            >>> csv_data = client.companies.export_csv({"ville": "Paris"})
            >>> with open("companies_paris.csv", "wb") as f:
            ...     f.write(csv_data)
        """
        
        params = {"format": "csv"}
        if filters:
            params["filters"] = json.dumps(filters)
        
        return self.client._request_raw("GET", "/external/export/companies", params=params)


class StatsAPI:
    """Interface pour l'API des statistiques"""
    
    def __init__(self, client: 'MAIntelligenceClient'):
        self.client = client
    
    def get_global_stats(
        self,
        include_trends: bool = False,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Récupère les statistiques globales.
        
        Args:
            include_trends: Inclure les tendances temporelles
            date_from: Date de début pour les tendances
            date_to: Date de fin pour les tendances
        
        Returns:
            Dict avec les statistiques
        
        Example:
            >>> stats = client.stats.get_global_stats(include_trends=True)
            >>> print(f"Total entreprises: {stats['total']}")
            >>> print(f"CA moyen: {stats['ca_moyen']:.2f}€")
        """
        
        params = {"include_trends": include_trends}
        
        if date_from:
            params["date_from"] = date_from.isoformat()
        if date_to:
            params["date_to"] = date_to.isoformat()
        
        response = self.client._request("GET", "/external/stats", params=params)
        return response["data"]


class MAIntelligenceClient:
    """
    Client principal pour l'API M&A Intelligence.
    
    Attributes:
        companies: Interface pour l'API des entreprises
        stats: Interface pour l'API des statistiques
    
    Example:
        >>> # Authentification par clé API
        >>> client = MAIntelligenceClient(
        ...     api_key="ak_your_api_key",
        ...     base_url="https://api.ma-intelligence.com"
        ... )
        
        >>> # Authentification OAuth2
        >>> client = MAIntelligenceClient(
        ...     access_token="your_oauth_token",
        ...     base_url="https://api.ma-intelligence.com"
        ... )
        
        >>> # Utilisation
        >>> companies = client.companies.list(city="Paris")
        >>> stats = client.stats.get_global_stats()
    """
    
    def __init__(
        self,
        base_url: str = "https://api.ma-intelligence.com",
        api_key: Optional[str] = None,
        access_token: Optional[str] = None,
        timeout: int = 30,
        max_retries: int = 3,
        **kwargs
    ):
        """
        Initialise le client.
        
        Args:
            base_url: URL de base de l'API
            api_key: Clé API pour l'authentification
            access_token: Token OAuth2 pour l'authentification
            timeout: Timeout des requêtes en secondes
            max_retries: Nombre maximum de tentatives
            **kwargs: Options additionnelles pour httpx
        
        Raises:
            ValueError: Si aucune méthode d'authentification n'est fournie
        """
        
        if not api_key and not access_token:
            raise ValueError("api_key ou access_token requis")
        
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        
        # Configuration des headers d'authentification
        self.headers = {
            "User-Agent": f"ma-intelligence-sdk-python/1.0.0",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        if api_key:
            self.headers["X-API-Key"] = api_key
        elif access_token:
            self.headers["Authorization"] = f"Bearer {access_token}"
        
        # Configuration du client HTTP
        client_config = {
            "timeout": timeout,
            "headers": self.headers,
            **kwargs
        }
        
        self._client = httpx.Client(**client_config)
        self._async_client = httpx.AsyncClient(**client_config)
        
        # Initialiser les interfaces API
        self.companies = CompaniesAPI(self)
        self.stats = StatsAPI(self)
    
    def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict] = None,
        json: Optional[Dict] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Effectue une requête HTTP synchrone.
        
        Args:
            method: Méthode HTTP (GET, POST, etc.)
            endpoint: Endpoint de l'API (sans base_url)
            params: Paramètres de requête
            json: Données JSON à envoyer
            **kwargs: Options additionnelles
        
        Returns:
            Dict: Réponse JSON parsée
        
        Raises:
            MAIntelligenceError: En cas d'erreur API
        """
        
        url = urljoin(self.base_url + "/api/v1", endpoint.lstrip("/"))
        
        for attempt in range(self.max_retries + 1):
            try:
                response = self._client.request(
                    method,
                    url,
                    params=params,
                    json=json,
                    **kwargs
                )
                
                return self._handle_response(response)
                
            except httpx.RequestError as e:
                if attempt == self.max_retries:
                    raise MAIntelligenceError(f"Erreur réseau: {e}")
                continue
    
    def _request_raw(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict] = None,
        **kwargs
    ) -> bytes:
        """
        Effectue une requête HTTP et retourne le contenu brut.
        
        Returns:
            bytes: Contenu brut de la réponse
        """
        
        url = urljoin(self.base_url + "/api/v1", endpoint.lstrip("/"))
        
        response = self._client.request(method, url, params=params, **kwargs)
        
        if response.status_code >= 400:
            self._handle_response(response)  # Déclenche exception
        
        return response.content
    
    async def _async_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict] = None,
        json: Optional[Dict] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Effectue une requête HTTP asynchrone.
        
        Returns:
            Dict: Réponse JSON parsée
        """
        
        url = urljoin(self.base_url + "/api/v1", endpoint.lstrip("/"))
        
        for attempt in range(self.max_retries + 1):
            try:
                response = await self._async_client.request(
                    method,
                    url,
                    params=params,
                    json=json,
                    **kwargs
                )
                
                return self._handle_response(response)
                
            except httpx.RequestError as e:
                if attempt == self.max_retries:
                    raise MAIntelligenceError(f"Erreur réseau: {e}")
                continue
    
    def _handle_response(self, response: httpx.Response) -> Dict[str, Any]:
        """
        Traite la réponse HTTP et gère les erreurs.
        
        Args:
            response: Réponse HTTP
        
        Returns:
            Dict: Données JSON parsées
        
        Raises:
            MAIntelligenceError: En cas d'erreur API
        """
        
        # Vérifier le rate limiting
        if response.status_code == 429:
            retry_after = response.headers.get("Retry-After", "60")
            raise RateLimitError(
                f"Rate limit atteint. Réessayez dans {retry_after} secondes.",
                retry_after=int(retry_after)
            )
        
        # Vérifier l'authentification
        if response.status_code == 401:
            raise AuthenticationError("Authentification échouée. Vérifiez vos identifiants.")
        
        # Vérifier les permissions
        if response.status_code == 403:
            raise AuthenticationError("Permissions insuffisantes.")
        
        # Vérifier ressource non trouvée
        if response.status_code == 404:
            raise NotFoundError("Ressource non trouvée.")
        
        # Vérifier erreurs de validation
        if response.status_code == 422:
            try:
                error_data = response.json()
                raise ValidationError(
                    "Erreur de validation.",
                    details=error_data.get("detail", [])
                )
            except (json.JSONDecodeError, KeyError):
                raise ValidationError("Erreur de validation.")
        
        # Vérifier autres erreurs client/serveur
        if response.status_code >= 400:
            try:
                error_data = response.json()
                message = error_data.get("error", {}).get("message", "Erreur inconnue")
            except json.JSONDecodeError:
                message = f"Erreur HTTP {response.status_code}"
            
            raise MAIntelligenceError(message, status_code=response.status_code)
        
        # Parsing de la réponse JSON
        try:
            return response.json()
        except json.JSONDecodeError:
            raise MAIntelligenceError("Réponse invalide (JSON attendu)")
    
    def test_connection(self) -> Dict[str, Any]:
        """
        Teste la connexion à l'API.
        
        Returns:
            Dict: Informations sur la connexion
        
        Example:
            >>> status = client.test_connection()
            >>> print(f"Connexion: {status['status']}")
        """
        
        try:
            response = self._request("GET", "/external/companies", params={"page": 1, "size": 1})
            return {
                "status": "connected",
                "api_version": "v1",
                "message": "Connexion réussie"
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }
    
    def close(self):
        """Ferme les connexions HTTP."""
        self._client.close()
    
    async def aclose(self):
        """Ferme les connexions HTTP asynchrones."""
        await self._async_client.aclose()
    
    def __enter__(self):
        """Support pour context manager synchrone."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Fermeture automatique en context manager."""
        self.close()
    
    async def __aenter__(self):
        """Support pour context manager asynchrone."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Fermeture automatique en context manager asynchrone."""
        await self.aclose()