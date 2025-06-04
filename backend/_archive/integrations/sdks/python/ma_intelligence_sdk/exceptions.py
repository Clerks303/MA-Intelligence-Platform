"""
Exceptions personnalisées pour le SDK Python
"""

from typing import Optional, List, Dict, Any


class MAIntelligenceError(Exception):
    """
    Exception de base pour toutes les erreurs du SDK.
    
    Attributes:
        message: Message d'erreur
        status_code: Code de statut HTTP (si applicable)
        request_id: ID de la requête (si disponible)
    """
    
    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        request_id: Optional[str] = None
    ):
        self.message = message
        self.status_code = status_code
        self.request_id = request_id
        super().__init__(message)
    
    def __str__(self) -> str:
        error_parts = [self.message]
        
        if self.status_code:
            error_parts.append(f"(HTTP {self.status_code})")
        
        if self.request_id:
            error_parts.append(f"[Request ID: {self.request_id}]")
        
        return " ".join(error_parts)


class AuthenticationError(MAIntelligenceError):
    """
    Erreur d'authentification.
    
    Levée quand:
    - Clé API invalide ou expirée
    - Token OAuth2 invalide ou expiré
    - Permissions insuffisantes
    """
    
    def __init__(self, message: str = "Erreur d'authentification", **kwargs):
        super().__init__(message, **kwargs)


class RateLimitError(MAIntelligenceError):
    """
    Erreur de limitation de taux.
    
    Levée quand la limite de requêtes par période est atteinte.
    
    Attributes:
        retry_after: Nombre de secondes à attendre avant de réessayer
    """
    
    def __init__(
        self,
        message: str = "Rate limit atteint",
        retry_after: Optional[int] = None,
        **kwargs
    ):
        self.retry_after = retry_after
        super().__init__(message, **kwargs)
    
    def __str__(self) -> str:
        error_str = super().__str__()
        if self.retry_after:
            error_str += f" (Réessayez dans {self.retry_after}s)"
        return error_str


class NotFoundError(MAIntelligenceError):
    """
    Erreur de ressource non trouvée.
    
    Levée quand:
    - Entreprise avec ID spécifié n'existe pas
    - Endpoint inexistant
    """
    
    def __init__(self, message: str = "Ressource non trouvée", **kwargs):
        super().__init__(message, status_code=404, **kwargs)


class ValidationError(MAIntelligenceError):
    """
    Erreur de validation des données.
    
    Levée quand:
    - Données requises manquantes
    - Format de données invalide
    - Contraintes de validation non respectées
    
    Attributes:
        details: Liste détaillée des erreurs de validation
    """
    
    def __init__(
        self,
        message: str = "Erreur de validation",
        details: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ):
        self.details = details or []
        super().__init__(message, status_code=422, **kwargs)
    
    def __str__(self) -> str:
        error_str = super().__str__()
        
        if self.details:
            detail_messages = []
            for detail in self.details:
                if isinstance(detail, dict):
                    field = detail.get("loc", ["unknown"])[-1]
                    msg = detail.get("msg", "Erreur inconnue")
                    detail_messages.append(f"{field}: {msg}")
                else:
                    detail_messages.append(str(detail))
            
            if detail_messages:
                error_str += f" - {'; '.join(detail_messages)}"
        
        return error_str


class ConnectionError(MAIntelligenceError):
    """
    Erreur de connexion réseau.
    
    Levée quand:
    - Impossible de se connecter à l'API
    - Timeout de connexion
    - Erreurs SSL/TLS
    """
    
    def __init__(self, message: str = "Erreur de connexion", **kwargs):
        super().__init__(message, **kwargs)


class ServerError(MAIntelligenceError):
    """
    Erreur serveur (5xx).
    
    Levée quand:
    - Erreur interne du serveur
    - Service temporairement indisponible
    - Erreur de configuration serveur
    """
    
    def __init__(self, message: str = "Erreur serveur", **kwargs):
        super().__init__(message, **kwargs)


class ConfigurationError(MAIntelligenceError):
    """
    Erreur de configuration du SDK.
    
    Levée quand:
    - Configuration manquante ou invalide
    - URL de base malformée
    - Paramètres de client HTTP invalides
    """
    
    def __init__(self, message: str = "Erreur de configuration", **kwargs):
        super().__init__(message, **kwargs)


class ResponseParsingError(MAIntelligenceError):
    """
    Erreur de parsing de la réponse API.
    
    Levée quand:
    - Réponse JSON malformée
    - Structure de données inattendue
    - Type de données incompatible
    """
    
    def __init__(self, message: str = "Erreur de parsing de la réponse", **kwargs):
        super().__init__(message, **kwargs)


# Exceptions compatibility aliases
APIError = MAIntelligenceError
AuthError = AuthenticationError
RateLimitExceeded = RateLimitError
NotFound = NotFoundError
InvalidData = ValidationError


def handle_http_error(status_code: int, response_data: Dict[str, Any]) -> MAIntelligenceError:
    """
    Crée l'exception appropriée selon le code de statut HTTP.
    
    Args:
        status_code: Code de statut HTTP
        response_data: Données de la réponse d'erreur
    
    Returns:
        MAIntelligenceError: Exception appropriée
    """
    
    error_info = response_data.get("error", {})
    message = error_info.get("message", f"Erreur HTTP {status_code}")
    request_id = error_info.get("request_id")
    
    if status_code == 401:
        return AuthenticationError(message, status_code=status_code, request_id=request_id)
    elif status_code == 403:
        return AuthenticationError(
            "Permissions insuffisantes", 
            status_code=status_code, 
            request_id=request_id
        )
    elif status_code == 404:
        return NotFoundError(message, request_id=request_id)
    elif status_code == 422:
        details = response_data.get("detail", [])
        return ValidationError(message, details=details, request_id=request_id)
    elif status_code == 429:
        retry_after = None
        if "rate_limit" in error_info:
            retry_after = error_info["rate_limit"].get("retry_after")
        return RateLimitError(message, retry_after=retry_after, request_id=request_id)
    elif 500 <= status_code < 600:
        return ServerError(message, status_code=status_code, request_id=request_id)
    else:
        return MAIntelligenceError(message, status_code=status_code, request_id=request_id)