"""
M&A Intelligence Platform SDK for Python
US-011: SDK client officiel pour intégration Python

Installation:
    pip install ma-intelligence-sdk

Usage rapide:
    from ma_intelligence_sdk import MAIntelligenceClient
    
    client = MAIntelligenceClient(
        api_key="ak_your_api_key",
        base_url="https://api.ma-intelligence.com"
    )
    
    companies = client.companies.list(city="Paris", ca_min=100000)
    for company in companies:
        print(f"{company.nom_entreprise} - {company.chiffre_affaires}")
"""

__version__ = "1.0.0"
__author__ = "M&A Intelligence Team"
__email__ = "sdk@ma-intelligence.com"

from .client import MAIntelligenceClient
from .models import Company, CompanyFilters, APIResponse, PaginatedResponse
from .exceptions import (
    MAIntelligenceError,
    AuthenticationError, 
    RateLimitError,
    NotFoundError,
    ValidationError
)

__all__ = [
    "MAIntelligenceClient",
    "Company",
    "CompanyFilters", 
    "APIResponse",
    "PaginatedResponse",
    "MAIntelligenceError",
    "AuthenticationError",
    "RateLimitError", 
    "NotFoundError",
    "ValidationError"
]