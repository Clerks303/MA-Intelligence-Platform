from fastapi import HTTPException, status

class CompanyNotFoundError(HTTPException):
    def __init__(self, identifier: str):
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Company with identifier {identifier} not found"
        )

class InvalidSIRENError(HTTPException):
    def __init__(self, siren: str):
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid SIREN format: {siren}. Must be 9 digits."
        )

class DatabaseError(HTTPException):
    def __init__(self, operation: str):
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error during {operation}"
        )

class ScrapingError(HTTPException):
    def __init__(self, source: str, message: str = "Scraping operation failed"):
        super().__init__(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"{source}: {message}"
        )

class QuotaExceededError(HTTPException):
    def __init__(self, service: str):
        super().__init__(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"API quota exceeded for {service}. Please try again later."
        )

class FileValidationError(HTTPException):
    def __init__(self, message: str):
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File validation error: {message}"
        )

class ValidationError(Exception):
    def __init__(self, message: str = "Validation error", payload: dict = None):
        self.message = message
        self.payload = payload or {}
        super().__init__(self.message)

    def to_dict(self):
        rv = dict(self.payload or ())
        rv['message'] = self.message
        return rv