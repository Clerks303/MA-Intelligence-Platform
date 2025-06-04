import re
from typing import Optional
from pydantic import BaseModel, validator
from fastapi import HTTPException, status

class SIRENValidator:
    @staticmethod
    def validate_siren(siren: str) -> str:
        """Validate SIREN format (9 digits)"""
        if not siren or not siren.isdigit() or len(siren) != 9:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="SIREN must be exactly 9 digits"
            )
        return siren

    @staticmethod
    def validate_siret(siret: str) -> str:
        """Validate SIRET format (14 digits)"""
        if not siret or not siret.isdigit() or len(siret) != 14:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="SIRET must be exactly 14 digits"
            )
        return siret

class PasswordValidator:
    @staticmethod
    def validate_password(password: str) -> str:
        """Validate password strength"""
        if len(password) < 8:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Password must be at least 8 characters long"
            )
        
        if not re.search(r'[A-Z]', password):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Password must contain at least one uppercase letter"
            )
        
        if not re.search(r'[a-z]', password):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Password must contain at least one lowercase letter"
            )
        
        if not re.search(r'[0-9]', password):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Password must contain at least one digit"
            )
        
        return password

class SearchValidator:
    @staticmethod
    def sanitize_search_term(term: str) -> str:
        """Sanitize search term to prevent SQL injection"""
        if not term:
            return ""
        
        # Remove SQL injection patterns
        sanitized = re.sub(r'[%_\\]', r'\\\g<0>', term)
        sanitized = re.sub(r'[\'";]', '', sanitized)
        
        # Limit length
        if len(sanitized) > 100:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Search term too long (max 100 characters)"
            )
        
        return sanitized

class FileValidator:
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS = {'.csv'}
    
    @classmethod
    def validate_upload_file(cls, filename: str, file_size: int) -> None:
        """Validate uploaded file"""
        if not filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Filename is required"
            )
        
        # Check extension
        extension = '.' + filename.split('.')[-1].lower() if '.' in filename else ''
        if extension not in cls.ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File type not allowed. Allowed types: {', '.join(cls.ALLOWED_EXTENSIONS)}"
            )
        
        # Check size
        if file_size > cls.MAX_FILE_SIZE:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File too large. Maximum size: {cls.MAX_FILE_SIZE // (1024*1024)}MB"
            )