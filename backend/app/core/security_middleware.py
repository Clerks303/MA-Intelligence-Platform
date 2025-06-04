import time
import logging
from typing import Set
from app.models.user import User
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
import re

logger = logging.getLogger(__name__)

class SecurityMiddleware(BaseHTTPMiddleware):
    """Comprehensive security middleware for FastAPI"""
    
    def __init__(self, app):
        super().__init__(app)
        
        # Malicious user agents patterns
        self.blocked_user_agents = {
            # Vulnerability scanners
            r'.*(?:nikto|sqlmap|nessus|openvas).*',
            # Common attack tools
            r'.*(?:masscan|nmap|zap).*',
            # Suspicious agents
            r'libwww-perl.*',
            r'.*scanner.*',
            # Empty or missing user agent
            r'^$',
            r'^\s*$'
        }
        
        # Compile regex patterns for performance
        self.blocked_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.blocked_user_agents]
        
        # Suspicious request patterns
        self.suspicious_paths = {
            # Admin paths
            '/admin', '/wp-admin', '/administrator',
            # Common vulnerable paths
            '/phpmyadmin', '/.env', '/config', '/backup',
            # API abuse patterns
            '/api/../', '//api',
        }
        
        # SQL injection patterns
        self.sql_injection_patterns = [
            re.compile(pattern, re.IGNORECASE) for pattern in [
                r"union\s+select",
                r"or\s+1\s*=\s*1",
                r"and\s+1\s*=\s*1",
                r"'\s*or\s*'",
                r";\s*drop\s+table",
                r";\s*delete\s+from",
                r"script\s*:",
                r"javascript\s*:",
                r"<\s*script",
                r"eval\s*\(",
                r"expression\s*\("
            ]
        ]
    
    def is_blocked_user_agent(self, user_agent: str) -> bool:
        """Check if user agent is blocked"""
        if not user_agent:
            return True
        
        for pattern in self.blocked_patterns:
            if pattern.match(user_agent):
                return True
        return False
    
    def has_suspicious_path(self, path: str) -> bool:
        """Check for suspicious path patterns"""
        path_lower = path.lower()
        
        # Check exact matches
        for suspicious_path in self.suspicious_paths:
            if suspicious_path in path_lower:
                return True
        
        # Check for path traversal
        if '../' in path or '..\\' in path:
            return True
        
        # Check for double slashes
        if '//' in path and not path.startswith('http'):
            return True
        
        return False
    
    def has_sql_injection(self, query_string: str) -> bool:
        """Check for SQL injection patterns"""
        content = query_string.lower()
        
        for pattern in self.sql_injection_patterns:
            if pattern.search(content):
                return True
        return False
    
    def get_client_info(self, request: Request) -> dict:
        """Extract client information for logging"""
        return {
            "ip": request.client.host if request.client else "unknown",
            "user_agent": request.headers.get("User-Agent", ""),
            "method": request.method,
            "path": request.url.path,
            "query": str(request.url.query),
            "referer": request.headers.get("Referer", ""),
            "x_forwarded_for": request.headers.get("X-Forwarded-For", ""),
            "x_real_ip": request.headers.get("X-Real-IP", "")
        }
    
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        client_info = self.get_client_info(request)
        
        # 1. Block malicious user agents
        user_agent = request.headers.get("User-Agent", "")
        if self.is_blocked_user_agent(user_agent):
            logger.warning(f"Blocked malicious user agent: {client_info}")
            return JSONResponse(
                status_code=403,
                content={"detail": "Access denied"}
            )
        
        # 2. Check for suspicious paths
        if self.has_suspicious_path(request.url.path):
            logger.warning(f"Suspicious path access: {client_info}")
            return JSONResponse(
                status_code=404,
                content={"detail": "Not found"}
            )
        
        # 3. Check for SQL injection attempts in query string only
        query_string = str(request.url.query)
        
        if self.has_sql_injection(query_string):
            logger.error(f"SQL injection attempt detected: {client_info}")
            return JSONResponse(
                status_code=400,
                content={"detail": "Invalid request"}
            )
        
        # 4. Process the request
        try:
            response = await call_next(request)
        except Exception as e:
            logger.error(f"Request processing error: {e} - {client_info}")
            return JSONResponse(
                status_code=500,
                content={"detail": "Internal server error"}
            )
        
        # 5. Add security headers
        self.add_security_headers(response)
        
        # 6. Log the request
        process_time = time.time() - start_time
        self.log_request(client_info, response.status_code, process_time)
        
        return response
    
    def add_security_headers(self, response: Response):
        """Add security headers to response"""
        # Prevent XSS attacks
        response.headers["X-XSS-Protection"] = "1; mode=block"
        
        # Prevent content type sniffing
        response.headers["X-Content-Type-Options"] = "nosniff"
        
        # Prevent clickjacking
        response.headers["X-Frame-Options"] = "DENY"
        
        # Content Security Policy
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: https:; "
            "connect-src 'self'"
        )
        
        # Referrer Policy
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        
        # Permissions Policy
        response.headers["Permissions-Policy"] = (
            "camera=(), microphone=(), geolocation=(), "
            "payment=(), usb=(), magnetometer=(), gyroscope=()"
        )
        
        # Remove server information
        if "server" in response.headers:
            del response.headers["server"]
    
    def log_request(self, client_info: dict, status_code: int, process_time: float):
        """Log request information"""
        # Log suspicious activity
        if status_code >= 400:
            if status_code == 403:
                logger.warning(f"Access denied: {client_info}")
            elif status_code == 404:
                logger.info(f"Not found: {client_info}")
            elif status_code >= 500:
                logger.error(f"Server error {status_code}: {client_info}")
        
        # Log slow requests
        if process_time > 5.0:  # Requests taking more than 5 seconds
            logger.warning(f"Slow request ({process_time:.2f}s): {client_info}")
        
        # Regular access log (debug level)
        logger.debug(
            f"{client_info['ip']} - {client_info['method']} {client_info['path']} "
            f"- {status_code} - {process_time:.3f}s"
        )