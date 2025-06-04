import time
import asyncio
from collections import defaultdict, deque
from typing import Dict, Tuple
from fastapi import Request, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
import logging

logger = logging.getLogger(__name__)

class InMemoryRateLimiter:
    """In-memory rate limiter using sliding window algorithm"""
    
    def __init__(self):
        self.clients: Dict[str, deque] = defaultdict(deque)
        self.lock = asyncio.Lock()
    
    async def is_allowed(self, client_id: str, limit: int, window: int) -> bool:
        """
        Check if client is allowed to make request
        Args:
            client_id: Client identifier (IP address)
            limit: Maximum requests allowed
            window: Time window in seconds
        Returns:
            True if allowed, False if rate limited
        """
        async with self.lock:
            now = time.time()
            client_requests = self.clients[client_id]
            
            # Remove old requests outside the window
            while client_requests and client_requests[0] <= now - window:
                client_requests.popleft()
            
            # Check if limit exceeded
            if len(client_requests) >= limit:
                return False
            
            # Add current request
            client_requests.append(now)
            return True

class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware for FastAPI"""
    
    def __init__(self, app, calls_per_minute: int = 100, calls_per_hour: int = 1000):
        super().__init__(app)
        self.limiter = InMemoryRateLimiter()
        self.calls_per_minute = calls_per_minute
        self.calls_per_hour = calls_per_hour
        
        # Special limits for specific endpoints
        self.endpoint_limits = {
            "/api/v1/auth/login": (5, 60),  # 5 per minute for login
            "/api/v1/companies/upload": (3, 3600),  # 3 per hour for uploads
            "/api/v1/scraping/": (2, 3600),  # 2 per hour for scraping
        }
    
    def get_client_id(self, request: Request) -> str:
        """Extract client identifier from request"""
        # Try to get real IP from headers (for reverse proxy setups)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        # Fallback to direct client IP
        return request.client.host if request.client else "unknown"
    
    def get_rate_limit(self, path: str) -> Tuple[int, int]:
        """Get rate limit for specific endpoint"""
        # Check for exact match first
        if path in self.endpoint_limits:
            return self.endpoint_limits[path]
        
        # Check for prefix matches
        for endpoint_path, limits in self.endpoint_limits.items():
            if path.startswith(endpoint_path):
                return limits
        
        # Default rate limit
        return self.calls_per_minute, 60
    
    async def dispatch(self, request: Request, call_next):
        # Skip rate limiting for health checks and docs
        if request.url.path in ["/health", "/docs", "/openapi.json", "/"]:
            return await call_next(request)
        
        client_id = self.get_client_id(request)
        path = request.url.path
        limit, window = self.get_rate_limit(path)
        
        # Check rate limit
        allowed = await self.limiter.is_allowed(client_id, limit, window)
        
        if not allowed:
            logger.warning(f"Rate limit exceeded for {client_id} on {path}")
            return HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded. Maximum {limit} requests per {window} seconds.",
                headers={"Retry-After": str(window)}
            ).get_response()
        
        # Add rate limit headers to response
        response = await call_next(request)
        
        # Get current usage for headers
        current_usage = len(self.limiter.clients[client_id])
        response.headers["X-RateLimit-Limit"] = str(limit)
        response.headers["X-RateLimit-Remaining"] = str(max(0, limit - current_usage))
        response.headers["X-RateLimit-Reset"] = str(int(time.time() + window))
        
        return response

class EndpointRateLimiter:
    """Decorator for specific endpoint rate limiting"""
    
    def __init__(self, calls: int, window: int):
        self.calls = calls
        self.window = window
        self.limiter = InMemoryRateLimiter()
    
    def __call__(self, func):
        async def wrapper(*args, **kwargs):
            # Extract request from args/kwargs
            request = None
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break
            
            if not request:
                # If no request found, proceed without rate limiting
                return await func(*args, **kwargs)
            
            client_id = request.client.host if request.client else "unknown"
            allowed = await self.limiter.is_allowed(client_id, self.calls, self.window)
            
            if not allowed:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=f"Rate limit exceeded. Maximum {self.calls} requests per {self.window} seconds."
                )
            
            return await func(*args, **kwargs)
        
        return wrapper

# Predefined rate limiters for common use cases
login_rate_limit = EndpointRateLimiter(calls=5, window=60)  # 5 per minute
upload_rate_limit = EndpointRateLimiter(calls=3, window=3600)  # 3 per hour
scraping_rate_limit = EndpointRateLimiter(calls=2, window=3600)  # 2 per hour