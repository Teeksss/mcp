from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
import time
from src.services.caching.smart_cache import SmartCache

class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app):
        super().__init__(app)
        self.cache = SmartCache()
    
    async def dispatch(self, request: Request, call_next):
        # Get client IP
        client_ip = request.client.host
        
        # Check rate limit
        if await self._is_rate_limited(client_ip):
            raise HTTPException(
                status_code=429,
                detail="Too many requests"
            )
        
        # Update request count
        await self._update_request_count(client_ip)
        
        response = await call_next(request)
        return response
    
    async def _is_rate_limited(self, client_ip: str) -> bool:
        key = f"rate_limit:{client_ip}"
        count = await self.cache.get(key) or 0
        return count >= 100  # 100 requests per minute
    
    async def _update_request_count(self, client_ip: str):
        key = f"rate_limit:{client_ip}"
        await self.cache.set(
            key,
            (await self.cache.get(key) or 0) + 1,
            ttl=60
        )