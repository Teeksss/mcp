from fastapi import Request, HTTPException
from fastapi.middleware.base import BaseHTTPMiddleware
from typing import Callable
import time
import logging
from prometheus_client import Counter, Histogram

logger = logging.getLogger(__name__)

class MetricsMiddleware(BaseHTTPMiddleware):
    def __init__(self, app):
        super().__init__(app)
        self.request_count = Counter(
            "http_requests_total",
            "Total HTTP requests",
            ["method", "endpoint", "status"]
        )
        self.request_latency = Histogram(
            "http_request_duration_seconds",
            "HTTP request duration",
            ["method", "endpoint"]
        )

    async def dispatch(
        self,
        request: Request,
        call_next: Callable
    ):
        start_time = time.time()
        
        try:
            response = await call_next(request)
            
            # Update metrics
            self.request_count.labels(
                method=request.method,
                endpoint=request.url.path,
                status=response.status_code
            ).inc()
            
            self.request_latency.labels(
                method=request.method,
                endpoint=request.url.path
            ).observe(time.time() - start_time)
            
            return response
            
        except Exception as e:
            logger.error(f"Request failed: {e}")
            self.request_count.labels(
                method=request.method,
                endpoint=request.url.path,
                status=500
            ).inc()
            raise

class AuthenticationMiddleware(BaseHTTPMiddleware):
    def __init__(self, app):
        super().__init__(app)
        self.auth_failures = Counter(
            "auth_failures_total",
            "Total authentication failures",
            ["method", "endpoint"]
        )

    async def dispatch(
        self,
        request: Request,
        call_next: Callable
    ):
        # Verify token
        token = request.headers.get("Authorization")
        if not token:
            self.auth_failures.labels(
                method=request.method,
                endpoint=request.url.path
            ).inc()
            raise HTTPException(
                status_code=401,
                detail="Missing authentication token"
            )
        
        # Verify token format
        if not token.startswith("Bearer "):
            self.auth_failures.labels(
                method=request.method,
                endpoint=request.url.path
            ).inc()
            raise HTTPException(
                status_code=401,
                detail="Invalid token format"
            )
        
        try:
            response = await call_next(request)
            return response
            
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            raise

class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app):
        super().__init__(app)
        self.rate_limit_exceeded = Counter(
            "rate_limit_exceeded_total",
            "Total rate limit exceeded",
            ["method", "endpoint"]
        )

    async def dispatch(
        self,
        request: Request,
        call_next: Callable
    ):
        # Check rate limit
        client_ip = request.client.host
        
        if await self._is_rate_limited(client_ip):
            self.rate_limit_exceeded.labels(
                method=request.method,
                endpoint=request.url.path
            ).inc()
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded"
            )
        
        return await call_next(request)
    
    async def _is_rate_limited(self, client_ip: str) -> bool:
        # Implementation depends on rate limiting strategy
        pass