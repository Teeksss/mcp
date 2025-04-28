from datetime import datetime
import asyncio
from typing import Dict, List, Optional
import aiohttp
import numpy as np
from dataclasses import dataclass
import logging
from prometheus_client import Counter, Histogram, Gauge

logger = logging.getLogger(__name__)

@dataclass
class RouterConfig:
    health_check_interval: int = 15
    connection_timeout: int = 5
    max_retries: int = 3
    circuit_breaker_threshold: int = 5
    recovery_timeout: int = 60

class ModelEndpoint:
    def __init__(
        self,
        url: str,
        weight: float = 1.0,
        max_concurrent: int = 100
    ):
        self.url = url
        self.weight = weight
        self.max_concurrent = max_concurrent
        self.current_load = 0
        self.error_count = 0
        self.last_error_time = None
        self.is_healthy = True
        self.response_times = []
        
        # Current context
        self.last_check = datetime.utcnow()

class SmartRouter:
    def __init__(self, config: RouterConfig):
        self.config = config
        self.endpoints: Dict[str, List[ModelEndpoint]] = {}
        self.metrics = self._setup_metrics()
        self.health_check_task = None
    
    def _setup_metrics(self):
        """Initialize Prometheus metrics"""
        return {
            "requests": Counter(
                "router_requests_total",
                "Total requests routed",
                ["model", "endpoint"]
            ),
            "errors": Counter(
                "router_errors_total",
                "Total routing errors",
                ["model", "endpoint", "error_type"]
            ),
            "latency": Histogram(
                "router_latency_seconds",
                "Request latency",
                ["model", "endpoint"]
            ),
            "endpoint_health": Gauge(
                "router_endpoint_health",
                "Endpoint health status",
                ["model", "endpoint"]
            )
        }
    
    async def start(self):
        """Start the router"""
        self.health_check_task = asyncio.create_task(
            self._health_check_loop()
        )
    
    async def stop(self):
        """Stop the router"""
        if self.health_check_task:
            self.health_check_task.cancel()
            try:
                await self.health_check_task
            except asyncio.CancelledError:
                pass
    
    def register_endpoint(
        self,
        model: str,
        endpoint: ModelEndpoint
    ):
        """Register a new endpoint for a model"""
        if model not in self.endpoints:
            self.endpoints[model] = []
        self.endpoints[model].append(endpoint)
        logger.info(
            f"Registered endpoint {endpoint.url} for model {model}"
        )
    
    async def route_request(
        self,
        model: str,
        request_data: Dict,
        timeout: Optional[int] = None
    ) -> Dict:
        """Route request to appropriate endpoint"""
        if model not in self.endpoints:
            raise ValueError(f"No endpoints registered for model {model}")
        
        # Get available endpoints
        available_endpoints = [
            ep for ep in self.endpoints[model]
            if ep.is_healthy and
            ep.current_load < ep.max_concurrent
        ]
        
        if not available_endpoints:
            raise RuntimeError(f"No available endpoints for model {model}")
        
        # Select endpoint using weighted random selection
        weights = [
            ep.weight * (1 - ep.current_load / ep.max_concurrent)
            for ep in available_endpoints
        ]
        selected_endpoint = np.random.choice(
            available_endpoints,
            p=np.array(weights) / sum(weights)
        )
        
        # Send request
        try:
            start_time = datetime.utcnow()
            selected_endpoint.current_load += 1
            
            response = await self._send_request(
                selected_endpoint,
                request_data,
                timeout or self.config.connection_timeout
            )
            
            # Update metrics
            duration = (datetime.utcnow() - start_time).total_seconds()
            self.metrics["latency"].labels(
                model=model,
                endpoint=selected_endpoint.url
            ).observe(duration)
            
            self.metrics["requests"].labels(
                model=model,
                endpoint=selected_endpoint.url
            ).inc()
            
            selected_endpoint.response_times.append(duration)
            if len(selected_endpoint.response_times) > 100:
                selected_endpoint.response_times.pop(0)
            
            return response
            
        except Exception as e:
            self.metrics["errors"].labels(
                model=model,
                endpoint=selected_endpoint.url,
                error_type=type(e).__name__
            ).inc()
            
            await self._handle_error(selected_endpoint, e)
            raise
            
        finally:
            selected_endpoint.current_load -= 1
    
    async def _send_request(
        self,
        endpoint: ModelEndpoint,
        data: Dict,
        timeout: int
    ) -> Dict:
        """Send request to endpoint with retry logic"""
        for attempt in range(self.config.max_retries):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        endpoint.url,
                        json=data,
                        timeout=timeout
                    ) as response:
                        if response.status == 200:
                            return await response.json()
                        else:
                            raise RuntimeError(
                                f"Endpoint returned status {response.status}"
                            )
            except Exception as e:
                if attempt == self.config.max_retries - 1:
                    raise
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
    
    async def _handle_error(
        self,
        endpoint: ModelEndpoint,
        error: Exception
    ):
        """Handle endpoint errors"""
        endpoint.error_count += 1
        endpoint.last_error_time = datetime.utcnow()
        
        if endpoint.error_count >= self.config.circuit_breaker_threshold:
            endpoint.is_healthy = False
            logger.warning(
                f"Circuit breaker triggered for endpoint {endpoint.url}"
            )
    
    async def _health_check_loop(self):
        """Periodically check endpoint health"""
        while True:
            try:
                for model, endpoints in self.endpoints.items():
                    for endpoint in endpoints:
                        await self._check_endpoint_health(model, endpoint)
                
            except Exception as e:
                logger.error(f"Health check error: {e}")
            
            await asyncio.sleep(self.config.health_check_interval)
    
    async def _check_endpoint_health(
        self,
        model: str,
        endpoint: ModelEndpoint
    ):
        """Check health of individual endpoint"""
        try:
            # Skip check if endpoint was marked unhealthy recently
            if (
                not endpoint.is_healthy and
                endpoint.last_error_time and
                (datetime.utcnow() - endpoint.last_error_time).seconds <
                self.config.recovery_timeout
            ):
                return
            
            # Send health check request
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{endpoint.url}/health",
                    timeout=self.config.connection_timeout
                ) as response:
                    if response.status == 200:
                        endpoint.is_healthy = True
                        endpoint.error_count = 0
                        self.metrics["endpoint_health"].labels(
                            model=model,
                            endpoint=endpoint.url
                        ).set(1)
                    else:
                        raise RuntimeError(
                            f"Health check failed with status {response.status}"
                        )
            
        except Exception as e:
            logger.error(f"Health check failed for {endpoint.url}: {e}")
            self.metrics["endpoint_health"].labels(
                model=model,
                endpoint=endpoint.url
            ).set(0)
            await self._handle_error(endpoint, e)