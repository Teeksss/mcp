import logging
import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration
from prometheus_client import Counter, Histogram, Gauge
import time
from typing import Dict, Optional
from dataclasses import dataclass
import psutil
import torch

@dataclass
class MonitoringConfig:
    sentry_dsn: str
    environment: str
    traces_sample_rate: float = 1.0
    performance_sample_rate: float = 0.1
    enable_profiling: bool = True

class MonitoringService:
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self._setup_sentry()
        self._setup_metrics()
        self._setup_logging()
    
    def _setup_sentry(self):
        """Initialize Sentry for error tracking"""
        sentry_sdk.init(
            dsn=self.config.sentry_dsn,
            environment=self.config.environment,
            traces_sample_rate=self.config.traces_sample_rate,
            profiles_sample_rate=self.config.performance_sample_rate,
            enable_tracing=True,
            integrations=[FastApiIntegration()]
        )
    
    def _setup_metrics(self):
        """Initialize Prometheus metrics"""
        # Request metrics
        self.request_counter = Counter(
            'mcp_requests_total',
            'Total requests processed',
            ['method', 'endpoint', 'status']
        )
        
        self.request_duration = Histogram(
            'mcp_request_duration_seconds',
            'Request duration in seconds',
            ['method', 'endpoint']
        )
        
        # Model metrics
        self.model_inference_time = Histogram(
            'mcp_model_inference_seconds',
            'Model inference time in seconds',
            ['model_name']
        )
        
        self.model_error_counter = Counter(
            'mcp_model_errors_total',
            'Total model errors',
            ['model_name', 'error_type']
        )
        
        # Resource metrics
        self.gpu_memory_usage = Gauge(
            'mcp_gpu_memory_bytes',
            'GPU memory usage in bytes',
            ['device']
        )
        
        self.cpu_usage = Gauge(
            'mcp_cpu_usage_percent',
            'CPU usage percentage'
        )
        
        # Cache metrics
        self.cache_hits = Counter(
            'mcp_cache_hits_total',
            'Total cache hits'
        )
        
        self.cache_misses = Counter(
            'mcp_cache_misses_total',
            'Total cache misses'
        )
    
    def _setup_logging(self):
        """Configure logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Add Sentry handler
        logging.getLogger().addHandler(
            logging.StreamHandler()
        )
    
    async def track_request(
        self,
        method: str,
        endpoint: str,
        status: int,
        duration: float
    ):
        """Track HTTP request metrics"""
        self.request_counter.labels(
            method=method,
            endpoint=endpoint,
            status=status
        ).inc()
        
        self.request_duration.labels(
            method=method,
            endpoint=endpoint
        ).observe(duration)
    
    async def track_model_inference(
        self,
        model_name: str,
        duration: float
    ):
        """Track model inference metrics"""
        self.model_inference_time.labels(
            model_name=model_name
        ).observe(duration)
    
    async def track_model_error(
        self,
        model_name: str,
        error_type: str
    ):
        """Track model errors"""
        self.model_error_counter.labels(
            model_name=model_name,
            error_type=error_type
        ).inc()
    
    async def update_resource_metrics(self):
        """Update system resource metrics"""
        # Update CPU metrics
        cpu_percent = psutil.cpu_percent()
        self.cpu_usage.set(cpu_percent)
        
        # Update GPU metrics if available
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                memory_allocated = torch.cuda.memory_allocated(i)
                self.gpu_memory_usage.labels(
                    device=f'cuda:{i}'
                ).set(memory_allocated)
    
    async def track_cache_access(self, hit: bool):
        """Track cache access metrics"""
        if hit:
            self.cache_hits.inc()
        else:
            self.cache_misses.inc()
    
    def capture_exception(self, exception: Exception):
        """Capture exception in Sentry"""
        sentry_sdk.capture_exception(exception)