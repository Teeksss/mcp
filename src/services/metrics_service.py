from prometheus_client import Counter, Histogram, Gauge
import logging
from datetime import datetime, timedelta
import asyncio
from collections import defaultdict
from src.config.settings import settings

logger = logging.getLogger(__name__)

class MetricsService:
    def __init__(self):
        # Request metrics
        self.request_count = Counter(
            f'{settings.monitoring.metrics_prefix}_request_total',
            'Total number of requests',
            ['model', 'status']
        )
        
        self.response_time = Histogram(
            f'{settings.monitoring.metrics_prefix}_response_time_seconds',
            'Response time in seconds',
            ['model'],
            buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0)
        )
        
        # Cache metrics
        self.cache_hits = Counter(
            f'{settings.monitoring.metrics_prefix}_cache_hits_total',
            'Number of cache hits'
        )
        self.cache_misses = Counter(
            f'{settings.monitoring.metrics_prefix}_cache_misses_total',
            'Number of cache misses'
        )
        
        # RAG metrics
        self.rag_retrieval_time = Histogram(
            f'{settings.monitoring.metrics_prefix}_rag_retrieval_time_seconds',
            'RAG retrieval time in seconds',
            buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)
        )
        self.rag_document_count = Gauge(
            f'{settings.monitoring.metrics_prefix}_rag_document_count',
            'Number of documents in RAG system'
        )
        
        # Model metrics
        self.model_token_count = Counter(
            f'{settings.monitoring.metrics_prefix}_model_tokens_total',
            'Total number of tokens processed',
            ['model']
        )
        self.model_error_count = Counter(
            f'{settings.monitoring.metrics_prefix}_model_errors_total',
            'Number of model errors',
            ['model', 'error_type']
        )
        
        # Resource metrics
        self.gpu_memory_usage = Gauge(
            f'{settings.monitoring.metrics_prefix}_gpu_memory_bytes',
            'GPU memory usage in bytes',
            ['device']
        )
        self.cpu_usage = Gauge(
            f'{settings.monitoring.metrics_prefix}_cpu_usage_percent',
            'CPU usage percentage'
        )
        
        # Initialize periodic metric collection
        self._start_periodic_collection()
    
    def _start_periodic_collection(self):
        asyncio.create_task(self._collect_resource_metrics())
    
    async def _collect_resource_metrics(self):
        while True:
            try:
                # Collect GPU metrics
                if torch.cuda.is_available():
                    for i in range(torch.cuda.device_count()):
                        memory_allocated = torch.cuda.memory_allocated(i)
                        self.gpu_memory_usage.labels(device=f'cuda:{i}').set(
                            memory_allocated
                        )
                
                # Collect CPU metrics
                cpu_percent = psutil.cpu_percent()
                self.cpu_usage.set(cpu_percent)
                
            except Exception as e:
                logger.error(f"Error collecting resource metrics: {e}")
            
            await asyncio.sleep(15)  # Collect every 15 seconds
    
    def record_request(
        self,
        model: str,
        status: str,
        duration: float,
        token_count: int
    ):
        """Record metrics for a single request"""
        self.request_count.labels(model=model, status=status).inc()
        self.response_time.labels(model=model).observe(duration)
        self.model_token_count.labels(model=model).inc(token_count)
    
    def record_cache_access(self, hit: bool):
        """Record cache hit/miss"""
        if hit:
            self.cache_hits.inc()
        else:
            self.cache_misses.inc()
    
    def record_rag_retrieval(self, duration: float):
        """Record RAG retrieval time"""
        self.rag_retrieval_time.observe(duration)
    
    def record_model_error(self, model: str, error_type: str):
        """Record model errors"""
        self.model_error_count.labels(
            model=model,
            error_type=error_type
        ).inc()
    
    def update_rag_document_count(self, count: int):
        """Update RAG document count"""
        self.rag_document_count.set(count)
    
    async def get_metrics_summary(self, time_window: int = 3600) -> dict:
        """Get a summary of metrics for the last hour"""
        now = datetime.now()
        window_start = now - timedelta(seconds=time_window)
        
        try:
            # Get request statistics
            total_requests = sum(
                self.request_count._value.values()
            )
            success_rate = (
                sum(v for k, v in self.request_count._value.items()
                    if k[1] == 'success') / total_requests
                if total_requests > 0 else 0
            )
            
            # Calculate cache hit rate
            total_cache_accesses = (
                self.cache_hits._value +
                self.cache_misses._value
            )
            cache_hit_rate = (
                self.cache_hits._value / total_cache_accesses
                if total_cache_accesses > 0 else 0
            )
            
            # Get average response times
            response_times = self.response_time._sum.values()
            avg_response_time = (
                sum(response_times) / len(response_times)
                if response_times else 0
            )
            
            return {
                'timestamp': now.isoformat(),
                'window_seconds': time_window,
                'total_requests': total_requests,
                'success_rate': success_rate,
                'cache_hit_rate': cache_hit_rate,
                'average_response_time': avg_response_time,
                'total_tokens_processed': sum(
                    self.model_token_count._value.values()
                ),
                'error_count': sum(
                    self.model_error_count._value.values()
                ),
                'rag_document_count': self.rag_document_count._value,
                'gpu_memory_usage': {
                    device: value
                    for device, value in self.gpu_memory_usage._value.items()
                },
                'cpu_usage': self.cpu_usage._value
            }
            
        except Exception as e:
            logger.error(f"Error getting metrics summary: {e}")
            return {}