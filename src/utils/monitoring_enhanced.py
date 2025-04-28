from typing import Dict, Optional
import logging
from logging.config import dictConfig
import structlog
from prometheus_client import start_http_server, Counter, Gauge, Histogram
import opentelemetry.trace as trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.exporter.prometheus import PrometheusMetricReader
import psutil
import GPUtil

class EnhancedMonitoring:
    def __init__(self, settings):
        self.settings = settings
        self.metrics = self._setup_metrics()
        self._setup_logging()
        self._setup_tracing()
        self._setup_metrics_export()
        
        # Start monitoring tasks
        self.start_monitoring()
    
    def _setup_metrics(self) -> Dict:
        return {
            # System metrics
            "system_cpu_usage": Gauge(
                "system_cpu_usage_percent",
                "System CPU usage"
            ),
            "system_memory_usage": Gauge(
                "system_memory_usage_bytes",
                "System memory usage"
            ),
            "system_gpu_usage": Gauge(
                "system_gpu_usage_percent",
                "GPU usage percentage",
                ["gpu_id"]
            ),
            
            # Application metrics
            "request_duration": Histogram(
                "app_request_duration_seconds",
                "Request duration in seconds",
                ["endpoint", "method", "status"]
            ),
            "active_connections": Gauge(
                "app_active_connections",
                "Number of active connections"
            ),
            
            # Model metrics
            "model_inference_time": Histogram(
                "model_inference_time_seconds",
                "Model inference time",
                ["model_name", "version"]
            ),
            "model_memory_usage": Gauge(
                "model_memory_usage_bytes",
                "Model memory usage",
                ["model_name", "version"]
            ),
            
            # Cache metrics
            "cache_hits": Counter(
                "cache_hits_total",
                "Cache hits",
                ["cache_type"]
            ),
            "cache_misses": Counter(
                "cache_misses_total",
                "Cache misses",
                ["cache_type"]
            ),
            
            # Error metrics
            "error_count": Counter(
                "error_count_total",
                "Error count",
                ["error_type", "severity"]
            )
        }
    
    def _setup_logging(self):
        """Setup enhanced logging"""
        logging_config = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "json": {
                    "()": structlog.stdlib.ProcessorFormatter,
                    "processor": structlog.processors.JSONRenderer(),
                },
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "json",
                },
                "file": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "filename": "logs/app.log",
                    "maxBytes": 10485760,
                    "backupCount": 5,
                    "formatter": "json",
                },
                "error_file": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "filename": "logs/error.log",
                    "maxBytes": 10485760,
                    "backupCount": 5,
                    "formatter": "json",
                },
            },
            "loggers": {
                "": {
                    "handlers": ["console", "file"],
                    "level": self.settings.LOG_LEVEL,
                },
                "error": {
                    "handlers": ["error_file"],
                    "level": "ERROR",
                    "propagate": False,
                },
            },
        }
        
        dictConfig(logging_config)
    
    def _setup_tracing(self):
        """Setup OpenTelemetry tracing"""
        trace.set_tracer_provider(TracerProvider())
        jaeger_exporter = JaegerExporter(
            agent_host_name=self.settings.JAEGER_HOST,
            agent_port=self.settings.JAEGER_PORT,
        )
        trace.get_tracer_provider().add_span_processor(
            BatchSpanProcessor(jaeger_exporter)
        )
    
    def _setup_metrics_export(self):
        """Setup metrics export"""
        if self.settings.PROMETHEUS_ENABLED:
            start_http_server(self.settings.METRICS_PORT)
    
    async def collect_system_metrics(self):
        """Collect system metrics"""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            self.metrics["system_cpu_usage"].set(cpu_percent)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            self.metrics["system_memory_usage"].set(memory.used)
            
            # GPU metrics
            try:
                gpus = GPUtil.getGPUs()
                for gpu in gpus:
                    self.metrics["system_gpu_usage"].labels(
                        gpu_id=gpu.id
                    ).set(gpu.load * 100)
            except Exception:
                pass
            
        except Exception as e:
            logging.error(f"Error collecting system metrics: {e}")
    
    async def record_request_metric(
        self,
        endpoint: str,
        method: str,
        status: int,
        duration: float
    ):
        """Record request metrics"""
        self.metrics["request_duration"].labels(
            endpoint=endpoint,
            method=method,
            status=status
        ).observe(duration)