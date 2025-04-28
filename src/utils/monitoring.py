import logging
from logging.config import dictConfig
import structlog
from prometheus_client import start_http_server
import opentelemetry.trace as trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

def setup_monitoring(settings):
    # Setup logging
    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "json": {
                "()": structlog.stdlib.ProcessorFormatter,
                "processor": structlog.processors.JSONRenderer(),
            },
            "standard": {
                "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "json",
            },
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "filename": "app.log",
                "maxBytes": 10485760,
                "backupCount": 5,
                "formatter": "json",
            },
        },
        "loggers": {
            "": {
                "handlers": ["console", "file"],
                "level": settings.LOG_LEVEL,
            }
        },
    }
    
    dictConfig(logging_config)
    
    # Setup tracing
    trace.set_tracer_provider(TracerProvider())
    jaeger_exporter = JaegerExporter(
        agent_host_name="localhost",
        agent_port=6831,
    )
    trace.get_tracer_provider().add_span_processor(
        BatchSpanProcessor(jaeger_exporter)
    )
    
    # Start Prometheus metrics server
    if settings.PROMETHEUS_ENABLED:
        start_http_server(settings.METRICS_PORT)