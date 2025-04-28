import logging
import logging.config
from typing import Dict
import json
from pythonjsonlogger import jsonlogger
import sys
from datetime import datetime

class CustomJsonFormatter(jsonlogger.JsonFormatter):
    def add_fields(
        self,
        log_record: Dict,
        record: logging.LogRecord,
        message_dict: Dict
    ) -> None:
        super().add_fields(log_record, record, message_dict)
        
        # Add custom fields
        log_record['timestamp'] = datetime.utcnow().isoformat()
        log_record['level'] = record.levelname
        log_record['logger'] = record.name
        
        # Add context if available
        if hasattr(record, 'user'):
            log_record['user'] = record.user
        if hasattr(record, 'request_id'):
            log_record['request_id'] = record.request_id

class RequestIdFilter(logging.Filter):
    """Add request ID to log records"""
    def __init__(self, request_id: str):
        super().__init__()
        self.request_id = request_id
    
    def filter(self, record):
        record.request_id = self.request_id
        return True

class UserFilter(logging.Filter):
    """Add user to log records"""
    def __init__(self, user: str):
        super().__init__()
        self.user = user
    
    def filter(self, record):
        record.user = self.user
        return True

def setup_logging(
    level: str = "INFO",
    json_format: bool = True,
    log_file: str = None
) -> None:
    """Setup logging configuration"""
    handlers = {
        'console': {
            'class': 'logging.StreamHandler',
            'stream': sys.stdout,
            'formatter': 'json' if json_format else 'standard'
        }
    }
    
    if log_file:
        handlers['file'] = {
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': log_file,
            'maxBytes': 10485760,  # 10MB
            'backupCount': 5,
            'formatter': 'json' if json_format else 'standard'
        }
    
    logging_config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S'
            },
            'json': {
                '()': CustomJsonFormatter,
                'fmt': '%(timestamp)s %(level)s %(name)s %(message)s'
            }
        },
        'handlers': handlers,
        'loggers': {
            '': {
                'handlers': list(handlers.keys()),
                'level': level,
                'propagate': True
            },
            'src': {
                'handlers': list(handlers.keys()),
                'level': level,
                'propagate': False
            }
        }
    }
    
    logging.config.dictConfig(logging_config)