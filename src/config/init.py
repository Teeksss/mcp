import os
import sys
from pathlib import Path
from typing import Dict, Any
import logging

from src.config.app_config import ConfigurationManager

def initialize_config() -> ConfigurationManager:
    """Initialize application configuration"""
    try:
        # Set up base paths
        root_dir = Path(__file__).parent.parent.parent
        config_dir = root_dir / "config"
        
        # Create required directories
        (root_dir / "logs").mkdir(exist_ok=True)
        (root_dir / "data").mkdir(exist_ok=True)
        
        # Initialize configuration manager
        config = ConfigurationManager(str(config_dir))
        
        # Validate configuration
        if not config.validate():
            raise ValueError("Configuration validation failed")
        
        # Set up enhanced logging
        _setup_enhanced_logging(config)
        
        return config
        
    except Exception as e:
        logging.error(f"Configuration initialization failed: {e}")
        sys.exit(1)

def _setup_enhanced_logging(config: ConfigurationManager):
    """Setup enhanced logging configuration"""
    log_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": config.logging.format,
                "datefmt": config.logging.date_format
            },
            "json": {
                "()": "pythonjsonlogger.jsonlogger.JsonFormatter",
                "fmt": "%(asctime)s %(name)s %(levelname)s %(message)s"
            }
        },
        "filters": {
            "environment": {
                "()": "src.utils.logging.EnvironmentFilter",
                "environment": config.app.env
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "standard",
                "filters": ["environment"],
                "stream": "ext://sys.stdout"
            },
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "formatter": "json",
                "filters": ["environment"],
                "filename": config.logging.file_path,
                "maxBytes": config.logging.max_size,
                "backupCount": config.logging.backup_count
            }
        },
        "loggers": {
            "": {
                "handlers": ["console", "file"],
                "level": config.logging.level,
                "propagate": True
            },
            "src": {
                "handlers": ["console", "file"],
                "level": config.logging.level,
                "propagate": False
            }
        }
    }
    
    logging.config.dictConfig(log_config)