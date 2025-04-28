from typing import Any, Dict, Optional
from pydantic import BaseSettings, PostgresDsn, RedisDsn, validator
import os
from pathlib import Path

class Settings(BaseSettings):
    # Application
    APP_NAME: str = "MCP_Server"
    ENV: str = "development"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"
    
    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 4
    
    # Database
    DATABASE_URL: PostgresDsn
    DB_POOL_SIZE: int = 20
    DB_MAX_OVERFLOW: int = 100
    
    # Redis
    REDIS_URL: RedisDsn
    REDIS_PASSWORD: str
    
    # Security
    JWT_SECRET: str
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRY: int = 3600
    MFA_ENABLED: bool = True
    
    # Model
    MODEL_BASE_PATH: Path
    MODEL_CACHE_SIZE: int = 8192
    MIN_FREE_MEMORY: float = 0.2
    BATCH_SIZE_MIN: int = 1
    BATCH_SIZE_MAX: int = 128
    
    # Monitoring
    PROMETHEUS_ENABLED: bool = True
    METRICS_PORT: int = 9090
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()