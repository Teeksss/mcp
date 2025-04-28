from typing import Dict, Any
import yaml
import os
from dataclasses import dataclass

@dataclass
class DatabaseConfig:
    url: str
    pool_size: int = 20
    max_overflow: int = 100
    echo: bool = False

@dataclass
class SecurityConfig:
    jwt_secret: str
    jwt_algorithm: str = "HS256"
    jwt_expiry: int = 3600
    mfa_enabled: bool = True

@dataclass
class CacheConfig:
    redis_url: str
    default_ttl: int = 3600
    max_memory_mb: int = 8192

@dataclass
class ModelConfig:
    base_path: str
    cache_size_mb: int = 8192
    min_free_memory: float = 0.2
    batch_size_range: tuple = (1, 128)

@dataclass
class AppConfig:
    database: DatabaseConfig
    security: SecurityConfig
    cache: CacheConfig
    model: ModelConfig
    environment: str = "production"
    debug: bool = False

class ConfigurationManager:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self) -> AppConfig:
        """Load configuration from YAML file"""
        config_file = os.path.join(
            self.config_path,
            f"{os.getenv('ENV', 'production')}.yaml"
        )
        
        with open(config_file, 'r') as f:
            config_data = yaml.safe_load(f)
        
        return self._create_config(config_data)
    
    def _create_config(self, data: Dict[str, Any]) -> AppConfig:
        """Create configuration object from dictionary"""
        return AppConfig(
            database=DatabaseConfig(**data["database"]),
            security=SecurityConfig(**data["security"]),
            cache=CacheConfig(**data["cache"]),
            model=ModelConfig(**data["model"]),
            environment=data.get("environment", "production"),
            debug=data.get("debug", False)
        )