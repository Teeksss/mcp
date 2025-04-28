from pydantic import BaseModel
from typing import Optional, List, Dict

class User(BaseModel):
    username: str
    hashed_password: str
    is_active: bool = True
    api_key: Optional[str] = None

class TokenData(BaseModel):
    username: Optional[str] = None
    exp: Optional[int] = None

class CacheConfig(BaseModel):
    ttl: int
    max_size: int

class ModelConfig(BaseModel):
    name: str
    fallback_models: List[str]
    timeout: int
    max_retries: int
    cache_config: CacheConfig