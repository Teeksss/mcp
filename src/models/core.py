from pydantic import BaseModel, validator, Field
from typing import Optional, List, Dict
from datetime import datetime
import pytz

class User(BaseModel):
    id: str
    username: str
    email: str
    hashed_password: str
    is_active: bool = True
    is_superuser: bool = False
    api_key: Optional[str] = None
    failed_login_attempts: int = 0
    last_failed_login: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime
    last_login: Optional[datetime] = None
    
    class Config:
        orm_mode = True

class TokenData(BaseModel):
    username: str
    exp: datetime
    type: str  # 'access' or 'refresh'
    user_id: str
    scopes: List[str] = []

class ModelRequest(BaseModel):
    prompt: str
    model_name: str
    use_rag: bool = False
    context_query: Optional[str] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    stream: bool = False
    user_id: str
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    
    @validator('temperature')
    def validate_temperature(cls, v):
        if v is not None and not (0.0 <= v <= 2.0):
            raise ValueError('temperature must be between 0.0 and 2.0')
        return v

class ModelResponse(BaseModel):
    response: str
    model_used: str
    execution_time: float
    token_count: int
    cached: bool = False
    rag_context: Optional[List[Dict[str, str]]] = None
    request_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(pytz.UTC))

class RagDocument(BaseModel):
    id: str
    content: str
    metadata: Dict[str, str]
    embedding: Optional[List[float]]
    created_at: datetime
    updated_at: datetime
    
    class Config:
        orm_mode = True

class MetricsData(BaseModel):
    request_count: int
    error_count: int
    average_response_time: float
    cache_hit_rate: float
    rag_retrieval_time: float
    model_usage: Dict[str, int]
    collected_at: datetime = Field(default_factory=lambda: datetime.now(pytz.UTC))