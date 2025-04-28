from pydantic import BaseSettings

class Settings(BaseSettings):
    JWT_SECRET_KEY: str = "your-secret-key"  # In production, use environment variable
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REDIS_URL: str = "redis://localhost:6379"
    SENTRY_DSN: str = ""  # Add your Sentry DSN here
    MODEL_TIMEOUT: int = 30
    MAX_RETRIES: int = 3
    
settings = Settings()