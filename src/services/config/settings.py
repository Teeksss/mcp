import os

class Settings:
    DEBUG = os.getenv("DEBUG", "false").lower() == "true"
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///mcp.db")
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    SECRET_KEY = os.getenv("SECRET_KEY", "changeme")