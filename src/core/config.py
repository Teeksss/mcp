import os

class Config:
    VECTOR_BACKEND = os.getenv("VECTOR_BACKEND", "chromadb")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    # Diğer genel ayarlar...