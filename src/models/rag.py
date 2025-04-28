from sqlalchemy import Column, Integer, String, JSON
from src.models.core import Base

class RagDocument(Base):
    __tablename__ = "rag_documents"
    id = Column(Integer, primary_key=True)
    content = Column(String)
    metadata = Column(JSON)