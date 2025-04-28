from sqlalchemy import Column, Integer, String, Float, JSON, DateTime, ForeignKey
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from src.database.session import Base

class Model(Base):
    __tablename__ = "models"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    version = Column(String)
    path = Column(String)
    config = Column(JSON)
    metrics = Column(JSON)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

class ModelVersion(Base):
    __tablename__ = "model_versions"
    
    id = Column(Integer, primary_key=True, index=True)
    model_id = Column(Integer, ForeignKey("models.id"))
    version = Column(String)
    path = Column(String)
    config = Column(JSON)
    metrics = Column(JSON)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    model = relationship("Model", back_populates="versions")