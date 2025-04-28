from sqlalchemy import Column, Integer, String, JSON, DateTime, ForeignKey
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from src.database.session import Base

class Query(Base):
    __tablename__ = "queries"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    model_id = Column(Integer, ForeignKey("models.id"))
    input_text = Column(String)
    output_text = Column(String)
    metadata = Column(JSON)
    metrics = Column(JSON)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    user = relationship("User", back_populates="queries")
    model = relationship("Model", back_populates="queries")