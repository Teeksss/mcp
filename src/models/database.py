from sqlalchemy import (
    Column, Integer, String, Float, DateTime, JSON, 
    ForeignKey, Boolean, Enum, Table
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
import enum

Base = declarative_base()

# Enums for status types
class ModelStatus(enum.Enum):
    DRAFT = "draft"
    TRAINING = "training"
    DEPLOYED = "deployed"
    ARCHIVED = "archived"
    FAILED = "failed"

class ExperimentStatus(enum.Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"

# Association tables for many-to-many relationships
model_tags = Table(
    'model_tags',
    Base.metadata,
    Column('model_id', String, ForeignKey('models.id')),
    Column('tag_id', Integer, ForeignKey('tags.id'))
)

class Model(Base):
    __tablename__ = 'models'
    
    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    version = Column(String, nullable=False)
    description = Column(String)
    framework = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    created_by = Column(String, nullable=False)
    status = Column(Enum(ModelStatus), default=ModelStatus.DRAFT)
    metrics = Column(JSON)
    parameters = Column(JSON)
    artifacts_path = Column(String)
    
    # Relationships
    deployments = relationship("ModelDeployment", back_populates="model")
    experiments = relationship("Experiment", back_populates="model")
    tags = relationship("Tag", secondary=model_tags, back_populates="models")

class ModelDeployment(Base):
    __tablename__ = 'model_deployments'
    
    id = Column(String, primary_key=True)
    model_id = Column(String, ForeignKey('models.id'))
    environment = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    created_by = Column(String, nullable=False)
    status = Column(String, nullable=False)
    config = Column(JSON)
    metrics = Column(JSON)
    endpoint = Column(String)
    
    # Relationships
    model = relationship("Model", back_populates="deployments")
    logs = relationship("DeploymentLog", back_populates="deployment")

class DeploymentLog(Base):
    __tablename__ = 'deployment_logs'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    deployment_id = Column(String, ForeignKey('model_deployments.id'))
    timestamp = Column(DateTime, default=datetime.utcnow)
    level = Column(String, nullable=False)
    message = Column(String, nullable=False)
    details = Column(JSON)
    
    # Relationships
    deployment = relationship("ModelDeployment", back_populates="logs")

class Experiment(Base):
    __tablename__ = 'experiments'
    
    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    model_id = Column(String, ForeignKey('models.id'))
    created_at = Column(DateTime, default=datetime.utcnow)
    created_by = Column(String, nullable=False)
    status = Column(Enum(ExperimentStatus), default=ExperimentStatus.RUNNING)
    config = Column(JSON)
    results = Column(JSON)
    
    # Relationships
    model = relationship("Model", back_populates="experiments")
    runs = relationship("ExperimentRun", back_populates="experiment")

class ExperimentRun(Base):
    __tablename__ = 'experiment_runs'
    
    id = Column(String, primary_key=True)
    experiment_id = Column(String, ForeignKey('experiments.id'))
    start_time = Column(DateTime, default=datetime.utcnow)
    end_time = Column(DateTime)
    status = Column(String, nullable=False)
    parameters = Column(JSON)
    metrics = Column(JSON)
    artifacts_path = Column(String)
    
    # Relationships
    experiment = relationship("Experiment", back_populates="runs")

class FeatureStore(Base):
    __tablename__ = 'feature_stores'
    
    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    created_by = Column(String, nullable=False)
    config = Column(JSON)
    schema = Column(JSON)
    statistics = Column(JSON)
    last_updated = Column(DateTime)
    
    # Relationships
    features = relationship("Feature", back_populates="store")

class Feature(Base):
    __tablename__ = 'features'
    
    id = Column(String, primary_key=True)
    store_id = Column(String, ForeignKey('feature_stores.id'))
    name = Column(String, nullable=False)
    description = Column(String)
    dtype = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    created_by = Column(String, nullable=False)
    transformation = Column(JSON)
    statistics = Column(JSON)
    
    # Relationships
    store = relationship("FeatureStore", back_populates="features")

class Tag(Base):
    __tablename__ = 'tags'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, nullable=False, unique=True)
    
    # Relationships
    models = relationship("Model", secondary=model_tags, back_populates="tags")

class AuditLog(Base):
    __tablename__ = 'audit_logs'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    user = Column(String, nullable=False)
    action = Column(String, nullable=False)
    resource_type = Column(String, nullable=False)
    resource_id = Column(String, nullable=False)
    details = Column(JSON)