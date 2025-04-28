from datetime import datetime
import hashlib
from typing import Dict, List, Optional
from dataclasses import dataclass
import json
import boto3
from sqlalchemy import Column, String, DateTime, JSON
from sqlalchemy.ext.declarative import declarative_base
import logging

logger = logging.getLogger(__name__)
Base = declarative_base()

@dataclass
class ModelVersion:
    model_id: str
    version: str
    created_at: datetime
    created_by: str
    metrics: Dict
    parameters: Dict
    artifacts_path: str
    status: str
    hash: str

class ModelVersionRecord(Base):
    __tablename__ = 'model_versions'
    
    model_id = Column(String, primary_key=True)
    version = Column(String, primary_key=True)
    created_at = Column(DateTime, nullable=False)
    created_by = Column(String, nullable=False)
    metrics = Column(JSON)
    parameters = Column(JSON)
    artifacts_path = Column(String)
    status = Column(String)
    hash = Column(String)

class ModelVersionTracker:
    def __init__(self, session, s3_bucket: str):
        self.session = session
        self.s3_client = boto3.client('s3')
        self.s3_bucket = s3_bucket
        
        # Current context
        self.current_user = "Teeksss"
        self.current_time = datetime.utcnow()
    
    async def register_version(
        self,
        model_id: str,
        parameters: Dict,
        artifacts_path: str,
        metrics: Optional[Dict] = None
    ) -> ModelVersion:
        """Register a new model version"""
        try:
            # Generate version hash
            version_hash = self._generate_hash(parameters)
            
            # Create version number
            version = f"{self.current_time.strftime('%Y%m%d')}-{version_hash[:8]}"
            
            # Create version record
            version_record = ModelVersionRecord(
                model_id=model_id,
                version=version,
                created_at=self.current_time,
                created_by=self.current_user,
                metrics=metrics or {},
                parameters=parameters,
                artifacts_path=artifacts_path,
                status="registered",
                hash=version_hash
            )
            
            # Save to database
            self.session.add(version_record)
            await self.session.commit()
            
            logger.info(f"Registered new version {version} for model {model_id}")
            
            return ModelVersion(**version_record.__dict__)
            
        except Exception as e:
            await self.session.rollback()
            logger.error(f"Version registration failed: {e}")
            raise
    
    async def update_version_status(
        self,
        model_id: str,
        version: str,
        status: str,
        metrics: Optional[Dict] = None
    ):
        """Update version status and metrics"""
        try:
            version_record = await self.session.query(ModelVersionRecord).filter_by(
                model_id=model_id,
                version=version
            ).first()
            
            if not version_record:
                raise ValueError(f"Version {version} not found for model {model_id}")
            
            version_record.status = status
            if metrics:
                version_record.metrics.update(metrics)
            
            await self.session.commit()
            
            logger.info(
                f"Updated version {version} status to {status} for model {model_id}"
            )
            
        except Exception as e:
            await self.session.rollback()
            logger.error(f"Version update failed: {e}")
            raise
    
    async def get_version_history(
        self,
        model_id: str,
        limit: int = 10
    ) -> List[ModelVersion]:
        """Get version history for a model"""
        try:
            records = await self.session.query(ModelVersionRecord).filter_by(
                model_id=model_id
            ).order_by(
                ModelVersionRecord.created_at.desc()
            ).limit(limit).all()
            
            return [ModelVersion(**record.__dict__) for record in records]
            
        except Exception as e:
            logger.error(f"Failed to get version history: {e}")
            raise
    
    def _generate_hash(self, parameters: Dict) -> str:
        """Generate deterministic hash for parameters"""
        sorted_params = json.dumps(parameters, sort_keys=True)
        return hashlib.sha256(sorted_params.encode()).hexdigest()
    
    async def archive_version(self, model_id: str, version: str):
        """Archive model version artifacts to S3"""
        try:
            version_record = await self.session.query(ModelVersionRecord).filter_by(
                model_id=model_id,
                version=version
            ).first()
            
            if not version_record:
                raise ValueError(f"Version {version} not found for model {model_id}")
            
            # Upload to S3
            s3_path = f"archives/{model_id}/{version}"
            self.s3_client.upload_file(
                version_record.artifacts_path,
                self.s3_bucket,
                s3_path
            )
            
            # Update record
            version_record.status = "archived"
            version_record.artifacts_path = f"s3://{self.s3_bucket}/{s3_path}"
            
            await self.session.commit()
            
            logger.info(f"Archived version {version} for model {model_id}")
            
        except Exception as e:
            await self.session.rollback()
            logger.error(f"Version archival failed: {e}")
            raise