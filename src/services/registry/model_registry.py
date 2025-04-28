from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import asyncio
import json
import boto3
from botocore.exceptions import ClientError
from sqlalchemy import create_engine, Column, String, DateTime, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from prometheus_client import Counter, Gauge
import logging

logger = logging.getLogger(__name__)
Base = declarative_base()

@dataclass
class RegistryConfig:
    db_url: str
    s3_bucket: str
    aws_region: str
    registry_path: str = "models"
    max_versions: int = 10
    cleanup_interval: int = 3600

class ModelMetadata(Base):
    __tablename__ = 'model_metadata'
    
    model_id = Column(String, primary_key=True)
    version = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    description = Column(String)
    framework = Column(String)
    created_at = Column(DateTime, nullable=False)
    created_by = Column(String, nullable=False)
    artifacts_path = Column(String)
    metrics = Column(JSON)
    parameters = Column(JSON)
    tags = Column(JSON)
    status = Column(String)

class ModelRegistry:
    def __init__(self, config: RegistryConfig):
        self.config = config
        self.metrics = self._setup_metrics()
        
        # Initialize database
        self.engine = create_engine(config.db_url)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        
        # Initialize S3 client
        self.s3 = boto3.client('s3', region_name=config.aws_region)
        
        # Current context
        self.timestamp = datetime.utcnow()
        self.current_user = "Teeksss"
        
        # Start cleanup task
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())
    
    def _setup_metrics(self) -> Dict:
        """Initialize registry metrics"""
        return {
            "model_versions": Counter(
                "registry_model_versions_total",
                "Total model versions registered",
                ["model_name"]
            ),
            "storage_usage": Gauge(
                "registry_storage_bytes",
                "Storage usage in bytes",
                ["model_name"]
            ),
            "active_models": Gauge(
                "registry_active_models",
                "Number of active models",
                ["status"]
            )
        }
    
    async def register_model(
        self,
        name: str,
        version: str,
        artifacts_path: str,
        description: Optional[str] = None,
        framework: Optional[str] = None,
        metrics: Optional[Dict] = None,
        parameters: Optional[Dict] = None,
        tags: Optional[Dict] = None
    ) -> str:
        """Register new model version"""
        try:
            session = self.Session()
            
            # Generate model ID
            model_id = f"{name}_{self.timestamp.strftime('%Y%m%d_%H%M%S')}"
            
            # Upload artifacts to S3
            s3_path = await self._upload_artifacts(
                model_id,
                version,
                artifacts_path
            )
            
            # Create metadata record
            model = ModelMetadata(
                model_id=model_id,
                version=version,
                name=name,
                description=description,
                framework=framework,
                created_at=self.timestamp,
                created_by=self.current_user,
                artifacts_path=s3_path,
                metrics=metrics or {},
                parameters=parameters or {},
                tags=tags or {},
                status="registered"
            )
            
            session.add(model)
            await session.commit()
            
            # Update metrics
            self.metrics["model_versions"].labels(
                model_name=name
            ).inc()
            
            self.metrics["active_models"].labels(
                status="registered"
            ).inc()
            
            logger.info(f"Registered model: {model_id} version {version}")
            return model_id
            
        except Exception as e:
            await session.rollback()
            logger.error(f"Model registration failed: {e}")
            raise
        
        finally:
            session.close()
    
    async def update_model_status(
        self,
        model_id: str,
        version: str,
        status: str,
        metrics: Optional[Dict] = None
    ):
        """Update model status and metrics"""
        try:
            session = self.Session()
            
            model = await session.query(ModelMetadata).filter_by(
                model_id=model_id,
                version=version
            ).first()
            
            if not model:
                raise ValueError(f"Model {model_id} version {version} not found")
            
            # Update status and metrics
            old_status = model.status
            model.status = status
            if metrics:
                model.metrics.update(metrics)
            
            await session.commit()
            
            # Update metrics
            self.metrics["active_models"].labels(
                status=old_status
            ).dec()
            self.metrics["active_models"].labels(
                status=status
            ).inc()
            
            logger.info(
                f"Updated model {model_id} version {version} status to {status}"
            )
            
        except Exception as e:
            await session.rollback()
            logger.error(f"Status update failed: {e}")
            raise
        
        finally:
            session.close()
    
    async def get_model_info(
        self,
        model_id: str,
        version: Optional[str] = None
    ) -> Union[ModelMetadata, List[ModelMetadata]]:
        """Get model information"""
        try:
            session = self.Session()
            
            if version:
                # Get specific version
                model = await session.query(ModelMetadata).filter_by(
                    model_id=model_id,
                    version=version
                ).first()
                
                if not model:
                    raise ValueError(
                        f"Model {model_id} version {version} not found"
                    )
                
                return model
            else:
                # Get all versions
                models = await session.query(ModelMetadata).filter_by(
                    model_id=model_id
                ).order_by(
                    ModelMetadata.created_at.desc()
                ).all()
                
                if not models:
                    raise ValueError(f"Model {model_id} not found")
                
                return models
            
        except Exception as e:
            logger.error(f"Model info retrieval failed: {e}")
            raise
        
        finally:
            session.close()
    
    async def download_artifacts(
        self,
        model_id: str,
        version: str,
        destination_path: str
    ):
        """Download model artifacts"""
        try:
            # Get model info
            model = await self.get_model_info(model_id, version)
            
            # Download from S3
            self.s3.download_file(
                self.config.s3_bucket,
                model.artifacts_path,
                destination_path
            )
            
            logger.info(
                f"Downloaded artifacts for model {model_id} version {version}"
            )
            
        except Exception as e:
            logger.error(f"Artifact download failed: {e}")
            raise
    
    async def _upload_artifacts(
        self,
        model_id: str,
        version: str,
        artifacts_path: str
    ) -> str:
        """Upload model artifacts to S3"""
        try:
            s3_path = f"{self.config.registry_path}/{model_id}/{version}"
            
            # Upload to S3
            self.s3.upload_file(
                artifacts_path,
                self.config.s3_bucket,
                s3_path
            )
            
            # Update storage metrics
            file_size = await self._get_file_size(artifacts_path)
            self.metrics["storage_usage"].labels(
                model_name=model_id
            ).set(file_size)
            
            return s3_path
            
        except Exception as e:
            logger.error(f"Artifact upload failed: {e}")
            raise
    
    async def _get_file_size(self, file_path: str) -> int:
        """Get file size in bytes"""
        try:
            return os.path.getsize(file_path)
        except Exception:
            return 0
    
    async def _cleanup_loop(self):
        """Periodic cleanup of old model versions"""
        while True:
            try:
                session = self.Session()
                
                # Get all models grouped by name
                models = await session.query(ModelMetadata).all()
                model_groups = {}
                
                for model in models:
                    if model.name not in model_groups:
                        model_groups[model.name] = []
                    model_groups[model.name].append(model)
                
                # Check each group
                for name, versions in model_groups.items():
                    if len(versions) > self.config.max_versions:
                        # Sort by creation time
                        versions.sort(
                            key=lambda x: x.created_at,
                            reverse=True
                        )
                        
                        # Remove old versions
                        for old_version in versions[self.config.max_versions:]:
                            # Delete from S3
                            try:
                                self.s3.delete_object(
                                    Bucket=self.config.s3_bucket,
                                    Key=old_version.artifacts_path
                                )
                            except ClientError:
                                pass
                            
                            # Delete from database
                            await session.delete(old_version)
                
                await session.commit()
                
            except Exception as e:
                logger.error(f"Cleanup failed: {e}")
                if session:
                    await session.rollback()
            
            finally:
                if session:
                    session.close()
            
            await asyncio.sleep(self.config.cleanup_interval)