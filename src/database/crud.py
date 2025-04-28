from typing import Dict, List, Optional, Any, Union
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc
from datetime import datetime

from src.models.database import (
    Model, ModelDeployment, Experiment, ExperimentRun,
    FeatureStore, Feature, Tag, AuditLog
)

class DatabaseManager:
    def __init__(self, session: Session):
        self.session = session
        self.current_user = "Teeksss"
        self.timestamp = datetime.utcnow()
    
    async def create_model(
        self,
        name: str,
        version: str,
        framework: str,
        description: Optional[str] = None,
        parameters: Optional[Dict] = None,
        artifacts_path: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> Model:
        """Create new model"""
        try:
            # Create model instance
            model = Model(
                id=f"{name}_{version}_{self.timestamp.strftime('%Y%m%d_%H%M%S')}",
                name=name,
                version=version,
                framework=framework,
                description=description,
                parameters=parameters or {},
                artifacts_path=artifacts_path,
                created_by=self.current_user
            )
            
            # Add tags
            if tags:
                for tag_name in tags:
                    tag = self.session.query(Tag).filter_by(
                        name=tag_name
                    ).first()
                    if not tag:
                        tag = Tag(name=tag_name)
                        self.session.add(tag)
                    model.tags.append(tag)
            
            self.session.add(model)
            await self.session.commit()
            
            # Create audit log
            await self.create_audit_log(
                action="create_model",
                resource_type="model",
                resource_id=model.id,
                details={"name": name, "version": version}
            )
            
            return model
            
        except Exception:
            await self.session.rollback()
            raise
    
    async def get_model(
        self,
        model_id: str
    ) -> Optional[Model]:
        """Get model by ID"""
        return self.session.query(Model).filter_by(id=model_id).first()
    
    async def update_model_status(
        self,
        model_id: str,
        status: str,
        metrics: Optional[Dict] = None
    ) -> Model:
        """Update model status and metrics"""
        try:
            model = await self.get_model(model_id)
            if not model:
                raise ValueError(f"Model {model_id} not found")
            
            model.status = status
            if metrics:
                model.metrics = metrics
            model.updated_at = self.timestamp
            
            await self.session.commit()
            
            # Create audit log
            await self.create_audit_log(
                action="update_model",
                resource_type="model",
                resource_id=model_id,
                details={"status": status, "metrics": metrics}
            )
            
            return model
            
        except Exception:
            await self.session.rollback()
            raise
    
    async def create_deployment(
        self,
        model_id: str,
        environment: str,
        config: Dict,
        endpoint: Optional[str] = None
    ) -> ModelDeployment:
        """Create model deployment"""
        try:
            deployment = ModelDeployment(
                id=f"deploy_{model_id}_{self.timestamp.strftime('%Y%m%d_%H%M%S')}",
                model_id=model_id,
                environment=environment,
                config=config,
                endpoint=endpoint,
                status="pending",
                created_by=self.current_user
            )
            
            self.session.add(deployment)
            await self.session.commit()
            
            # Create audit log
            await self.create_audit_log(
                action="create_deployment",
                resource_type="deployment",
                resource_id=deployment.id,
                details={"model_id": model_id, "environment": environment}
            )
            
            return deployment
            
        except Exception:
            await self.session.rollback()
            raise
    
    async def create_experiment(
        self,
        name: str,
        model_id: str,
        config: Dict
    ) -> Experiment:
        """Create new experiment"""
        try:
            experiment = Experiment(
                id=f"exp_{self.timestamp.strftime('%Y%m%d_%H%M%S')}",
                name=name,
                model_id=model_id,
                config=config,
                created_by=self.current_user
            )
            
            self.session.add(experiment)
            await self.session.commit()
            
            # Create audit log
            await self.create_audit_log(
                action="create_experiment",
                resource_type="experiment",
                resource_id=experiment.id,
                details={"name": name, "model_id": model_id}
            )
            
            return experiment
            
        except Exception:
            await self.session.rollback()
            raise
    
    async def create_feature_store(
        self,
        name: str,
        config: Dict,
        schema: Dict
    ) -> FeatureStore:
        """Create new feature store"""
        try:
            store = FeatureStore(
                id=f"store_{self.timestamp.strftime('%Y%m%d_%H%M%S')}",
                name=name,
                config=config,
                schema=schema,
                created_by=self.current_user
            )
            
            self.session.add(store)
            await self.session.commit()
            
            # Create audit log
            await self.create_audit_log(
                action="create_feature_store",
                resource_type="feature_store",
                resource_id=store.id,
                details={"name": name}
            )
            
            return store
            
        except Exception:
            await self.session.rollback()
            raise
    
    async def create_audit_log(
        self,
        action: str,
        resource_type: str,
        resource_id: str,
        details: Optional[Dict] = None
    ) -> AuditLog:
        """Create audit log entry"""
        try:
            log = AuditLog(
                user=self.current_user,
                action=action,
                resource_type=resource_type,
                resource_id=resource_id,
                details=details
            )
            
            self.session.add(log)
            await self.session.commit()
            
            return log
            
        except Exception:
            await self.session.rollback()
            raise