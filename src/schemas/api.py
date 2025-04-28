from pydantic import BaseModel, Field
from typing import Dict, List, Optional
from datetime import datetime

class ModelConfig(BaseModel):
    name: str
    version: str
    framework: str
    parameters: Dict
    artifacts_path: Optional[str]
    description: Optional[str]

class DataConfig(BaseModel):
    source: str
    format: str
    schema: Dict
    validation_rules: Optional[Dict]

class DeploymentConfig(BaseModel):
    environment: str
    resources: Dict
    scaling_config: Dict
    monitoring_config: Dict
    alerts_config: Optional[Dict]

class ModelTrainingRequest(BaseModel):
    model_config: ModelConfig
    training_data: DataConfig
    deployment_config: DeploymentConfig
    
    class Config:
        schema_extra = {
            "example": {
                "model_config": {
                    "name": "recommendation_engine",
                    "version": "1.0.0",
                    "framework": "pytorch",
                    "parameters": {
                        "learning_rate": 0.001,
                        "batch_size": 32,
                        "epochs": 100
                    }
                },
                "training_data": {
                    "source": "s3://training-data/recommendations/",
                    "format": "parquet",
                    "schema": {
                        "user_id": "string",
                        "item_id": "string",
                        "rating": "float"
                    }
                },
                "deployment_config": {
                    "environment": "production",
                    "resources": {
                        "cpu": 4,
                        "memory": "16Gi",
                        "gpu": 1
                    },
                    "scaling_config": {
                        "min_replicas": 2,
                        "max_replicas": 10,
                        "target_cpu_utilization": 70
                    },
                    "monitoring_config": {
                        "metrics_interval": 60,
                        "log_level": "INFO"
                    }
                }
            }
        }

class ModelDeploymentRequest(BaseModel):
    environment: str
    resources: Dict
    scaling_config: Dict
    monitoring_config: Dict
    alerts_config: Optional[Dict]

class ModelInferenceRequest(BaseModel):
    data: Dict
    parameters: Optional[Dict]
    request_id: Optional[str]
    
    class Config:
        schema_extra = {
            "example": {
                "data": {
                    "user_id": "user_123",
                    "context": {
                        "time": "2025-04-28T07:58:44",
                        "location": "US"
                    }
                },
                "parameters": {
                    "num_recommendations": 5,
                    "diversity_weight": 0.3
                }
            }
        }

class ExperimentRequest(BaseModel):
    name: str
    config: Dict
    tags: Optional[Dict]
    
    class Config:
        schema_extra = {
            "example": {
                "name": "hyperparameter_tuning_exp_1",
                "config": {
                    "objective": "minimize",
                    "metric": "validation_loss",
                    "parameters": {
                        "learning_rate": {
                            "distribution": "log_uniform",
                            "min": 1e-4,
                            "max": 1e-2
                        }
                    }
                },
                "tags": {
                    "model_type": "recommendation",
                    "experiment_type": "hyperparameter_tuning"
                }
            }
        }

class FeatureExtractionRequest(BaseModel):
    config: Dict
    data: Optional[Dict]
    
    class Config:
        schema_extra = {
            "example": {
                "config": {
                    "feature_groups": ["user_features", "item_features"],
                    "transformations": {
                        "user_features": [
                            {"type": "normalize", "columns": ["age", "income"]},
                            {"type": "one_hot", "columns": ["country"]}
                        ]
                    }
                },
                "data": {
                    "source": "s3://raw-data/users/",
                    "format": "parquet"
                }
            }
        }

class OperationResponse(BaseModel):
    operation_id: str
    status: str
    timestamp: datetime
    details: Optional[Dict]

class ErrorResponse(BaseModel):
    error: str
    error_type: str
    timestamp: datetime
    details: Optional[Dict]