from datetime import datetime
from typing import Dict, List, Optional, Any, Union
import asyncio
import json
from dataclasses import dataclass
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
import numpy as np
from prometheus_client import Gauge, Counter
import logging

logger = logging.getLogger(__name__)

@dataclass
class ExperimentConfig:
    mlflow_tracking_uri: str
    artifact_store_path: str
    metrics_update_interval: int = 60
    retention_days: int = 90
    max_artifacts_size: int = 1024 * 1024 * 1024  # 1GB

class ExperimentTracker:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.client = MlflowClient(config.mlflow_tracking_uri)
        self.metrics = self._setup_metrics()
        self.active_runs = {}
        
        # Current context
        self.timestamp = datetime.utcnow()
        self.current_user = "Teeksss"
        
        # Initialize MLflow
        mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    
    def _setup_metrics(self) -> Dict:
        """Initialize experiment metrics"""
        return {
            "active_experiments": Gauge(
                "active_experiments",
                "Number of active experiments",
                ["model_type"]
            ),
            "experiment_metrics": Gauge(
                "experiment_metrics",
                "Experiment performance metrics",
                ["experiment_id", "metric_name"]
            ),
            "artifact_size": Gauge(
                "experiment_artifact_size_bytes",
                "Total size of experiment artifacts",
                ["experiment_id"]
            )
        }
    
    async def create_experiment(
        self,
        name: str,
        model_type: str,
        config: Dict,
        tags: Optional[Dict] = None
    ) -> str:
        """Create new experiment"""
        try:
            # Create MLflow experiment
            experiment_id = mlflow.create_experiment(
                name,
                artifact_location=f"{self.config.artifact_store_path}/{name}"
            )
            
            # Set experiment tags
            base_tags = {
                "model_type": model_type,
                "created_by": self.current_user,
                "created_at": self.timestamp.isoformat(),
                "status": "active"
            }
            
            if tags:
                base_tags.update(tags)
            
            for key, value in base_tags.items():
                self.client.set_experiment_tag(experiment_id, key, value)
            
            # Store configuration
            with mlflow.start_run(experiment_id=experiment_id):
                mlflow.log_params(config)
            
            # Update metrics
            self.metrics["active_experiments"].labels(
                model_type=model_type
            ).inc()
            
            logger.info(f"Created experiment {name} with ID {experiment_id}")
            return experiment_id
            
        except Exception as e:
            logger.error(f"Experiment creation failed: {e}")
            raise
    
    async def start_run(
        self,
        experiment_id: str,
        run_name: Optional[str] = None,
        params: Optional[Dict] = None
    ) -> str:
        """Start new experiment run"""
        try:
            # Start MLflow run
            run = mlflow.start_run(
                experiment_id=experiment_id,
                run_name=run_name or f"run_{self.timestamp.strftime('%Y%m%d_%H%M%S')}"
            )
            
            run_id = run.info.run_id
            self.active_runs[run_id] = run
            
            # Log parameters
            if params:
                mlflow.log_params(params)
            
            # Log basic run info
            mlflow.set_tags({
                "started_by": self.current_user,
                "started_at": self.timestamp.isoformat(),
                "status": "running"
            })
            
            logger.info(f"Started run {run_id} for experiment {experiment_id}")
            return run_id
            
        except Exception as e:
            logger.error(f"Run start failed: {e}")
            raise
    
    async def log_metrics(
        self,
        run_id: str,
        metrics: Dict[str, float],
        step: Optional[int] = None
    ):
        """Log metrics for experiment run"""
        try:
            if run_id not in self.active_runs:
                raise ValueError(f"Run {run_id} not found or not active")
            
            # Log to MLflow
            mlflow.log_metrics(metrics, step=step)
            
            # Update Prometheus metrics
            for metric_name, value in metrics.items():
                self.metrics["experiment_metrics"].labels(
                    experiment_id=self.active_runs[run_id].info.experiment_id,
                    metric_name=metric_name
                ).set(value)
            
        except Exception as e:
            logger.error(f"Metric logging failed: {e}")
            raise
    
    async def log_artifacts(
        self,
        run_id: str,
        artifacts: Dict[str, Union[str, bytes, np.ndarray]]
    ):
        """Log artifacts for experiment run"""
        try:
            if run_id not in self.active_runs:
                raise ValueError(f"Run {run_id} not found or not active")
            
            total_size = 0
            
            for name, artifact in artifacts.items():
                # Handle different artifact types
                if isinstance(artifact, str):
                    # Text artifact
                    mlflow.log_text(artifact, f"{name}.txt")
                    total_size += len(artifact.encode())
                    
                elif isinstance(artifact, bytes):
                    # Binary artifact
                    mlflow.log_binary(artifact, f"{name}.bin")
                    total_size += len(artifact)
                    
                elif isinstance(artifact, np.ndarray):
                    # NumPy array
                    np.save(f"/tmp/{name}.npy", artifact)
                    mlflow.log_artifact(f"/tmp/{name}.npy")
                    total_size += artifact.nbytes
                
                else:
                    logger.warning(f"Unsupported artifact type for {name}")
                    continue
            
            # Check size limit
            if total_size > self.config.max_artifacts_size:
                logger.warning(f"Artifact size ({total_size} bytes) exceeds limit")
            
            # Update size metric
            self.metrics["artifact_size"].labels(
                experiment_id=self.active_runs[run_id].info.experiment_id
            ).set(total_size)
            
        except Exception as e:
            logger.error(f"Artifact logging failed: {e}")
            raise
    
    async def end_run(
        self,
        run_id: str,
        status: str = "FINISHED",
        end_time: Optional[datetime] = None
    ):
        """End experiment run"""
        try:
            if run_id not in self.active_runs:
                raise ValueError(f"Run {run_id} not found or not active")
            
            # Set end time
            end_time = end_time or self.timestamp
            
            # Log final status
            mlflow.set_tag("status", status)
            mlflow.set_tag("ended_at", end_time.isoformat())
            mlflow.set_tag("duration_seconds", 
                          (end_time - datetime.fromisoformat(
                              mlflow.get_tag("started_at")
                          )).total_seconds())
            
            # End run
            mlflow.end_run(status=status)
            del self.active_runs[run_id]
            
            logger.info(f"Ended run {run_id} with status {status}")
            
        except Exception as e:
            logger.error(f"Run end failed: {e}")
            raise
    
    async def get_run_history(
        self,
        experiment_id: str,
        metric_name: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> pd.DataFrame:
        """Get historical metrics for experiment"""
        try:
            # Get all runs for experiment
            runs = self.client.search_runs(
                experiment_ids=[experiment_id],
                filter_string=""
            )
            
            # Collect metrics
            history = []
            for run in runs:
                metrics = self.client.get_metric_history(
                    run.info.run_id,
                    metric_name
                )
                
                for metric in metrics:
                    timestamp = datetime.fromtimestamp(metric.timestamp / 1000.0)
                    
                    if start_time and timestamp < start_time:
                        continue
                    if end_time and timestamp > end_time:
                        continue
                    
                    history.append({
                        "timestamp": timestamp,
                        "value": metric.value,
                        "step": metric.step,
                        "run_id": run.info.run_id
                    })
            
            return pd.DataFrame(history)
            
        except Exception as e:
            logger.error(f"History retrieval failed: {e}")
            raise
    
    async def compare_runs(
        self,
        experiment_id: str,
        metric_names: List[str],
        n_runs: int = 5
    ) -> Dict:
        """Compare recent runs for experiment"""
        try:
            # Get recent runs
            runs = self.client.search_runs(
                experiment_ids=[experiment_id],
                order_by=["start_time DESC"],
                max_results=n_runs
            )
            
            comparison = {}
            for metric_name in metric_names:
                metric_values = []
                for run in runs:
                    value = run.data.metrics.get(metric_name)
                    if value is not None:
                        metric_values.append({
                            "run_id": run.info.run_id,
                            "value": value,
                            "start_time": datetime.fromtimestamp(
                                run.info.start_time / 1000.0
                            ).isoformat()
                        })
                
                comparison[metric_name] = metric_values
            
            return comparison
            
        except Exception as e:
            logger.error(f"Run comparison failed: {e}")
            raise