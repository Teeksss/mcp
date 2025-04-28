from datetime import datetime
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
import asyncio
import torch.distributed as dist
import horovod.torch as hvd
from prometheus_client import Counter, Gauge, Histogram
import kubernetes
from kubernetes import client, config
import logging
import ray
from ray import train
from ray.train import Trainer

logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    num_workers: int = 4
    gpu_per_worker: int = 1
    cpu_per_worker: int = 4
    memory_per_worker: str = "16Gi"
    batch_size_per_worker: int = 32
    max_epochs: int = 100
    checkpoint_interval: int = 10

class DistributedTrainer:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.metrics = self._setup_metrics()
        self.k8s_client = self._setup_kubernetes()
        
        # Initialize Ray
        ray.init(address="auto")
        
        # Current context
        self.timestamp = datetime.utcnow()
        self.current_user = "Teeksss"
    
    def _setup_metrics(self) -> Dict:
        """Initialize training metrics"""
        return {
            "training_time": Histogram(
                "training_time_seconds",
                "Total training time",
                ["model_name"]
            ),
            "gpu_utilization": Gauge(
                "gpu_utilization_percent",
                "GPU utilization",
                ["worker_id", "gpu_id"]
            ),
            "worker_status": Gauge(
                "worker_status",
                "Worker status (0=offline, 1=training, 2=error)",
                ["worker_id"]
            ),
            "training_loss": Gauge(
                "training_loss",
                "Training loss",
                ["model_name", "epoch"]
            )
        }
    
    def _setup_kubernetes(self) -> kubernetes.client.CoreV1Api:
        """Initialize Kubernetes client"""
        try:
            config.load_incluster_config()
        except config.ConfigException:
            config.load_kube_config()
        return client.CoreV1Api()
    
    async def start_distributed_training(
        self,
        model_config: Dict,
        dataset_config: Dict,
        training_name: str
    ) -> str:
        """Start distributed training job"""
        try:
            # Create training spec
            training_spec = self._create_training_spec(
                model_config,
                dataset_config,
                training_name
            )
            
            # Initialize trainer
            trainer = Trainer(
                backend="torch",
                num_workers=self.config.num_workers,
                use_gpu=True,
                resources_per_worker={
                    "CPU": self.config.cpu_per_worker,
                    "GPU": self.config.gpu_per_worker
                }
            )
            
            # Start training
            result_future = await self._run_training(
                trainer,
                training_spec
            )
            
            logger.info(f"Started distributed training: {training_name}")
            return result_future
            
        except Exception as e:
            logger.error(f"Training start failed: {e}")
            raise
    
    async def _run_training(
        self,
        trainer: Trainer,
        training_spec: Dict
    ):
        """Execute distributed training"""
        @ray.remote(num_gpus=self.config.gpu_per_worker)
        class TrainingWorker:
            def __init__(self, worker_id: int):
                self.worker_id = worker_id
                self.device = torch.device(
                    f"cuda:{worker_id % torch.cuda.device_count()}"
                )
                
                # Initialize Horovod
                hvd.init()
                torch.cuda.set_device(hvd.local_rank())
            
            async def train(self, spec: Dict):
                try:
                    # Setup model
                    model = self._setup_model(spec["model_config"])
                    model = model.to(self.device)
                    
                    # Setup optimizer with Horovod
                    optimizer = torch.optim.Adam(
                        model.parameters(),
                        lr=spec["learning_rate"]
                    )
                    optimizer = hvd.DistributedOptimizer(
                        optimizer,
                        named_parameters=model.named_parameters()
                    )
                    
                    # Setup data loader
                    train_loader = self._setup_dataloader(
                        spec["dataset_config"]
                    )
                    
                    # Training loop
                    for epoch in range(spec["num_epochs"]):
                        model.train()
                        epoch_loss = 0.0
                        
                        for batch_idx, (data, target) in enumerate(train_loader):
                            data, target = data.to(self.device), target.to(self.device)
                            optimizer.zero_grad()
                            
                            # Forward pass
                            output = model(data)
                            loss = self._compute_loss(output, target)
                            
                            # Backward pass
                            loss.backward()
                            optimizer.step()
                            
                            epoch_loss += loss.item()
                            
                            # Report metrics
                            if hvd.rank() == 0 and batch_idx % 10 == 0:
                                train.report(
                                    {"loss": loss.item(), "epoch": epoch}
                                )
                        
                        # Checkpoint saving
                        if hvd.rank() == 0 and epoch % spec["checkpoint_interval"] == 0:
                            self._save_checkpoint(model, optimizer, epoch)
                    
                    return {"status": "completed", "final_loss": epoch_loss}
                    
                except Exception as e:
                    logger.error(f"Worker {self.worker_id} failed: {e}")
                    return {"status": "failed", "error": str(e)}
            
            def _setup_model(self, model_config: Dict):
                """Setup model for distributed training"""
                # Implementation depends on model architecture
                pass
            
            def _setup_dataloader(self, dataset_config: Dict):
                """Setup distributed data loader"""
                # Implementation depends on dataset
                pass
            
            def _compute_loss(self, output, target):
                """Compute loss function"""
                # Implementation depends on task
                pass
            
            def _save_checkpoint(self, model, optimizer, epoch: int):
                """Save training checkpoint"""
                checkpoint = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict()
                }
                torch.save(
                    checkpoint,
                    f"checkpoints/epoch_{epoch}.pt"
                )
        
        # Create workers
        workers = [
            TrainingWorker.remote(i)
            for i in range(self.config.num_workers)
        ]
        
        # Start training on all workers
        futures = [
            worker.train.remote(training_spec)
            for worker in workers
        ]
        
        return ray.get(futures)
    
    def _create_training_spec(
        self,
        model_config: Dict,
        dataset_config: Dict,
        training_name: str
    ) -> Dict:
        """Create training specification"""
        return {
            "name": training_name,
            "model_config": model_config,
            "dataset_config": dataset_config,
            "num_epochs": self.config.max_epochs,
            "checkpoint_interval": self.config.checkpoint_interval,
            "batch_size": self.config.batch_size_per_worker,
            "learning_rate": 0.001,  # Could be part of model_config
            "timestamp": self.timestamp.isoformat(),
            "user": self.current_user
        }
    
    async def monitor_training(self, training_name: str):
        """Monitor training progress"""
        try:
            while True:
                # Get worker status
                workers = self.k8s_client.list_namespaced_pod(
                    namespace="default",
                    label_selector=f"training-job={training_name}"
                )
                
                for worker in workers.items:
                    worker_id = worker.metadata.labels.get("worker-id")
                    status = self._get_worker_status(worker.status.phase)
                    
                    self.metrics["worker_status"].labels(
                        worker_id=worker_id
                    ).set(status)
                    
                    # Get GPU metrics if available
                    if worker.status.phase == "Running":
                        gpu_metrics = await self._get_gpu_metrics(worker.metadata.name)
                        for gpu_id, utilization in gpu_metrics.items():
                            self.metrics["gpu_utilization"].labels(
                                worker_id=worker_id,
                                gpu_id=gpu_id
                            ).set(utilization)
                
                await asyncio.sleep(10)
                
        except Exception as e:
            logger.error(f"Training monitoring failed: {e}")
    
    def _get_worker_status(self, phase: str) -> int:
        """Convert Kubernetes pod phase to status code"""
        status_map = {
            "Pending": 0,
            "Running": 1,
            "Failed": 2,
            "Unknown": 0
        }
        return status_map.get(phase, 0)
    
    async def _get_gpu_metrics(self, pod_name: str) -> Dict[str, float]:
        """Get GPU metrics from pod"""
        try:
            # Execute nvidia-smi in pod
            exec_command = [
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu",
                "--format=csv,noheader,nounits"
            ]
            
            response = self.k8s_client.read_namespaced_pod_exec(
                pod_name,
                "default",
                command=exec_command,
                stdout=True,
                stderr=True
            )
            
            # Parse response
            metrics = {}
            for line in response.split("\n"):
                if line.strip():
                    gpu_id, utilization = line.split(",")
                    metrics[gpu_id.strip()] = float(utilization.strip())
            
            return metrics
            
        except Exception as e:
            logger.error(f"GPU metrics collection failed: {e}")
            return {}