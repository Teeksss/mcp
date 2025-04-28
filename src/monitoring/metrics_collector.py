from typing import Dict, List, Optional
import time
from prometheus_client import Counter, Gauge, Histogram, Summary
import psutil
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class MetricsCollector:
    def __init__(self):
        self.timestamp = datetime.utcnow()
        self.current_user = "Teeksss"
        
        # System metrics
        self.system_metrics = {
            "cpu_usage": Gauge(
                "system_cpu_usage_percent",
                "CPU usage percentage",
                ["cpu_type"]
            ),
            "memory_usage": Gauge(
                "system_memory_usage_bytes",
                "Memory usage in bytes",
                ["memory_type"]
            ),
            "disk_usage": Gauge(
                "system_disk_usage_bytes",
                "Disk usage in bytes",
                ["mount_point"]
            ),
            "network_io": Counter(
                "system_network_bytes_total",
                "Network I/O bytes",
                ["direction"]
            )
        }
        
        # Application metrics
        self.app_metrics = {
            "requests_total": Counter(
                "app_requests_total",
                "Total application requests",
                ["endpoint", "method", "status"]
            ),
            "request_duration": Histogram(
                "app_request_duration_seconds",
                "Request duration in seconds",
                ["endpoint"],
                buckets=[.01, .05, .1, .5, 1, 5]
            ),
            "active_users": Gauge(
                "app_active_users",
                "Number of active users",
                ["endpoint"]
            )
        }
        
        # Model metrics
        self.model_metrics = {
            "training_duration": Histogram(
                "model_training_duration_seconds",
                "Model training duration",
                ["model_name", "version"]
            ),
            "inference_latency": Histogram(
                "model_inference_latency_seconds",
                "Model inference latency",
                ["model_name", "version"]
            ),
            "prediction_errors": Counter(
                "model_prediction_errors_total",
                "Total prediction errors",
                ["model_name", "error_type"]
            ),
            "model_accuracy": Gauge(
                "model_accuracy",
                "Model accuracy metrics",
                ["model_name", "metric_type"]
            )
        }
        
        # Resource metrics
        self.resource_metrics = {
            "gpu_utilization": Gauge(
                "gpu_utilization_percent",
                "GPU utilization percentage",
                ["gpu_id"]
            ),
            "gpu_memory": Gauge(
                "gpu_memory_usage_bytes",
                "GPU memory usage",
                ["gpu_id"]
            ),
            "batch_size": Histogram(
                "batch_size_distribution",
                "Distribution of batch sizes",
                ["model_name"]
            )
        }
    
    async def collect_system_metrics(self):
        """Collect system-level metrics"""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1, percpu=True)
            for i, cpu in enumerate(cpu_percent):
                self.system_metrics["cpu_usage"].labels(
                    cpu_type=f"cpu_{i}"
                ).set(cpu)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            self.system_metrics["memory_usage"].labels(
                memory_type="total"
            ).set(memory.total)
            self.system_metrics["memory_usage"].labels(
                memory_type="used"
            ).set(memory.used)
            
            # Disk metrics
            for partition in psutil.disk_partitions():
                usage = psutil.disk_usage(partition.mountpoint)
                self.system_metrics["disk_usage"].labels(
                    mount_point=partition.mountpoint
                ).set(usage.used)
            
            # Network metrics
            net_io = psutil.net_io_counters()
            self.system_metrics["network_io"].labels(
                direction="sent"
            ).inc(net_io.bytes_sent)
            self.system_metrics["network_io"].labels(
                direction="received"
            ).inc(net_io.bytes_recv)
            
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
    
    async def collect_model_metrics(
        self,
        model_name: str,
        version: str,
        metrics: Dict
    ):
        """Collect model-specific metrics"""
        try:
            # Training metrics
            if "training_duration" in metrics:
                self.model_metrics["training_duration"].labels(
                    model_name=model_name,
                    version=version
                ).observe(metrics["training_duration"])
            
            # Accuracy metrics
            for metric_type, value in metrics.get("accuracy", {}).items():
                self.model_metrics["model_accuracy"].labels(
                    model_name=model_name,
                    metric_type=metric_type
                ).set(value)
            
            # Error tracking
            if "errors" in metrics:
                for error_type, count in metrics["errors"].items():
                    self.model_metrics["prediction_errors"].labels(
                        model_name=model_name,
                        error_type=error_type
                    ).inc(count)
            
        except Exception as e:
            logger.error(f"Failed to collect model metrics: {e}")
    
    async def collect_gpu_metrics(self):
        """Collect GPU-related metrics"""
        try:
            # Using nvidia-smi for GPU metrics
            import pynvml
            pynvml.nvmlInit()
            
            device_count = pynvml.nvmlDeviceGetCount()
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                
                # GPU utilization
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                self.resource_metrics["gpu_utilization"].labels(
                    gpu_id=i
                ).set(utilization.gpu)
                
                # GPU memory
                memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                self.resource_metrics["gpu_memory"].labels(
                    gpu_id=i
                ).set(memory.used)
            
            pynvml.nvmlShutdown()
            
        except Exception as e:
            logger.error(f"Failed to collect GPU metrics: {e}")
    
    def track_request(
        self,
        endpoint: str,
        method: str,
        status: int,
        duration: float
    ):
        """Track API request metrics"""
        try:
            self.app_metrics["requests_total"].labels(
                endpoint=endpoint,
                method=method,
                status=status
            ).inc()
            
            self.app_metrics["request_duration"].labels(
                endpoint=endpoint
            ).observe(duration)
            
        except Exception as e:
            logger.error(f"Failed to track request metrics: {e}")
    
    def track_batch_size(
        self,
        model_name: str,
        batch_size: int
    ):
        """Track batch size distribution"""
        try:
            self.resource_metrics["batch_size"].labels(
                model_name=model_name
            ).observe(batch_size)
            
        except Exception as e:
            logger.error(f"Failed to track batch size: {e}")