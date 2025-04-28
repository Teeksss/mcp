from datetime import datetime
import asyncio
from typing import Dict, List, Optional
import psutil
import torch
import numpy as np
from prometheus_client import Gauge, Counter, Histogram
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ResourceMonitorConfig:
    check_interval: int = 15  # seconds
    gpu_memory_threshold: float = 0.8  # 80%
    cpu_threshold: float = 0.7  # 70%
    memory_threshold: float = 0.8  # 80%
    scale_up_threshold: int = 3  # consecutive checks
    scale_down_threshold: int = 5  # consecutive checks

class ResourceMonitor:
    def __init__(self, config: ResourceMonitorConfig):
        self.config = config
        self.metrics = self._setup_metrics()
        self.high_usage_count = 0
        self.low_usage_count = 0
    
    def _setup_metrics(self) -> Dict:
        """Setup Prometheus metrics"""
        return {
            "gpu_memory_usage": Gauge(
                "mcp_gpu_memory_usage",
                "GPU memory usage percentage",
                ["device"]
            ),
            "cpu_usage": Gauge(
                "mcp_cpu_usage",
                "CPU usage percentage"
            ),
            "memory_usage": Gauge(
                "mcp_memory_usage",
                "Memory usage percentage"
            ),
            "scale_events": Counter(
                "mcp_scale_events_total",
                "Number of scaling events",
                ["direction"]
            )
        }
    
    async def start_monitoring(self):
        """Start resource monitoring loop"""
        while True:
            try:
                await self._check_resources()
                await asyncio.sleep(self.config.check_interval)
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
    
    async def _check_resources(self):
        """Check system resources and trigger scaling if needed"""
        metrics = await self._collect_metrics()
        
        # Update Prometheus metrics
        self._update_prometheus_metrics(metrics)
        
        # Check for scaling needs
        await self._evaluate_scaling(metrics)
    
    async def _collect_metrics(self) -> Dict:
        """Collect system resource metrics"""
        metrics = {
            "timestamp": datetime.utcnow().isoformat(),
            "cpu_usage": psutil.cpu_percent() / 100,
            "memory_usage": psutil.virtual_memory().percent / 100,
            "gpu_usage": {}
        }
        
        # Collect GPU metrics if available
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                memory_allocated = torch.cuda.memory_allocated(i)
                memory_total = torch.cuda.get_device_properties(i).total_memory
                metrics["gpu_usage"][f"gpu_{i}"] = memory_allocated / memory_total
        
        return metrics
    
    def _update_prometheus_metrics(self, metrics: Dict):
        """Update Prometheus metrics"""
        self.metrics["cpu_usage"].set(metrics["cpu_usage"] * 100)
        self.metrics["memory_usage"].set(metrics["memory_usage"] * 100)
        
        for device, usage in metrics["gpu_usage"].items():
            self.metrics["gpu_memory_usage"].labels(device).set(usage * 100)
    
    async def _evaluate_scaling(self, metrics: Dict):
        """Evaluate if scaling is needed"""
        # Calculate overall resource pressure
        gpu_pressure = max(metrics["gpu_usage"].values()) if metrics["gpu_usage"] else 0
        system_pressure = max(
            metrics["cpu_usage"],
            metrics["memory_usage"],
            gpu_pressure
        )
        
        # Check if we need to scale up
        if system_pressure > self.config.gpu_memory_threshold:
            self.high_usage_count += 1
            self.low_usage_count = 0
            
            if self.high_usage_count >= self.config.scale_up_threshold:
                await self._trigger_scale_up()
                self.high_usage_count = 0
        
        # Check if we can scale down
        elif system_pressure < (self.config.gpu_memory_threshold * 0.5):
            self.low_usage_count += 1
            self.high_usage_count = 0
            
            if self.low_usage_count >= self.config.scale_down_threshold:
                await self._trigger_scale_down()
                self.low_usage_count = 0
        
        else:
            self.high_usage_count = 0
            self.low_usage_count = 0
    
    async def _trigger_scale_up(self):
        """Trigger scaling up of resources"""
        try:
            self.metrics["scale_events"].labels("up").inc()
            # Implement your scaling logic here
            logger.info("Triggering scale up")
        except Exception as e:
            logger.error(f"Scale up failed: {e}")
    
    async def _trigger_scale_down(self):
        """Trigger scaling down of resources"""
        try:
            self.metrics["scale_events"].labels("down").inc()
            # Implement your scaling logic here
            logger.info("Triggering scale down")
        except Exception as e:
            logger.error(f"Scale down failed: {e}")