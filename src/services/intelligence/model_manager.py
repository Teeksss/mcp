from typing import Dict, List, Optional, Union, Tuple
import torch
import numpy as np
from dataclasses import dataclass
import logging
import asyncio
from datetime import datetime
import psutil
import GPUtil
from prometheus_client import Gauge, Histogram

logger = logging.getLogger(__name__)

@dataclass
class ModelProfile:
    name: str
    size: int  # Model size in MB
    avg_inference_time: float
    gpu_memory_required: int
    complexity: int  # 1-10 scale
    supported_tasks: List[str]
    fallback_models: List[str]
    min_batch_size: int
    max_batch_size: int
    optimal_batch_size: int

class IntelligentModelManager:
    def __init__(self):
        self.models = {}
        self.model_profiles = {}
        self.loaded_models = {}
        self.active_sessions = {}
        self.performance_metrics = self._setup_metrics()
        
        # Current context
        self.timestamp = datetime.utcnow()
        self.current_user = "Teeksss"
        
        # Resource monitoring
        self.resource_monitor = ResourceMonitor()
    
    def _setup_metrics(self) -> Dict:
        """Initialize performance metrics"""
        return {
            "model_load_time": Histogram(
                "model_load_time_seconds",
                "Time taken to load model",
                ["model_name", "version"]
            ),
            "model_memory_usage": Gauge(
                "model_memory_usage_bytes",
                "Memory used by model",
                ["model_name", "type"]
            ),
            "model_inference_time": Histogram(
                "model_inference_time_seconds",
                "Model inference time",
                ["model_name", "batch_size"]
            ),
            "model_accuracy": Gauge(
                "model_accuracy",
                "Model accuracy metrics",
                ["model_name", "metric_type"]
            )
        }
    
    async def register_model(
        self,
        model_name: str,
        model_profile: ModelProfile
    ):
        """Register a new model with its profile"""
        try:
            self.model_profiles[model_name] = model_profile
            logger.info(f"Registered model: {model_name}")
            
        except Exception as e:
            logger.error(f"Failed to register model: {e}")
            raise
    
    async def select_optimal_model(
        self,
        user_query: str,
        user_context: Dict,
        resource_constraints: Optional[Dict] = None
    ) -> Tuple[str, Dict]:
        """Dynamically select the optimal model based on various factors"""
        try:
            # Analyze query complexity
            query_complexity = await self._analyze_query_complexity(user_query)
            
            # Get available resources
            available_resources = await self.resource_monitor.get_resources()
            
            # Score each model
            model_scores = {}
            for model_name, profile in self.model_profiles.items():
                score = await self._calculate_model_score(
                    model_name=model_name,
                    profile=profile,
                    query_complexity=query_complexity,
                    user_context=user_context,
                    available_resources=available_resources,
                    resource_constraints=resource_constraints
                )
                model_scores[model_name] = score
            
            # Select best model
            selected_model = max(model_scores.items(), key=lambda x: x[1])[0]
            
            return selected_model, {
                "scores": model_scores,
                "query_complexity": query_complexity,
                "available_resources": available_resources
            }
            
        except Exception as e:
            logger.error(f"Model selection failed: {e}")
            raise
    
    async def _analyze_query_complexity(self, query: str) -> float:
        """Analyze query complexity"""
        complexity_factors = {
            "length": len(query),
            "special_tokens": len([c for c in query if not c.isalnum()]),
            "unique_words": len(set(query.split())),
            "nested_operations": query.count("(") + query.count("[")
        }
        
        # Normalize and weight factors
        weights = {
            "length": 0.3,
            "special_tokens": 0.2,
            "unique_words": 0.3,
            "nested_operations": 0.2
        }
        
        max_values = {
            "length": 1000,
            "special_tokens": 50,
            "unique_words": 100,
            "nested_operations": 10
        }
        
        complexity = sum(
            weights[factor] * (value / max_values[factor])
            for factor, value in complexity_factors.items()
        )
        
        return min(complexity, 1.0)
    
    async def _calculate_model_score(
        self,
        model_name: str,
        profile: ModelProfile,
        query_complexity: float,
        user_context: Dict,
        available_resources: Dict,
        resource_constraints: Optional[Dict]
    ) -> float:
        """Calculate model score based on multiple factors"""
        scores = {
            "complexity_match": self._score_complexity_match(
                profile.complexity,
                query_complexity
            ),
            "resource_availability": self._score_resource_availability(
                profile,
                available_resources,
                resource_constraints
            ),
            "performance_history": await self._score_performance_history(
                model_name,
                user_context
            ),
            "task_suitability": self._score_task_suitability(
                profile,
                user_context
            )
        }
        
        weights = {
            "complexity_match": 0.3,
            "resource_availability": 0.3,
            "performance_history": 0.2,
            "task_suitability": 0.2
        }
        
        return sum(
            score * weights[factor]
            for factor, score in scores.items()
        )
    
    def _score_complexity_match(
        self,
        model_complexity: int,
        query_complexity: float
    ) -> float:
        """Score how well model complexity matches query complexity"""
        normalized_model_complexity = model_complexity / 10
        difference = abs(normalized_model_complexity - query_complexity)
        return 1 - difference
    
    def _score_resource_availability(
        self,
        profile: ModelProfile,
        available_resources: Dict,
        resource_constraints: Optional[Dict]
    ) -> float:
        """Score based on resource availability"""
        if resource_constraints:
            if (
                profile.gpu_memory_required > resource_constraints.get("gpu_memory", float("inf")) or
                profile.size > resource_constraints.get("memory", float("inf"))
            ):
                return 0.0
        
        memory_score = min(
            available_resources["memory"] / profile.size,
            1.0
        )
        gpu_score = min(
            available_resources["gpu_memory"] / profile.gpu_memory_required
            if available_resources["gpu_memory"] > 0
            else 0.0,
            1.0
        )
        
        return (memory_score + gpu_score) / 2
    
    async def _score_performance_history(
        self,
        model_name: str,
        user_context: Dict
    ) -> float:
        """Score based on historical performance"""
        # Implementation depends on performance tracking system
        return 0.8  # Placeholder
    
    def _score_task_suitability(
        self,
        profile: ModelProfile,
        user_context: Dict
    ) -> float:
        """Score how suitable the model is for the task"""
        if user_context.get("task") in profile.supported_tasks:
            return 1.0
        return 0.0

class ResourceMonitor:
    def __init__(self):
        self.update_interval = 5  # seconds
        self._last_update = 0
        self._cached_resources = {}
    
    async def get_resources(self) -> Dict:
        """Get available system resources"""
        current_time = time.time()
        if current_time - self._last_update > self.update_interval:
            self._cached_resources = await self._update_resources()
            self._last_update = current_time
        
        return self._cached_resources
    
    async def _update_resources(self) -> Dict:
        """Update resource availability"""
        # Memory
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        # CPU
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # GPU
        gpu_resources = {"memory": 0, "utilization": 0}
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu_resources = {
                    "memory": sum(gpu.memoryFree for gpu in gpus),
                    "utilization": sum(gpu.load for gpu in gpus) / len(gpus)
                }
        except Exception:
            pass
        
        return {
            "memory": memory.available,
            "swap": swap.free,
            "cpu": 100 - cpu_percent,
            "gpu_memory": gpu_resources["memory"],
            "gpu_utilization": gpu_resources["utilization"]
        }