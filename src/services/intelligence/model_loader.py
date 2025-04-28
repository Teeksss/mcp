from typing import Dict, Optional, Any
import torch
import gc
import asyncio
from datetime import datetime
import logging
from prometheus_client import Counter, Gauge, Histogram
import numpy as np

logger = logging.getLogger(__name__)

class DynamicModelLoader:
    def __init__(
        self,
        cache_size_mb: int = 8192,  # 8GB default cache
        min_free_memory: float = 0.2  # 20% minimum free memory
    ):
        self.cache_size_mb = cache_size_mb
        self.min_free_memory = min_free_memory
        self.loaded_models = {}
        self.model_usage = {}
        self.model_locks = {}
        
        # Performance metrics
        self.metrics = self._setup_metrics()
        
        # Current context
        self.timestamp = datetime.utcnow()
        self.current_user = "Teeksss"
    
    def _setup_metrics(self) -> Dict:
        """Initialize metrics collection"""
        return {
            "model_cache_size": Gauge(
                "model_cache_size_bytes",
                "Current model cache size",
                ["type"]
            ),
            "model_load_count": Counter(
                "model_load_count_total",
                "Number of model loads",
                ["model_name", "reason"]
            ),
            "model_unload_count": Counter(
                "model_unload_count_total",
                "Number of model unloads",
                ["model_name", "reason"]
            ),
            "cache_hits": Counter(
                "model_cache_hits_total",
                "Number of cache hits",
                ["model_name"]
            ),
            "cache_misses": Counter(
                "model_cache_misses_total",
                "Number of cache misses",
                ["model_name"]
            )
        }
    
    async def load_model(
        self,
        model_name: str,
        model_config: Dict,
        force: bool = False
    ) -> Any:
        """Load model with lazy loading and caching"""
        try:
            # Check if model is already loaded
            if model_name in self.loaded_models and not force:
                self.metrics["cache_hits"].labels(
                    model_name=model_name
                ).inc()
                self.model_usage[model_name] = self.timestamp
                return self.loaded_models[model_name]
            
            self.metrics["cache_misses"].labels(
                model_name=model_name
            ).inc()
            
            # Acquire lock for this model
            if model_name not in self.model_locks:
                self.model_locks[model_name] = asyncio.Lock()
            
            async with self.model_locks[model_name]:
                # Check memory availability
                await self._ensure_memory_available(
                    model_config.get("size_mb", 0)
                )
                
                # Load model
                model = await self._load_model_implementation(
                    model_name,
                    model_config
                )
                
                # Update cache
                self.loaded_models[model_name] = model
                self.model_usage[model_name] = self.timestamp
                
                # Update metrics
                self.metrics["model_load_count"].labels(
                    model_name=model_name,
                    reason="initial_load"
                ).inc()
                
                self.metrics["model_cache_size"].labels(
                    type="gpu" if model_config.get("use_gpu", False) else "cpu"
                ).inc(model_config.get("size_mb", 0))
                
                return model
                
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise
    
    async def unload_model(
        self,
        model_name: str,
        reason: str = "manual"
    ):
        """Unload model from memory"""
        try:
            if model_name in self.loaded_models:
                async with self.model_locks[model_name]:
                    model = self.loaded_models[model_name]
                    
                    # Move model to CPU if it's on GPU
                    if hasattr(model, "to"):
                        model.to("cpu")
                    
                    # Clear from memory
                    del self.loaded_models[model_name]
                    del self.model_usage[model_name]
                    
                    # Force garbage collection
                    gc.collect()
                    torch.cuda.empty_cache()
                    
                    # Update metrics
                    self.metrics["model_unload_count"].labels(
                        model_name=model_name,
                        reason=reason
                    ).inc()
                    
                    logger.info(f"Unloaded model {model_name}")
            
        except Exception as e:
            logger.error(f"Failed to unload model {model_name}: {e}")
            raise
    
    async def _ensure_memory_available(self, required_mb: int):
        """Ensure sufficient memory is available"""
        while True:
            # Check current memory usage
            memory_usage = self._get_current_memory_usage()
            
            if memory_usage + required_mb <= self.cache_size_mb:
                break
            
            # Find least recently used model
            if not self.model_usage:
                raise RuntimeError("No models to unload")
            
            lru_model = min(
                self.model_usage.items(),
                key=lambda x: x[1]
            )[0]
            
            # Unload least recently used model
            await self.unload_model(lru_model, reason="memory_pressure")
    
    def _get_current_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024
        return 0
    
    async def _load_model_implementation(
        self,
        model_name: str,
        model_config: Dict
    ) -> Any:
        """Actual model loading implementation"""
        try:
            # Custom loading logic based on model type
            if model_config.get("framework") == "pytorch":
                return await self._load_pytorch_model(
                    model_config["path"],
                    model_config.get("use_gpu", False)
                )
            elif model_config.get("framework") == "tensorflow":
                return await self._load_tensorflow_model(
                    model_config["path"],
                    model_config.get("use_gpu", False)
                )
            else:
                raise ValueError(f"Unsupported framework: {model_config.get('framework')}")
            
        except Exception as e:
            logger.error(f"Model loading implementation failed: {e}")
            raise
    
    async def _load_pytorch_model(
        self,
        model_path: str,
        use_gpu: bool
    ) -> torch.nn.Module:
        """Load PyTorch model"""
        model = torch.load(model_path)
        if use_gpu and torch.cuda.is_available():
            model = model.cuda()
        return model
    
    async def _load_tensorflow_model(
        self,
        model_path: str,
        use_gpu: bool
    ) -> Any:
        """Load TensorFlow model"""
        import tensorflow as tf
        if not use_gpu:
            tf.config.set_visible_devices([], 'GPU')
        return tf.saved_model.load(model_path)