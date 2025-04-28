import torch
import torch.quantization
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from typing import Dict, Optional, List
import logging
from dataclasses import dataclass
import gc
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

@dataclass
class OptimizationConfig:
    quantization_type: str = "dynamic"  # dynamic, static, or None
    pruning_level: float = 0.3  # Pruning threshold
    use_gpu: bool = True
    batch_size: int = 32
    use_half_precision: bool = True
    enable_cuda_graphs: bool = True
    model_parallel: bool = False
    num_gpu_shards: int = 1

class ModelOptimizer:
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.device = self._setup_device()
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.models: Dict[str, Dict] = {}
    
    def _setup_device(self) -> torch.device:
        if self.config.use_gpu and torch.cuda.is_available():
            # Enable TF32 for better performance on Ampere GPUs
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            return torch.device("cuda")
        return torch.device("cpu")
    
    async def optimize_model(
        self,
        model_name: str,
        model_path: str
    ) -> None:
        try:
            # Load model with initial optimizations
            model = await self._load_model(model_path)
            
            # Apply quantization if enabled
            if self.config.quantization_type != "none":
                model = await self._apply_quantization(model)
            
            # Apply pruning if threshold is set
            if self.config.pruning_level > 0:
                model = await self._apply_pruning(model)
            
            # Move to device and apply additional optimizations
            model = await self._optimize_for_device(model)
            
            # Store optimized model
            self.models[model_name] = {
                "model": model,
                "tokenizer": AutoTokenizer.from_pretrained(model_path)
            }
            
            logger.info(f"Successfully optimized model: {model_name}")
            
        except Exception as e:
            logger.error(f"Model optimization failed for {model_name}: {e}")
            raise
    
    async def _load_model(self, model_path: str):
        try:
            # Load with half precision if enabled
            dtype = torch.float16 if self.config.use_half_precision else torch.float32
            
            if self.config.model_parallel:
                # Load model in sharded mode for multi-GPU
                return AutoModelForCausalLM.from_pretrained(
                    model_path,
                    device_map="auto",
                    torch_dtype=dtype,
                    low_cpu_mem_usage=True
                )
            else:
                return AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=dtype
                )
                
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            raise
    
    async def _apply_quantization(self, model):
        try:
            if self.config.quantization_type == "dynamic":
                # Dynamic quantization
                return torch.quantization.quantize_dynamic(
                    model,
                    {torch.nn.Linear},
                    dtype=torch.qint8
                )
            elif self.config.quantization_type == "static":
                # Static quantization (requires calibration)
                model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
                torch.quantization.prepare(model, inplace=True)
                # Note: Would need calibration data here
                return torch.quantization.convert(model)
            
            return model
            
        except Exception as e:
            logger.error(f"Quantization failed: {e}")
            return model
    
    async def _apply_pruning(self, model):
        try:
            for name, parameter in model.named_parameters():
                if 'weight' in name:
                    # Calculate threshol