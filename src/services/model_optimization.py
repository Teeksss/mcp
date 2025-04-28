import torch
import torch.quantization
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
from typing import Dict, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ModelOptimizationConfig:
    quantization_dtype: torch.dtype = torch.float16
    use_gpu: bool = True
    batch_size: int = 32
    enable_cuda_graphs: bool = True
    use_flash_attention: bool = True
    enable_kernel_fusion: bool = True

class OptimizedModelService:
    def __init__(self, config: ModelOptimizationConfig):
        self.config = config
        self.device = self._setup_device()
        self.models: Dict[str, Dict] = {}
        
    def _setup_device(self) -> torch.device:
        if self.config.use_gpu and torch.cuda.is_available():
            # Enable TF32 for better performance on Ampere GPUs
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            return torch.device("cuda")
        return torch.device("cpu")
    
    async def load_model(self, model_name: str, model_path: str):
        try:
            # Load model with optimizations
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=self.config.quantization_dtype,
                device_map="auto",
                use_flash_attention=self.config.use_flash_attention
            )
            
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            # Apply quantization if on CPU
            if self.device.type == "cpu":
                model = self._quantize_model(model)
            
            # Enable fusion for faster inference
            if self.config.enable_kernel_fusion:
                model = self._apply_fusion_optimizations(model)
            
            self.models[model_name] = {
                "model": model,
                "tokenizer": tokenizer
            }
            
            logger.info(f"Successfully loaded and optimized model: {model_name}")
            
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            raise
    
    def _quantize_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """Apply dynamic quantization to the model"""
        try:
            # Configure quantization
            model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
            
            # Prepare for quantization
            torch.quantization.prepare(model, inplace=True)
            
            # Convert to quantized model
            quantized_model = torch.quantization.convert(model)
            
            logger.info("Successfully applied quantization")
            return quantized_model
            
        except Exception as e:
            logger.error(f"Quantization failed: {e}")
            return model
    
    def _apply_fusion_optimizations(self, model: torch.nn.Module) -> torch.nn.Module:
        """Apply fusion optimizations for faster inference"""
        try:
            if self.device.type == "cuda":
                # Enable CUDA graphs for repeated operations
                torch.cuda.enable_cuda_graphs()
                
                # Apply fusion to attention layers
                for layer in model.modules():
                    if hasattr(layer, "fused_attention"):
                        layer.fused_attention = True
            
            return model
            
        except Exception as e:
            logger.error(f"Fusion optimization failed: {e}")
            return model
    
    async def generate(
        self,
        model_name: str,
        prompt: str,
        max_length: int = 512,
        temperature: float = 0.7,
        batch_size: Optional[int] = None
    ) -> str:
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not loaded")
        
        try:
            model_data = self.models[model_name]
            model, tokenizer = model_data["model"], model_data["tokenizer"]
            
            # Tokenize input
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length
            ).to(self.device)
            
            # Use CUDA graphs for repeated operations
            if self.config.enable_cuda_graphs and self.device.type == "cuda":
                with torch.cuda.graph(model):
                    outputs = model.generate(
                        **inputs,
                        max_length=max_length,
                        temperature=temperature,
                        batch_size=batch_size or self.config.batch_size,
                        use_cache=True
                    )
            else:
                outputs = model.generate(
                    **inputs,
                    max_length=max_length,
                    temperature=temperature,
                    batch_size=batch_size or self.config.batch_size,
                    use_cache=True
                )
            
            return tokenizer.decode(outputs[0], skip_special_tokens=True)
            
        except Exception as e:
            logger.error(f"Generation failed for model {model_name}: {e}")
            raise