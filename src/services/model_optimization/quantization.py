import torch
import torch.quantization
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
from typing import Dict, Optional, Union
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class QuantizationConfig:
    dtype: torch.dtype = torch.float16
    quantization_method: str = "dynamic"  # dynamic or static
    bits: int = 8
    symmetric: bool = True
    per_channel: bool = True
    calibration_samples: int = 100

class ModelQuantizer:
    def __init__(self, config: QuantizationConfig):
        self.config = config
        self.calibration_data = None
    
    async def quantize_model(
        self,
        model: AutoModelForCausalLM,
        calibration_data: Optional[torch.Tensor] = None
    ) -> AutoModelForCausalLM:
        try:
            if self.config.quantization_method == "dynamic":
                return await self._apply_dynamic_quantization(model)
            else:
                return await self._apply_static_quantization(
                    model,
                    calibration_data
                )
        except Exception as e:
            logger.error(f"Quantization failed: {e}")
            return model
    
    async def _apply_dynamic_quantization(
        self,
        model: AutoModelForCausalLM
    ) -> AutoModelForCausalLM:
        try:
            # Configure quantization
            model.qconfig = torch.quantization.get_default_dynamic_qconfig(
                dtype=self.config.dtype
            )
            
            # Prepare model for quantization
            model_prepared = torch.quantization.prepare(model)
            
            # Convert to quantized model
            model_quantized = torch.quantization.convert(model_prepared)
            
            logger.info(
                f"Successfully applied dynamic quantization with "
                f"{self.config.bits} bits"
            )
            return model_quantized
            
        except Exception as e:
            logger.error(f"Dynamic quantization failed: {e}")
            return model
    
    async def _apply_static_quantization(
        self,
        model: AutoModelForCausalLM,
        calibration_data: Optional[torch.Tensor]
    ) -> AutoModelForCausalLM:
        try:
            if calibration_data is None:
                raise ValueError("Calibration data required for static quantization")
            
            # Configure static quantization
            model.qconfig = torch.quantization.get_default_qconfig(
                'fbgemm' if torch.cuda.is_available() else 'qnnpack'
            )
            
            # Prepare model for quantization
            model_prepared = torch.quantization.prepare(model)
            
            # Calibrate with sample data
            with torch.no_grad():
                for sample in calibration_data:
                    model_prepared(sample)
            
            # Convert to quantized model
            model_quantized = torch.quantization.convert(model_prepared)
            
            logger.info(
                f"Successfully applied static quantization with "
                f"{self.config.bits} bits"
            )
            return model_quantized
            
        except Exception as e:
            logger.error(f"Static quantization failed: {e}")
            return model