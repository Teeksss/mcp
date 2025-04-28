import torch
import torch.nn.utils.prune as prune
from transformers import AutoModelForCausalLM
import logging
from typing import Dict, Optional, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class PruningConfig:
    method: str = "l1_unstructured"
    amount: float = 0.3
    n_iterative_steps: int = 10
    min_layers_to_keep: int = 3
    importance_scores: Optional[Dict[str, float]] = None

class ModelPruner:
    def __init__(self, config: PruningConfig):
        self.config = config
        self.pruning_methods = {
            "l1_unstructured": prune.L1Unstructured,
            "random_unstructured": prune.RandomUnstructured,
            "ln_structured": prune.LnStructured
        }
    
    async def prune_model(
        self,
        model: AutoModelForCausalLM,
        importance_scores: Optional[Dict[str, float]] = None
    ) -> AutoModelForCausalLM:
        try:
            # Apply iterative pruning
            for step in range(self.config.n_iterative_steps):
                amount_per_step = self.config.amount / self.config.n_iterative_steps
                
                # Get pruning method
                pruning_method = self.pruning_methods.get(
                    self.config.method,
                    prune.L1Unstructured
                )
                
                # Apply pruning to each layer
                for name, module in model.named_modules():
                    if isinstance(module, torch.nn.Linear):
                        # Check if layer should be preserved
                        if self._should_preserve_layer(name):
                            continue
                        
                        # Apply pruning
                        pruning_method.apply(
                            module,
                            name='weight',
                            amount=amount_per_step
                        )
                        
                        # Remove pruning reparametrization
                        prune.remove(module, 'weight')
                
                logger.info(
                    f"Completed pruning step {step + 1}/{self.config.n_iterative_steps}"
                )
            
            return model
            
        except Exception as e:
            logger.error(f"Pruning failed: {e}")
            return model
    
    def _should_preserve_layer(self, layer_name: str) -> bool:
        """Determine if a layer should be preserved from pruning"""
        # Preserve important layers
        if "embedding" in layer_name or "output" in layer_name:
            return True
        
        # Check importance scores
        if self.config.importance_scores:
            score = self.config.importance_scores.get(layer_name, 0.0)
            if score > 0.8:  # Preserve highly important layers
                return True
        
        return False
    
    async def calculate_sparsity(self, model: AutoModelForCausalLM) -> Dict[str, float]:
        """Calculate the sparsity of each layer after pruning"""
        sparsity_stats = {}
        
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                weight = module.weight
                total_params = weight.numel()
                zero_params = (weight == 0).sum().item()
                sparsity = zero_params / total_params
                sparsity_stats[name] = sparsity
        
        return sparsity_stats