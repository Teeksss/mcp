import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class DistillationConfig:
    temperature: float = 2.0
    alpha: float = 0.5
    batch_size: int = 32
    epochs: int = 3
    learning_rate: float = 1e-4
    warmup_steps: int = 100

class KnowledgeDistiller:
    def __init__(self, config: DistillationConfig):
        self.config = config
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    
    async def distill_knowledge(
        self,
        teacher_model: AutoModelForCausalLM,
        student_model: AutoModelForCausalLM,
        train_dataloader: torch.utils.data.DataLoader
    ) -> AutoModelForCausalLM:
        try:
            # Move models to device
            teacher_model = teacher_model.to(self.device)
            student_model = student_model.to(self.device)
            
            # Set teacher model to eval mode
            teacher_model.eval()
            
            # Prepare optimizer and scheduler
            optimizer = torch.optim.AdamW(
                student_model.parameters(),
                lr=self.config.learning_rate
            )
            
            scheduler = self._get_scheduler(optimizer, len(train_dataloader))
            
            # Training loop
            for epoch in range(self.config.epochs):
                total_loss = 0
                student_model.train()
                
                for batch in train_dataloader:
                    loss = await self._distillation_step(
                        teacher_model,
                        student_model,
                        batch,
                        optimizer,
                        scheduler
                    )
                    total_loss += loss
                
                avg_loss = total_loss / len(train_dataloader)
                logger.info(f"Epoch {epoch + 1}, Average Loss: {avg_loss:.4f}")
            
            return student_model
            
        except Exception as e:
            logger.error(f"Distillation failed: {e}")
            return student_model
    
    async def _distillation_step(
        self,
        teacher_model: AutoModelForCausalLM,
        student_model: AutoModelForCausalLM,
        batch: Dict[str, torch.Tensor],
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler
    ) -> float:
        try:
            # Move batch to device
            inputs = {k: v.to(self.device) for k, v in batch.items()}
            
            # Get teacher predictions
            with torch.no_grad():
                teacher_outputs = teacher_model(**inputs)
                teacher_logits = teacher_outputs.logits
            
            # Get student predictions
            student_outputs = student_model(**inputs)
            student_logits = student_outputs.logits
            
            # Calculate losses
            distillation_loss = self._compute_distillation_loss(
                student_logits,
                teacher_logits
            )
            
            task_loss = student_outputs.loss if hasattr(
                student_outputs, 'loss'
            ) else 0
            
            # Combine losses
            total_loss = (
                self.config.alpha * task_loss +
                (1 - self.config.alpha) * distillation_loss
            )
            
            # Backpropagate and optimize
            total_loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            return total_loss.item()
            
        except Exception as e:
            logger.error(f"Distillation step failed: {e}")
            raise
    
    def _compute_distillation_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor
    ) -> torch.Tensor:
        """Compute the distillation loss using KL divergence"""
        distillation_loss = nn.KLDivLoss(reduction='batchmean')(
            nn.functional.log_softmax(
                student_logits / self.config.temperature,
                dim=-1
            ),
            nn.functional.softmax(
                teacher_logits / self.config.temperature,
                dim=-1
            )
        ) * (self.config.temperature ** 2)
        
        return distillation_loss
    
    def _get_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
        num_training_steps: int
    ) -> torch.optim.lr_scheduler.LRScheduler:
        """Get learning rate scheduler"""
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.config.learning_rate,
            steps_per_epoch=num_training_steps,
            epochs=self.config.epochs,
            pct_start=self.config.warmup_steps / num_training_steps
        )