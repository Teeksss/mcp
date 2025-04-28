from typing import Optional, Dict, List
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging
import time
from datetime import datetime
import traceback
from src.config.settings import settings
from src.models.core import ModelRequest, ModelResponse

logger = logging.getLogger(__name__)

class ModelService:
    def __init__(self):
        self.model_pool = {}
        self.model_locks = {}
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    async def get_model(self, model_name: str):
        if model_name not in self.model_locks:
            self.model_locks[model_name] = asyncio.Lock()
        
        async with self.model_locks[model_name]:
            if model_name not in self.model_pool:
                model_settings = settings.models[model_name]
                
                try:
                    model = AutoModelForCausalLM.from_pretrained(
                        model_settings.name,
                        torch_dtype=torch.float16,
                        device_map="auto"
                    )
                    tokenizer = AutoTokenizer.from_pretrained(model_settings.name)
                    
                    self.model_pool[model_name] = {
                        'model': model,
                        'tokenizer': tokenizer,
                        'last_used': datetime.now(),
                        'total_requests': 0,
                        'total_tokens': 0
                    }
                except Exception as e:
                    logger.error(f"Model loading error for {model_name}: {e}")
                    raise
        
        return self.model_pool[model_name]
    
    async def generate_response(
        self,
        request: ModelRequest,
        context: Optional[List[Dict]] = None
    ) -> ModelResponse:
        start_time = time.time()
        model_name = request.model_name
        
        try:
            model_instance = await self.get_model(model_name)
            
            # Prepare prompt with context
            full_prompt = self._prepare_prompt(request.prompt, context)
            
            # Tokenize input
            inputs = await self._tokenize(
                model_instance['tokenizer'],
                full_prompt,
                request
            )
            
            # Generate response
            outputs = await self._generate(
                model_instance['model'],
                inputs,
                request
            )
            
            # Decode response
            response_text = model_instance['tokenizer'].decode(
                outputs[0],
                skip_special_tokens=True
            )
            
            # Update model statistics
            self._update_stats(model_name, outputs[0].shape[0])
            
            return ModelResponse(
                response=response_text,
                model_used=model_name,
                execution_time=time.time() - start_time,
                token_count=outputs[0].shape[0],
                request_id=request.request_id,
                rag_context=[doc['content'] for doc in (context or [])]
            )
            
        except Exception as e:
            logger.error(f"Generation error: {traceback.format_exc()}")
            # Try fallback models
            for fallback_model in settings.models[model_name].fallback_models:
                try:
                    request.model_name = fallback_model
                    return await self.generate_response(request, context)
                except Exception as fallback_e:
                    logger.error(f"Fallback {fallback_model} error: {fallback_e}")
            raise
    
    def _prepare_prompt(
        self,
        prompt: str,
        context: Optional[List[Dict]]
    ) -> str:
        if not context:
            return prompt
            
        context_text = "\n".join([
            f"Context {i+1}: {doc['content']}"
            for i, doc in enumerate(context)
        ])
        
        return f"""Given the following context:

{context_text}

User Question: {prompt}

Assistant Response:"""
    
    async def _tokenize(
        self,
        tokenizer,
        text: str,
        request: ModelRequest
    ) -> Dict:
        return await asyncio.get_event_loop().run_in_executor(
            self.executor,
            lambda: tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=request.max_tokens or settings.models[request.model_name].max_tokens
            ).to("cuda")
        )
    
    async def _generate(
        self,
        model,
        inputs: Dict,
        request: ModelRequest
    ) -> torch.Tensor:
        model_settings = settings.models[request.model_name]
        
        generation_config = {
            "max_length": request.max_tokens or model_settings.max_tokens,
            "temperature": request.temperature or model_settings.temperature,
            "top_p": model_settings.top_p,
            "presence_penalty": model_settings.presence_penalty,
            "frequency_penalty": model_settings.frequency_penalty,
            "do_sample": True
        }
        
        return await asyncio.get_event_loop().run_in_executor(
            self.executor,
            lambda: model.generate(**inputs, **generation_config)
        )
    
    def _update_stats(self, model_name: str, token_count: int):
        stats = self.model_pool[model_name]
        stats['last_used'] = datetime.now()
        stats['total_requests'] += 1
        stats['total_tokens'] += token_count