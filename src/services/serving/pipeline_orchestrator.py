from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
import asyncio
import json
from prometheus_client import Counter, Histogram, Gauge
import logging
import aioredis
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel

logger = logging.getLogger(__name__)

@dataclass
class PipelineConfig:
    redis_url: str
    max_concurrent_requests: int = 100
    request_timeout: int = 30
    max_retries: int = 3
    batch_size: int = 32
    enable_caching: bool = True
    cache_ttl: int = 3600

class PipelineStage(BaseModel):
    name: str
    endpoint: str
    timeout: int
    retry_count: int
    required: bool = True
    cache_results: bool = True

class PipelineOrchestrator:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.metrics = self._setup_metrics()
        self.redis = aioredis.from_url(config.redis_url)
        self.pipelines: Dict[str, List[PipelineStage]] = {}
        self.semaphore = asyncio.Semaphore(config.max_concurrent_requests)
        
        # Current context
        self.timestamp = datetime.utcnow()
        self.current_user = "Teeksss"
        
        # Initialize FastAPI app
        self.app = FastAPI()
        self._setup_routes()
    
    def _setup_metrics(self) -> Dict:
        """Initialize pipeline metrics"""
        return {
            "pipeline_requests": Counter(
                "pipeline_requests_total",
                "Total pipeline requests",
                ["pipeline_name"]
            ),
            "stage_latency": Histogram(
                "pipeline_stage_latency_seconds",
                "Stage processing time",
                ["pipeline_name", "stage_name"]
            ),
            "pipeline_errors": Counter(
                "pipeline_errors_total",
                "Pipeline execution errors",
                ["pipeline_name", "error_type"]
            ),
            "active_requests": Gauge(
                "pipeline_active_requests",
                "Currently active pipeline requests",
                ["pipeline_name"]
            )
        }
    
    def _setup_routes(self):
        """Setup FastAPI routes"""
        @self.app.post("/pipeline/{pipeline_name}")
        async def execute_pipeline(
            pipeline_name: str,
            request_data: Dict,
            background_tasks: BackgroundTasks
        ):
            try:
                # Validate pipeline exists
                if pipeline_name not in self.pipelines:
                    raise ValueError(f"Pipeline {pipeline_name} not found")
                
                # Execute pipeline
                result = await self.execute_pipeline(
                    pipeline_name,
                    request_data
                )
                
                # Schedule cleanup in background
                background_tasks.add_task(
                    self._cleanup_pipeline_data,
                    pipeline_name,
                    result.get("request_id")
                )
                
                return result
                
            except Exception as e:
                logger.error(f"Pipeline execution failed: {e}")
                self.metrics["pipeline_errors"].labels(
                    pipeline_name=pipeline_name,
                    error_type="execution"
                ).inc()
                raise
    
    async def register_pipeline(
        self,
        name: str,
        stages: List[PipelineStage]
    ):
        """Register new pipeline"""
        try:
            # Validate stages
            for stage in stages:
                if not await self._validate_stage(stage):
                    raise ValueError(
                        f"Invalid stage configuration: {stage.name}"
                    )
            
            self.pipelines[name] = stages
            logger.info(f"Registered pipeline: {name}")
            
        except Exception as e:
            logger.error(f"Pipeline registration failed: {e}")
            raise
    
    async def execute_pipeline(
        self,
        pipeline_name: str,
        input_data: Dict
    ) -> Dict:
        """Execute pipeline stages"""
        async with self.semaphore:
            try:
                start_time = self.timestamp
                request_id = f"{pipeline_name}_{start_time.timestamp()}"
                
                self.metrics["pipeline_requests"].labels(
                    pipeline_name=pipeline_name
                ).inc()
                
                self.metrics["active_requests"].labels(
                    pipeline_name=pipeline_name
                ).inc()
                
                # Initialize pipeline context
                context = {
                    "request_id": request_id,
                    "input": input_data,
                    "start_time": start_time.isoformat(),
                    "user": self.current_user,
                    "intermediate_results": {},
                    "final_result": None
                }
                
                # Execute stages
                stages = self.pipelines[pipeline_name]
                for stage in stages:
                    try:
                        # Check cache if enabled
                        if (
                            self.config.enable_caching and
                            stage.cache_results
                        ):
                            cached_result = await self._get_cached_result(
                                request_id,
                                stage.name
                            )
                            if cached_result:
                                context["intermediate_results"][stage.name] = cached_result
                                continue
                        
                        # Execute stage
                        result = await self._execute_stage(
                            stage,
                            context
                        )
                        
                        # Cache result if enabled
                        if (
                            self.config.enable_caching and
                            stage.cache_results
                        ):
                            await self._cache_result(
                                request_id,
                                stage.name,
                                result
                            )
                        
                        context["intermediate_results"][stage.name] = result
                        
                    except Exception as e:
                        if stage.required:
                            raise
                        logger.warning(
                            f"Non-critical stage {stage.name} failed: {e}"
                        )
                
                # Prepare final result
                context["final_result"] = await self._prepare_final_result(
                    context
                )
                
                return context
                
            except Exception as e:
                logger.error(f"Pipeline execution failed: {e}")
                self.metrics["pipeline_errors"].labels(
                    pipeline_name=pipeline_name,
                    error_type="execution"
                ).inc()
                raise
            
            finally:
                self.metrics["active_requests"].labels(
                    pipeline_name=pipeline_name
                ).dec()
    
    async def _execute_stage(
        self,
        stage: PipelineStage,
        context: Dict
    ) -> Any:
        """Execute single pipeline stage"""
        start_time = datetime.utcnow()
        
        try:
            # Prepare stage input
            stage_input = await self._prepare_stage_input(
                stage,
                context
            )
            
            # Execute stage with retry logic
            for attempt in range(stage.retry_count + 1):
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.post(
                            stage.endpoint,
                            json=stage_input,
                            timeout=stage.timeout
                        ) as response:
                            if response.status == 200:
                                result = await response.json()
                                break
                            else:
                                raise RuntimeError(
                                    f"Stage returned status {response.status}"
                                )
                except Exception as e:
                    if attempt == stage.retry_count:
                        raise
                    await asyncio.sleep(2 ** attempt)
            
            # Update metrics
            duration = (datetime.utcnow() - start_time).total_seconds()
            self.metrics["stage_latency"].labels(
                pipeline_name=context["pipeline_name"],
                stage_name=stage.name
            ).observe(duration)
            
            return result
            
        except Exception as e:
            logger.error(f"Stage execution failed: {e}")
            self.metrics["pipeline_errors"].labels(
                pipeline_name=context["pipeline_name"],
                error_type=f"stage_{stage.name}"
            ).inc()
            raise
    
    async def _validate_stage(self, stage: PipelineStage) -> bool:
        """Validate stage configuration"""
        try:
            # Validate endpoint
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{stage.endpoint}/health"
                ) as response:
                    return response.status == 200
        except Exception:
            return False
    
    async def _get_cached_result(
        self,
        request_id: str,
        stage_name: str
    ) -> Optional[Any]:
        """Get cached stage result"""
        try:
            cache_key = f"pipeline:{request_id}:{stage_name}"
            cached_data = await self.redis.get(cache_key)
            return json.loads(cached_data) if cached_data else None
        except Exception as e:
            logger.error(f"Cache retrieval failed: {e}")
            return None
    
    async def _cache_result(
        self,
        request_id: str,
        stage_name: str,
        result: Any
    ):
        """Cache stage result"""
        try:
            cache_key = f"pipeline:{request_id}:{stage_name}"
            await self.redis.setex(
                cache_key,
                self.config.cache_ttl,
                json.dumps(result)
            )
        except Exception as e:
            logger.error(f"Cache storage failed: {e}")
    
    async def _prepare_stage_input(
        self,
        stage: PipelineStage,
        context: Dict
    ) -> Dict:
        """Prepare input for stage execution"""
        return {
            "request_id": context["request_id"],
            "input": context["input"],
            "intermediate_results": context["intermediate_results"],
            "stage_name": stage.name,
            "timestamp": self.timestamp.isoformat(),
            "user": self.current_user
        }
    
    async def _prepare_final_result(self, context: Dict) -> Dict:
        """Prepare final pipeline result"""
        return {
            "request_id": context["request_id"],
            "status": "completed",
            "execution_time": (
                datetime.utcnow() - 
                datetime.fromisoformat(context["start_time"])
            ).total_seconds(),
            "results": context["intermediate_results"]
        }
    
    async def _cleanup_pipeline_data(
        self,
        pipeline_name: str,
        request_id: str
    ):
        """Cleanup pipeline execution data"""
        try:
            # Clean up cached results
            pipeline_keys = await self.redis.keys(
                f"pipeline:{request_id}:*"
            )
            if pipeline_keys:
                await self.redis.delete(*pipeline_keys)
            
        except Exception as e:
            logger.error(f"Pipeline cleanup failed: {e}")