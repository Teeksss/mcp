from typing import Dict, List, Optional
import asyncio
from datetime import datetime
import logging
from prometheus_client import Histogram

logger = logging.getLogger(__name__)

class InferencePipeline:
    def __init__(
        self,
        cache,
        load_balancer,
        optimizer,
        error_recovery,
        version_manager,
        monitoring
    ):
        self.cache = cache
        self.load_balancer = load_balancer
        self.optimizer = optimizer
        self.error_recovery = error_recovery
        self.version_manager = version_manager
        self.monitoring = monitoring
        
        # Metrics
        self.inference_time = Histogram(
            "inference_pipeline_duration_seconds",
            "Time taken for complete inference pipeline",
            ["model_name", "version"]
        )
        
        # Current context
        self.timestamp = datetime.utcnow()
        self.current_user = "Teeksss"
    
    async def process(
        self,
        query: str,
        model_name: str,
        context: Optional[Dict] = None
    ) -> Dict:
        """Process inference request through pipeline"""
        start_time = self.timestamp
        
        try:
            # 1. Check cache
            cache_key = self._generate_cache_key(query, model_name, context)
            cached_result = await self.cache.get(cache_key)
            
            if cached_result:
                logger.info(f"Cache hit for query: {query[:50]}...")
                return cached_result
            
            # 2. Get active model version
            model_version = await self.version_manager.get_active_version(
                model_name
            )
            
            # 3. Load balance request
            node = await self.load_balancer.get_optimal_node(
                model_name,
                model_version["version"],
                context
            )
            
            # 4. Optimize performance
            optimization = await self.optimizer.optimize_performance(
                component_id=f"{model_name}_{model_version['version']}",
                performance_data={
                    "query_length": len(query),
                    "context_size": len(context) if context else 0,
                    "node_metrics": node.metrics
                }
            )
            
            # 5. Execute inference
            try:
                result = await node.execute(
                    query=query,
                    model_version=model_version,
                    optimization=optimization,
                    context=context
                )
            except Exception as e:
                # Handle error with recovery system
                recovery_result = await self.error_recovery.handle_error(
                    error=e,
                    context={
                        "component_id": f"{model_name}_{model_version['version']}",
                        "node_id": node.id,
                        "query": query,
                        "context": context
                    },
                    severity=ErrorSeverity.HIGH,
                    category=ErrorCategory.MODEL
                )
                
                if recovery_result["status"] == "recovered":
                    result = recovery_result["result"]
                else:
                    raise
            
            # 6. Cache result
            await self.cache.set(
                cache_key,
                result,
                ttl=self._calculate_cache_ttl(result)
            )
            
            # 7. Update metrics
            duration = (datetime.utcnow() - start_time).total_seconds()
            self.inference_time.labels(
                model_name=model_name,
                version=model_version["version"]
            ).observe(duration)
            
            # 8. Record telemetry
            await self.monitoring.record_inference(
                model_name=model_name,
                version=model_version["version"],
                duration=duration,
                node_id=node.id,
                optimization=optimization,
                result_metrics=result.get("metrics", {})
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Inference pipeline failed: {e}")
            raise
    
    def _generate_cache_key(
        self,
        query: str,
        model_name: str,
        context: Optional[Dict]
    ) -> str:
        """Generate cache key for query"""
        key_parts = [
            query,
            model_name,
            str(hash(str(context))) if context else "no_context"
        ]
        return ":".join(key_parts)
    
    def _calculate_cache_ttl(self, result: Dict) -> int:
        """Calculate cache TTL based on result"""
        base_ttl = 3600  # 1 hour
        
        # Adjust TTL based on result confidence
        confidence = result.get("confidence", 0.5)
        ttl_multiplier = max(0.5, min(2.0, confidence))
        
        return int(base_ttl * ttl_multiplier)