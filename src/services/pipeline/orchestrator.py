from typing import Dict, List, Optional, Union
import asyncio
from datetime import datetime
import logging
from prometheus_client import Counter, Histogram
from src.services.intelligence.model_manager import ModelManager
from src.services.intelligence.rag_enhancer import RAGEnhancer
from src.services.optimization.performance_optimizer import PerformanceOptimizer
from src.services.recovery.error_recovery import ErrorRecoverySystem

logger = logging.getLogger(__name__)

class PipelineOrchestrator:
    def __init__(
        self,
        model_manager: ModelManager,
        rag_enhancer: RAGEnhancer,
        performance_optimizer: PerformanceOptimizer,
        error_recovery: ErrorRecoverySystem
    ):
        self.model_manager = model_manager
        self.rag_enhancer = rag_enhancer
        self.performance_optimizer = performance_optimizer
        self.error_recovery = error_recovery
        self.metrics = self._setup_metrics()
        
        # Current context
        self.timestamp = datetime.utcnow()
        self.current_user = "Teeksss"
    
    def _setup_metrics(self) -> Dict:
        return {
            "pipeline_executions": Counter(
                "pipeline_executions_total",
                "Total pipeline executions",
                ["status", "pipeline_type"]
            ),
            "pipeline_duration": Histogram(
                "pipeline_duration_seconds",
                "Pipeline execution duration",
                ["pipeline_type"]
            )
        }
    
    async def process_query(
        self,
        query: str,
        pipeline_config: Dict,
        context: Optional[Dict] = None
    ) -> Dict:
        """Process query through the pipeline"""
        try:
            start_time = self.timestamp
            
            # Enhance query with RAG
            enhanced_query = await self.rag_enhancer.enhance_query(
                query,
                context
            )
            
            # Select optimal model
            model = await self.model_manager.select_optimal_model(
                enhanced_query["enhanced_query"],
                pipeline_config
            )
            
            # Optimize performance
            optimization = await self.performance_optimizer.optimize_performance(
                model.id,
                {
                    "query_length": len(query),
                    "context_size": len(enhanced_query["contexts"]),
                    "model_size": model.config["size"]
                }
            )
            
            # Execute model
            try:
                result = await self.model_manager.execute_model(
                    model.id,
                    enhanced_query["enhanced_query"],
                    optimization["results"]
                )
            except Exception as e:
                # Handle error with recovery system
                recovery_result = await self.error_recovery.handle_error(
                    e,
                    {
                        "component_id": model.id,
                        "query": enhanced_query["enhanced_query"],
                        "context": context
                    },
                    severity=ErrorSeverity.HIGH,
                    category=ErrorCategory.MODEL
                )
                
                if recovery_result["status"] == "recovered":
                    result = recovery_result["result"]
                else:
                    raise
            
            # Update metrics
            self.metrics["pipeline_executions"].labels(
                status="success",
                pipeline_type="query"
            ).inc()
            
            self.metrics["pipeline_duration"].labels(
                pipeline_type="query"
            ).observe(
                (datetime.utcnow() - start_time).total_seconds()
            )
            
            return {
                "result": result,
                "enhanced_query": enhanced_query,
                "model": model.name,
                "optimization": optimization,
                "processing_time": (
                    datetime.utcnow() - start_time
                ).total_seconds()
            }
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            self.metrics["pipeline_executions"].labels(
                status="failure",
                pipeline_type="query"
            ).inc()
            raise