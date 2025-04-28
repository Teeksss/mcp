from datetime import datetime
import asyncio
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from dataclasses import dataclass
import logging
from prometheus_client import Histogram, Gauge
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

logger = logging.getLogger(__name__)

@dataclass
class AnalyzerConfig:
    analysis_interval: int = 3600  # 1 hour
    min_samples: int = 100
    performance_threshold: float = 0.8
    latency_threshold: float = 1.0  # seconds
    memory_threshold: float = 0.85  # 85% usage

class ModelPerformanceAnalyzer:
    def __init__(self, config: AnalyzerConfig):
        self.config = config
        self.metrics = self._setup_metrics()
        self.analysis_task = None
        self.performance_history = {}
        
        # Current context
        self.timestamp = datetime.utcnow()
        self.current_user = "Teeksss"
    
    def _setup_metrics(self) -> Dict:
        """Initialize performance metrics"""
        return {
            "inference_time": Histogram(
                "model_inference_seconds",
                "Model inference time",
                ["model_name", "version"]
            ),
            "memory_usage": Gauge(
                "model_memory_usage_bytes",
                "Model memory usage",
                ["model_name", "version"]
            ),
            "accuracy": Gauge(
                "model_accuracy",
                "Model accuracy",
                ["model_name", "version"]
            ),
            "throughput": Gauge(
                "model_throughput",
                "Model throughput (requests/second)",
                ["model_name", "version"]
            )
        }
    
    async def start(self):
        """Start performance analysis"""
        self.analysis_task = asyncio.create_task(
            self._analysis_loop()
        )
    
    async def stop(self):
        """Stop performance analysis"""
        if self.analysis_task:
            self.analysis_task.cancel()
            try:
                await self.analysis_task
            except asyncio.CancelledError:
                pass
    
    async def record_inference(
        self,
        model_name: str,
        version: str,
        input_data: Dict,
        output_data: Dict,
        duration: float,
        memory_used: int
    ):
        """Record model inference statistics"""
        try:
            # Update metrics
            self.metrics["inference_time"].labels(
                model_name=model_name,
                version=version
            ).observe(duration)
            
            self.metrics["memory_usage"].labels(
                model_name=model_name,
                version=version
            ).set(memory_used)
            
            # Store performance data
            key = f"{model_name}:{version}"
            if key not in self.performance_history:
                self.performance_history[key] = []
            
            self.performance_history[key].append({
                "timestamp": self.timestamp,
                "duration": duration,
                "memory_used": memory_used,
                "input_data": input_data,
                "output_data": output_data
            })
            
            # Trim history if too long
            if len(self.performance_history[key]) > 1000:
                self.performance_history[key] = self.performance_history[key][-1000:]
            
        except Exception as e:
            logger.error(f"Error recording inference: {e}")
    
    async def analyze_performance(
        self,
        model_name: str,
        version: str
    ) -> Dict:
        """Analyze model performance"""
        try:
            key = f"{model_name}:{version}"
            if key not in self.performance_history:
                return {}
            
            history = self.performance_history[key]
            if len(history) < self.config.min_samples:
                return {}
            
            # Calculate performance metrics
            durations = [h["duration"] for h in history]
            memory_usage = [h["memory_used"] for h in history]
            
            # Calculate throughput
            total_time = sum(durations)
            throughput = len(durations) / total_time if total_time > 0 else 0
            
            # Update throughput metric
            self.metrics["throughput"].labels(
                model_name=model_name,
                version=version
            ).set(throughput)
            
            return {
                "latency": {
                    "mean": np.mean(durations),
                    "p50": np.percentile(durations, 50),
                    "p95": np.percentile(durations, 95),
                    "p99": np.percentile(durations, 99)
                },
                "memory": {
                    "mean": np.mean(memory_usage),
                    "max": max(memory_usage),
                    "min": min(memory_usage)
                },
                "throughput": throughput,
                "samples": len(history),
                "optimization_needed": await self._check_optimization_needed(
                    durations,
                    memory_usage
                )
            }
            
        except Exception as e:
            logger.error(f"Performance analysis failed: {e}")
            return {}
    
    async def _check_optimization_needed(
        self,
        durations: List[float],
        memory_usage: List[int]
    ) -> Tuple[bool, str]:
        """Check if model needs optimization"""
        reasons = []
        
        # Check latency
        mean_latency = np.mean(durations)
        if mean_latency > self.config.latency_threshold:
            reasons.append(
                f"High latency: {mean_latency:.2f}s > "
                f"{self.config.latency_threshold}s"
            )
        
        # Check memory usage
        max_memory = max(memory_usage)
        if max_memory > self.config.memory_threshold:
            reasons.append(
                f"High memory usage: {max_memory:.2%} > "
                f"{self.config.memory_threshold:.2%}"
            )
        
        return bool(reasons), ", ".join(reasons)
    
    async def suggest_optimizations(
        self,
        model_name: str,
        version: str
    ) -> List[Dict]:
        """Suggest possible optimizations"""
        try:
            analysis = await self.analyze_performance(model_name, version)
            if not analysis:
                return []
            
            suggestions = []
            
            # Check latency optimizations
            if analysis["latency"]["mean"] > self.config.latency_threshold:
                suggestions.append({
                    "type": "latency",
                    "priority": "high",
                    "suggestion": "Consider quantization or pruning",
                    "details": {
                        "current_latency": analysis["latency"]["mean"],
                        "target_latency": self.config.latency_threshold
                    }
                })
            
            # Check memory optimizations
            if analysis["memory"]["mean"] > self.config.memory_threshold:
                suggestions.append({
                    "type": "memory",
                    "priority": "high",
                    "suggestion": "Consider model distillation or compression",
                    "details": {
                        "current_memory": analysis["memory"]["mean"],
                        "target_memory": self.config.memory_threshold
                    }
                })
            
            # Check throughput optimizations
            if analysis["throughput"] < 10:  # Less than 10 requests/second
                suggestions.append({
                    "type": "throughput",
                    "priority": "medium",
                    "suggestion": "Consider batch processing or model parallelization",
                    "details": {
                        "current_throughput": analysis["throughput"],
                        "target_throughput": 10
                    }
                })
            
            return suggestions
            
        except Exception as e:
            logger.error(f"Optimization suggestion failed: {e}")
            return []
    
    async def _analysis_loop(self):
        """Periodic performance analysis loop"""
        while True:
            try:
                for key in self.performance_history:
                    model_name, version = key.split(":")
                    analysis = await self.analyze_performance(
                        model_name,
                        version
                    )
                    
                    if analysis.get("optimization_needed"):
                        suggestions = await self.suggest_optimizations(
                            model_name,
                            version
                        )
                        if suggestions:
                            logger.warning(
                                f"Model {model_name}:{version} needs optimization: "
                                f"{suggestions}"
                            )
                
            except Exception as e:
                logger.error(f"Analysis loop error: {e}")
            
            await asyncio.sleep(self.config.analysis_interval)