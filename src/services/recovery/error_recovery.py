from typing import Dict, List, Optional, Union, Callable
import asyncio
from datetime import datetime, timedelta
import logging
from prometheus_client import Counter, Gauge, Histogram
import traceback
import json
from dataclasses import dataclass
import sys
from enum import Enum

logger = logging.getLogger(__name__)

class ErrorSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ErrorCategory(Enum):
    SYSTEM = "system"
    APPLICATION = "application"
    NETWORK = "network"
    DATABASE = "database"
    SECURITY = "security"
    MODEL = "model"
    RESOURCE = "resource"

@dataclass
class RecoveryConfig:
    max_retries: int = 3
    retry_delay: int = 5
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: int = 300
    fallback_enabled: bool = True
    auto_recovery_enabled: bool = True
    health_check_interval: int = 60
    error_expiry: int = 3600
    snapshot_interval: int = 300

class ErrorRecoverySystem:
    def __init__(
        self,
        config: RecoveryConfig
    ):
        self.config = config
        self.metrics = self._setup_metrics()
        
        # Error tracking
        self.error_history = {}
        self.circuit_breakers = {}
        self.recovery_states = {}
        self.snapshots = {}
        
        # Current context
        self.timestamp = datetime.utcnow()
        self.current_user = "Teeksss"
        
        # Start recovery monitoring
        asyncio.create_task(self._monitor_system_health())
    
    def _setup_metrics(self) -> Dict:
        """Initialize recovery metrics"""
        return {
            "errors_total": Counter(
                "recovery_errors_total",
                "Total errors encountered",
                ["severity", "category"]
            ),
            "recovery_attempts": Counter(
                "recovery_attempts_total",
                "Total recovery attempts",
                ["status", "strategy"]
            ),
            "circuit_breaker_trips": Counter(
                "circuit_breaker_trips_total",
                "Total circuit breaker trips",
                ["component"]
            ),
            "recovery_time": Histogram(
                "recovery_time_seconds",
                "Time taken for recovery",
                ["strategy"]
            ),
            "active_failures": Gauge(
                "active_failures_total",
                "Current number of active failures",
                ["severity"]
            )
        }
    
    async def handle_error(
        self,
        error: Exception,
        context: Dict,
        severity: ErrorSeverity,
        category: ErrorCategory
    ) -> Dict:
        """Handle and recover from error"""
        try:
            error_id = self._generate_error_id(error, context)
            
            # Log error
            await self._log_error(
                error_id,
                error,
                context,
                severity,
                category
            )
            
            # Update metrics
            self.metrics["errors_total"].labels(
                severity=severity.value,
                category=category.value
            ).inc()
            
            # Check circuit breaker
            if await self._should_break_circuit(
                context.get("component_id"),
                category
            ):
                await self._trip_circuit_breaker(
                    context.get("component_id")
                )
                return {
                    "status": "circuit_breaker_tripped",
                    "error_id": error_id
                }
            
            # Attempt recovery
            recovery_result = await self._attempt_recovery(
                error_id,
                error,
                context,
                severity,
                category
            )
            
            # Update state
            await self._update_recovery_state(
                error_id,
                recovery_result
            )
            
            return {
                "status": recovery_result["status"],
                "error_id": error_id,
                "recovery": recovery_result
            }
            
        except Exception as e:
            logger.error(f"Error handling failed: {e}")
            raise
    
    async def register_fallback(
        self,
        component_id: str,
        fallback_handler: Callable
    ):
        """Register fallback handler for component"""
        try:
            self.recovery_states[component_id] = {
                "fallback_handler": fallback_handler,
                "last_updated": self.timestamp.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Fallback registration failed: {e}")
            raise
    
    async def create_snapshot(
        self,
        component_id: str,
        state: Dict
    ):
        """Create system state snapshot"""
        try:
            snapshot_id = f"snapshot_{self.timestamp.strftime('%Y%m%d_%H%M%S')}"
            
            self.snapshots[snapshot_id] = {
                "component_id": component_id,
                "state": state,
                "timestamp": self.timestamp.isoformat()
            }
            
            # Cleanup old snapshots
            await self._cleanup_old_snapshots()
            
            return snapshot_id
            
        except Exception as e:
            logger.error(f"Snapshot creation failed: {e}")
            raise
    
    async def restore_snapshot(
        self,
        snapshot_id: str
    ) -> Dict:
        """Restore system from snapshot"""
        try:
            if snapshot_id not in self.snapshots:
                raise RecoveryException("Snapshot not found")
            
            snapshot = self.snapshots[snapshot_id]
            
            # Perform restoration
            await self._restore_component_state(
                snapshot["component_id"],
                snapshot["state"]
            )
            
            return {
                "status": "restored",
                "snapshot_id": snapshot_id,
                "timestamp": self.timestamp.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Snapshot restoration failed: {e}")
            raise
    
    async def _attempt_recovery(
        self,
        error_id: str,
        error: Exception,
        context: Dict,
        severity: ErrorSeverity,
        category: ErrorCategory
    ) -> Dict:
        """Attempt error recovery"""
        try:
            start_time = self.timestamp
            
            # Get recovery strategy
            strategy = await self._determine_recovery_strategy(
                error,
                context,
                severity,
                category
            )
            
            # Execute strategy
            for attempt in range(self.config.max_retries):
                try:
                    result = await self._execute_recovery_strategy(
                        strategy,
                        context
                    )
                    
                    # Update metrics
                    self.metrics["recovery_attempts"].labels(
                        status="success",
                        strategy=strategy["type"]
                    ).inc()
                    
                    self.metrics["recovery_time"].labels(
                        strategy=strategy["type"]
                    ).observe(
                        (datetime.utcnow() - start_time).total_seconds()
                    )
                    
                    return {
                        "status": "recovered",
                        "strategy": strategy,
                        "attempt": attempt + 1,
                        "result": result
                    }
                    
                except Exception as recovery_error:
                    logger.warning(
                        f"Recovery attempt {attempt + 1} failed: {recovery_error}"
                    )
                    
                    if attempt < self.config.max_retries - 1:
                        await asyncio.sleep(
                            self.config.retry_delay * (attempt + 1)
                        )
            
            # All attempts failed
            self.metrics["recovery_attempts"].labels(
                status="failure",
                strategy=strategy["type"]
            ).inc()
            
            # Try fallback if enabled
            if self.config.fallback_enabled:
                fallback_result = await self._execute_fallback(
                    context
                )
                return {
                    "status": "fallback",
                    "strategy": strategy,
                    "attempts": self.config.max_retries,
                    "fallback_result": fallback_result
                }
            
            return {
                "status": "failed",
                "strategy": strategy,
                "attempts": self.config.max_retries
            }
            
        except Exception as e:
            logger.error(f"Recovery attempt failed: {e}")
            raise
    
    async def _determine_recovery_strategy(
        self,
        error: Exception,
        context: Dict,
        severity: ErrorSeverity,
        category: ErrorCategory
    ) -> Dict:
        """Determine appropriate recovery strategy"""
        strategies = {
            ErrorCategory.SYSTEM: self._get_system_recovery_strategy,
            ErrorCategory.APPLICATION: self._get_application_recovery_strategy,
            ErrorCategory.NETWORK: self._get_network_recovery_strategy,
            ErrorCategory.DATABASE: self._get_database_recovery_strategy,
            ErrorCategory.SECURITY: self._get_security_recovery_strategy,
            ErrorCategory.MODEL: self._get_model_recovery_strategy,
            ErrorCategory.RESOURCE: self._get_resource_recovery_strategy
        }
        
        strategy_func = strategies.get(
            category,
            self._get_default_recovery_strategy
        )
        
        return await strategy_func(
            error,
            context,
            severity
        )
    
    async def _execute_recovery_strategy(
        self,
        strategy: Dict,
        context: Dict
    ) -> Dict:
        """Execute recovery strategy"""
        if strategy["type"] == "restart":
            return await self._restart_component(
                context["component_id"]
            )
        
        elif strategy["type"] == "rollback":
            return await self._rollback_component(
                context["component_id"],
                strategy["version"]
            )
        
        elif strategy["type"] == "scale":
            return await self._scale_component(
                context["component_id"],
                strategy["replicas"]
            )
        
        elif strategy["type"] == "failover":
            return await self._failover_component(
                context["component_id"],
                strategy["target"]
            )
        
        elif strategy["type"] == "repair":
            return await self._repair_component(
                context["component_id"],
                strategy["repairs"]
            )
        
        raise RecoveryException(f"Unknown strategy type: {strategy['type']}")
    
    async def _execute_fallback(
        self,
        context: Dict
    ) -> Dict:
        """Execute fallback handler"""
        component_id = context.get("component_id")
        
        if component_id in self.recovery_states:
            handler = self.recovery_states[component_id]["fallback_handler"]
            
            try:
                result = await handler(context)
                return {
                    "status": "success",
                    "result": result
                }
            except Exception as e:
                logger.error(f"Fallback execution failed: {e}")
                return {
                    "status": "failed",
                    "error": str(e)
                }
        
        return {"status": "no_fallback"}
    
    async def _monitor_system_health(self):
        """Monitor system health and perform maintenance"""
        while True:
            try:
                # Check circuit breakers
                await self._check_circuit_breakers()
                
                # Create system snapshots
                if self.config.auto_recovery_enabled:
                    await self._create_system_snapshot()
                
                # Cleanup old errors
                await self._cleanup_old_errors()
                
            except Exception as e:
                logger.error(f"Health monitoring failed: {e}")
            
            await asyncio.sleep(self.config.health_check_interval)
    
    def _generate_error_id(
        self,
        error: Exception,
        context: Dict
    ) -> str:
        """Generate unique error ID"""
        error_info = {
            "type": error.__class__.__name__,
            "message": str(error),
            "context": context,
            "timestamp": self.timestamp.isoformat()
        }
        
        return f"error_{hash(json.dumps(error_info))}"
    
    async def _should_break_circuit(
        self,
        component_id: str,
        category: ErrorCategory
    ) -> bool:
        """Check if circuit should be broken"""
        if not component_id:
            return False
        
        recent_errors = [
            e for e in self.error_history.values()
            if (
                e["component_id"] == component_id and
                e["category"] == category.value and
                (
                    self.timestamp - datetime.fromisoformat(e["timestamp"])
                ).total_seconds() < self.config.circuit_breaker_timeout
            )
        ]
        
        return len(recent_errors) >= self.config.circuit_breaker_threshold

class RecoveryException(Exception):
    """Custom recovery exception"""
    pass