from datetime import datetime
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
import traceback
import asyncio
from prometheus_client import Counter, Histogram
import logging
import sentry_sdk
from functools import wraps

logger = logging.getLogger(__name__)

@dataclass
class ErrorConfig:
    max_retries: int = 3
    retry_delay: int = 5
    error_threshold: int = 10
    recovery_timeout: int = 300
    alert_threshold: int = 5

class ErrorType:
    VALIDATION_ERROR = "validation_error"
    PROCESSING_ERROR = "processing_error"
    SYSTEM_ERROR = "system_error"
    NETWORK_ERROR = "network_error"
    DATABASE_ERROR = "database_error"
    TIMEOUT_ERROR = "timeout_error"

class ErrorManager:
    def __init__(self, config: ErrorConfig):
        self.config = config
        self.metrics = self._setup_metrics()
        self.error_counts = {}
        self.recovery_handlers = {}
        self.alert_handlers = []
        
        # Current context
        self.timestamp = datetime.utcnow()
        self.current_user = "Teeksss"
    
    def _setup_metrics(self) -> Dict:
        """Initialize error metrics"""
        return {
            "errors": Counter(
                "error_total",
                "Total errors by type",
                ["error_type", "component"]
            ),
            "recovery_time": Histogram(
                "error_recovery_seconds",
                "Time taken to recover from errors",
                ["error_type"]
            ),
            "retry_count": Counter(
                "error_retry_total",
                "Total retry attempts",
                ["error_type"]
            )
        }
    
    def register_recovery_handler(
        self,
        error_type: str,
        handler: Callable
    ):
        """Register recovery handler for error type"""
        self.recovery_handlers[error_type] = handler
    
    def register_alert_handler(self, handler: Callable):
        """Register alert handler"""
        self.alert_handlers.append(handler)
    
    async def handle_error(
        self,
        error: Exception,
        component: str,
        context: Optional[Dict] = None
    ):
        """Handle and recover from error"""
        try:
            # Determine error type
            error_type = self._classify_error(error)
            
            # Update metrics
            self.metrics["errors"].labels(
                error_type=error_type,
                component=component
            ).inc()
            
            # Track error count
            key = f"{error_type}:{component}"
            self.error_counts[key] = self.error_counts.get(key, 0) + 1
            
            # Check if we need to alert
            if self.error_counts[key] >= self.config.alert_threshold:
                await self._send_alerts(error, error_type, component, context)
            
            # Attempt recovery
            if error_type in self.recovery_handlers:
                start_time = datetime.utcnow()
                
                try:
                    await self.recovery_handlers[error_type](error, context)
                    
                    # Update recovery time metric
                    recovery_time = (datetime.utcnow() - start_time).total_seconds()
                    self.metrics["recovery_time"].labels(
                        error_type=error_type
                    ).observe(recovery_time)
                    
                except Exception as recovery_error:
                    logger.error(
                        f"Recovery failed for {error_type}: {recovery_error}"
                    )
                    # Send additional alert for recovery failure
                    await self._send_alerts(
                        recovery_error,
                        f"{error_type}_recovery_failed",
                        component,
                        context
                    )
            
            # Log error details
            logger.error(
                f"Error in {component}: {str(error)}",
                extra={
                    "error_type": error_type,
                    "stack_trace": traceback.format_exc(),
                    "context": context,
                    "timestamp": self.timestamp.isoformat(),
                    "user": self.current_user
                }
            )
            
            # Send to Sentry
            sentry_sdk.capture_exception(error)
            
        except Exception as e:
            logger.error(f"Error handling failed: {e}")
    
    def retry_on_error(
        self,
        retryable_errors: tuple = (Exception,),
        max_retries: Optional[int] = None,
        delay: Optional[int] = None
    ):
        """Decorator for automatic retry on error"""
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                retries = 0
                max_attempts = max_retries or self.config.max_retries
                retry_delay = delay or self.config.retry_delay
                
                while retries < max_attempts:
                    try:
                        return await func(*args, **kwargs)
                    except retryable_errors as e:
                        retries += 1
                        self.metrics["retry_count"].labels(
                            error_type=self._classify_error(e)
                        ).inc()
                        
                        if retries == max_attempts:
                            raise
                        
                        await asyncio.sleep(retry_delay * (2 ** (retries - 1)))
                
                raise Exception("Maximum retries exceeded")
            
            return wrapper
        return decorator
    
    def _classify_error(self, error: Exception) -> str:
        """Classify error type"""
        if isinstance(error, ValueError):
            return ErrorType.VALIDATION_ERROR
        elif isinstance(error, TimeoutError):
            return ErrorType.TIMEOUT_ERROR
        elif isinstance(error, ConnectionError):
            return ErrorType.NETWORK_ERROR
        elif "database" in str(error).lower():
            return ErrorType.DATABASE_ERROR
        else:
            return ErrorType.SYSTEM_ERROR
    
    async def _send_alerts(
        self,
        error: Exception,
        error_type: str,
        component: str,
        context: Optional[Dict]
    ):
        """Send alerts to registered handlers"""
        alert_data = {
            "error": str(error),
            "error_type": error_type,
            "component": component,
            "stack_trace": traceback.format_exc(),
            "context": context,
            "timestamp": self.timestamp.isoformat(),
            "user": self.current_user
        }
        
        for handler in self.alert_handlers:
            try:
                await handler(alert_data)
            except Exception as e:
                logger.error(f"Alert handler failed: {e}")
    
    async def reset_error_counts(self):
        """Reset error counts after recovery period"""
        self.error_counts = {}
        logger.info("Reset error counts")