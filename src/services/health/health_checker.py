from typing import Dict, List
import asyncio
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class HealthChecker:
    def __init__(
        self,
        components: Dict,
        monitoring
    ):
        self.components = components
        self.monitoring = monitoring
        self.status = {}
        
        # Current context
        self.timestamp = datetime.utcnow()
        self.current_user = "Teeksss"
    
    async def check_health(self) -> Dict:
        """Check health of all components"""
        results = {}
        
        for name, component in self.components.items():
            try:
                # Check component health
                status = await component.health_check()
                
                # Update status
                results[name] = {
                    "status": "healthy" if status else "unhealthy",
                    "last_check": self.timestamp.isoformat(),
                    "metrics": component.metrics if hasattr(component, "metrics") else {}
                }
                
                # Record metrics
                await self.monitoring.record_health_check(
                    component=name,
                    status=results[name]["status"],
                    metrics=results[name]["metrics"]
                )
                
            except Exception as e:
                logger.error(f"Health check failed for {name}: {e}")
                results[name] = {
                    "status": "error",
                    "error": str(e),
                    "last_check": self.timestamp.isoformat()
                }
        
        self.status = results
        return results
    
    async def get_system_status(self) -> Dict:
        """Get overall system status"""
        health_status = await self.check_health()
        
        # Calculate system health
        total_components = len(health_status)
        healthy_components = sum(
            1 for status in health_status.values()
            if status["status"] == "healthy"
        )
        
        system_health = healthy_components / total_components
        
        return {
            "status": "healthy" if system_health >= 0.8 else "degraded",
            "health_percentage": system_health * 100,
            "timestamp": self.timestamp.isoformat(),
            "components": health_status
        }