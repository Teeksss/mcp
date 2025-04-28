from typing import Dict, List, Optional
import aiohttp
import asyncio
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class HealthCheck:
    def __init__(self):
        self.timestamp = datetime.utcnow()
        self.current_user = "Teeksss"
        self.services = {}
        self.dependencies = {}
    
    def register_service(
        self,
        name: str,
        check_url: str,
        dependencies: Optional[List[str]] = None,
        timeout: int = 5
    ):
        """Register service for health checking"""
        self.services[name] = {
            "check_url": check_url,
            "status": "unknown",
            "last_check": None,
            "timeout": timeout
        }
        
        if dependencies:
            self.dependencies[name] = dependencies
    
    async def check_health(self) -> Dict:
        """Perform health check on all registered services"""
        results = {}
        
        try:
            # Check all services
            for name, service in self.services.items():
                status = await self._check_service(
                    name,
                    service["check_url"],
                    service["timeout"]
                )
                
                # Update service status
                self.services[name]["status"] = status["status"]
                self.services[name]["last_check"] = self.timestamp
                
                results[name] = {
                    "status": status["status"],
                    "details": status.get("details", {}),
                    "last_check": self.timestamp.isoformat(),
                    "dependencies": await self._check_dependencies(name)
                }
            
            return {
                "status": "healthy" if all(
                    s["status"] == "healthy" 
                    for s in results.values()
                ) else "unhealthy",
                "timestamp": self.timestamp.isoformat(),
                "services": results
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "error",
                "timestamp": self.timestamp.isoformat(),
                "error": str(e)
            }
    
    async def _check_service(
        self,
        name: str,
        url: str,
        timeout: int
    ) -> Dict:
        """Check individual service health"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    timeout=timeout
                ) as response:
                    if response.status == 200:
                        return {
                            "status": "healthy",
                            "details": await response.json()
                        }
                    else:
                        return {
                            "status": "unhealthy",
                            "details": {
                                "status_code": response.status,
                                "reason": response.reason
                            }
                        }
                        
        except asyncio.TimeoutError:
            return {
                "status": "unhealthy",
                "details": {"error": "timeout"}
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "details": {"error": str(e)}
            }
    
    async def _check_dependencies(self, service_name: str) -> Dict:
        """Check service dependencies"""
        if service_name not in self.dependencies:
            return {}
        
        results = {}
        for dep in self.dependencies[service_name]:
            if dep in self.services:
                results[dep] = {
                    "status": self.services[dep]["status"],
                    "last_check": self.services[dep]["last_check"].isoformat()
                    if self.services[dep]["last_check"]
                    else None
                }
        
        return results