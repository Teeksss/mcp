from typing import Dict, List, Optional
import asyncio
from datetime import datetime
import logging
from prometheus_client import Counter, Gauge, Histogram
import semver
import hashlib
import json

logger = logging.getLogger(__name__)

class ModelVersionManager:
    def __init__(self):
        self.metrics = self._setup_metrics()
        self.versions = {}
        self.active_versions = {}
        
        # Current context
        self.timestamp = datetime.utcnow()
        self.current_user = "Teeksss"
    
    def _setup_metrics(self) -> Dict:
        return {
            "version_updates": Counter(
                "model_version_updates_total",
                "Total model version updates",
                ["model_name", "update_type"]
            ),
            "active_versions": Gauge(
                "model_active_versions",
                "Number of active model versions",
                ["model_name"]
            ),
            "version_rollbacks": Counter(
                "model_version_rollbacks_total",
                "Total model version rollbacks",
                ["model_name", "reason"]
            )
        }
    
    async def create_version(
        self,
        model_name: str,
        version_data: Dict,
        metadata: Optional[Dict] = None
    ) -> Dict:
        """Create new model version"""
        try:
            # Generate version hash
            version_hash = self._generate_version_hash(version_data)
            
            # Create version info
            version_info = {
                "version": self._generate_version_number(model_name),
                "hash": version_hash,
                "data": version_data,
                "metadata": metadata or {},
                "created_at": self.timestamp.isoformat(),
                "created_by": self.current_user,
                "status": "created"
            }
            
            # Store version
            if model_name not in self.versions:
                self.versions[model_name] = []
            
            self.versions[model_name].append(version_info)
            
            # Update metrics
            self.metrics["version_updates"].labels(
                model_name=model_name,
                update_type="create"
            ).inc()
            
            return version_info
            
        except Exception as e:
            logger.error(f"Version creation failed: {e}")
            raise
    
    async def activate_version(
        self,
        model_name: str,
        version: str
    ) -> Dict:
        """Activate specific model version"""
        try:
            # Find version
            version_info = self._find_version(model_name, version)
            if not version_info:
                raise ValueError(f"Version {version} not found")
            
            # Update status
            version_info["status"] = "active"
            version_info["activated_at"] = self.timestamp.isoformat()
            version_info["activated_by"] = self.current_user
            
            # Update active versions
            self.active_versions[model_name] = version_info
            
            # Update metrics
            self.metrics["active_versions"].labels(
                model_name=model_name
            ).set(1)
            
            return version_info
            
        except Exception as e:
            logger.error(f"Version activation failed: {e}")
            raise
    
    async def rollback_version(
        self,
        model_name: str,
        version: str,
        reason: str
    ) -> Dict:
        """Rollback to previous version"""
        try:
            # Find version
            version_info = self._find_version(model_name, version)
            if not version_info:
                raise ValueError(f"Version {version} not found")
            
            # Update status
            current_active = self.active_versions.get(model_name)
            if current_active:
                current_active["status"] = "rolled_back"
                current_active["rolled_back_at"] = self.timestamp.isoformat()
                current_active["rolled_back_by"] = self.current_user
                current_active["rollback_reason"] = reason
            
            # Activate rollback version
            await self.activate_version(model_name, version)
            
            # Update metrics
            self.metrics["version_rollbacks"].labels(
                model_name=model_name,
                reason=reason
            ).inc()
            
            return self.active_versions[model_name]
            
        except Exception as e:
            logger.error(f"Version rollback failed: {e}")
            raise
    
    def _generate_version_hash(self, data: Dict) -> str:
        """Generate hash for version data"""
        return hashlib.sha256(
            json.dumps(data, sort_keys=True).encode()
        ).hexdigest()
    
    def _generate_version_number(self, model_name: str) -> str:
        """Generate semantic version number"""
        versions = self.versions.get(model_name, [])
        if not versions:
            return "1.0.0"
        
        latest = max(
            versions,
            key=lambda x: semver.VersionInfo.parse(x["version"])
        )
        
        current = semver.VersionInfo.parse(latest["version"])
        return str(current.bump_minor())
    
    def _find_version(
        self,
        model_name: str,
        version: str
    ) -> Optional[Dict]:
        """Find specific version info"""
        versions = self.versions.get(model_name, [])
        for v in versions:
            if v["version"] == version:
                return v
        return None