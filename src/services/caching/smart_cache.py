from typing import Dict, Any, Optional, List, Tuple
import redis
import pickle
import json
import hashlib
from datetime import datetime, timedelta
import asyncio
import logging
from prometheus_client import Counter, Histogram, Gauge

logger = logging.getLogger(__name__)

class SmartCache:
    def __init__(
        self,
        redis_client: redis.Redis,
        default_ttl: int = 3600,
        max_memory_mb: int = 1024
    ):
        self.redis = redis_client
        self.default_ttl = default_ttl
        self.max_memory_mb = max_memory_mb
        self.metrics = self._setup_metrics()
        
        # Cache statistics
        self.access_patterns = {}
        self.size_tracking = {}
        
        # Current context
        self.timestamp = datetime.utcnow()
        self.current_user = "Teeksss"
    
    def _setup_metrics(self) -> Dict:
        """Initialize cache metrics"""
        return {
            "cache_hits": Counter(
                "cache_hits_total",
                "Number of cache hits",
                ["cache_type"]
            ),
            "cache_misses": Counter(
                "cache_misses_total",
                "Number of cache misses",
                ["cache_type"]
            ),
            "cache_size": Gauge(
                "cache_size_bytes",
                "Current cache size",
                ["cache_type"]
            ),
            "eviction_count": Counter(
                "cache_evictions_total",
                "Number of cache evictions",
                ["reason"]
            ),
            "operation_latency": Histogram(
                "cache_operation_latency_seconds",
                "Cache operation latency",
                ["operation"]
            )
        }
    
    async def get(
        self,
        key: str,
        cache_type: str = "default"
    ) -> Optional[Any]:
        """Get value from cache"""
        try:
            start_time = self.timestamp
            
            # Generate cache key
            full_key = self._generate_key(key, cache_type)
            
            # Get from Redis
            value = await self.redis.get(full_key)
            
            if value:
                # Update access patterns
                await self._update_access_pattern(full_key)
                
                # Update metrics
                self.metrics["cache_hits"].labels(
                    cache_type=cache_type
                ).inc()
                
                # Deserialize
                result = pickle.loads(value)
                
                # Record latency
                self.metrics["operation_latency"].labels(
                    operation="get"
                ).observe(
                    (datetime.utcnow() - start_time).total_seconds()
                )
                
                return result
            
            self.metrics["cache_misses"].labels(
                cache_type=cache_type
            ).inc()
            return None
            
        except Exception as e:
            logger.error(f"Cache get failed: {e}")
            return None
    
    async def set(
        self,
        key: str,
        value: Any,
        cache_type: str = "default",
        ttl: Optional[int] = None,
        metadata: Optional[Dict] = None
    ):
        """Set value in cache"""
        try:
            start_time = self.timestamp
            
            # Generate cache key
            full_key = self._generate_key(key, cache_type)
            
            # Serialize value
            serialized = pickle.dumps(value)
            value_size = len(serialized)
            
            # Check memory limits
            await self._ensure_memory_available(value_size)
            
            # Store value
            ttl = ttl or self.default_ttl
            await self.redis.setex(
                full_key,
                ttl,
                serialized
            )
            
            # Store metadata
            if metadata:
                meta_key = f"{full_key}:meta"
                await self.redis.setex(
                    meta_key,
                    ttl,
                    json.dumps(metadata)
                )
            
            # Update size tracking
            self.size_tracking[full_key] = {
                "size": value_size,
                "timestamp": self.timestamp.isoformat()
            }
            
            # Update metrics
            self.metrics["cache_size"].labels(
                cache_type=cache_type
            ).inc(value_size)
            
            # Record latency
            self.metrics["operation_latency"].labels(
                operation="set"
            ).observe(
                (datetime.utcnow() - start_time).total_seconds()
            )
            
        except Exception as e:
            logger.error(f"Cache set failed: {e}")
            raise
    
    async def invalidate(
        self,
        key: str,
        cache_type: str = "default"
    ):
        """Invalidate cache entry"""
        try:
            # Generate cache key
            full_key = self._generate_key(key, cache_type)
            
            # Remove value and metadata
            await self.redis.delete(full_key)
            await self.redis.delete(f"{full_key}:meta")
            
            # Update size tracking
            if full_key in self.size_tracking:
                size = self.size_tracking[full_key]["size"]
                self.metrics["cache_size"].labels(
                    cache_type=cache_type
                ).dec(size)
                del self.size_tracking[full_key]
            
            # Update metrics
            self.metrics["eviction_count"].labels(
                reason="manual"
            ).inc()
            
        except Exception as e:
            logger.error(f"Cache invalidation failed: {e}")
            raise
    
    async def _ensure_memory_available(self, required_size: int):
        """Ensure sufficient memory is available"""
        try:
            current_size = sum(
                info["size"] for info in self.size_tracking.values()
            )
            
            if current_size + required_size > self.max_memory_mb * 1024 * 1024:
                # Need to evict items
                await self._evict_items(required_size)
            
        except Exception as e:
            logger.error(f"Memory management failed: {e}")
            raise
    
    async def _evict_items(self, required_size: int):
        """Evict items from cache"""
        try:
            # Get sorted items by priority
            items = await self._get_items_by_priority()
            
            freed_space = 0
            for item in items:
                # Stop if we've freed enough space
                if freed_space >= required_size:
                    break
                
                # Evict item
                await self.invalidate(
                    item["key"],
                    item["cache_type"]
                )
                
                freed_space += item["size"]
                
                # Update metrics
                self.metrics["eviction_count"].labels(
                    reason="memory_pressure"
                ).inc()
            
        except Exception as e:
            logger.error(f"Cache eviction failed: {e}")
            raise
    
    async def _get_items_by_priority(self) -> List[Dict]:
        """Get cache items sorted by eviction priority"""
        items = []
        
        for key, info in self.size_tracking.items():
            # Calculate priority score
            access_count = self.access_patterns.get(key, {}).get(
                "count",
                0
            )
            last_access = datetime.fromisoformat(
                self.access_patterns.get(key, {}).get(
                    "last_access",
                    self.timestamp.isoformat()
                )
            )
            
            age = (self.timestamp - last_access).total_seconds()
            
            priority_score = (
                age / 3600 +  # Hours since last access
                1 / (access_count + 1)  # Inverse of access count
            )
            
            items.append({
                "key": key,
                "cache_type": key.split(":")[0],
                "size": info["size"],
                "priority_score": priority_score
            })
        
        # Sort by priority (highest first)
        items.sort(key=lambda x: x["priority_score"], reverse=True)
        return items
    
    async def _update_access_pattern(self, key: str):
        """Update access pattern statistics"""
        if key not in self.access_patterns:
            self.access_patterns[key] = {
                "count": 0,
                "first_access": self.timestamp.isoformat()
            }
        
        self.access_patterns[key].update({
            "count": self.access_patterns[key]["count"] + 1,
            "last_access": self.timestamp.isoformat()
        })
    
    def _generate_key(self, key: str, cache_type: str) -> str:
        """Generate full cache key"""
        return f"{cache_type}:{hashlib.md5(key.encode()).hexdigest()}"