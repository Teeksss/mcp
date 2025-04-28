from typing import Any, Dict, Optional, List
import asyncio
from datetime import datetime, timedelta
import json
import zlib
from dataclasses import dataclass
import redis
import diskcache
from functools import lru_cache
import logging

logger = logging.getLogger(__name__)

@dataclass
class CacheConfig:
    redis_url: str
    local_cache_dir: str
    memory_cache_size: int = 1000
    disk_cache_size: int = 10 * 1024 * 1024 * 1024  # 10GB
    compression_level: int = 6
    default_ttl: int = 3600

class HierarchicalCache:
    def __init__(self, config: CacheConfig):
        self.config = config
        self.memory_cache = self._setup_memory_cache()
        self.disk_cache = self._setup_disk_cache()
        self.redis_cache = self._setup_redis_cache()
        
        # Statistics
        self.stats = {
            "memory_hits": 0,
            "disk_hits": 0,
            "redis_hits": 0,
            "misses": 0
        }
    
    def _setup_memory_cache(self):
        """Setup in-memory LRU cache"""
        return lru_cache(maxsize=self.config.memory_cache_size)
    
    def _setup_disk_cache(self):
        """Setup disk cache"""
        return diskcache.Cache(
            self.config.local_cache_dir,
            size_limit=self.config.disk_cache_size
        )
    
    def _setup_redis_cache(self):
        """Setup Redis cache"""
        return redis.Redis.from_url(self.config.redis_url)
    
    async def get(
        self,
        key: str,
        default: Any = None
    ) -> Optional[Any]:
        """Get value from cache hierarchy"""
        try:
            # Try memory cache first
            value = self.memory_cache(key)
            if value is not None:
                self.stats["memory_hits"] += 1
                return value
            
            # Try disk cache
            value = self.disk_cache.get(key)
            if value is not None:
                self.stats["disk_hits"] += 1
                # Promote to memory cache
                self.memory_cache(key)(value)
                return value
            
            # Try Redis cache
            value = await self.redis_cache.get(key)
            if value is not None:
                self.stats["redis_hits"] += 1
                value = self._decompress_value(value)
                # Promote to local caches
                self.memory_cache(key)(value)
                self.disk_cache.set(key, value)
                return value
            
            self.stats["misses"] += 1
            return default
            
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return default
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        tags: Optional[List[str]] = None
    ):
        """Set value in cache hierarchy"""
        try:
            # Set in memory cache
            self.memory_cache(key)(value)
            
            # Set in disk cache
            self.disk_cache.set(
                key,
                value,
                expire=ttl or self.config.default_ttl
            )
            
            # Set in Redis cache
            compressed_value = self._compress_value(value)
            await self.redis_cache.set(
                key,
                compressed_value,
                ex=ttl or self.config.default_ttl
            )
            
            # Store tags for easy invalidation
            if tags:
                for tag in tags:
                    await self.redis_cache.sadd(f"tag:{tag}", key)
            
        except Exception as e:
            logger.error(f"Cache set error: {e}")
    
    async def invalidate_by_tags(self, tags: List[str]):
        """Invalidate cache entries by tags"""
        try:
            # Get all keys for given tags
            keys = set()
            for tag in tags:
                tag_keys = await self.redis_cache.smembers(f"tag:{tag}")
                keys.update(tag_keys)
            
            # Invalidate from all cache levels
            for key in keys:
                await self.invalidate(key)
            
        except Exception as e:
            logger.error(f"Cache invalidation error: {e}")
    
    async def invalidate(self, key: str):
        """Invalidate cache entry from all levels"""
        try:
            # Remove from memory cache
            self.memory_cache.cache_clear()
            
            # Remove from disk cache
            self.disk_cache.delete(key)
            
            # Remove from Redis cache
            await self.redis_cache.delete(key)
            
        except Exception as e:
            logger.error(f"Cache invalidation error: {e}")
    
    def _compress_value(self, value: Any) -> bytes:
        """Compress value for storage"""
        serialized = json.dumps(value)
        return zlib.compress(
            serialized.encode(),
            level=self.config.compression_level
        )
    
    def _decompress_value(self, value: bytes) -> Any:
        """Decompress stored value"""
        decompressed = zlib.decompress(value)
        return json.loads(decompressed.decode())
    
    async def get_stats(self) -> Dict:
        """Get cache statistics"""
        total_requests = sum(self.stats.values())
        return {
            "total_requests": total_requests,
            "hit_rate": {
                "memory": self.stats["memory_hits"] / total_requests,
                "disk": self.stats["disk_hits"] / total_requests,
                "redis": self.stats["redis_hits"] / total_requests
            },
            "miss_rate": self.stats["misses"] / total_requests,
            "memory_cache_info": self.memory_cache.cache_info(),
            "disk_cache_size": self.disk_cache.volume(),
            "redis_info": await self.redis_cache.info()
        }