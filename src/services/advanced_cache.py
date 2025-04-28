import redis
from typing import Any, Optional, List, Dict
import pickle
import zlib
import asyncio
import logging
from datetime import datetime, timedelta
import json
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class CacheConfig:
    redis_url: str
    max_memory_mb: int = 1024
    eviction_policy: str = "volatile-lru"
    compression_level: int = 6
    default_ttl: int = 3600
    enable_persistence: bool = True
    max_connections: int = 10

class AdvancedCache:
    def __init__(self, config: CacheConfig):
        self.config = config
        self._pool = None
        self._client = None
        self.setup_redis()
    
    def setup_redis(self):
        try:
            self._pool = redis.ConnectionPool.from_url(
                self.config.redis_url,
                max_connections=self.config.max_connections
            )
            
            self._client = redis.Redis(
                connection_pool=self._pool,
                decode_responses=False
            )
            
            # Configure Redis
            self._client.config_set(
                "maxmemory",
                f"{self.config.max_memory_mb}mb"
            )
            self._client.config_set(
                "maxmemory-policy",
                self.config.eviction_policy
            )
            
            if self.config.enable_persistence:
                self._client.config_set("save", "900 1 300 10 60 10000")
            
            logger.info("Redis cache initialized successfully")
            
        except Exception as e:
            logger.error(f"Redis initialization failed: {e}")
            raise
    
    async def get(
        self,
        key: str,
        default: Any = None
    ) -> Optional[Any]:
        try:
            data = await self._client.get(key)
            if data is None:
                return default
            
            # Decompress and deserialize
            decompressed = zlib.decompress(data)
            return pickle.loads(decompressed)
            
        except Exception as e:
            logger.error(f"Cache get error for key {key}: {e}")
            return default
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        tags: Optional[List[str]] = None
    ) -> bool:
        try:
            # Serialize and compress
            serialized = pickle.dumps(value)
            compressed = zlib.compress(
                serialized,
                level=self.config.compression_level
            )
            
            # Store the main data
            success = await self._client.set(
                key,
                compressed,
                ex=ttl or self.config.default_ttl
            )
            
            # Store tags for easier lookup
            if tags:
                for tag in tags:
                    await self._client.sadd(f"tag:{tag}", key)
            
            return bool(success)
            
        except Exception as e:
            logger.error(f"Cache set error for key {key}: {e}")
            return False
    
    async def delete_by_pattern(self, pattern: str) -> int:
        try:
            keys = await self._client.keys(pattern)
            if keys:
                return await self._client.delete(*keys)
            return 0
            
        except Exception as e:
            logger.error(f"Cache pattern deletion error: {e}")
            return 0
    
    async def delete_by_tags(self, tags: List[str]) -> int:
        try:
            # Get all keys for given tags
            keys = set()
            for tag in tags:
                tag_keys = await self._client.smembers(f"tag:{tag}")
                keys.update(tag_keys)
            
            if keys:
                return await self._client.delete(*keys)
            return 0
            
        except Exception as e:
            logger.error(f"Cache tag deletion error: {e}")
            return 0
    
    async def get_stats(self) -> Dict:
        try:
            info = await self._client.info()
            return {
                "used_memory": info["used_memory_human"],
                "evicted_keys": info["evicted_keys"],
                "hit_rate": self._calculate_hit_rate(info),
                "total_connections": info["total_connections_received"],
                "connected_clients": info["connected_clients"]
            }
            
        except Exception as e:
            logger.error(f"Cache stats error: {e}")
            return {}
    
    def _calculate_hit_rate(self, info: Dict) -> float:
        hits = info.get("keyspace_hits", 0)
        misses = info.get("keyspace_misses", 0)
        total = hits + misses
        return (hits / total) * 100 if total > 0 else 0
    
    async def cleanup_expired_tags(self):
        """Periodic cleanup of expired tag references"""
        try:
            tag_keys = await self._client.keys("tag:*")
            for tag_key in tag_keys:
                members = await self._client.smembers(tag_key)
                for member in members:
                    if not await self._client.exists(member):
                        await self._client.srem(tag_key, member)
                        
        except Exception as e:
            logger.error(f"Tag cleanup error: {e}")