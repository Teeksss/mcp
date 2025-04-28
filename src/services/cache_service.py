import redis
import json
import zlib
from typing import Optional, Any
from datetime import datetime
import logging
from src.config.settings import settings

logger = logging.getLogger(__name__)

class CacheService:
    def __init__(self):
        self.redis_pool = redis.ConnectionPool.from_url(
            settings.redis_url,
            max_connections=settings.redis_pool_size
        )
        self._client = None
        
    @property
    def client(self):
        if self._client is None:
            self._client = redis.Redis(connection_pool=self.redis_pool)
        return self._client
    
    def _compress(self, data: str) -> bytes:
        return zlib.compress(json.dumps(data).encode())
    
    def _decompress(self, data: bytes) -> Any:
        return json.loads(zlib.decompress(data).decode())
    
    async def get(self, key: str) -> Optional[Any]:
        try:
            data = await self.client.get(key)
            if data:
                if settings.cache.compression:
                    return self._decompress(data)
                return json.loads(data)
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return None
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        nx: bool = False
    ) -> bool:
        try:
            if settings.cache.compression:
                value = self._compress(value)
            else:
                value = json.dumps(value)
            
            if nx:
                return await self.client.set(
                    key,
                    value,
                    ex=ttl or settings.cache.ttl,
                    nx=True
                )
            
            return await self.client.set(
                key,
                value,
                ex=ttl or settings.cache.ttl
            )
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        try:
            return bool(await self.client.delete(key))
        except Exception as e:
            logger.error(f"Cache delete error: {e}")
            return False
    
    async def clear_pattern(self, pattern: str) -> int:
        try:
            keys = await self.client.keys(pattern)
            if keys:
                return await self.client.delete(*keys)
            return 0
        except Exception as e:
            logger.error(f"Cache clear pattern error: {e}")
            return 0
    
    async def get_stats(self) -> dict:
        try:
            info = await self.client.info()
            return {
                'used_memory': info['used_memory'],
                'hits': info['keyspace_hits'],
                'misses': info['keyspace_misses'],
                'evicted_keys': info['evicted_keys']
            }
        except Exception as e:
            logger.error(f"Cache stats error: {e}")
            return {}