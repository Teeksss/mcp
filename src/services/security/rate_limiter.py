from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
import asyncio
import redis
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class RateLimitConfig:
    redis_url: str
    default_rate: int = 60  # requests per minute
    default_burst: int = 10
    window_size: int = 60  # seconds
    cleanup_interval: int = 300  # seconds

class TokenBucketRateLimiter:
    def __init__(self, config: RateLimitConfig):
        self.config = config
        self.redis = redis.Redis.from_url(config.redis_url)
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        # Current context
        self.current_user = "Teeksss"
        self.current_time = datetime.utcnow()
    
    async def check_rate_limit(
        self,
        key: str,
        rate: Optional[int] = None,
        burst: Optional[int] = None
    ) -> Tuple[bool, Dict]:
        """Check if request is within rate limits"""
        try:
            rate = rate or self.config.default_rate
            burst = burst or self.config.default_burst
            
            # Get current token count
            token_key = f"ratelimit:{key}:tokens"
            last_update_key = f"ratelimit:{key}:last_update"
            
            # Get current values
            pipeline = self.redis.pipeline()
            pipeline.get(token_key)
            pipeline.get(last_update_key)
            tokens, last_update = pipeline.execute()
            
            current_tokens = float(tokens) if tokens else burst
            last_update = float(last_update) if last_update else 0
            
            # Calculate token refill
            now = datetime.utcnow().timestamp()
            time_passed = now - last_update
            tokens_to_add = time_passed * (rate / 60.0)
            new_tokens = min(burst, current_tokens + tokens_to_add)
            
            # Check if we have enough tokens
            if new_tokens >= 1:
                # Consume token
                new_tokens -= 1
                
                # Update Redis
                pipeline = self.redis.pipeline()
                pipeline.set(token_key, new_tokens)
                pipeline.set(last_update_key, now)
                pipeline.execute()
                
                return True, {
                    "remaining": new_tokens,
                    "reset": now + ((burst - new_tokens) * 60 / rate)
                }
            
            return False, {
                "remaining": 0,
                "reset": last_update + (60 / rate)
            }
            
        except Exception as e:
            logger.error(f"Rate limit check error: {e}")
            # Default to allowing request in case of errors
            return True, {"remaining": 1, "reset": 0}
    
    async def _cleanup_loop(self):
        """Periodically clean up expired rate limit data"""
        while True:
            try:
                # Get all rate limit keys
                keys = self.redis.keys("ratelimit:*")
                
                # Check each key for expiration
                for key in keys:
                    last_update = self.redis.get(f"{key}:last_update")
                    if last_update:
                        last_update = float(last_update)
                        if datetime.utcnow().timestamp() - last_update > self.config.window_size:
                            # Remove expired keys
                            self.redis.delete(key)
                            self.redis.delete(f"{key}:last_update")
                
            except Exception as e:
                logger.error(f"Rate limit cleanup error: {e}")
            
            await asyncio.sleep(self.config.cleanup_interval)
    
    def get_user_rate_limit(self, user_id: str) -> Tuple[int, int]:
        """Get rate limit configuration for user"""
        # Example: different limits based on user role
        if user_id == "admin":
            return 1000, 50  # rate, burst
        elif user_id.startswith("premium_"):
            return 200, 20
        else:
            return self.config.default_rate, self.config.default_burst
    
    async def close(self):
        """Cleanup resources"""
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass