from datetime import datetime, timedelta
from typing import Dict, List, Optional
import asyncio
from collections import defaultdict
import numpy as np
from dataclasses import dataclass
import json
import aioredis
from prometheus_client import Counter, Histogram
import logging

logger = logging.getLogger(__name__)

@dataclass
class AnalyticsConfig:
    redis_url: str
    window_size: int = 3600  # 1 hour
    update_interval: int = 10
    max_events: int = 10000

class RealtimeAnalytics:
    def __init__(self, config: AnalyticsConfig):
        self.config = config
        self.redis = aioredis.from_url(config.redis_url)
        self.metrics = self._setup_metrics()
        self.update_task = None
        
        # Current context
        self.current_time = datetime.utcnow()
        self.current_user = "Teeksss"
    
    def _setup_metrics(self):
        """Initialize Prometheus metrics"""
        return {
            "events": Counter(
                "analytics_events_total",
                "Total events processed",
                ["event_type"]
            ),
            "processing_time": Histogram(
                "analytics_processing_seconds",
                "Event processing time",
                ["event_type"]
            )
        }
    
    async def start(self):
        """Start analytics service"""
        self.update_task = asyncio.create_task(
            self._update_loop()
        )
    
    async def stop(self):
        """Stop analytics service"""
        if self.update_task:
            self.update_task.cancel()
            try:
                await self.update_task
            except asyncio.CancelledError:
                pass
    
    async def track_event(
        self,
        event_type: str,
        data: Dict,
        timestamp: Optional[datetime] = None
    ):
        """Track a new event"""
        try:
            timestamp = timestamp or datetime.utcnow()
            event = {
                "type": event_type,
                "data": data,
                "timestamp": timestamp.isoformat(),
                "user": self.current_user
            }
            
            # Store event in Redis
            await self.redis.xadd(
                f"events:{event_type}",
                event,
                maxlen=self.config.max_events
            )
            
            # Update metrics
            self.metrics["events"].labels(
                event_type=event_type
            ).inc()
            
            # Update real-time statistics
            await self._update_statistics(event_type, data)
            
        except Exception as e:
            logger.error(f"Event tracking failed: {e}")
    
    async def get_statistics(
        self,
        event_type: str,
        window: Optional[int] = None
    ) -> Dict:
        """Get statistics for event type"""
        try:
            window = window or self.config.window_size
            start_time = datetime.utcnow() - timedelta(seconds=window)
            
            # Get events from Redis
            events = await self.redis.xrange(
                f"events:{event_type}",
                min=start_time.timestamp() * 1000,
                max='+',
                count=self.config.max_events
            )
            
            # Process events
            return await self._calculate_statistics(events)
            
        except Exception as e:
            logger.error(f"Statistics calculation failed: {e}")
            return {}
    
    async def _update_statistics(
        self,
        event_type: str,
        data: Dict
    ):
        """Update real-time statistics"""
        try:
            # Update various statistics in Redis
            pipeline = self.redis.pipeline()
            
            # Update count
            pipeline.hincrby(
                f"stats:{event_type}:count",
                datetime.utcnow().strftime("%Y-%m-%d-%H"),
                1
            )
            
            # Update user statistics
            pipeline.hincrby(
                f"stats:{event_type}:users",
                self.current_user,
                1
            )
            
            # Track numeric values
            for key, value in data.items():
                if isinstance(value, (int, float)):
                    pipeline.rpush(
                        f"stats:{event_type}:values:{key}",
                        value
                    )
                    pipeline.ltrim(
                        f"stats:{event_type}:values:{key}",
                        -100,
                        -1
                    )
            
            await pipeline.execute()
            
        except Exception as e:
            logger.error(f"Statistics update failed: {e}")
    
    async def _calculate_statistics(self, events: List) -> Dict:
        """Calculate statistics from events"""
        stats = {
            "total_events": len(events),
            "unique_users": set(),
            "values": defaultdict(list)
        }
        
        for event_id, event_data in events:
            try:
                event = json.loads(event_data["data"])
                stats["unique_users"].add(event_data["user"])
                
                # Process numeric values
                for key, value in event.items():
                    if isinstance(value, (int, float)):
                        stats["values"][key].append(value)
            except Exception as e:
                logger.error(f"Event processing failed: {e}")
        
        # Calculate aggregates
        for key, values in stats["values"].items():
            if values:
                stats[key] = {
                    "mean": np.mean(values),
                    "median": np.median(values),
                    "std": np.std(values),
                    "min": min(values),
                    "max": max(values)
                }
        
        stats["unique_users"] = len(stats["unique_users"])
        return stats
    
    async def _update_loop(self):
        """Periodic update loop"""
        while True:
            try:
                # Clean up old data
                cutoff = datetime.utcnow() - timedelta(
                    seconds=self.config.window_size
                )
                
                # Delete old events
                for event_type in await self.redis.keys("events:*"):
                    await self.redis.xtrim(
                        event_type,
                        maxlen=self.config.max_events
                    )
                
                # Delete old statistics
                for key in await self.redis.keys("stats:*"):
                    if key.endswith(":count"):
                        old_hours = await self.redis.hkeys(key)
                        for hour in old_hours:
                            if datetime.strptime(
                                hour.decode(),
                                "%Y-%m-%d-%H"
                            ) < cutoff:
                                await self.redis.hdel(key, hour)
                
            except Exception as e:
                logger.error(f"Update loop error: {e}")
            
            await asyncio.sleep(self.config.update_interval)