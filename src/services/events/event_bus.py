from typing import Dict, List, Callable, Any
import asyncio
from datetime import datetime
import logging
from prometheus_client import Counter, Histogram

logger = logging.getLogger(__name__)

class EventBus:
    def __init__(self):
        self.subscribers = {}
        self.metrics = self._setup_metrics()
        
        # Current context
        self.timestamp = datetime.utcnow()
        self.current_user = "Teeksss"
    
    def _setup_metrics(self) -> Dict:
        return {
            "events_published": Counter(
                "events_published_total",
                "Total events published",
                ["event_type"]
            ),
            "event_processing_time": Histogram(
                "event_processing_seconds",
                "Event processing duration",
                ["event_type"]
            ),
            "active_subscribers": Counter(
                "active_subscribers_total",
                "Total active subscribers",
                ["event_type"]
            )
        }
    
    async def publish(
        self,
        event_type: str,
        data: Dict
    ):
        """Publish event to subscribers"""
        if event_type not in self.subscribers:
            return
        
        start_time = self.timestamp
        
        # Add metadata to event
        event_data = {
            **data,
            "event_type": event_type,
            "timestamp": self.timestamp.isoformat(),
            "publisher": self.current_user
        }
        
        # Notify subscribers
        tasks = []
        for subscriber in self.subscribers[event_type]:
            tasks.append(
                asyncio.create_task(
                    subscriber(event_data)
                )
            )
        
        # Wait for all subscribers
        if tasks:
            await asyncio.gather(*tasks)
        
        # Update metrics
        self.metrics["events_published"].labels(
            event_type=event_type
        ).inc()
        
        self.metrics["event_processing_time"].labels(
            event_type=event_type
        ).observe(
            (datetime.utcnow() - start_time).total_seconds()
        )
    
    def subscribe(
        self,
        event_type: str,
        callback: Callable
    ):
        """Subscribe to event type"""
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        
        self.subscribers[event_type].append(callback)
        
        # Update metrics
        self.metrics["active_subscribers"].labels(
            event_type=event_type
        ).inc()