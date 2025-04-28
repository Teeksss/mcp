from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
import asyncio
import json
import aiokafka
from prometheus_client import Counter, Histogram
import logging

logger = logging.getLogger(__name__)

@dataclass
class StreamConfig:
    kafka_brokers: List[str]
    consumer_group: str
    max_batch_size: int = 100
    commit_interval: int = 5
    retry_interval: int = 30

class EventStream:
    def __init__(self, config: StreamConfig):
        self.config = config
        self.metrics = self._setup_metrics()
        self.producer = None
        self.consumers = {}
        self.handlers = {}
        
        # Current context
        self.timestamp = datetime.utcnow()
        self.current_user = "Teeksss"
    
    def _setup_metrics(self) -> Dict:
        """Initialize streaming metrics"""
        return {
            "messages_produced": Counter(
                "stream_messages_produced_total",
                "Total messages produced",
                ["topic"]
            ),
            "messages_consumed": Counter(
                "stream_messages_consumed_total",
                "Total messages consumed",
                ["topic", "consumer_group"]
            ),
            "processing_time": Histogram(
                "stream_processing_seconds",
                "Message processing time",
                ["topic", "consumer_group"]
            ),
            "processing_errors": Counter(
                "stream_processing_errors_total",
                "Processing errors",
                ["topic", "error_type"]
            )
        }
    
    async def start(self):
        """Start event streaming"""
        # Initialize producer
        self.producer = aiokafka.AIOKafkaProducer(
            bootstrap_servers=self.config.kafka_brokers
        )
        await self.producer.start()
    
    async def stop(self):
        """Stop event streaming"""
        # Stop producer
        if self.producer:
            await self.producer.stop()
        
        # Stop consumers
        for consumer in self.consumers.values():
            await consumer.stop()
    
    async def publish_event(
        self,
        topic: str,
        event_data: Dict,
        key: Optional[str] = None
    ):
        """Publish event to topic"""
        try:
            # Add metadata
            event = {
                "data": event_data,
                "metadata": {
                    "timestamp": self.timestamp.isoformat(),
                    "user": self.current_user,
                    "version": "1.0"
                }
            }
            
            # Publish to Kafka
            await self.producer.send_and_wait(
                topic,
                json.dumps(event).encode(),
                key=key.encode() if key else None
            )
            
            # Update metrics
            self.metrics["messages_produced"].labels(
                topic=topic
            ).inc()
            
            logger.info(f"Published event to topic {topic}")
            
        except Exception as e:
            logger.error(f"Event publishing failed: {e}")
            self.metrics["processing_errors"].labels(
                topic=topic,
                error_type="publish"
            ).inc()
            raise
    
    async def subscribe(
        self,
        topic: str,
        handler: Callable,
        group_id: Optional[str] = None
    ):
        """Subscribe to topic"""
        try:
            group_id = group_id or f"{self.config.consumer_group}_{topic}"
            
            # Create consumer
            consumer = aiokafka.AIOKafkaConsumer(
                topic,
                bootstrap_servers=self.config.kafka_brokers,
                group_id=group_id,
                enable_auto_commit=False
            )
            
            # Store handler
            self.handlers[topic] = handler
            
            # Start consumer
            await consumer.start()
            self.consumers[topic] = consumer
            
            # Start processing task
            asyncio.create_task(
                self._process_messages(topic, consumer)
            )
            
            logger.info(f"Subscribed to topic {topic}")
            
        except Exception as e:
            logger.error(f"Subscription failed: {e}")
            raise
    
    async def _process_messages(
        self,
        topic: str,
        consumer: aiokafka.AIOKafkaConsumer
    ):
        """Process messages from topic"""
        try:
            batch = []
            last_commit = self.timestamp
            
            async for message in consumer:
                try:
                    start_time = datetime.utcnow()
                    
                    # Parse message
                    event = json.loads(message.value.decode())
                    
                    # Add to batch
                    batch.append(event)
                    
                    # Process batch if full
                    if len(batch) >= self.config.max_batch_size:
                        await self._process_batch(topic, batch)
                        batch = []
                        await consumer.commit()
                        last_commit = datetime.utcnow()
                    
                    # Commit periodically
                    elif (datetime.utcnow() - last_commit).seconds >= self.config.commit_interval:
                        if batch:
                            await self._process_batch(topic, batch)
                            batch = []
                        await consumer.commit()
                        last_commit = datetime.utcnow()
                    
                    # Update metrics
                    processing_time = (
                        datetime.utcnow() - start_time
                    ).total_seconds()
                    self.metrics["processing_time"].labels(
                        topic=topic,
                        consumer_group=consumer.group_id
                    ).observe(processing_time)
                    
                    self.metrics["messages_consumed"].labels(
                        topic=topic,
                        consumer_group=consumer.group_id
                    ).inc()
                    
                except Exception as e:
                    logger.error(f"Message processing failed: {e}")
                    self.metrics["processing_errors"].labels(
                        topic=topic,
                        error_type="processing"
                    ).inc()
            
        except Exception as e:
            logger.error(f"Consumer failed: {e}")
            
            # Retry subscription
            await asyncio.sleep(self.config.retry_interval)
            await self.subscribe(
                topic,
                self.handlers[topic],
                consumer.group_id
            )
    
    async def _process_batch(self, topic: str, batch: List[Dict]):
        """Process batch of messages"""
        try:
            if topic in self.handlers:
                await self.handlers[topic](batch)
            
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            self.metrics["processing_errors"].labels(
                topic=topic,
                error_type="batch"
            ).inc()
            raise