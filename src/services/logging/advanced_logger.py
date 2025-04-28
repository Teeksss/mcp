from datetime import datetime
import logging
import json
import threading
import queue
from typing import Dict, Any, Optional
from dataclasses import dataclass
import asyncio
import aioboto3
import structlog
from src.config import settings

@dataclass
class LogConfig:
    app_name: str = "mcp-server"
    environment: str = settings.ENVIRONMENT
    log_level: str = "INFO"
    aws_cloudwatch_group: str = f"/mcp-server/{settings.ENVIRONMENT}"
    aws_cloudwatch_stream: str = f"application-{datetime.utcnow().strftime('%Y-%m-%d')}"
    local_log_path: str = "logs/mcp-server.log"
    batch_size: int = 100
    flush_interval: int = 10  # seconds

class AdvancedLogger:
    def __init__(self, config: LogConfig):
        self.config = config
        self.log_queue = queue.Queue()
        self.session = aioboto3.Session()
        
        # Initialize structured logger
        self.logger = self._setup_structured_logger()
        
        # Start background worker
        self.worker_thread = threading.Thread(
            target=self._process_log_queue,
            daemon=True
        )
        self.worker_thread.start()
    
    def _setup_structured_logger(self) -> structlog.BoundLogger:
        """Setup structured logger with processors"""
        processors = [
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ]
        
        structlog.configure(
            processors=processors,
            context_class=dict,
            logger_factory=structlog.PrintLoggerFactory(),
            wrapper_class=structlog.BoundLogger,
            cache_logger_on_first_use=True,
        )
        
        return structlog.get_logger()
    
    async def log(
        self,
        level: str,
        message: str,
        extra: Optional[Dict[str, Any]] = None
    ):
        """Log a message with additional context"""
        try:
            log_entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "level": level,
                "message": message,
                "app_name": self.config.app_name,
                "environment": self.config.environment,
                "user": "Teeksss",  # Current user
                **extra if extra else {}
            }
            
            # Add to queue for async processing
            self.log_queue.put(log_entry)
            
            # Also log to structured logger
            self.logger.msg(
                message,
                level=level,
                **log_entry
            )
            
        except Exception as e:
            print(f"Logging error: {e}")
    
    async def _flush_to_cloudwatch(self, logs: list):
        """Flush logs to CloudWatch"""
        try:
            async with self.session.client('logs') as logs_client:
                await logs_client.put_log_events(
                    logGroupName=self.config.aws_cloudwatch_group,
                    logStreamName=self.config.aws_cloudwatch_stream,
                    logEvents=[
                        {
                            'timestamp': int(
                                datetime.fromisoformat(
                                    log['timestamp']
                                ).timestamp() * 1000
                            ),
                            'message': json.dumps(log)
                        }
                        for log in logs
                    ]
                )
        except Exception as e:
            print(f"CloudWatch flush error: {e}")
    
    def _process_log_queue(self):
        """Process logs from queue in batches"""
        while True:
            logs = []
            try:
                # Collect logs from queue
                while len(logs) < self.config.batch_size:
                    try:
                        log = self.log_queue.get(
                            timeout=self.config.flush_interval
                        )
                        logs.append(log)
                    except queue.Empty:
                        break
                
                if logs:
                    # Create event loop for async operations
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    
                    # Flush to CloudWatch
                    loop.run_until_complete(
                        self._flush_to_cloudwatch(logs)
                    )
                    
                    # Write to local file
                    with open(self.config.local_log_path, 'a') as f:
                        for log in logs:
                            f.write(json.dumps(log) + '\n')
                    
            except Exception as e:
                print(f"Log processing error: {e}")