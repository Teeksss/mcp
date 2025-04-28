from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
import asyncio
import json
import etcd3
from prometheus_client import Counter
import logging
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

logger = logging.getLogger(__name__)

@dataclass
class ConfigurationConfig:
    etcd_host: str
    etcd_port: int
    local_config_path: str
    refresh_interval: int = 30
    max_versions: int = 10

class ConfigurationManager:
    def __init__(self, config: ConfigurationConfig):
        self.config = config
        self.metrics = self._setup_metrics()
        self.etcd = etcd3.client(
            host=config.etcd_host,
            port=config.etcd_port
        )
        self.local_cache = {}
        self.watchers = {}
        self.file_observer = None
        
        # Current context
        self.timestamp = datetime.utcnow()
        self.current_user = "Teeksss"
    
    def _setup_metrics(self) -> Dict:
        """Initialize configuration metrics"""
        return {
            "config_updates": Counter(
                "config_updates_total",
                "Configuration update count",
                ["config_key"]
            ),
            "config_errors": Counter(
                "config_errors_total",
                "Configuration error count",
                ["error_type"]
            )
        }
    
    async def start(self):
        """Start configuration manager"""
        # Start file watcher
        self.file_observer = Observer()
        event_handler = ConfigFileHandler(self)
        self.file_observer.schedule(
            event_handler,
            self.config.local_config_path,
            recursive=True
        )
        self.file_observer.start()
        
        # Load initial configurations
        await self._load_configs()
    
    async def stop(self):
        """Stop configuration manager"""
        if self.file_observer:
            self.file_observer.stop()
            self.file_observer.join()
        
        # Stop etcd watchers
        for watcher in self.watchers.values():
            watcher.cancel()
    
    async def get_config(
        self,
        key: str,
        default: Any = None
    ) -> Optional[Any]:
        """Get configuration value"""
        try:
            # Check local cache first
            if key in self.local_cache:
                return self.local_cache[key]
            
            # Get from etcd
            value, metadata = self.etcd.get(key)
            if value is not None:
                value = json.loads(value.decode())
                self.local_cache[key] = value
                return value
            
            return default
            
        except Exception as e:
            logger.error(f"Configuration retrieval failed: {e}")
            self.metrics["config_errors"].labels(
                error_type="retrieval"
            ).inc()
            return default
    
    async def set_config(
        self,
        key: str,
        value: Any,
        version: Optional[str] = None
    ):
        """Set configuration value"""
        try:
            # Validate value
            if not self._validate_config(key, value):
                raise ValueError(f"Invalid configuration value for {key}")
            
            # Store value
            encoded_value = json.dumps(value).encode()
            self.etcd.put(key, encoded_value)
            
            # Store version if provided
            if version:
                version_key = f"{key}_versions"
                versions = await self.get_config(version_key, [])
                versions.append({
                    "version": version,
                    "value": value,
                    "timestamp": self.timestamp.isoformat(),
                    "user": self.current_user
                })
                
                # Keep only recent versions
                if len(versions) > self.config.max_versions:
                    versions = versions[-self.config.max_versions:]
                
                self.etcd.put(
                    version_key,
                    json.dumps(versions).encode()
                )
            
            # Update local cache
            self.local_cache[key] = value
            
            # Update metrics
            self.metrics["config_updates"].labels(
                config_key=key
            ).inc()
            
            logger.info(f"Updated configuration {key}")
            
        except Exception as e:
            logger.error(f"Configuration update failed: {e}")
            self.metrics["config_errors"].labels(
                error_type="update"
            ).inc()
            raise
    
    async def watch_config(
        self,
        key: str,
        callback: callable
    ):
        """Watch for configuration changes"""
        try:
            # Cancel existing watcher if any
            if key in self.watchers:
                self.watchers[key].cancel()
            
            # Start new watcher
            events_iterator, cancel = self.etcd.watch(key)
            self.watchers[key] = cancel
            
            # Handle events in background
            asyncio.create_task(
                self._handle_watch_events(
                    key,
                    events_iterator,
                    callback
                )
            )
            
        except Exception as e:
            logger.error(f"Configuration watch setup failed: {e}")
            self.metrics["config_errors"].labels(
                error_type="watch"
            ).inc()
    
    async def _handle_watch_events(
        self,
        key: str,
        events_iterator: Any,
        callback: callable
    ):
        """Handle configuration change events"""
        try:
            for event in events_iterator:
                if event.events:
                    for evt in event.events:
                        if evt.value:
                            value = json.loads(evt.value.decode())
                            self.local_cache[key] = value
                            await callback(key, value)
                
        except Exception as e:
            logger.error(f"Watch event handling failed: {e}")
            self.metrics["config_errors"].labels(
                error_type="watch_handler"
            ).inc()
    
    async def _load_configs(self):
        """Load configurations from etcd"""
        try:
            for item in self.etcd.get_all():
                key = item[1].key.decode()
                value = json.loads(item[0].decode())
                self.local_cache[key] = value
            
        except Exception as e:
            logger.error(f"Configuration loading failed: {e}")
            self.metrics["config_errors"].labels(
                error_type="load"
            ).inc()
    
    def _validate_config(self, key: str, value: Any) -> bool:
        """Validate configuration value"""
        try:
            # Basic validation
            if not isinstance(value, (dict, list, str, int, float, bool)):
                return False
            
            # Key-specific validation
            if key.startswith("model_"):
                return self._validate_model_config(value)
            elif key.startswith("feature_"):
                return self._validate_feature_config(value)
            
            return True
            
        except Exception:
            return False
    
    def _validate_model_config(self, config: Dict) -> bool:
        """Validate model configuration"""
        required_fields = [
            "model_type",
            "version",
            "parameters"
        ]
        return all(field in config for field in required_fields)
    
    def _validate_feature_config(self, config: Dict) -> bool:
        """Validate feature configuration"""
        required_fields = [
            "feature_name",
            "data_type",
            "transform"
        ]
        return all(field in config for field in required_fields)

class ConfigFileHandler(FileSystemEventHandler):
    def __init__(self, manager: ConfigurationManager):
        self.manager = manager
    
    def on_modified(self, event):
        if not event.is_directory:
            asyncio.create_task(
                self._handle_file_change(event.src_path)
            )
    
    async def _handle_file_change(self, file_path: str):
        """Handle configuration file changes"""
        try:
            with open(file_path, 'r') as f:
                config = json.load(f)
            
            # Update configurations
            for key, value in config.items():
                await self.manager.set_config(key, value)
            
        except Exception as e:
            logger.error(f"File change handling failed: {e}")
            self.manager.metrics["config_errors"].labels(
                error_type="file_change"
            ).inc()