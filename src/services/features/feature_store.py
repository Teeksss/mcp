from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
import asyncio
import pandas as pd
import numpy as np
from feast import FeatureStore, Entity, Feature, FeatureView
from feast.infra.offline_stores.file import FileOfflineStore
from feast.infra.online_stores.redis import RedisOnlineStore
import redis
from prometheus_client import Counter, Histogram
import logging

logger = logging.getLogger(__name__)

@dataclass
class FeatureStoreConfig:
    redis_url: str
    offline_store_path: str
    feature_ttl: int = 86400  # 1 day
    batch_size: int = 1000
    update_interval: int = 300  # 5 minutes

class EnhancedFeatureStore:
    def __init__(self, config: FeatureStoreConfig):
        self.config = config
        self.metrics = self._setup_metrics()
        self.redis = redis.Redis.from_url(config.redis_url)
        self.update_task = None
        
        # Initialize Feast store
        self.store = FeatureStore(
            repo_path=config.offline_store_path,
            provider="local"
        )
        
        # Current context
        self.timestamp = datetime.utcnow()
        self.current_user = "Teeksss"
    
    def _setup_metrics(self) -> Dict:
        """Initialize feature store metrics"""
        return {
            "feature_requests": Counter(
                "feature_requests_total",
                "Total feature requests",
                ["feature_view", "type"]
            ),
            "feature_latency": Histogram(
                "feature_retrieval_seconds",
                "Feature retrieval latency",
                ["feature_view"]
            ),
            "cache_hits": Counter(
                "feature_cache_hits_total",
                "Feature cache hits",
                ["feature_view"]
            )
        }
    
    async def start(self):
        """Start feature store services"""
        self.update_task = asyncio.create_task(
            self._update_loop()
        )
    
    async def stop(self):
        """Stop feature store services"""
        if self.update_task:
            self.update_task.cancel()
            try:
                await self.update_task
            except asyncio.CancelledError:
                pass
    
    async def get_online_features(
        self,
        feature_view: str,
        entity_keys: List[str]
    ) -> Dict[str, List[Any]]:
        """Get online features for entities"""
        try:
            start_time = datetime.utcnow()
            
            # Try cache first
            cached_features = await self._get_from_cache(
                feature_view,
                entity_keys
            )
            
            if cached_features:
                self.metrics["cache_hits"].labels(
                    feature_view=feature_view
                ).inc()
                return cached_features
            
            # Get from feature store
            features = self.store.get_online_features(
                feature_view=feature_view,
                entity_keys=entity_keys
            ).to_dict()
            
            # Update cache
            await self._update_cache(
                feature_view,
                entity_keys,
                features
            )
            
            # Update metrics
            duration = (datetime.utcnow() - start_time).total_seconds()
            self.metrics["feature_latency"].labels(
                feature_view=feature_view
            ).observe(duration)
            
            self.metrics["feature_requests"].labels(
                feature_view=feature_view,
                type="online"
            ).inc()
            
            return features
            
        except Exception as e:
            logger.error(f"Online feature retrieval failed: {e}")
            raise
    
    async def get_historical_features(
        self,
        feature_view: str,
        entity_df: pd.DataFrame,
        feature_names: List[str]
    ) -> pd.DataFrame:
        """Get historical features"""
        try:
            start_time = datetime.utcnow()
            
            # Get features from offline store
            features = self.store.get_historical_features(
                entity_df=entity_df,
                feature_view=feature_view,
                feature_names=feature_names
            ).to_df()
            
            # Update metrics
            duration = (datetime.utcnow() - start_time).total_seconds()
            self.metrics["feature_latency"].labels(
                feature_view=feature_view
            ).observe(duration)
            
            self.metrics["feature_requests"].labels(
                feature_view=feature_view,
                type="historical"
            ).inc()
            
            return features
            
        except Exception as e:
            logger.error(f"Historical feature retrieval failed: {e}")
            raise
    
    async def _get_from_cache(
        self,
        feature_view: str,
        entity_keys: List[str]
    ) -> Optional[Dict]:
        """Get features from Redis cache"""
        try:
            features = {}
            for key in entity_keys:
                cache_key = f"feature:{feature_view}:{key}"
                value = self.redis.get(cache_key)
                if value:
                    features[key] = value
            
            return features if features else None
            
        except Exception as e:
            logger.error(f"Cache retrieval failed: {e}")
            return None
    
    async def _update_cache(
        self,
        feature_view: str,
        entity_keys: List[str],
        features: Dict
    ):
        """Update Redis cache with features"""
        try:
            pipeline = self.redis.pipeline()
            
            for key in entity_keys:
                cache_key = f"feature:{feature_view}:{key}"
                pipeline.setex(
                    cache_key,
                    self.config.feature_ttl,
                    features[key]
                )
            
            pipeline.execute()
            
        except Exception as e:
            logger.error(f"Cache update failed: {e}")
    
    async def _update_loop(self):
        """Periodic update of feature views"""
        while True:
            try:
                # Update all feature views
                feature_views = self.store.list_feature_views()
                
                for view in feature_views:
                    await self._update_feature_view(view)
                
            except Exception as e:
                logger.error(f"Feature view update failed: {e}")
            
            await asyncio.sleep(self.config.update_interval)
    
    async def _update_feature_view(self, feature_view: FeatureView):
        """Update single feature view"""
        try:
            # Get latest data
            current_time = datetime.utcnow()
            start_time = current_time - timedelta(
                seconds=self.config.update_interval
            )
            
            # Update offline store
            self.store.materialize_incremental(
                feature_view_name=feature_view.name,
                start_time=start_time,
                end_time=current_time
            )
            
            logger.info(f"Updated feature view: {feature_view.name}")
            
        except Exception as e:
            logger.error(
                f"Feature view {feature_view.name} update failed: {e}"
            )