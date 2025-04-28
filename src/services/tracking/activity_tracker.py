from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
import asyncio
import json
from dataclasses import dataclass
import redis
from prometheus_client import Counter, Histogram
import logging

logger = logging.getLogger(__name__)

@dataclass
class ActivityConfig:
    redis_url: str
    session_timeout: int = 1800  # 30 minutes
    cleanup_interval: int = 300  # 5 minutes
    max_session_history: int = 100

class UserActivityTracker:
    def __init__(self, config: ActivityConfig):
        self.config = config
        self.redis = redis.Redis.from_url(config.redis_url)
        self.metrics = self._setup_metrics()
        self.cleanup_task = None
        
        # Current context
        self.timestamp = datetime.utcnow()
        self.current_user = "Teeksss"
    
    def _setup_metrics(self) -> Dict:
        """Initialize activity metrics"""
        return {
            "sessions": Counter(
                "user_sessions_total",
                "Total user sessions",
                ["user_id"]
            ),
            "actions": Counter(
                "user_actions_total",
                "Total user actions",
                ["user_id", "action_type"]
            ),
            "session_duration": Histogram(
                "session_duration_seconds",
                "Session duration in seconds",
                ["user_id"]
            )
        }
    
    async def start(self):
        """Start activity tracking"""
        self.cleanup_task = asyncio.create_task(
            self._cleanup_loop()
        )
    
    async def stop(self):
        """Stop activity tracking"""
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
    
    async def track_session_start(
        self,
        user_id: str,
        session_data: Dict
    ) -> str:
        """Track new session start"""
        try:
            session_id = f"session:{user_id}:{self.timestamp.timestamp()}"
            
            # Store session data
            session_info = {
                "session_id": session_id,
                "user_id": user_id,
                "start_time": self.timestamp.isoformat(),
                "last_activity": self.timestamp.isoformat(),
                "data": session_data
            }
            
            # Store in Redis
            self.redis.setex(
                f"sessions:{session_id}",
                self.config.session_timeout,
                json.dumps(session_info)
            )
            
            # Update metrics
            self.metrics["sessions"].labels(
                user_id=user_id
            ).inc()
            
            return session_id
            
        except Exception as e:
            logger.error(f"Session start tracking failed: {e}")
            raise
    
    async def track_activity(
        self,
        session_id: str,
        action_type: str,
        action_data: Dict
    ):
        """Track user activity within session"""
        try:
            # Get session info
            session_key = f"sessions:{session_id}"
            session_data = self.redis.get(session_key)
            
            if not session_data:
                raise ValueError(f"Session {session_id} not found")
            
            session_info = json.loads(session_data)
            
            # Record activity
            activity = {
                "timestamp": self.timestamp.isoformat(),
                "action_type": action_type,
                "action_data": action_data
            }
            
            # Store activity
            activity_key = f"activities:{session_id}"
            self.redis.rpush(
                activity_key,
                json.dumps(activity)
            )
            
            # Update session last activity
            session_info["last_activity"] = self.timestamp.isoformat()
            self.redis.setex(
                session_key,
                self.config.session_timeout,
                json.dumps(session_info)
            )
            
            # Update metrics
            self.metrics["actions"].labels(
                user_id=session_info["user_id"],
                action_type=action_type
            ).inc()
            
        except Exception as e:
            logger.error(f"Activity tracking failed: {e}")
            raise
    
    async def end_session(self, session_id: str):
        """End user session"""
        try:
            # Get session info
            session_key = f"sessions:{session_id}"
            session_data = self.redis.get(session_key)
            
            if not session_data:
                return
            
            session_info = json.loads(session_data)
            
            # Calculate session duration
            start_time = datetime.fromisoformat(session_info["start_time"])
            duration = (self.timestamp - start_time).total_seconds()
            
            # Update metrics
            self.metrics["session_duration"].labels(
                user_id=session_info["user_id"]
            ).observe(duration)
            
            # Archive session data
            await self._archive_session(session_id, session_info, duration)
            
            # Clean up session data
            self.redis.delete(session_key)
            self.redis.delete(f"activities:{session_id}")
            
        except Exception as e:
            logger.error(f"Session end failed: {e}")
    
    async def get_user_sessions(
        self,
        user_id: str,
        limit: int = 10
    ) -> List[Dict]:
        """Get user's recent sessions"""
        try:
            sessions = []
            pattern = f"sessions:session:{user_id}:*"
            
            for key in self.redis.scan_iter(pattern):
                session_data = self.redis.get(key)
                if session_data:
                    sessions.append(json.loads(session_data))
            
            return sorted(
                sessions,
                key=lambda x: x["start_time"],
                reverse=True
            )[:limit]
            
        except Exception as e:
            logger.error(f"Session retrieval failed: {e}")
            return []
    
    async def _archive_session(
        self,
        session_id: str,
        session_info: Dict,
        duration: float
    ):
        """Archive session data"""
        try:
            # Get session activities
            activities = []
            activity_key = f"activities:{session_id}"
            
            for activity_data in self.redis.lrange(
                activity_key,
                0,
                -1
            ):
                activities.append(json.loads(activity_data))
            
            # Create archive record
            archive_data = {
                **session_info,
                "end_time": self.timestamp.isoformat(),
                "duration": duration,
                "activities": activities
            }
            
            # Store in Redis with expiration
            archive_key = f"archive:sessions:{session_id}"
            self.redis.setex(
                archive_key,
                86400 * 30,  # 30 days
                json.dumps(archive_data)
            )
            
        except Exception as e:
            logger.error(f"Session archival failed: {e}")
    
    async def _cleanup_loop(self):
        """Periodic cleanup of expired sessions"""
        while True:
            try:
                # Find expired sessions
                for key in self.redis.scan_iter("sessions:*"):
                    session_data = self.redis.get(key)
                    if session_data:
                        session_info = json.loads(session_data)
                        last_activity = datetime.fromisoformat(
                            session_info["last_activity"]
                        )
                        
                        if (self.timestamp - last_activity).seconds > self.config.session_timeout:
                            await self.end_session(session_info["session_id"])
                
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")
            
            await asyncio.sleep(self.config.cleanup_interval)