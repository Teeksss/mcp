from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Union
from dataclasses import dataclass
import asyncio
import json
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from prometheus_client import Counter, Gauge
import logging

logger = logging.getLogger(__name__)

@dataclass
class SchedulerConfig:
    max_concurrent_jobs: int = 10
    job_timeout: int = 300
    max_retries: int = 3
    history_retention_days: int = 30

class JobStatus:
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"

class BackgroundScheduler:
    def __init__(self, config: SchedulerConfig):
        self.config = config
        self.metrics = self._setup_metrics()
        self.scheduler = AsyncIOScheduler()
        self.active_jobs: Dict[str, Dict] = {}
        self.job_history: List[Dict] = []
        self.semaphore = asyncio.Semaphore(config.max_concurrent_jobs)
        
        # Current context
        self.timestamp = datetime.utcnow()
        self.current_user = "Teeksss"
    
    def _setup_metrics(self) -> Dict:
        """Initialize scheduler metrics"""
        return {
            "scheduled_jobs": Counter(
                "scheduler_jobs_total",
                "Total scheduled jobs",
                ["job_type"]
            ),
            "active_jobs": Gauge(
                "scheduler_active_jobs",
                "Currently active jobs",
                ["job_type"]
            ),
            "job_errors": Counter(
                "scheduler_job_errors_total",
                "Job execution errors",
                ["job_type", "error_type"]
            )
        }
    
    async def start(self):
        """Start scheduler"""
        self.scheduler.start()
        asyncio.create_task(self._cleanup_history())
    
    async def stop(self):
        """Stop scheduler"""
        self.scheduler.shutdown()
    
    async def schedule_job(
        self,
        job_type: str,
        func: Callable,
        trigger: Union[CronTrigger, IntervalTrigger],
        args: Optional[List] = None,
        kwargs: Optional[Dict] = None,
        job_id: Optional[str] = None
    ) -> str:
        """Schedule new job"""
        try:
            job_id = job_id or f"{job_type}_{self.timestamp.timestamp()}"
            
            # Create job wrapper
            wrapped_func = self._create_job_wrapper(
                job_id,
                job_type,
                func,
                args or [],
                kwargs or {}
            )
            
            # Add job to scheduler
            self.scheduler.add_job(
                wrapped_func,
                trigger=trigger,
                id=job_id,
                replace_existing=True
            )
            
            # Initialize job state
            self.active_jobs[job_id] = {
                "type": job_type,
                "status": JobStatus.PENDING,
                "created_at": self.timestamp.isoformat(),
                "created_by": self.current_user,
                "last_run": None,
                "next_run": None,
                "error_count": 0
            }
            
            # Update metrics
            self.metrics["scheduled_jobs"].labels(
                job_type=job_type
            ).inc()
            
            logger.info(f"Scheduled job: {job_id}")
            return job_id
            
        except Exception as e:
            logger.error(f"Job scheduling failed: {e}")
            self.metrics["job_errors"].labels(
                job_type=job_type,
                error_type="scheduling"
            ).inc()
            raise
    
    def _create_job_wrapper(
        self,
        job_id: str,
        job_type: str,
        func: Callable,
        args: List,
        kwargs: Dict
    ) -> Callable:
        """Create wrapper for job execution"""
        async def wrapper():
            async with self.semaphore:
                try:
                    # Update job state
                    self.active_jobs[job_id].update({
                        "status": JobStatus.RUNNING,
                        "last_run": self.timestamp.isoformat()
                    })
                    
                    self.metrics["active_jobs"].labels(
                        job_type=job_type
                    ).inc()
                    
                    # Execute job with timeout
                    result = await asyncio.wait_for(
                        func(*args, **kwargs),
                        timeout=self.config.job_timeout
                    )
                    
                    # Update job state
                    self.active_jobs[job_id].update({
                        "status": JobStatus.COMPLETED,
                        "last_result": result,
                        "completed_at": datetime.utcnow().isoformat()
                    })
                    
                    # Archive job history
                    await self._archive_job_run(
                        job_id,
                        JobStatus.COMPLETED,
                        result
                    )
                    
                except Exception as e:
                    # Handle job failure
                    error_count = self.active_jobs[job_id]["error_count"] + 1
                    
                    if error_count <= self.config.max_retries:
                        # Retry job
                        self.active_jobs[job_id].update({
                            "status": JobStatus.RETRYING,
                            "error_count": error_count,
                            "last_error": str(e)
                        })
                        
                        # Schedule retry
                        retry_delay = 2 ** (error_count - 1)  # Exponential backoff
                        self.scheduler.add_job(
                            wrapper,
                            'date',
                            run_date=datetime.utcnow() + timedelta(
                                seconds=retry_delay
                            ),
                            id=f"{job_id}_retry_{error_count}"
                        )
                        
                    else:
                        # Mark job as failed
                        self.active_jobs[job_id].update({
                            "status": JobStatus.FAILED,
                            "error_count": error_count,
                            "last_error": str(e)
                        })
                        
                        # Archive failed job
                        await self._archive_job_run(
                            job_id,
                            JobStatus.FAILED,
                            error=str(e)
                        )
                    
                    # Update error metrics
                    self.metrics["job_errors"].labels(
                        job_type=job_type,
                        error_type="execution"
                    ).inc()
                    
                    logger.error(f"Job {job_id} failed: {e}")
                    
                finally:
                    self.metrics["active_jobs"].labels(
                        job_type=job_type
                    ).dec()
        
        return wrapper
    
    async def _archive_job_run(
        self,
        job_id: str,
        status: str,
        result: Any = None,
        error: Optional[str] = None
    ):
        """Archive job execution history"""
        try:
            job_info = self.active_jobs[job_id].copy()
            job_info.update({
                "execution_id": f"{job_id}_{self.timestamp.timestamp()}",
                "execution_status": status,
                "execution_result": result,
                "execution_error": error,
                "archived_at": self.timestamp.isoformat()
            })
            
            self.job_history.append(job_info)
            
        except Exception as e:
            logger.error(f"Job archival failed: {e}")
    
    async def _cleanup_history(self):
        """Cleanup old job history"""
        while True:
            try:
                cutoff_date = self.timestamp - timedelta(
                    days=self.config.history_retention_days
                )
                
                self.job_history = [
                    job for job in self.job_history
                    if datetime.fromisoformat(job["archived_at"]) > cutoff_date
                ]
                
            except Exception as e:
                logger.error(f"History cleanup failed: {e}")
            
            await asyncio.sleep(86400)  # Daily cleanup