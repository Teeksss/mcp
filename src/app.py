from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import logging
from contextlib import asynccontextmanager
from datetime import datetime

from src.config.settings import settings
from src.services.caching.smart_cache import SmartCache
from src.services.load_balancing.intelligent_balancer import IntelligentLoadBalancer
from src.services.security.advanced_security import AdvancedSecurityManager
from src.services.optimization.performance_optimizer import PerformanceOptimizer
from src.services.recovery.error_recovery import ErrorRecoverySystem
from src.services.versioning.model_version_manager import ModelVersionManager
from src.services.pipeline.orchestrator import PipelineOrchestrator
from src.utils.monitoring import EnhancedMonitoring

logger = logging.getLogger(__name__)

class MCPServer:
    def __init__(self):
        self.app = FastAPI(
            title="MCP Server",
            description="Multi-Component Processing Server",
            version="1.0.0",
            lifespan=self.lifespan_context
        )
        
        # Initialize components
        self.cache = SmartCache()
        self.load_balancer = IntelligentLoadBalancer()
        self.security = AdvancedSecurityManager()
        self.optimizer = PerformanceOptimizer()
        self.error_recovery = ErrorRecoverySystem()
        self.version_manager = ModelVersionManager()
        
        # Initialize orchestrator
        self.pipeline = PipelineOrchestrator(
            model_manager=self.model_manager,
            rag_enhancer=self.rag_enhancer,
            performance_optimizer=self.optimizer,
            error_recovery=self.error_recovery
        )
        
        # Initialize monitoring
        self.monitoring = EnhancedMonitoring(settings)
        
        # Setup routes and middleware
        self._setup_routes()
        self._setup_middleware()
        self._setup_error_handlers()
        
        # Current context
        self.timestamp = datetime.utcnow()
        self.current_user = "Teeksss"
    
    @asynccontextmanager
    async def lifespan_context(self, app: FastAPI):
        """Lifespan context manager for FastAPI"""
        # Startup
        logger.info("Starting MCP Server")
        await self._startup()
        
        yield
        
        # Shutdown
        logger.info("Shutting down MCP Server")
        await self._shutdown()
    
    def _setup_routes(self):
        """Setup API routes"""
        from src.api.endpoints import (
            model,
            pipeline,
            query,
            version
        )
        
        # Add routers
        self.app.include_router(
            model.router,
            prefix="/api/v1/models",
            tags=["models"]
        )
        self.app.include_router(
            pipeline.router,
            prefix="/api/v1/pipeline",
            tags=["pipeline"]
        )
        self.app.include_router(
            query.router,
            prefix="/api/v1/query",
            tags=["query"]
        )
        self.app.include_router(
            version.router,
            prefix="/api/v1/versions",
            tags=["versions"]
        )
    
    def _setup_middleware(self):
        """Setup middleware"""
        from src.api.middleware.auth import AuthenticationMiddleware
        from src.api.middleware.metrics import MetricsMiddleware
        from src.api.middleware.rate_limit import RateLimitMiddleware
        
        self.app.add_middleware(AuthenticationMiddleware)
        self.app.add_middleware(MetricsMiddleware)
        self.app.add_middleware(RateLimitMiddleware)
    
    def _setup_error_handlers(self):
        """Setup error handlers"""
        @self.app.exception_handler(Exception)
        async def global_error_handler(
            request: Request,
            exc: Exception
        ) -> JSONResponse:
            # Handle error with recovery system
            try:
                recovery_result = await self.error_recovery.handle_error(
                    error=exc,
                    context={
                        "request": {
                            "method": request.method,
                            "url": str(request.url),
                            "headers": dict(request.headers),
                            "client": request.client.host
                        },
                        "timestamp": self.timestamp.isoformat(),
                        "user": self.current_user
                    },
                    severity=ErrorSeverity.HIGH,
                    category=ErrorCategory.APPLICATION
                )
                
                if recovery_result["status"] == "recovered":
                    return JSONResponse(
                        status_code=200,
                        content=recovery_result["result"]
                    )
                
            except Exception as recovery_error:
                logger.error(f"Error recovery failed: {recovery_error}")
            
            return JSONResponse(
                status_code=500,
                content={
                    "error": str(exc),
                    "timestamp": self.timestamp.isoformat()
                }
            )
    
    async def _startup(self):
        """Startup tasks"""
        try:
            # Initialize cache
            await self.cache.initialize()
            
            # Initialize load balancer
            await self.load_balancer.initialize()
            
            # Initialize security
            await self.security.initialize()
            
            # Initialize optimizer
            await self.optimizer.initialize()
            
            # Initialize version manager
            await self.version_manager.initialize()
            
            # Start monitoring
            await self.monitoring.start()
            
            logger.info("All components initialized successfully")
            
        except Exception as e:
            logger.error(f"Startup failed: {e}")
            raise
    
    async def _shutdown(self):
        """Shutdown tasks"""
        try:
            # Cleanup cache
            await self.cache.cleanup()
            
            # Cleanup load balancer
            await self.load_balancer.cleanup()
            
            # Cleanup security
            await self.security.cleanup()
            
            # Cleanup optimizer
            await self.optimizer.cleanup()
            
            # Cleanup version manager
            await self.version_manager.cleanup()
            
            # Stop monitoring
            await self.monitoring.stop()
            
            logger.info("All components cleaned up successfully")
            
        except Exception as e:
            logger.error(f"Shutdown failed: {e}")
            raise

# Create application instance
app = MCPServer().app