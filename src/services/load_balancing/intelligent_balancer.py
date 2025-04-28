from typing import Dict, List, Optional, Tuple
import asyncio
from datetime import datetime
import logging
import numpy as np
from prometheus_client import Counter, Gauge, Histogram
import aiohttp
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class NodeConfig:
    host: str
    port: int
    weight: float = 1.0
    max_concurrent: int = 100
    health_check_interval: int = 30
    scaling_threshold: float = 0.8

@dataclass
class BalancerConfig:
    algorithm: str = "adaptive"  # adaptive, round_robin, least_conn
    health_check_path: str = "/health"
    scaling_enabled: bool = True
    min_nodes: int = 2
    max_nodes: int = 10

class IntelligentLoadBalancer:
    def __init__(
        self,
        config: BalancerConfig
    ):
        self.config = config
        self.nodes: Dict[str, NodeConfig] = {}
        self.node_stats: Dict[str, Dict] = {}
        self.metrics = self._setup_metrics()
        
        # State tracking
        self.active_connections = {}
        self.node_health = {}
        self.last_node_index = 0
        
        # Current context
        self.timestamp = datetime.utcnow()
        self.current_user = "Teeksss"
        
        # Start monitoring
        asyncio.create_task(self._monitor_nodes())
    
    def _setup_metrics(self) -> Dict:
        """Initialize load balancing metrics"""
        return {
            "requests_total": Counter(
                "lb_requests_total",
                "Total requests handled",
                ["node", "status"]
            ),
            "active_connections": Gauge(
                "lb_active_connections",
                "Current active connections",
                ["node"]
            ),
            "response_time": Histogram(
                "lb_response_time_seconds",
                "Request response time",
                ["node"]
            ),
            "node_health": Gauge(
                "lb_node_health",
                "Node health status",
                ["node"]
            ),
            "load_score": Gauge(
                "lb_load_score",
                "Node load score",
                ["node"]
            )
        }
    
    async def register_node(
        self,
        node_id: str,
        config: NodeConfig
    ):
        """Register new node"""
        try:
            self.nodes[node_id] = config
            self.node_stats[node_id] = {
                "total_requests": 0,
                "success_rate": 1.0,
                "avg_response_time": 0.0,
                "last_health_check": None
            }
            
            # Initialize metrics
            self.metrics["node_health"].labels(node=node_id).set(1)
            self.metrics["active_connections"].labels(node=node_id).set(0)
            
            logger.info(f"Registered node: {node_id}")
            
        except Exception as e:
            logger.error(f"Node registration failed: {e}")
            raise
    
    async def route_request(
        self,
        request_data: Dict,
        request_type: str
    ) -> Tuple[str, str]:
        """Route request to optimal node"""
        try:
            # Select best node
            node_id = await self._select_node(request_data, request_type)
            if not node_id:
                raise RuntimeError("No healthy nodes available")
            
            # Get node config
            node = self.nodes[node_id]
            
            # Update metrics
            self.metrics["requests_total"].labels(
                node=node_id,
                status="routed"
            ).inc()
            
            # Generate endpoint
            endpoint = f"http://{node.host}:{node.port}"
            
            return node_id, endpoint
            
        except Exception as e:
            logger.error(f"Request routing failed: {e}")
            raise
    
    async def _select_node(
        self,
        request_data: Dict,
        request_type: str
    ) -> Optional[str]:
        """Select optimal node based on algorithm"""
        try:
            if self.config.algorithm == "adaptive":
                return await self._adaptive_selection(
                    request_data,
                    request_type
                )
            elif self.config.algorithm == "round_robin":
                return self._round_robin_selection()
            elif self.config.algorithm == "least_conn":
                return self._least_connections_selection()
            else:
                raise ValueError(f"Unknown algorithm: {self.config.algorithm}")
            
        except Exception as e:
            logger.error(f"Node selection failed: {e}")
            raise
    
    async def _adaptive_selection(
        self,
        request_data: Dict,
        request_type: str
    ) -> Optional[str]:
        """Adaptive node selection based on multiple factors"""
        scores = {}
        
        for node_id, node in self.nodes.items():
            if not self._is_node_healthy(node_id):
                continue
            
            # Calculate load score
            load_score = await self._calculate_load_score(node_id)
            
            # Calculate performance score
            perf_score = self._calculate_performance_score(node_id)
            
            # Calculate request compatibility score
            compat_score = self._calculate_compatibility_score(
                node_id,
                request_type
            )
            
            # Calculate final score
            scores[node_id] = (
                load_score * 0.4 +
                perf_score * 0.4 +
                compat_score * 0.2
            )
            
            # Update metrics
            self.metrics["load_score"].labels(node=node_id).set(scores[node_id])
        
        if not scores:
            return None
        
        # Select node with highest score
        return max(scores.items(), key=lambda x: x[1])[0]
    
    def _round_robin_selection(self) -> Optional[str]:
        """Simple round-robin selection"""
        healthy_nodes = [
            node_id for node_id in self.nodes
            if self._is_node_healthy(node_id)
        ]
        
        if not healthy_nodes:
            return None
        
        self.last_node_index = (
            self.last_node_index + 1
        ) % len(healthy_nodes)
        
        return healthy_nodes[self.last_node_index]
    
    def _least_connections_selection(self) -> Optional[str]:
        """Select node with least active connections"""
        scores = {}
        
        for node_id in self.nodes:
            if not self._is_node_healthy(node_id):
                continue
            
            active_conns = self.active_connections.get(node_id, 0)
            max_conns = self.nodes[node_id].max_concurrent
            
            scores[node_id] = 1 - (active_conns / max_conns)
        
        if not scores:
            return None
        
        return max(scores.items(), key=lambda x: x[1])[0]
    
    async def _calculate_load_score(self, node_id: str) -> float:
        """Calculate node load score"""
        active_conns = self.active_connections.get(node_id, 0)
        max_conns = self.nodes[node_id].max_concurrent
        
        # Get CPU and memory usage
        node_metrics = await self._get_node_metrics(node_id)
        
        # Calculate weighted score
        conn_score = 1 - (active_conns / max_conns)
        cpu_score = 1 - node_metrics.get("cpu_usage", 0)
        mem_score = 1 - node_metrics.get("memory_usage", 0)
        
        return (
            conn_score * 0.4 +
            cpu_score * 0.3 +
            mem_score * 0.3
        )
    
    def _calculate_performance_score(self, node_id: str) -> float:
        """Calculate node performance score"""
        stats = self.node_stats[node_id]
        
        # Normalize response time (0-1, lower is better)
        resp_time_score = 1 - min(
            stats["avg_response_time"] / 1.0,  # 1 second baseline
            1.0
        )
        
        return (
            stats["success_rate"] * 0.6 +
            resp_time_score * 0.4
        )
    
    def _calculate_compatibility_score(
        self,
        node_id: str,
        request_type: str
    ) -> float:
        """Calculate request-node compatibility score"""
        # Implementation depends on node capabilities
        return 1.0
    
    def _is_node_healthy(self, node_id: str) -> bool:
        """Check if node is healthy"""
        last_check = self.node_health.get(node_id, {}).get("last_check")
        if not last_check:
            return False
        
        # Check if health check is recent
        time_since_check = (
            self.timestamp - last_check
        ).total_seconds()
        
        return (
            time_since_check <= self.nodes[node_id].health_check_interval * 2 and
            self.node_health[node_id]["status"] == "healthy"
        )
    
    async def _monitor_nodes(self):
        """Continuous node monitoring"""
        while True:
            try:
                for node_id, node in self.nodes.items():
                    # Perform health check
                    health = await self._check_node_health(node_id)
                    
                    # Update metrics
                    self.metrics["node_health"].labels(
                        node=node_id
                    ).set(1 if health["status"] == "healthy" else 0)
                    
                    # Check scaling
                    if self.config.scaling_enabled:
                        await self._check_scaling_needs()
                
            except Exception as e:
                logger.error(f"Node monitoring failed: {e}")
            
            await asyncio.sleep(min(
                node.health_check_interval
                for node in self.nodes.values()
            ))
    
    async def _check_node_health(self, node_id: str) -> Dict:
        """Check health of specific node"""
        try:
            node = self.nodes[node_id]
            health_url = f"http://{node.host}:{node.port}{self.config.health_check_path}"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(health_url) as response:
                    status = "healthy" if response.status == 200 else "unhealthy"
                    
                    self.node_health[node_id] = {
                        "status": status,
                        "last_check": self.timestamp,
                        "details": await response.json()
                    }
                    
                    return self.node_health[node_id]
                    
        except Exception as e:
            logger.error(f"Health check failed for node {node_id}: {e}")
            self.node_health[node_id] = {
                "status": "unhealthy",
                "last_check": self.timestamp,
                "error": str(e)
            }
            return self.node_health[node_id]
    
    async def _check_scaling_needs(self):
        """Check if scaling is needed"""
        try:
            # Calculate average load
            total_load = sum(
                await self._calculate_load_score(node_id)
                for node_id in self.nodes
            )
            avg_load = total_load / len(self.nodes)
            
            # Check scaling conditions
            if (
                avg_load > self.nodes[
                    next(iter(self.nodes))
                ].scaling_threshold and
                len(self.nodes) < self.config.max_nodes
            ):
                await self._scale_up()
            elif (
                avg_load < 0.3 and
                len(self.nodes) > self.config.min_nodes
            ):
                await self._scale_down()
            
        except Exception as e:
            logger.error(f"Scaling check failed: {e}")
    
    async def _scale_up(self):
        """Scale up by adding new node"""
        # Implementation depends on infrastructure
        pass
    
    async def _scale_down(self):
        """Scale down by removing node"""
        # Implementation depends on infrastructure
        pass
    
    async def _get_node_metrics(self, node_id: str) -> Dict:
        """Get node metrics"""
        # Implementation depends on monitoring system
        pass