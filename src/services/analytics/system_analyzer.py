from typing import Dict, List, Optional
import asyncio
from datetime import datetime, timedelta
import logging
import numpy as np
import pandas as pd
from prometheus_client import Gauge

logger = logging.getLogger(__name__)

class SystemAnalyzer:
    def __init__(
        self,
        event_bus,
        monitoring,
        components: Dict
    ):
        self.event_bus = event_bus
        self.monitoring = monitoring
        self.components = components
        
        self.metrics = self._setup_metrics()
        self.analysis_state = {}
        
        # Start analysis loop
        asyncio.create_task(self._analysis_loop())
        
        # Current context
        self.timestamp = datetime.utcnow()
        self.current_user = "Teeksss"
    
    def _setup_metrics(self) -> Dict:
        return {
            "system_health_score": Gauge(
                "system_health_score",
                "Overall system health score",
                ["component"]
            ),
            "performance_score": Gauge(
                "performance_score",
                "System performance score",
                ["component"]
            ),
            "reliability_score": Gauge(
                "reliability_score",
                "System reliability score",
                ["component"]
            )
        }
    
    async def _analysis_loop(self):
        """Continuous system analysis loop"""
        while True:
            try:
                # Perform analysis
                analysis = await self.analyze_system()
                
                # Update metrics
                self._update_metrics(analysis)
                
                # Check for issues
                if issues := self._detect_issues(analysis):
                    await self._handle_issues(issues)
                
                # Wait for next interval
                await asyncio.sleep(300)  # 5 minutes
                
            except Exception as e:
                logger.error(f"Analysis loop failed: {e}")
                await asyncio.sleep(60)  # Wait 1 minute on error
    
    async def analyze_system(self) -> Dict:
        """Analyze system state"""
        try:
            analysis = {
                "timestamp": self.timestamp.isoformat(),
                "components": {},
                "overall": {}
            }
            
            # Analyze each component
            for name, component in self.components.items():
                analysis["components"][name] = await self._analyze_component(
                    name,
                    component
                )
            
            # Calculate overall metrics
            analysis["overall"] = self._calculate_overall_metrics(
                analysis["components"]
            )
            
            return analysis
            
        except Exception as e:
            logger.error(f"System analysis failed: {e}")
            raise
    
    async def _analyze_component(
        self,
        name: str,
        component: Any
    ) -> Dict:
        """Analyze individual component"""
        metrics = component.metrics if hasattr(component, "metrics") else {}
        
        return {
            "health": await self._calculate_health_score(name, metrics),
            "performance": await self._calculate_performance_score(name, metrics),
            "reliability": await self._calculate_reliability_score(name, metrics),
            "metrics": metrics
        }
    
    def _calculate_overall_metrics(
        self,
        component_analysis: Dict
    ) -> Dict:
        """Calculate overall system metrics"""
        scores = {
            "health": [],
            "performance": [],
            "reliability": []
        }
        
        for component in component_analysis.values():
            scores["health"].append(component["health"])
            scores["performance"].append(component["performance"])
            scores["reliability"].append(component["reliability"])
        
        return {
            "health": np.mean(scores["health"]),
            "performance": np.mean(scores["performance"]),
            "reliability": np.mean(scores["reliability"])
        }
    
    def _detect_issues(self, analysis: Dict) -> List[Dict]:
        """Detect system issues"""
        issues = []
        
        # Check overall health
        if analysis["overall"]["health"] < 0.8:
            issues.append({
                "type": "system_health",
                "severity": "high",
                "description": "System health below threshold",
                "score": analysis["overall"]["health"]
            })
        
        # Check component issues
        for name, component in analysis["components"].items():
            if component["health"] < 0.7:
                issues.append({
                    "type": "component_health",
                    "component": name,
                    "severity": "high",
                    "description": f"Component {name} health critical",
                    "score": component["health"]
                })
        
        return issues
    
    async def _handle_issues(self, issues: List[Dict]):
        """Handle detected issues"""
        for issue in issues:
            # Publish issue event
            await self.event_bus.publish(
                "system_issue_detected",
                {
                    "issue": issue,
                    "timestamp": self.timestamp.isoformat(),
                    "detected_by": self.current_user
                }
            )