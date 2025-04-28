from datetime import datetime
import asyncio
from typing import Dict, List, Optional
import yaml
import kubernetes
from kubernetes import client, config
from kubernetes.client.rest import ApiException
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class DeploymentConfig:
    namespace: str = "mcp-server"
    image_registry: str = "your-registry.azurecr.io"
    deployment_timeout: int = 600
    max_surge: str = "25%"
    max_unavailable: str = "25%"
    readiness_probe_delay: int = 30
    liveness_probe_delay: int = 60

class DeploymentManager:
    def __init__(self, config: DeploymentConfig):
        self.config = config
        self._setup_kubernetes()
    
    def _setup_kubernetes(self):
        """Initialize Kubernetes client"""
        try:
            config.load_incluster_config()
        except config.ConfigException:
            config.load_kube_config()
        
        self.k8s_apps = client.AppsV1Api()
        self.k8s_core = client.CoreV1Api()
    
    async def deploy_model(
        self,
        model_name: str,
        model_version: str,
        resource_requirements: Dict
    ):
        """Deploy a new model version"""
        try:
            deployment_name = f"model-{model_name}-{model_version}"
            
            # Create deployment configuration
            deployment = self._create_deployment_spec(
                deployment_name,
                model_name,
                model_version,
                resource_requirements
            )
            
            # Apply deployment
            try:
                self.k8s_apps.create_namespaced_deployment(
                    namespace=self.config.namespace,
                    body=deployment
                )
            except ApiException as e:
                if e.status == 409:  # Already exists
                    self.k8s_apps.patch_namespaced_deployment(
                        name=deployment_name,
                        namespace=self.config.namespace,
                        body=deployment
                    )
                else:
                    raise
            
            # Wait for deployment
            await self._wait_for_deployment(deployment_name)
            
            logger.info(
                f"Successfully deployed {model_name} version {model_version}"
            )
            
        except Exception as e:
            logger.error(f"Deployment failed: {e}")
            raise
    
    def _create_deployment_spec(
        self,
        deployment_name: str,
        model_name: str,
        model_version: str,
        resource_requirements: Dict
    ) -> Dict:
        """Create Kubernetes deployment specification"""
        return {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": deployment_name,
                "namespace": self.config.namespace,
                "labels": {
                    "app": "model-server",
                    "model": model_name,
                    "version": model_version
                }
            },
            "spec": {
                "replicas": resource_requirements.get("replicas", 1),
                "strategy": {
                    "type": "RollingUpdate",
                    "rollingUpdate": {
                        "maxSurge": self.config.max_surge,
                        "maxUnavailable": self.config.max_unavailable
                    }
                },
                "selector": {
                    "matchLabels": {
                        "app": "model-server",
                        "model": model_name,
                        "version": model_version
                    }
                },
                "template": {
                    "metadata": {
                        "labels": {
                            "app": "model-server",
                            "model": model_name,
                            "version": model_version
                        }
                    },
                    "spec": {
                        "containers": [{
                            "name": "model-server",
                            "image": f"{self.config.image_registry}/model-server:{model_version}",
                            "resources": {
                                "requests": resource_requirements.get("requests", {
                                    "cpu": "500m",
                                    "memory": "1Gi"
                                }),
                                "limits": resource_requirements.get("limits", {
                                    "cpu": "2",
                                    "memory": "4Gi",
                                    "nvidia.com/gpu": "1"
                                })
                            },
                            "readinessProbe": {
                                "httpGet": {
                                    "path": "/health",
                                    "port": 8000
                                },
                                "initialDelaySeconds": self.config.readiness_probe_delay,
                                "periodSeconds": 10
                            },
                            "livenessProbe": {
                                "httpGet": {
                                    "path": "/health",
                                    "port": 8000
                                },
                                "initialDelaySeconds": self.config.liveness_probe_delay,
                                "periodSeconds": 30
                            },
                            "env": [
                                {
                                    "name": "MODEL_NAME",
                                    "value": model_name
                                },
                                {
                                    "name": "MODEL_VERSION",
                                    "value": model_version
                                }
                            ]
                        }]
                    }
                }
            }
        }
    
    async def _wait_for_deployment(self, deployment_name: str):
        """Wait for deployment to be ready"""
        start_time = datetime.now()
        while True:
            try:
                deployment = self.k8s_apps.read_namespaced_deployment_status(
                    name=deployment_name,
                    namespace=self.config.namespace
                )
                
                if deployment.status.available_replicas == deployment.spec.replicas:
                    return
                
            except ApiException as e:
                logger.error(f"Error checking deployment status: {e}")
            
            if (datetime.now() - start_time).seconds > self.config.deployment_timeout:
                raise TimeoutError("Deployment timed out")
            
            await asyncio.sleep(5)