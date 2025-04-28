from typing import Dict, List, Optional
import asyncio
import logging
from datetime import datetime
import yaml
import kubernetes
from kubernetes import client, config
import docker
import subprocess
import os
import json

logger = logging.getLogger(__name__)

class DeploymentManager:
    def __init__(self):
        self.k8s_config = self._load_k8s_config()
        self.docker_client = docker.from_env()
        
        # Current context
        self.timestamp = datetime.utcnow()
        self.current_user = "Teeksss"
        
        # Initialize Kubernetes client
        config.load_kube_config()
        self.k8s_api = client.CoreV1Api()
        self.k8s_apps_api = client.AppsV1Api()
    
    def _load_k8s_config(self) -> Dict:
        """Load Kubernetes configuration"""
        config_path = os.path.join(
            os.path.dirname(__file__),
            '../config/kubernetes'
        )
        
        with open(f"{config_path}/config.yaml", 'r') as f:
            return yaml.safe_load(f)
    
    async def deploy(
        self,
        version: str,
        environment: str = "production",
        rolling: bool = True
    ) -> Dict:
        """Deploy application to Kubernetes"""
        try:
            logger.info(f"Starting deployment of version {version}")
            
            # 1. Build Docker image
            image = await self._build_docker_image(version)
            
            # 2. Run tests
            await self._run_tests()
            
            # 3. Push image to registry
            await self._push_docker_image(image)
            
            # 4. Update Kubernetes manifests
            await self._update_k8s_manifests(version)
            
            # 5. Apply Kubernetes changes
            if rolling:
                await self._rolling_update(version)
            else:
                await self._apply_k8s_changes()
            
            # 6. Verify deployment
            await self._verify_deployment()
            
            logger.info(f"Deployment of version {version} completed")
            
            return {
                "status": "success",
                "version": version,
                "timestamp": self.timestamp.isoformat(),
                "environment": environment,
                "deployer": self.current_user
            }
            
        except Exception as e:
            logger.error(f"Deployment failed: {e}")
            await self._handle_deployment_failure(version)
            raise
    
    async def _build_docker_image(self, version: str) -> str:
        """Build Docker image"""
        try:
            logger.info("Building Docker image")
            
            # Build image
            image_name = f"mcp-server:{version}"
            self.docker_client.images.build(
                path=".",
                tag=image_name,
                dockerfile="docker/Dockerfile"
            )
            
            return image_name
            
        except Exception as e:
            logger.error(f"Docker build failed: {e}")
            raise
    
    async def _run_tests(self):
        """Run test suite"""
        try:
            logger.info("Running tests")
            
            result = subprocess.run(
                ["pytest", "tests/", "-v", "--junitxml=test-results.xml"],
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                raise Exception(f"Tests failed: {result.stderr}")
            
        except Exception as e:
            logger.error(f"Test execution failed: {e}")
            raise
    
    async def _push_docker_image(self, image: str):
        """Push Docker image to registry"""
        try:
            logger.info(f"Pushing image {image}")
            
            registry = self.k8s_config["registry"]
            repository = f"{registry}/{image}"
            
            # Tag image
            self.docker_client.images.get(image).tag(repository)
            
            # Push image
            for line in self.docker_client.images.push(
                repository,
                stream=True,
                decode=True
            ):
                if 'error' in line:
                    raise Exception(line['error'])
            
        except Exception as e:
            logger.error(f"Image push failed: {e}")
            raise
    
    async def _update_k8s_manifests(self, version: str):
        """Update Kubernetes manifests"""
        try:
            logger.info("Updating Kubernetes manifests")
            
            manifests_dir = "deployment/kubernetes"
            for filename in os.listdir(manifests_dir):
                if filename.endswith('.yaml'):
                    path = os.path.join(manifests_dir, filename)
                    
                    with open(path, 'r') as f:
                        manifest = yaml.safe_load(f)
                    
                    # Update image version
                    if manifest['kind'] in ['Deployment', 'StatefulSet']:
                        containers = manifest['spec']['template']['spec']['containers']
                        for container in containers:
                            if container['name'] == 'mcp-server':
                                container['image'] = f"mcp-server:{version}"
                    
                    with open(path, 'w') as f:
                        yaml.dump(manifest, f)
            
        except Exception as e:
            logger.error(f"Manifest update failed: {e}")
            raise
    
    async def _rolling_update(self, version: str):
        """Perform rolling update"""
        try:
            logger.info("Performing rolling update")
            
            # Update deployments
            deployments = self.k8s_apps_api.list_namespaced_deployment(
                namespace="mcp"
            )
            
            for deployment in deployments.items:
                if deployment.metadata.name.startswith('mcp-'):
                    # Update container image
                    deployment.spec.template.spec.containers[0].image = \
                        f"mcp-server:{version}"
                    
                    # Update deployment
                    self.k8s_apps_api.patch_namespaced_deployment(
                        name=deployment.metadata.name,
                        namespace="mcp",
                        body=deployment
                    )
                    
                    # Wait for rollout
                    await self._wait_for_rollout(deployment.metadata.name)
            
        except Exception as e:
            logger.error(f"Rolling update failed: {e}")
            raise
    
    async def _wait_for_rollout(self, deployment_name: str):
        """Wait for deployment rollout to complete"""
        while True:
            deployment = self.k8s_apps_api.read_namespaced_deployment(
                name=deployment_name,
                namespace="mcp"
            )
            
            if deployment.status.updated_replicas == deployment.spec.replicas:
                break
            
            await asyncio.sleep(5)
    
    async def _verify_deployment(self):
        """Verify deployment status"""
        try:
            logger.info("Verifying deployment")
            
            # Check pod status
            pods = self.k8s_api.list_namespaced_pod(
                namespace="mcp",
                label_selector="app=mcp-server"
            )
            
            for pod in pods.items:
                if pod.status.phase != 'Running':
                    raise Exception(
                        f"Pod {pod.metadata.name} is not running: "
                        f"{pod.status.phase}"
                    )
            
            # Check service health
            service = self.k8s_api.read_namespaced_service(
                name="mcp-server",
                namespace="mcp"
            )
            
            if not service:
                raise Exception("Service not found")
            
        except Exception as e:
            logger.error(f"Deployment verification failed: {e}")
            raise
    
    async def _handle_deployment_failure(self, version: str):
        """Handle deployment failure"""
        try:
            logger.info("Handling deployment failure")
            
            # Rollback deployments
            deployments = self.k8s_apps_api.list_namespaced_deployment(
                namespace="mcp"
            )
            
            for deployment in deployments.items:
                if deployment.metadata.name.startswith('mcp-'):
                    # Rollback to previous revision
                    self.k8s_apps_api.rollback_namespaced_deployment_rollback(
                        name=deployment.metadata.name,
                        namespace="mcp",
                        body=client.V1beta1DeploymentRollback(
                            name=deployment.metadata.name
                        )
                    )
            
            logger.info("Rollback completed")
            
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            raise