#!/bin/bash

# Deployment automation script
set -e

# Configuration
DEPLOY_ENV=${1:-production}
VERSION=${2:-latest}
NAMESPACE="mcp-${DEPLOY_ENV}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Functions
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%dT%H:%M:%S%z')] $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%dT%H:%M:%S%z')] ERROR: $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%dT%H:%M:%S%z')] WARNING: $1${NC}"
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check kubectl
    if ! command -v kubectl &> /dev/null; then
        error "kubectl is not installed"
        exit 1
    fi
    
    # Check docker
    if ! command -v docker &> /dev/null; then
        error "docker is not installed"
        exit 1
    }
    
    # Check environment
    if [[ ! "$DEPLOY_ENV" =~ ^(production|staging)$ ]]; then
        error "Invalid environment: $DEPLOY_ENV"
        exit 1
    }
}

# Build application
build_application() {
    log "Building application..."
    
    # Build Docker image
    docker build \
        -t "mcp-server:${VERSION}" \
        -f docker/Dockerfile \
        --build-arg BUILD_ENV="$DEPLOY_ENV" \
        .
    
    if [ $? -ne 0 ]; then
        error "Docker build failed"
        exit 1
    fi
}

# Run tests
run_tests() {
    log "Running tests..."
    
    # Run pytest
    python -m pytest tests/ -v --junitxml=test-results.xml
    
    if [ $? -ne 0 ]; then
        error "Tests failed"
        exit 1
    }
}

# Deploy to Kubernetes
deploy_to_kubernetes() {
    log "Deploying to Kubernetes..."
    
    # Update kube config
    kubectl config use-context "mcp-${DEPLOY_ENV}-cluster"
    
    # Apply configurations
    kubectl apply -f deployment/kubernetes/ -n "$NAMESPACE"
    
    if [ $? -ne 0 ]; then
        error "Kubernetes deployment failed"
        exit 1
    }
    
    # Wait for rollout
    kubectl rollout status deployment/mcp-server -n "$NAMESPACE" --timeout=300s
    
    if [ $? -ne 0 ]; then
        error "Deployment rollout failed"
        exit 1
    }
}

# Verify deployment
verify_deployment() {
    log "Verifying deployment..."
    
    # Check pods
    READY_PODS=$(kubectl get pods -n "$NAMESPACE" -l app=mcp-server -o json | jq '.items[] | select(.status.phase=="Running") | .metadata.name' | wc -l)
    TOTAL_PODS=$(kubectl get pods -n "$NAMESPACE" -l app=mcp-server --no-headers | wc -l)
    
    if [ "$READY_PODS" -ne "$TOTAL_PODS" ]; then
        error "Not all pods are running"
        exit 1
    }
    
    # Check service
    kubectl get service mcp-server -n "$NAMESPACE" &> /dev/null
    
    if [ $? -ne 0 ]; then
        error "Service not found"
        exit 1
    }
}

# Main deployment process
main() {
    log "Starting deployment process..."
    
    # Run steps
    check_prerequisites
    build_application
    run_tests
    deploy_to_kubernetes
    verify_deployment
    
    log "Deployment completed successfully!"
}

# Run main process
main
