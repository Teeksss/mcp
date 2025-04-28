from fastapi import APIRouter, Depends
from typing import Dict
from src.services.health.health_checker import HealthChecker
from src.schemas.health import SystemHealth, ComponentHealth

router = APIRouter()

@router.get(
    "/health",
    response_model=SystemHealth
)
async def get_system_health(
    health_checker: HealthChecker = Depends()
):
    """Get system health status"""
    return await health_checker.get_system_status()

@router.get(
    "/health/{component_name}",
    response_model=ComponentHealth
)
async def get_component_health(
    component_name: str,
    health_checker: HealthChecker = Depends()
):
    """Get specific component health"""
    status = await health_checker.check_health()
    return status.get(component_name, {
        "status": "not_found",
        "error": f"Component {component_name} not found"
    })