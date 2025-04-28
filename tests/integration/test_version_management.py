import pytest
from fastapi.testclient import TestClient
from datetime import datetime
from src.services.versioning.model_version_manager import ModelVersionManager

@pytest.mark.asyncio
async def test_version_creation(
    client: TestClient,
    db_session,
    mock_version_manager
):
    # Test data
    model_name = "test_model"
    version_data = {
        "weights": "path/to/weights",
        "config": {"param1": "value1"}
    }
    
    # Create version
    response = await client.post(
        f"/api/v1/versions/{model_name}",
        json={
            "data": version_data,
            "metadata": {"author": "test_user"}
        }
    )
    
    assert response.status_code == 200
    assert "version" in response.json()
    assert response.json()["status"] == "created"

@pytest.mark.asyncio
async def test_version_activation(
    client: TestClient,
    db_session,
    mock_version_manager
):
    # Setup
    model_name = "test_model"
    version = "1.0.0"
    
    # Create and activate version
    await mock_version_manager.create_version(
        model_name,
        {"test": "data"}
    )
    
    response = await client.post(
        f"/api/v1/versions/{model_name}/activate",
        json={"version": version}
    )
    
    assert response.status_code == 200
    assert response.json()["status"] == "active"

@pytest.mark.asyncio
async def test_version_rollback(
    client: TestClient,
    db_session,
    mock_version_manager
):
    # Setup
    model_name = "test_model"
    version = "1.0.0"
    
    # Test rollback
    response = await client.post(
        f"/api/v1/versions/{model_name}/rollback",
        json={
            "version": version,
            "reason": "performance_degradation"
        }
    )
    
    assert response.status_code == 200
    assert response.json()["status"] == "active"
    assert "rolled_back_at" in response.json()