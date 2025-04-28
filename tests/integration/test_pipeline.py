import pytest
from fastapi.testclient import TestClient
from datetime import datetime, timedelta
from src.services.pipeline.orchestrator import PipelineOrchestrator

@pytest.mark.asyncio
async def test_pipeline_execution(
    client: TestClient,
    db_session,
    mock_model_manager,
    mock_rag_enhancer
):
    # Prepare test data
    test_query = "What is the meaning of life?"
    test_config = {
        "model_type": "gpt-4",
        "max_tokens": 100,
        "temperature": 0.7
    }
    
    # Execute pipeline
    response = await client.post(
        "/api/v1/pipeline/execute",
        json={
            "query": test_query,
            "config": test_config
        }
    )
    
    assert response.status_code == 200
    assert "result" in response.json()
    assert "processing_time" in response.json()

@pytest.mark.asyncio
async def test_pipeline_error_recovery(
    client: TestClient,
    db_session,
    mock_model_manager,
    mock_error_recovery
):
    # Simulate error condition
    mock_model_manager.execute_model.side_effect = Exception("Model failed")
    
    # Execute pipeline
    response = await client.post(
        "/api/v1/pipeline/execute",
        json={
            "query": "Test query",
            "config": {"model_type": "gpt-4"}
        }
    )
    
    assert response.status_code == 200
    assert response.json()["status"] == "recovered"

@pytest.mark.asyncio
async def test_pipeline_performance_optimization(
    client: TestClient,
    db_session,
    mock_performance_optimizer
):
    # Test optimization
    response = await client.post(
        "/api/v1/pipeline/optimize",
        json={
            "component_id": "test_model",
            "performance_data": {
                "latency": [100, 150, 200],
                "throughput": [10, 15, 20]
            }
        }
    )
    
    assert response.status_code == 200
    assert "optimization" in response.json()