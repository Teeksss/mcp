from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)

def test_pipeline_execute_success():
    payload = {
        "query": "Dünya'nın en yüksek dağı nedir?",
        "config": {"model_type": "gpt-4", "max_tokens": 50, "temperature": 0.6}
    }
    response = client.post("/api/v1/pipeline/execute", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "result" in data
    assert "processing_time" in data
    assert data["status"] == "success"