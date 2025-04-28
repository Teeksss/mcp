from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)

def test_pipeline_execute():
    response = client.post("/api/v1/pipeline/execute", json={
        "query": "Merhaba Dünya!",
        "config": {"model_type": "llama", "max_tokens": 100, "temperature": 0.7}
    })
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "result" in data