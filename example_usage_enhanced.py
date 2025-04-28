import requests

# Get access token
token_response = requests.post(
    "http://localhost:8000/token",
    data={"username": "test", "password": "test"}
)
access_token = token_response.json()["access_token"]

# Make authenticated request
headers = {"Authorization": f"Bearer {access_token}"}
response = requests.post(
    "http://localhost:8000/query",
    headers=headers,
    json={
        "prompt": "What is machine learning?",
        "model_name": "llama",
        "use_rag": True,
        "context_query": "machine learning fundamentals"
    }
)

# Check metrics
metrics = requests.get("http://localhost:8000/metrics")
print(metrics.text)