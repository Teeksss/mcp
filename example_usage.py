import requests

# Add a document for RAG
requests.post(
    "http://localhost:8000/add_document",
    json={
        "document": "AI models can process natural language queries.",
        "metadata": {"source": "example"}
    }
)

# Query the model without RAG
response = requests.post(
    "http://localhost:8000/query",
    json={
        "prompt": "What is machine learning?",
        "model_name": "llama",
        "use_rag": False
    }
)

# Query with RAG
response = requests.post(
    "http://localhost:8000/query",
    json={
        "prompt": "How do AI models handle queries?",
        "model_name": "llama",
        "use_rag": True,
        "context_query": "AI models processing"
    }
)