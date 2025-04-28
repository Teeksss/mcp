from src.services.rag_service import RAGService
from src.services.rag.data_connectors import fetch_from_web

rag = RAGService()
web_text = fetch_from_web("https://en.wikipedia.org/wiki/Retrieval-augmented_generation")
await rag.add_document({"content": web_text, "metadata": {"source": "wikipedia"}})