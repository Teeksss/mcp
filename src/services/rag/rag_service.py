from src.services.rag.vector_store_factory import get_vector_store
from src.services.rag.data_connectors import process_file, fetch_web_content

class RAGService:
    def __init__(self, backend="chromadb"):
        self.store = get_vector_store(backend, {})

    async def add_document(self, doc):
        return self.store.insert([{"content": doc.content, "metadata": doc.metadata}])

    async def list_documents(self):
        return self.store.get_all()

    async def ingest_file(self, file, ocr_lang):
        docs = await process_file(file, ocr_lang)
        for d in docs:
            await self.add_document(d)
        return {"status": "success", "count": len(docs)}

    def retrieve_docs(self, query, top_k=4):
        return self.store.similarity_search(query, k=top_k)