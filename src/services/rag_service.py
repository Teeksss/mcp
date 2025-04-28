class RAGService:
    # ... init vs.
    async def add_document(self, document):
        # Gerekirse embedding hesapla
        embedding = await self._embed(document.content)
        return await self._add_to_vector_store(document.content, document.metadata, embedding)
    
    async def _embed(self, text):
        # HF veya OpenAI gibi modelden embedding al
        pass

    async def _add_to_vector_store(self, content, metadata, embedding):
        # VectorDB (ChromaDB, FAISS, Pinecone vs.) ekleme işlemi
        pass