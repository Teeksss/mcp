def get_vector_store(backend, config):
    if backend == "chromadb":
        # Dummy: Basit in-memory örnek
        class DummyStore:
            def __init__(self):
                self.docs = []
            def insert(self, docs):
                self.docs.extend(docs)
                return True
            def get_all(self):
                return self.docs
            def similarity_search(self, query, k=4):
                # Kısa içerik benzerliği için dummy
                return sorted(self.docs, key=lambda d: -d["content"].count(query))[:k]
        return DummyStore()
    elif backend == "weaviate":
        # Weaviate client örneği buraya eklenir
        pass
    else:
        raise NotImplementedError("Desteklenmeyen backend")