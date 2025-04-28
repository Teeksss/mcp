# (Opsiyonel) Reranker stub. Geliştirilebilir.
class Reranker:
    def rerank(self, docs, query):
        # Skorla ve sırala (dummy örnek)
        return sorted(docs, key=lambda d: -len(d.get("content", "")))