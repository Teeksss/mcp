class PipelineOrchestrator:
    def __init__(self, model_manager, rag_service, reranker=None):
        self.model_manager = model_manager
        self.rag_service = rag_service
        self.reranker = reranker

    async def run(self, query, config):
        model = self.model_manager.get(config["model"])
        context = []
        if config.get("use_rag"):
            context = await self.rag_service.retrieve(query)
        if self.reranker:
            context = self.reranker.rerank(query, context)
        answer = await self.model_manager.infer(model, query, context)
        return {"result": answer, "context": context}