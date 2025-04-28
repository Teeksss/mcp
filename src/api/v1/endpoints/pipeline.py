from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from src.services.rag.rag_service import RAGService
from src.services.intelligence.model_manager import mm

router = APIRouter(prefix="/api/v1/pipeline", tags=["Pipeline"])

class PipelineRequest(BaseModel):
    query: str
    model_key: str = "gpt4"
    top_k: int = 4

@router.post("/rag-inference")
def rag_inference(req: PipelineRequest):
    rag = RAGService()
    docs = rag.retrieve_docs(req.query, top_k=req.top_k)
    context = "\n".join([d['content'] for d in docs])
    prompt = f"Cevap için sadece aşağıdaki bağlamı kullan:\n{context}\n\nSoru: {req.query}\nCevap:"
    answer = mm.generate(req.model_key, prompt)
    return {
        "answer": answer,
        "docs": docs
    }