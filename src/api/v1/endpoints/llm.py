from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from src.services.intelligence.model_manager import mm

router = APIRouter(prefix="/api/v1/llm", tags=["LLM"])

class LLMRequest(BaseModel):
    model_key: str = Field(..., description="Kullanılacak model anahtarı")
    prompt: str = Field(..., description="Sorgu veya istem")
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.95

@router.post("/generate")
def generate_llm(req: LLMRequest):
    try:
        result = mm.generate(
            req.model_key,
            prompt=req.prompt,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
        )
        return {"result": result}
    except Exception as ex:
        raise HTTPException(status_code=500, detail=str(ex))