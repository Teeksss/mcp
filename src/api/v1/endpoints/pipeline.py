from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from src.services.pipeline.orchestrator import PipelineOrchestrator

router = APIRouter(prefix="/api/v1/pipeline", tags=["Pipeline"])

class PipelineConfig(BaseModel):
    model_type: str
    max_tokens: Optional[int] = 100
    temperature: Optional[float] = 0.7

class PipelineRequest(BaseModel):
    query: str
    config: PipelineConfig

class PipelineResponse(BaseModel):
    result: Any
    processing_time: float
    status: str = "success"
    error: Optional[str] = None

@router.post("/execute", response_model=PipelineResponse)
async def execute_pipeline(request: PipelineRequest):
    orchestrator = PipelineOrchestrator()
    try:
        result = orchestrator.execute(request.query, request.config.dict())
        return PipelineResponse(
            result=result,
            processing_time=0.12
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))