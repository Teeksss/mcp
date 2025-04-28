from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List, Optional
from src.database.session import get_db
from src.services.intelligence.model_manager import ModelManager
from src.services.security.advanced_security import AdvancedSecurityManager
from src.schemas import (
    QueryRequest,
    QueryResponse,
    ModelCreate,
    ModelResponse
)

router = APIRouter()

@router.post("/query", response_model=QueryResponse)
async def process_query(
    request: QueryRequest,
    db: Session = Depends(get_db),
    model_manager: ModelManager = Depends(),
    security: AdvancedSecurityManager = Depends()
):
    try:
        # Validate request
        await security.validate_request(request)
        
        # Process query
        result = await model_manager.process_query(
            query=request.query,
            model_name=request.model_name,
            context=request.context
        )
        
        return QueryResponse(
            result=result,
            model=request.model_name,
            processing_time=result.processing_time
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/models", response_model=ModelResponse)
async def create_model(
    model: ModelCreate,
    db: Session = Depends(get_db),
    model_manager: ModelManager = Depends(),
    security: AdvancedSecurityManager = Depends()
):
    try:
        # Create model
        result = await model_manager.create_model(
            name=model.name,
            version=model.version,
            config=model.config
        )
        
        return ModelResponse(
            id=result.id,
            name=result.name,
            version=result.version,
            status="created"
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/models/{model_id}/metrics")
async def get_model_metrics(
    model_id: int,
    db: Session = Depends(get_db),
    model_manager: ModelManager = Depends(),
    security: AdvancedSecurityManager = Depends()
):
    try:
        metrics = await model_manager.get_model_metrics(model_id)
        return metrics
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))