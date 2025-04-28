from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List, Optional
from src.database.session import get_db
from src.services.pipeline.orchestrator import PipelineOrchestrator
from src.schemas.pipeline import (
    PipelineRequest,
    PipelineResponse,
    PipelineConfig,
    PipelineMetrics
)

router = APIRouter()

@router.post("/execute", response_model=PipelineResponse)
async def execute_pipeline(
    request: PipelineRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    orchestrator: PipelineOrchestrator = Depends()
):
    try:
        # Execute pipeline
        result = await orchestrator.process_query(
            query=request.query,
            pipeline_config=request.config,
            context=request.context
        )
        
        # Schedule metrics update
        background_tasks.add_task(
            update_pipeline_metrics,
            db=db,
            pipeline_id=result["pipeline_id"],
            metrics=result["metrics"]
        )
        
        return PipelineResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/config", response_model=PipelineConfig)
async def update_pipeline_config(
    config: PipelineConfig,
    db: Session = Depends(get_db),
    orchestrator: PipelineOrchestrator = Depends()
):
    try:
        # Update configuration
        updated_config = await orchestrator.update_config(config.dict())
        return PipelineConfig(**updated_config)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/metrics", response_model=List[PipelineMetrics])
async def get_pipeline_metrics(
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    pipeline_type: Optional[str] = None,
    db: Session = Depends(get_db),
    orchestrator: PipelineOrchestrator = Depends()
):
    try:
        metrics = await orchestrator.get_metrics(
            start_time=start_time,
            end_time=end_time,
            pipeline_type=pipeline_type
        )
        return [PipelineMetrics(**m) for m in metrics]
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))