from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List, Optional
from src.database.session import get_db
from src.services.versioning.model_version_manager import ModelVersionManager
from src.schemas.version import (
    VersionCreate,
    VersionResponse,
    VersionActivate,
    VersionRollback
)

router = APIRouter()

@router.post(
    "/versions/{model_name}",
    response_model=VersionResponse
)
async def create_version(
    model_name: str,
    version: VersionCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    version_manager: ModelVersionManager = Depends()
):
    try:
        # Create version
        result = await version_manager.create_version(
            model_name=model_name,
            version_data=version.data,
            metadata=version.metadata
        )
        
        # Schedule validation
        background_tasks.add_task(
            validate_version,
            db=db,
            model_name=model_name,
            version=result["version"]
        )
        
        return VersionResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post(
    "/versions/{model_name}/activate",
    response_model=VersionResponse
)
async def activate_version(
    model_name: str,
    activation: VersionActivate,
    db: Session = Depends(get_db),
    version_manager: ModelVersionManager = Depends()
):
    try:
        result = await version_manager.activate_version(
            model_name=model_name,
            version=activation.version
        )
        return VersionResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post(
    "/versions/{model_name}/rollback",
    response_model=VersionResponse
)
async def rollback_version(
    model_name: str,
    rollback: VersionRollback,
    db: Session = Depends(get_db),
    version_manager: ModelVersionManager = Depends()
):
    try:
        result = await version_manager.rollback_version(
            model_name=model_name,
            version=rollback.version,
            reason=rollback.reason
        )
        return VersionResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get(
    "/versions/{model_name}/history",
    response_model=List[VersionResponse]
)
async def get_version_history(
    model_name: str,
    limit: Optional[int] = 10,
    db: Session = Depends(get_db),
    version_manager: ModelVersionManager = Depends()
):
    try:
        versions = version_manager.versions.get(model_name, [])
        return [
            VersionResponse(**v)
            for v in sorted(
                versions,
                key=lambda x: x["created_at"],
                reverse=True
            )[:limit]
        ]
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))