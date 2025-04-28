from fastapi import APIRouter
from fastapi.responses import EventSourceResponse
import asyncio
import random

router = APIRouter(prefix="/api/v1/logs", tags=["Logs"])

@router.get("/stream")
async def log_stream():
    async def event_generator():
        levels = ["INFO", "WARNING", "ERROR"]
        while True:
            await asyncio.sleep(1)
            yield f"data: {random.choice(levels)}: Dummy log satırı\n\n"
    return EventSourceResponse(event_generator())