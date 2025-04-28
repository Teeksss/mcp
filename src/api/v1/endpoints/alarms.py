from fastapi import APIRouter
from pydantic import BaseModel
from typing import List
import time, asyncio

router = APIRouter(prefix="/api/v1/alarms", tags=["Alarms"])

class Alarm(BaseModel):
    id: int
    level: str  # "critical", "warning", "info"
    message: str
    time: float

alarms: List[Alarm] = [
    Alarm(id=1, level="critical", message="GPT-4 endpoint down!", time=time.time()),
    Alarm(id=2, level="warning", message="Yüksek gecikme", time=time.time()),
]

@router.get("/", response_model=List[Alarm])
def get_alarms():
    return alarms

@router.get("/stream")
async def alarm_stream():
    from fastapi.responses import EventSourceResponse
    async def generator():
        idx = len(alarms)
        while True:
            await asyncio.sleep(2)
            if len(alarms) > idx:
                yield f"data: {alarms[-1].json()}\n\n"
                idx += 1
    return EventSourceResponse(generator())