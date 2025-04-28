from fastapi import APIRouter
import platform, socket, os, psutil

router = APIRouter(prefix="/api/v1/monitoring", tags=["Monitoring"])

@router.get("/health")
def health():
    return {"status": "ok"}

@router.get("/system-info")
def system_info():
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "cpu_percent": psutil.cpu_percent(),
        "memory": psutil.virtual_memory()._asdict(),
        "loadavg": os.getloadavg()
    }