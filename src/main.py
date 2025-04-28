import uvicorn
import os
from src.app import create_app

app = create_app()

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=os.getenv("ENV") == "development",
        workers=int(os.getenv("WORKERS", 4)),
        log_level=os.getenv("LOG_LEVEL", "info")
    )