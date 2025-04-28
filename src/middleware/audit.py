from starlette.middleware.base import BaseHTTPMiddleware
import time

class AuditLogMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        start = time.time()
        response = await call_next(request)
        end = time.time()
        user = request.headers.get("X-User", "unknown")  # JWT'den de çıkarılabilir
        log_entry = {
            "user": user,
            "path": request.url.path,
            "method": request.method,
            "status": response.status_code,
            "duration": end - start,
            "timestamp": start,
        }
        print(log_entry)  # Gerçek ortamda log dosyasına veya DB'ye yaz
        return response