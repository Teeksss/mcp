from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from src.services.security.advanced_security import AdvancedSecurityManager
from src.config.settings import settings

class AuthenticationMiddleware(BaseHTTPMiddleware):
    def __init__(self, app):
        super().__init__(app)
        self.security_manager = AdvancedSecurityManager()
    
    async def dispatch(self, request: Request, call_next):
        try:
            # Extract token
            token = request.headers.get("Authorization")
            if not token:
                raise HTTPException(
                    status_code=401,
                    detail="Missing authentication token"
                )
            
            # Validate token
            user = await self.security_manager.validate_token(
                token.replace("Bearer ", "")
            )
            
            # Add user to request state
            request.state.user = user
            
            response = await call_next(request)
            return response
            
        except HTTPException as e:
            raise e
        except Exception as e:
            raise HTTPException(
                status_code=401,
                detail="Invalid authentication token"
            )