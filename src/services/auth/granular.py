from fastapi import Depends, HTTPException
from src.api.v1.endpoints.auth import get_current_user

def require_permission(module: str, action: str):
    def _wrapper(user=Depends(get_current_user)):
        perms = user.get("permissions", {})
        if action in perms.get(module, []) or "admin" in user.get("roles", []):
            return user
        raise HTTPException(status_code=403, detail="Yetkiniz yok")
    return _wrapper