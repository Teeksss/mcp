from fastapi import Depends, HTTPException

def require_role(role: str):
    def wrapper(user=Depends(get_current_user)):
        if role not in user.roles:
            raise HTTPException(status_code=403, detail="Yetkiniz yok")
        return user
    return wrapper

def require_permission(permission: str):
    def wrapper(user=Depends(get_current_user)):
        if not user.permissions.get(permission, False):
            raise HTTPException(status_code=403, detail="Yetkiniz yok")
        return user
    return wrapper