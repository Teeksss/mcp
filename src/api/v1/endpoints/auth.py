from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from src.core.rbac import check_permission

router = APIRouter(prefix="/api/v1/auth", tags=["Auth"])

class LoginRequest(BaseModel):
    username: str
    password: str

@router.post("/login")
def login(req: LoginRequest):
    if req.username == "admin" and req.password == "123456":
        return {"token": "dummy-token", "role": "admin"}
    raise HTTPException(status_code=401, detail="Geçersiz kullanıcı")

@router.get("/check")
def check(token: str = Depends(check_permission)):
    return {"status": "ok"}