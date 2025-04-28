from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional

router = APIRouter(prefix="/api/v1/pipeline/modules", tags=["Modules"])

class Module(BaseModel):
    id: str
    name: str
    type: str
    status: str

# Dummy storage
modules = [
    Module(id="1", name="GPT-4", type="model", status="active"),
    Module(id="2", name="RAG", type="rag", status="active"),
]

@router.get("/", response_model=List[Module])
def list_modules():
    return modules

@router.post("/", response_model=Module)
def add_module(mod: Module):
    modules.append(mod)
    return mod

@router.put("/{mod_id}", response_model=Module)
def update_module(mod_id: str, mod: Module):
    for i, m in enumerate(modules):
        if m.id == mod_id:
            modules[i] = mod
            return mod
    raise HTTPException(404)

@router.delete("/{mod_id}")
def delete_module(mod_id: str):
    global modules
    modules = [m for m in modules if m.id != mod_id]
    return {"ok": True}