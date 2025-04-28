from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Query
from pydantic import BaseModel
from typing import Optional, List
from src.services.rag.rag_service import RAGService

router = APIRouter(prefix="/api/v1/rag", tags=["RAG"])

class DocumentIn(BaseModel):
    content: str
    metadata: dict = {}

@router.post("/documents/add")
async def add_document(doc: DocumentIn):
    rag = RAGService()
    ok = await rag.add_document(doc)
    if not ok:
        raise HTTPException(status_code=500, detail="Döküman eklenemedi")
    return {"status": "success"}

@router.get("/documents/list")
async def list_documents(backend: str = Query("chromadb")):
    rag = RAGService(backend=backend)
    docs = await rag.list_documents()
    return docs

@router.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    backend: str = Form("chromadb"),
    ocr_lang: str = Form("eng+tur")
):
    rag = RAGService(backend=backend)
    return await rag.ingest_file(file, ocr_lang)