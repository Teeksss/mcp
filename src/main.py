from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.v1.endpoints import pipeline, rag_documents, llm, monitoring, auth

app = FastAPI(
    title="MCP Server",
    description="Akıllı Multi-Model + RAG + LLM Platform",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(pipeline.router)
app.include_router(rag_documents.router)
app.include_router(llm.router)
app.include_router(monitoring.router)
app.include_router(auth.router)