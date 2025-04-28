from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import chromadb
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = FastAPI(title="MVP Multi-Model RAG Server")

# Initialize ChromaDB
chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="documents")

# Model registry - in MVP we'll start with one model
models = {
    "llama": {
        "name": "meta-llama/Llama-2-7b-chat-hf",
        "model": None,
        "tokenizer": None
    }
}

class QueryRequest(BaseModel):
    prompt: str
    model_name: str = "llama"  # Default model
    use_rag: bool = False
    context_query: Optional[str] = None

class QueryResponse(BaseModel):
    response: str
    model_used: str
    rag_context: Optional[List[str]] = None

@app.on_event("startup")
async def load_models():
    """Load models on startup - In MVP we load just one model"""
    model_config = models["llama"]
    model_config["tokenizer"] = AutoTokenizer.from_pretrained(model_config["name"])
    model_config["model"] = AutoModelForCausalLM.from_pretrained(
        model_config["name"],
        torch_dtype=torch.float16,
        device_map="auto"
    )

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    if request.model_name not in models:
        raise HTTPException(status_code=400, detail="Model not found")
    
    context = []
    if request.use_rag and request.context_query:
        # Simple RAG implementation for MVP
        results = collection.query(
            query_texts=[request.context_query],
            n_results=2
        )
        context = results['documents'][0]
    
    # Prepare prompt with context if RAG is used
    final_prompt = (
        f"Context: {' '.join(context)}\n\n" if context else ""
    ) + request.prompt
    
    # Generate response using the selected model
    model_config = models[request.model_name]
    inputs = model_config["tokenizer"](
        final_prompt, 
        return_tensors="pt"
    ).to("cuda")
    
    outputs = model_config["model"].generate(
        **inputs,
        max_length=512,
        temperature=0.7
    )
    
    response = model_config["tokenizer"].decode(
        outputs[0], 
        skip_special_tokens=True
    )
    
    return QueryResponse(
        response=response,
        model_used=request.model_name,
        rag_context=context if context else None
    )

@app.post("/add_document")
async def add_document(document: str, metadata: dict = None):
    """Add documents to RAG database"""
    collection.add(
        documents=[document],
        metadatas=[metadata or {}],
        ids=[f"doc_{collection.count() + 1}"]
    )
    return {"status": "success"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)