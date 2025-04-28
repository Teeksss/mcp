from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from jose import JWTError, jwt
from passlib.context import CryptContext
from datetime import datetime, timedelta
import redis
import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration
import logging
import time
import asyncio
from typing import Dict, List, Optional
import prometheus_client
from prometheus_client import Counter, Histogram
import chromadb
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from concurrent.futures import ThreadPoolExecutor
import functools

from config import settings
from models import User, TokenData, CacheConfig, ModelConfig

# Initialize Sentry for error tracking
sentry_sdk.init(
    dsn=settings.SENTRY_DSN,
    integrations=[FastApiIntegration()],
    traces_sample_rate=1.0,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Redis for caching
redis_client = redis.from_url(settings.REDIS_URL)

# Initialize Prometheus metrics
REQUEST_COUNT = Counter('request_count', 'Number of requests processed')
RESPONSE_TIME = Histogram('response_time_seconds', 'Response time in seconds')
MODEL_ERRORS = Counter('model_errors', 'Number of model errors')
CACHE_HITS = Counter('cache_hits', 'Number of cache hits')
RAG_RETRIEVAL_TIME = Histogram('rag_retrieval_time', 'RAG retrieval time in seconds')

app = FastAPI(title="Enhanced Multi-Model RAG Server")

# Security
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Initialize ChromaDB with better configuration
chroma_client = chromadb.Client(settings={
    "chroma_db_impl": "duckdb+parquet",
    "persist_directory": "./chroma_db"
})
collection = chroma_client.get_or_create_collection(
    name="documents",
    metadata={"hnsw:space": "cosine", "hnsw:construction_ef": 100}
)

# Enhanced model registry with fallback configuration
models: Dict[str, ModelConfig] = {
    "llama": {
        "name": "meta-llama/Llama-2-7b-chat-hf",
        "fallback_models": ["mistral", "phi"],
        "timeout": 30,
        "max_retries": 3,
        "cache_config": {"ttl": 3600, "max_size": 1000}
    }
}

# Model instance pool
model_pool = {}
model_locks = {}

async def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    credentials_exception = HTTPException(
        status_code=401,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(
            token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM]
        )
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    # In production, fetch user from database
    user = User(username=token_data.username, hashed_password="")
    return user

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(
        to_encode, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM
    )
    return encoded_jwt

@app.post("/token")
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    # In production, validate against database
    if form_data.username != "test" or form_data.password != "test":
        raise HTTPException(status_code=400, detail="Incorrect username or password")
    access_token = create_access_token(data={"sub": form_data.username})
    return {"access_token": access_token, "token_type": "bearer"}

async def cache_response(cache_key: str, response: str, ttl: int):
    try:
        await redis_client.setex(cache_key, ttl, response)
    except Exception as e:
        logger.error(f"Cache error: {e}")
        sentry_sdk.capture_exception(e)

async def get_cached_response(cache_key: str) -> Optional[str]:
    try:
        cached = await redis_client.get(cache_key)
        if cached:
            CACHE_HITS.inc()
            return cached.decode()
    except Exception as e:
        logger.error(f"Cache retrieval error: {e}")
        sentry_sdk.capture_exception(e)
    return None

class RateLimiter:
    def __init__(self, requests_per_minute: int):
        self.requests_per_minute = requests_per_minute
        self.requests = {}

    async def check(self, user_id: str) -> bool:
        now = time.time()
        user_requests = self.requests.get(user_id, [])
        user_requests = [req for req in user_requests if req > now - 60]
        
        if len(user_requests) >= self.requests_per_minute:
            return False
        
        user_requests.append(now)
        self.requests[user_id] = user_requests
        return True

rate_limiter = RateLimiter(requests_per_minute=60)

@app.post("/query")
async def process_query(
    request: dict,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user)
):
    REQUEST_COUNT.inc()
    start_time = time.time()
    
    if not await rate_limiter.check(current_user.username):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    try:
        # Check cache first
        cache_key = f"{current_user.username}:{request['prompt']}"
        cached_response = await get_cached_response(cache_key)
        if cached_response:
            return {"response": cached_response, "source": "cache"}

        # RAG processing with improved retrieval
        context = []
        if request.get('use_rag'):
            rag_start = time.time()
            results = await process_rag_query(request)
            context = results.get('documents', [])
            RAG_RETRIEVAL_TIME.observe(time.time() - rag_start)

        # Model processing with fallback
        response = await process_with_fallback(request, context)
        
        # Cache the response
        background_tasks.add_task(
            cache_response,
            cache_key,
            response,
            models[request['model_name']]['cache_config']['ttl']
        )

        RESPONSE_TIME.observe(time.time() - start_time)
        return {"response": response, "context": context}

    except Exception as e:
        MODEL_ERRORS.inc()
        logger.error(f"Query processing error: {e}")
        sentry_sdk.capture_exception(e)
        raise HTTPException(status_code=500, detail=str(e))

async def process_with_fallback(request: dict, context: List[str]) -> str:
    model_config = models[request['model_name']]
    
    for model_name in [request['model_name']] + model_config['fallback_models']:
        try:
            return await process_with_model(model_name, request, context)
        except Exception as e:
            logger.warning(f"Model {model_name} failed: {e}")
            continue
    
    raise HTTPException(status_code=503, detail="All models failed")

async def process_with_model(model_name: str, request: dict, context: List[str]) -> str:
    async with asyncio.Lock():
        if model_name not in model_pool:
            model_config = models[model_name]
            model_pool[model_name] = {
                'model': AutoModelForCausalLM.from_pretrained(
                    model_config['name'],
                    torch_dtype=torch.float16,
                    device_map="auto"
                ),
                'tokenizer': AutoTokenizer.from_pretrained(model_config['name'])
            }

    model_instance = model_pool[model_name]
    
    with ThreadPoolExecutor() as executor:
        future = executor.submit(
            functools.partial(
                generate_response,
                model_instance,
                request['prompt'],
                context
            )
        )
        
        try:
            return await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: future.result(timeout=settings.MODEL_TIMEOUT)
            )
        except TimeoutError:
            raise HTTPException(status_code=504, detail="Model timeout")

def generate_response(model_instance: dict, prompt: str, context: List[str]) -> str:
    full_prompt = f"Context: {' '.join(context)}\n\n{prompt}" if context else prompt
    inputs = model_instance['tokenizer'](
        full_prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    ).to("cuda")
    
    outputs = model_instance['model'].generate(
        **inputs,
        max_length=512,
        temperature=0.7,
        do_sample=True,
        top_p=0.9
    )
    
    return model_instance['tokenizer'].decode(outputs[0], skip_special_tokens=True)

async def process_rag_query(request: dict) -> dict:
    try:
        results = collection.query(
            query_texts=[request.get('context_query', request['prompt'])],
            n_results=5,
            include_metadata=True
        )
        
        # Implement more sophisticated ranking
        ranked_results = rank_results(results)
        return ranked_results
    except Exception as e:
        logger.error(f"RAG processing error: {e}")
        sentry_sdk.capture_exception(e)
        return {"documents": []}

def rank_results(results: dict) -> dict:
    # Implement more sophisticated ranking logic
    # This is a placeholder for the actual implementation
    return results

@app.get("/metrics")
async def metrics():
    return prometheus_client.generate_latest()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)