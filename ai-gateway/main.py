"""
Entrypoint. Run with:
  uvicorn main:app --reload

This is the equivalent of your Express/Node index.ts.
FastAPI auto-generates interactive API docs at http://localhost:8000/docs
"""
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from gateway.models import ChatRequest, ChatResponse
from gateway.router import router
from gateway.rate_limiter import rate_limiter

from config.settings import settings
print(">>> DEFAULT_PROVIDER:", settings.default_provider)
print(">>> FALLBACK_PROVIDER:", settings.fallback_provider)
print(">>> REGION:", settings.aws_region)

app = FastAPI(
    title="AI Gateway",
    description="Multi-provider LLM gateway with retry, fallback, and rate limiting",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    x_api_key: str = Header(default="dev-key"),  # simple auth header
):
    # 1. Rate limit check
    rate_limiter.check(x_api_key)

    # 2. Route to provider
    return await router.route(request)


@app.get("/providers")
async def list_providers():
    """Shows available providers and their default models."""
    return {
        "default": "openai",
        "fallback": "anthropic",
        "available": ["openai", "anthropic", "bedrock"],
    }
