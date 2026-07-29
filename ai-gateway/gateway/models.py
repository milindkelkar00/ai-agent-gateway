"""
Data models for the API.
Pydantic models = TypeScript interfaces, but with runtime validation.
"""
from pydantic import BaseModel
from typing import Literal, Optional

class Message(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str

class ChatRequest(BaseModel):
    messages: list[Message]
    provider: Optional[Literal["openai", "anthropic", "bedrock"]] = None
    model: Optional[str] = None
    max_tokens: int = 1000
    stream: bool = False

class ChatResponse(BaseModel):
    content: str
    provider_used: str
    model_used: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
