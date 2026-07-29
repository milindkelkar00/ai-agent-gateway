"""
Central config — reads from .env file.
Pydantic BaseSettings validates types automatically,
just like Zod does in TypeScript.
"""
from pydantic_settings import BaseSettings
from typing import Literal

class Settings(BaseSettings):
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    aws_region: str = "us-east-1"

    default_provider: Literal["openai", "anthropic", "bedrock", "bedrock2"] = "bedrock"
    fallback_provider: Literal["openai", "anthropic", "bedrock", "bedrock2"] = "bedrock2"
    max_requests_per_minute: int = 60

    redis_url: str = "redis://localhost:6379"

    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()
