"""
Provider abstraction layer.

KEY CONCEPT: We define a common interface so the router doesn't
care which provider it's calling — just like a TypeScript interface
where multiple classes implement the same contract.
"""
import anthropic
import openai
from abc import ABC, abstractmethod
from gateway.models import ChatRequest, ChatResponse
from config.settings import settings

COSTS = {
    "gpt-4o":                   {"input": 0.0025,  "output": 0.010},
    "gpt-4o-mini":              {"input": 0.00015, "output": 0.0006},
    "claude-3-5-haiku-latest":  {"input": 0.0008,  "output": 0.004},
    "claude-3-5-sonnet-latest": {"input": 0.003,   "output": 0.015},
}

def calc_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    rate = COSTS.get(model, {"input": 0.001, "output": 0.002})
    return (input_tokens * rate["input"] + output_tokens * rate["output"]) / 1000


class LLMProvider(ABC):
    @abstractmethod
    async def chat(self, request: ChatRequest) -> ChatResponse:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass


class OpenAIProvider(LLMProvider):
    name = "openai"
    default_model = "gpt-4o-mini"

    def __init__(self):
        self.client = openai.AsyncOpenAI(api_key=settings.openai_api_key)

    async def chat(self, request: ChatRequest) -> ChatResponse:
        model = request.model or self.default_model
        messages = [m.model_dump() for m in request.messages]

        response = await self.client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=request.max_tokens,
        )

        content = response.choices[0].message.content
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens

        return ChatResponse(
            content=content,
            provider_used=self.name,
            model_used=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=calc_cost(model, input_tokens, output_tokens),
        )


class AnthropicProvider(LLMProvider):
    name = "anthropic"
    default_model = "claude-3-5-haiku-latest"

    def __init__(self):
        self.client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)

    async def chat(self, request: ChatRequest) -> ChatResponse:
        model = request.model or self.default_model

        system = next(
            (m.content for m in request.messages if m.role == "system"),
            "You are a helpful assistant."
        )
        messages = [
            {"role": m.role, "content": m.content}
            for m in request.messages if m.role != "system"
        ]

        response = await self.client.messages.create(
            model=model,
            system=system,
            messages=messages,
            max_tokens=request.max_tokens,
        )

        content = response.content[0].text
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens

        return ChatResponse(
            content=content,
            provider_used=self.name,
            model_used=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=calc_cost(model, input_tokens, output_tokens),
        )

class BedrockProvider(LLMProvider):
    name = "bedrock"
    default_model = "anthropic.claude-3-haiku-20240307-v1:0"

    def __init__(self):
        self.client = boto3.client(
            service_name="bedrock-runtime",
            region_name=settings.aws_region,
            aws_access_key_id=settings.aws_access_key_id,
            aws_secret_access_key=settings.aws_secret_access_key,
        )

    async def chat(self, request: ChatRequest) -> ChatResponse:
        import asyncio
        model = request.model or self.default_model

        messages = [
            {"role": m.role, "content": m.content}
            for m in request.messages if m.role != "system"
        ]
        system = next(
            (m.content for m in request.messages if m.role == "system"),
            "You are a helpful assistant."
        )

        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": request.max_tokens,
            "system": system,
            "messages": messages,
        })

        # boto3 is sync, so we run it in a thread to not block async
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self.client.invoke_model(modelId=model, body=body)
        )

        result = json.loads(response["body"].read())
        content = result["content"][0]["text"]
        input_tokens = result["usage"]["input_tokens"]
        output_tokens = result["usage"]["output_tokens"]

        return ChatResponse(
            content=content,
            provider_used=self.name,
            model_used=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=calc_cost(model, input_tokens, output_tokens),
        )

class TitanProvider(LLMProvider):
    name = "titan"
    default_model = "amazon.titan-text-lite-v1"

    def __init__(self):
        self.client = boto3.client(
            service_name="bedrock-runtime",
            region_name=settings.aws_region,
            aws_access_key_id=settings.aws_access_key_id,
            aws_secret_access_key=settings.aws_secret_access_key,
        )

    async def chat(self, request: ChatRequest) -> ChatResponse:
        import asyncio
        import json
        model = request.model or self.default_model

        # Titan has a completely different request format than Claude
        prompt = "\n".join(
            f"{m.role.upper()}: {m.content}"
            for m in request.messages
        ) + "\nASSISTANT:"

        body = json.dumps({
            "inputText": prompt,
            "textGenerationConfig": {
                "maxTokenCount": request.max_tokens,
                "temperature": 0.7,
                "topP": 0.9,
            }
        })

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self.client.invoke_model(modelId=model, body=body)
        )

        result = json.loads(response["body"].read())
        content = result["results"][0]["outputText"].strip()
        input_tokens = result.get("inputTextTokenCount", 0)
        output_tokens = result["results"][0].get("tokenCount", 0)

        return ChatResponse(
            content=content,
            provider_used=self.name,
            model_used=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=calc_cost(model, input_tokens, output_tokens),
        )
        
class BedrockFallbackProvider(LLMProvider):
    name = "bedrock2"
    default_model = "anthropic.claude-3-5-haiku-20241022-v1:0"

    def __init__(self):
        self.client = boto3.client(
            service_name="bedrock-runtime",
            region_name=settings.aws_region,
            aws_access_key_id=settings.aws_access_key_id,
            aws_secret_access_key=settings.aws_secret_access_key,
        )

    async def chat(self, request: ChatRequest) -> ChatResponse:
        import asyncio
        model = request.model or self.default_model

        system = next(
            (m.content for m in request.messages if m.role == "system"),
            "You are a helpful assistant."
        )
        messages = [
            {"role": m.role, "content": m.content}
            for m in request.messages if m.role != "system"
        ]

        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": request.max_tokens,
            "system": system,
            "messages": messages,
        })

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self.client.invoke_model(modelId=model, body=body)
        )

        result = json.loads(response["body"].read())
        content = result["content"][0]["text"]
        input_tokens = result["usage"]["input_tokens"]
        output_tokens = result["usage"]["output_tokens"]

        return ChatResponse(
            content=content,
            provider_used=self.name,
            model_used=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=calc_cost(model, input_tokens, output_tokens),
        )
        
import boto3
import json