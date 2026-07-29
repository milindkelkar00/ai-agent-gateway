"""
The router decides WHICH provider to use and handles
retry + fallback when a provider fails.
"""
import logging
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from gateway.providers import LLMProvider, BedrockProvider, BedrockFallbackProvider, TitanProvider
from gateway.models import ChatRequest, ChatResponse
from config.settings import settings
import logging

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

PROVIDERS: dict[str, LLMProvider] = {
    "bedrock": BedrockProvider(),
    "bedrock2": BedrockFallbackProvider()
}


class GatewayRouter:
    async def route(self, request: ChatRequest) -> ChatResponse:
        primary_name = request.provider or settings.default_provider
        fallback_name = settings.fallback_provider

        primary = PROVIDERS.get(primary_name)
        fallback = PROVIDERS.get(fallback_name)

        if not primary:
            raise ValueError(f"Unknown provider: {primary_name}")

        try:
            return await self._call_with_retry(primary, request)
        except Exception as e:
            logger.error(f"Primary provider {primary_name} failed with: {type(e).__name__}: {e}")

        if fallback and fallback.name != primary_name:
            try:
                return await self._call_with_retry(fallback, request)
            except Exception as e:
                logger.error(f"Fallback provider {fallback_name} failed with: {type(e).__name__}: {e}")

        raise RuntimeError("All providers failed. Check terminal logs above for the real error.")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type((ConnectionError, TimeoutError)),
    )
    async def _call_with_retry(
        self, provider: LLMProvider, request: ChatRequest
    ) -> ChatResponse:
        return await provider.chat(request)


router = GatewayRouter()
