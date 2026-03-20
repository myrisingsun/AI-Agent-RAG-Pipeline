import json
from collections.abc import AsyncIterator

import httpx

from src.common.exceptions import RAGBaseError
from src.common.logging import get_logger
from src.rag.config import RAGConfig

logger = get_logger(__name__)


class LLMError(RAGBaseError):
    pass


class LLMClient:
    """OpenAI-compatible client for self-hosted vLLM."""

    def __init__(self, config: RAGConfig) -> None:
        self._api_url = config.vllm_api_url.rstrip("/")
        self._model = config.vllm_model
        self._http = httpx.AsyncClient(timeout=120.0)

    async def complete(self, prompt: str) -> str:
        """Non-streaming completion. Returns full response text."""
        payload = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 2048,
        }
        try:
            response = await self._http.post(
                f"{self._api_url}/chat/completions",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            return str(data["choices"][0]["message"]["content"])
        except httpx.HTTPError as exc:
            raise LLMError(f"LLM request failed: {exc}") from exc

    async def stream(self, prompt: str) -> AsyncIterator[str]:
        """Streaming completion. Yields text tokens as they arrive."""
        payload = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 2048,
            "stream": True,
        }
        try:
            async with self._http.stream(
                "POST",
                f"{self._api_url}/chat/completions",
                json=payload,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    data = json.loads(data_str)
                    delta = data["choices"][0]["delta"].get("content", "")
                    if delta:
                        yield delta
        except httpx.HTTPError as exc:
            raise LLMError(f"LLM stream failed: {exc}") from exc

    async def close(self) -> None:
        await self._http.aclose()
