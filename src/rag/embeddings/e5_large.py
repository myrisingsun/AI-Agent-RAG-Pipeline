import asyncio
from typing import Any

from src.common.exceptions import EmbeddingError
from src.common.logging import get_logger
from src.rag.config import RAGConfig
from src.rag.embeddings.base import EmbeddingProvider

logger = get_logger(__name__)

_VECTOR_SIZE = 1024


class E5LargeEmbeddingProvider(EmbeddingProvider):
    """
    Embedding provider using intfloat/multilingual-e5-large.

    Requires query/passage prefixes:
      - indexing: "passage: {text}"
      - retrieval: "query: {text}"
    Vectors are L2-normalized (unit length) — cosine similarity = dot product.
    """

    def __init__(self, config: RAGConfig) -> None:
        self._config = config
        self._model: Any = None

    async def initialize(self) -> None:
        """Load model weights. Must be called once before any embed_* call."""
        from sentence_transformers import SentenceTransformer

        loop = asyncio.get_event_loop()
        try:
            self._model = await loop.run_in_executor(
                None,
                lambda: SentenceTransformer(
                    self._config.embedding_model_name,
                    device=self._config.embedding_device,
                    cache_folder=self._config.hf_home,
                ),
            )
        except Exception as exc:
            raise EmbeddingError(f"Failed to load e5-large: {exc}") from exc

        logger.info(
            "e5-large loaded",
            model=self._config.embedding_model_name,
            device=self._config.embedding_device,
        )

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed passage texts for indexing (prepends 'passage: ' prefix)."""
        prefixed = [f"passage: {t}" for t in texts]
        return await self._encode_batch(prefixed)

    async def embed_query(self, text: str) -> list[float]:
        """Embed a single query for retrieval (prepends 'query: ' prefix)."""
        vectors = await self._encode_batch([f"query: {text}"])
        return vectors[0]

    async def _encode_batch(self, texts: list[str]) -> list[list[float]]:
        self._ensure_initialized()
        loop = asyncio.get_event_loop()
        try:
            result = await loop.run_in_executor(
                None,
                lambda: self._model.encode(
                    texts,
                    batch_size=self._config.embedding_batch_size,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                ),
            )
        except Exception as exc:
            raise EmbeddingError(f"Encoding failed: {exc}") from exc
        return result.tolist()

    def _ensure_initialized(self) -> None:
        if self._model is None:
            raise EmbeddingError(
                "E5LargeEmbeddingProvider is not initialized. Call await initialize() first."
            )

    @property
    def vector_size(self) -> int:
        return _VECTOR_SIZE

    @property
    def model_name(self) -> str:
        return self._config.embedding_model_name
