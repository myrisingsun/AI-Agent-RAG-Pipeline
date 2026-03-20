import asyncio
from typing import Any

from src.common.exceptions import RAGBaseError
from src.common.logging import get_logger
from src.rag.config import RAGConfig
from src.rag.reranker.base import Reranker

logger = get_logger(__name__)


class RerankerError(RAGBaseError):
    pass


class CrossEncoderReranker(Reranker):
    """
    Cross-encoder reranker using mmarco-mMiniLMv2-L12-H384.
    Scores (query, passage) pairs and reranks top-N candidates.
    Much more accurate than cosine similarity for final ranking.
    """

    def __init__(self, config: RAGConfig) -> None:
        self._config = config
        self._model: Any = None

    async def initialize(self) -> None:
        """Load cross-encoder weights. Runs in executor to avoid blocking event loop."""
        from sentence_transformers import CrossEncoder

        loop = asyncio.get_event_loop()
        try:
            self._model = await loop.run_in_executor(
                None,
                lambda: CrossEncoder(
                    self._config.reranker_model_name,
                    max_length=512,
                    device=self._config.embedding_device,
                ),
            )
        except Exception as exc:
            raise RerankerError(f"Failed to load reranker '{self._config.reranker_model_name}': {exc}") from exc

        logger.info(
            "reranker loaded",
            model=self._config.reranker_model_name,
            device=self._config.embedding_device,
        )

    async def rerank(
        self,
        query: str,
        hits: list[dict],  # type: ignore[type-arg]
        top_k: int,
    ) -> list[dict]:  # type: ignore[type-arg]
        """Score all hits with the cross-encoder and return the top_k by score."""
        if not hits:
            return []
        if self._model is None:
            raise RerankerError("CrossEncoderReranker not initialized. Call await initialize() first.")

        texts = [str(hit.get("payload", {}).get("text", "")) for hit in hits]
        pairs = [(query, t) for t in texts]

        loop = asyncio.get_event_loop()
        try:
            scores: list[float] = await loop.run_in_executor(
                None,
                lambda: self._model.predict(pairs).tolist(),
            )
        except Exception as exc:
            raise RerankerError(f"Reranking failed: {exc}") from exc

        ranked = sorted(
            zip(hits, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        result = []
        for hit, score in ranked[:top_k]:
            reranked_hit = dict(hit)
            reranked_hit["score"] = round(float(score), 4)
            result.append(reranked_hit)

        logger.debug(
            "reranking done",
            candidates=len(hits),
            top_k=top_k,
            top_score=result[0]["score"] if result else 0,
        )
        return result
