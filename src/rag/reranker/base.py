from abc import ABC, abstractmethod


class Reranker(ABC):
    """Abstract base for cross-encoder rerankers."""

    async def initialize(self) -> None:
        """Load model weights. Called once at startup."""

    @abstractmethod
    async def rerank(
        self,
        query: str,
        hits: list[dict],  # type: ignore[type-arg]
        top_k: int,
    ) -> list[dict]:  # type: ignore[type-arg]
        """
        Rerank search hits by (query, chunk_text) relevance score.
        Returns top_k hits sorted by descending reranker score.
        Each hit dict gets its 'score' field overwritten with the reranker score.
        """
        ...
