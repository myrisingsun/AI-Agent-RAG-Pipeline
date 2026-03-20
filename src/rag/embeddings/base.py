from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    """Abstract base for all embedding providers."""

    async def initialize(self) -> None:
        """Load model weights. Called once at startup. Override if needed."""

    @abstractmethod
    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of passage texts for indexing."""
        ...

    @abstractmethod
    async def embed_query(self, text: str) -> list[float]:
        """Embed a single query. May use a different prefix than embed_texts."""
        ...

    @property
    @abstractmethod
    def vector_size(self) -> int:
        """Dimensionality of produced vectors."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Model identifier string."""
        ...
