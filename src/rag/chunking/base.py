from abc import ABC, abstractmethod

from src.schemas.chunk import Chunk
from src.schemas.document import ParsedDocument


class ChunkingStrategy(ABC):
    """Abstract base for all chunking strategies."""

    @abstractmethod
    def chunk_document(self, doc: ParsedDocument) -> list[Chunk]:
        """
        Split a parsed document into chunks.
        Every returned Chunk must have a fully populated ChunkMetadata.
        """
        ...
