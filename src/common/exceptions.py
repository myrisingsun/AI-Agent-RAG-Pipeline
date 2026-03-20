class RAGBaseError(Exception):
    """Base exception for all RAG pipeline errors."""


class CollectionNotFoundError(RAGBaseError):
    """Qdrant collection does not exist."""


class EmbeddingError(RAGBaseError):
    """Embedding model failure or not initialized."""


class ChunkingError(RAGBaseError):
    """Document chunking failure."""


class VectorStoreError(RAGBaseError):
    """Qdrant connection or operation failure."""


class DocumentNotFoundError(RAGBaseError):
    """Document not found in storage."""


class ConfigurationError(RAGBaseError):
    """Invalid or missing configuration."""
