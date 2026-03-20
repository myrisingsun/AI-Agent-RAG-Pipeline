import uuid
from datetime import UTC, datetime

from src.common.exceptions import ChunkingError
from src.common.logging import get_logger
from src.rag.chunking.fixed_size import FixedSizeChunkingStrategy
from src.rag.config import CollectionName, RAGConfig
from src.rag.embeddings.base import EmbeddingProvider
from src.rag.vectorstore.operations import VectorStoreOperations
from src.schemas.api import DocumentUploadResponse
from src.schemas.document import DocType, ParsedDocument

logger = get_logger(__name__)

# doc_type → target collection
_COLLECTION_MAP: dict[DocType, CollectionName] = {
    DocType.CONTRACT: CollectionName.CURRENT_PACKAGE,
    DocType.FINANCIAL_REPORT: CollectionName.CURRENT_PACKAGE,
    DocType.NORMATIVE: CollectionName.NORMATIVE_BASE,
    DocType.TEMPLATE: CollectionName.REFERENCE_TEMPLATES,
    DocType.UNKNOWN: CollectionName.CURRENT_PACKAGE,
}


def _extract_text(file_bytes: bytes, filename: str) -> str:
    """
    Extract plain text from an uploaded file.
    MVP: .txt only. Production: use Docling for PDF/DOCX.
    """
    lower = filename.lower()
    if lower.endswith(".txt"):
        return file_bytes.decode("utf-8", errors="replace")
    raise ChunkingError(
        f"Unsupported file type: '{filename}'. "
        "MVP supports .txt only. Configure Docling for PDF/DOCX in production."
    )


class IngestionService:
    """Orchestrates the full ingestion pipeline: parse → chunk → embed → upsert."""

    def __init__(
        self,
        config: RAGConfig,
        embedding_provider: EmbeddingProvider,
        vs_operations: VectorStoreOperations,
    ) -> None:
        self._config = config
        self._embedding = embedding_provider
        self._vs = vs_operations
        self._chunker = FixedSizeChunkingStrategy(config)

    async def ingest(
        self,
        file_bytes: bytes,
        filename: str,
        doc_type: DocType,
        session_id: str | None = None,
    ) -> DocumentUploadResponse:
        """
        Full ingestion pipeline.
        Returns a DocumentUploadResponse with chunk count and target collection.
        """
        doc_id = uuid.uuid4()
        text = _extract_text(file_bytes, filename)

        parsed = ParsedDocument(
            id=doc_id,
            filename=filename,
            doc_type=doc_type,
            content=text,
            metadata={"session_id": session_id or ""},
        )

        chunks = self._chunker.chunk_document(parsed)
        if not chunks:
            raise ChunkingError(f"No chunks produced from '{filename}'")

        # Attach session_id to each chunk for current_package filtering
        if session_id:
            for chunk in chunks:
                chunk.metadata.session_id = session_id

        # Embed all chunks in one batch call
        texts = [c.text for c in chunks]
        embeddings = await self._embedding.embed_texts(texts)
        for chunk, emb in zip(chunks, embeddings, strict=True):
            chunk.embedding = emb

        collection = _COLLECTION_MAP.get(doc_type, CollectionName.CURRENT_PACKAGE)
        count = await self._vs.upsert_chunks(collection, chunks)

        logger.info(
            "document ingested",
            doc_id=str(doc_id),
            filename=filename,
            doc_type=doc_type,
            collection=collection,
            chunks=count,
        )

        return DocumentUploadResponse(
            id=doc_id,
            filename=filename,
            doc_type=doc_type,
            collection=collection,
            chunk_count=count,
            created_at=datetime.now(UTC),
        )
