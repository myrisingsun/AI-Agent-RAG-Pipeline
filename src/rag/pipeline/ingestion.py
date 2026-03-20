import uuid
from datetime import UTC, datetime

from src.common.exceptions import ChunkingError
from src.common.logging import get_logger
from src.common.storage import MinIOStorage
from src.rag.chunking.factory import ChunkingStrategyFactory
from src.rag.config import CollectionName, RAGConfig
from src.rag.embeddings.base import EmbeddingProvider
from src.rag.parsing.factory import get_parser
from src.rag.vectorstore.operations import VectorStoreOperations
from src.rag.vectorstore.sparse import compute_sparse_vector
from src.schemas.api import DocumentUploadResponse
from src.schemas.document import DocType

logger = get_logger(__name__)

# doc_type → target Qdrant collection
_COLLECTION_MAP: dict[DocType, CollectionName] = {
    DocType.CONTRACT: CollectionName.CURRENT_PACKAGE,
    DocType.FINANCIAL_REPORT: CollectionName.CURRENT_PACKAGE,
    DocType.NORMATIVE: CollectionName.NORMATIVE_BASE,
    DocType.TEMPLATE: CollectionName.REFERENCE_TEMPLATES,
    DocType.UNKNOWN: CollectionName.CURRENT_PACKAGE,
}


class IngestionService:
    """Orchestrates the full ingestion pipeline: parse → store → chunk → embed → upsert."""

    def __init__(
        self,
        config: RAGConfig,
        embedding_provider: EmbeddingProvider,
        vs_operations: VectorStoreOperations,
        storage: MinIOStorage,
    ) -> None:
        self._config = config
        self._embedding = embedding_provider
        self._vs = vs_operations
        self._storage = storage
        self._chunking_factory = ChunkingStrategyFactory(config)

    async def ingest(
        self,
        file_bytes: bytes,
        filename: str,
        doc_type: DocType,
        session_id: str | None = None,
    ) -> DocumentUploadResponse:
        """
        Full ingestion pipeline:
        1. Parse file (TXT / PDF / DOCX)
        2. Upload original to MinIO
        3. Chunk (strategy selected by doc_type)
        4. Compute sparse vectors (BM25 TF for hybrid search)
        5. Embed (dense)
        6. Upsert to Qdrant
        """
        doc_id = uuid.uuid4()

        # 1. Parse
        parser = get_parser(filename)
        parsed = parser.parse(file_bytes, filename, doc_type)
        parsed.id = doc_id

        # 2. Upload original to MinIO (non-fatal)
        source_path: str | None = None
        try:
            source_path = await self._storage.upload(str(doc_id), filename, file_bytes)
        except Exception as exc:
            logger.warning("MinIO upload failed, continuing without source_path", error=str(exc))

        # 3. Chunk using strategy appropriate for doc_type
        chunker = self._chunking_factory.get(doc_type)
        chunks = chunker.chunk_document(parsed)
        if not chunks:
            raise ChunkingError(f"No chunks produced from '{filename}'")

        # Attach session_id and source_path to each chunk
        for chunk in chunks:
            if session_id:
                chunk.metadata.session_id = session_id
            if source_path:
                chunk.metadata.source_path = source_path

        # 4. Compute sparse vectors for hybrid search (uses embedding tokenizer)
        if self._config.hybrid_search_enabled:
            tokenizer = getattr(self._embedding, "_tokenizer", None)
            if tokenizer is not None:
                for chunk in chunks:
                    chunk.sparse_vector = compute_sparse_vector(chunk.text, tokenizer)

        # 5. Embed dense vectors
        texts = [c.text for c in chunks]
        embeddings = await self._embedding.embed_texts(texts)
        for chunk, emb in zip(chunks, embeddings, strict=True):
            chunk.embedding = emb

        # 6. Upsert to Qdrant
        collection = _COLLECTION_MAP.get(doc_type, CollectionName.CURRENT_PACKAGE)
        count = await self._vs.upsert_chunks(collection, chunks)

        logger.info(
            "document ingested",
            doc_id=str(doc_id),
            filename=filename,
            doc_type=doc_type,
            collection=collection,
            chunks=count,
            pages=parsed.pages,
            hybrid=self._config.hybrid_search_enabled,
        )

        return DocumentUploadResponse(
            id=doc_id,
            filename=filename,
            doc_type=doc_type,
            collection=collection,
            chunk_count=count,
            created_at=datetime.now(UTC),
        )
