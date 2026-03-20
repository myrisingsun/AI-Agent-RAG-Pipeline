from typing import Any

from src.common.exceptions import ChunkingError
from src.common.logging import get_logger
from src.rag.chunking.base import ChunkingStrategy
from src.rag.config import RAGConfig
from src.schemas.chunk import Chunk, ChunkMetadata
from src.schemas.document import ParsedDocument

logger = get_logger(__name__)


class FixedSizeChunkingStrategy(ChunkingStrategy):
    """
    Token-aware sliding window chunking.
    Uses the same tokenizer as the embedding model to count tokens accurately.

    chunk_size=400, overlap=60 → step=340 tokens per window.
    """

    def __init__(self, config: RAGConfig) -> None:
        self._chunk_size = config.chunking_chunk_size
        self._chunk_overlap = config.chunking_chunk_overlap
        self._tokenizer = self._load_tokenizer(config)

    def _load_tokenizer(self, config: RAGConfig) -> Any:
        try:
            from transformers import AutoTokenizer

            return AutoTokenizer.from_pretrained(
                config.embedding_model_name,
                cache_dir=config.hf_home,
            )
        except Exception as exc:
            raise ChunkingError(f"Failed to load tokenizer: {exc}") from exc

    def chunk_document(self, doc: ParsedDocument) -> list[Chunk]:
        if not doc.content.strip():
            return []

        try:
            token_ids: list[int] = self._tokenizer.encode(
                doc.content, add_special_tokens=False
            )
        except Exception as exc:
            raise ChunkingError(f"Tokenization failed for {doc.filename}: {exc}") from exc

        windows = self._sliding_windows(token_ids)
        chunks: list[Chunk] = []

        for idx, window in enumerate(windows):
            try:
                text: str = self._tokenizer.decode(window, skip_special_tokens=True)
            except Exception as exc:
                raise ChunkingError(f"Decode failed at window {idx}: {exc}") from exc

            metadata = ChunkMetadata(
                document_id=doc.id,
                doc_type=doc.doc_type.value,
                chunk_index=idx,
                total_chunks=len(windows),
            )
            chunks.append(Chunk(text=text, token_count=len(window), metadata=metadata))

        logger.debug(
            "document chunked",
            filename=doc.filename,
            doc_type=doc.doc_type,
            total_tokens=len(token_ids),
            total_chunks=len(chunks),
        )
        return chunks

    def _sliding_windows(self, token_ids: list[int]) -> list[list[int]]:
        step = self._chunk_size - self._chunk_overlap
        windows: list[list[int]] = []
        start = 0
        while start < len(token_ids):
            end = min(start + self._chunk_size, len(token_ids))
            windows.append(token_ids[start:end])
            if end == len(token_ids):
                break
            start += step
        return windows
