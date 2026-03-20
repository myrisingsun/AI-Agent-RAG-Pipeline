"""Semantic chunking strategy — split on paragraph boundaries."""

import re

from src.common.logging import get_logger
from src.rag.chunking.base import ChunkingStrategy
from src.rag.config import RAGConfig
from src.schemas.chunk import Chunk, ChunkMetadata
from src.schemas.document import ParsedDocument

logger = get_logger(__name__)

_MIN_CHUNK_CHARS = 100


class SemanticChunkingStrategy(ChunkingStrategy):
    """
    Paragraph-aware chunking for contracts and free-form documents.

    Splits on double newlines, then merges short paragraphs with the next
    one until the combined length exceeds `min_chunk_chars`.
    Each chunk gets a `section` index in metadata.
    """

    def __init__(self, config: RAGConfig) -> None:
        self._max_chars = config.chunking_chunk_size * 4  # rough char budget
        self._min_chars = _MIN_CHUNK_CHARS

    def chunk_document(self, doc: ParsedDocument) -> list[Chunk]:
        if not doc.content.strip():
            return []

        paragraphs = [p.strip() for p in re.split(r"\n{2,}", doc.content) if p.strip()]
        merged = self._merge_short(paragraphs)

        chunks: list[Chunk] = []
        for idx, text in enumerate(merged):
            metadata = ChunkMetadata(
                document_id=doc.id,
                doc_type=doc.doc_type.value,
                chunk_index=idx,
                total_chunks=len(merged),
                section=str(idx + 1),
            )
            chunks.append(Chunk(text=text, token_count=len(text.split()), metadata=metadata))

        logger.debug(
            "semantic chunking done",
            filename=doc.filename,
            paragraphs=len(paragraphs),
            chunks=len(chunks),
        )
        return chunks

    def _merge_short(self, paragraphs: list[str]) -> list[str]:
        """Merge consecutive short paragraphs; split oversized ones."""
        result: list[str] = []
        buffer = ""
        for para in paragraphs:
            if not buffer:
                buffer = para
            elif len(buffer) < self._min_chars:
                buffer = buffer + "\n\n" + para
            else:
                if len(buffer) > self._max_chars:
                    result.extend(self._split_long(buffer))
                else:
                    result.append(buffer)
                buffer = para
        if buffer:
            if len(buffer) > self._max_chars:
                result.extend(self._split_long(buffer))
            else:
                result.append(buffer)
        return result

    def _split_long(self, text: str) -> list[str]:
        """Split an oversized paragraph by sentence boundaries."""
        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks: list[str] = []
        buf = ""
        for sent in sentences:
            if len(buf) + len(sent) > self._max_chars and buf:
                chunks.append(buf.strip())
                buf = sent
            else:
                buf = (buf + " " + sent).strip() if buf else sent
        if buf:
            chunks.append(buf.strip())
        return chunks
