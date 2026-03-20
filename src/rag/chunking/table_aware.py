"""Table-aware chunking — keeps markdown tables intact, splits prose normally."""

import re

from src.common.logging import get_logger
from src.rag.chunking.base import ChunkingStrategy
from src.rag.config import RAGConfig
from src.schemas.chunk import Chunk, ChunkMetadata
from src.schemas.document import ParsedDocument

logger = get_logger(__name__)

# A block is "table-like" if at least half its non-empty lines contain `|`
_TABLE_LINE_RE = re.compile(r"\|")
_MIN_TABLE_RATIO = 0.5

# Form-feed separates pages in PDF-parsed content (see PdfParser)
_PAGE_SEP = "\x0c"


def _is_table_block(text: str) -> bool:
    lines = [l for l in text.splitlines() if l.strip()]
    if not lines:
        return False
    table_lines = sum(1 for l in lines if _TABLE_LINE_RE.search(l))
    return table_lines / len(lines) >= _MIN_TABLE_RATIO


def _page_at_offset(content: str, offset: int) -> int:
    """Return 1-based page number for a character offset using \\x0c markers."""
    return content[:offset].count(_PAGE_SEP) + 1


class TableAwareChunkingStrategy(ChunkingStrategy):
    """
    Chunking strategy for financial reports.

    - Markdown table blocks (lines with `|`) are kept as single chunks.
    - Prose paragraphs are merged until they approach max_chars, then split.
    - Page numbers are tracked via \\x0c (form-feed) separators from PdfParser.
    """

    def __init__(self, config: RAGConfig) -> None:
        self._max_chars = config.chunking_chunk_size * 4  # ~1600 chars ≈ 400 tokens

    def chunk_document(self, doc: ParsedDocument) -> list[Chunk]:
        if not doc.content.strip():
            return []

        # Split content into raw blocks (by double-newline or form-feed)
        raw_blocks = re.split(r"\x0c|\n{2,}", doc.content)

        # Classify blocks and compute their start offsets (for page tracking)
        typed_blocks: list[tuple[str, str, int]] = []  # (kind, text, start_offset)
        offset = 0
        for block in raw_blocks:
            block_stripped = block.strip()
            if block_stripped:
                kind = "table" if _is_table_block(block_stripped) else "text"
                typed_blocks.append((kind, block_stripped, offset))
            offset += len(block) + 2  # approximate offset including separator

        # Build chunks: table blocks → one chunk each; prose blocks → merged
        chunks: list[Chunk] = []
        prose_buf: list[str] = []
        prose_offset: int = 0

        def _flush_prose() -> None:
            nonlocal prose_buf, prose_offset
            if not prose_buf:
                return
            text = "\n\n".join(prose_buf)
            page = _page_at_offset(doc.content, prose_offset)
            metadata = ChunkMetadata(
                document_id=doc.id,
                doc_type=doc.doc_type.value,
                chunk_index=len(chunks),
                total_chunks=0,  # fixed up below
                page=page,
                section=str(len(chunks) + 1),
            )
            chunks.append(Chunk(text=text, token_count=len(text.split()), metadata=metadata))
            prose_buf = []
            prose_offset = 0

        for kind, text, start in typed_blocks:
            if kind == "table":
                _flush_prose()
                page = _page_at_offset(doc.content, start)
                metadata = ChunkMetadata(
                    document_id=doc.id,
                    doc_type=doc.doc_type.value,
                    chunk_index=len(chunks),
                    total_chunks=0,
                    page=page,
                    section=str(len(chunks) + 1),
                )
                chunks.append(
                    Chunk(text=text, token_count=len(text.split()), metadata=metadata)
                )
            else:
                # Accumulate prose; flush when buffer would exceed max_chars
                combined = "\n\n".join(prose_buf + [text]) if prose_buf else text
                if len(combined) > self._max_chars and prose_buf:
                    _flush_prose()
                if not prose_buf:
                    prose_offset = start
                prose_buf.append(text)

        _flush_prose()

        # Fix total_chunks
        for chunk in chunks:
            chunk.metadata.total_chunks = len(chunks)

        logger.debug(
            "table-aware chunking done",
            filename=doc.filename,
            chunks=len(chunks),
        )
        return chunks
