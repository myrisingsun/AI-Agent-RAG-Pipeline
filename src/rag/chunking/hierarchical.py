"""Hierarchical chunking strategy — splits legal/normative documents by articles."""

import re

from src.common.logging import get_logger
from src.rag.chunking.base import ChunkingStrategy
from src.rag.config import RAGConfig
from src.schemas.chunk import Chunk, ChunkMetadata
from src.schemas.document import ParsedDocument

logger = get_logger(__name__)

# Matches headers like: "Статья 5", "Пункт 3.1", "Раздел IV", "п. 2.3.1", "Глава 2"
_ARTICLE_PATTERN = re.compile(
    r"^(Статья|Пункт|Раздел|Глава|п\.|ст\.|Art\.)\s+[\dIVXivx][\d\.\-]*",
    re.MULTILINE | re.IGNORECASE,
)


class HierarchicalChunkingStrategy(ChunkingStrategy):
    """
    Article-aware chunking for normative documents (laws, CBR regulations).

    Splits document at article/section headers detected by regex.
    Each article becomes one chunk. `metadata.law_article` is populated
    with the matched header text.
    """

    def __init__(self, config: RAGConfig) -> None:
        self._max_chars = config.chunking_chunk_size * 4

    def chunk_document(self, doc: ParsedDocument) -> list[Chunk]:
        if not doc.content.strip():
            return []

        sections = self._split_by_articles(doc.content)

        # Fallback to paragraph-based split if no article markers found
        if len(sections) == 0:
            paragraphs = [p.strip() for p in re.split(r"\n{2,}", doc.content) if p.strip()]
            sections = [(None, p) for p in paragraphs] if paragraphs else [(None, doc.content)]

        chunks: list[Chunk] = []
        for idx, (article_label, text) in enumerate(sections):
            text = text.strip()
            if not text:
                continue
            metadata = ChunkMetadata(
                document_id=doc.id,
                doc_type=doc.doc_type.value,
                chunk_index=idx,
                total_chunks=len(sections),
                law_article=article_label,
                section=str(idx + 1),
            )
            chunks.append(Chunk(text=text, token_count=len(text.split()), metadata=metadata))

        # Fix total_chunks after filtering empty sections
        for chunk in chunks:
            chunk.metadata.total_chunks = len(chunks)

        logger.debug(
            "hierarchical chunking done",
            filename=doc.filename,
            sections=len(sections),
            chunks=len(chunks),
        )
        return chunks

    def _split_by_articles(self, content: str) -> list[tuple[str | None, str]]:
        """
        Returns list of (article_label, text) pairs.
        Text before first article has label=None.
        """
        matches = list(_ARTICLE_PATTERN.finditer(content))
        if not matches:
            return []

        sections: list[tuple[str | None, str]] = []

        # Preamble before first article
        preamble = content[: matches[0].start()].strip()
        if preamble:
            sections.append((None, preamble))

        for i, match in enumerate(matches):
            label = match.group(0).strip()
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
            text = content[start:end].strip()
            sections.append((label, text))

        return sections
