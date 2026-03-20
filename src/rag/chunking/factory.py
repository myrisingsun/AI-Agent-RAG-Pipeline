"""Chunking strategy factory — maps doc_type to the right strategy."""

from src.rag.chunking.base import ChunkingStrategy
from src.rag.chunking.fixed_size import FixedSizeChunkingStrategy
from src.rag.chunking.hierarchical import HierarchicalChunkingStrategy
from src.rag.chunking.semantic import SemanticChunkingStrategy
from src.rag.config import RAGConfig
from src.schemas.document import DocType


class ChunkingStrategyFactory:
    """
    Pre-initialises one instance per strategy type, then dispatches by doc_type.

    Mapping (per CLAUDE.md):
      normative         → hierarchical (article-aware)
      contract          → semantic (paragraph-aware)
      financial_report  → fixed_size (table_aware TODO)
      template          → fixed_size
      unknown           → fixed_size
    """

    def __init__(self, config: RAGConfig) -> None:
        _fixed = FixedSizeChunkingStrategy(config)
        self._strategies: dict[DocType, ChunkingStrategy] = {
            DocType.NORMATIVE: HierarchicalChunkingStrategy(config),
            DocType.CONTRACT: SemanticChunkingStrategy(config),
            DocType.FINANCIAL_REPORT: _fixed,  # TODO: replace with TableAwareChunkingStrategy
            DocType.TEMPLATE: _fixed,
            DocType.UNKNOWN: _fixed,
        }

    def get(self, doc_type: DocType) -> ChunkingStrategy:
        return self._strategies.get(doc_type, self._strategies[DocType.UNKNOWN])
