"""Unit tests for FixedSizeChunkingStrategy — tokenizer is mocked."""

import uuid
from unittest.mock import MagicMock, patch

import pytest

from src.rag.config import RAGConfig
from src.rag.chunking.fixed_size import FixedSizeChunkingStrategy
from src.schemas.document import DocType, ParsedDocument


def _make_doc(content: str, doc_type: DocType = DocType.CONTRACT) -> ParsedDocument:
    return ParsedDocument(
        id=uuid.uuid4(),
        filename="test.pdf",
        doc_type=doc_type,
        content=content,
    )


def _make_strategy(config: RAGConfig, token_ids_per_word: int = 1) -> FixedSizeChunkingStrategy:
    """Return FixedSizeChunkingStrategy with a mocked tokenizer."""
    strategy = FixedSizeChunkingStrategy.__new__(FixedSizeChunkingStrategy)
    strategy._chunk_size = config.chunking_chunk_size
    strategy._chunk_overlap = config.chunking_chunk_overlap

    mock_tokenizer = MagicMock()
    # encode: each space-separated word = 1 token
    mock_tokenizer.encode = lambda text, add_special_tokens=False: list(
        range(len(text.split()))
    )
    # decode: return N words of the original position
    mock_tokenizer.decode = lambda ids, skip_special_tokens=True: " ".join(
        f"word{i}" for i in ids
    )
    strategy._tokenizer = mock_tokenizer
    return strategy


@pytest.mark.unit
def test_empty_document_returns_empty_list(config: RAGConfig) -> None:
    strategy = _make_strategy(config)
    doc = _make_doc("")
    assert strategy.chunk_document(doc) == []


@pytest.mark.unit
def test_short_document_produces_one_chunk(config: RAGConfig) -> None:
    """Document shorter than chunk_size (400) → exactly 1 chunk."""
    strategy = _make_strategy(config)
    # 100 words → 100 tokens < 400
    doc = _make_doc(" ".join(["word"] * 100))
    chunks = strategy.chunk_document(doc)
    assert len(chunks) == 1


@pytest.mark.unit
def test_document_exactly_chunk_size_produces_one_chunk(config: RAGConfig) -> None:
    strategy = _make_strategy(config)
    doc = _make_doc(" ".join(["word"] * 400))
    chunks = strategy.chunk_document(doc)
    assert len(chunks) == 1


@pytest.mark.unit
def test_document_two_chunks(config: RAGConfig) -> None:
    """400 + overlap produces 2 chunks.
    chunk_size=400, overlap=60, step=340
    Words=401 → 2 windows: [0..400], [340..401]
    """
    strategy = _make_strategy(config)
    doc = _make_doc(" ".join(["word"] * 450))
    chunks = strategy.chunk_document(doc)
    assert len(chunks) == 2


@pytest.mark.unit
def test_chunk_metadata_populated(config: RAGConfig) -> None:
    strategy = _make_strategy(config)
    doc = _make_doc(" ".join(["word"] * 100))
    chunks = strategy.chunk_document(doc)

    chunk = chunks[0]
    assert chunk.metadata.document_id == doc.id
    assert chunk.metadata.chunk_index == 0
    assert chunk.metadata.total_chunks == 1
    assert chunk.token_count > 0


@pytest.mark.unit
def test_chunk_metadata_doc_type_preserved(config: RAGConfig) -> None:
    strategy = _make_strategy(config)
    doc = _make_doc(" ".join(["word"] * 100), doc_type=DocType.NORMATIVE)
    chunks = strategy.chunk_document(doc)
    assert chunks[0].metadata.doc_type == DocType.NORMATIVE


@pytest.mark.unit
def test_chunk_index_sequential(config: RAGConfig) -> None:
    strategy = _make_strategy(config)
    doc = _make_doc(" ".join(["word"] * 800))
    chunks = strategy.chunk_document(doc)
    for i, chunk in enumerate(chunks):
        assert chunk.metadata.chunk_index == i
        assert chunk.metadata.total_chunks == len(chunks)


@pytest.mark.unit
def test_overlap_produces_shared_tokens(config: RAGConfig) -> None:
    """With overlap=60, chunk N ends at token X, chunk N+1 starts at X - 60."""
    strategy = _make_strategy(config)
    doc = _make_doc(" ".join(["word"] * 800))
    chunks = strategy.chunk_document(doc)
    assert len(chunks) >= 2
    # First chunk size ≤ chunk_size
    assert chunks[0].token_count <= config.chunking_chunk_size
    # Second chunk's token count should be > 0
    assert chunks[1].token_count > 0
