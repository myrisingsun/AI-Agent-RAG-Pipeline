"""Unit tests for TableAwareChunkingStrategy."""

import uuid

import pytest

from src.rag.chunking.table_aware import TableAwareChunkingStrategy, _is_table_block
from src.rag.config import RAGConfig
from src.schemas.document import DocType, ParsedDocument


def _make_doc(content: str) -> ParsedDocument:
    return ParsedDocument(
        id=uuid.uuid4(),
        filename="report.pdf",
        doc_type=DocType.FINANCIAL_REPORT,
        content=content,
    )


def _make_strategy() -> TableAwareChunkingStrategy:
    config = RAGConfig.model_construct(
        chunking_chunk_size=400,
        chunking_chunk_overlap=60,
    )
    return TableAwareChunkingStrategy(config)


# ─── _is_table_block ─────────────────────────────────────────────────────────

@pytest.mark.unit
def test_is_table_block_detects_markdown_table() -> None:
    block = "| Статья | Ставка |\n| --- | --- |\n| 334 ГК | 12% |"
    assert _is_table_block(block) is True


@pytest.mark.unit
def test_is_table_block_rejects_plain_text() -> None:
    block = "Банк обязуется предоставить кредит на следующих условиях."
    assert _is_table_block(block) is False


@pytest.mark.unit
def test_is_table_block_mixed_majority_pipes() -> None:
    # 3 lines with pipes out of 4 → table
    block = "| A | B |\n| 1 | 2 |\n| 3 | 4 |\nПримечание: данные условные."
    assert _is_table_block(block) is True


# ─── chunk_document ───────────────────────────────────────────────────────────

@pytest.mark.unit
def test_empty_document_returns_no_chunks() -> None:
    strategy = _make_strategy()
    chunks = strategy.chunk_document(_make_doc(""))
    assert chunks == []


@pytest.mark.unit
def test_table_block_becomes_single_chunk() -> None:
    table = "| Показатель | Значение |\n| --- | --- |\n| Активы | 100 |\n| Пассивы | 90 |"
    strategy = _make_strategy()
    chunks = strategy.chunk_document(_make_doc(table))
    assert len(chunks) == 1
    assert "|" in chunks[0].text


@pytest.mark.unit
def test_text_and_table_produce_separate_chunks() -> None:
    content = (
        "Финансовая отчётность за 2024 год.\n\n"
        "| Статья | Сумма |\n| --- | --- |\n| Выручка | 500 |\n| Расходы | 400 |\n\n"
        "Итого прибыль составила 100 млн руб."
    )
    strategy = _make_strategy()
    chunks = strategy.chunk_document(_make_doc(content))
    assert len(chunks) == 3
    table_chunks = [c for c in chunks if "|" in c.text]
    assert len(table_chunks) == 1


@pytest.mark.unit
def test_chunks_have_correct_total_chunks() -> None:
    content = (
        "Вводный текст.\n\n"
        "| A | B |\n| --- | --- |\n| 1 | 2 |\n\n"
        "Заключение."
    )
    strategy = _make_strategy()
    chunks = strategy.chunk_document(_make_doc(content))
    for chunk in chunks:
        assert chunk.metadata.total_chunks == len(chunks)


@pytest.mark.unit
def test_page_number_tracked_via_form_feed() -> None:
    """Page 1 content before \\x0c, page 2 content after."""
    content = "Стр. 1 текст.\n\nОтчётность за квартал.\x0cСтр. 2 таблица.\n\n| A | B |\n| --- | --- |\n| 1 | 2 |"
    strategy = _make_strategy()
    chunks = strategy.chunk_document(_make_doc(content))
    pages = [c.metadata.page for c in chunks if c.metadata.page is not None]
    assert 1 in pages
    assert 2 in pages


@pytest.mark.unit
def test_multiple_prose_blocks_merged_when_small() -> None:
    """Short prose paragraphs should be merged into fewer chunks."""
    content = "\n\n".join([f"Параграф {i}." for i in range(5)])
    strategy = _make_strategy()
    chunks = strategy.chunk_document(_make_doc(content))
    # All 5 short paragraphs should fit in 1 chunk
    assert len(chunks) == 1


@pytest.mark.unit
def test_metadata_doc_id_matches_document() -> None:
    doc = _make_doc("Текст отчёта.")
    strategy = _make_strategy()
    chunks = strategy.chunk_document(doc)
    assert all(c.metadata.document_id == doc.id for c in chunks)
