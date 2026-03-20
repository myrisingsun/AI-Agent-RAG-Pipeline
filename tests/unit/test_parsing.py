"""Unit tests for document parsers."""

import pytest

from src.rag.parsing.factory import get_parser
from src.rag.parsing.txt import TxtParser
from src.rag.parsing.pdf import PdfParser
from src.rag.parsing.docx import DocxParser
from src.common.exceptions import ChunkingError
from src.schemas.document import DocType


# ─── Factory ─────────────────────────────────────────────────────────────────

@pytest.mark.unit
def test_factory_returns_txt_parser() -> None:
    parser = get_parser("document.txt")
    assert isinstance(parser, TxtParser)


@pytest.mark.unit
def test_factory_returns_pdf_parser() -> None:
    parser = get_parser("report.PDF")  # uppercase extension
    assert isinstance(parser, PdfParser)


@pytest.mark.unit
def test_factory_returns_docx_parser() -> None:
    parser = get_parser("contract.docx")
    assert isinstance(parser, DocxParser)


@pytest.mark.unit
def test_factory_raises_for_unsupported() -> None:
    with pytest.raises(ChunkingError, match="Unsupported file type"):
        get_parser("spreadsheet.xlsx")


# ─── TxtParser ───────────────────────────────────────────────────────────────

@pytest.mark.unit
def test_txt_parser_basic() -> None:
    content = "Кредитный договор. Ставка 12%."
    result = TxtParser().parse(content.encode("utf-8"), "test.txt", DocType.CONTRACT)
    assert result.content == content
    assert result.doc_type == DocType.CONTRACT
    assert result.pages == 1


@pytest.mark.unit
def test_txt_parser_invalid_bytes_replaced() -> None:
    bad_bytes = b"Hello \xff\xfe World"
    result = TxtParser().parse(bad_bytes, "test.txt", DocType.UNKNOWN)
    assert "Hello" in result.content
    assert "World" in result.content


# ─── PdfParser ───────────────────────────────────────────────────────────────

@pytest.mark.unit
def test_pdf_parser_raises_on_invalid_bytes() -> None:
    with pytest.raises(ChunkingError, match="Failed to read PDF"):
        PdfParser().parse(b"not a pdf", "fake.pdf", DocType.CONTRACT)


# ─── DocxParser ──────────────────────────────────────────────────────────────

@pytest.mark.unit
def test_docx_parser_raises_on_invalid_bytes() -> None:
    with pytest.raises(ChunkingError, match="Failed to read DOCX"):
        DocxParser().parse(b"not a docx", "fake.docx", DocType.CONTRACT)
