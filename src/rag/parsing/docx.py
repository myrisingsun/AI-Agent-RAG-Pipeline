import io

from src.common.exceptions import ChunkingError
from src.rag.parsing.base import DocumentParser
from src.schemas.document import DocType, ParsedDocument


def _format_table_markdown(table: "Table") -> str:  # type: ignore[name-defined]
    """Render a python-docx Table as a Markdown table string."""
    rows: list[list[str]] = []
    for row in table.rows:
        cells = [cell.text.replace("\n", " ").strip() for cell in row.cells]
        rows.append(cells)

    if not rows:
        return ""

    # Deduplicate merged cells (python-docx repeats merged cell text)
    deduped: list[list[str]] = []
    for row in rows:
        deduped_row: list[str] = []
        prev = object()
        for cell in row:
            deduped_row.append(cell if cell != prev else "")
            prev = cell
        deduped.append(deduped_row)

    col_count = max(len(r) for r in deduped)
    # Pad rows to equal width
    padded = [r + [""] * (col_count - len(r)) for r in deduped]

    lines: list[str] = []
    for i, row in enumerate(padded):
        lines.append("| " + " | ".join(row) + " |")
        if i == 0:
            lines.append("| " + " | ".join(["---"] * col_count) + " |")

    return "\n".join(lines)


class DocxParser(DocumentParser):
    """
    DOCX parser using python-docx.
    Preserves document order (paragraphs and tables interleaved).
    Tables are rendered as Markdown for better structure in chunks.
    Note: never use WidthType.PERCENTAGE — it breaks Google Docs compatibility.
    """

    def parse(self, file_bytes: bytes, filename: str, doc_type: DocType) -> ParsedDocument:
        try:
            from docx import Document
            from docx.table import Table
            from docx.text.paragraph import Paragraph
        except ImportError as exc:
            raise ChunkingError(
                "python-docx is not installed. Run: pip install python-docx"
            ) from exc

        try:
            doc = Document(io.BytesIO(file_bytes))
        except Exception as exc:
            raise ChunkingError(f"Failed to read DOCX '{filename}': {exc}") from exc

        parts: list[str] = []

        # Iterate body children in document order to preserve paragraph/table interleaving
        for child in doc.element.body:
            tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag

            if tag == "p":
                para = Paragraph(child, doc)
                text = para.text.strip()
                if text:
                    parts.append(text)

            elif tag == "tbl":
                table = Table(child, doc)
                md = _format_table_markdown(table)
                if md:
                    parts.append(md)

        content = "\n\n".join(parts)
        if not content:
            raise ChunkingError(f"DOCX '{filename}' contains no extractable text.")

        return ParsedDocument(
            filename=filename,
            doc_type=doc_type,
            content=content,
            pages=0,  # DOCX has no fixed page concept
        )

    @classmethod
    def supported_extensions(cls) -> list[str]:
        return [".docx"]
