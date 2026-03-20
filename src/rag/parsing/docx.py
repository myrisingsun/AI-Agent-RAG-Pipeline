import io

from src.common.exceptions import ChunkingError
from src.rag.parsing.base import DocumentParser
from src.schemas.document import DocType, ParsedDocument


class DocxParser(DocumentParser):
    """
    DOCX parser using python-docx.
    Extracts paragraphs and table cell text preserving document order.
    Note: never use WidthType.PERCENTAGE — it breaks Google Docs compatibility.
    """

    def parse(self, file_bytes: bytes, filename: str, doc_type: DocType) -> ParsedDocument:
        try:
            from docx import Document
        except ImportError as exc:
            raise ChunkingError(
                "python-docx is not installed. Run: pip install python-docx"
            ) from exc

        try:
            doc = Document(io.BytesIO(file_bytes))
        except Exception as exc:
            raise ChunkingError(f"Failed to read DOCX '{filename}': {exc}") from exc

        parts: list[str] = []

        # Paragraphs
        for para in doc.paragraphs:
            text = para.text.strip()
            if text:
                parts.append(text)

        # Tables (cell text, row by row)
        for table in doc.tables:
            for row in table.rows:
                row_text = " | ".join(
                    cell.text.strip() for cell in row.cells if cell.text.strip()
                )
                if row_text:
                    parts.append(row_text)

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
