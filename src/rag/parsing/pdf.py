import io

from src.common.exceptions import ChunkingError
from src.rag.parsing.base import DocumentParser
from src.schemas.document import DocType, ParsedDocument

# Form-feed character (\x0c) separates pages in the extracted content.
# Chunkers can use content[:offset].count('\x0c') + 1 to determine page number.
PAGE_SEPARATOR = "\x0c"


class PdfParser(DocumentParser):
    """
    PDF parser using pypdf. Extracts text page-by-page.
    Pages are joined with \\x0c (form-feed) so chunkers can track page numbers.
    """

    def parse(self, file_bytes: bytes, filename: str, doc_type: DocType) -> ParsedDocument:
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise ChunkingError("pypdf is not installed. Run: pip install pypdf") from exc

        try:
            reader = PdfReader(io.BytesIO(file_bytes))
        except Exception as exc:
            raise ChunkingError(f"Failed to read PDF '{filename}': {exc}") from exc

        page_texts: list[str] = []
        for page in reader.pages:
            text = page.extract_text() or ""
            stripped = text.strip()
            if stripped:
                page_texts.append(stripped)

        content = PAGE_SEPARATOR.join(page_texts)
        if not content:
            raise ChunkingError(f"PDF '{filename}' contains no extractable text (scanned image?).")

        return ParsedDocument(
            filename=filename,
            doc_type=doc_type,
            content=content,
            pages=len(reader.pages),
        )

    @classmethod
    def supported_extensions(cls) -> list[str]:
        return [".pdf"]
