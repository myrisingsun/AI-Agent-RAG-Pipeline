from src.rag.parsing.base import DocumentParser
from src.schemas.document import DocType, ParsedDocument


class TxtParser(DocumentParser):
    """Plain text parser — UTF-8 with replacement for invalid bytes."""

    def parse(self, file_bytes: bytes, filename: str, doc_type: DocType) -> ParsedDocument:
        content = file_bytes.decode("utf-8", errors="replace")
        return ParsedDocument(
            filename=filename,
            doc_type=doc_type,
            content=content,
            pages=1,
        )

    @classmethod
    def supported_extensions(cls) -> list[str]:
        return [".txt"]
