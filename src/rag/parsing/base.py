from abc import ABC, abstractmethod

from src.schemas.document import DocType, ParsedDocument


class DocumentParser(ABC):
    """Abstract base for file format parsers."""

    @abstractmethod
    def parse(self, file_bytes: bytes, filename: str, doc_type: DocType) -> ParsedDocument:
        """Parse raw bytes into a ParsedDocument. doc_type is set by the caller."""
        ...

    @classmethod
    @abstractmethod
    def supported_extensions(cls) -> list[str]:
        """File extensions this parser handles, e.g. ['.pdf']."""
        ...
