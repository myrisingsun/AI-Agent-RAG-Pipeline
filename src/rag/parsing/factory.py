from pathlib import Path

from src.common.exceptions import ChunkingError
from src.rag.parsing.base import DocumentParser
from src.rag.parsing.docx import DocxParser
from src.rag.parsing.pdf import PdfParser
from src.rag.parsing.txt import TxtParser

_REGISTRY: dict[str, type[DocumentParser]] = {}

for _cls in (TxtParser, PdfParser, DocxParser):
    for _ext in _cls.supported_extensions():
        _REGISTRY[_ext] = _cls


def get_parser(filename: str) -> DocumentParser:
    """Return the correct parser for the given filename. Raises ChunkingError if unsupported."""
    ext = Path(filename).suffix.lower()
    cls = _REGISTRY.get(ext)
    if cls is None:
        supported = ", ".join(sorted(_REGISTRY))
        raise ChunkingError(
            f"Unsupported file type '{ext}' for '{filename}'. "
            f"Supported: {supported}"
        )
    return cls()
