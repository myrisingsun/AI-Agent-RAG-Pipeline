import uuid
from pydantic import BaseModel, Field


class ChunkMetadata(BaseModel):
    """Authoritative payload schema for all 4 Qdrant collections."""

    document_id: uuid.UUID
    doc_type: str
    chunk_index: int
    total_chunks: int
    page: int | None = None
    section: str | None = None
    source_path: str | None = None

    # normative_base
    law_article: str | None = None
    effective_date: str | None = None

    # deal_precedents
    deal_id: str | None = None
    outcome: str | None = None
    risk_flags: list[str] = Field(default_factory=list)

    # reference_templates
    template_version: str | None = None

    # current_package
    session_id: str | None = None


class Chunk(BaseModel):
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    text: str
    token_count: int
    metadata: ChunkMetadata
    embedding: list[float] | None = None
    sparse_vector: dict[int, float] | None = None
