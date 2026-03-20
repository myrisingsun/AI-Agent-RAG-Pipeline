import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from src.rag.config import CollectionName
from src.schemas.document import DocType


# ─── Document ────────────────────────────────────────────────────────────────

class DocumentUploadResponse(BaseModel):
    id: uuid.UUID
    filename: str
    doc_type: DocType
    collection: CollectionName
    chunk_count: int
    created_at: datetime


class DocumentResponse(BaseModel):
    id: uuid.UUID
    filename: str
    doc_type: DocType
    collection: CollectionName
    chunk_count: int
    created_at: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)


# ─── Search ──────────────────────────────────────────────────────────────────

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    collection: CollectionName = CollectionName.CURRENT_PACKAGE
    limit: int = Field(default=5, ge=1, le=20)
    session_id: str | None = None
    filters: dict[str, str] = Field(default_factory=dict)


class Citation(BaseModel):
    chunk_id: str
    text: str
    score: float
    source: str | None = None
    page: int | None = None
    section: str | None = None
    law_article: str | None = None


class SearchResponse(BaseModel):
    answer: str
    query: str
    citations: list[Citation]
    collection: CollectionName
    latency_ms: float


# ─── Validate ────────────────────────────────────────────────────────────────

class ValidateRequest(BaseModel):
    session_id: str
    document_id: uuid.UUID | None = None


class ValidationIssue(BaseModel):
    severity: str  # "critical" | "warning" | "info"
    description: str
    law_article: str | None = None
    citation: Citation | None = None


class ValidateResponse(BaseModel):
    session_id: str
    status: str  # "compliant" | "non_compliant" | "review_required"
    issues: list[ValidationIssue]
    summary: str
    checked_articles: list[str]


# ─── Collections ─────────────────────────────────────────────────────────────

class CollectionStat(BaseModel):
    name: CollectionName
    point_count: int
    vector_size: int
    status: str


class CollectionsStatsResponse(BaseModel):
    collections: list[CollectionStat]


# ─── WebSocket ───────────────────────────────────────────────────────────────

class WsChatMessage(BaseModel):
    query: str
    session_id: str
    collection: CollectionName = CollectionName.CURRENT_PACKAGE


class WsToken(BaseModel):
    type: str = "token"
    content: str


class WsCitation(BaseModel):
    type: str = "citation"
    citations: list[Citation]


class WsError(BaseModel):
    type: str = "error"
    message: str


class WsDone(BaseModel):
    type: str = "done"
    latency_ms: float
