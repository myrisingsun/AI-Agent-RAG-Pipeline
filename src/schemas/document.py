import uuid
from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class DocType(StrEnum):
    CONTRACT = "contract"
    FINANCIAL_REPORT = "financial_report"
    NORMATIVE = "normative"
    TEMPLATE = "template"
    UNKNOWN = "unknown"


class ParsedDocument(BaseModel):
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    filename: str
    doc_type: DocType
    content: str
    pages: int = 0
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, str | int | float | bool] = Field(default_factory=dict)
