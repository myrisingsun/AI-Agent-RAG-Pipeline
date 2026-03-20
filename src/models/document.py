import uuid
from datetime import datetime

from sqlalchemy import DateTime, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from src.common.database import Base


class DocumentRecord(Base):
    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True)
    filename: Mapped[str] = mapped_column(String(500))
    doc_type: Mapped[str] = mapped_column(String(50))
    collection: Mapped[str] = mapped_column(String(100))
    chunk_count: Mapped[int] = mapped_column(Integer)
    session_id: Mapped[str | None] = mapped_column(String(200), nullable=True, index=True)
    source_path: Mapped[str | None] = mapped_column(String(1000), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
