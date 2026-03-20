import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.document import DocumentRecord
from src.schemas.api import DocumentUploadResponse


class DocumentRepository:
    """Thin async repository over the `documents` table."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def save(
        self,
        doc: DocumentUploadResponse,
        session_id: str | None = None,
        source_path: str | None = None,
    ) -> None:
        record = DocumentRecord(
            id=doc.id,
            filename=doc.filename,
            doc_type=str(doc.doc_type),
            collection=str(doc.collection),
            chunk_count=doc.chunk_count,
            session_id=session_id,
            source_path=source_path,
            created_at=doc.created_at,
        )
        self._session.add(record)
        await self._session.commit()

    async def get_by_id(self, doc_id: uuid.UUID) -> DocumentRecord | None:
        result = await self._session.execute(
            select(DocumentRecord).where(DocumentRecord.id == doc_id)
        )
        return result.scalar_one_or_none()

    async def list_by_session(self, session_id: str) -> list[DocumentRecord]:
        result = await self._session.execute(
            select(DocumentRecord).where(DocumentRecord.session_id == session_id)
        )
        return list(result.scalars().all())
