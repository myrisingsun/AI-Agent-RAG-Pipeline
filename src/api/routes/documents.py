import uuid
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Form, HTTPException, UploadFile, status

from src.api.deps import get_document_registry, get_ingestion_service
from src.api.middleware.auth import require_auth
from src.common.exceptions import ChunkingError
from src.rag.pipeline.ingestion import IngestionService
from src.schemas.api import DocumentResponse, DocumentUploadResponse
from src.schemas.document import DocType

router = APIRouter(prefix="/documents", tags=["documents"])

_MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB


@router.post("/upload", response_model=DocumentUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile,
    doc_type: Annotated[DocType, Form()],
    session_id: Annotated[str | None, Form()] = None,
    ingestion: IngestionService = Depends(get_ingestion_service),
    registry: dict[str, Any] = Depends(get_document_registry),
    _user: dict[str, Any] = Depends(require_auth),
) -> DocumentUploadResponse:
    """
    Upload a document and index it into the appropriate Qdrant collection.
    Returns document id, chunk count, and target collection.
    """
    if file.filename is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Filename is required")

    file_bytes = await file.read()
    if len(file_bytes) == 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="File is empty")
    if len(file_bytes) > _MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds {_MAX_FILE_SIZE // (1024 * 1024)} MB limit",
        )

    try:
        result = await ingestion.ingest(
            file_bytes=file_bytes,
            filename=file.filename,
            doc_type=doc_type,
            session_id=session_id,
        )
    except ChunkingError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    registry[str(result.id)] = result.model_dump()
    return result


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: uuid.UUID,
    registry: dict[str, Any] = Depends(get_document_registry),
    _user: dict[str, Any] = Depends(require_auth),
) -> DocumentResponse:
    """Return document metadata by id."""
    doc = registry.get(str(document_id))
    if doc is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
    return DocumentResponse(**doc)
