from typing import Any

from fastapi import APIRouter, Depends

from src.api.deps import get_retrieval_service
from src.api.middleware.auth import require_auth
from src.rag.pipeline.retrieval import RetrievalService
from src.schemas.api import SearchRequest, SearchResponse

router = APIRouter(prefix="/search", tags=["search"])


@router.post("", response_model=SearchResponse)
async def search(
    request: SearchRequest,
    retrieval: RetrievalService = Depends(get_retrieval_service),
    _user: dict[str, Any] = Depends(require_auth),
) -> SearchResponse:
    """
    Q&A endpoint: embed query → retrieve relevant chunks → answer with citations.
    """
    return await retrieval.search(
        query=request.query,
        collection=request.collection,
        limit=request.limit,
        session_id=request.session_id,
        filters=request.filters or None,
    )
