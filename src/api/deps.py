from collections.abc import AsyncIterator

from fastapi import Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession

from src.common.storage import MinIOStorage
from src.rag.repositories.document_repository import DocumentRepository
from src.rag.embeddings.base import EmbeddingProvider
from src.rag.llm.client import LLMClient
from src.rag.pipeline.ingestion import IngestionService
from src.rag.pipeline.retrieval import RetrievalService
from src.rag.pipeline.validation import ValidationService
from src.rag.reranker.base import Reranker
from src.rag.router import QueryRouter
from src.rag.vectorstore.client import QdrantManager
from src.rag.vectorstore.operations import VectorStoreOperations


def get_qdrant_manager(request: Request) -> QdrantManager:
    return request.app.state.qdrant_manager  # type: ignore[no-any-return]


def get_embedding_provider(request: Request) -> EmbeddingProvider:
    return request.app.state.embedding_provider  # type: ignore[no-any-return]


def get_vs_operations(request: Request) -> VectorStoreOperations:
    return request.app.state.vs_operations  # type: ignore[no-any-return]


def get_llm_client(request: Request) -> LLMClient:
    return request.app.state.llm_client  # type: ignore[no-any-return]


def get_reranker(request: Request) -> Reranker:
    return request.app.state.reranker  # type: ignore[no-any-return]


def get_router(request: Request) -> QueryRouter:
    return request.app.state.query_router  # type: ignore[no-any-return]


def get_storage(request: Request) -> MinIOStorage:
    return request.app.state.storage  # type: ignore[no-any-return]


def get_ingestion_service(request: Request) -> IngestionService:
    return request.app.state.ingestion_service  # type: ignore[no-any-return]


def get_retrieval_service(request: Request) -> RetrievalService:
    return request.app.state.retrieval_service  # type: ignore[no-any-return]


def get_validation_service(request: Request) -> ValidationService:
    return request.app.state.validation_service  # type: ignore[no-any-return]


async def get_db_session(request: Request) -> AsyncIterator[AsyncSession]:
    factory = request.app.state.async_session_factory
    if factory is None:
        raise HTTPException(status_code=503, detail="Database unavailable")
    async with factory() as session:
        yield session


async def get_document_repository(
    session: AsyncSession = Depends(get_db_session),
) -> DocumentRepository:
    return DocumentRepository(session)
