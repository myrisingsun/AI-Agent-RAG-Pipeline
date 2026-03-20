from fastapi import Request

from src.rag.embeddings.base import EmbeddingProvider
from src.rag.llm.client import LLMClient
from src.rag.pipeline.ingestion import IngestionService
from src.rag.pipeline.retrieval import RetrievalService
from src.rag.pipeline.validation import ValidationService
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


def get_ingestion_service(request: Request) -> IngestionService:
    return request.app.state.ingestion_service  # type: ignore[no-any-return]


def get_retrieval_service(request: Request) -> RetrievalService:
    return request.app.state.retrieval_service  # type: ignore[no-any-return]


def get_validation_service(request: Request) -> ValidationService:
    return request.app.state.validation_service  # type: ignore[no-any-return]


def get_document_registry(request: Request) -> dict:  # type: ignore[type-arg]
    return request.app.state.document_registry  # type: ignore[no-any-return]
