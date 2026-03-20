from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.middleware.logging import RequestLoggingMiddleware
from src.api.routes import collections, documents, search, validate, websocket
from src.common.logging import configure_logging, get_logger
from src.rag.config import get_config
from src.rag.embeddings.e5_large import E5LargeEmbeddingProvider
from src.rag.llm.client import LLMClient
from src.rag.pipeline.ingestion import IngestionService
from src.rag.pipeline.retrieval import RetrievalService
from src.rag.pipeline.validation import ValidationService
from src.rag.vectorstore.client import QdrantManager
from src.rag.vectorstore.operations import VectorStoreOperations

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    config = get_config()
    configure_logging(config.rag_log_level)
    logger.info("starting RAG API", log_level=config.rag_log_level)

    # Infrastructure
    qdrant_manager = QdrantManager(config)
    await qdrant_manager.initialize()

    embedding_provider = E5LargeEmbeddingProvider(config)
    await embedding_provider.initialize()

    vs_operations = VectorStoreOperations(qdrant_manager, config)
    llm_client = LLMClient(config)

    # Services
    ingestion_service = IngestionService(config, embedding_provider, vs_operations)
    retrieval_service = RetrievalService(config, embedding_provider, vs_operations, llm_client)
    validation_service = ValidationService(config, embedding_provider, vs_operations, llm_client)

    # Attach to app state
    app.state.config = config
    app.state.qdrant_manager = qdrant_manager
    app.state.embedding_provider = embedding_provider
    app.state.vs_operations = vs_operations
    app.state.llm_client = llm_client
    app.state.ingestion_service = ingestion_service
    app.state.retrieval_service = retrieval_service
    app.state.validation_service = validation_service
    app.state.document_registry: dict = {}  # MVP: in-memory; production: PostgreSQL

    logger.info("RAG API ready")
    yield

    # Cleanup
    await qdrant_manager.close()
    await llm_client.close()
    logger.info("RAG API shutdown complete")


def create_app() -> FastAPI:
    app = FastAPI(
        title="RAG Pipeline API",
        description="Credit collateral document analysis — Q&A + normative compliance",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173"],  # Vite dev server
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    api_v1_prefix = "/api/v1"
    app.include_router(documents.router, prefix=api_v1_prefix)
    app.include_router(search.router, prefix=api_v1_prefix)
    app.include_router(validate.router, prefix=api_v1_prefix)
    app.include_router(collections.router, prefix=api_v1_prefix)
    app.include_router(websocket.router)  # /ws/chat — no /api/v1 prefix

    @app.get("/health", tags=["health"])
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()
