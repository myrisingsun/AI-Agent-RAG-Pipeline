from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.middleware.logging import RequestLoggingMiddleware
from src.common.database import build_engine, build_session_factory, init_tables
from src.api.routes import collections, documents, search, validate, websocket
from src.common.logging import configure_logging, get_logger
from src.common.storage import MinIOStorage
from src.rag.config import get_config
from src.rag.embeddings.e5_large import E5LargeEmbeddingProvider
from src.rag.llm.client import LLMClient
from src.rag.pipeline.ingestion import IngestionService
from src.rag.pipeline.retrieval import RetrievalService
from src.rag.pipeline.validation import ValidationService
from src.rag.reranker.cross_encoder import CrossEncoderReranker
from src.rag.router import QueryRouter
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

    # Reranker (optional — disable with RERANKER_ENABLED=false)
    reranker = CrossEncoderReranker(config)
    if config.reranker_enabled:
        await reranker.initialize()
    else:
        logger.info("reranker disabled via RERANKER_ENABLED=false")

    # Query router
    query_router = QueryRouter()

    # MinIO storage
    storage = MinIOStorage(config)
    try:
        await storage.initialize()
    except Exception as exc:
        logger.warning("MinIO unavailable, uploads will skip file storage", error=str(exc))

    # Services
    ingestion_service = IngestionService(config, embedding_provider, vs_operations, storage)
    retrieval_service = RetrievalService(
        config, embedding_provider, vs_operations, llm_client, reranker, query_router
    )
    validation_service = ValidationService(config, embedding_provider, vs_operations, llm_client)

    # Attach to app state
    app.state.config = config
    app.state.qdrant_manager = qdrant_manager
    app.state.embedding_provider = embedding_provider
    app.state.vs_operations = vs_operations
    app.state.llm_client = llm_client
    app.state.reranker = reranker
    app.state.query_router = query_router
    app.state.storage = storage
    app.state.ingestion_service = ingestion_service
    app.state.retrieval_service = retrieval_service
    app.state.validation_service = validation_service
    # PostgreSQL
    db_engine = build_engine(config)
    try:
        await init_tables(db_engine)
        app.state.async_session_factory = build_session_factory(db_engine)
        app.state.db_engine = db_engine
    except Exception as exc:
        logger.warning("PostgreSQL unavailable, document registry disabled", error=str(exc))
        app.state.async_session_factory = None
        app.state.db_engine = None

    logger.info("RAG API ready", reranker=config.reranker_enabled)
    yield

    # Cleanup
    await qdrant_manager.close()
    await llm_client.close()
    if app.state.db_engine:
        await app.state.db_engine.dispose()
    logger.info("RAG API shutdown complete")


def create_app() -> FastAPI:
    app = FastAPI(
        title="RAG Pipeline API",
        description="Credit collateral document analysis — Q&A + normative compliance",
        version="0.2.0",
        lifespan=lifespan,
    )

    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    api_v1_prefix = "/api/v1"
    app.include_router(documents.router, prefix=api_v1_prefix)
    app.include_router(search.router, prefix=api_v1_prefix)
    app.include_router(validate.router, prefix=api_v1_prefix)
    app.include_router(collections.router, prefix=api_v1_prefix)
    app.include_router(websocket.router)

    @app.get("/health", tags=["health"])
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()
