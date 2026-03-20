"""
Integration tests for the FastAPI endpoints.
Requires: Qdrant running (make up-infra) + collections initialized (make init-collections).
LLM calls are mocked to avoid vLLM dependency in tests.
"""

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from src.api.main import create_app
from src.rag.config import CollectionName
from src.schemas.api import DocumentUploadResponse, SearchResponse
from src.schemas.document import DocType


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def app():
    return create_app()


@pytest_asyncio.fixture
async def client(app):
    """
    Create a test client with mocked infrastructure:
    - Real Qdrant (requires make up-infra)
    - Mocked embedding provider (no model weights needed)
    - Mocked LLM client (no vLLM needed)
    """
    from src.rag.config import get_config
    from src.rag.vectorstore.client import QdrantManager
    from src.rag.vectorstore.operations import VectorStoreOperations
    from src.rag.pipeline.ingestion import IngestionService
    from src.rag.pipeline.retrieval import RetrievalService
    from src.rag.pipeline.validation import ValidationService

    config = get_config()

    # Real Qdrant
    qdrant_manager = QdrantManager(config)
    await qdrant_manager.initialize()
    vs_operations = VectorStoreOperations(qdrant_manager, config)

    # Mock embedding provider — returns 1024-dim unit vectors
    mock_embedding = MagicMock()
    mock_embedding.embed_texts = AsyncMock(
        side_effect=lambda texts: [[0.0] * 1023 + [1.0]] * len(texts)
    )
    mock_embedding.embed_query = AsyncMock(return_value=[0.0] * 1023 + [1.0])

    # Mock LLM client
    mock_llm = MagicMock()
    mock_llm.complete = AsyncMock(return_value="Тестовый ответ LLM.")
    mock_llm.close = AsyncMock()

    ingestion_service = IngestionService(config, mock_embedding, vs_operations)
    retrieval_service = RetrievalService(config, mock_embedding, vs_operations, mock_llm)
    validation_service = ValidationService(config, mock_embedding, vs_operations, mock_llm)

    # Bypass lifespan — inject state directly
    app.state.config = config
    app.state.qdrant_manager = qdrant_manager
    app.state.embedding_provider = mock_embedding
    app.state.vs_operations = vs_operations
    app.state.llm_client = mock_llm
    app.state.ingestion_service = ingestion_service
    app.state.retrieval_service = retrieval_service
    app.state.validation_service = validation_service
    app.state.document_registry = {}

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as ac:
        yield ac

    await qdrant_manager.close()


# ─── Health ──────────────────────────────────────────────────────────────────

@pytest.mark.integration
async def test_health(client: AsyncClient) -> None:
    response = await client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# ─── Documents ───────────────────────────────────────────────────────────────

@pytest.mark.integration
async def test_upload_txt_document(client: AsyncClient) -> None:
    content = b"Test document content for credit analysis."
    response = await client.post(
        "/api/v1/documents/upload",
        data={"doc_type": DocType.CONTRACT, "session_id": "test-session-001"},
        files={"file": ("test.txt", content, "text/plain")},
    )
    assert response.status_code == 201
    body = response.json()
    assert body["filename"] == "test.txt"
    assert body["doc_type"] == DocType.CONTRACT
    assert body["chunk_count"] >= 1
    assert body["collection"] == CollectionName.CURRENT_PACKAGE


@pytest.mark.integration
async def test_upload_unsupported_format_returns_422(client: AsyncClient) -> None:
    response = await client.post(
        "/api/v1/documents/upload",
        data={"doc_type": DocType.CONTRACT},
        files={"file": ("doc.docx", b"fake docx bytes", "application/vnd.openxmlformats")},
    )
    assert response.status_code == 422


@pytest.mark.integration
async def test_get_document_after_upload(client: AsyncClient) -> None:
    content = b"Кредитный договор с условиями залога."
    upload = await client.post(
        "/api/v1/documents/upload",
        data={"doc_type": DocType.CONTRACT, "session_id": "sess-get-test"},
        files={"file": ("contract.txt", content, "text/plain")},
    )
    assert upload.status_code == 201
    doc_id = upload.json()["id"]

    response = await client.get(f"/api/v1/documents/{doc_id}")
    assert response.status_code == 200
    assert response.json()["id"] == doc_id


@pytest.mark.integration
async def test_get_nonexistent_document_returns_404(client: AsyncClient) -> None:
    response = await client.get(f"/api/v1/documents/{uuid.uuid4()}")
    assert response.status_code == 404


# ─── Search ──────────────────────────────────────────────────────────────────

@pytest.mark.integration
async def test_search_returns_response(client: AsyncClient) -> None:
    response = await client.post(
        "/api/v1/search",
        json={
            "query": "условия кредитного договора",
            "collection": CollectionName.CURRENT_PACKAGE,
            "limit": 3,
            "session_id": "test-session-001",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert "answer" in body
    assert "citations" in body
    assert body["query"] == "условия кредитного договора"


@pytest.mark.integration
async def test_search_query_too_long_returns_422(client: AsyncClient) -> None:
    response = await client.post(
        "/api/v1/search",
        json={"query": "x" * 2001},
    )
    assert response.status_code == 422


# ─── Validate ────────────────────────────────────────────────────────────────

@pytest.mark.integration
async def test_validate_empty_session(client: AsyncClient) -> None:
    response = await client.post(
        "/api/v1/validate",
        json={"session_id": "nonexistent-session-xyz"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "review_required"
    assert len(body["issues"]) > 0


# ─── Collections stats ───────────────────────────────────────────────────────

@pytest.mark.integration
async def test_collections_stats(client: AsyncClient) -> None:
    response = await client.get("/api/v1/collections/stats")
    assert response.status_code == 200
    body = response.json()
    names = [c["name"] for c in body["collections"]]
    assert CollectionName.NORMATIVE_BASE in names
    assert CollectionName.CURRENT_PACKAGE in names
    assert len(body["collections"]) == 4
