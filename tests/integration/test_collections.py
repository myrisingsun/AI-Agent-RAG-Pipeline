"""
Integration tests for Qdrant collection management.

Runs against the local Qdrant instance (make up-infra required).
Uses a dedicated test prefix to isolate from production collections.

Run: pytest tests/integration/ -v -m integration
     make test-integration
"""

import uuid

import pytest

from src.common.exceptions import VectorStoreError
from src.rag.config import CollectionName, RAGConfig
from src.rag.vectorstore.client import QdrantManager
from src.rag.vectorstore.collections import SPARSE_VECTOR_NAME
from src.rag.vectorstore.operations import VectorStoreOperations
from src.schemas.chunk import Chunk, ChunkMetadata


@pytest.fixture(scope="module")
def integration_config() -> RAGConfig:
    """Config pointing to local Qdrant (make up-infra must be running)."""
    return RAGConfig(
        postgres_password="testpassword",
        minio_root_password="testpassword",
        qdrant_host="localhost",
        qdrant_port=6333,
        qdrant_prefer_grpc=False,  # REST for tests — simpler debugging
    )


@pytest.fixture(scope="module")
async def manager(integration_config: RAGConfig) -> QdrantManager:  # type: ignore[misc]
    """Initialized QdrantManager. Requires local Qdrant to be running."""
    mgr = QdrantManager(integration_config)
    await mgr.initialize()
    yield mgr
    await mgr.close()


@pytest.mark.integration
async def test_health_check_succeeds(manager: QdrantManager) -> None:
    # initialize() already calls health_check() — reaching here means it passed
    assert manager.client is not None


@pytest.mark.integration
async def test_all_four_collections_created(manager: QdrantManager) -> None:
    await manager.ensure_collections_exist()
    for name in CollectionName:
        exists = await manager.client.collection_exists(name)
        assert exists, f"Collection '{name}' was not created"


@pytest.mark.integration
async def test_collection_creation_idempotent(manager: QdrantManager) -> None:
    """Calling ensure_collections_exist() twice must not raise."""
    await manager.ensure_collections_exist()
    await manager.ensure_collections_exist()


@pytest.mark.integration
async def test_normative_base_vector_size_1024(manager: QdrantManager) -> None:
    info = await manager.collection_info(CollectionName.NORMATIVE_BASE)
    vectors_config = info["config"]["params"]["vectors_config"]
    assert "text" in vectors_config, "Named vector 'text' not found"
    assert vectors_config["text"]["size"] == 1024


@pytest.mark.integration
async def test_sparse_vector_config_present(manager: QdrantManager) -> None:
    for name in CollectionName:
        info = await manager.collection_info(name)
        sparse = info["config"]["params"].get("sparse_vectors_config") or {}
        assert SPARSE_VECTOR_NAME in sparse, (
            f"Collection '{name}' missing sparse vector '{SPARSE_VECTOR_NAME}'"
        )


@pytest.mark.integration
async def test_payload_index_session_id_on_current_package(
    manager: QdrantManager,
) -> None:
    info = await manager.collection_info(CollectionName.CURRENT_PACKAGE)
    payload_schema = info.get("payload_schema") or {}
    assert "session_id" in payload_schema, (
        "session_id payload index missing on current_package"
    )


@pytest.mark.integration
async def test_upsert_and_search_one_point(
    manager: QdrantManager, integration_config: RAGConfig
) -> None:
    """End-to-end: upsert a fake 1024-dim vector and retrieve it by search."""
    ops = VectorStoreOperations(manager, integration_config)

    # Fake unit vector (normalized)
    dim = 1024
    vector = [1.0 / dim**0.5] * dim

    test_session = f"test-integration-{uuid.uuid4().hex[:8]}"
    chunk = Chunk(
        id=uuid.uuid4(),
        text="тестовый документ залога",
        token_count=3,
        metadata=ChunkMetadata(
            document_id=uuid.uuid4(),
            doc_type="contract",
            chunk_index=0,
            total_chunks=1,
            session_id=test_session,
        ),
        embedding=vector,
    )

    count = await ops.upsert_chunks(CollectionName.CURRENT_PACKAGE, [chunk])
    assert count == 1

    results = await ops.search(
        CollectionName.CURRENT_PACKAGE,
        query_vector=vector,
        limit=1,
        filter_payload={"session_id": test_session},
    )
    assert len(results) == 1
    assert results[0]["score"] > 0.99

    # Cleanup: remove test session points
    await ops.delete_by_session(test_session)
    results_after = await ops.search(
        CollectionName.CURRENT_PACKAGE,
        query_vector=vector,
        limit=1,
        filter_payload={"session_id": test_session},
    )
    assert len(results_after) == 0
