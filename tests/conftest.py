import pytest

from src.rag.config import RAGConfig


@pytest.fixture(scope="session")
def config() -> RAGConfig:
    """Test config with safe defaults (no real services required for unit tests)."""
    return RAGConfig(
        postgres_password="testpassword",
        minio_root_password="testpassword",
        qdrant_prefer_grpc=False,
        hf_home="./models",
    )
