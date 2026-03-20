from enum import StrEnum
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class EmbeddingProviderType(StrEnum):
    E5_LARGE = "e5_large"
    BGE_M3 = "bge_m3"
    QWEN3 = "qwen3"


class ChunkingStrategyType(StrEnum):
    FIXED_SIZE = "fixed_size"
    SEMANTIC = "semantic"
    TABLE_AWARE = "table_aware"
    HIERARCHICAL = "hierarchical"


class CollectionName(StrEnum):
    NORMATIVE_BASE = "normative_base"
    DEAL_PRECEDENTS = "deal_precedents"
    REFERENCE_TEMPLATES = "reference_templates"
    CURRENT_PACKAGE = "current_package"


class RAGConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_grpc_port: int = 6334
    qdrant_api_key: str | None = None
    qdrant_prefer_grpc: bool = False  # Enable in production with TLS

    # PostgreSQL
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_user: str = "raguser"
    postgres_password: str = Field(default="changeme")
    postgres_db: str = "ragdb"

    @property
    def postgres_dsn(self) -> str:
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # MinIO
    minio_endpoint: str = "localhost:9000"
    minio_root_user: str = "minioadmin"
    minio_root_password: str = Field(default="changeme")
    minio_bucket_documents: str = "documents"
    minio_secure: bool = False

    # Embedding
    embedding_provider: EmbeddingProviderType = EmbeddingProviderType.E5_LARGE
    embedding_model_name: str = "intfloat/multilingual-e5-large"
    embedding_device: str = "cpu"
    embedding_batch_size: int = 32
    embedding_max_tokens: int = 512
    hf_home: str = "./models"

    # Chunking
    chunking_chunk_size: int = 400
    chunking_chunk_overlap: int = 60

    # LLM
    vllm_api_url: str = "http://localhost:8001/v1"
    vllm_model: str = "Qwen/Qwen2.5-7B-Instruct"

    # Application
    rag_log_level: str = "INFO"

    # JWT auth (set to anything other than "dev" in production)
    jwt_secret: str = "dev"
    jwt_algorithm: str = "HS256"

    @property
    def vector_size(self) -> int:
        sizes = {
            EmbeddingProviderType.E5_LARGE: 1024,
            EmbeddingProviderType.BGE_M3: 1024,
            EmbeddingProviderType.QWEN3: 1536,
        }
        return sizes[self.embedding_provider]


@lru_cache(maxsize=1)
def get_config() -> RAGConfig:
    """Return singleton RAGConfig. Use this everywhere instead of RAGConfig()."""
    return RAGConfig()
