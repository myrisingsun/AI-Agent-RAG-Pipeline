from qdrant_client import AsyncQdrantClient
from qdrant_client.models import PayloadSchemaType

from src.common.exceptions import VectorStoreError
from src.common.logging import get_logger
from src.rag.config import CollectionName, RAGConfig
from src.rag.vectorstore.collections import (
    COLLECTION_CONFIGS,
    DENSE_VECTOR_NAME,
    SPARSE_VECTOR_NAME,
    get_dense_vector_params,
    get_sparse_vector_params,
)

logger = get_logger(__name__)

_PAYLOAD_TYPE_MAP: dict[str, PayloadSchemaType] = {
    "keyword": PayloadSchemaType.KEYWORD,
    "integer": PayloadSchemaType.INTEGER,
    "float": PayloadSchemaType.FLOAT,
    "datetime": PayloadSchemaType.DATETIME,
}


class QdrantManager:
    """
    Manages a single AsyncQdrantClient for the application lifetime.
    All methods are async — call initialize() before use.
    """

    def __init__(self, config: RAGConfig) -> None:
        self._config = config
        self._client: AsyncQdrantClient | None = None

    async def initialize(self) -> None:
        """Create client and verify connectivity."""
        self._client = AsyncQdrantClient(
            host=self._config.qdrant_host,
            port=self._config.qdrant_port,
            grpc_port=self._config.qdrant_grpc_port,
            prefer_grpc=self._config.qdrant_prefer_grpc,
            api_key=self._config.qdrant_api_key or None,
            https=self._config.qdrant_prefer_grpc,  # HTTPS only with gRPC+TLS in production
            timeout=30,
        )
        await self.health_check()
        logger.info(
            "Qdrant client initialized",
            host=self._config.qdrant_host,
            port=self._config.qdrant_port,
            grpc=self._config.qdrant_prefer_grpc,
        )

    async def health_check(self) -> None:
        """Raise VectorStoreError if Qdrant is unreachable."""
        try:
            await self.client.get_collections()
        except Exception as exc:
            raise VectorStoreError(
                f"Cannot connect to Qdrant at {self._config.qdrant_host}:{self._config.qdrant_port}: {exc}"
            ) from exc

    async def ensure_collections_exist(self) -> None:
        """
        Create all 4 collections with correct vector configs and payload indexes.
        Idempotent: skips collections that already exist.
        """
        for name, definition in COLLECTION_CONFIGS.items():
            exists = await self.client.collection_exists(name)
            if exists:
                logger.info("collection already exists, skipping", collection=name)
                continue

            await self.client.create_collection(
                collection_name=name,
                vectors_config={DENSE_VECTOR_NAME: get_dense_vector_params(self._config.vector_size)},
                sparse_vectors_config={SPARSE_VECTOR_NAME: get_sparse_vector_params()},
            )
            logger.info(
                "collection created",
                collection=name,
                vector_size=self._config.vector_size,
            )

            for field_name, field_type in definition.payload_indexes.items():
                schema_type = _PAYLOAD_TYPE_MAP.get(field_type)
                if schema_type is None:
                    logger.warning("unknown payload type, skipping index", field=field_name, type=field_type)
                    continue
                await self.client.create_payload_index(
                    collection_name=name,
                    field_name=field_name,
                    field_schema=schema_type,
                )
                logger.debug("payload index created", collection=name, field=field_name)

    async def collection_info(self, name: CollectionName) -> dict:  # type: ignore[type-arg]
        """Return collection info as dict."""
        info = await self.client.get_collection(name)
        return info.model_dump()

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()
            logger.info("Qdrant client closed")

    @property
    def client(self) -> AsyncQdrantClient:
        if self._client is None:
            raise VectorStoreError("QdrantManager not initialized. Call await initialize() first.")
        return self._client
