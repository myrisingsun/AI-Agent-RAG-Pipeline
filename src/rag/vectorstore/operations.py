import uuid

from qdrant_client.models import Filter, FilterSelector, FieldCondition, MatchValue, NamedVector, PointStruct

from src.common.exceptions import VectorStoreError
from src.common.logging import get_logger
from src.rag.config import CollectionName, RAGConfig
from src.rag.vectorstore.client import QdrantManager
from src.rag.vectorstore.collections import DENSE_VECTOR_NAME
from src.schemas.chunk import Chunk

logger = get_logger(__name__)


class VectorStoreOperations:
    """High-level Qdrant operations used by indexing and retrieval pipelines."""

    def __init__(self, manager: QdrantManager, config: RAGConfig) -> None:
        self._manager = manager
        self._config = config

    async def upsert_chunks(
        self,
        collection: CollectionName,
        chunks: list[Chunk],
    ) -> int:
        """
        Upsert chunks into the given collection.
        Each Chunk must have embedding populated before calling this.
        Returns the number of upserted points.
        """
        points: list[PointStruct] = []
        for chunk in chunks:
            if chunk.embedding is None:
                raise VectorStoreError(
                    f"Chunk {chunk.id} has no embedding. Embed before upsert."
                )
            payload = chunk.metadata.model_dump(exclude_none=True)
            payload["text"] = chunk.text  # stored for context retrieval
            points.append(
                PointStruct(
                    id=str(chunk.id),
                    vector={DENSE_VECTOR_NAME: chunk.embedding},
                    payload=payload,
                )
            )

        await self._manager.client.upsert(
            collection_name=collection,
            points=points,
            wait=True,
        )
        logger.info("chunks upserted", collection=collection, count=len(points))
        return len(points)

    async def search(
        self,
        collection: CollectionName,
        query_vector: list[float],
        limit: int = 10,
        filter_payload: dict[str, str] | None = None,
    ) -> list[dict]:  # type: ignore[type-arg]
        """
        Dense vector search. Returns list of payload dicts with score.
        Hybrid search (BM25 + RRF) added in Sprint 4.
        """
        qdrant_filter = None
        if filter_payload:
            qdrant_filter = Filter(
                must=[
                    FieldCondition(key=k, match=MatchValue(value=v))
                    for k, v in filter_payload.items()
                ]
            )

        results = await self._manager.client.query_points(
            collection_name=collection,
            query=query_vector,
            using=DENSE_VECTOR_NAME,
            limit=limit,
            query_filter=qdrant_filter,
            with_payload=True,
        )
        return [{"id": r.id, "score": r.score, "payload": r.payload} for r in results.points]

    async def delete_by_session(self, session_id: str) -> None:
        """
        Delete all points in current_package matching the given session_id.
        Uses payload index on session_id for O(log n) lookup.
        """
        await self._manager.client.delete(
            collection_name=CollectionName.CURRENT_PACKAGE,
            points_selector=FilterSelector(
                filter=Filter(
                    must=[
                        FieldCondition(
                            key="session_id",
                            match=MatchValue(value=session_id),
                        )
                    ]
                )
            ),
            wait=True,
        )
        logger.info("session deleted from current_package", session_id=session_id)

    async def count(self, collection: CollectionName) -> int:
        """Return number of points in collection."""
        result = await self._manager.client.count(collection_name=collection, exact=True)
        return result.count
