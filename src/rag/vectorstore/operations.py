from typing import Any

from qdrant_client.models import (
    FieldCondition,
    Filter,
    FilterSelector,
    Fusion,
    FusionQuery,
    MatchValue,
    PointStruct,
    Prefetch,
    SparseVector,
)

from src.common.exceptions import VectorStoreError
from src.common.logging import get_logger
from src.rag.config import CollectionName, RAGConfig
from src.rag.vectorstore.client import QdrantManager
from src.rag.vectorstore.collections import DENSE_VECTOR_NAME, SPARSE_VECTOR_NAME
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
        Each Chunk must have embedding populated. sparse_vector is optional.
        Returns number of upserted points.
        """
        points: list[PointStruct] = []
        for chunk in chunks:
            if chunk.embedding is None:
                raise VectorStoreError(
                    f"Chunk {chunk.id} has no embedding. Embed before upsert."
                )
            payload: dict[str, Any] = chunk.metadata.model_dump(exclude_none=True)
            payload["text"] = chunk.text

            vector: dict[str, Any] = {DENSE_VECTOR_NAME: chunk.embedding}
            if chunk.sparse_vector:
                indices = list(chunk.sparse_vector.keys())
                values = list(chunk.sparse_vector.values())
                vector[SPARSE_VECTOR_NAME] = SparseVector(indices=indices, values=values)

            points.append(PointStruct(id=str(chunk.id), vector=vector, payload=payload))

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
        sparse_vector: dict[int, float] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Vector search. Uses hybrid RRF (dense + BM25) when sparse_vector is provided
        and hybrid_search_enabled=True, otherwise dense-only.
        """
        qdrant_filter = None
        if filter_payload:
            qdrant_filter = Filter(
                must=[
                    FieldCondition(key=k, match=MatchValue(value=v))
                    for k, v in filter_payload.items()
                ]
            )

        candidate_limit = max(limit, self._config.reranker_candidate_limit)

        if sparse_vector and self._config.hybrid_search_enabled:
            results = await self._manager.client.query_points(
                collection_name=collection,
                prefetch=[
                    Prefetch(
                        query=query_vector,
                        using=DENSE_VECTOR_NAME,
                        limit=candidate_limit,
                        filter=qdrant_filter,
                    ),
                    Prefetch(
                        query=SparseVector(
                            indices=list(sparse_vector.keys()),
                            values=list(sparse_vector.values()),
                        ),
                        using=SPARSE_VECTOR_NAME,
                        limit=candidate_limit,
                        filter=qdrant_filter,
                    ),
                ],
                query=FusionQuery(fusion=Fusion.RRF),
                limit=limit,
                with_payload=True,
            )
        else:
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
        """Delete all current_package points matching session_id."""
        await self._manager.client.delete(
            collection_name=CollectionName.CURRENT_PACKAGE,
            points_selector=FilterSelector(
                filter=Filter(
                    must=[FieldCondition(key="session_id", match=MatchValue(value=session_id))]
                )
            ),
            wait=True,
        )
        logger.info("session deleted from current_package", session_id=session_id)

    async def count(self, collection: CollectionName) -> int:
        result = await self._manager.client.count(collection_name=collection, exact=True)
        return result.count
