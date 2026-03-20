from typing import Any

from fastapi import APIRouter, Depends

from src.api.deps import get_qdrant_manager, get_vs_operations
from src.api.middleware.auth import require_auth
from src.rag.config import CollectionName, RAGConfig
from src.rag.vectorstore.client import QdrantManager
from src.rag.vectorstore.operations import VectorStoreOperations
from src.schemas.api import CollectionStat, CollectionsStatsResponse

router = APIRouter(prefix="/collections", tags=["collections"])


def _get_config(request: Any) -> RAGConfig:
    return request.app.state.config  # type: ignore[no-any-return]


@router.get("/stats", response_model=CollectionsStatsResponse)
async def get_collections_stats(
    manager: QdrantManager = Depends(get_qdrant_manager),
    vs: VectorStoreOperations = Depends(get_vs_operations),
    _user: dict[str, Any] = Depends(require_auth),
) -> CollectionsStatsResponse:
    """Return point counts, vector sizes, and status for all 4 Qdrant collections."""
    stats: list[CollectionStat] = []
    for name in CollectionName:
        try:
            info = await manager.collection_info(name)
            vectors_config = info.get("config", {}).get("params", {}).get("vectors", {})
            # Extract size from named vectors config {"text": {"size": 1024, ...}}
            text_cfg = vectors_config.get("text", {})
            vector_size = int(text_cfg.get("size", 0))
            point_count = await vs.count(name)
            qdrant_status = str(info.get("status", "unknown"))
        except Exception:
            point_count = -1
            vector_size = 0
            qdrant_status = "unavailable"

        stats.append(
            CollectionStat(
                name=name,
                point_count=point_count,
                vector_size=vector_size,
                status=qdrant_status,
            )
        )

    return CollectionsStatsResponse(collections=stats)
