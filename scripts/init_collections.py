"""
Creates all 4 Qdrant collections with correct vector configs and payload indexes.
Idempotent: safe to run multiple times.

Usage:
    python scripts/init_collections.py
    make init-collections
"""

import asyncio
import sys
from pathlib import Path

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.common.logging import configure_logging, get_logger
from src.rag.config import CollectionName, get_config
from src.rag.vectorstore.client import QdrantManager


async def main() -> int:
    config = get_config()
    configure_logging(config.rag_log_level)
    logger = get_logger("init_collections")

    logger.info(
        "connecting to Qdrant",
        host=config.qdrant_host,
        port=config.qdrant_port,
    )

    manager = QdrantManager(config)
    try:
        await manager.initialize()
    except Exception as exc:
        logger.error("failed to connect to Qdrant", error=str(exc))
        logger.error("is Qdrant running? try: make up-infra")
        return 1

    await manager.ensure_collections_exist()

    # Print final summary
    for name in CollectionName:
        info = await manager.collection_info(name)
        vectors_count = info.get("vectors_count", 0)
        logger.info("collection ready", collection=name, vectors_count=vectors_count)

    await manager.close()
    logger.info("init_collections complete — all 4 collections ready")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
