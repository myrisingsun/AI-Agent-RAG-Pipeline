"""Async SQLAlchemy engine + session factory."""

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from src.common.logging import get_logger
from src.rag.config import RAGConfig

logger = get_logger(__name__)


class Base(DeclarativeBase):
    pass


def build_engine(config: RAGConfig) -> AsyncEngine:
    return create_async_engine(
        config.postgres_dsn,
        echo=False,
        pool_size=5,
        max_overflow=10,
    )


def build_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False)


async def init_tables(engine: AsyncEngine) -> None:
    """Create all tables if they don't exist (idempotent for schema additions via ALTER TABLE)."""
    # Import models so their metadata is registered on Base
    import src.models.document  # noqa: F401

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("database tables initialised")
