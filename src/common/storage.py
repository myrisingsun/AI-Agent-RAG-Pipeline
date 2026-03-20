import asyncio
import io
import mimetypes
from pathlib import Path

from src.common.logging import get_logger
from src.rag.config import RAGConfig

logger = get_logger(__name__)


class MinIOStorage:
    """
    Async wrapper around the synchronous minio client.
    Stores original document files in the `documents` bucket.
    """

    def __init__(self, config: RAGConfig) -> None:
        self._config = config
        self._client = None

    async def initialize(self) -> None:
        from minio import Minio

        self._client = Minio(
            endpoint=self._config.minio_endpoint,
            access_key=self._config.minio_root_user,
            secret_key=self._config.minio_root_password,
            secure=self._config.minio_secure,
        )
        await self._ensure_bucket()
        logger.info("MinIO storage initialized", endpoint=self._config.minio_endpoint)

    async def _ensure_bucket(self) -> None:
        loop = asyncio.get_event_loop()
        bucket = self._config.minio_bucket_documents
        exists = await loop.run_in_executor(None, lambda: self._client.bucket_exists(bucket))  # type: ignore[union-attr]
        if not exists:
            await loop.run_in_executor(None, lambda: self._client.make_bucket(bucket))  # type: ignore[union-attr]
            logger.info("MinIO bucket created", bucket=bucket)

    async def upload(self, doc_id: str, filename: str, file_bytes: bytes) -> str:
        """
        Upload file bytes to MinIO.
        Returns the object path: '{bucket}/{doc_id}/{filename}'
        """
        content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
        object_name = f"{doc_id}/{Path(filename).name}"
        bucket = self._config.minio_bucket_documents

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: self._client.put_object(  # type: ignore[union-attr]
                bucket_name=bucket,
                object_name=object_name,
                data=io.BytesIO(file_bytes),
                length=len(file_bytes),
                content_type=content_type,
            ),
        )
        path = f"{bucket}/{object_name}"
        logger.info("file uploaded to MinIO", path=path, size=len(file_bytes))
        return path
