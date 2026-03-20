"""Unit tests for E5LargeEmbeddingProvider — no model loading required."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.common.exceptions import EmbeddingError
from src.rag.config import RAGConfig
from src.rag.embeddings.base import EmbeddingProvider
from src.rag.embeddings.e5_large import E5LargeEmbeddingProvider


@pytest.mark.unit
def test_e5_provider_implements_interface(config: RAGConfig) -> None:
    provider = E5LargeEmbeddingProvider(config)
    assert isinstance(provider, EmbeddingProvider)


@pytest.mark.unit
def test_vector_size_is_1024(config: RAGConfig) -> None:
    provider = E5LargeEmbeddingProvider(config)
    assert provider.vector_size == 1024


@pytest.mark.unit
def test_model_name_matches_config(config: RAGConfig) -> None:
    provider = E5LargeEmbeddingProvider(config)
    assert provider.model_name == config.embedding_model_name


@pytest.mark.unit
async def test_uninitialized_raises_embedding_error(config: RAGConfig) -> None:
    provider = E5LargeEmbeddingProvider(config)
    with pytest.raises(EmbeddingError, match="not initialized"):
        await provider.embed_texts(["test"])


@pytest.mark.unit
async def test_embed_texts_adds_passage_prefix(config: RAGConfig) -> None:
    provider = E5LargeEmbeddingProvider(config)
    fake_vector = [0.1] * 1024

    with patch.object(
        provider,
        "_encode_batch",
        new_callable=AsyncMock,
        return_value=[fake_vector],
    ) as mock_encode:
        provider._model = MagicMock()  # mark as initialized
        result = await provider.embed_texts(["hello world"])

    mock_encode.assert_called_once_with(["passage: hello world"])
    assert result == [fake_vector]


@pytest.mark.unit
async def test_embed_query_adds_query_prefix(config: RAGConfig) -> None:
    provider = E5LargeEmbeddingProvider(config)
    fake_vector = [0.2] * 1024

    with patch.object(
        provider,
        "_encode_batch",
        new_callable=AsyncMock,
        return_value=[fake_vector],
    ) as mock_encode:
        provider._model = MagicMock()  # mark as initialized
        result = await provider.embed_query("what is the loan rate?")

    mock_encode.assert_called_once_with(["query: what is the loan rate?"])
    assert result == fake_vector


@pytest.mark.unit
async def test_embed_texts_batch_returns_correct_count(config: RAGConfig) -> None:
    provider = E5LargeEmbeddingProvider(config)
    fake_vectors = [[float(i)] * 1024 for i in range(3)]

    with patch.object(
        provider,
        "_encode_batch",
        new_callable=AsyncMock,
        return_value=fake_vectors,
    ):
        provider._model = MagicMock()
        result = await provider.embed_texts(["a", "b", "c"])

    assert len(result) == 3
    assert len(result[0]) == 1024
