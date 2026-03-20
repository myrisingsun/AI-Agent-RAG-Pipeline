"""Unit tests for the cross-encoder reranker (model mocked)."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.rag.reranker.cross_encoder import CrossEncoderReranker
from src.rag.reranker.base import Reranker
from src.rag.config import RAGConfig


@pytest.fixture
def config() -> RAGConfig:
    return RAGConfig()


@pytest.fixture
def reranker(config: RAGConfig) -> CrossEncoderReranker:
    r = CrossEncoderReranker(config)
    # Inject a mock model
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([0.3, 0.9, 0.1])
    r._model = mock_model
    return r


def _make_hits(n: int) -> list[dict]:
    return [
        {"id": f"id-{i}", "score": 0.5, "payload": {"text": f"passage {i}"}}
        for i in range(n)
    ]


@pytest.mark.unit
def test_reranker_implements_base() -> None:
    config = RAGConfig()
    assert isinstance(CrossEncoderReranker(config), Reranker)


@pytest.mark.unit
async def test_reranker_sorts_by_score(reranker: CrossEncoderReranker) -> None:
    hits = _make_hits(3)
    result = await reranker.rerank("запрос", hits, top_k=3)
    # Mock returns [0.3, 0.9, 0.1] → sorted: hit-1, hit-0, hit-2
    assert result[0]["id"] == "id-1"
    assert result[1]["id"] == "id-0"
    assert result[2]["id"] == "id-2"


@pytest.mark.unit
async def test_reranker_respects_top_k(reranker: CrossEncoderReranker) -> None:
    hits = _make_hits(3)
    result = await reranker.rerank("запрос", hits, top_k=2)
    assert len(result) == 2


@pytest.mark.unit
async def test_reranker_empty_hits(reranker: CrossEncoderReranker) -> None:
    result = await reranker.rerank("запрос", [], top_k=5)
    assert result == []


@pytest.mark.unit
async def test_reranker_overwrites_score(reranker: CrossEncoderReranker) -> None:
    hits = _make_hits(3)
    result = await reranker.rerank("запрос", hits, top_k=3)
    # Scores should now be reranker scores, not original 0.5
    assert result[0]["score"] == round(0.9, 4)


@pytest.mark.unit
async def test_reranker_not_initialized_raises() -> None:
    from src.rag.reranker.cross_encoder import RerankerError
    config = RAGConfig()
    r = CrossEncoderReranker(config)  # _model is None
    with pytest.raises(RerankerError, match="not initialized"):
        await r.rerank("query", _make_hits(2), top_k=1)
