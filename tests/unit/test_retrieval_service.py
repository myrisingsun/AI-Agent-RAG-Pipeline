"""Unit tests for RetrievalService — reranker integration and search flow."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.rag.config import CollectionName, RAGConfig
from src.rag.pipeline.retrieval import RetrievalService


def _make_hits(n: int, base_score: float = 0.8) -> list[dict]:
    return [
        {
            "id": f"chunk-{i}",
            "score": round(base_score - i * 0.05, 4),
            "payload": {
                "text": f"Текст чанка {i}",
                "source_path": f"docs/doc{i}.txt",
                "session_id": "sess-1",
            },
        }
        for i in range(n)
    ]


def _make_service(
    *,
    reranker_enabled: bool = True,
    candidate_limit: int = 50,
    top_k: int = 5,
    hits: list[dict] | None = None,
    reranked: list[dict] | None = None,
) -> tuple[RetrievalService, MagicMock, MagicMock]:
    config = RAGConfig.model_construct(
        reranker_enabled=reranker_enabled,
        reranker_candidate_limit=candidate_limit,
        reranker_top_k=top_k,
        hybrid_search_enabled=False,
        vllm_api_url="http://localhost:8001/v1",
        vllm_model="stub",
    )

    embedding = MagicMock()
    embedding.embed_query = AsyncMock(return_value=[0.1] * 1024)

    _hits = hits if hits is not None else _make_hits(10)
    vs_ops = MagicMock()
    vs_ops.search = AsyncMock(return_value=_hits)

    _reranked = reranked if reranked is not None else _make_hits(top_k)
    reranker = MagicMock()
    reranker.rerank = AsyncMock(return_value=_reranked)

    llm = MagicMock()
    llm.complete = AsyncMock(return_value="Ответ LLM.")

    router = MagicMock()
    router.route = MagicMock(return_value=CollectionName.CURRENT_PACKAGE)

    svc = RetrievalService(config, embedding, vs_ops, llm, reranker, router)
    return svc, vs_ops, reranker


# ─── Reranker enabled ────────────────────────────────────────────────────────

@pytest.mark.unit
async def test_reranker_called_with_candidate_limit() -> None:
    """When reranker enabled, vs.search is called with candidate_limit, not user limit."""
    svc, vs_ops, reranker = _make_service(reranker_enabled=True, candidate_limit=50)

    await svc.search("запрос о залоге", limit=5, session_id="sess-1")

    # search must be called with candidate_limit=50, not limit=5
    call_kwargs = vs_ops.search.call_args
    assert call_kwargs.kwargs["limit"] == 50


@pytest.mark.unit
async def test_reranker_called_with_correct_top_k() -> None:
    """Reranker.rerank is called with top_k equal to the user-requested limit."""
    svc, _, reranker = _make_service(reranker_enabled=True, candidate_limit=50)

    await svc.search("запрос", limit=3)

    reranker.rerank.assert_called_once()
    _, call_kwargs = reranker.rerank.call_args
    assert call_kwargs["top_k"] == 3


@pytest.mark.unit
async def test_reranker_receives_all_candidates() -> None:
    """Reranker.rerank receives the full candidate list from vs.search."""
    candidates = _make_hits(50)
    svc, _, reranker = _make_service(
        reranker_enabled=True, candidate_limit=50, hits=candidates
    )

    await svc.search("запрос", limit=5)

    args, _ = reranker.rerank.call_args
    assert len(args[1]) == 50  # second positional arg is hits


@pytest.mark.unit
async def test_response_contains_reranked_results() -> None:
    """SearchResponse citations come from reranker output, not raw vs.search hits."""
    top5 = _make_hits(5, base_score=0.95)
    svc, _, _ = _make_service(reranker_enabled=True, reranked=top5)

    response = await svc.search("запрос", limit=5)

    chunk_ids = {c.chunk_id for c in response.citations}
    expected_ids = {h["id"] for h in top5}
    assert chunk_ids == expected_ids


# ─── Reranker disabled ───────────────────────────────────────────────────────

@pytest.mark.unit
async def test_reranker_disabled_uses_limit_directly() -> None:
    """When reranker disabled, vs.search is called with user limit, not candidate_limit."""
    svc, vs_ops, reranker = _make_service(reranker_enabled=False, candidate_limit=50)

    await svc.search("запрос", limit=5)

    call_kwargs = vs_ops.search.call_args
    assert call_kwargs.kwargs["limit"] == 5


@pytest.mark.unit
async def test_reranker_disabled_not_called() -> None:
    """When reranker disabled, reranker.rerank is never called."""
    svc, _, reranker = _make_service(reranker_enabled=False)

    await svc.search("запрос", limit=5)

    reranker.rerank.assert_not_called()


@pytest.mark.unit
async def test_reranker_disabled_trims_to_limit() -> None:
    """When reranker disabled, hits are trimmed to limit."""
    hits_20 = _make_hits(20)
    svc, _, _ = _make_service(reranker_enabled=False, hits=hits_20)

    response = await svc.search("запрос", limit=5)

    assert len(response.citations) == 5


# ─── Session filter ──────────────────────────────────────────────────────────

@pytest.mark.unit
async def test_session_filter_passed_to_search() -> None:
    """session_id is forwarded as payload filter to vs.search for CURRENT_PACKAGE."""
    svc, vs_ops, _ = _make_service()

    await svc.search("запрос", session_id="my-session-123")

    call_kwargs = vs_ops.search.call_args
    assert call_kwargs.kwargs["filter_payload"] == {"session_id": "my-session-123"}


# ─── Response structure ──────────────────────────────────────────────────────

@pytest.mark.unit
async def test_response_has_answer_and_citations() -> None:
    svc, _, _ = _make_service()

    response = await svc.search("запрос")

    assert response.answer == "Ответ LLM."
    assert len(response.citations) > 0
    assert response.latency_ms >= 0


@pytest.mark.unit
async def test_citation_fields_populated() -> None:
    hit = {
        "id": "chunk-abc",
        "score": 0.92,
        "payload": {
            "text": "Текст из документа.",
            "source_path": "docs/contract.pdf",
            "page": 3,
            "section": "2",
            "law_article": "Статья 334",
        },
    }
    svc, _, _ = _make_service(reranker_enabled=False, hits=[hit])

    response = await svc.search("запрос", limit=1)

    c = response.citations[0]
    assert c.chunk_id == "chunk-abc"
    assert c.score == 0.92
    assert c.source == "docs/contract.pdf"
    assert c.page == 3
    assert c.law_article == "Статья 334"
