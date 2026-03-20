import time

from src.common.logging import get_logger
from src.rag.config import CollectionName, RAGConfig
from src.rag.embeddings.base import EmbeddingProvider
from src.rag.llm.client import LLMClient
from src.rag.vectorstore.operations import VectorStoreOperations
from src.schemas.api import Citation, SearchResponse

logger = get_logger(__name__)

_QA_PROMPT_PATH = "prompts/rag/qa.txt"
_DEFAULT_QA_PROMPT = """\
Вы — помощник кредитного аналитика банка. Ответьте на вопрос, используя только предоставленный контекст.
Если контекст не содержит ответа, так и скажите.
Ссылайтесь на источники по номерам [1], [2] и т.д.

Контекст:
{context}

Вопрос: {query}

Ответ:"""


def _load_prompt(path: str, default: str) -> str:
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return default


def _build_context(hits: list[dict]) -> tuple[str, list[Citation]]:  # type: ignore[type-arg]
    """Convert search hits to LLM context string and Citation list."""
    citations: list[Citation] = []
    parts: list[str] = []

    for i, hit in enumerate(hits, 1):
        payload = hit.get("payload", {})
        text = str(payload.get("text", f"[chunk {i}]"))
        citation = Citation(
            chunk_id=str(hit["id"]),
            text=text[:500],
            score=round(float(hit["score"]), 4),
            source=payload.get("source_path"),
            page=payload.get("page"),
            section=payload.get("section"),
            law_article=payload.get("law_article"),
        )
        citations.append(citation)
        parts.append(f"[{i}] {text[:800]}")

    return "\n\n".join(parts), citations


class RetrievalService:
    """Embed → search → build context → LLM answer with citations."""

    def __init__(
        self,
        config: RAGConfig,
        embedding_provider: EmbeddingProvider,
        vs_operations: VectorStoreOperations,
        llm_client: LLMClient,
    ) -> None:
        self._config = config
        self._embedding = embedding_provider
        self._vs = vs_operations
        self._llm = llm_client
        self._qa_prompt = _load_prompt(_QA_PROMPT_PATH, _DEFAULT_QA_PROMPT)

    async def search(
        self,
        query: str,
        collection: CollectionName,
        limit: int = 5,
        session_id: str | None = None,
        filters: dict[str, str] | None = None,
    ) -> SearchResponse:
        t0 = time.perf_counter()

        query_vector = await self._embedding.embed_query(query)

        payload_filter: dict[str, str] = {}
        if session_id and collection == CollectionName.CURRENT_PACKAGE:
            payload_filter["session_id"] = session_id
        if filters:
            payload_filter.update(filters)

        hits = await self._vs.search(
            collection=collection,
            query_vector=query_vector,
            limit=limit,
            filter_payload=payload_filter or None,
        )

        context, citations = _build_context(hits)
        prompt = self._qa_prompt.format(context=context, query=query)
        answer = await self._llm.complete(prompt)

        latency_ms = round((time.perf_counter() - t0) * 1000, 1)
        logger.info(
            "retrieval completed",
            query_len=len(query),
            hits=len(hits),
            collection=collection,
            latency_ms=latency_ms,
        )

        return SearchResponse(
            answer=answer,
            query=query,
            citations=citations,
            collection=collection,
            latency_ms=latency_ms,
        )
