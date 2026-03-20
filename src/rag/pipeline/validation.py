from src.common.logging import get_logger
from src.rag.config import CollectionName, RAGConfig
from src.rag.embeddings.base import EmbeddingProvider
from src.rag.llm.client import LLMClient
from src.rag.vectorstore.operations import VectorStoreOperations
from src.schemas.api import Citation, ValidateResponse, ValidationIssue

logger = get_logger(__name__)

_VALIDATE_PROMPT_PATH = "prompts/rag/validate.txt"
_DEFAULT_VALIDATE_PROMPT = """\
Вы — юрист банка, специализирующийся на нормативном соответствии.
Проверьте кредитный документ на соответствие нормативным требованиям ЦБ РФ.

Документ:
{document_context}

Применимые нормативы:
{normative_context}

Выявите нарушения (критичные / предупреждения / информационные).
Укажите конкретные статьи и пункты, которым документ не соответствует.
Если нарушений нет — напишите "Документ соответствует нормативным требованиям."

Заключение:"""

# Representative query used to retrieve document chunks for validation
_VALIDATION_QUERY = "кредитное соглашение обеспечение залог условия договора"


def _load_prompt(path: str, default: str) -> str:
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return default


class ValidationService:
    """Check current_package session against normative_base for compliance."""

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
        self._validate_prompt = _load_prompt(_VALIDATE_PROMPT_PATH, _DEFAULT_VALIDATE_PROMPT)

    async def validate(
        self,
        session_id: str,
        document_id: str | None = None,
    ) -> ValidateResponse:
        """
        Retrieve session documents, search relevant normative articles,
        and call LLM to produce a compliance report.
        """
        query_vector = await self._embedding.embed_query(_VALIDATION_QUERY)

        # 1. Fetch document chunks from current_package
        session_filter: dict[str, str] = {"session_id": session_id}
        if document_id:
            session_filter["document_id"] = document_id

        doc_hits = await self._vs.search(
            collection=CollectionName.CURRENT_PACKAGE,
            query_vector=query_vector,
            limit=10,
            filter_payload=session_filter,
        )

        if not doc_hits:
            return ValidateResponse(
                session_id=session_id,
                status="review_required",
                issues=[
                    ValidationIssue(
                        severity="warning",
                        description=(
                            "Документы сессии не найдены. "
                            "Загрузите документы перед запуском проверки."
                        ),
                    )
                ],
                summary="Нет документов для проверки.",
                checked_articles=[],
            )

        # 2. Build document context
        doc_parts = [
            f"[{i}] {str(hit['payload'].get('text', ''))[:600]}"
            for i, hit in enumerate(doc_hits, 1)
        ]
        doc_context = "\n\n".join(doc_parts)

        # 3. Search normative_base for relevant articles
        norm_hits = await self._vs.search(
            collection=CollectionName.NORMATIVE_BASE,
            query_vector=query_vector,
            limit=5,
        )
        norm_parts = [
            f"[Ст. {hit['payload'].get('law_article', '?')}] "
            f"{str(hit['payload'].get('text', ''))[:400]}"
            for hit in norm_hits
        ]
        norm_context = "\n\n".join(norm_parts)
        checked_articles = [
            str(hit["payload"]["law_article"])
            for hit in norm_hits
            if hit["payload"].get("law_article")
        ]

        # 4. Call LLM
        prompt = self._validate_prompt.format(
            document_context=doc_context,
            normative_context=norm_context,
        )
        llm_response = await self._llm.complete(prompt)

        # 5. Parse issues (MVP: keyword heuristic; production: structured JSON output)
        issues = self._parse_issues(llm_response, norm_hits)
        if any(i.severity == "critical" for i in issues):
            status = "non_compliant"
        elif any(i.severity == "warning" for i in issues):
            status = "review_required"
        else:
            status = "compliant"

        logger.info(
            "validation completed",
            session_id=session_id,
            status=status,
            issues=len(issues),
        )

        return ValidateResponse(
            session_id=session_id,
            status=status,
            issues=issues,
            summary=llm_response[:500],
            checked_articles=checked_articles,
        )

    def _parse_issues(
        self,
        llm_response: str,
        norm_hits: list[dict],  # type: ignore[type-arg]
    ) -> list[ValidationIssue]:
        """MVP: single issue from raw LLM text. Production: parse structured JSON."""
        if not llm_response.strip():
            return []

        lower = llm_response.lower()
        if any(kw in lower for kw in ["нарушение", "не соответствует", "критично"]):
            severity = "critical"
        elif any(kw in lower for kw in ["рекомендуется", "следует", "предупреждение"]):
            severity = "warning"
        else:
            severity = "info"

        article = None
        citation = None
        if norm_hits:
            p = norm_hits[0]["payload"]
            article = p.get("law_article")
            citation = Citation(
                chunk_id=str(norm_hits[0]["id"]),
                text=str(p.get("text", ""))[:300],
                score=round(float(norm_hits[0]["score"]), 4),
                law_article=article,
            )

        return [
            ValidationIssue(
                severity=severity,
                description=llm_response[:1000],
                law_article=article,
                citation=citation,
            )
        ]
