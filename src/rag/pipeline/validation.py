import json
import re

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

Верните ответ в формате JSON: issues (массив с полями severity/article/violation/recommendation) и summary.

JSON:"""

# Representative query used to retrieve document chunks for validation
_VALIDATION_QUERY = "кредитное соглашение обеспечение залог условия договора"

_VALID_SEVERITIES = {"critical", "warning", "info"}


def _load_prompt(path: str, default: str) -> str:
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return default


def _extract_json(text: str) -> str:
    """Extract the first JSON object from LLM output (strips markdown fences etc.)."""
    # Strip markdown code fences
    text = re.sub(r"```(?:json)?\s*", "", text).strip()
    # Find first `{` … last `}`
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text


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
        and call LLM to produce a structured compliance report.
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
        prompt = (
            self._validate_prompt
            .replace("{document_context}", doc_context)
            .replace("{normative_context}", norm_context)
        )
        llm_response = await self._llm.complete(prompt)

        # 5. Parse structured issues from JSON output (with keyword fallback)
        issues = self._parse_issues(llm_response, norm_hits)
        summary = self._extract_summary(llm_response)

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
            summary=summary,
            checked_articles=checked_articles,
        )

    def _parse_issues(
        self,
        llm_response: str,
        norm_hits: list[dict],  # type: ignore[type-arg]
    ) -> list[ValidationIssue]:
        """Parse issues from structured JSON output. Falls back to keyword heuristic."""
        if not llm_response.strip():
            return []

        # --- Try JSON path ---
        try:
            data = json.loads(_extract_json(llm_response))
            raw_issues = data.get("issues", [])
            if isinstance(raw_issues, list):
                result: list[ValidationIssue] = []
                for item in raw_issues:
                    if not isinstance(item, dict):
                        continue
                    severity = str(item.get("severity", "info")).lower()
                    if severity not in _VALID_SEVERITIES:
                        severity = "info"
                    article = item.get("article") or None
                    violation = str(item.get("violation", "")).strip()
                    recommendation = str(item.get("recommendation", "")).strip()
                    description = violation
                    if recommendation:
                        description = f"{violation}\nРекомендация: {recommendation}"

                    citation = self._find_citation(norm_hits, article)
                    result.append(
                        ValidationIssue(
                            severity=severity,
                            description=description,
                            law_article=article,
                            citation=citation,
                        )
                    )
                return result
        except (json.JSONDecodeError, KeyError, TypeError):
            logger.debug("JSON parse failed for validation response, using heuristic fallback")

        # --- Keyword heuristic fallback ---
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

    def _extract_summary(self, llm_response: str) -> str:
        """Extract summary field from JSON response, or use first 500 chars."""
        try:
            data = json.loads(_extract_json(llm_response))
            summary = data.get("summary", "")
            if summary:
                return str(summary)
        except (json.JSONDecodeError, KeyError, TypeError):
            pass
        return llm_response[:500]

    def _find_citation(
        self,
        norm_hits: list[dict],  # type: ignore[type-arg]
        article: str | None,
    ) -> Citation | None:
        """Find the most relevant norm_hit for the given article label."""
        if not norm_hits:
            return None
        # Try to match by article label first
        if article:
            for hit in norm_hits:
                hit_article = str(hit["payload"].get("law_article", ""))
                if article.lower() in hit_article.lower() or hit_article.lower() in article.lower():
                    return Citation(
                        chunk_id=str(hit["id"]),
                        text=str(hit["payload"].get("text", ""))[:300],
                        score=round(float(hit["score"]), 4),
                        law_article=hit_article or None,
                    )
        # Default to top hit
        p = norm_hits[0]["payload"]
        return Citation(
            chunk_id=str(norm_hits[0]["id"]),
            text=str(p.get("text", ""))[:300],
            score=round(float(norm_hits[0]["score"]), 4),
            law_article=p.get("law_article"),
        )
