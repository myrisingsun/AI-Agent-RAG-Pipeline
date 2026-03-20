"""Unit tests for ValidationService._parse_issues and _extract_summary."""

import pytest

from src.rag.config import RAGConfig
from src.rag.pipeline.validation import ValidationService, _extract_json


# ─── _extract_json ────────────────────────────────────────────────────────────

@pytest.mark.unit
def test_extract_json_plain() -> None:
    raw = '{"issues": [], "summary": "OK"}'
    assert _extract_json(raw) == raw


@pytest.mark.unit
def test_extract_json_strips_markdown_fence() -> None:
    raw = "```json\n{\"issues\": []}\n```"
    result = _extract_json(raw)
    assert result.startswith("{")
    assert result.endswith("}")


@pytest.mark.unit
def test_extract_json_extracts_embedded() -> None:
    raw = 'Some preamble text. {"issues": [{"severity": "info"}]} trailing text.'
    result = _extract_json(raw)
    assert result.startswith("{")
    assert '"issues"' in result


# ─── _parse_issues (JSON path) ────────────────────────────────────────────────

def _make_service() -> ValidationService:
    """Return ValidationService with all deps stubbed out."""
    config = RAGConfig.model_construct()
    svc = ValidationService.__new__(ValidationService)
    svc._config = config
    svc._validate_prompt = ""
    return svc  # type: ignore[return-value]


@pytest.mark.unit
def test_parse_issues_empty_json_returns_empty() -> None:
    svc = _make_service()
    issues = svc._parse_issues('{"issues": [], "summary": "OK"}', [])
    assert issues == []


@pytest.mark.unit
def test_parse_issues_single_critical() -> None:
    svc = _make_service()
    raw = '{"issues": [{"severity": "critical", "article": "Ст. 334", "violation": "Нет залога", "recommendation": "Добавить"}], "summary": "Нарушение"}'
    issues = svc._parse_issues(raw, [])
    assert len(issues) == 1
    assert issues[0].severity == "critical"
    assert issues[0].law_article == "Ст. 334"
    assert "Нет залога" in issues[0].description
    assert "Добавить" in issues[0].description


@pytest.mark.unit
def test_parse_issues_multiple_severities() -> None:
    svc = _make_service()
    raw = """{
        "issues": [
            {"severity": "critical", "article": "п. 1", "violation": "V1", "recommendation": "R1"},
            {"severity": "warning", "article": "п. 2", "violation": "V2", "recommendation": "R2"},
            {"severity": "info", "article": null, "violation": "V3", "recommendation": ""}
        ],
        "summary": "Найдено 2 нарушения"
    }"""
    issues = svc._parse_issues(raw, [])
    assert len(issues) == 3
    assert issues[0].severity == "critical"
    assert issues[1].severity == "warning"
    assert issues[2].severity == "info"
    assert issues[2].law_article is None


@pytest.mark.unit
def test_parse_issues_invalid_severity_defaults_to_info() -> None:
    svc = _make_service()
    raw = '{"issues": [{"severity": "unknown_value", "violation": "X", "recommendation": ""}]}'
    issues = svc._parse_issues(raw, [])
    assert len(issues) == 1
    assert issues[0].severity == "info"


@pytest.mark.unit
def test_parse_issues_fallback_to_keyword_heuristic() -> None:
    """Non-JSON response triggers keyword fallback."""
    svc = _make_service()
    raw = "Выявлено нарушение требований ЦБ. Документ не соответствует нормативам."
    issues = svc._parse_issues(raw, [])
    assert len(issues) == 1
    assert issues[0].severity == "critical"


@pytest.mark.unit
def test_parse_issues_fallback_warning() -> None:
    svc = _make_service()
    raw = "Рекомендуется добавить пункт о страховании залога."
    issues = svc._parse_issues(raw, [])
    assert issues[0].severity == "warning"


@pytest.mark.unit
def test_parse_issues_empty_response_returns_empty() -> None:
    svc = _make_service()
    assert svc._parse_issues("", []) == []


# ─── _extract_summary ────────────────────────────────────────────────────────

@pytest.mark.unit
def test_extract_summary_from_json() -> None:
    svc = _make_service()
    raw = '{"issues": [], "summary": "Документ соответствует требованиям."}'
    assert svc._extract_summary(raw) == "Документ соответствует требованиям."


@pytest.mark.unit
def test_extract_summary_fallback_to_first_500_chars() -> None:
    svc = _make_service()
    raw = "Это обычный текст без JSON."
    result = svc._extract_summary(raw)
    assert result == raw


# ─── _find_citation ──────────────────────────────────────────────────────────

@pytest.mark.unit
def test_find_citation_matches_article_label() -> None:
    svc = _make_service()
    norm_hits = [
        {"id": "aaa", "score": 0.9, "payload": {"law_article": "Ст. 334", "text": "Текст статьи"}},
        {"id": "bbb", "score": 0.8, "payload": {"law_article": "Ст. 337", "text": "Другая статья"}},
    ]
    citation = svc._find_citation(norm_hits, "Ст. 334")
    assert citation is not None
    assert citation.chunk_id == "aaa"


@pytest.mark.unit
def test_find_citation_falls_back_to_top_hit() -> None:
    svc = _make_service()
    norm_hits = [
        {"id": "aaa", "score": 0.9, "payload": {"law_article": "Ст. 334", "text": "Текст"}},
    ]
    citation = svc._find_citation(norm_hits, "Ст. 999")
    assert citation is not None
    assert citation.chunk_id == "aaa"


@pytest.mark.unit
def test_find_citation_returns_none_on_empty_hits() -> None:
    svc = _make_service()
    assert svc._find_citation([], "Ст. 334") is None
