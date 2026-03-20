"""Unit tests for QueryRouter."""

import pytest

from src.rag.config import CollectionName
from src.rag.router import QueryRouter


@pytest.fixture
def router() -> QueryRouter:
    return QueryRouter()


@pytest.mark.unit
def test_session_id_always_routes_to_current_package(router: QueryRouter) -> None:
    result = router.route("закон о залоге", session_id="sess-123")
    assert result == CollectionName.CURRENT_PACKAGE


@pytest.mark.unit
def test_normative_keyword_routes_to_normative_base(router: QueryRouter) -> None:
    assert router.route("статья 254-П положение ЦБ") == CollectionName.NORMATIVE_BASE


@pytest.mark.unit
def test_normative_keyword_case_insensitive(router: QueryRouter) -> None:
    assert router.route("ЗАКОН о банках") == CollectionName.NORMATIVE_BASE


@pytest.mark.unit
def test_precedent_keyword_routes_to_deal_precedents(router: QueryRouter) -> None:
    assert router.route("похожая сделка с залогом") == CollectionName.DEAL_PRECEDENTS


@pytest.mark.unit
def test_template_keyword_routes_to_reference_templates(router: QueryRouter) -> None:
    assert router.route("типовой шаблон договора") == CollectionName.REFERENCE_TEMPLATES


@pytest.mark.unit
def test_unknown_query_routes_to_current_package(router: QueryRouter) -> None:
    assert router.route("какова сумма кредита?") == CollectionName.CURRENT_PACKAGE


@pytest.mark.unit
def test_normative_takes_priority_over_default(router: QueryRouter) -> None:
    # session_id overrides everything
    result = router.route("статья 254-П", session_id="s1")
    assert result == CollectionName.CURRENT_PACKAGE


@pytest.mark.unit
def test_empty_query_routes_to_current_package(router: QueryRouter) -> None:
    assert router.route("") == CollectionName.CURRENT_PACKAGE
