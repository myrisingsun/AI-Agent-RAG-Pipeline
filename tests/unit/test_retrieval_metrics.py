"""Unit tests for retrieval metrics: MRR@k, Recall@k, Precision@k."""

import pytest

from src.rag.evaluation.retrieval_metrics import compute_mrr, compute_precision, compute_recall


@pytest.mark.unit
def test_mrr_first_result_relevant() -> None:
    assert compute_mrr(["a", "b", "c"], {"a"}, k=10) == 1.0


@pytest.mark.unit
def test_mrr_second_result_relevant() -> None:
    assert compute_mrr(["x", "a", "b"], {"a"}, k=10) == pytest.approx(0.5)


@pytest.mark.unit
def test_mrr_third_result_relevant() -> None:
    assert compute_mrr(["x", "y", "a"], {"a"}, k=10) == pytest.approx(1 / 3)


@pytest.mark.unit
def test_mrr_not_in_top_k() -> None:
    assert compute_mrr(["x", "y", "z"], {"a"}, k=3) == 0.0


@pytest.mark.unit
def test_mrr_empty_retrieved() -> None:
    assert compute_mrr([], {"a"}, k=10) == 0.0


@pytest.mark.unit
def test_recall_all_found() -> None:
    assert compute_recall(["a", "b", "c", "d", "e"], {"a", "b"}, k=5) == 1.0


@pytest.mark.unit
def test_recall_partial() -> None:
    assert compute_recall(["a", "x", "x", "x", "x"], {"a", "b"}, k=5) == pytest.approx(0.5)


@pytest.mark.unit
def test_recall_none_found() -> None:
    assert compute_recall(["x", "y", "z"], {"a", "b"}, k=5) == 0.0


@pytest.mark.unit
def test_recall_empty_relevant_returns_one() -> None:
    assert compute_recall(["a", "b"], set(), k=5) == 1.0


@pytest.mark.unit
def test_recall_cutoff_respected() -> None:
    # "a" is at position 6, outside k=5 window
    assert compute_recall(["x", "x", "x", "x", "x", "a"], {"a"}, k=5) == 0.0


@pytest.mark.unit
def test_precision_all_relevant() -> None:
    assert compute_precision(["a", "b", "c"], {"a", "b", "c"}, k=3) == 1.0


@pytest.mark.unit
def test_precision_none_relevant() -> None:
    assert compute_precision(["x", "y", "z"], {"a"}, k=3) == 0.0


@pytest.mark.unit
def test_precision_k_zero() -> None:
    assert compute_precision(["a"], {"a"}, k=0) == 0.0
