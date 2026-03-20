"""Retrieval quality metrics: MRR@k and Recall@k.

Pure functions — no infrastructure dependencies, fully unit-testable.
"""


def compute_mrr(retrieved_ids: list[str], relevant_ids: set[str], k: int = 10) -> float:
    """
    Mean Reciprocal Rank at k.

    Returns the reciprocal of the rank of the first relevant result in the
    top-k retrieved list. Returns 0.0 if no relevant result found in top-k.

    Args:
        retrieved_ids: Ordered list of retrieved document/chunk IDs.
        relevant_ids:  Set of ground-truth relevant IDs.
        k:             Cutoff rank.
    """
    for rank, doc_id in enumerate(retrieved_ids[:k], start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def compute_recall(retrieved_ids: list[str], relevant_ids: set[str], k: int = 5) -> float:
    """
    Recall at k.

    Fraction of relevant documents that appear in the top-k retrieved list.
    Returns 1.0 if relevant_ids is empty (vacuously true).

    Args:
        retrieved_ids: Ordered list of retrieved document/chunk IDs.
        relevant_ids:  Set of ground-truth relevant IDs.
        k:             Cutoff rank.
    """
    if not relevant_ids:
        return 1.0
    top_k = set(retrieved_ids[:k])
    return len(top_k & relevant_ids) / len(relevant_ids)


def compute_precision(retrieved_ids: list[str], relevant_ids: set[str], k: int = 5) -> float:
    """
    Precision at k.

    Fraction of top-k retrieved documents that are relevant.
    Returns 0.0 if k == 0.
    """
    if k == 0:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return hits / k
