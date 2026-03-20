"""BM25 sparse vector computation for hybrid search."""

from typing import Any


def compute_sparse_vector(text: str, tokenizer: Any) -> dict[int, float]:
    """
    Compute a sparse TF vector from token IDs for BM25 hybrid search.

    Token IDs from the embedding tokenizer are used directly as sparse indices.
    Qdrant applies IDF scoring automatically via `Modifier.IDF` in collection config.

    Returns: {token_id: normalized_term_frequency}
    Returns empty dict if text produces no tokens.
    """
    token_ids: list[int] = tokenizer.encode(text, add_special_tokens=False)
    if not token_ids:
        return {}

    freq: dict[int, int] = {}
    for tid in token_ids:
        freq[tid] = freq.get(tid, 0) + 1

    total = len(token_ids)
    return {tid: count / total for tid, count in freq.items()}
