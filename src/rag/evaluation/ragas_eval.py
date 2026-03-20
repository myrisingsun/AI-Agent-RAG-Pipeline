"""
RAGAS + retrieval metrics evaluation harness.

Usage:
    python -m src.rag.evaluation.ragas_eval

Reads:  data/ground_truth/eval_dataset.json
Writes: data/eval_results/{timestamp}.json

Dataset format:
    [
      {
        "query": "...",
        "ground_truth": "...",
        "relevant_chunk_ids": ["uuid1", "uuid2"]   // optional, for MRR/Recall
      },
      ...
    ]
"""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from src.common.logging import configure_logging, get_logger
from src.rag.config import get_config
from src.rag.evaluation.retrieval_metrics import compute_mrr, compute_recall

logger = get_logger(__name__)

DATASET_PATH = Path("data/ground_truth/eval_dataset.json")
RESULTS_DIR = Path("data/eval_results")


def _load_dataset(path: Path) -> list[dict]:  # type: ignore[type-arg]
    if not path.exists():
        raise FileNotFoundError(f"Eval dataset not found: {path}")
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or not data:
        raise ValueError("eval_dataset.json must be a non-empty JSON array")
    return data  # type: ignore[return-value]


async def _run_retrieval(query: str) -> list[str]:
    """Run retrieval pipeline and return ordered chunk IDs."""
    from src.rag.config import get_config
    from src.rag.embeddings.e5_large import E5LargeEmbeddingProvider
    from src.rag.vectorstore.client import QdrantManager
    from src.rag.vectorstore.operations import VectorStoreOperations
    from src.rag.vectorstore.sparse import compute_sparse_vector
    from src.rag.router import QueryRouter

    config = get_config()
    qdrant = QdrantManager(config)
    await qdrant.initialize()

    embedding = E5LargeEmbeddingProvider(config)
    await embedding.initialize()

    vs = VectorStoreOperations(qdrant, config)
    router = QueryRouter()

    collection = router.route(query)
    query_vector = await embedding.embed_query(query)

    sparse: dict[int, float] | None = None
    if config.hybrid_search_enabled:
        tokenizer = getattr(embedding, "_tokenizer", None)
        if tokenizer is not None:
            sparse = compute_sparse_vector(query, tokenizer)

    hits = await vs.search(
        collection=collection,
        query_vector=query_vector,
        limit=10,
        sparse_vector=sparse,
    )
    await qdrant.close()
    return [str(h["id"]) for h in hits]


async def _evaluate() -> dict:  # type: ignore[type-arg]
    config = get_config()
    configure_logging(config.rag_log_level)

    dataset = _load_dataset(DATASET_PATH)
    logger.info("eval started", queries=len(dataset))

    mrr_scores: list[float] = []
    recall_scores: list[float] = []
    rows: list[dict] = []  # type: ignore[type-arg]

    for item in dataset:
        query: str = item["query"]
        relevant_ids: set[str] = set(item.get("relevant_chunk_ids", []))

        retrieved_ids = await _run_retrieval(query)

        mrr = compute_mrr(retrieved_ids, relevant_ids, k=10) if relevant_ids else None
        recall = compute_recall(retrieved_ids, relevant_ids, k=5) if relevant_ids else None

        if mrr is not None:
            mrr_scores.append(mrr)
        if recall is not None:
            recall_scores.append(recall)

        rows.append({
            "query": query,
            "retrieved_ids": retrieved_ids[:5],
            "mrr": mrr,
            "recall_at_5": recall,
        })
        logger.info("query evaluated", query=query[:60], mrr=mrr, recall=recall)

    # Attempt RAGAS metrics if library available
    ragas_metrics: dict = {}  # type: ignore[type-arg]
    try:
        from ragas import EvaluationDataset, evaluate  # type: ignore[import]
        from ragas.metrics import AnswerRelevancy, ContextPrecision, Faithfulness  # type: ignore[import]

        ragas_rows = [
            {
                "user_input": item["query"],
                "response": item.get("ground_truth", ""),
                "retrieved_contexts": [],
                "reference": item.get("ground_truth", ""),
            }
            for item in dataset
        ]
        eval_dataset = EvaluationDataset.from_list(ragas_rows)
        result = evaluate(eval_dataset, metrics=[Faithfulness(), ContextPrecision(), AnswerRelevancy()])
        ragas_metrics = result.to_pandas().mean(numeric_only=True).to_dict()
        logger.info("ragas metrics", **ragas_metrics)
    except ImportError:
        logger.info("ragas not installed, skipping RAGAS metrics (pip install '.[eval]')")
    except Exception as exc:
        logger.warning("ragas evaluation failed", error=str(exc))

    results = {
        "evaluated_at": datetime.now(UTC).isoformat(),
        "queries": len(dataset),
        "mrr_at_10": round(sum(mrr_scores) / len(mrr_scores), 4) if mrr_scores else None,
        "recall_at_5": round(sum(recall_scores) / len(recall_scores), 4) if recall_scores else None,
        "ragas": ragas_metrics,
        "rows": rows,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"eval_{ts}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info(
        "eval complete",
        mrr_at_10=results["mrr_at_10"],
        recall_at_5=results["recall_at_5"],
        output=str(out_path),
    )
    return results


if __name__ == "__main__":
    results = asyncio.run(_evaluate())
    print(json.dumps(
        {k: v for k, v in results.items() if k != "rows"},
        ensure_ascii=False,
        indent=2,
    ))
    sys.exit(0)
