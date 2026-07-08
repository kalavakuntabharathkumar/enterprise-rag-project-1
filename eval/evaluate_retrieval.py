"""Retrieval evaluation against `eval/test_queries.json`.

Same metrics as `evaluation/evaluate_retrieval.py`, but scored against the
larger, query-type-labeled test set used by the rest of the `eval/`
harness, and broken down by query type in addition to the overall
aggregate. Results are written to `eval/retrieval_results.csv` with one
row per query plus a final aggregate row.

Usage:
    python -m eval.evaluate_retrieval [test_queries.json] [k]
"""
import json
import os
import sys

import pandas as pd

from backend.retriever import DocumentRetriever
from evaluation.metrics import precision_at_k, recall_at_k, reciprocal_rank


def load_test_queries(path: str) -> list:
    with open(path) as f:
        data = json.load(f)
    return [item for item in data if "_comment" not in item]


def doc_id(document) -> str:
    meta = document.metadata
    return f"{os.path.basename(str(meta.get('source', 'unknown')))}#{meta.get('chunk_index', '?')}"


def run_evaluation(test_queries_path: str = "eval/test_queries.json", k: int = 5,
                    output_csv: str = "eval/retrieval_results.csv") -> pd.DataFrame:
    dataset = load_test_queries(test_queries_path)
    retriever = DocumentRetriever()
    retriever.load_vectorstore()

    rows = []
    for item in dataset:
        docs = retriever.retrieve(item["query"], k=k)
        retrieved_ids = [doc_id(doc) for doc in docs]
        relevant_ids = set(item["relevant_chunk_ids"])

        rows.append({
            "id": item.get("id"),
            "query": item["query"],
            "query_type": item.get("query_type", "unknown"),
            "precision_at_k": precision_at_k(retrieved_ids, relevant_ids, k),
            "recall_at_k": recall_at_k(retrieved_ids, relevant_ids, k),
            "reciprocal_rank": reciprocal_rank(retrieved_ids, relevant_ids),
        })

    results = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    results.to_csv(output_csv, index=False)
    return results


def summarize(results: pd.DataFrame) -> dict:
    if results.empty:
        return {"num_queries": 0, "precision_at_k": 0.0, "recall_at_k": 0.0, "mrr": 0.0}
    return {
        "num_queries": len(results),
        "precision_at_k": float(results["precision_at_k"].mean()),
        "recall_at_k": float(results["recall_at_k"].mean()),
        "mrr": float(results["reciprocal_rank"].mean()),
        "by_query_type": {
            qtype: {
                "precision_at_k": float(group["precision_at_k"].mean()),
                "recall_at_k": float(group["recall_at_k"].mean()),
                "mrr": float(group["reciprocal_rank"].mean()),
            }
            for qtype, group in results.groupby("query_type")
        },
    }


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "eval/test_queries.json"
    k_value = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    df = run_evaluation(path, k_value)
    print(json.dumps(summarize(df), indent=2))
