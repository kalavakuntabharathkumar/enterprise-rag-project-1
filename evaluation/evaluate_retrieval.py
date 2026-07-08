"""Retrieval evaluation against a labeled query set.

Each entry in `labeled_queries.json` pairs a question with the chunk ids
that are actually relevant to it, so we can compute real precision@k,
recall@k and MRR instead of eyeballing answers. Chunk ids follow the
`<source_filename>#<chunk_index>` convention produced by the preprocessing
pipeline, which you can read off `logs/app.log` after uploading a document.

Usage:
    python -m evaluation.evaluate_retrieval [labeled_queries.json] [k]
"""
import json
import os
import sys

import numpy as np

from backend.retriever import DocumentRetriever
from evaluation.metrics import precision_at_k, recall_at_k, reciprocal_rank


def load_labeled_queries(path: str) -> list:
    with open(path) as f:
        data = json.load(f)
    return [item for item in data if "_comment" not in item]


def doc_id(document) -> str:
    meta = document.metadata
    return f"{os.path.basename(str(meta.get('source', 'unknown')))}#{meta.get('chunk_index', '?')}"


def run_evaluation(labeled_path: str = "evaluation/labeled_queries.json", k: int = 3) -> dict:
    dataset = load_labeled_queries(labeled_path)
    if not dataset:
        return {"num_queries": 0, "precision_at_k": 0.0, "recall_at_k": 0.0, "mrr": 0.0, "k": k}

    retriever = DocumentRetriever()
    retriever.load_vectorstore()

    precisions, recalls, rr_scores = [], [], []
    for item in dataset:
        docs = retriever.retrieve(item["question"], k=k)
        retrieved_ids = [doc_id(doc) for doc in docs]
        relevant_ids = set(item["relevant_ids"])

        precisions.append(precision_at_k(retrieved_ids, relevant_ids, k))
        recalls.append(recall_at_k(retrieved_ids, relevant_ids, k))
        rr_scores.append(reciprocal_rank(retrieved_ids, relevant_ids))

    return {
        "num_queries": len(dataset),
        "precision_at_k": float(np.mean(precisions)),
        "recall_at_k": float(np.mean(recalls)),
        "mrr": float(np.mean(rr_scores)),
        "k": k,
    }


if __name__ == "__main__":
    labeled_path = sys.argv[1] if len(sys.argv) > 1 else "evaluation/labeled_queries.json"
    k_value = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    print(json.dumps(run_evaluation(labeled_path, k_value), indent=2))
