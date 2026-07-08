"""Compare retrieval quality and latency across a handful of named
chunk_size/top_k configurations, scored against `eval/test_queries.json`.

This is a smaller, opinionated sibling of `tuning/hyperparameter_search.py`
(which does an exhaustive grid search): three specific configurations are
built and compared side by side so a default can be picked and justified,
rather than searching the whole grid every time.

Requires OPENAI_API_KEY (each config re-embeds the document).

Usage:
    python -m eval.compare_configs data/your_document.pdf
"""
import os
import shutil
import sys
import time

import pandas as pd
from langchain_core.documents import Document

from backend.retriever import DocumentRetriever
from backend.utils import extract_text_from_pdf
from eval.evaluate_retrieval import doc_id, load_test_queries
from evaluation.metrics import precision_at_k, recall_at_k
from preprocessing.pipeline import PreprocessingPipeline

CONFIGS = [
    {"name": "small_chunks", "chunk_size": 256, "chunk_overlap": 30, "top_k": 3},
    {"name": "medium_chunks", "chunk_size": 512, "chunk_overlap": 50, "top_k": 5},
    {"name": "large_chunks", "chunk_size": 1024, "chunk_overlap": 100, "top_k": 10},
]


def _build_index(pdf_path: str, chunk_size: int, chunk_overlap: int, vectorstore_path: str) -> DocumentRetriever:
    text = extract_text_from_pdf(pdf_path)
    records = PreprocessingPipeline(chunk_size=chunk_size, chunk_overlap=chunk_overlap).run(text, source=pdf_path)
    documents = [Document(page_content=r["text"], metadata=r["metadata"]) for r in records]

    retriever = DocumentRetriever()
    retriever.vectorstore_path = vectorstore_path
    retriever.add_documents(documents)
    return retriever


def run_comparison(pdf_path: str, test_queries_path: str = "eval/test_queries.json",
                    output_csv: str = "eval/config_comparison.csv") -> pd.DataFrame:
    dataset = load_test_queries(test_queries_path)
    rows = []

    for cfg in CONFIGS:
        vectorstore_path = f"vectorstore/_compare_{cfg['name']}"
        try:
            retriever = _build_index(pdf_path, cfg["chunk_size"], cfg["chunk_overlap"], vectorstore_path)

            precisions, recalls, latencies_ms = [], [], []
            for item in dataset:
                start = time.time()
                docs = retriever.retrieve(item["query"], k=cfg["top_k"])
                latencies_ms.append((time.time() - start) * 1000)

                retrieved_ids = [doc_id(d) for d in docs]
                relevant_ids = set(item["relevant_chunk_ids"])
                precisions.append(precision_at_k(retrieved_ids, relevant_ids, cfg["top_k"]))
                recalls.append(recall_at_k(retrieved_ids, relevant_ids, cfg["top_k"]))

            n = len(dataset) or 1
            rows.append({
                "config": cfg["name"],
                "chunk_size": cfg["chunk_size"],
                "chunk_overlap": cfg["chunk_overlap"],
                "top_k": cfg["top_k"],
                "precision_at_5": sum(precisions) / n,
                "recall_at_5": sum(recalls) / n,
                "avg_latency_ms": sum(latencies_ms) / n,
            })
        finally:
            shutil.rmtree(vectorstore_path, ignore_errors=True)

    results = pd.DataFrame(rows)
    if not results.empty:
        results = results.sort_values("precision_at_5", ascending=False)

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    results.to_csv(output_csv, index=False)
    return results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m eval.compare_configs <path_to_pdf>")
        sys.exit(1)

    df = run_comparison(sys.argv[1])
    print(df.to_string(index=False))
    if not df.empty:
        best = df.iloc[0]
        print(f"\nBest by precision@5: {best['config']} "
              f"(chunk_size={best['chunk_size']}, top_k={best['top_k']})")
