"""Grid search over chunk_size / chunk_overlap / top_k.

This rebuilds a throwaway vector index for each candidate configuration
from a source PDF, scores it against `evaluation/labeled_queries.json`
using the same precision@k / recall@k / MRR metrics as the evaluation
suite, and writes every result to a CSV table so configurations can be
compared side by side instead of hardcoded by guesswork.

Requires OPENAI_API_KEY to be set (each candidate re-embeds the document)
and a populated labeled query set to score against.

Usage:
    python -m tuning.hyperparameter_search data/your_document.pdf
"""
import itertools
import os
import shutil
import sys
import time

import pandas as pd
from langchain_core.documents import Document

from backend.retriever import DocumentRetriever
from backend.utils import extract_text_from_pdf
from evaluation.evaluate_retrieval import doc_id, load_labeled_queries
from evaluation.metrics import precision_at_k, recall_at_k, reciprocal_rank
from preprocessing.pipeline import PreprocessingPipeline

PARAM_GRID = {
    "chunk_size": [300, 400, 600],
    "chunk_overlap": [30, 50, 100],
    "top_k": [3, 5],
}


def _build_index(pdf_path: str, chunk_size: int, chunk_overlap: int, vectorstore_path: str) -> DocumentRetriever:
    text = extract_text_from_pdf(pdf_path)
    records = PreprocessingPipeline(chunk_size=chunk_size, chunk_overlap=chunk_overlap).run(text, source=pdf_path)
    documents = [Document(page_content=r["text"], metadata=r["metadata"]) for r in records]

    retriever = DocumentRetriever()
    retriever.vectorstore_path = vectorstore_path
    retriever.add_documents(documents)
    return retriever


def _score(retriever: DocumentRetriever, labeled_queries: list, k: int) -> dict:
    precisions, recalls, rr_scores = [], [], []
    for item in labeled_queries:
        docs = retriever.retrieve(item["question"], k=k)
        retrieved_ids = [doc_id(d) for d in docs]
        relevant_ids = set(item["relevant_ids"])
        precisions.append(precision_at_k(retrieved_ids, relevant_ids, k))
        recalls.append(recall_at_k(retrieved_ids, relevant_ids, k))
        rr_scores.append(reciprocal_rank(retrieved_ids, relevant_ids))

    n = len(labeled_queries)
    return {
        "precision_at_k": sum(precisions) / n if n else 0.0,
        "recall_at_k": sum(recalls) / n if n else 0.0,
        "mrr": sum(rr_scores) / n if n else 0.0,
    }


def run_grid_search(pdf_path: str, labeled_queries_path: str = "evaluation/labeled_queries.json",
                     output_csv: str = "tuning/tuning_results.csv") -> pd.DataFrame:
    labeled_queries = load_labeled_queries(labeled_queries_path)
    rows = []

    for chunk_size, chunk_overlap, top_k in itertools.product(
        PARAM_GRID["chunk_size"], PARAM_GRID["chunk_overlap"], PARAM_GRID["top_k"]
    ):
        if chunk_overlap >= chunk_size:
            continue

        vectorstore_path = f"vectorstore/_tuning_{chunk_size}_{chunk_overlap}"
        start = time.time()
        try:
            retriever = _build_index(pdf_path, chunk_size, chunk_overlap, vectorstore_path)
            metrics = _score(retriever, labeled_queries, top_k)
            build_seconds = time.time() - start
            rows.append({
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "top_k": top_k,
                "build_seconds": round(build_seconds, 3),
                **metrics,
            })
        finally:
            shutil.rmtree(vectorstore_path, ignore_errors=True)

    results = pd.DataFrame(rows)
    if not results.empty:
        results = results.sort_values("precision_at_k", ascending=False)

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    results.to_csv(output_csv, index=False)
    return results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m tuning.hyperparameter_search <path_to_pdf>")
        sys.exit(1)

    results_df = run_grid_search(sys.argv[1])
    print(results_df.to_string(index=False))
