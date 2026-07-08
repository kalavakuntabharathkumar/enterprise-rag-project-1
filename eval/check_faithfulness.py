"""Faithfulness / groundedness check for generated answers.

For each query in `eval/test_queries.json`, runs the real RAG pipeline and
scores the generated answer's lexical overlap against the chunks that were
actually retrieved, using a ROUGE-L-style longest-common-subsequence F1.
This is a cheap proxy for groundedness, not a substitute for a human or
LLM-judge review: it catches answers with no textual relationship to the
retrieved context, which is usually (though not always) a sign the model
answered from outside knowledge instead of the document.

Usage:
    python -m eval.check_faithfulness [threshold]
"""
import json
import os
import sys

import pandas as pd

from backend.rag_pipeline import RAGPipeline
from eval.evaluate_retrieval import load_test_queries

DEFAULT_THRESHOLD = 0.2


def _lcs_length(a: list, b: list) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        curr = [0] * (len(b) + 1)
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    return prev[len(b)]


def rouge_l_f1(candidate: str, reference: str) -> float:
    """Longest-common-subsequence-based F1, the core of ROUGE-L, computed
    without any external dependency."""
    cand_tokens = candidate.lower().split()
    ref_tokens = reference.lower().split()
    if not cand_tokens or not ref_tokens:
        return 0.0

    lcs = _lcs_length(cand_tokens, ref_tokens)
    if lcs == 0:
        return 0.0

    precision = lcs / len(cand_tokens)
    recall = lcs / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def run_faithfulness_check(test_queries_path: str = "eval/test_queries.json",
                            threshold: float = DEFAULT_THRESHOLD,
                            output_csv: str = "eval/faithfulness_results.csv") -> pd.DataFrame:
    dataset = load_test_queries(test_queries_path)
    pipeline = RAGPipeline()

    rows = []
    for item in dataset:
        result = pipeline.ask_question(item["query"])
        answer = result["answer"]

        candidates = pipeline.retriever.retrieve(item["query"], k=5)
        context = "\n".join(doc.page_content for doc in candidates)

        score = rouge_l_f1(answer, context) if context else 0.0
        rows.append({
            "id": item.get("id"),
            "query": item["query"],
            "answer": answer,
            "faithfulness_score": score,
            "grounded": score >= threshold,
        })

    results = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    results.to_csv(output_csv, index=False)
    return results


if __name__ == "__main__":
    threshold_value = float(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_THRESHOLD
    df = run_faithfulness_check(threshold=threshold_value)
    grounded_pct = 100.0 * df["grounded"].mean() if not df.empty else 0.0
    print(json.dumps({
        "num_answers": len(df),
        "grounded_pct": grounded_pct,
        "threshold": threshold_value,
    }, indent=2))
