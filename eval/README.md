# End-to-end evaluation harness

This is the query-level evaluation toolkit: a fixed test set plus scripts
that measure retrieval quality, config sensitivity and answer
faithfulness against it. It builds on the ranking metrics in
`evaluation/metrics.py` (the same precision@k / recall@k / MRR functions
used by `tuning/hyperparameter_search.py`) rather than re-implementing
them, so a number computed here means the same thing everywhere else in
the project.

## Methodology

**Building the test set.** `test_queries.json` holds 41 template queries
spanning three query types (`factual`, `summarization`, `comparison` —
the same labels `models/intent_classifier.py` predicts). Each entry has:

- `query` — the question to ask
- `query_type` — used to break results down by query type
- `expected_answer` — a human-written reference answer
- `relevant_chunk_ids` — the chunk ids that actually answer it, in
  `<filename>#<chunk_index>` form (0-indexed, in the order the
  preprocessing pipeline produced them — check `logs/app.log` after
  uploading a document)

The shipped file is a template: fill in `expected_answer` and
`relevant_chunk_ids` from your own uploaded document(s) before the scripts
below produce meaningful numbers. Running them against the template as-is
will report near-zero scores, since `your_document.pdf#0` won't match
anything you've actually indexed.

**Computing the metrics.**

- *Retrieval quality* (`evaluate_retrieval.py`): for each query, run
  retrieval and compare the returned chunk ids against
  `relevant_chunk_ids`, computing precision@k, recall@k and MRR.
- *Config sensitivity* (`compare_configs.py`): rebuild the index under a
  handful of chunk_size/top_k configurations and re-run the same scoring,
  so a config choice is backed by a number instead of a guess.
- *Faithfulness* (`check_faithfulness.py`): for each query, generate an
  answer through the real RAG pipeline and score its lexical overlap
  (ROUGE-L-style longest-common-subsequence F1) against the chunks that
  were actually retrieved. An answer below `--threshold` (default 0.2) is
  flagged as ungrounded — a cheap proxy for hallucination, not a
  substitute for human review.

## Running

```bash
# Retrieval metrics -> eval/retrieval_results.csv
python -m eval.evaluate_retrieval

# Compare 3 chunk_size/top_k configs against the same test set -> eval/config_comparison.csv
python -m eval.compare_configs data/your_document.pdf

# Faithfulness of generated answers -> eval/faithfulness_results.csv (requires OPENAI_API_KEY)
python -m eval.check_faithfulness
```

All three require a document to already be indexed (`POST /upload` or
`tuning`/`evaluation` scripts populate the same `vectorstore/`).
