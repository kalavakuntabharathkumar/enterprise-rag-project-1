# End-to-end evaluation harness

This is the query-level evaluation toolkit: a fixed test set plus scripts
that measure retrieval quality, config sensitivity and answer faithfulness
against it.  It builds on the ranking metrics in `evaluation/metrics.py`
(the same precision@k / recall@k / MRR functions used by
`tuning/hyperparameter_search.py`) rather than re-implementing them, so a
number computed here means the same thing everywhere else in the project.

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
  were actually retrieved.  An answer below `--threshold` (default 0.2) is
  flagged as ungrounded — a cheap proxy for hallucination, not a substitute
  for human review.

## Running

```bash
# Retrieval metrics -> eval/retrieval_results.csv
python -m eval.evaluate_retrieval

# Compare 3 chunk_size/top_k configs against the same test set -> eval/config_comparison.csv
python -m eval.compare_configs data/your_document.pdf

# Faithfulness of generated answers -> eval/faithfulness_results.csv
python -m eval.check_faithfulness
```

All three require a document to already be indexed (`POST /upload` or
`tuning`/`evaluation` scripts populate the same `vectorstore/`).

## Compatibility with the self-hosted inference backend

All three scripts work without any changes against the self-hosted
generation backend (vLLM or llama.cpp).

**Why no changes were needed.**  The evaluation suite is split cleanly into
a retrieval layer and a generation layer:

- `evaluate_retrieval.py` and `compare_configs.py` only exercise
  retrieval — FAISS similarity search, chunking and index construction.
  They never call the LLM at all, so the generation backend is irrelevant.

- `check_faithfulness.py` calls `RAGPipeline.ask_question()`, which now
  routes generation through `backend/inference_client.py`.  The faithfulness
  score is computed from the returned answer string using a local
  ROUGE-L-style lexical overlap; it has no dependency on *how* the answer
  was generated.

**Running the faithfulness check against the self-hosted server.**  Make
sure the model server is running and `LLM_BACKEND` / `LLM_BASE_URL` are
set correctly (see `.env.example`), then run as normal:

```bash
LLM_BACKEND=vllm LLM_BASE_URL=http://localhost:8000 python -m eval.check_faithfulness
```

The output CSV (`eval/faithfulness_results.csv`) has the same schema as
before — `id`, `query`, `answer`, `faithfulness_score`, `grounded` — so
existing tooling and result comparisons remain valid.

**Comparing results before and after the backend switch.**  The retrieval
metrics (precision@k, recall@k, MRR) are unchanged by definition — the
FAISS index, embeddings and re-ranker were not modified.  For faithfulness,
any shift in scores reflects differences in generation quality between the
cloud API and the self-hosted model, which is expected and worth tracking.
Run the faithfulness check on the same `test_queries.json` set with both
backends and compare the `faithfulness_score` columns to quantify the
difference.
