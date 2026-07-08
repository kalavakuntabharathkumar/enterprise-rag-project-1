# Hyperparameter tuning

`hyperparameter_search.py` grid-searches chunk size, chunk overlap and
top-k, rebuilding a throwaway FAISS index per configuration and scoring it
against `evaluation/labeled_queries.json` (precision@k, recall@k, MRR).

## Running

```bash
python -m tuning.hyperparameter_search data/your_document.pdf
```

Results are written to `tuning/tuning_results.csv`, sorted best-first by
precision@k, so a configuration change is a data-backed decision rather
than a guess. This needs a populated labeled query set — see
`evaluation/README.md` for how to build one.
