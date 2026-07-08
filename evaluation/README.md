# Retrieval evaluation

`evaluate_retrieval.py` computes precision@k, recall@k and MRR against a
labeled query set, so retrieval quality is measured instead of eyeballed.

## Building your own labeled set

1. Upload a document through the app (or call `POST /upload` directly).
2. Check `logs/app.log` for the preprocessing line — it logs how many
   chunks were kept for that source.
3. For a handful of representative questions, work out which chunks
   actually answer them. Chunk ids follow `<filename>#<chunk_index>`,
   0-indexed in the order they were produced by the preprocessing pipeline.
4. Add `{ "question": "...", "relevant_ids": ["file.pdf#2", "file.pdf#5"] }`
   entries to `labeled_queries.json`.

## Running

```bash
python -m evaluation.evaluate_retrieval evaluation/labeled_queries.json 3
```

This is also what `tuning/hyperparameter_search.py` uses as its scoring
function when comparing chunk size / overlap / top-k configurations.
