# Enterprise AI Document Assistant (RAG-based)

A production-oriented AI document assistant that answers questions from
uploaded PDFs using Retrieval-Augmented Generation. It combines a
FAISS retrieval pipeline with a small classical ML layer (query routing,
lexical re-ranking) and the operational scaffolding a real service needs:
preprocessing, analytics, evaluation, and tuning.

Generation is handled by a **self-hosted model server** (vLLM or
llama.cpp). OpenAI embeddings are still used for FAISS indexing and
retrieval; the OpenAI API is not used for inference.

## Architecture

```mermaid
flowchart TD
    subgraph Client
        UI[Streamlit frontend]
    end

    subgraph API[FastAPI backend]
        UP["/upload"]
        ASK["/ask"]
        HEALTH["/health"]
        AN["/analytics/summary"]
    end

    subgraph Ingestion
        EXTRACT[PDF text extraction]
        PRE[Preprocessing pipeline\nclean -> split -> dedup -> quality filter]
        EMB1[OpenAI embeddings]
        STORE[(FAISS vector store)]
    end

    subgraph QueryPath
        INTENT[Intent classifier\nTF-IDF + LogisticRegression]
        RETRIEVE[Vector similarity search]
        RERANK[Lexical re-ranker\nTF-IDF cosine + embedding score]
        LLM[Self-hosted model server\nvLLM or llama.cpp]
        CONF[Confidence estimation]
    end

    subgraph Observability
        LOG[Query analytics log\npandas/numpy]
        EVAL[Evaluation suite\nprecision / recall / MRR]
        TUNE[Hyperparameter search\nchunk size, overlap, top-k]
    end

    UI --> UP --> EXTRACT --> PRE --> EMB1 --> STORE
    UI --> ASK --> INTENT
    INTENT -- document_question --> RETRIEVE --> STORE
    RETRIEVE --> RERANK --> LLM --> CONF --> ASK
    INTENT -- greeting / chit_chat --> ASK
    ASK --> LOG
    LOG --> AN
    EVAL -.-> STORE
    TUNE -.-> EVAL
```

## Features

- PDF upload and text extraction
- Preprocessing pipeline: cleaning, deduplication, metadata extraction and
  chunk-quality scoring — not just a raw text splitter
- OpenAI embeddings + FAISS vector store for semantic retrieval
- Query-intent routing (TF-IDF + Logistic Regression) so greetings/small
  talk never touch the vector store or the LLM
- TF-IDF lexical re-ranking on top of embedding similarity
- Self-hosted generation via vLLM or llama.cpp, switchable with one env var
- Query/retrieval analytics: latency distribution, similarity stats,
  precision@k proxy, and per-request backend tracking
- Retrieval evaluation suite (precision@k, recall@k, MRR) against a
  labeled query set
- Hyperparameter grid search over chunk size, overlap and top-k
- Confidence scoring with a low-confidence fallback
- Auth stub, structured error handling, and a health-check endpoint
- Docker + docker-compose for local/production runs on Linux

## Tech stack

- **Backend**: FastAPI (Python)
- **Frontend**: Streamlit
- **LLM inference**: vLLM or llama.cpp (self-hosted, OpenAI-compatible API)
- **Embeddings**: OpenAI API (`text-embedding-3-small` by default)
- **Retrieval framework**: LangChain
- **Vector store**: FAISS (embedded, no separate server)
- **Classical ML**: scikit-learn (TF-IDF, Logistic Regression, cosine similarity)
- **Analytics**: pandas, numpy

## Project structure

```
enterprise-rag-project/
├── backend/
│   ├── main.py              # FastAPI app: routes, auth stub, error handling
│   ├── schemas.py           # Request/response models
│   ├── rag_pipeline.py      # Orchestrates preprocessing, retrieval, generation
│   ├── retriever.py         # FAISS-backed retrieval, with and without scores
│   ├── inference_client.py  # HTTP client for the self-hosted model server
│   ├── embeddings.py        # OpenAI embeddings helper (used for indexing)
│   ├── utils.py             # PDF extraction, file saving
│   ├── config.py            # Environment-driven configuration
│   └── logger.py            # File-based logging setup
├── preprocessing/           # Cleaning, dedup, metadata, quality scoring
├── analytics/               # Query/retrieval logging and pandas/numpy metrics
├── ml/                      # Greeting/chit-chat intent router and lexical re-ranker
├── models/                  # Query-type classifier (factual/summarization/comparison)
├── evaluation/              # Precision/recall/MRR evaluation suite (single labeled set)
├── eval/                    # End-to-end harness: test set, config comparison, faithfulness
├── tuning/                  # Hyperparameter grid search
├── tests/                   # Smoke tests for retrieval, routing and re-ranking
├── frontend/
│   └── app.py               # Streamlit UI
├── data/                    # Uploaded PDFs (gitignored)
├── vectorstore/             # FAISS index (gitignored)
├── logs/                    # App + analytics logs (gitignored)
├── Dockerfile               # Backend image
├── Dockerfile.frontend      # Frontend image
├── docker-compose.yml       # Backend + frontend + model server, shared volumes
├── requirements.txt
├── .env.example
└── README.md
```

## Why self-hosted inference

**Cost and data ownership.** Routing every generation call through a
third-party API means paying per token and sending document content to an
external service. A self-hosted model keeps inference costs fixed (hardware
only), and document content never leaves the deployment environment. For
internal or sensitive documents this is often a hard requirement.

**vLLM vs llama.cpp.** vLLM is optimised for throughput on CUDA GPUs —
continuous batching and PagedAttention make it the better choice when
multiple requests arrive concurrently. llama.cpp targets CPU-first
deployment with optional GPU offload via GGUF-quantised models; it requires
no GPU and runs on any Linux server. Both expose the same OpenAI-compatible
`/v1/chat/completions` endpoint, so the application code is identical for
either backend.

**Embeddings still use the OpenAI API.** Generating high-quality embeddings
locally requires a dedicated embedding model and additional infrastructure.
Since embeddings are only produced during document ingestion (not at query
time), the latency and cost are much lower than generation, and the tradeoff
of keeping OpenAI for this one step is reasonable for most deployments.
This can be swapped out for a local embedding server following the same
pattern as `inference_client.py` when full air-gap operation is needed.

## Design decisions and tradeoffs

**FAISS over a hosted vector database.** The corpus size this app targets
(documents a single user or team uploads) fits comfortably in memory, so
an embedded, file-backed index avoids running and paying for a separate
database service. The tradeoff is no built-in horizontal scaling or
multi-writer support — if the corpus grows into the millions of chunks or
needs concurrent writers, a managed vector DB (pgvector, Pinecone, Qdrant)
is the better fit.

**Recursive character splitting over semantic/sentence-based chunking.**
`RecursiveCharacterTextSplitter` is cheap, deterministic and works
reasonably well across document types without a model call. The
preprocessing pipeline compensates for its main weakness — chunks that are
too short, near-duplicate, or largely stopwords — with an explicit
dedup + quality-scoring pass rather than relying on the splitter alone.

**A classifier for query routing instead of a fixed keyword blocklist.**
A TF-IDF + Logistic Regression model generalises past the exact phrasing
of a hand-written keyword list, at the cost of needing (a small amount
of) labeled training data. The seed dataset here is intentionally small;
it is meant to catch greetings/small talk cheaply, not to be a
general-purpose intent model.

**Lexical re-ranking blended with embedding similarity, not a learned
re-ranker.** Embedding similarity alone can miss exact keyword or clause
matches. A learned cross-encoder re-ranker would be more accurate but adds
an inference-time model and its own training data requirement; TF-IDF
cosine similarity gets most of the benefit (surfacing exact term matches)
for negligible cost.

**Two classifiers, not one.** `ml/intent_classifier.py` and
`models/intent_classifier.py` both use TF-IDF + Logistic Regression, but
answer different questions. The first asks "should this touch the document
corpus at all?" (greeting/chit-chat vs. document question) and runs before
retrieval. The second only ever sees questions already headed for the
documents and asks "what shape of answer does this need?"
(factual/summarization/comparison), so factual questions retrieval answers
with high confidence can skip the LLM call entirely. Merging them into one
multi-class model was considered and rejected — the two decisions have
different failure costs and different training data, so keeping them
separate keeps each one simple enough to actually reason about.

**A cheap lexical overlap score for faithfulness, not an LLM judge.**
`eval/check_faithfulness.py` scores generated answers against retrieved
context with a ROUGE-L-style longest-common-subsequence F1 instead of
asking another LLM to grade groundedness. An LLM judge would catch
paraphrased hallucination this can't, but it also costs a completion call
per answer and introduces its own failure modes. The overlap score is a
fast, deterministic first pass — flag anything below threshold for human
review rather than treat it as a final verdict.

**Similarity-threshold precision@k proxy in `/analytics/summary`, real
precision@k in `evaluation/`.** Live traffic has no ground truth, so the
analytics endpoint approximates relevance from retrieval similarity scores
for a fast, always-available signal. The `evaluation` module computes the
real metric against a labeled query set for a trustworthy read on retrieval
quality; the two are complementary, not redundant.

## Setup

### Prerequisites

- Python 3.11+
- An OpenAI API key (embeddings only)
- A Linux host with Docker and docker-compose

### Local installation

```bash
git clone <this-repo>
cd enterprise-rag-project
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env      # then fill in OPENAI_API_KEY and LLM_* vars
```

### Running locally (without Docker)

Start the model server first, then the backend and frontend:

```bash
# Terminal 1 — vLLM (requires CUDA GPU)
python -m vllm.entrypoints.openai.api_server \
  --model mistral-7b-instruct --port 8000

# Terminal 1 (alternative) — llama.cpp
./server -m models/mistral-7b-instruct.Q4_K_M.gguf --port 8080

# Terminal 2 — FastAPI backend
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

# Terminal 3 — Streamlit frontend
streamlit run frontend/app.py
```

Open `http://localhost:8501`.

### Running with Docker (recommended)

```bash
# vLLM backend (GPU required)
export OPENAI_API_KEY=sk-...
docker compose --profile vllm up --build

# llama.cpp backend (CPU)
export OPENAI_API_KEY=sk-...
export LLAMA_MODEL_FILE=mistral-7b-instruct.Q4_K_M.gguf
docker compose --profile llamacpp up --build
```

The FastAPI backend is on `http://localhost:8000`, the Streamlit frontend
on `http://localhost:8501`.  Model weights are persisted in the
`models_data` Docker volume so they are downloaded once and reused.

## API

| Method | Path                | Description                                   |
|--------|---------------------|------------------------------------------------|
| POST   | `/upload`           | Upload a PDF, preprocess and index it          |
| POST   | `/ask`              | Ask a question, routed through intent + RAG   |
| GET    | `/health`           | Vectorstore and embedding-configuration status |
| GET    | `/analytics/summary`| Aggregated query/retrieval stats              |
| GET    | `/stats`            | Index size and per-query token estimates      |

A latency-logging middleware times every request and writes
`retrieval_time_ms`/`generation_time_ms`/`total_time_ms` to
`logs/latency.csv`.  Each row in `logs/query_analytics.csv` now also
records `llm_backend` — which server handled generation for that request.
`/ask` returns 502 specifically when the model server is unreachable or
returns an error, so an inference failure is distinguishable from an
application bug.

Set `API_KEY` in the environment to require an `X-API-Key` header on
`/upload`, `/ask` and `/analytics/summary`.

## Evaluation and tuning

See `evaluation/README.md` for building a labeled query set and running
the precision/recall/MRR suite, and `tuning/README.md` for the
chunk-size/overlap/top-k grid search.

`eval/` is the broader, query-level harness — see `eval/README.md` for the
full methodology, including how to run the faithfulness checker against the
self-hosted backend. In short:

- `eval/test_queries.json` — 41 template queries labeled by type; fill in
  real answers/chunk ids from your own document before running.
- `python -m eval.evaluate_retrieval` — precision@k/recall@k/MRR to
  `eval/retrieval_results.csv`.
- `python -m eval.compare_configs data/your_document.pdf` — precision,
  recall and latency for three configurations to `eval/config_comparison.csv`.
- `python -m eval.check_faithfulness` — lexical-overlap groundedness score
  to `eval/faithfulness_results.csv`.
- `python -m models.train_query_type_classifier` — retrain the
  factual/summarization/comparison classifier.

## Results

Numbers below are placeholders — the shipped `eval/test_queries.json` is
a template. Fill in real reference answers and chunk ids for your document,
then run the `eval` and `models` scripts above and drop the actual numbers
in here.

| Metric                      | Before tuning | After tuning |
|------------------------------|:---:|:---:|
| Precision@5                  | — | — |
| Recall@5                     | — | — |
| MRR                          | — | — |
| Avg latency (ms)             | — | — |
| Faithfulness (% grounded)    | — | — |
| Query-type classifier F1     | — | — |

## Usage

1. Upload a PDF document.
2. Wait for it to be processed — text is cleaned, chunked, deduplicated,
   quality-filtered and embedded into the vector store.
3. Ask a question. Greetings and off-topic chit-chat get an instant canned
   reply; document questions are retrieved, re-ranked and answered by the
   self-hosted model with source citations and a confidence score.
