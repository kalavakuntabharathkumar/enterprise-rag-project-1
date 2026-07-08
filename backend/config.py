import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    VECTORSTORE_PATH = os.getenv("VECTORSTORE_PATH", "./vectorstore")
    DATA_PATH = os.getenv("DATA_PATH", "./data")
    LOGS_PATH = os.getenv("LOGS_PATH", "./logs")
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 400))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))
    TOP_K = int(os.getenv("TOP_K", 3))
    CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", 0.7))
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    MIN_CHUNK_QUALITY = float(os.getenv("MIN_CHUNK_QUALITY", 0.35))
    RERANK_ALPHA = float(os.getenv("RERANK_ALPHA", 0.6))
    # Stub auth: unset means the API is open. Set API_KEY to require callers
    # to send a matching X-API-Key header on /upload and /ask.
    API_KEY = os.getenv("API_KEY")
    # Factual questions whose top retrieved chunk clears this similarity
    # bar are answered by returning that chunk directly, skipping the LLM
    # call entirely. Summarization/comparison questions always go through
    # the LLM since they need synthesis across chunks.
    FACTUAL_DIRECT_THRESHOLD = float(os.getenv("FACTUAL_DIRECT_THRESHOLD", 0.82))
    # Rough per-1K-token cost estimates (USD) used only for the /stats
    # endpoint's estimated_cost_per_query figure. These are illustrative
    # defaults, not live pricing -- override via env vars if your actual
    # rates differ.
    LLM_INPUT_COST_PER_1K = float(os.getenv("LLM_INPUT_COST_PER_1K", 0.0005))
    LLM_OUTPUT_COST_PER_1K = float(os.getenv("LLM_OUTPUT_COST_PER_1K", 0.0015))