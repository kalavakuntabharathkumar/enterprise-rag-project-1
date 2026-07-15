/**
 * TypeScript interfaces that mirror the FastAPI Pydantic response schemas.
 * Keep these in sync with backend/schemas.py.
 */

// ── Requests ──────────────────────────────────────────────────────────────

export interface QuestionRequest {
  question: string;
  /** Optional thread id for multi-turn memory (LangGraph MemorySaver). */
  session_id?: string;
}

// ── /ask ──────────────────────────────────────────────────────────────────

export interface AskResponse {
  answer: string;
  sources: string[];
  confidence: number;
  intent?: string;
  query_type?: string;
}

// ── /health ───────────────────────────────────────────────────────────────

export interface HealthResponse {
  status: string;
  vectorstore_ready: boolean;
  openai_configured: boolean;
}

// ── /analytics/summary ────────────────────────────────────────────────────

export interface LatencyStats {
  count: number;
  mean_ms: number;
  p50_ms: number;
  p90_ms: number;
  p99_ms: number;
  max_ms: number;
}

export interface SimilarityStats {
  mean: number;
  std: number;
  min: number;
  max: number;
}

export interface AnalyticsSummaryResponse {
  total_queries: number;
  answer_rate: number;
  latency: Partial<LatencyStats>;
  similarity: Partial<SimilarityStats>;
  precision_at_k_proxy: number;
}

// ── /stats ────────────────────────────────────────────────────────────────

export interface StatsResponse {
  total_documents_indexed: number;
  total_chunks: number;
  vector_db_size_bytes: number;
  avg_tokens_per_query: number;
  estimated_cost_per_query_usd: number;
  cache_hits: number;
  cache_misses: number;
  cache_hit_rate: number;
  cache_size: number;
  cache_maxsize: number;
}

// ── /upload ───────────────────────────────────────────────────────────────

export interface UploadResponse {
  message: string;
  file_id: string;
  chunks_indexed: number;
}

// ── Error envelope ────────────────────────────────────────────────────────

export interface ApiError {
  detail: string;
}
