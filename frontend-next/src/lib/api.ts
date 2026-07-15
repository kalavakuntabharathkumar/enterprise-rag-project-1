/**
 * Typed API client for the FastAPI backend.
 *
 * The backend URL is read from the NEXT_PUBLIC_API_URL environment variable
 * at runtime. Set it in .env.local for development or in your deployment
 * environment for production. Falls back to http://localhost:8000.
 */

import type {
  AskResponse,
  AnalyticsSummaryResponse,
  StatsResponse,
  UploadResponse,
  QuestionRequest,
} from "@/types/api";

const API_BASE =
  process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, "") ?? "http://localhost:8000";

class ApiClientError extends Error {
  constructor(
    public status: number,
    public detail: string,
  ) {
    super(`API ${status}: ${detail}`);
    this.name = "ApiClientError";
  }
}

async function request<T>(
  path: string,
  init?: RequestInit,
): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json", ...init?.headers },
    ...init,
  });

  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      if (typeof body?.detail === "string") detail = body.detail;
    } catch {
      // ignore parse error — keep statusText
    }
    throw new ApiClientError(res.status, detail);
  }

  return res.json() as Promise<T>;
}

export async function askQuestion(
  question: string,
  sessionId?: string,
): Promise<AskResponse> {
  const body: QuestionRequest = { question };
  if (sessionId) body.session_id = sessionId;
  return request<AskResponse>("/ask", {
    method: "POST",
    body: JSON.stringify(body),
  });
}

export async function getStats(): Promise<StatsResponse> {
  return request<StatsResponse>("/stats");
}

export async function getAnalyticsSummary(): Promise<AnalyticsSummaryResponse> {
  return request<AnalyticsSummaryResponse>("/analytics/summary");
}

export async function uploadPdf(file: File): Promise<UploadResponse> {
  const form = new FormData();
  form.append("file", file);
  return request<UploadResponse>("/upload", {
    method: "POST",
    // Let the browser set the multipart boundary automatically
    headers: {},
    body: form,
  });
}

export { ApiClientError };
