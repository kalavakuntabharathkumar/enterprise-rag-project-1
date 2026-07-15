"use client";

import { useEffect, useState } from "react";
import { getStats, getAnalyticsSummary, ApiClientError } from "@/lib/api";
import type { StatsResponse, AnalyticsSummaryResponse } from "@/types/api";

// ── Helpers ───────────────────────────────────────────────────────────────

function fmt(n: number, decimals = 0): string {
  if (n === undefined || n === null || isNaN(n)) return "—";
  return n.toLocaleString("en-US", {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  });
}

function bytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / 1024 / 1024).toFixed(2)} MB`;
}

function pct(n: number): string {
  return `${(n * 100).toFixed(1)}%`;
}

// ── Stat card ─────────────────────────────────────────────────────────────

function StatCard({
  label,
  value,
  sub,
  accent,
}: {
  label: string;
  value: string;
  sub?: string;
  accent?: string;
}) {
  return (
    <div
      style={{
        background: "#fff",
        border: "1px solid #e5e5e5",
        borderRadius: 10,
        padding: "18px 20px",
        display: "flex",
        flexDirection: "column",
        gap: 4,
      }}
    >
      <span style={{ fontSize: 11, fontWeight: 600, color: "#9ca3af", textTransform: "uppercase", letterSpacing: "0.05em" }}>
        {label}
      </span>
      <span
        style={{
          fontSize: 26,
          fontWeight: 700,
          letterSpacing: "-0.02em",
          color: accent ?? "#111",
          lineHeight: 1.2,
        }}
      >
        {value}
      </span>
      {sub && (
        <span style={{ fontSize: 12, color: "#6b7280" }}>{sub}</span>
      )}
    </div>
  );
}

// ── Cache hit-rate bar ────────────────────────────────────────────────────

function HitRateBar({ rate, hits, misses }: { rate: number; hits: number; misses: number }) {
  const pctNum = Math.round(rate * 100);
  const color = pctNum >= 60 ? "#166534" : pctNum >= 30 ? "#92400e" : "#991b1b";
  const barColor = pctNum >= 60 ? "#22c55e" : pctNum >= 30 ? "#f59e0b" : "#ef4444";

  return (
    <div
      style={{
        background: "#fff",
        border: "1px solid #e5e5e5",
        borderRadius: 10,
        padding: "18px 20px",
        gridColumn: "span 2",
      }}
    >
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 12 }}>
        <span style={{ fontSize: 11, fontWeight: 600, color: "#9ca3af", textTransform: "uppercase", letterSpacing: "0.05em" }}>
          Cache Hit Rate
        </span>
        <span style={{ fontSize: 22, fontWeight: 700, color, letterSpacing: "-0.02em" }}>
          {pct(rate)}
        </span>
      </div>

      {/* Bar */}
      <div
        style={{
          height: 10,
          background: "#f0f0f0",
          borderRadius: 5,
          overflow: "hidden",
          marginBottom: 10,
        }}
      >
        <div
          style={{
            height: "100%",
            width: `${pctNum}%`,
            background: barColor,
            borderRadius: 5,
            transition: "width 0.6s ease",
          }}
        />
      </div>

      <div style={{ display: "flex", gap: 20 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <span style={{ width: 8, height: 8, borderRadius: "50%", background: "#22c55e", display: "inline-block" }} />
          <span style={{ fontSize: 12, color: "#6b7280" }}>
            <strong style={{ color: "#111" }}>{fmt(hits)}</strong> hits
          </span>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <span style={{ width: 8, height: 8, borderRadius: "50%", background: "#d1d5db", display: "inline-block" }} />
          <span style={{ fontSize: 12, color: "#6b7280" }}>
            <strong style={{ color: "#111" }}>{fmt(misses)}</strong> misses
          </span>
        </div>
      </div>
    </div>
  );
}

// ── Section heading ───────────────────────────────────────────────────────

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div style={{ marginBottom: 32 }}>
      <h2
        style={{
          fontSize: 12,
          fontWeight: 700,
          color: "#6b7280",
          textTransform: "uppercase",
          letterSpacing: "0.06em",
          marginBottom: 12,
        }}
      >
        {title}
      </h2>
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fill, minmax(180px, 1fr))",
          gap: 12,
        }}
      >
        {children}
      </div>
    </div>
  );
}

// ── Skeleton loader ───────────────────────────────────────────────────────

function Skeleton() {
  return (
    <div
      style={{
        background: "#e5e7eb",
        borderRadius: 10,
        height: 90,
        animation: "pulse 1.5s ease-in-out infinite",
      }}
    >
      <style>{`
        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.5; }
        }
      `}</style>
    </div>
  );
}

// ── Main page ─────────────────────────────────────────────────────────────

export default function AnalyticsPage() {
  const [stats, setStats] = useState<StatsResponse | null>(null);
  const [summary, setSummary] = useState<AnalyticsSummaryResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    (async () => {
      setLoading(true);
      setError(null);
      try {
        const [s, a] = await Promise.all([getStats(), getAnalyticsSummary()]);
        setStats(s);
        setSummary(a);
      } catch (err) {
        setError(
          err instanceof ApiClientError
            ? `${err.status}: ${err.detail}`
            : "Failed to reach the backend.",
        );
      } finally {
        setLoading(false);
      }
    })();
  }, []);

  return (
    <div
      style={{
        height: "100%",
        overflowY: "auto",
        padding: "32px 24px",
      }}
    >
      <div style={{ maxWidth: 860, margin: "0 auto" }}>

        {/* Header */}
        <div style={{ marginBottom: 28 }}>
          <h1 style={{ fontSize: 22, fontWeight: 700, letterSpacing: "-0.02em", color: "#111" }}>
            Analytics
          </h1>
          <p style={{ fontSize: 13, color: "#6b7280", marginTop: 4 }}>
            Live stats from the running FastAPI backend.
          </p>
        </div>

        {error && (
          <div
            style={{
              padding: "12px 16px",
              background: "#fee2e2",
              border: "1px solid #fca5a5",
              borderRadius: 8,
              color: "#991b1b",
              fontSize: 13,
              marginBottom: 24,
            }}
          >
            {error}
          </div>
        )}

        {/* Index */}
        <Section title="Vector Index">
          {loading ? (
            [0, 1, 2].map((i) => <Skeleton key={i} />)
          ) : stats ? (
            <>
              <StatCard label="Documents indexed" value={fmt(stats.total_documents_indexed)} />
              <StatCard label="Total chunks" value={fmt(stats.total_chunks)} />
              <StatCard label="Vector store size" value={bytes(stats.vector_db_size_bytes)} />
            </>
          ) : null}
        </Section>

        {/* Cache */}
        <Section title="Retrieval Cache">
          {loading ? (
            [0, 1, 2].map((i) => <Skeleton key={i} />)
          ) : stats ? (
            <>
              <HitRateBar
                rate={stats.cache_hit_rate}
                hits={stats.cache_hits}
                misses={stats.cache_misses}
              />
              <StatCard
                label="Cache entries"
                value={`${fmt(stats.cache_size)} / ${fmt(stats.cache_maxsize)}`}
                sub="current / max capacity"
              />
            </>
          ) : null}
        </Section>

        {/* Token / cost */}
        <Section title="Per-Query Cost Estimates">
          {loading ? (
            [0, 1].map((i) => <Skeleton key={i} />)
          ) : stats ? (
            <>
              <StatCard
                label="Avg tokens / query"
                value={fmt(stats.avg_tokens_per_query, 1)}
                sub="input + output estimate"
              />
              <StatCard
                label="Avg cost / query"
                value={
                  stats.estimated_cost_per_query_usd === 0
                    ? "$0.00"
                    : `$${stats.estimated_cost_per_query_usd.toFixed(5)}`
                }
                sub="self-hosted LLM · embeddings only"
              />
            </>
          ) : null}
        </Section>

        {/* Query log summary */}
        <Section title="Query Log">
          {loading ? (
            [0, 1, 2].map((i) => <Skeleton key={i} />)
          ) : summary ? (
            <>
              <StatCard label="Total queries" value={fmt(summary.total_queries)} />
              <StatCard
                label="Answer rate"
                value={pct(summary.answer_rate)}
                sub="queries that returned an answer"
                accent={summary.answer_rate >= 0.7 ? "#166534" : "#92400e"}
              />
              <StatCard
                label="Precision@k proxy"
                value={pct(summary.precision_at_k_proxy)}
                sub="similarity ≥ 0.75 threshold"
              />
              {summary.latency && "mean_ms" in summary.latency && (
                <>
                  <StatCard
                    label="Avg latency"
                    value={`${fmt(summary.latency.mean_ms ?? 0)} ms`}
                    sub={`p90: ${fmt(summary.latency.p90_ms ?? 0)} ms`}
                  />
                  <StatCard
                    label="p99 latency"
                    value={`${fmt(summary.latency.p99_ms ?? 0)} ms`}
                  />
                </>
              )}
            </>
          ) : !error ? (
            <p style={{ color: "#9ca3af", fontSize: 13 }}>
              No query log data yet — ask some questions first.
            </p>
          ) : null}
        </Section>

        {/* Refresh */}
        <div style={{ textAlign: "center", paddingBottom: 16 }}>
          <button
            onClick={() => {
              setLoading(true);
              Promise.all([getStats(), getAnalyticsSummary()])
                .then(([s, a]) => { setStats(s); setSummary(a); setError(null); })
                .catch((err) =>
                  setError(err instanceof ApiClientError ? err.detail : "Refresh failed"),
                )
                .finally(() => setLoading(false));
            }}
            disabled={loading}
            style={{
              padding: "7px 18px",
              border: "1px solid #d1d5db",
              borderRadius: 7,
              background: "#fff",
              fontSize: 13,
              fontWeight: 500,
              cursor: loading ? "not-allowed" : "pointer",
              color: "#374151",
              opacity: loading ? 0.5 : 1,
            }}
          >
            {loading ? "Loading…" : "↻ Refresh"}
          </button>
        </div>

      </div>
    </div>
  );
}
