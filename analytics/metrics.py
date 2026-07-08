"""Retrieval-quality statistics computed directly with pandas/numpy over
the query analytics log.
"""
import numpy as np
import pandas as pd


def latency_distribution(df: pd.DataFrame) -> dict:
    if df.empty or "latency_ms" not in df:
        return {}
    latencies = pd.to_numeric(df["latency_ms"], errors="coerce").dropna().to_numpy(dtype=float)
    if latencies.size == 0:
        return {}
    return {
        "count": int(latencies.size),
        "mean_ms": float(np.mean(latencies)),
        "p50_ms": float(np.percentile(latencies, 50)),
        "p90_ms": float(np.percentile(latencies, 90)),
        "p99_ms": float(np.percentile(latencies, 99)),
        "max_ms": float(np.max(latencies)),
    }


def similarity_stats(df: pd.DataFrame) -> dict:
    if df.empty or "avg_similarity" not in df:
        return {}
    sims = pd.to_numeric(df["avg_similarity"], errors="coerce").dropna().to_numpy(dtype=float)
    if sims.size == 0:
        return {}
    return {
        "mean": float(np.mean(sims)),
        "std": float(np.std(sims)),
        "min": float(np.min(sims)),
        "max": float(np.max(sims)),
    }


def precision_at_k_proxy(df: pd.DataFrame, similarity_threshold: float = 0.75) -> float:
    """Approximate precision@k from logged data: the fraction of answered
    queries whose average retrieval similarity clears a relevance
    threshold. This is a proxy for real precision@k (see the `evaluation`
    module for the ground-truth version against a labeled query set).
    """
    if df.empty or "avg_similarity" not in df:
        return 0.0
    sims = pd.to_numeric(df["avg_similarity"], errors="coerce").dropna()
    if sims.empty:
        return 0.0
    return float((sims >= similarity_threshold).mean())


def summarize(df: pd.DataFrame) -> dict:
    answered = pd.to_numeric(df["answered"], errors="coerce") if "answered" in df else pd.Series(dtype=float)
    return {
        "total_queries": int(len(df)),
        "answer_rate": float(answered.mean()) if not answered.empty else 0.0,
        "latency": latency_distribution(df),
        "similarity": similarity_stats(df),
        "precision_at_k_proxy": precision_at_k_proxy(df),
    }
