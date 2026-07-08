"""Per-request phase latency: retrieval time and generation time, captured
inside `RAGPipeline.ask_question` and logged by FastAPI middleware once
the response is ready.

A contextvar is used (rather than passing a recorder object through every
function signature) so `rag_pipeline.py` doesn't need to know it's being
timed, and each concurrent request gets its own recorder.
"""
import contextvars
import os
import time

import pandas as pd

from backend.config import Config

_current_recorder = contextvars.ContextVar("latency_recorder", default=None)

LOG_COLUMNS = ["timestamp", "path", "retrieval_time_ms", "generation_time_ms", "total_time_ms"]


class LatencyRecorder:
    def __init__(self):
        self.retrieval_ms = None
        self.generation_ms = None


def start_recording() -> LatencyRecorder:
    recorder = LatencyRecorder()
    _current_recorder.set(recorder)
    return recorder


def record_retrieval_time(ms: float) -> None:
    recorder = _current_recorder.get()
    if recorder is not None:
        recorder.retrieval_ms = ms


def record_generation_time(ms: float) -> None:
    recorder = _current_recorder.get()
    if recorder is not None:
        recorder.generation_ms = ms


class timed:
    """Context manager that measures elapsed wall time and reports it
    through one of the record_* functions above.

    Usage: `with timed(record_retrieval_time): ...`
    """

    def __init__(self, reporter):
        self.reporter = reporter
        self._start = None

    def __enter__(self):
        self._start = time.time()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.reporter((time.time() - self._start) * 1000)
        return False


class LatencyLog:
    """Append-only CSV log of per-request latency, one row per HTTP
    request handled by the FastAPI app."""

    def __init__(self, path: str = None):
        self.path = path or os.path.join(Config.LOGS_PATH, "latency.csv")

    def log(self, record: dict) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        row = {col: record.get(col) for col in LOG_COLUMNS}
        df = pd.DataFrame([row], columns=LOG_COLUMNS)
        write_header = not os.path.exists(self.path)
        df.to_csv(self.path, mode="a", header=write_header, index=False)

    def load(self) -> pd.DataFrame:
        if not os.path.exists(self.path):
            return pd.DataFrame(columns=LOG_COLUMNS)
        return pd.read_csv(self.path)
