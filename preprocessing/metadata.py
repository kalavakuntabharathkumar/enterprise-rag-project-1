"""Metadata extraction for individual chunks.

Every chunk gets a small structured record on top of its source file, which
the analytics layer and evaluation suite both rely on to identify chunks
(`source#chunk_index`) and reason about chunk composition.
"""
import re

WORD_RE = re.compile(r"\b\w+\b")
HEADING_RE = re.compile(r"^[A-Z0-9 \-:]{4,}$")


def extract_chunk_metadata(chunk: str, source: str, chunk_index: int) -> dict:
    words = WORD_RE.findall(chunk)
    first_line = chunk.strip().split("\n")[0] if chunk.strip() else ""

    return {
        "source": source,
        "chunk_index": chunk_index,
        "char_count": len(chunk),
        "word_count": len(words),
        "has_numbers": any(ch.isdigit() for ch in chunk),
        "looks_like_heading": bool(HEADING_RE.match(first_line)),
    }
