"""Chunk deduplication.

Repeated boilerplate (legal disclaimers, repeated headers, table-of-contents
entries) can end up as multiple near-identical chunks after splitting. Exact
duplicates are caught with a content hash; near-duplicates are caught with a
sequence-similarity ratio, bounded to chunks of similar length to keep the
comparison cheap.
"""
import hashlib
from difflib import SequenceMatcher


def _normalize_for_hash(text: str) -> str:
    return " ".join(text.lower().split())


def content_hash(text: str) -> str:
    return hashlib.sha256(_normalize_for_hash(text).encode("utf-8")).hexdigest()


def deduplicate_exact(chunks: list) -> list:
    """Remove exact duplicate chunks (after whitespace/case normalization),
    preserving first-seen order."""
    seen = set()
    unique = []
    for chunk in chunks:
        digest = content_hash(chunk)
        if digest not in seen:
            seen.add(digest)
            unique.append(chunk)
    return unique


def deduplicate_near(chunks: list, threshold: float = 0.92) -> list:
    """Remove near-duplicate chunks using a sequence similarity ratio.

    O(n^2) in the worst case, which is fine here since a single document
    typically produces low hundreds of chunks, not millions.
    """
    unique: list = []
    for chunk in chunks:
        is_duplicate = False
        for kept in unique:
            length_diff_ratio = abs(len(chunk) - len(kept)) / max(len(kept), 1)
            if length_diff_ratio > 0.3:
                continue
            if SequenceMatcher(None, chunk, kept).quick_ratio() >= threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            unique.append(chunk)
    return unique
