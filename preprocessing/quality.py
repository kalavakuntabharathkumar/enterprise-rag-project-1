"""Chunk quality scoring.

Naive fixed-size splitting produces some chunks that are nearly useless for
retrieval: fragments that are too short, chunks that are mostly numbers or
table artifacts, or chunks dominated by stopwords with little real content.
This module scores each chunk in [0, 1] and filters out the low-quality
tail before it ever reaches the vector store.
"""

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "of", "to",
    "in", "on", "at", "for", "with", "by", "is", "are", "was", "were", "be",
    "been", "being", "this", "that", "these", "those", "it", "its", "as",
    "from", "into", "than", "so", "such", "not", "no", "do", "does", "did",
    "will", "would", "can", "could", "should", "may", "might", "must",
    "have", "has", "had", "i", "you", "he", "she", "we", "they",
}


def score_chunk_quality(chunk: str) -> float:
    """Heuristic quality score combining chunk length, alphabetic density
    and lexical information density (inverse stopword ratio)."""
    stripped = chunk.strip()
    if not stripped:
        return 0.0

    words = stripped.split()
    length_score = min(len(words) / 5, 1.0)

    alpha_chars = sum(c.isalpha() for c in stripped)
    alpha_ratio = alpha_chars / max(len(stripped), 1)

    lowered_words = [w.lower().strip(".,;:!?()[]\"'") for w in words]
    lowered_words = [w for w in lowered_words if w]
    stopword_ratio = (
        sum(1 for w in lowered_words if w in STOPWORDS) / len(lowered_words)
        if lowered_words else 1.0
    )
    info_density = 1 - stopword_ratio

    score = 0.4 * length_score + 0.35 * alpha_ratio + 0.25 * info_density
    return round(max(0.0, min(score, 1.0)), 4)


def filter_low_quality(chunks: list, min_score: float = 0.35):
    """Return (kept_chunks, all_scores) where all_scores is a list of
    (chunk, score) pairs for every input chunk, kept or not."""
    scored = [(chunk, score_chunk_quality(chunk)) for chunk in chunks]
    kept = [chunk for chunk, score in scored if score >= min_score]
    return kept, scored
