"""Text cleaning utilities applied before chunking.

PDF extraction tends to leave behind artifacts that hurt both chunk quality
and embedding relevance: hyphenated line breaks from justified text,
duplicated whitespace, stray control characters, and short lines that are
really page numbers or running headers/footers. This module cleans those up
so downstream chunking works on normalized text.
"""
import re
import unicodedata

CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")
MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
HYPHEN_LINEBREAK_RE = re.compile(r"(\w)-\n(\w)")


def clean_text(text: str) -> str:
    """Normalize unicode, strip control characters, fix hyphenated
    line-wraps and collapse redundant whitespace."""
    if not text:
        return ""

    text = unicodedata.normalize("NFKC", text)
    text = CONTROL_CHAR_RE.sub("", text)
    text = HYPHEN_LINEBREAK_RE.sub(r"\1\2", text)
    text = MULTI_SPACE_RE.sub(" ", text)
    text = MULTI_NEWLINE_RE.sub("\n\n", text)

    lines = [line.strip() for line in text.split("\n")]
    return "\n".join(lines).strip()


def strip_boilerplate(text: str, min_digit_line_len: int = 3) -> str:
    """Drop lines that are almost certainly page numbers (short,
    all-digit lines). Headers/footers are highly document-specific, so we
    keep this conservative rather than guessing at repeated line removal.
    """
    kept = []
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped and stripped.isdigit() and len(stripped) < min_digit_line_len:
            continue
        kept.append(line)
    return "\n".join(kept)
