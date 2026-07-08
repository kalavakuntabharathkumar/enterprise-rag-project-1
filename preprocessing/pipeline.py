"""Preprocessing pipeline orchestrator.

Replaces naive "split and embed everything" with: clean -> split ->
deduplicate -> quality-filter -> attach metadata. This is what
`RAGPipeline.process_pdf` calls instead of using LangChain's splitter
directly.
"""
from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.config import Config
from backend.logger import app_logger
from preprocessing.cleaner import clean_text, strip_boilerplate
from preprocessing.dedup import deduplicate_exact, deduplicate_near
from preprocessing.metadata import extract_chunk_metadata
from preprocessing.quality import filter_low_quality, score_chunk_quality


class PreprocessingPipeline:
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None,
                 min_quality: float = 0.35):
        self.chunk_size = chunk_size or Config.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or Config.CHUNK_OVERLAP
        self.min_quality = min_quality
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
        )

    def run(self, raw_text: str, source: str) -> list:
        """Run the full pipeline and return a list of
        {"text": ..., "metadata": ...} records ready to become documents."""
        cleaned = clean_text(raw_text)
        cleaned = strip_boilerplate(cleaned)

        raw_chunks = self.splitter.split_text(cleaned)
        deduped = deduplicate_exact(raw_chunks)
        deduped = deduplicate_near(deduped)

        kept_chunks, _ = filter_low_quality(deduped, self.min_quality)

        records = []
        for index, chunk in enumerate(kept_chunks):
            meta = extract_chunk_metadata(chunk, source, index)
            meta["quality_score"] = score_chunk_quality(chunk)
            records.append({"text": chunk, "metadata": meta})

        app_logger.info(
            f"Preprocessing '{source}': {len(raw_chunks)} raw chunks -> "
            f"{len(deduped)} after dedup -> {len(kept_chunks)} kept "
            f"(min_quality={self.min_quality})"
        )
        return records
