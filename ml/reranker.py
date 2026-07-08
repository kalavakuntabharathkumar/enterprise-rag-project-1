"""Lexical re-ranker.

Embedding similarity alone can miss exact keyword matches (e.g. a specific
clause number or defined term) because it optimizes for semantic closeness,
not lexical overlap. This re-ranker blends normalized embedding similarity
with TF-IDF cosine similarity so exact term matches get a boost without
throwing away the semantic signal entirely.
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class LexicalReranker:
    def rerank(self, query: str, doc_texts: list, embedding_scores: list, alpha: float = 0.6) -> list:
        """Return the indices of `doc_texts`, reordered best-first by a
        weighted combination of embedding similarity (`alpha`) and TF-IDF
        lexical similarity (`1 - alpha`).
        """
        if not doc_texts:
            return []
        if len(doc_texts) == 1:
            return [0]

        corpus = doc_texts + [query]
        vectorizer = TfidfVectorizer(stop_words="english")
        try:
            tfidf_matrix = vectorizer.fit_transform(corpus)
        except ValueError:
            # Corpus was empty after stopword removal (e.g. very short chunks).
            return list(range(len(doc_texts)))

        query_vec = tfidf_matrix[-1]
        doc_vecs = tfidf_matrix[:-1]
        lexical_scores = cosine_similarity(doc_vecs, query_vec).flatten()

        combined_scores = []
        for i in range(len(doc_texts)):
            emb_score = embedding_scores[i] if i < len(embedding_scores) else 0.0
            combined_scores.append(alpha * emb_score + (1 - alpha) * lexical_scores[i])

        return sorted(range(len(doc_texts)), key=lambda i: combined_scores[i], reverse=True)
