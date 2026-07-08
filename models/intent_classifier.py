"""TF-IDF + Logistic Regression query-type classifier.

Distinct from `ml/intent_classifier.py` (which filters out greetings and
chit-chat before retrieval even runs). This one only ever sees questions
that are already headed for the document corpus, and labels *what kind*
of answer they need: `factual`, `summarization` or `comparison`. The RAG
pipeline uses that label to skip the LLM for factual questions retrieval
already answers confidently, per the "route queries through this
classifier before hitting the LLM" requirement.

Reports held-out accuracy/F1 whenever it (re)trains, since a classifier
with no reported quality number isn't trustworthy to route production
traffic on.
"""
import os

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from backend.logger import app_logger
from models.data.query_type_training_data import TRAINING_EXAMPLES

MODEL_PATH = os.path.join(os.path.dirname(__file__), "trained", "query_type_classifier.joblib")


class QueryTypeClassifier:
    LABELS = ("factual", "summarization", "comparison")

    def __init__(self, model_path: str = None):
        self.model_path = model_path or MODEL_PATH
        self.pipeline = None
        self.last_eval = None
        self._load_or_train()

    @staticmethod
    def _build_pipeline() -> Pipeline:
        return Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, stop_words="english")),
            ("clf", LogisticRegression(max_iter=1000)),
        ])

    def _load_or_train(self) -> None:
        if os.path.exists(self.model_path):
            try:
                self.pipeline = joblib.load(self.model_path)
                app_logger.info("Loaded query-type classifier from disk")
                return
            except Exception as e:
                app_logger.warning(f"Could not load saved query-type classifier, retraining: {e}")
        self.train()

    def train(self, examples: list = None, test_size: float = 0.25, random_state: int = 42) -> dict:
        """Fit on a train split, report accuracy/F1 on a held-out test
        split, then refit on all data before persisting -- so the saved
        model uses every labeled example while the reported metric is
        still an honest held-out number."""
        examples = examples or TRAINING_EXAMPLES
        texts, labels = zip(*examples)

        x_train, x_test, y_train, y_test = train_test_split(
            list(texts), list(labels), test_size=test_size, random_state=random_state, stratify=list(labels)
        )

        eval_pipeline = self._build_pipeline()
        eval_pipeline.fit(x_train, y_train)
        predictions = eval_pipeline.predict(x_test)

        self.last_eval = {
            "accuracy": accuracy_score(y_test, predictions),
            "f1_macro": f1_score(y_test, predictions, average="macro"),
            "test_size": len(x_test),
            "train_size": len(x_train),
        }
        app_logger.info(
            f"Query-type classifier held-out eval: accuracy={self.last_eval['accuracy']:.3f} "
            f"f1_macro={self.last_eval['f1_macro']:.3f} (n_test={len(x_test)})"
        )

        self.pipeline = self._build_pipeline()
        self.pipeline.fit(list(texts), list(labels))
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(self.pipeline, self.model_path)
        return self.last_eval

    def predict(self, text: str) -> str:
        if not text or not text.strip():
            return "factual"
        return self.pipeline.predict([text])[0]

    def predict_proba(self, text: str) -> dict:
        proba = self.pipeline.predict_proba([text])[0]
        return dict(zip(self.pipeline.classes_, proba.tolist()))
