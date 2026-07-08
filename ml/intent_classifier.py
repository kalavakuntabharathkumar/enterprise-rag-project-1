"""TF-IDF + Logistic Regression query-intent classifier.

Routes incoming questions into `greeting`, `chit_chat` or
`document_question` before the RAG pipeline decides whether to touch the
vector store and the LLM at all. Greetings and chit-chat get an instant
canned response, saving an embedding call, a retrieval call and an LLM
call on traffic that was never going to need the document corpus.
"""
import os

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from backend.logger import app_logger
from ml.data.intent_training_data import TRAINING_EXAMPLES

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "intent_classifier.joblib")


class IntentClassifier:
    def __init__(self, model_path: str = None):
        self.model_path = model_path or MODEL_PATH
        self.pipeline = None
        self._load_or_train()

    @staticmethod
    def _build_pipeline() -> Pipeline:
        return Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
            ("clf", LogisticRegression(max_iter=1000)),
        ])

    def _load_or_train(self) -> None:
        if os.path.exists(self.model_path):
            try:
                self.pipeline = joblib.load(self.model_path)
                app_logger.info("Loaded intent classifier from disk")
                return
            except Exception as e:
                app_logger.warning(f"Could not load saved intent classifier, retraining: {e}")
        self.train()

    def train(self, examples: list = None) -> None:
        examples = examples or TRAINING_EXAMPLES
        texts, labels = zip(*examples)
        self.pipeline = self._build_pipeline()
        self.pipeline.fit(list(texts), list(labels))
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(self.pipeline, self.model_path)
        app_logger.info(f"Trained intent classifier on {len(texts)} examples")

    def predict(self, text: str) -> str:
        if not text or not text.strip():
            return "chit_chat"
        return self.pipeline.predict([text])[0]

    def predict_proba(self, text: str) -> dict:
        proba = self.pipeline.predict_proba([text])[0]
        return dict(zip(self.pipeline.classes_, proba.tolist()))
