"""Standalone script to (re)train the query-type classifier and print its
held-out accuracy/F1.

Usage:
    python -m models.train_query_type_classifier
"""
import json

from models.intent_classifier import QueryTypeClassifier


def main():
    classifier = QueryTypeClassifier()
    eval_result = classifier.train()
    print(f"Query-type classifier trained and saved to {classifier.model_path}")
    print(json.dumps(eval_result, indent=2))


if __name__ == "__main__":
    main()
