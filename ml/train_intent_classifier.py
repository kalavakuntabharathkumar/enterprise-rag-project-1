"""Standalone script to (re)train and persist the intent classifier.

Usage:
    python -m ml.train_intent_classifier
"""
from ml.intent_classifier import IntentClassifier


def main():
    classifier = IntentClassifier()
    classifier.train()
    print(f"Intent classifier trained and saved to {classifier.model_path}")


if __name__ == "__main__":
    main()
