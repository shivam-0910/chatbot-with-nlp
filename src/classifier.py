"""
Intent classification for the chatbot NLP pipeline.

This module trains and uses a Logistic Regression classifier on top of
TF-IDF feature vectors produced by src.feature_extraction. It is
intentionally independent of preprocessing and vectorization: it only
deals with feature matrices (X) and intent labels (y).

Typical usage
-------------
Training:
    classifier = create_classifier()
    train_classifier(classifier, X_train, y_train)
    save_classifier(classifier, "models/intent_classifier.pkl")

Inference:
    classifier = load_classifier("models/intent_classifier.pkl")
    intent, confidence = predict_intent_with_confidence(classifier, X_new)
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
from scipy.sparse import spmatrix
from sklearn.linear_model import LogisticRegression

FeatureMatrix = Union[np.ndarray, spmatrix]


def create_classifier(**kwargs) -> LogisticRegression:
    """Create a new, untrained Logistic Regression classifier.

    Uses sensible defaults for a small multi-class text dataset
    (max_iter=1000 so convergence isn't an issue on sparse TF-IDF
    features, random_state=42 for reproducibility). Extra keyword
    arguments are forwarded to LogisticRegression.
    """
    defaults = {"max_iter": 1000, "random_state": 42}
    defaults.update(kwargs)
    return LogisticRegression(**defaults)


def train_classifier(
    classifier: LogisticRegression, X: FeatureMatrix, y: List[str]
) -> LogisticRegression:
    """Train the classifier on TF-IDF features X and intent labels y.

    Returns the same classifier instance, now fitted, for convenience.
    """
    _validate_training_data(X, y)
    classifier.fit(X, y)
    return classifier


def predict_intent(classifier: LogisticRegression, X: FeatureMatrix) -> Union[str, List[str]]:
    """Predict the most likely intent(s) for feature vector(s) X.

    Returns a single label if X contains one sample, or a list of labels
    for a batch of samples.
    """
    _ensure_fitted(classifier)
    predictions = classifier.predict(X)
    predictions = list(predictions)
    if len(predictions) == 1:
        return predictions[0]
    return predictions


def predict_intent_with_confidence(
    classifier: LogisticRegression, X: FeatureMatrix
) -> Union[Tuple[str, float], List[Tuple[str, float]]]:
    """Predict intent(s) along with the model's confidence (probability).

    Confidence is the actual predicted-class probability from
    classifier.predict_proba() -- never a fabricated value.

    Returns a single (intent, confidence) tuple if X contains one sample,
    or a list of such tuples for a batch of samples.
    """
    _ensure_fitted(classifier)
    probabilities = classifier.predict_proba(X)
    classes = classifier.classes_

    results = []
    for row in probabilities:
        best_index = int(np.argmax(row))
        intent = classes[best_index]
        confidence = float(row[best_index])
        results.append((intent, confidence))

    if len(results) == 1:
        return results[0]
    return results


def predict_intent_with_threshold(
    classifier: LogisticRegression,
    X: FeatureMatrix,
    threshold: float = 0.50,
    fallback: str = "fallback",
) -> Union[str, List[str]]:
    """Predict intent(s), returning `fallback` when confidence < threshold.

    `threshold` is a simple, configurable cutoff (not scientifically
    derived) that can be tuned later using validation data. This is a
    separate mechanism from the dataset's own "fallback" intent class --
    it does not assume the model has learned a perfect fallback detector.
    """
    if not (0.0 <= threshold <= 1.0):
        raise ValueError(f"threshold must be between 0 and 1, got {threshold}.")

    predictions = predict_intent_with_confidence(classifier, X)
    if isinstance(predictions, tuple):
        predictions = [predictions]
        single = True
    else:
        single = False

    results = [
        intent if confidence >= threshold else fallback
        for intent, confidence in predictions
    ]

    if single:
        return results[0]
    return results


def save_classifier(classifier: LogisticRegression, path: Union[str, Path]) -> None:
    """Persist a trained classifier to disk using pickle."""
    _ensure_fitted(classifier)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(classifier, f)


def load_classifier(path: Union[str, Path]) -> LogisticRegression:
    """Load a previously saved, trained classifier from disk."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No classifier found at: {path}")
    with open(path, "rb") as f:
        classifier = pickle.load(f)
    _ensure_fitted(classifier)
    return classifier


def _validate_training_data(X: FeatureMatrix, y: List[str]) -> None:
    """Validate obvious training-data problems with clear errors."""
    if X is None or y is None:
        raise TypeError("X and y must not be None.")

    n_samples = X.shape[0] if hasattr(X, "shape") else len(X)

    if n_samples == 0:
        raise ValueError("Cannot train on empty feature matrix X.")
    if len(y) == 0:
        raise ValueError("Cannot train on empty labels y.")
    if n_samples != len(y):
        raise ValueError(
            f"Mismatched sample/label counts: X has {n_samples} samples, "
            f"y has {len(y)} labels."
        )
    if any(label is None for label in y):
        raise ValueError("Labels in y must not contain None.")


def _ensure_fitted(classifier: LogisticRegression) -> None:
    """Raise a clear error if the classifier has not been trained yet."""
    if not hasattr(classifier, "classes_"):
        raise ValueError(
            "Classifier has not been trained yet. Call train_classifier() first."
        )