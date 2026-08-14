"""
TF-IDF feature extraction for the chatbot intent-classification pipeline.

This module converts already-preprocessed text (see src.preprocessing) into
numerical TF-IDF feature vectors using scikit-learn's TfidfVectorizer.

Typical usage
-------------
Training:
    vectorizer, X_train = fit_transform_texts(training_texts)
    save_vectorizer(vectorizer, "models/tfidf_vectorizer.pkl")

Inference:
    vectorizer = load_vectorizer("models/tfidf_vectorizer.pkl")
    X_new = transform_text(vectorizer, [preprocessed_query])

The vectorizer must only ever be fit once, on the training corpus. New/unseen
text at inference time is transformed using the already-fitted vectorizer.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import List, Union

from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer


def create_vectorizer(**kwargs) -> TfidfVectorizer:
    """Create a new, unfitted TfidfVectorizer.

    Extra keyword arguments are forwarded to TfidfVectorizer, allowing
    callers to customize parameters (e.g. ngram_range) if ever needed.
    """
    return TfidfVectorizer(**kwargs)


def fit_vectorizer(vectorizer: TfidfVectorizer, texts: List[str]) -> TfidfVectorizer:
    """Fit a vectorizer's vocabulary on a list of (preprocessed) training texts.

    Returns the same vectorizer instance, now fitted, for convenience.
    """
    texts = _validate_texts(texts)
    if not texts:
        raise ValueError("Cannot fit a vectorizer on an empty list of texts.")
    vectorizer.fit(texts)
    return vectorizer


def transform_text(vectorizer: TfidfVectorizer, texts: List[str]) -> csr_matrix:
    """Transform text into TF-IDF vectors using an already-fitted vectorizer.

    Use this at inference time. Never fits the vectorizer; raises if the
    vectorizer has not been fitted yet.
    """
    texts = _validate_texts(texts)
    _ensure_fitted(vectorizer)
    if not texts:
        # Return a well-formed, empty (0 rows) sparse matrix with the
        # correct number of columns instead of raising an obscure error.
        n_features = len(vectorizer.get_feature_names_out())
        return csr_matrix((0, n_features))
    return vectorizer.transform(texts)


def fit_transform_texts(texts: List[str], **kwargs) -> tuple[TfidfVectorizer, csr_matrix]:
    """Create a vectorizer, fit it on texts, and return (vectorizer, X).

    Convenience wrapper for the common training-time workflow.
    """
    texts = _validate_texts(texts)
    if not texts:
        raise ValueError("Cannot fit a vectorizer on an empty list of texts.")
    vectorizer = create_vectorizer(**kwargs)
    X = vectorizer.fit_transform(texts)
    return vectorizer, X


def save_vectorizer(vectorizer: TfidfVectorizer, path: Union[str, Path]) -> None:
    """Persist a fitted vectorizer to disk using pickle."""
    _ensure_fitted(vectorizer)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(vectorizer, f)


def load_vectorizer(path: Union[str, Path]) -> TfidfVectorizer:
    """Load a previously saved, fitted vectorizer from disk."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No vectorizer found at: {path}")
    with open(path, "rb") as f:
        vectorizer = pickle.load(f)
    _ensure_fitted(vectorizer)
    return vectorizer


def _validate_texts(texts: List[str]) -> List[str]:
    """Basic sanity check: texts must be a list (or tuple) of strings."""
    if texts is None:
        raise TypeError("texts must be a list of strings, got None.")
    if not isinstance(texts, (list, tuple)):
        raise TypeError(f"texts must be a list of strings, got {type(texts).__name__}.")
    for t in texts:
        if not isinstance(t, str):
            raise TypeError(f"All items in texts must be strings, got {type(t).__name__}.")
    return list(texts)


def _ensure_fitted(vectorizer: TfidfVectorizer) -> None:
    """Raise a clear error if the vectorizer has not been fitted yet."""
    if not hasattr(vectorizer, "vocabulary_"):
        raise ValueError(
            "Vectorizer has not been fitted yet. Call fit_vectorizer() or "
            "fit_transform_texts() first."
        )
    