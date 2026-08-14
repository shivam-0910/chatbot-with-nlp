"""
train.py

Training pipeline for the Chatbot with NLP intent classifier.

This script wires together the already-implemented, validated components:

    data/intents.json  ->  src.preprocessing  ->  src.feature_extraction  ->  src.classifier

It does not duplicate any preprocessing, TF-IDF, or classifier logic --
it only loads data, calls the existing APIs, and saves the results.

Pipeline
--------
    data/intents.json
            |
            v
    load intents + patterns
            |
            v
    extract (pattern, tag) pairs
            |
            v
    preprocess_text() for every pattern      (src.preprocessing)
            |
            v
    fit_transform_texts()                    (src.feature_extraction)
            |
            v
    create_classifier() + train_classifier() (src.classifier)
            |
            v
    save_vectorizer() / save_classifier()
            |
            v
    models/tfidf_vectorizer.pkl, models/intent_classifier.pkl

Usage
-----
    python -m src.train
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

from src.preprocessing import preprocess_text
from src.feature_extraction import fit_transform_texts, save_vectorizer
from src.classifier import create_classifier, train_classifier, save_classifier, predict_intent_with_confidence
from src.feature_extraction import transform_text

# Default locations, relative to the project root.
DEFAULT_INTENTS_PATH = Path("data/intents.json")
DEFAULT_MODELS_DIR = Path("models")
DEFAULT_VECTORIZER_PATH = DEFAULT_MODELS_DIR / "tfidf_vectorizer.pkl"
DEFAULT_CLASSIFIER_PATH = DEFAULT_MODELS_DIR / "intent_classifier.pkl"

# Tags we prefer to show in the sanity-check summary, if present in the
# dataset. This is just a display preference -- every tag is still
# eligible, and we fall back gracefully if any of these are missing.
PREFERRED_SANITY_TAGS = [
    "greeting",
    "courses",
    "fees",
    "library",
    "attendance",
    "examinations",
]


def load_training_data(intents_path: Path = DEFAULT_INTENTS_PATH) -> Dict:
    """Load and minimally validate the raw intents.json structure.

    Returns the parsed JSON dict. Raises clear errors for structural
    problems (missing 'intents' key, an intent missing a tag, an intent
    with no patterns).
    """
    intents_path = Path(intents_path)
    if not intents_path.exists():
        raise FileNotFoundError(f"Could not find intents file at: {intents_path}")

    with open(intents_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "intents" not in data or not isinstance(data["intents"], list):
        raise ValueError("intents.json must contain a top-level 'intents' list.")

    if len(data["intents"]) == 0:
        raise ValueError("intents.json contains no intents.")

    for intent in data["intents"]:
        if "tag" not in intent or not intent["tag"]:
            raise ValueError(f"Found an intent with a missing/empty 'tag': {intent}")
        if "patterns" not in intent or not intent["patterns"]:
            raise ValueError(
                f"Intent '{intent.get('tag')}' has no patterns; every intent "
                f"must have at least one training pattern."
            )

    return data


def prepare_training_data(data: Dict) -> Tuple[List[str], List[str]]:
    """Extract (pattern, tag) pairs from the loaded intents data.

    Returns two parallel lists: raw patterns and their corresponding
    intent labels. Raises a clear error if the resulting dataset is
    empty or inconsistent.
    """
    patterns: List[str] = []
    labels: List[str] = []

    for intent in data["intents"]:
        tag = intent["tag"]
        for pattern in intent["patterns"]:
            if not pattern or not isinstance(pattern, str):
                continue
            patterns.append(pattern)
            labels.append(tag)

    if len(patterns) == 0:
        raise ValueError("No training patterns were extracted from intents.json.")

    if len(patterns) != len(labels):
        # Should be unreachable given the loop above, but kept as an
        # explicit invariant check per the task requirements.
        raise ValueError(
            f"Mismatched patterns/labels counts: {len(patterns)} patterns, "
            f"{len(labels)} labels."
        )

    return patterns, labels


def preprocess_patterns(patterns: List[str]) -> List[str]:
    """Run every raw pattern through the existing preprocess_text()."""
    return [preprocess_text(pattern) for pattern in patterns]


def train_model(
    processed_patterns: List[str], labels: List[str]
) -> Tuple[TfidfVectorizer, LogisticRegression]:
    """Fit the TF-IDF vectorizer and train the classifier.

    The vectorizer is fit exactly once, on the full set of training
    patterns, using the existing fit_transform_texts() helper. The
    classifier is trained on the resulting feature matrix using the
    existing create_classifier() / train_classifier() helpers.
    """
    vectorizer, X_train = fit_transform_texts(processed_patterns)

    classifier = create_classifier()
    train_classifier(classifier, X_train, labels)

    return vectorizer, classifier


def save_models(
    vectorizer: TfidfVectorizer,
    classifier: LogisticRegression,
    vectorizer_path: Path = DEFAULT_VECTORIZER_PATH,
    classifier_path: Path = DEFAULT_CLASSIFIER_PATH,
) -> None:
    """Save the fitted vectorizer and trained classifier to disk."""
    save_vectorizer(vectorizer, vectorizer_path)
    save_classifier(classifier, classifier_path)


def run_sanity_predictions(
    data: Dict,
    vectorizer: TfidfVectorizer,
    classifier: LogisticRegression,
) -> List[Tuple[str, str, str, float]]:
    """Run a handful of real dataset examples through the trained pipeline.

    This is only a sanity check that the full pipeline (preprocessing ->
    TF-IDF -> classifier) works end-to-end -- not a formal accuracy
    evaluation. Examples are real patterns taken from intents.json,
    preferring a fixed set of representative tags when they exist in
    the dataset, otherwise falling back to the first few intents found.

    Returns a list of (pattern, expected_tag, predicted_tag, confidence).
    """
    tag_to_pattern: Dict[str, str] = {}
    for intent in data["intents"]:
        tag = intent["tag"]
        if intent["patterns"]:
            tag_to_pattern[tag] = intent["patterns"][0]

    chosen_tags = [tag for tag in PREFERRED_SANITY_TAGS if tag in tag_to_pattern]
    if not chosen_tags:
        chosen_tags = list(tag_to_pattern.keys())[:6]

    results = []
    for tag in chosen_tags:
        pattern = tag_to_pattern[tag]
        processed = preprocess_text(pattern)
        X = transform_text(vectorizer, [processed])
        predicted_tag, confidence = predict_intent_with_confidence(classifier, X)
        results.append((pattern, tag, predicted_tag, confidence))

    return results


def print_training_summary(
    patterns: List[str],
    labels: List[str],
    vectorizer: TfidfVectorizer,
    vectorizer_path: Path,
    classifier_path: Path,
) -> None:
    """Print a concise, dynamically-calculated training summary."""
    num_classes = len(set(labels))
    num_features = len(vectorizer.get_feature_names_out())

    print("Training completed successfully!\n")
    print(f"Training patterns: {len(patterns)}")
    print(f"Intent classes: {num_classes}")
    print(f"TF-IDF features: {num_features}\n")
    print("Model files:")
    print(f"\u2713 {vectorizer_path}")
    print(f"\u2713 {classifier_path}")


def print_sanity_predictions(results: List[Tuple[str, str, str, float]]) -> None:
    """Print sanity-check predictions (not a formal accuracy evaluation)."""
    print("\nSample predictions (sanity check only, not an accuracy evaluation):\n")
    for pattern, expected_tag, predicted_tag, confidence in results:
        print(f'"{pattern}"')
        print(f"Expected: {expected_tag}")
        print(f"Predicted: {predicted_tag} ({confidence:.2f})\n")


def main() -> None:
    """Run the full training pipeline end-to-end."""
    data = load_training_data(DEFAULT_INTENTS_PATH)
    patterns, labels = prepare_training_data(data)

    processed_patterns = preprocess_patterns(patterns)

    vectorizer, classifier = train_model(processed_patterns, labels)

    save_models(vectorizer, classifier, DEFAULT_VECTORIZER_PATH, DEFAULT_CLASSIFIER_PATH)

    print_training_summary(
        patterns, labels, vectorizer, DEFAULT_VECTORIZER_PATH, DEFAULT_CLASSIFIER_PATH
    )

    sanity_results = run_sanity_predictions(data, vectorizer, classifier)
    print_sanity_predictions(sanity_results)


if __name__ == "__main__":
    main()