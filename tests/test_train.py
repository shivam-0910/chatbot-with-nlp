"""
tests/test_train.py

Tests for the training pipeline in src/train.py.

These tests exercise the real project components (src.preprocessing,
src.feature_extraction, src.classifier) and, where practical, the real
data/intents.json dataset. Model files produced during tests are written
to pytest's tmp_path fixture so nothing is left behind in the repo.
"""

from pathlib import Path

import pytest

from src.train import (
    DEFAULT_INTENTS_PATH,
    load_training_data,
    prepare_training_data,
    preprocess_patterns,
    train_model,
    save_models,
    run_sanity_predictions,
)
from src.feature_extraction import load_vectorizer, transform_text
from src.classifier import load_classifier, predict_intent_with_confidence


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def test_real_intents_json_loads():
    data = load_training_data(DEFAULT_INTENTS_PATH)
    assert "intents" in data
    assert len(data["intents"]) > 0


def test_patterns_and_labels_extracted():
    data = load_training_data(DEFAULT_INTENTS_PATH)
    patterns, labels = prepare_training_data(data)
    assert len(patterns) > 0
    assert len(labels) > 0


def test_pattern_and_label_counts_match():
    data = load_training_data(DEFAULT_INTENTS_PATH)
    patterns, labels = prepare_training_data(data)
    assert len(patterns) == len(labels)


def test_expected_intents_discovered_dynamically():
    data = load_training_data(DEFAULT_INTENTS_PATH)
    patterns, labels = prepare_training_data(data)
    discovered_tags = set(labels)

    # These are the tags actually present in the real dataset; verify
    # discovery is dynamic (from the file) rather than hard-coded, by
    # cross-checking against the file's own intent tags.
    expected_tags = {intent["tag"] for intent in data["intents"]}
    assert discovered_tags == expected_tags


def test_missing_intents_key_raises(tmp_path):
    bad_file = tmp_path / "bad_intents.json"
    bad_file.write_text('{"not_intents": []}', encoding="utf-8")
    with pytest.raises(ValueError):
        load_training_data(bad_file)


def test_intent_missing_tag_raises(tmp_path):
    bad_file = tmp_path / "bad_intents.json"
    bad_file.write_text(
        '{"intents": [{"patterns": ["hi"]}]}', encoding="utf-8"
    )
    with pytest.raises(ValueError):
        load_training_data(bad_file)


def test_intent_with_no_patterns_raises(tmp_path):
    bad_file = tmp_path / "bad_intents.json"
    bad_file.write_text(
        '{"intents": [{"tag": "greeting", "patterns": []}]}', encoding="utf-8"
    )
    with pytest.raises(ValueError):
        load_training_data(bad_file)


def test_missing_file_raises(tmp_path):
    missing_path = tmp_path / "does_not_exist.json"
    with pytest.raises(FileNotFoundError):
        load_training_data(missing_path)


# ---------------------------------------------------------------------------
# Data preparation / preprocessing
# ---------------------------------------------------------------------------

def test_patterns_pass_through_preprocessing():
    data = load_training_data(DEFAULT_INTENTS_PATH)
    patterns, labels = prepare_training_data(data)
    processed = preprocess_patterns(patterns)

    assert len(processed) == len(patterns)
    # Preprocessing should lowercase and strip punctuation somewhere in
    # the corpus (real dataset has patterns with capitals/punctuation).
    assert any(p != p.lower() for p in patterns)
    assert all(p == p.lower() or p == "" for p in processed if p)


def test_empty_training_data_raises():
    empty_data = {"intents": []}
    with pytest.raises(ValueError):
        prepare_training_data(empty_data)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def test_tfidf_and_classifier_train_successfully():
    data = load_training_data(DEFAULT_INTENTS_PATH)
    patterns, labels = prepare_training_data(data)
    processed = preprocess_patterns(patterns)

    vectorizer, classifier = train_model(processed, labels)

    assert hasattr(vectorizer, "vocabulary_")
    assert hasattr(classifier, "classes_")


def test_classifier_has_multiple_classes():
    data = load_training_data(DEFAULT_INTENTS_PATH)
    patterns, labels = prepare_training_data(data)
    processed = preprocess_patterns(patterns)

    _, classifier = train_model(processed, labels)

    assert len(classifier.classes_) > 1
    assert len(classifier.classes_) == len(set(labels))


# ---------------------------------------------------------------------------
# End-to-end training using the real dataset
# ---------------------------------------------------------------------------

def test_end_to_end_training_with_real_dataset():
    data = load_training_data(DEFAULT_INTENTS_PATH)
    patterns, labels = prepare_training_data(data)
    processed = preprocess_patterns(patterns)
    vectorizer, classifier = train_model(processed, labels)

    results = run_sanity_predictions(data, vectorizer, classifier)

    assert len(results) > 0
    for pattern, expected_tag, predicted_tag, confidence in results:
        assert isinstance(pattern, str) and pattern
        assert isinstance(expected_tag, str) and expected_tag
        assert isinstance(predicted_tag, str) and predicted_tag
        assert 0.0 <= confidence <= 1.0


# ---------------------------------------------------------------------------
# Model persistence
# ---------------------------------------------------------------------------

def test_save_load_transform_predict_roundtrip(tmp_path):
    data = load_training_data(DEFAULT_INTENTS_PATH)
    patterns, labels = prepare_training_data(data)
    processed = preprocess_patterns(patterns)
    vectorizer, classifier = train_model(processed, labels)

    vectorizer_path = tmp_path / "tfidf_vectorizer.pkl"
    classifier_path = tmp_path / "intent_classifier.pkl"

    save_models(vectorizer, classifier, vectorizer_path, classifier_path)

    assert vectorizer_path.exists()
    assert classifier_path.exists()

    loaded_vectorizer = load_vectorizer(vectorizer_path)
    loaded_classifier = load_classifier(classifier_path)

    # Use a real pattern from the dataset for a full predict roundtrip.
    sample_pattern = patterns[0]
    sample_processed = processed[0]
    X = transform_text(loaded_vectorizer, [sample_processed])
    predicted_tag, confidence = predict_intent_with_confidence(loaded_classifier, X)

    assert isinstance(predicted_tag, str)
    assert 0.0 <= confidence <= 1.0
    assert sample_pattern  # sanity: still a real dataset pattern


def test_models_directory_created_automatically(tmp_path):
    data = load_training_data(DEFAULT_INTENTS_PATH)
    patterns, labels = prepare_training_data(data)
    processed = preprocess_patterns(patterns)
    vectorizer, classifier = train_model(processed, labels)

    nested_dir = tmp_path / "nested" / "models"
    vectorizer_path = nested_dir / "tfidf_vectorizer.pkl"
    classifier_path = nested_dir / "intent_classifier.pkl"

    assert not nested_dir.exists()
    save_models(vectorizer, classifier, vectorizer_path, classifier_path)
    assert nested_dir.exists()
    assert vectorizer_path.exists()
    assert classifier_path.exists()