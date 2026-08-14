"""Tests for src/classifier.py."""

import sys
from pathlib import Path

import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.classifier import (
    create_classifier,
    train_classifier,
    predict_intent,
    predict_intent_with_confidence,
    predict_intent_with_threshold,
    save_classifier,
    load_classifier,
)


# Small synthetic dataset with three well-separated intents so predictions
# are deterministic and not dependent on the full intents.json dataset.
TRAIN_TEXTS = [
    "hello there", "hi", "good morning", "hey how are you",
    "how much are the fees", "what is the tuition cost", "fee payment details",
    "library opening hours", "where is the library", "book borrowing rules",
]
TRAIN_LABELS = [
    "greeting", "greeting", "greeting", "greeting",
    "fees", "fees", "fees",
    "library", "library", "library",
]


@pytest.fixture
def fitted_vectorizer_and_X():
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(TRAIN_TEXTS)
    return vectorizer, X


@pytest.fixture
def trained_classifier(fitted_vectorizer_and_X):
    _, X = fitted_vectorizer_and_X
    classifier = create_classifier()
    train_classifier(classifier, X, TRAIN_LABELS)
    return classifier


class TestClassifierCreation:
    def test_create_classifier_returns_logistic_regression(self):
        classifier = create_classifier()
        assert isinstance(classifier, LogisticRegression)

    def test_create_classifier_is_untrained(self):
        classifier = create_classifier()
        assert not hasattr(classifier, "classes_")


class TestTraining:
    def test_training_succeeds(self, fitted_vectorizer_and_X):
        _, X = fitted_vectorizer_and_X
        classifier = create_classifier()
        trained = train_classifier(classifier, X, TRAIN_LABELS)
        assert hasattr(trained, "classes_")

    def test_multiple_classes_supported(self, trained_classifier):
        assert set(trained_classifier.classes_) == {"greeting", "fees", "library"}

    def test_empty_training_data_raises(self):
        classifier = create_classifier()
        with pytest.raises((ValueError, TypeError)):
            train_classifier(classifier, [], [])

    def test_mismatched_sample_label_counts_raises(self, fitted_vectorizer_and_X):
        _, X = fitted_vectorizer_and_X
        classifier = create_classifier()
        with pytest.raises(ValueError):
            train_classifier(classifier, X, TRAIN_LABELS[:-1])  # one label short

    def test_none_labels_raise(self, fitted_vectorizer_and_X):
        _, X = fitted_vectorizer_and_X
        classifier = create_classifier()
        with pytest.raises(TypeError):
            train_classifier(classifier, X, None)


class TestPrediction:
    def test_predict_single_returns_known_intent(self, fitted_vectorizer_and_X, trained_classifier):
        vectorizer, _ = fitted_vectorizer_and_X
        X_new = vectorizer.transform(["hi there"])
        prediction = predict_intent(trained_classifier, X_new)
        assert prediction in {"greeting", "fees", "library"}

    def test_predict_batch_returns_list(self, fitted_vectorizer_and_X, trained_classifier):
        vectorizer, _ = fitted_vectorizer_and_X
        X_new = vectorizer.transform(["hi there", "library hours please"])
        predictions = predict_intent(trained_classifier, X_new)
        assert isinstance(predictions, list)
        assert len(predictions) == 2

    def test_predict_before_fitting_raises(self, fitted_vectorizer_and_X):
        vectorizer, _ = fitted_vectorizer_and_X
        X_new = vectorizer.transform(["hi there"])
        classifier = create_classifier()
        with pytest.raises(ValueError):
            predict_intent(classifier, X_new)


class TestConfidence:
    def test_confidence_between_0_and_1(self, fitted_vectorizer_and_X, trained_classifier):
        vectorizer, _ = fitted_vectorizer_and_X
        X_new = vectorizer.transform(["good morning"])
        intent, confidence = predict_intent_with_confidence(trained_classifier, X_new)
        assert 0.0 <= confidence <= 1.0

    def test_confidence_matches_highest_probability(self, fitted_vectorizer_and_X, trained_classifier):
        vectorizer, _ = fitted_vectorizer_and_X
        X_new = vectorizer.transform(["good morning"])
        intent, confidence = predict_intent_with_confidence(trained_classifier, X_new)

        probs = trained_classifier.predict_proba(X_new)[0]
        max_prob = max(probs)
        assert confidence == pytest.approx(max_prob)

    def test_predicted_intent_matches_predict(self, fitted_vectorizer_and_X, trained_classifier):
        vectorizer, _ = fitted_vectorizer_and_X
        X_new = vectorizer.transform(["good morning"])
        intent, _ = predict_intent_with_confidence(trained_classifier, X_new)
        assert intent == predict_intent(trained_classifier, X_new)

    def test_batch_confidence_returns_list_of_tuples(self, fitted_vectorizer_and_X, trained_classifier):
        vectorizer, _ = fitted_vectorizer_and_X
        X_new = vectorizer.transform(["good morning", "library hours"])
        results = predict_intent_with_confidence(trained_classifier, X_new)
        assert isinstance(results, list)
        assert len(results) == 2
        for intent, confidence in results:
            assert isinstance(intent, str)
            assert 0.0 <= confidence <= 1.0


class TestThreshold:
    def test_high_confidence_returns_predicted_intent(self, fitted_vectorizer_and_X, trained_classifier):
        vectorizer, _ = fitted_vectorizer_and_X
        # Very close match to training data -> should be high confidence.
        X_new = vectorizer.transform(["hello there"])
        result = predict_intent_with_threshold(trained_classifier, X_new, threshold=0.0)
        assert result != "fallback"

    def test_impossible_threshold_forces_fallback(self, fitted_vectorizer_and_X, trained_classifier):
        vectorizer, _ = fitted_vectorizer_and_X
        X_new = vectorizer.transform(["hello there"])
        # With only 3 classes and soft probabilities, confidence is virtually
        # never >= 0.999999, so fallback should be triggered deterministically.
        result = predict_intent_with_threshold(trained_classifier, X_new, threshold=0.999999)
        assert result == "fallback"

    def test_invalid_threshold_raises(self, fitted_vectorizer_and_X, trained_classifier):
        vectorizer, _ = fitted_vectorizer_and_X
        X_new = vectorizer.transform(["hello there"])
        with pytest.raises(ValueError):
            predict_intent_with_threshold(trained_classifier, X_new, threshold=1.5)

    def test_custom_fallback_label(self, fitted_vectorizer_and_X, trained_classifier):
        vectorizer, _ = fitted_vectorizer_and_X
        X_new = vectorizer.transform(["hello there"])
        result = predict_intent_with_threshold(
            trained_classifier, X_new, threshold=0.999999, fallback="unknown"
        )
        assert result == "unknown"


class TestSaveLoad:
    def test_save_and_load_round_trip(self, trained_classifier, tmp_path):
        path = tmp_path / "intent_classifier.pkl"
        save_classifier(trained_classifier, path)
        assert path.exists()

        loaded = load_classifier(path)
        assert set(loaded.classes_) == set(trained_classifier.classes_)

    def test_loaded_classifier_predicts_same(self, fitted_vectorizer_and_X, trained_classifier, tmp_path):
        vectorizer, _ = fitted_vectorizer_and_X
        X_new = vectorizer.transform(["hi there"])

        path = tmp_path / "intent_classifier.pkl"
        save_classifier(trained_classifier, path)
        loaded = load_classifier(path)

        assert predict_intent(loaded, X_new) == predict_intent(trained_classifier, X_new)

    def test_save_untrained_classifier_raises(self, tmp_path):
        classifier = create_classifier()
        path = tmp_path / "untrained.pkl"
        with pytest.raises(ValueError):
            save_classifier(classifier, path)

    def test_load_missing_file_raises(self, tmp_path):
        path = tmp_path / "does_not_exist.pkl"
        with pytest.raises(FileNotFoundError):
            load_classifier(path)