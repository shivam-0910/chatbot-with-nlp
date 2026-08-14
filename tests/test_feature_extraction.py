"""Tests for src/feature_extraction.py."""

import sys
from pathlib import Path

import pytest
from scipy.sparse import csr_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.feature_extraction import (
    create_vectorizer,
    fit_vectorizer,
    transform_text,
    fit_transform_texts,
    save_vectorizer,
    load_vectorizer,
)


TRAIN_TEXTS = [
    "computer science course",
    "mechanical engineering course",
    "computer science admission",
]


class TestBasicTfidf:
    def test_fit_transform_texts_succeeds(self):
        vectorizer, X = fit_transform_texts(TRAIN_TEXTS)
        assert vectorizer is not None
        assert X is not None

    def test_output_row_count_matches_input(self):
        _, X = fit_transform_texts(TRAIN_TEXTS)
        assert X.shape[0] == len(TRAIN_TEXTS)

    def test_output_is_sparse_matrix(self):
        _, X = fit_transform_texts(TRAIN_TEXTS)
        assert isinstance(X, csr_matrix)

    def test_vocabulary_is_non_empty(self):
        vectorizer, _ = fit_transform_texts(TRAIN_TEXTS)
        assert len(vectorizer.get_feature_names_out()) > 0

    def test_fit_vectorizer_returns_fitted_instance(self):
        vectorizer = create_vectorizer()
        fitted = fit_vectorizer(vectorizer, TRAIN_TEXTS)
        assert hasattr(fitted, "vocabulary_")


class TestNewTextTransformation:
    def test_transform_unseen_text_succeeds(self):
        vectorizer, X_train = fit_transform_texts(TRAIN_TEXTS)
        X_new = transform_text(vectorizer, ["computer science research"])
        assert X_new is not None

    def test_column_count_matches_training(self):
        vectorizer, X_train = fit_transform_texts(TRAIN_TEXTS)
        X_new = transform_text(vectorizer, ["computer science research"])
        assert X_new.shape[1] == X_train.shape[1]

    def test_vocabulary_unchanged_after_transform(self):
        vectorizer, _ = fit_transform_texts(TRAIN_TEXTS)
        vocab_before = dict(vectorizer.vocabulary_)
        transform_text(vectorizer, ["a completely new unseen sentence here"])
        assert vectorizer.vocabulary_ == vocab_before

    def test_transform_raises_if_not_fitted(self):
        vectorizer = create_vectorizer()
        with pytest.raises(ValueError):
            transform_text(vectorizer, ["computer science"])


class TestEmptyInput:
    def test_fit_on_empty_list_raises_clear_error(self):
        with pytest.raises(ValueError):
            fit_transform_texts([])

    def test_transform_empty_list_returns_empty_matrix(self):
        vectorizer, _ = fit_transform_texts(TRAIN_TEXTS)
        X_empty = transform_text(vectorizer, [])
        assert X_empty.shape[0] == 0
        assert X_empty.shape[1] == len(vectorizer.get_feature_names_out())

    def test_none_input_raises_type_error(self):
        vectorizer, _ = fit_transform_texts(TRAIN_TEXTS)
        with pytest.raises(TypeError):
            transform_text(vectorizer, None)

    def test_non_string_item_raises_type_error(self):
        vectorizer, _ = fit_transform_texts(TRAIN_TEXTS)
        with pytest.raises(TypeError):
            transform_text(vectorizer, [123])


class TestSaveLoad:
    def test_save_and_load_round_trip(self, tmp_path):
        vectorizer, _ = fit_transform_texts(TRAIN_TEXTS)
        path = tmp_path / "tfidf_vectorizer.pkl"
        save_vectorizer(vectorizer, path)
        assert path.exists()

        loaded = load_vectorizer(path)
        assert loaded.vocabulary_ == vectorizer.vocabulary_

    def test_loaded_vectorizer_produces_equivalent_features(self, tmp_path):
        vectorizer, X_original = fit_transform_texts(TRAIN_TEXTS)
        path = tmp_path / "tfidf_vectorizer.pkl"
        save_vectorizer(vectorizer, path)
        loaded = load_vectorizer(path)

        X_loaded = transform_text(loaded, TRAIN_TEXTS)
        assert (X_original != X_loaded).nnz == 0

    def test_save_unfitted_vectorizer_raises(self, tmp_path):
        vectorizer = create_vectorizer()
        path = tmp_path / "unfitted.pkl"
        with pytest.raises(ValueError):
            save_vectorizer(vectorizer, path)

    def test_load_missing_file_raises(self, tmp_path):
        path = tmp_path / "does_not_exist.pkl"
        with pytest.raises(FileNotFoundError):
            load_vectorizer(path)


class TestReproducibility:
    def test_repeated_transform_is_consistent(self):
        vectorizer, _ = fit_transform_texts(TRAIN_TEXTS)
        X1 = transform_text(vectorizer, ["computer science course"])
        X2 = transform_text(vectorizer, ["computer science course"])
        assert (X1 != X2).nnz == 0