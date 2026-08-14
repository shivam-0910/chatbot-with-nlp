"""
Unit tests for src/preprocessing.py

Run with:
    pytest tests/test_preprocessing.py -v
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocessing import (
    clean_text,
    tokenize_text,
    remove_stopwords,
    stem_tokens,
    lemmatize_tokens,
    preprocess_text,
)


# ---------- clean_text ----------

def test_clean_text_lowercases():
    assert clean_text("Hello, How Are You?") == "hello how are you"


def test_clean_text_removes_punctuation():
    result = clean_text("hello!!! how are you?")
    assert "!" not in result
    assert "?" not in result
    assert result == "hello how are you"


def test_clean_text_collapses_whitespace():
    result = clean_text("hello    there,   world!")
    assert result == "hello there world"


def test_clean_text_empty_input():
    assert clean_text("") == ""


def test_clean_text_none_input():
    assert clean_text(None) == ""


# ---------- tokenize_text ----------

def test_tokenize_text_basic():
    result = tokenize_text("hello how are you")
    assert result == ["hello", "how", "are", "you"]


def test_tokenize_text_empty_input():
    assert tokenize_text("") == []


def test_tokenize_text_none_input():
    assert tokenize_text(None) == []


# ---------- remove_stopwords ----------

def test_remove_stopwords_filters_common_words():
    tokens = ["what", "is", "the", "fee", "structure"]
    result = remove_stopwords(tokens)
    assert "is" not in result
    assert "the" not in result
    assert "fee" in result
    assert "structure" in result


def test_remove_stopwords_empty_input():
    assert remove_stopwords([]) == []


# ---------- stem_tokens ----------

def test_stem_tokens_reduces_to_root_form():
    tokens = ["playing", "played", "plays"]
    result = stem_tokens(tokens)
    # Porter stemmer should reduce all three to the same root
    assert len(set(result)) == 1
    assert result[0] == "play"


def test_stem_tokens_empty_input():
    assert stem_tokens([]) == []


# ---------- lemmatize_tokens ----------

def test_lemmatize_tokens_reduces_plurals():
    tokens = ["courses", "classes"]
    result = lemmatize_tokens(tokens)
    assert result == ["course", "class"]


def test_lemmatize_tokens_empty_input():
    assert lemmatize_tokens([]) == []


# ---------- preprocess_text (full pipeline) ----------

def test_preprocess_text_full_pipeline():
    result = preprocess_text("Hello!!! What courses are you offering?")
    # Should be lowercased, punctuation-free, stopwords removed, lemmatized
    assert "!" not in result
    assert "?" not in result
    assert "are" not in result.split()
    assert "you" not in result.split()
    assert "course" in result  # "courses" -> "course" after lemmatization


def test_preprocess_text_empty_input():
    assert preprocess_text("") == ""


def test_preprocess_text_none_input():
    assert preprocess_text(None) == ""  # pyright: ignore[reportArgumentType]


def test_preprocess_text_returns_string():
    result = preprocess_text("What are the library timings?")
    assert isinstance(result, str)