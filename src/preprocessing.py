"""
preprocessing.py

Text preprocessing utilities for the Chatbot with NLP project.

Pipeline stages implemented:
    1. Lowercase conversion
    2. Punctuation removal
    3. Tokenization
    4. Stopword removal
    5. Stemming (Porter Stemmer)
    6. Lemmatization (WordNet Lemmatizer)

Design note:
    Stemming and lemmatization are both required by the internship handbook,
    but chaining them (stem -> lemmatize) on the same tokens is not useful:
    a stemmer often produces truncated, non-dictionary word fragments
    (e.g. "studies" -> "studi"), and feeding that into a lemmatizer does not
    reliably recover a valid lemma. Instead, this module exposes stemming
    and lemmatization as independent, parallel operations on the same
    stopword-filtered tokens. The main pipeline (`preprocess_text`) uses
    lemmatization for the final normalized output passed to TF-IDF, since
    lemmatization produces real dictionary words and tends to work better
    for downstream feature extraction. `stem_tokens` remains available for
    direct use, testing, or comparison, satisfying the requirement that
    both techniques be implemented.
"""

import string
from typing import List

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

try:
    _STEMMER = PorterStemmer()
    _LEMMATIZER = WordNetLemmatizer()
except LookupError:
    _STEMMER = None
    _LEMMATIZER = None

_STOPWORDS: set = set()
_RESOURCES_READY = False


def download_nltk_resources() -> None:
    """
    Ensure required NLTK resources are available locally.

    Checks for each resource before downloading, so this does not hit the
    network (or fail due to no internet access) if the resources are
    already installed. Safe to call multiple times.
    """
    global _RESOURCES_READY, _STOPWORDS

    resources = {
        "corpora/stopwords": "stopwords",
        "tokenizers/punkt": "punkt",
        "tokenizers/punkt_tab": "punkt_tab",
        "corpora/wordnet": "wordnet",
        "corpora/omw-1.4": "omw-1.4",
    }

    for resource_path, package_name in resources.items():
        try:
            nltk.data.find(resource_path)
        except LookupError:
            try:
                nltk.download(package_name, quiet=True)
            except Exception as exc:
                raise RuntimeError(
                    f"Required NLTK resource '{package_name}' is not installed "
                    f"and could not be downloaded automatically (no internet "
                    f"access or download failed). Please run "
                    f"`python -m nltk.downloader {package_name}` manually."
                ) from exc

    _STOPWORDS = set(stopwords.words("english"))
    _RESOURCES_READY = True


def _ensure_ready() -> None:
    """Lazily initialize NLTK resources on first use."""
    if not _RESOURCES_READY:
        download_nltk_resources()


def clean_text(text: str) -> str:
    """
    Lowercase text and strip punctuation.

    Args:
        text: Raw input string.

    Returns:
        Cleaned text: lowercased, punctuation removed, extra whitespace
        collapsed. Returns "" for None or empty input.
    """
    if not text:
        return ""

    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = " ".join(text.split())
    return text


def tokenize_text(text: str) -> List[str]:
    """
    Split cleaned text into individual word tokens.

    Args:
        text: Cleaned (lowercased, punctuation-free) text.

    Returns:
        List of word tokens. Empty list for None or empty input.
    """
    if not text:
        return []

    _ensure_ready()
    return word_tokenize(text)


def remove_stopwords(tokens: List[str]) -> List[str]:
    """
    Remove common English stopwords from a list of tokens.

    Args:
        tokens: List of word tokens.

    Returns:
        List of tokens with stopwords removed. Empty list for empty input.
    """
    if not tokens:
        return []

    _ensure_ready()
    return [token for token in tokens if token not in _STOPWORDS]


def stem_tokens(tokens: List[str]) -> List[str]:
    """
    Reduce tokens to their root form using the Porter Stemmer.

    Args:
        tokens: List of word tokens.

    Returns:
        List of stemmed tokens. Empty list for empty input.
    """
    if not tokens:
        return []

    return [_STEMMER.stem(token) for token in tokens]


def lemmatize_tokens(tokens: List[str]) -> List[str]:
    """
    Reduce tokens to their dictionary base form using WordNet Lemmatizer.

    Args:
        tokens: List of word tokens.

    Returns:
        List of lemmatized tokens. Empty list for empty input.
    """
    if not tokens:
        return []

    _ensure_ready()
    return [_LEMMATIZER.lemmatize(token) for token in tokens]


def preprocess_text(text: str) -> str:
    """
    Run the full preprocessing pipeline on raw input text.

    Pipeline: lowercase -> remove punctuation -> tokenize -> remove
    stopwords -> lemmatize -> rejoin into a single string.

    This final string is the normalized representation intended for
    TF-IDF feature extraction downstream.

    Args:
        text: Raw input string (e.g. a user message).

    Returns:
        A single space-joined string of processed tokens. Returns "" for
        None or empty input.
    """
    if not text:
        return ""

    cleaned = clean_text(text)
    tokens = tokenize_text(cleaned)
    tokens = remove_stopwords(tokens)
    tokens = lemmatize_tokens(tokens)
    return " ".join(tokens)


if __name__ == "__main__":
    sample = "Hello!!! What courses are you offering this year?"
    print("Raw:        ", sample)
    print("Cleaned:    ", clean_text(sample))
    print("Tokenized:  ", tokenize_text(clean_text(sample)))
    print("No stopwords:", remove_stopwords(tokenize_text(clean_text(sample))))
    print("Stemmed:    ", stem_tokens(remove_stopwords(tokenize_text(clean_text(sample)))))
    print("Lemmatized: ", lemmatize_tokens(remove_stopwords(tokenize_text(clean_text(sample)))))
    print("Final:      ", preprocess_text(sample))