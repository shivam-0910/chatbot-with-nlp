"""
tests/test_response_generator.py

Tests for src/response_generator.py.

Uses the actual data/intents.json dataset wherever practical. Because
response selection is randomized for intents with multiple responses,
tests check *membership* in the real response list rather than an exact
string match.
"""

from pathlib import Path

import pytest

from src.response_generator import (
    DEFAULT_INTENTS_PATH,
    FALLBACK_TAG,
    load_responses,
    generate_response,
)


@pytest.fixture(scope="module")
def responses_map():
    return load_responses(DEFAULT_INTENTS_PATH)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def test_load_responses_from_real_dataset(responses_map):
    assert isinstance(responses_map, dict)
    assert len(responses_map) > 0


def test_all_discovered_tags_have_nonempty_response_lists(responses_map):
    for tag, responses in responses_map.items():
        assert isinstance(responses, list)
        assert len(responses) > 0


def test_fallback_tag_present_in_real_dataset(responses_map):
    assert FALLBACK_TAG in responses_map


def test_load_responses_missing_file_raises(tmp_path):
    missing_path = tmp_path / "does_not_exist.json"
    with pytest.raises(FileNotFoundError):
        load_responses(missing_path)


# ---------------------------------------------------------------------------
# Basic response generation
# ---------------------------------------------------------------------------

def test_known_intent_returns_a_response(responses_map):
    response = generate_response("greeting", responses_map)
    assert isinstance(response, str)
    assert response


def test_returned_response_belongs_to_intent(responses_map):
    response = generate_response("fees", responses_map)
    assert response in responses_map["fees"]


def test_multiple_calls_return_valid_responses(responses_map):
    for _ in range(20):
        response = generate_response("courses", responses_map)
        assert response in responses_map["courses"]


# ---------------------------------------------------------------------------
# Multiple-response intent
# ---------------------------------------------------------------------------

def test_multi_response_intent_can_return_different_responses(responses_map):
    # Not guaranteed on any single run, but with enough draws from a real
    # multi-response intent we should see more than one distinct value.
    intent = "greeting"
    assert len(responses_map[intent]) > 1

    seen = {generate_response(intent, responses_map) for _ in range(50)}
    assert seen.issubset(set(responses_map[intent]))
    assert len(seen) > 1


# ---------------------------------------------------------------------------
# Unknown intent / fallback
# ---------------------------------------------------------------------------

def test_unknown_intent_returns_fallback_response(responses_map):
    response = generate_response("not_a_real_intent", responses_map)
    assert response in responses_map[FALLBACK_TAG]


def test_fallback_intent_uses_real_dataset_fallback_responses(responses_map):
    response = generate_response(FALLBACK_TAG, responses_map)
    assert response in responses_map[FALLBACK_TAG]


def test_missing_fallback_intent_uses_minimal_fallback():
    responses_map_without_fallback = {"greeting": ["Hi!"]}
    response = generate_response("unknown_intent", responses_map_without_fallback)
    assert isinstance(response, str)
    assert response


# ---------------------------------------------------------------------------
# Empty / invalid input
# ---------------------------------------------------------------------------

def test_empty_string_intent_returns_fallback(responses_map):
    response = generate_response("", responses_map)
    assert response in responses_map[FALLBACK_TAG]


def test_none_intent_returns_fallback(responses_map):
    response = generate_response(None, responses_map)
    assert response in responses_map[FALLBACK_TAG]


def test_empty_responses_map_raises():
    with pytest.raises(ValueError):
        generate_response("greeting", {})


# ---------------------------------------------------------------------------
# Non-mutation
# ---------------------------------------------------------------------------

def test_generate_response_does_not_mutate_original_list(responses_map):
    original = list(responses_map["fees"])
    for _ in range(10):
        generate_response("fees", responses_map)
    assert responses_map["fees"] == original