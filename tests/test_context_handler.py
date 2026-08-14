"""Tests for src/context_handler.py."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.context_handler import (
    create_context,
    set_context,
    get_context,
    get_last_user_input,
    clear_context,
    is_follow_up,
    resolve_context,
)


# ---------------------------------------------------------------------------
# Context storage
# ---------------------------------------------------------------------------

class TestContextStorage:
    def test_initial_context_has_no_intent(self):
        context = create_context()
        assert get_context(context) is None

    def test_initial_context_has_no_last_user_input(self):
        context = create_context()
        assert get_last_user_input(context) is None

    def test_setting_intent_works(self):
        context = create_context()
        context = set_context(context, "courses")
        assert get_context(context) == "courses"

    def test_retrieving_intent_returns_stored_value(self):
        context = create_context()
        context = set_context(context, "fees", user_input="what are the fees")
        assert get_context(context) == "fees"
        assert get_last_user_input(context) == "what are the fees"

    def test_updating_intent_replaces_previous_value(self):
        context = create_context()
        context = set_context(context, "courses")
        context = set_context(context, "fees")
        assert get_context(context) == "fees"

    def test_clearing_context_works(self):
        context = create_context()
        context = set_context(context, "library")
        context = clear_context(context)
        assert get_context(context) is None
        assert get_last_user_input(context) is None

    def test_set_context_does_not_mutate_original(self):
        context = create_context()
        new_context = set_context(context, "greeting")
        assert get_context(context) is None
        assert get_context(new_context) == "greeting"


# ---------------------------------------------------------------------------
# Follow-up detection
# ---------------------------------------------------------------------------

class TestFollowUpDetection:
    @pytest.mark.parametrize(
        "text",
        [
            "What about CSE?",
            "What about it?",
            "Tell me more.",
            "And the fees?",
            "What about that?",
            "How about the library?",
            "What else can you tell me?",
        ],
    )
    def test_obvious_follow_ups_detected(self, text):
        assert is_follow_up(text) is True

    @pytest.mark.parametrize(
        "text",
        [
            "What courses do you offer?",
            "How do I apply for admission?",
            "Hello",
            "What is the fee structure?",
            "Where is the library located?",
        ],
    )
    def test_non_follow_ups_not_detected(self, text):
        assert is_follow_up(text) is False

    def test_empty_string_is_not_follow_up(self):
        assert is_follow_up("") is False

    def test_none_is_not_follow_up(self):
        assert is_follow_up(None) is False

    def test_whitespace_only_is_not_follow_up(self):
        assert is_follow_up("   ") is False

    def test_case_insensitive(self):
        assert is_follow_up("WHAT ABOUT CSE?") is True


# ---------------------------------------------------------------------------
# Context resolution
# ---------------------------------------------------------------------------

class TestContextResolution:
    def test_follow_up_with_existing_context_returns_previous_intent(self):
        context = create_context()
        context = set_context(context, "courses")
        resolved = resolve_context(context, "What about CSE?", predicted_intent="fallback")
        assert resolved == "courses"

    def test_follow_up_with_no_context_does_not_invent_intent(self):
        context = create_context()
        resolved = resolve_context(context, "What about CSE?", predicted_intent="fallback")
        assert resolved == "fallback"

    def test_follow_up_with_no_context_and_no_predicted_intent_returns_none(self):
        context = create_context()
        resolved = resolve_context(context, "What about CSE?", predicted_intent=None)
        assert resolved is None

    def test_normal_input_uses_predicted_intent_not_previous_context(self):
        context = create_context()
        context = set_context(context, "courses")
        resolved = resolve_context(context, "What is the fee structure?", predicted_intent="fees")
        assert resolved == "fees"

    def test_normal_input_with_no_context_uses_predicted_intent(self):
        context = create_context()
        resolved = resolve_context(context, "Hello", predicted_intent="greeting")
        assert resolved == "greeting"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_input_to_resolve_context(self):
        context = create_context()
        context = set_context(context, "courses")
        resolved = resolve_context(context, "", predicted_intent="fallback")
        assert resolved == "fallback"

    def test_none_input_to_resolve_context(self):
        context = create_context()
        context = set_context(context, "courses")
        resolved = resolve_context(context, None, predicted_intent="fallback")
        assert resolved == "fallback"

    def test_clearing_already_empty_context(self):
        context = create_context()
        cleared = clear_context(context)
        assert get_context(cleared) is None

    def test_setting_empty_intent_does_not_overwrite_existing(self):
        context = create_context()
        context = set_context(context, "courses")
        context = set_context(context, "")
        assert get_context(context) == "courses"

    def test_setting_none_intent_does_not_overwrite_existing(self):
        context = create_context()
        context = set_context(context, "courses")
        context = set_context(context, None)
        assert get_context(context) == "courses"

    def test_setting_empty_intent_on_empty_context_stays_empty(self):
        context = create_context()
        context = set_context(context, "")
        assert get_context(context) is None