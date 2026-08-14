"""
context_handler.py

Lightweight conversation context management for the Chatbot with NLP
project.

This module does NOT classify intents and does NOT generate responses.
Those responsibilities belong to src.classifier and
src.response_generator, respectively. This module only tracks the most
recently known intent and offers a simple, rule-based way to decide
whether a new user input is a follow-up that should reuse that intent.

Architecture
------------
    user input
        |
        v
    preprocessing -> TF-IDF -> classifier -> predicted intent
        |
        v
    context_handler.resolve_context(user_input, predicted_intent)
        |
        v
    response_generator

Typical usage
-------------
    context = create_context()
    context = set_context(context, "courses", user_input="What courses do you offer?")

    # Later, on a follow-up message:
    resolved_intent = resolve_context(context, "What about CSE?", predicted_intent="fallback")

Design notes
------------
- Context is intentionally tiny: just the last intent (and optionally the
  last user input that produced it). No conversation history is kept.
- Follow-up detection is a simple, deterministic, rule-based check
  against a small list of known follow-up phrases/patterns. It does not
  attempt any real NLP and does not know about specific college facts.
- resolve_context() never invents a previous intent: if there is no
  stored context, a follow-up input simply resolves to whatever the
  caller passed in as the current/predicted intent (which may be None).
"""

from __future__ import annotations

import re
from typing import Dict, Optional, TypedDict


class ConversationContext(TypedDict):
    last_intent: Optional[str]
    last_user_input: Optional[str]


# A small set of rule-based patterns for detecting obvious follow-up
# questions. Kept intentionally simple -- no NLP, no college-specific
# vocabulary. Patterns are matched against the lowercased, stripped input.
_FOLLOW_UP_PATTERNS = [
    r"^what about\b",
    r"^and\b",
    r"^what about it\??$",
    r"^what about that\??$",
    r"tell me more\b",
    r"^more\??$",
    r"^and\s+the\b",
    r"^what else\b",
    r"^anything else\b",
    r"^what if\b",
    r"^how about\b",
    r"^and what about\b",
]

_FOLLOW_UP_REGEXES = [re.compile(pattern) for pattern in _FOLLOW_UP_PATTERNS]


def create_context() -> ConversationContext:
    """Create a new, empty conversation context.

    Returns a dict with `last_intent` and `last_user_input` both set to
    None. This is the starting state before any intent has been seen.
    """
    return {"last_intent": None, "last_user_input": None}


def set_context(
    context: ConversationContext,
    intent: Optional[str],
    user_input: Optional[str] = None,
) -> ConversationContext:
    """Return a new context with `last_intent` (and optionally
    `last_user_input`) updated.

    Does not mutate the passed-in context; returns a new dict.

    An empty/None `intent` is treated as "nothing to store" and does not
    overwrite the existing context -- it is returned unchanged (aside
    from a normalized copy). This avoids accidentally erasing a valid
    previous intent because of a bad/empty classifier result; use
    clear_context() to intentionally reset.
    """
    if not intent:
        return dict(context)  # type: ignore[return-value]

    return {"last_intent": intent, "last_user_input": user_input}


def get_context(context: ConversationContext) -> Optional[str]:
    """Return the stored last intent, or None if there isn't one."""
    if not context:
        return None
    return context.get("last_intent")


def get_last_user_input(context: ConversationContext) -> Optional[str]:
    """Return the stored last user input, or None if there isn't one."""
    if not context:
        return None
    return context.get("last_user_input")


def clear_context(context: ConversationContext) -> ConversationContext:
    """Return a fresh, empty context. Safe to call on an already-empty
    context."""
    return create_context()


def is_follow_up(text: Optional[str]) -> bool:
    """Return True if `text` looks like a short follow-up question.

    This is a simple, deterministic, rule-based check against a small
    list of common follow-up phrases/patterns (e.g. "What about CSE?",
    "Tell me more.", "And the fees?"). It does not perform any real NLP
    and knows nothing about specific college facts.

    Returns False for None or empty/whitespace-only input.
    """
    if not text or not text.strip():
        return False

    normalized = text.strip().lower()
    normalized = normalized.rstrip("?!.")

    for regex in _FOLLOW_UP_REGEXES:
        if regex.search(normalized):
            return True

    return False


def resolve_context(
    context: ConversationContext,
    user_input: Optional[str],
    predicted_intent: Optional[str] = None,
) -> Optional[str]:
    """Decide which intent should actually be used for this turn.

    Behavior:
    - If `user_input` looks like a follow-up (per is_follow_up) AND the
      context has a stored `last_intent`, return that stored intent.
    - Otherwise, return `predicted_intent` as-is (this covers: normal,
      non-follow-up input; a follow-up input with no prior context, in
      which case we do NOT invent a previous intent; and any other
      case).

    This function never classifies text itself -- `predicted_intent` is
    expected to already come from src.classifier (or be None/unknown,
    e.g. "fallback").
    """
    last_intent = get_context(context)

    if is_follow_up(user_input) and last_intent:
        return last_intent

    return predicted_intent