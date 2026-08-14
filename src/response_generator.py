"""
response_generator.py

Response selection for the Chatbot with NLP project.

This module is intentionally simple: it does not classify anything and
does not load any trained model. It receives an already-predicted intent
(from src.classifier) and returns one of that intent's predefined
responses from data/intents.json.

Architecture
------------
    user input
        |
        v
    classifier (elsewhere)
        |
        v
    predicted intent (a string, e.g. "fees")
        |
        v
    response_generator.generate_response()
        |
        v
    one response string, taken from intents.json

Typical usage
-------------
    responses_map = load_responses("data/intents.json")
    reply = generate_response("fees", responses_map)
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Union

DEFAULT_INTENTS_PATH = Path("data/intents.json")
FALLBACK_TAG = "fallback"

# Used only if the dataset itself has no "fallback" intent at all (an
# unexpected/degraded dataset state). This is a last-resort safety net,
# not the normal fallback path -- the normal path uses the dataset's own
# fallback responses.
_MINIMAL_FALLBACK_RESPONSE = "I'm sorry, I didn't understand that."


def load_responses(intents_path: Union[str, Path] = DEFAULT_INTENTS_PATH) -> Dict[str, List[str]]:
    """Load the intent -> responses mapping from intents.json.

    Returns a dict like {"greeting": [...], "fees": [...], ...}. Does not
    modify or duplicate the underlying dataset validation logic beyond
    what's needed to build this mapping safely.
    """
    intents_path = Path(intents_path)
    if not intents_path.exists():
        raise FileNotFoundError(f"Could not find intents file at: {intents_path}")

    with open(intents_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "intents" not in data or not isinstance(data["intents"], list):
        raise ValueError("intents.json must contain a top-level 'intents' list.")

    responses_map: Dict[str, List[str]] = {}
    for intent in data["intents"]:
        tag = intent.get("tag")
        responses = intent.get("responses")
        if tag and responses:
            responses_map[tag] = list(responses)

    return responses_map


def generate_response(intent: str, responses_map: Dict[str, List[str]]) -> str:
    """Return one response belonging to `intent`.

    If `intent` has multiple responses, one is chosen at random using the
    standard-library `random` module. If `intent` is unknown, empty, or
    otherwise invalid, falls back to the dataset's own "fallback" intent
    responses. If even that is missing, a minimal hard-coded message is
    used as a last resort.

    The original response lists in `responses_map` are never mutated.
    """
    if not responses_map:
        raise ValueError("responses_map is empty; cannot generate a response.")

    if intent and isinstance(intent, str) and intent in responses_map:
        return random.choice(responses_map[intent])

    # Unknown/empty/invalid intent -> use the dataset's own fallback intent.
    if FALLBACK_TAG in responses_map:
        return random.choice(responses_map[FALLBACK_TAG])

    # Last-resort safety net if the dataset has no fallback intent at all.
    return _MINIMAL_FALLBACK_RESPONSE