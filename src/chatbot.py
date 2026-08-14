"""
chatbot.py

Integration/orchestration layer for the Chatbot with NLP project.

This module does NOT implement any new NLP, machine learning, or business
logic. It only wires together the already-implemented, validated
components:

    src.preprocessing        -> preprocess_text()
    src.feature_extraction   -> load_vectorizer(), transform_text()
    src.classifier           -> load_classifier(), predict_intent_with_confidence()
    src.context_handler      -> create_context(), resolve_context(), set_context(), ...
    src.response_generator   -> load_responses(), generate_response()

Runtime architecture
---------------------
    User message
        |
        v
    preprocessing.preprocess_text()
        |
        v
    feature_extraction.transform_text()   (already-fitted TF-IDF vectorizer)
        |
        v
    classifier.predict_intent_with_confidence()  (real predicted_intent, real probability)
        |
        v
    Chatbot-level confidence threshold check
        |
        v
    context_handler.resolve_context()     (follow-up detection / intent resolution)
        |
        v
    response_generator.generate_response()
        |
        v
    context_handler.set_context()         (update context for next turn)
        |
        v
    Final response string

Model loading
-------------
`chatbot.py` never trains a model. It only LOADS the artifacts already
produced by `src/train.py`:

    models/tfidf_vectorizer.pkl
    models/intent_classifier.pkl

If those files are missing or invalid, a clear, actionable error is
raised at initialization time rather than silently creating fake models.

Confidence threshold: integration-level, NOT the classifier's default
---------------------------------------------------------------------
`src/classifier.py` exposes `predict_intent_with_threshold()` with its own
default threshold of 0.50. That default is left completely unchanged in
this integration -- `classifier.py` is not modified in any way.

However, 0.50 is not a suitable operating threshold for the CURRENT,
real, trained model. As observed directly from `src/train.py`'s own
sanity-check output (see docstring below and the project's training
run), the current model:

    - has 16 intent classes
    - is trained on 138 total patterns (an average of well under 10
      patterns per class)
    - produces genuine, unmodified `predict_proba()`-based confidence
      values on KNOWN, real dataset examples in roughly the 0.25-0.36
      range (e.g. "Hello" -> greeting (0.25), "What is the fee
      structure?" -> fees (0.36), "What are the library timings?" ->
      library (0.33)).

This is an entirely expected consequence of a 16-way softmax-style
probability distribution fit on a small dataset: with 16 classes, a
uniform/no-information probability would be ~0.0625, so confidences in
the 0.25-0.36 range represent the model being meaningfully more
confident than chance, even though they fall well short of 0.50.

Given that, this module introduces its own, SEPARATE, configurable
`confidence_threshold` on the `Chatbot` class, defaulted to 0.20. This
value is:

    - an INTEGRATION-LEVEL decision, made in `chatbot.py` only.
    - selected by inspecting the CURRENT model's observed confidence
      range on real dataset examples (~0.25-0.36), so that ordinary,
      correctly-classified known inputs are not incorrectly routed to
      fallback just because the classifier's own 0.50 default doesn't
      fit a 16-class / 138-pattern model.
    - NOT scientifically derived, NOT claimed to be optimal, and NOT a
      replacement for proper threshold tuning against a held-out
      validation set. It is a pragmatic default that can and should be
      revisited once more data/validation is available.
    - fully overridable via `Chatbot(confidence_threshold=...)`.

Crucially, this module does NOT re-derive, normalize, rescale, or
fabricate confidence in any way. It calls the existing
`predict_intent_with_confidence()` to obtain the real predicted intent
and the real, unmodified `predict_proba()`-based probability, and only
then compares that real probability against the chatbot's own
`confidence_threshold`. `predict_intent_with_threshold()` (which bakes in
a threshold internally and only returns the resulting label) is
intentionally NOT used here, because the chatbot's response flow needs
to retain the actual confidence value for its own decisions and for
callers/tests -- calling the threshold-baking helper would discard that
value.

Typical usage
-------------
    chatbot = Chatbot()
    response = chatbot.respond("What courses do you offer?")
    chatbot.reset_context()
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from src.preprocessing import preprocess_text
from src.feature_extraction import load_vectorizer, transform_text
from src.classifier import load_classifier, predict_intent_with_confidence
from src.context_handler import (
    create_context,
    set_context,
    resolve_context,
    clear_context,
    ConversationContext,
)
from src.response_generator import (
    load_responses,
    generate_response,
    DEFAULT_INTENTS_PATH,
    FALLBACK_TAG,
)

# Default artifact locations, relative to the project root. These match
# the paths already used/produced by src/train.py.
DEFAULT_MODELS_DIR = Path("models")
DEFAULT_VECTORIZER_PATH = DEFAULT_MODELS_DIR / "tfidf_vectorizer.pkl"
DEFAULT_CLASSIFIER_PATH = DEFAULT_MODELS_DIR / "intent_classifier.pkl"

# Chatbot-level (integration-level) default confidence threshold.
#
# NOT the same as classifier.py's own 0.50 default used by
# predict_intent_with_threshold(). See the module docstring above for
# the full rationale. In short: 0.20 was chosen by inspecting the
# CURRENT model's real, observed confidence on known dataset examples
# (~0.25-0.36 for a 16-class / 138-pattern model), so that correctly
# classified, real inputs are not routed to fallback purely because a
# threshold designed for a different (e.g. larger, more separable)
# dataset doesn't fit this one. This is a pragmatic, documented
# integration choice -- not a scientifically tuned or claimed-optimal
# value.
DEFAULT_CONFIDENCE_THRESHOLD = 0.20


class Chatbot:
    """Orchestrates the full chatbot pipeline using existing, validated
    components.

    This class does not implement any NLP/ML logic itself. It loads the
    already-trained TF-IDF vectorizer and classifier, loads the response
    data from `data/intents.json`, and on each `respond()` call routes
    the user's message through preprocessing -> vectorization ->
    classification -> context resolution -> response generation, in that
    order, using only the existing module APIs.
    """

    def __init__(
        self,
        vectorizer_path: Union[str, Path] = DEFAULT_VECTORIZER_PATH,
        classifier_path: Union[str, Path] = DEFAULT_CLASSIFIER_PATH,
        intents_path: Union[str, Path] = DEFAULT_INTENTS_PATH,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    ) -> None:
        """Load all trained artifacts and initialize an empty context.

        Args:
            vectorizer_path: Path to the saved, fitted TF-IDF vectorizer
                (produced by src.train / src.feature_extraction.save_vectorizer).
            classifier_path: Path to the saved, trained classifier
                (produced by src.train / src.classifier.save_classifier).
            intents_path: Path to data/intents.json, used to load the real
                response data via src.response_generator.load_responses().
            confidence_threshold: The chatbot-level (integration-level)
                minimum confidence required to trust the classifier's
                prediction. See module docstring for full rationale.
                Defaults to 0.20. Must be between 0.0 and 1.0.

        Raises:
            FileNotFoundError: If the vectorizer or classifier model file
                does not exist at the given path. The underlying
                FileNotFoundError from src.feature_extraction.load_vectorizer
                / src.classifier.load_classifier is allowed to propagate
                with its original, already-clear message.
            ValueError: If `confidence_threshold` is not within [0.0, 1.0].
        """
        if not (0.0 <= confidence_threshold <= 1.0):
            raise ValueError(
                f"confidence_threshold must be between 0 and 1, got "
                f"{confidence_threshold}."
            )

        self.confidence_threshold = confidence_threshold

        # Load trained artifacts. No training happens here -- these
        # calls only deserialize already-fitted/trained objects produced
        # by src/train.py. Missing files raise a clear FileNotFoundError
        # via the existing load_vectorizer()/load_classifier() helpers.
        self.vectorizer = load_vectorizer(vectorizer_path)
        self.classifier = load_classifier(classifier_path)

        # Load real response data from data/intents.json. Raises
        # FileNotFoundError/ValueError via the existing load_responses()
        # helper if the file is missing or malformed.
        self.responses_map = load_responses(intents_path)

        # Start with a fresh, empty conversation context.
        self.context: ConversationContext = create_context()

    def respond(self, user_input: Optional[str]) -> str:
        """Process a single user message and return the chatbot's reply.

        Pipeline (each stage delegates to the existing, validated module
        responsible for it):

            1. Validate input.
            2. preprocess_text() (src.preprocessing)
            3. transform_text() (src.feature_extraction) using the
               already-fitted vectorizer.
            4. predict_intent_with_confidence() (src.classifier) to get
               the real predicted intent and the real, unmodified
               predict_proba()-based confidence.
            5. Compare that real confidence against this chatbot's own
               `confidence_threshold`. Below threshold -> treat as the
               dataset's "fallback" intent tag; at/above threshold ->
               keep the real predicted intent.
            6. resolve_context() (src.context_handler) to let a real
               follow-up question reuse the stored previous intent when
               appropriate.
            7. generate_response() (src.response_generator) to pick an
               actual response from data/intents.json for the resolved
               intent.
            8. set_context() (src.context_handler) to update context for
               the next turn, using the newly resolved intent.

        Args:
            user_input: The raw user message. May be None, empty, or
                whitespace-only.

        Returns:
            A response string. For invalid input (None, empty, or
            whitespace-only), a real fallback response drawn from
            data/intents.json's own "fallback" intent is returned via
            generate_response(), rather than raising an exception --
            this keeps the input-validation behavior consistent with the
            rest of the pipeline's "never fabricate, always use the real
            dataset fallback" convention.

        Raises:
            TypeError: If `user_input` is not a string and not None
                (e.g. an int or list), matching the type-checking
                convention already used elsewhere in this project (see
                src.feature_extraction._validate_texts).
        """
        if user_input is not None and not isinstance(user_input, str):
            raise TypeError(
                f"user_input must be a string or None, got "
                f"{type(user_input).__name__}."
            )

        # Empty / whitespace-only / None input never reaches the
        # classifier -- there is nothing meaningful to classify. Use the
        # dataset's own fallback response directly.
        if not user_input or not user_input.strip():
            return generate_response(None, self.responses_map)

        # 1. Preprocessing (existing module).
        processed = preprocess_text(user_input)

        # 2. TF-IDF vectorization using the already-fitted vectorizer
        #    (existing module; never re-fit here).
        features = transform_text(self.vectorizer, [processed])

        # 3. Classification with REAL, unmodified confidence. This is
        #    the actual predict_proba()-based value -- never fabricated
        #    or normalized.
        predicted_intent, confidence = predict_intent_with_confidence(
            self.classifier, features
        )

        # 4. Apply the chatbot's own (not classifier.py's) threshold to
        #    the real confidence value. We intentionally do NOT call
        #    predict_intent_with_threshold() here, since that would only
        #    hand back a label and silently discard the actual
        #    confidence this chatbot needs to retain.
        if confidence >= self.confidence_threshold:
            classifier_intent = predicted_intent
        else:
            classifier_intent = FALLBACK_TAG

        # 5. Context resolution (existing module). A genuine follow-up
        #    (per context_handler.is_follow_up) with existing stored
        #    context takes precedence over the classifier's result;
        #    otherwise the classifier's (possibly fallback) result is
        #    used as-is.
        resolved_intent = resolve_context(
            self.context, user_input, predicted_intent=classifier_intent
        )

        # 6. Response generation from the real dataset (existing
        #    module). Any unknown/empty/None resolved_intent is handled
        #    internally by generate_response() via the dataset's own
        #    fallback responses.
        response = generate_response(resolved_intent, self.responses_map)

        # 7. Update context for the next turn using the resolved intent
        #    (existing module). set_context() itself no-ops on an
        #    empty/None intent, preserving prior context rather than
        #    erasing it.
        self.context = set_context(self.context, resolved_intent, user_input=user_input)

        return response

    def reset_context(self) -> None:
        """Clear the conversation context, starting a fresh session.

        Delegates entirely to src.context_handler.clear_context(); does
        not implement its own context-clearing logic.
        """
        self.context = clear_context(self.context)