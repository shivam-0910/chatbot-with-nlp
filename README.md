# Chatbot with NLP

An NLP-based intent-classification chatbot built for a college helpdesk use case. It uses TF-IDF feature extraction and a Logistic Regression classifier to identify what a user is asking, then returns a matching response from a predefined intents dataset.

## Overview

This chatbot answers common college-related questions — courses, fees, admissions, attendance, library hours, and more — by classifying free-text user input into one of 16 predefined intents. The pipeline is built entirely from classical NLP and machine learning techniques (no deep learning, transformers, or LLMs).

Key behaviors:
- **Intent classification** using TF-IDF + Logistic Regression
- **Context-aware follow-ups** — a short follow-up like "What about that?" can resolve to the previous turn's intent
- **Confidence-based fallback** — low-confidence predictions are routed to a fallback response instead of a possibly-wrong answer

The final implementation is exposed as a Python class (`Chatbot` in `src/chatbot.py`) rather than a web application. No Streamlit/Flask/FastAPI interface or `app.py` is part of the final architecture.

## Features

- Text preprocessing: lowercasing, punctuation removal, tokenization, stopword removal, stemming, and lemmatization
- TF-IDF feature extraction (179 features from the current dataset)
- Logistic Regression intent classification across 16 intent classes
- Confidence-based fallback handling with a configurable threshold
- Context-aware follow-up resolution (e.g. "What about that?")
- Response generation from a real intents dataset (`data/intents.json`)
- Trained model persistence (`models/tfidf_vectorizer.pkl`, `models/intent_classifier.pkl`)
- 163 automated tests (122 unit tests + 41 chatbot integration tests), all passing

## Architecture

The actual runtime pipeline, as implemented in `src/chatbot.py`, is:

```
User Input
    ↓
Text Preprocessing (src/preprocessing.py)
    ↓
TF-IDF Feature Extraction (src/feature_extraction.py)
    ↓
Logistic Regression Intent Classification (src/classifier.py)
    ↓
Confidence Threshold Check (chatbot-level, default 0.20)
    ↓
Context / Follow-up Resolution (src/context_handler.py)
    ↓
Response Generation (src/response_generator.py)
    ↓
Final Response
```

**Stage notes:**
1. **Preprocessing** — cleans, tokenizes, removes stopwords, and lemmatizes the raw input into a normalized string.
2. **TF-IDF extraction** — transforms the normalized text into a feature vector using the already-fitted vectorizer (never re-fit at inference time).
3. **Classification** — the Logistic Regression model returns a predicted intent and its real `predict_proba()` confidence.
4. **Confidence threshold** — if confidence is below the chatbot's threshold (default `0.20`), the intent is replaced with the dataset's `fallback` tag *before* context resolution runs.
5. **Context resolution** — if the input looks like a follow-up (e.g. starts with "what about", "and", "how about") and a previous intent is stored, that previous intent is reused instead of the (possibly fallback) classifier result.
6. **Response generation** — a response is picked at random from the resolved intent's response list in `data/intents.json`.
7. Context is then updated with the resolved intent for the next turn.

## Dataset

Location: `data/intents.json`

Verified statistics (counted directly from the file):
- **16 intent classes**
- **138 total training patterns**
- Each intent has 3 predefined responses
- A dedicated `fallback` intent (7 patterns, 3 responses) is used whenever the classifier's confidence is too low or the resolved intent is unrecognized

Each intent entry has a `tag`, a list of `patterns` (example user phrasings used for training), and a list of `responses` (candidate bot replies, one chosen at random per call).

The same file is used twice: `src/train.py` reads the patterns/tags to train the model, and `src/response_generator.py` reads the tags/responses at runtime to generate replies.

## Machine Learning Pipeline

1. **Data loading** — `src/train.py` loads and validates `data/intents.json` (every intent must have a tag and at least one pattern).
2. **Preprocessing** — every training pattern is passed through `preprocess_text()`.
3. **TF-IDF fitting** — a `TfidfVectorizer` is fit once on the full set of preprocessed patterns (`fit_transform_texts()`).
4. **Classifier training** — a Logistic Regression classifier (`max_iter=1000`, `random_state=42`) is trained on the resulting feature matrix.
5. **Model serialization** — the fitted vectorizer and trained classifier are pickled to `models/tfidf_vectorizer.pkl` and `models/intent_classifier.pkl`.
6. **Prediction** — at inference time, the already-fitted vectorizer transforms new text (never refit), and the classifier returns a predicted intent and its `predict_proba()`-based confidence.
7. **Confidence handling** — the chatbot compares that real confidence against its own configurable threshold before deciding whether to trust the prediction.

## Model

- **Algorithm:** Logistic Regression (`sklearn.linear_model.LogisticRegression`)
- **Artifacts:** `models/tfidf_vectorizer.pkl`, `models/intent_classifier.pkl`, produced by `src/train.py`
- **Prediction/confidence:** `predict_intent_with_confidence()` returns the predicted label and the corresponding class probability from `predict_proba()` — this value is never modified, normalized, or fabricated.

**On accuracy:** there is currently no formal held-out evaluation (no train/test split, no accuracy/precision/recall/F1 metric, no confusion matrix). The 163 automated tests validate implementation behavior and pipeline correctness — not model accuracy. Any claim of a specific accuracy percentage would not be supported by the current implementation.

## Context Handling

Implemented in `src/context_handler.py`:
- The chatbot stores only the **most recent** resolved intent and the user input that produced it — no full conversation history.
- Follow-up detection (`is_follow_up()`) is a simple rule-based check against a small set of patterns (e.g. text starting with "what about", "and", "how about", or containing "tell me more").
- If the current input looks like a follow-up **and** a previous intent is stored, `resolve_context()` returns that stored intent instead of the classifier's result.
- If there is no stored context, a follow-up input does not invent a previous intent — the classifier's (or fallback) result is used as-is.
- `chatbot.reset_context()` clears the stored context, starting a fresh session.

Example:
```
User: What courses do you offer?
Bot: We offer a variety of undergraduate and postgraduate programs across multiple departments.

User: What about that?
Bot: [resolves to the "courses" intent again via stored context]
```

## Fallback / Confidence Handling

The `Chatbot` class uses its own confidence threshold, defaulting to **0.20**, which is separate from `classifier.py`'s own `predict_intent_with_threshold()` default of 0.50.

This 0.20 default was chosen by observing the current model's real confidence scores on known dataset examples, which fall roughly in the 0.25–0.36 range (e.g. "Hello" → greeting at 0.25, fee-related queries around 0.36). With 16 classes, this is meaningfully above the ~0.0625 chance baseline, but well below 0.50. This value is a pragmatic, documented integration choice — **not** derived from a formal ROC/PR analysis or validated against held-out data, and it is fully overridable via `Chatbot(confidence_threshold=...)`.

## Project Structure

```
chatbot-with-nlp/
├── data/
│   └── intents.json
├── models/
│   ├── tfidf_vectorizer.pkl
│   └── intent_classifier.pkl
├── src/
│   ├── preprocessing.py
│   ├── feature_extraction.py
│   ├── classifier.py
│   ├── context_handler.py
│   ├── response_generator.py
│   ├── train.py
│   └── chatbot.py
├── tests/
│   ├── test_preprocessing.py
│   ├── test_feature_extraction.py
│   ├── test_classifier.py
│   ├── test_context_handler.py
│   ├── test_response_generator.py
│   ├── test_train.py
│   └── test_chatbot.py
├── docs/
│   ├── screenshots/
│   └── screenrecord/
├── requirements.txt
├── README.md
├── project_report.md
└── .gitignore
```

## Installation

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Training

```bash
python -m src.train
```

This loads `data/intents.json`, preprocesses all patterns, fits a TF-IDF vectorizer, trains the Logistic Regression classifier, and saves both artifacts to `models/`. It also prints a sanity-check summary (pattern/class/feature counts and a handful of sample predictions) — this is not a formal accuracy benchmark.

## Usage

```python
from src.chatbot import Chatbot

chatbot = Chatbot()
response = chatbot.respond("Hello")
print(response)
```

Example conversation:
```
>>> chatbot.respond("Hello")
'Hello! How can I help you today?'
>>> chatbot.respond("What courses do you offer?")
'We offer a variety of undergraduate and postgraduate programs across multiple departments.'
>>> chatbot.respond("What is the fee structure?")
'The fee structure varies by program. Please refer to the official fee details for accurate figures.'
>>> chatbot.respond("What about it?")
'Scholarships and fee waivers may be available for eligible students based on merit or need.'
>>> chatbot.respond("asdlkjasldkj")
"I'm sorry, I didn't quite understand that. Could you rephrase your question?"
```

## Testing

```bash
python -m pytest tests/test_chatbot.py -v
python -m pytest tests/ -v
```

Verified results:
- **41/41** chatbot integration tests passing
- **163/163** total tests passing (122 unit tests across preprocessing, feature extraction, classifier, context handler, response generator, and training + 41 chatbot integration tests)

## Evidence / Demonstration

| Evidence | Purpose |
|---|---|
| `docs/screenshots/image1.png` | Full test suite run in the terminal (163 passed) |
| `docs/screenshots/image2.png` | Training run output (`python -m src.train`) showing dataset stats and sample predictions |
| `docs/screenshots/image3.png` | Interactive Python REPL session demonstrating `Chatbot.respond()`, including a context-dependent follow-up and a fallback case |

A screen-recording video is referenced as part of the project's `docs/` folder; its exact filename was not available for inspection in this documentation pass — update this section with the real path once confirmed.

## Limitations

- Small dataset (138 patterns across 16 intents), which limits robustness to phrasing not resembling the training patterns
- Traditional ML (TF-IDF + Logistic Regression) rather than generative or embedding-based NLP
- Confidence scores are `predict_proba()` outputs, not calibrated probabilities of correctness
- No web UI in the final implementation — the chatbot is used via the Python API
- Responses are selected from a fixed, predefined set — no dynamically generated text
- No formal held-out ML evaluation (no accuracy/precision/recall/F1 metrics)
- The 0.20 confidence threshold is a pragmatic default, not a scientifically tuned value

## Future Improvements

The following are proposed directions, not implemented features:
- A larger, more diverse training dataset
- Broader intent coverage
- A formal evaluation dataset with train/test split and standard ML metrics
- Improved, calibrated confidence scoring
- Semantic embeddings for better generalization to unseen phrasing
- A web-based user interface
- Deployment to a hosted environment
- Multilingual support

## License

This project is licensed under the MIT License. See the `LICENSE` file for full terms.