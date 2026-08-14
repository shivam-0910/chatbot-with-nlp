# Project Report
## NLP-Based College FAQ Chatbot

### 1. Introduction

This report documents the design, implementation, and testing of an NLP-based chatbot developed as an AI internship project. The chatbot answers common college-related questions by classifying free-text user input into one of a fixed set of intents and returning an appropriate predefined response. The system is built entirely on classical Natural Language Processing (NLP) and machine learning techniques — text preprocessing, TF-IDF feature extraction, and a Logistic Regression classifier — rather than deep learning or generative language models.

### 2. Problem Statement

Prospective and current students frequently ask the same categories of questions about a college (admissions, fees, courses, timetables, library rules, and so on). Answering these manually is repetitive and does not scale. This project addresses that problem by building an automated chatbot that recognizes the *intent* behind a user's question, even when phrased differently from the training examples, and responds with relevant, predefined information.

### 3. Objectives

- Build a text preprocessing pipeline that normalizes raw user input for machine learning.
- Represent text numerically using TF-IDF so a classifier can operate on it.
- Train a multi-class classifier capable of distinguishing between 16 distinct college-related intents.
- Handle low-confidence predictions gracefully via a fallback mechanism rather than returning a likely-wrong answer.
- Support simple context-aware follow-up questions across two conversational turns.
- Validate the full pipeline with automated tests covering each component individually and the integrated chatbot end-to-end.

### 4. Project Scope

The current implementation covers:
- A dataset of 16 intents and 138 training patterns (`data/intents.json`).
- A full text preprocessing pipeline (cleaning, tokenization, stopword removal, stemming, lemmatization).
- TF-IDF feature extraction and a trained Logistic Regression intent classifier.
- Confidence-threshold-based fallback handling.
- Single-step (last-turn-only) conversational context for follow-up resolution.
- A Python API (`Chatbot` class) as the interface to the system.
- An automated test suite of 163 tests.

Out of scope for the current implementation: any web or GUI interface, deployment, multi-turn conversation history beyond one prior turn, deep learning or transformer-based models, and formal statistical evaluation of classifier accuracy.

### 5. Requirements

**Software requirements**
- Python 3.x
- A virtual environment (`.venv`) for dependency isolation

**Python / dependency requirements**
- `scikit-learn` (TF-IDF vectorization, Logistic Regression)
- `nltk` (tokenization, stopwords, stemming, lemmatization)
- `pytest` (test execution)
- Exact versions are pinned in `requirements.txt`, which was not available for direct inspection during this documentation pass; refer to that file directly for the authoritative dependency list.

**Functional requirements**
- Classify a user's text input into one of the dataset's intents.
- Return a relevant response drawn from the dataset for the classified intent.
- Detect and resolve simple follow-up questions using the previous turn's intent.
- Fall back to a generic "didn't understand" response when confidence is low or input is invalid.
- Persist and reload trained model artifacts without retraining on every run.

**Non-functional requirements**
- The pipeline should not silently invent or fabricate confidence values — all confidence scores must be the classifier's real `predict_proba()` output.
- Model loading failures must raise clear, actionable errors rather than failing silently or creating fake models.
- Existing, validated modules (preprocessing, feature extraction, classifier, context handler, response generator) must not be reimplemented or duplicated by the orchestration layer.

### 6. System Architecture

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

`src/chatbot.py` is a thin orchestration layer: it does not implement any NLP or ML logic itself, only calls the above modules in this fixed order. Each module is independently testable and has its own dedicated unit test file, in addition to the end-to-end integration tests in `tests/test_chatbot.py`.

An important architectural detail, confirmed directly from the source: the confidence-threshold check happens **before** context resolution, and the (possibly fallback-substituted) intent — not the raw classifier prediction — is what gets passed into `resolve_context()`. This means a low-confidence prediction on a follow-up turn can still be correctly recovered via stored context.

### 7. Dataset

Location: `data/intents.json`

Verified directly from the file:

| Metric | Value |
|---|---|
| Number of intents | 16 |
| Total training patterns | 138 |
| Responses per intent | 3 |
| Fallback intent present | Yes (`fallback`, 7 patterns, 3 responses) |

Intents cover: greeting, goodbye, thanks, college_info, courses, admission, fees, attendance, examinations, timetable, library, contact_info, working_hours, facilities, events, and fallback.

Each intent record has three fields:
- `tag` — the intent's unique label
- `patterns` — example phrasings of that intent, used only during training
- `responses` — one or more candidate replies; one is chosen at random per matching request

The dataset is read twice by the system: once by `src/train.py` (patterns + tags, for training) and once by `src/response_generator.py` (tags + responses, for generating replies at runtime).

### 8. NLP Preprocessing

Implemented in `src/preprocessing.py`. The pipeline, in order, is:

1. **Lowercasing and punctuation removal** (`clean_text`) — converts to lowercase and strips all punctuation, collapsing extra whitespace.
2. **Tokenization** (`tokenize_text`) — splits cleaned text into word tokens using NLTK's `word_tokenize`.
3. **Stopword removal** (`remove_stopwords`) — filters out common English stopwords using NLTK's stopword list.
4. **Lemmatization** (`lemmatize_tokens`) — reduces tokens to dictionary base forms using NLTK's `WordNetLemmatizer`. This is the technique used in the main `preprocess_text()` pipeline.

**Stemming** (`stem_tokens`, using NLTK's `PorterStemmer`) is also implemented and independently testable, but is *not* chained into the main pipeline. The module's own documentation explains why: stemming a token and then lemmatizing the stemmed result does not reliably produce a valid word (e.g. "studies" → "studi" via stemming is not a valid lemmatizer input), so the two techniques are exposed as parallel, independent operations on the same stopword-filtered tokens rather than a serial chain. `preprocess_text()` — the function used everywhere else in the pipeline — uses lemmatization for its final output, since it produces real dictionary words that are better suited to TF-IDF downstream.

NLTK resource downloads (`stopwords`, `punkt`, `punkt_tab`, `wordnet`, `omw-1.4`) are handled lazily and idempotently by `download_nltk_resources()`, checking local availability before attempting any network access.

### 9. Feature Extraction

Implemented in `src/feature_extraction.py` using scikit-learn's `TfidfVectorizer`.

- `fit_transform_texts()` is the training-time entry point: it creates a new vectorizer and fits it once on the full corpus of preprocessed training patterns.
- `transform_text()` is the inference-time entry point: it transforms new text using an **already-fitted** vectorizer and explicitly refuses to re-fit.
- The vectorizer is persisted via `save_vectorizer()`/`load_vectorizer()` (pickle-based), so training happens exactly once and inference reuses the saved vocabulary.

On the current dataset (138 preprocessed patterns), fitting produces **179 TF-IDF features**, confirmed by directly running `python -m src.train` against the actual dataset.

### 10. Intent Classification

Implemented in `src/classifier.py` using `sklearn.linear_model.LogisticRegression` (`max_iter=1000`, `random_state=42` by default).

- `train_classifier()` fits the classifier on the TF-IDF feature matrix and intent labels.
- `predict_intent()` returns the single most likely label.
- `predict_intent_with_confidence()` returns both the predicted label and the real `predict_proba()`-based probability for that label — never a fabricated or rescaled value.
- `predict_intent_with_threshold()` is a separate convenience helper with its own default threshold of `0.50`; it is intentionally **not** used by `chatbot.py`, because it discards the confidence value that the chatbot needs to retain for its own threshold logic.

On the current model, real observed confidence for correctly classified, in-dataset examples falls roughly in the 0.25–0.36 range (verified via the training script's sanity-check output — see Section 16), which is well above the ~0.0625 no-information baseline for a 16-way classification problem but below the classifier module's own 0.50 default.

### 11. Context Management

Implemented in `src/context_handler.py`. Context is intentionally minimal: a `ConversationContext` dict stores only `last_intent` and `last_user_input` — no multi-turn history.

- `is_follow_up()` is a simple, deterministic, rule-based check against a fixed list of regex patterns (e.g. text starting with "what about", "and", "how about", or containing "tell me more", "what else", "anything else"). It performs no semantic analysis.
- `resolve_context()` returns the stored `last_intent` only if the current input is judged a follow-up **and** a previous intent exists; otherwise it returns whatever intent was passed in (which, per the pipeline order in Section 6, is the post-threshold classifier result).
- `set_context()` returns a new context rather than mutating the old one, and deliberately does *not* overwrite existing context with an empty/None intent — this prevents an accidental bad classification from erasing a valid prior context.
- `clear_context()` resets context to empty, used by `Chatbot.reset_context()`.

### 12. Response Generation

Implemented in `src/response_generator.py`.

- `load_responses()` parses `data/intents.json` into a `{tag: [responses]}` mapping.
- `generate_response()` picks one response at random (via the standard library's `random.choice`) from the resolved intent's list. If the intent is unknown, empty, or `None`, it falls back to the dataset's own `fallback` intent responses. If even the `fallback` intent is missing from the dataset (a degraded state), a single hard-coded minimal message is used as a last resort.
- Response selection never mutates the original response lists in the loaded mapping.

### 13. Chatbot Integration

Implemented in `src/chatbot.py`. The `Chatbot` class is purely an orchestration layer over the modules described above; it implements no new NLP or ML logic.

On initialization, it:
1. Validates the given `confidence_threshold` is within `[0.0, 1.0]`.
2. Loads the pre-trained vectorizer and classifier from disk (never trains).
3. Loads the response mapping from `data/intents.json`.
4. Creates a fresh, empty conversation context.

On each call to `respond(user_input)`:
1. Validates the input type (raises `TypeError` for non-string, non-`None` input).
2. Empty/whitespace/`None` input skips classification entirely and returns the dataset's fallback response directly.
3. Preprocesses the input, vectorizes it, and gets a real predicted intent + confidence from the classifier.
4. Compares that confidence against the chatbot's own threshold (default `0.20`); below threshold, the intent is replaced with the fallback tag.
5. Passes the resulting intent through `resolve_context()`, allowing a genuine follow-up to override it with the stored previous intent.
6. Generates a response for the resolved intent.
7. Updates stored context with the resolved intent for the next turn.

### 14. Training Process

Implemented in `src/train.py`, run via `python -m src.train`.

Flow:
1. `load_training_data()` loads and validates the raw JSON structure (every intent must have a non-empty tag and at least one pattern).
2. `prepare_training_data()` flattens the dataset into parallel `patterns` and `labels` lists.
3. `preprocess_patterns()` runs every pattern through `preprocess_text()`.
4. `train_model()` fits a new TF-IDF vectorizer on the processed patterns and trains a new Logistic Regression classifier on the resulting features.
5. `save_models()` persists both artifacts to `models/tfidf_vectorizer.pkl` and `models/intent_classifier.pkl`, creating the `models/` directory if needed.
6. `print_training_summary()` prints pattern/class/feature counts and the artifact paths.
7. `run_sanity_predictions()` runs a handful of real dataset patterns back through the freshly trained pipeline as an end-to-end sanity check — explicitly documented in the code as *not* a formal accuracy evaluation.

Running this script against the actual `data/intents.json` produces:
```
Training patterns: 138
Intent classes: 16
TF-IDF features: 179
```
confirmed by direct execution during this documentation pass, matching the provided training-run screenshot exactly.

### 15. Testing and Validation

The test suite is organized by module, with one dedicated test file per source module, plus an integration test file for the orchestrated chatbot:

| Test file | Scope |
|---|---|
| `test_preprocessing.py` | Text cleaning, tokenization, stopword removal, stemming, lemmatization |
| `test_feature_extraction.py` | TF-IDF fitting, transforming, save/load, empty-input handling |
| `test_classifier.py` | Classifier creation, training, prediction, confidence, thresholding, save/load |
| `test_context_handler.py` | Context storage, follow-up detection, context resolution, edge cases |
| `test_response_generator.py` | Response loading and selection, fallback behavior, non-mutation |
| `test_train.py` | Dataset loading/validation, end-to-end training, model persistence |
| `test_chatbot.py` | Full integration: initialization, real predictions, threshold behavior, fallback, context/follow-up, multi-turn sequences, context reset, input validation, model-loading failures |

Running the suite (directly verified during this documentation pass):
```
python -m pytest tests/test_chatbot.py -v   →  41 passed
python -m pytest tests/ -v                  →  163 passed
```

This gives 122 non-chatbot unit tests (163 − 41) spanning preprocessing, feature extraction, classifier, context handler, response generator, and training.

**Important distinction:** these tests validate implementation *behavior* — correct wiring between modules, correct handling of edge cases (empty input, missing files, invalid thresholds), and correct persistence/reload of trained artifacts. They do **not** constitute a formal machine learning accuracy evaluation (no held-out test set, no accuracy/precision/recall/F1 metric was computed).

### 16. Demonstration

A representative demonstration sequence, based on the actual dataset and verified via a live REPL session (see `docs/screenshots/image3.png`):

1. **Greeting** — `chatbot.respond("Hello")` → a greeting response, demonstrating basic intent classification.
2. **Courses question** — `chatbot.respond("What courses do you offer?")` → a courses response, demonstrating classification of a domain-specific FAQ.
3. **Fees question** — `chatbot.respond("What is the fee structure?")` → a fees response.
4. **Context-dependent follow-up** — `chatbot.respond("What about it?")` → the follow-up is detected and resolved using the stored `fees` context from the previous turn, returning a related fees response (scholarships/waivers) rather than falling back.
5. **Unknown input / fallback** — `chatbot.respond("asdlkjasldkj")` → the input does not match any real intent with sufficient confidence, and the dataset's fallback response is returned.

Each step demonstrates a distinct part of the pipeline: classification accuracy on in-domain phrasing, context carry-over across turns, and graceful degradation on nonsense input.

### 17. Results

Verified, reproducible results from directly running the project's own scripts and test suite against the actual repository contents:

| Metric | Verified value |
|---|---|
| Intent classes | 16 |
| Training patterns | 138 |
| TF-IDF features | 179 |
| Trained model artifacts | `models/tfidf_vectorizer.pkl`, `models/intent_classifier.pkl` (both produced successfully) |
| Chatbot integration tests | 41/41 passing |
| Total test suite | 163/163 passing |

No accuracy, precision, recall, F1-score, latency, or other benchmark metrics are reported, because no such formal evaluation exists in the current implementation.

### 18. Limitations

- The training dataset is small (138 patterns across 16 classes — an average of under 9 patterns per class), which limits generalization to phrasings very different from the training examples.
- The system uses traditional ML (TF-IDF + Logistic Regression), not generative or embedding-based NLP, so it cannot compose novel responses — it only selects from a fixed response set.
- Classifier confidence scores are raw `predict_proba()` outputs; they are not calibrated probabilities of correctness.
- There is no web or graphical interface in the final implementation; the chatbot is accessed via a Python class.
- No formal held-out evaluation exists, so no accuracy-style claim can be made about the classifier's real-world performance.
- The chatbot's 0.20 confidence threshold is a pragmatic, manually chosen default based on observed confidence ranges, not a value tuned via formal threshold-optimization (e.g. ROC/PR curve analysis).
- Conversational context is limited to exactly one prior turn; there is no longer multi-turn memory.

### 19. Future Enhancements

The following are proposed, **not implemented**, directions for future work:
- Expanding the dataset with more patterns per intent and additional intents.
- Building a proper held-out validation/test split and reporting standard classification metrics.
- Calibrating confidence scores (e.g. via Platt scaling) so thresholds have a clearer probabilistic meaning.
- Exploring semantic embeddings (e.g. sentence embeddings) in place of or alongside TF-IDF for better generalization.
- Adding a web-based front end for broader accessibility.
- Deploying the chatbot as a hosted service.
- Extending conversational context to support multi-turn history rather than a single prior turn.
- Adding multilingual support.

### 20. Conclusion

The project successfully implements a complete, working intent-classification chatbot for a college FAQ use case, using a fully traditional NLP/ML pipeline: text preprocessing (cleaning, tokenization, stopword removal, stemming, and lemmatization), TF-IDF feature extraction, and a Logistic Regression classifier. The system includes practical engineering considerations beyond the core classifier — a configurable, documented confidence threshold for fallback handling, and simple rule-based context resolution for follow-up questions. Every core module is independently unit-tested, and the fully assembled chatbot is separately validated with 41 integration tests, for a combined 163/163 passing automated tests. The implementation is honestly scoped: no deep learning, web interface, or formal accuracy benchmark is claimed, and known limitations (dataset size, unvalidated threshold, lack of formal evaluation) are documented rather than hidden.

### 21. References

- Internship program guide provided for this project (referenced for objectives, terminology, and expected deliverables; not independently reproduced in this report).
- scikit-learn documentation, for `TfidfVectorizer` and `LogisticRegression` — used as implemented in `src/feature_extraction.py` and `src/classifier.py`.
- NLTK documentation, for tokenization, stopword lists, `PorterStemmer`, and `WordNetLemmatizer` — used as implemented in `src/preprocessing.py`.