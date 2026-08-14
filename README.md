# Chatbot with NLP

An NLP-based college helpdesk chatbot that classifies user queries into intents and provides appropriate responses.

## Project Overview

This project implements a simple, rule-assisted machine learning chatbot that understands natural language questions about a college (admissions, fees, courses, timetable, library, etc.) and responds with relevant information.

The chatbot uses classical Natural Language Processing (NLP) techniques — text cleaning, tokenization, stemming, lemmatization, and TF-IDF feature extraction — combined with a simple machine learning classifier to detect user intent and generate an appropriate response.

This project is being developed as part of the **Bright Hub Private Limited AI Internship Program**, Project 2 of the internship curriculum. It is intentionally kept simple and practical, focusing on core NLP and machine learning fundamentals rather than complex or heavyweight AI systems.

## Features

- **Greeting detection** — recognizes common greetings and responds naturally
- **FAQ responses** — answers common college-related questions (courses, fees, admissions, etc.)
- **Intent classification** — uses TF-IDF + a machine learning classifier to detect user intent
- **Context handling** — *planned*
- **Multiple intent detection** — *planned*
- **Exit commands** — recognizes when a user wants to end the conversation
- **Streamlit interface** — *planned*, a simple web-based chat interface

## NLP Pipeline

```text
User Query
    ↓
Text Cleaning
    ↓
Tokenization
    ↓
Stemming
    ↓
Lemmatization
    ↓
TF-IDF Feature Extraction
    ↓
Intent Classification
    ↓
Response Generation
    ↓
Chatbot Response
```

**Stage descriptions:**

1. **Text Cleaning** — Converts input to lowercase and removes punctuation and stopwords.
2. **Tokenization** — Splits cleaned text into individual word tokens.
3. **Stemming** — Reduces words to their root form (e.g., "running" → "run").
4. **Lemmatization** — Converts words to their dictionary base form using context-aware rules.
5. **TF-IDF Feature Extraction** — Converts processed text into numeric feature vectors.
6. **Intent Classification** — A trained machine learning model predicts the most likely intent.
7. **Response Generation** — Selects an appropriate response based on the predicted intent.
8. **Chatbot Response** — The final reply is returned to the user.

## Technology Stack

| Technology   | Purpose                   |
| ------------ | -------------------------- |
| Python       | Main programming language |
| NLTK         | NLP preprocessing         |
| Scikit-learn | Machine learning / TF-IDF |
| TF-IDF       | Text feature extraction   |
| Streamlit    | Web application            |

## Project Structure

```text
chatbot-with-nlp/
│
├── data/
│   └── intents.json              # Intent dataset (tags, patterns, responses)
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py          # Text cleaning, tokenization, stemming, lemmatization
│   ├── feature_extraction.py     # TF-IDF vectorization
│   ├── train.py                  # Trains and saves the intent classifier
│   ├── classifier.py             # Loads model and predicts intent
│   ├── response_generator.py     # Selects response based on predicted intent
│   ├── context_handler.py        # Tracks simple conversation context
│   └── chatbot.py                # Orchestrates the full chatbot pipeline
│
├── models/
│   ├── tfidf_vectorizer.pkl      # Saved TF-IDF vectorizer
│   └── intent_classifier.pkl     # Saved trained classifier
│
├── tests/
│   ├── test_preprocessing.py
│   ├── test_feature_extraction.py
│   └── test_chatbot.py
│
├── docs/
│   ├── project_report.md
│   ├── screenshots/
│   └── screenrecord/
│
├── app.py                        # Streamlit application entry point
├── requirements.txt
├── README.md
└── .gitignore
```

> **Note:** Some files above are part of the planned project structure and are not yet implemented. See [Project Status](#project-status) for current progress.

## Dataset

The chatbot uses an intent-based dataset stored in `data/intents.json`. Each entry in the dataset contains:

- **Intent tag** — a label identifying the type of query (e.g., `greeting`, `fees`, `admission`)
- **Patterns** — example user questions/messages belonging to that intent
- **Responses** — one or more possible bot replies for that intent

The current domain covered by the dataset is a **college/student helpdesk chatbot**, addressing common topics such as admissions, courses, fees, timetable, examinations, library, and general college information.

## Installation

Clone the repository and set up a virtual environment:

```bash
git clone <repository-url>
cd chatbot-with-nlp

python -m venv venv
```

Activate the virtual environment:

**Windows:**
```bash
venv\Scripts\activate
```

**Linux/macOS:**
```bash
source venv/bin/activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Usage (Planned Workflow)

```text
1. Prepare dataset            → data/intents.json
2. Preprocess text            → src/preprocessing.py
3. Extract features (TF-IDF)  → src/feature_extraction.py
4. Train classifier           → src/train.py
5. Save trained model         → models/
6. Run Streamlit application  → app.py
```

> These steps reflect the intended workflow. Some scripts are still under development — see [Project Status](#project-status).

## Running the Application

Once implementation is complete, the chatbot will be launched with:

```bash
streamlit run app.py
```

This is the intended application entry point.

## Model Files

```text
models/
├── tfidf_vectorizer.pkl   # Fitted TF-IDF vectorizer used to transform user input
└── intent_classifier.pkl  # Trained ML model used to predict intent
```

These files are generated by running `src/train.py` and are loaded at runtime by the chatbot to avoid retraining on every request.

## Testing

Planned test coverage:

```text
tests/
├── test_preprocessing.py       # Unit tests for text cleaning/tokenization/stemming/lemmatization
├── test_feature_extraction.py  # Unit tests for TF-IDF feature extraction
└── test_chatbot.py             # End-to-end chatbot response tests
```

Tests have not yet been implemented or executed. This section will be updated once test coverage is added and verified.

## Project Status

🚧 **Under development.**

This project is in early implementation stages. The project structure and documentation are in place; core NLP and machine learning components are being built incrementally.

## Internship Requirements

This project is being developed as part of the **Bright Hub Private Limited AI Internship Program**, fulfilling the requirements for Project 2 — Chatbot with NLP.

## Future Scope

The following are potential future improvements, not current features:

- Larger and more diverse intent dataset
- Improved context management across multi-turn conversations
- More sophisticated intent classification techniques
- Support for additional domains beyond the college helpdesk use case

## License

License information will be added before public release.# chatbot-with-nlp
