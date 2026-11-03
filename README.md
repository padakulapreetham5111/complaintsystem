# AI-Based Complaint Analysis & Automated Response System

A two-stage NLP pipeline that classifies customer complaints, detects sentiment,
prioritises tickets, generates an automated reply, and learns from human feedback.

## Architecture

```
complaint text
   |
   +--> VADER sentiment ------------------------------+
   |                                                  |
   +--> Stage 1: TF-IDF + LogReg  -> category         |
   |        (confidence < 0.40 -> escalate to human)  |
   +--> Stage 2: TF-IDF + LogReg  -> intent           |
            (one model per category)                  |
                     |                                |
                     v                                v
             Priority scoring  <-- sentiment + urgency keywords + category risk
                     |
                     v
        Automated reply (intent template + sentiment-aware opening + SLA)
                     |
                     v
        Feedback (helpful? / correct labels) -> data/feedback.csv
                     |
                     v
        retrain.py merges feedback (x5 weight) -> better models
```

## Setup

```bash
pip install -r requirements.txt
python generate_dataset.py      # creates data/complaints.csv (synthetic)
python train.py                 # trains both stages, saves to models/
```

## Run

```bash
python main.py "I was charged twice and need this fixed immediately!"   # one-shot CLI
python main.py                                                          # interactive + feedback prompts
python app.py                                                           # web UI at http://127.0.0.1:5000
python retrain.py                                                       # apply collected feedback
pytest                                                                  # run tests
```

## API

| Method | Route            | Body                                                                 |
|--------|------------------|----------------------------------------------------------------------|
| POST   | `/api/analyze`   | `{"text": "..."}`                                                    |
| POST   | `/api/feedback`  | `{ticket_id, text, category, intent, helpful, true_category?, true_intent?}` |
| GET    | `/api/stats`     | feedback statistics                                                  |
| POST   | `/api/retrain`   | retrain with feedback and hot-reload models                          |

## Project structure

```
complaint_system/
  app.py                 Flask web app + JSON API
  main.py                CLI
  train.py / retrain.py  training and feedback-driven retraining
  generate_dataset.py    synthetic dataset builder
  src/
    config.py            paths, thresholds
    preprocess.py        text cleaning
    classifier.py        TF-IDF + LogisticRegression (stage 1 & 2)
    sentiment.py         VADER + priority scoring
    responder.py         reply generation
    feedback.py          feedback store
    pipeline.py          end-to-end orchestration
  data/                  complaints.csv, responses.json, feedback.csv
  models/                trained .joblib files, metrics.json
  templates/index.html   web UI
  tests/                 pytest suite
```

## Important notes

* The bundled dataset is **synthetic** and built from a few phrasings per
  intent, so test accuracy is ~100%. That number is not realistic. For real use,
  replace `data/complaints.csv` (columns: `text, category, intent`) with real
  labelled data (e.g. the CFPB Consumer Complaint Database) and retrain.
* To add a category/intent: add labelled rows, add a template to
  `data/responses.json`, and run `python train.py`.
* Tune `CONFIDENCE_THRESHOLD` and `FEEDBACK_WEIGHT` in `src/config.py`.
