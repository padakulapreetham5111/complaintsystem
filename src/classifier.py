"""Two-stage TF-IDF + Logistic Regression classifier.

Stage 1: complaint text      -> category   (e.g. Billing)
Stage 2: complaint text      -> intent     (e.g. double_charge), using a
         separate model trained only on that category's data.
"""
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .config import RANDOM_STATE
from .preprocess import clean_text


def build_model() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(preprocessor=clean_text, ngram_range=(1, 2),
                                  sublinear_tf=True, min_df=1)),
        ("clf", LogisticRegression(C=10, max_iter=2000,
                                   random_state=RANDOM_STATE)),
    ])


def train_stage1(df):
    model = build_model()
    model.fit(df["text"], df["category"])
    return model


def train_stage2(df) -> dict:
    """One intent model per category."""
    models = {}
    for category, group in df.groupby("category"):
        if group["intent"].nunique() < 2:
            continue  # nothing to discriminate
        m = build_model()
        m.fit(group["text"], group["intent"])
        models[category] = m
    return models


def predict_with_confidence(model, text: str):
    probs = model.predict_proba([text])[0]
    idx = probs.argmax()
    return model.classes_[idx], float(probs[idx])


def save(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)


def load(path):
    return joblib.load(path)
