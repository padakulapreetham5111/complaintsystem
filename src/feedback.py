"""Feedback store: collects human corrections used to refine the models."""
import csv
from datetime import datetime
import pandas as pd

from .config import FEEDBACK_PATH

COLUMNS = ["timestamp", "ticket_id", "text", "pred_category", "pred_intent",
           "true_category", "true_intent", "helpful"]


def record_feedback(ticket_id, text, pred_category, pred_intent,
                    helpful: bool, true_category=None, true_intent=None):
    """Save one feedback row.

    If the user marks the result helpful, the prediction is treated as correct.
    If not helpful, `true_category` / `true_intent` carry the correction.
    """
    if helpful:
        true_category, true_intent = pred_category, pred_intent
    FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
    new_file = not FEEDBACK_PATH.exists()
    with open(FEEDBACK_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(COLUMNS)
        w.writerow([datetime.now().isoformat(timespec="seconds"), ticket_id, text,
                    pred_category, pred_intent, true_category or "",
                    true_intent or "", int(helpful)])


def load_feedback() -> pd.DataFrame:
    if not FEEDBACK_PATH.exists():
        return pd.DataFrame(columns=COLUMNS)
    return pd.read_csv(FEEDBACK_PATH).fillna("")


def feedback_stats() -> dict:
    df = load_feedback()
    if df.empty:
        return {"total": 0, "helpful_rate": None, "corrections": 0}
    helpful = df["helpful"].astype(int)
    return {"total": len(df),
            "helpful_rate": round(float(helpful.mean()), 3),
            "corrections": int((helpful == 0).sum())}


def training_rows_from_feedback() -> pd.DataFrame:
    """Rows with a usable (text, category, intent) label."""
    df = load_feedback()
    if df.empty:
        return pd.DataFrame(columns=["text", "category", "intent"])
    df = df[(df["true_category"] != "") & (df["true_intent"] != "")]
    return df.rename(columns={"true_category": "category",
                              "true_intent": "intent"})[["text", "category", "intent"]]
