"""Train both stages and save models + metrics.

Usage:  python train.py [--no-feedback]
By default, verified rows from data/feedback.csv are merged into training
(see retrain.py for the feedback-driven loop).
"""
import argparse
import json

import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from src import classifier
from src.config import (DATASET_PATH, FEEDBACK_WEIGHT, METRICS_PATH, RANDOM_STATE,
                        STAGE1_MODEL_PATH, STAGE2_MODEL_PATH)
from src.feedback import training_rows_from_feedback


def load_training_data(use_feedback: bool = True) -> pd.DataFrame:
    base = pd.read_csv(DATASET_PATH)
    if use_feedback:
        fb = training_rows_from_feedback()
        if not fb.empty:
            print(f"Merging {len(fb)} feedback rows (x{FEEDBACK_WEIGHT} weight)")
            base = pd.concat([base] + [fb] * FEEDBACK_WEIGHT, ignore_index=True)
    return base


def main(use_feedback: bool = True) -> dict:
    df = load_training_data(use_feedback)
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=RANDOM_STATE, stratify=df["category"])

    # ---- Stage 1: category ----
    s1 = classifier.train_stage1(train_df)
    pred1 = s1.predict(test_df["text"])
    rep1 = classification_report(test_df["category"], pred1, output_dict=True, zero_division=0)
    print("=== Stage 1: Category ===")
    print(classification_report(test_df["category"], pred1, zero_division=0))

    # ---- Stage 2: intent (per category) ----
    s2 = classifier.train_stage2(train_df)
    correct, total = 0, 0
    for cat, model in s2.items():
        sub = test_df[test_df["category"] == cat]
        if sub.empty:
            continue
        p = model.predict(sub["text"])
        correct += int((p == sub["intent"]).sum())
        total += len(sub)
    s2_acc = correct / total if total else 0.0
    print(f"=== Stage 2: Intent accuracy: {s2_acc:.3f} ({correct}/{total}) ===")

    # Final models are retrained on ALL data so nothing is wasted.
    classifier.save(classifier.train_stage1(df), STAGE1_MODEL_PATH)
    classifier.save(classifier.train_stage2(df), STAGE2_MODEL_PATH)

    metrics = {"rows": len(df),
               "stage1_accuracy": round(rep1["accuracy"], 4),
               "stage1_macro_f1": round(rep1["macro avg"]["f1-score"], 4),
               "stage2_accuracy": round(s2_acc, 4)}
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print("Saved models.", metrics)
    return metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-feedback", action="store_true")
    main(use_feedback=not ap.parse_args().no_feedback)
