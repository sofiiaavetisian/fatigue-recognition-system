"""
Train a sklearn classifier that maps geometric features (EAR, MAR, head pitch,
eye-closed progress, yawn progress) to a fatigue probability.

Reads `data/processed/features.csv` produced by `preprocess_videos.py`,
splits it 70/15/15, fits a RandomForestClassifier (with LogisticRegression as
a calibrated fallback), and writes:
    models/classical_classifier.pkl     - {"model": ..., "feature_order": [...]}
    models/classical_classifier_summary.json

Run:
    python -m src.tools.train_classical_classifier
"""
from __future__ import annotations

import csv
import json
import os
import pickle
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

FEATURES_CSV = "data/processed/features.csv"
MODEL_PATH = "models/classical_classifier.pkl"
SUMMARY_PATH = "models/classical_classifier_summary.json"

FEATURE_ORDER = ["ear", "mar", "pitch_deg", "eye_closed_progress", "yawn_progress"]
RANDOM_SEED = 42


def _load_features(path: str) -> tuple[np.ndarray, np.ndarray]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run `python -m src.tools.preprocess_videos` first."
        )
    X, y = [], []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            X.append([float(row[k]) for k in FEATURE_ORDER])
            y.append(int(row["label"]))
    if not X:
        raise RuntimeError(f"No rows in {path}")
    return np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.int64)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def train() -> dict:
    X, y = _load_features(FEATURES_CSV)
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    print(f"Loaded {len(X)} rows ({pos} fatigue / {neg} alert)")

    if pos == 0 or neg == 0:
        raise RuntimeError(
            "Need both alert and fatigue examples to train a classifier. "
            "Adjust thresholds in preprocess_videos.py or capture more data."
        )

    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.30, random_state=RANDOM_SEED, stratify=y,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.50, random_state=RANDOM_SEED, stratify=y_tmp,
    )

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    rf = RandomForestClassifier(
        n_estimators=200, max_depth=None, class_weight="balanced",
        n_jobs=-1, random_state=RANDOM_SEED,
    ).fit(X_train_s, y_train)

    lr = LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=RANDOM_SEED,
    ).fit(X_train_s, y_train)

    rf_val = _metrics(y_val, rf.predict(X_val_s))
    lr_val = _metrics(y_val, lr.predict(X_val_s))
    print(f"RandomForest val: {rf_val}")
    print(f"LogisticReg  val: {lr_val}")

    chosen_name, chosen = (
        ("random_forest", rf) if rf_val["f1"] >= lr_val["f1"] else ("logistic_regression", lr)
    )
    print(f"Selected: {chosen_name}")

    test_metrics = _metrics(y_test, chosen.predict(X_test_s))
    print(f"Test metrics: {test_metrics}")
    print(classification_report(y_test, chosen.predict(X_test_s),
                                target_names=["alert", "fatigue"], zero_division=0))

    bundle = {
        "model": chosen,
        "scaler": scaler,
        "feature_order": FEATURE_ORDER,
        "model_kind": chosen_name,
    }
    Path(MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(bundle, f)
    print(f"Saved classifier to {MODEL_PATH}")

    summary = {
        "model_kind": chosen_name,
        "feature_order": FEATURE_ORDER,
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
        "n_test": int(len(y_test)),
        "val_metrics_random_forest": rf_val,
        "val_metrics_logistic_regression": lr_val,
        "test_metrics": test_metrics,
    }
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {SUMMARY_PATH}")
    return summary


if __name__ == "__main__":
    train()
