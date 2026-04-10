from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


def compute_classification_metrics(y_true, y_pred, y_score) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
    }


def score_to_prediction(y_score) -> np.ndarray:
    y_score = np.asarray(y_score)
    if ((y_score >= 0.0) & (y_score <= 1.0)).all():
        return (y_score >= 0.5).astype(int)
    return (y_score >= 0.0).astype(int)
