"""Classification metrics and confusion-matrix helpers."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


def compute_metrics(
    y_true: list[int] | np.ndarray,
    y_pred: list[int] | np.ndarray,
    target_names: list[str],
) -> dict:
    """Compute the headline metrics shared by all methods."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(target_names)), zero_division=0
    )
    per_class = {
        name: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i, name in enumerate(target_names)
    }
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "n_samples": int(len(y_true)),
        "per_class": per_class,
    }


def confusion(
    y_true: list[int] | np.ndarray,
    y_pred: list[int] | np.ndarray,
    n_classes: int,
) -> np.ndarray:
    """Raw (unnormalized) confusion matrix."""
    return confusion_matrix(y_true, y_pred, labels=range(n_classes))


def report_text(
    y_true: list[int] | np.ndarray,
    y_pred: list[int] | np.ndarray,
    target_names: list[str],
) -> str:
    """Full sklearn classification report as text."""
    return classification_report(
        y_true, y_pred, labels=range(len(target_names)),
        target_names=target_names, zero_division=0,
    )