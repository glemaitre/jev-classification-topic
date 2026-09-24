"""Method A: TF-IDF + linear/naive-Bayes classifiers with grid search.

Mirrors the scikit-learn example
``model_selection/plot_grid_search_text_feature_extraction.py`` and adds timing
instrumentation. Several classifiers can be plugged in behind the same TF-IDF
front end:

- ``linsvc``: ``LinearSVC`` (the original baseline),
- ``lr``: ``LogisticRegression``,
- ``nb``: ``MultinomialNB``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from .data import Dataset
from .metrics import compute_metrics, confusion, report_text
from .timing import Timer, TimingResult, repeat

CLASSIFIER_FACTORIES: dict[str, Callable[[], Any]] = {
    "linsvc": lambda: LinearSVC(dual="auto"),
    "lr": lambda: LogisticRegression(max_iter=1000),
    "nb": lambda: MultinomialNB(),
}


def default_param_grid(classifier: str = "linsvc") -> dict[str, list[Any]]:
    """A modest grid per classifier over TF-IDF and model hyper-parameters."""
    grid: dict[str, list[Any]] = {
        "vect__max_df": [0.5, 1.0],
        "vect__max_features": [None, 50_000],
        "vect__ngram_range": [(1, 1), (1, 2)],
    }
    if classifier == "linsvc":
        grid["clf__C"] = [0.1, 1.0, 10.0]
    elif classifier == "lr":
        grid["clf__C"] = [0.1, 1.0, 10.0]
    elif classifier == "nb":
        grid["clf__alpha"] = [0.1, 1.0]
    return grid


def build_pipeline(classifier: str = "linsvc") -> Pipeline:
    try:
        clf = CLASSIFIER_FACTORIES[classifier]()
    except KeyError:
        raise ValueError(
            f"unknown classifier {classifier!r}; choose from "
            f"{sorted(CLASSIFIER_FACTORIES)}"
        ) from None
    return Pipeline([("vect", TfidfVectorizer()), ("clf", clf)])


@dataclass
class TfidfResult:
    name: str = "tfidf:linsvc"
    classifier: str = "linsvc"
    best_params: dict[str, Any] = field(default_factory=dict)
    best_cv_score: float = 0.0
    scoring: str = "accuracy"
    cv_results: pd.DataFrame | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    report: str = ""
    confusion: np.ndarray | None = None
    grid_search_seconds: float = 0.0
    fit_timing: dict[str, float] = field(default_factory=dict)
    predict_timing: dict[str, float] = field(default_factory=dict)
    throughput_docs_per_s: float = 0.0

    def timing_row(self) -> dict[str, Any]:
        return {
            "method": self.name,
            "grid_search_seconds": self.grid_search_seconds,
            "encode_train_median": 0.0,
            "encode_test_median": 0.0,
            **{f"fit_{k}": v for k, v in self.fit_timing.items()},
            **{f"predict_{k}": v for k, v in self.predict_timing.items()},
            "throughput_docs_per_s": self.throughput_docs_per_s,
        }


def run_tfidf(
    dataset: Dataset,
    classifier: str = "linsvc",
    param_grid: dict[str, list[Any]] | None = None,
    cv: int = 5,
    scoring: str = "accuracy",
    n_jobs: int = -1,
    n_repeats: int = 3,
    verbose: int = 1,
) -> TfidfResult:
    """Run the grid search and time refit/predict for the chosen classifier."""
    param_grid = param_grid or default_param_grid(classifier)
    pipeline = build_pipeline(classifier)

    search = GridSearchCV(
        pipeline, param_grid, cv=cv, scoring=scoring, n_jobs=n_jobs, verbose=verbose
    )
    with Timer() as grid_timer:
        search.fit(dataset.x_train, dataset.y_train)
    grid_seconds = grid_timer.elapsed

    best = search.best_estimator_
    best_params = {k: str(v) for k, v in search.best_params_.items()}

    estimator = clone(best)
    _, fit_timing = repeat(
        lambda: estimator.fit(dataset.x_train, dataset.y_train), n_repeats=n_repeats
    )
    y_pred, predict_timing = repeat(
        lambda: estimator.predict(dataset.x_test), n_repeats=n_repeats
    )

    throughput = len(dataset.x_test) / predict_timing.median if predict_timing.median else 0.0

    return TfidfResult(
        name=f"tfidf:{classifier}",
        classifier=classifier,
        best_params=best_params,
        best_cv_score=float(search.best_score_),
        scoring=scoring,
        cv_results=pd.DataFrame(search.cv_results_),
        metrics=compute_metrics(dataset.y_test, y_pred, dataset.target_names),
        report=report_text(dataset.y_test, y_pred, dataset.target_names),
        confusion=confusion(dataset.y_test, y_pred, dataset.n_classes),
        grid_search_seconds=grid_seconds,
        fit_timing=fit_timing.as_dict(),
        predict_timing=predict_timing.as_dict(),
        throughput_docs_per_s=throughput,
    )