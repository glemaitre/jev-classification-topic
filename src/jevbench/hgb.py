"""Method D: HistGradientBoosting on LSA(TF-IDF) features.

HistGradientBoosting requires dense input, so it consumes the shared
TF-IDF -> truncated SVD representation produced by :mod:`jevbench.features`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from .data import Dataset
from .features import build_lsa_features
from .metrics import compute_metrics, confusion, report_text
from .timing import repeat

PARAMS = {"early_stopping": True, "random_state": 42}


@dataclass
class HgbResult:
    name: str = "hgb:lsa"
    n_components: int = 100
    params: dict[str, Any] = field(default_factory=lambda: dict(PARAMS))
    metrics: dict[str, Any] = field(default_factory=dict)
    report: str = ""
    confusion: np.ndarray | None = None
    encode_train_timing: dict[str, float] = field(default_factory=dict)
    encode_test_timing: dict[str, float] = field(default_factory=dict)
    fit_timing: dict[str, float] = field(default_factory=dict)
    predict_timing: dict[str, float] = field(default_factory=dict)
    throughput_docs_per_s: float = 0.0

    def timing_row(self) -> dict[str, Any]:
        return {
            "method": self.name,
            "grid_search_seconds": 0.0,
            "encode_train_median": self.encode_train_timing.get("median", 0.0),
            "encode_test_median": self.encode_test_timing.get("median", 0.0),
            **{f"fit_{k}": v for k, v in self.fit_timing.items()},
            **{f"predict_{k}": v for k, v in self.predict_timing.items()},
            "throughput_docs_per_s": self.throughput_docs_per_s,
        }


def run_hgb(
    dataset: Dataset,
    n_components: int = 100,
    n_repeats: int = 3,
    use_cache: bool = True,
) -> HgbResult:
    """Train HistGradientBoosting on LSA features and evaluate on the test set."""
    features = build_lsa_features(
        dataset, n_components=n_components, n_repeats=n_repeats, use_cache=use_cache
    )

    clf = HistGradientBoostingClassifier(**PARAMS)

    _, fit_timing = repeat(
        lambda: clf.fit(features.x_train, dataset.y_train), n_repeats=n_repeats
    )
    y_pred, predict_timing = repeat(
        lambda: clf.predict(features.x_test), n_repeats=n_repeats
    )

    throughput = (
        len(dataset.x_test) / predict_timing.median if predict_timing.median else 0.0
    )

    return HgbResult(
        name=f"hgb:lsa{n_components}",
        n_components=n_components,
        metrics=compute_metrics(dataset.y_test, y_pred, dataset.target_names),
        report=report_text(dataset.y_test, y_pred, dataset.target_names),
        confusion=confusion(dataset.y_test, y_pred, dataset.n_classes),
        encode_train_timing=features.fit_timing,
        encode_test_timing=features.transform_timing,
        fit_timing=fit_timing.as_dict(),
        predict_timing=predict_timing.as_dict(),
        throughput_docs_per_s=throughput,
    )