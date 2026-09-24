"""Method B: sentence-embedding pipeline (skrub TableVectorizer + classifier).

The whole thing is a single scikit-learn pipeline: a skrub ``TableVectorizer``
holding a ``TextEncoder`` for the string column, followed by a logistic
regression. Because the encoder lives inside the pipeline, the fit and predict
timings account for encoding the train and test text respectively — there is no
separate, untimed encoding step.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from .config import RANDOM_STATE
from .data import Dataset
from .metrics import compute_metrics, confusion, report_text
from .timing import repeat

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass
class EmbedResult:
    name: str
    model_name: str
    metrics: dict[str, Any] = field(default_factory=dict)
    report: str = ""
    confusion: np.ndarray | None = None
    embedding_dim: int = 0
    fit_timing: dict[str, float] = field(default_factory=dict)
    predict_timing: dict[str, float] = field(default_factory=dict)
    throughput_docs_per_s: float = 0.0
    device: str = ""

    def timing_row(self) -> dict[str, Any]:
        return {
            "method": self.name,
            "grid_search_seconds": 0.0,
            "encode_train_median": 0.0,
            "encode_test_median": 0.0,
            **{f"fit_{k}": v for k, v in self.fit_timing.items()},
            **{f"predict_{k}": v for k, v in self.predict_timing.items()},
            "throughput_docs_per_s": self.throughput_docs_per_s,
        }


def _device() -> str:
    import torch

    return "mps" if torch.backends.mps.is_available() else "cpu"


def _build_pipeline(model_name: str, device: str, C: float, batch_size: int):
    from skrub import TableVectorizer, TextEncoder

    encoder = TextEncoder(
        model_name=model_name,
        n_components=None,  # keep the full encoder dimensionality
        device=device,
        batch_size=batch_size,
        random_state=RANDOM_STATE,
    )
    vectorizer = TableVectorizer(specific_transformers=[(encoder, ["text"])])
    return make_pipeline(vectorizer, LogisticRegression(max_iter=1000, C=C))


def run_embed(
    dataset: Dataset,
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 32,
    n_repeats: int = 3,
    C: float = 10.0,
    **_: Any,
) -> EmbedResult:
    """Fit and evaluate the embedding pipeline end to end."""
    device = _device()
    pipeline = _build_pipeline(model_name, device, C, batch_size)

    x_train = pd.DataFrame({"text": dataset.x_train})
    x_test = pd.DataFrame({"text": dataset.x_test})

    _, fit_timing = repeat(
        lambda: pipeline.fit(x_train, dataset.y_train), n_repeats=n_repeats
    )
    y_pred, predict_timing = repeat(
        lambda: pipeline.predict(x_test), n_repeats=n_repeats
    )

    embedding_dim = int(pipeline.named_steps["tablevectorizer"]
                        .transform(x_test.iloc[:1]).shape[1])
    throughput = (
        len(dataset.x_test) / predict_timing.median if predict_timing.median else 0.0
    )

    return EmbedResult(
        name=f"embed:{model_name.split('/')[-1]}",
        model_name=model_name,
        metrics=compute_metrics(dataset.y_test, y_pred, dataset.target_names),
        report=report_text(dataset.y_test, y_pred, dataset.target_names),
        confusion=confusion(dataset.y_test, y_pred, dataset.n_classes),
        embedding_dim=embedding_dim,
        fit_timing=fit_timing.as_dict(),
        predict_timing=predict_timing.as_dict(),
        throughput_docs_per_s=throughput,
        device=device,
    )