"""Method B: sentence-embedding encoder + linear classifier.

Documents are encoded with a Sentence-Transformers model (a language-model text
encoder that captures semantics beyond lexical overlap), then a logistic
regression is trained on the frozen embeddings. Embeddings are cached to disk.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression

from .config import CACHE_DIR
from .data import Dataset
from .metrics import compute_metrics, confusion, report_text
from .timing import TimingResult, repeat


def _embedding_path(model_name: str, split: str) -> Path:
    safe = model_name.replace("/", "__")
    return CACHE_DIR / f"embeddings_{safe}_{split}.npy"


@dataclass
class EmbedResult:
    name: str
    model_name: str
    metrics: dict[str, Any] = field(default_factory=dict)
    report: str = ""
    confusion: np.ndarray | None = None
    embedding_dim: int = 0
    train_encode_timing: dict[str, float] = field(default_factory=dict)
    test_encode_timing: dict[str, float] = field(default_factory=dict)
    fit_timing: dict[str, float] = field(default_factory=dict)
    predict_timing: dict[str, float] = field(default_factory=dict)
    throughput_docs_per_s: float = 0.0
    device: str = ""

    def timing_row(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "method": self.name,
            "grid_search_seconds": 0.0,
            "encode_train_median": self.train_encode_timing.get("median", 0.0),
            "encode_test_median": self.test_encode_timing.get("median", 0.0),
            **{f"fit_{k}": v for k, v in self.fit_timing.items()},
            **{f"predict_{k}": v for k, v in self.predict_timing.items()},
            "throughput_docs_per_s": self.throughput_docs_per_s,
        }
        return row


def _load_sentence_transformer(model_name: str):
    import torch
    from sentence_transformers import SentenceTransformer

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = SentenceTransformer(model_name, device=device)
    return model, device


def _encode(model, texts: list[str], batch_size: int, show_progress: bool) -> np.ndarray:
    return model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=show_progress,
    )


def _timed_encode(
    model,
    texts: list[str],
    batch_size: int,
    show_progress: bool,
    n_repeats: int,
) -> tuple[np.ndarray, TimingResult]:
    """Encode ``texts`` ``n_repeats`` times, returning the last result + timings."""
    timing = TimingResult(label="encode")
    embeddings: np.ndarray | None = None
    for _ in range(n_repeats):
        start = time.perf_counter()
        embeddings = _encode(model, texts, batch_size, show_progress)
        timing.seconds.append(time.perf_counter() - start)
    assert embeddings is not None
    return embeddings, timing


def run_embed(
    dataset: Dataset,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 64,
    n_repeats: int = 3,
    use_cache: bool = True,
    show_progress: bool = True,
    C: float = 10.0,
) -> EmbedResult:
    """Encode the corpus and train a logistic regression on the embeddings."""
    model, device = _load_sentence_transformer(model_name)

    train_path = _embedding_path(model_name, "train")
    test_path = _embedding_path(model_name, "test")

    if use_cache and train_path.exists() and test_path.exists():
        emb_train = np.load(train_path)
        emb_test = np.load(test_path)
        train_encode = TimingResult(label="embed_train")
        test_encode = TimingResult(label="embed_test")
    else:
        emb_train, train_encode = _timed_encode(
            model, dataset.x_train, batch_size, show_progress, n_repeats
        )
        emb_test, test_encode = _timed_encode(
            model, dataset.x_test, batch_size, show_progress, n_repeats
        )
        np.save(train_path, emb_train)
        np.save(test_path, emb_test)

    clf = LogisticRegression(max_iter=1000, C=C)

    _, fit_timing = repeat(
        lambda: clf.fit(emb_train, dataset.y_train), n_repeats=n_repeats
    )
    y_pred, predict_timing = repeat(
        lambda: clf.predict(emb_test), n_repeats=n_repeats
    )

    throughput = (
        len(dataset.x_test) / predict_timing.median if predict_timing.median else 0.0
    )

    return EmbedResult(
        name=f"embed:{model_name.split('/')[-1]}",
        model_name=model_name,
        metrics=compute_metrics(dataset.y_test, y_pred, dataset.target_names),
        report=report_text(dataset.y_test, y_pred, dataset.target_names),
        confusion=confusion(dataset.y_test, y_pred, dataset.n_classes),
        embedding_dim=int(emb_train.shape[1]),
        train_encode_timing=train_encode.as_dict(),
        test_encode_timing=test_encode.as_dict(),
        fit_timing=fit_timing.as_dict(),
        predict_timing=predict_timing.as_dict(),
        throughput_docs_per_s=throughput,
        device=device,
    )