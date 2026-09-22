"""Method C: zero-shot classification with Jev (TypeSafe System One).

Each test document is sent as ``state`` with a single ``choice`` question whose
criteria are the 20 newsgroup names. The returned choice is the prediction and
its confidence is recorded. There is no training step — the model is used
zero-shot — so the only cost measured here is inference (API) time.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from tqdm import tqdm

from .config import CATEGORY_DESCRIPTIONS
from .data import Dataset
from .jev_client import (
    JevClient,
    JevError,
    build_classification_questions,
    parse_choice,
)
from .metrics import compute_metrics, confusion, report_text
from .timing import format_seconds, latency_summary


@dataclass
class DocResult:
    index: int
    true_label: str
    pred_label: str | None
    confidence: float
    latency: float
    error: str | None = None


@dataclass
class JevResult:
    name: str
    model: str
    endpoint: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    report: str = ""
    confusion: np.ndarray | None = None
    latency: dict[str, float] = field(default_factory=dict)
    wall_clock_seconds: float = 0.0
    throughput_docs_per_s: float = 0.0
    n_errors: int = 0
    n_truncated: int = 0
    usage: dict[str, Any] = field(default_factory=dict)
    client_stats: dict[str, int] = field(default_factory=dict)
    docs: list[DocResult] = field(default_factory=list)

    def timing_row(self) -> dict[str, Any]:
        return {
            "method": self.name,
            "grid_search_seconds": 0.0,
            "encode_train_median": 0.0,
            "encode_test_median": 0.0,
            "fit_mean": 0.0,
            "fit_median": 0.0,
            "fit_stdev": 0.0,
            "fit_min": 0.0,
            "fit_max": 0.0,
            "fit_n_repeats": 0,
            "predict_mean": self.latency.get("mean", 0.0),
            "predict_median": self.latency.get("median", 0.0),
            "predict_stdev": self.latency.get("stdev", 0.0),
            "predict_min": self.latency.get("min", 0.0),
            "predict_max": self.latency.get("max", 0.0),
            "throughput_docs_per_s": self.throughput_docs_per_s,
            "wall_clock_seconds": self.wall_clock_seconds,
            "n_errors": self.n_errors,
            "n_truncated": self.n_truncated,
            "input_tokens": self.usage.get("input_tokens", 0),
            "output_tokens": self.usage.get("output_tokens", 0),
            "cost_usd": self.usage.get("cost", 0.0),
        }


def run_jev(
    dataset: Dataset,
    model: str,
    concurrency: int = 8,
    limit: int | None = None,
    allow_other: bool = False,
    show_progress: bool = True,
    client: JevClient | None = None,
    use_cache: bool = True,
) -> JevResult:
    """Classify the test set zero-shot with Jev.

    ``use_cache=False`` bypasses the on-disk response cache so a run measures
    real API latency and cost even if the documents were classified before.
    """
    client = client or JevClient(model=model)
    endpoint = client.probe()

    questions = build_classification_questions(
        dataset.target_names,
        descriptions=CATEGORY_DESCRIPTIONS,
        allow_other=allow_other,
    )
    name_to_index = {name: i for i, name in enumerate(dataset.target_names)}

    indices = list(range(dataset.n_test))
    if limit is not None:
        indices = indices[:limit]

    def classify(i: int) -> DocResult:
        text = dataset.x_test[i]
        true_label = dataset.target_names[dataset.y_test[i]]
        start = time.perf_counter()
        try:
            body = client.decide(text, questions, use_cache=use_cache)
        except JevError as exc:
            return DocResult(
                index=i,
                true_label=true_label,
                pred_label=None,
                confidence=0.0,
                latency=time.perf_counter() - start,
                error=str(exc),
            )
        choice, confidence = parse_choice(body)
        return DocResult(
            index=i,
            true_label=true_label,
            pred_label=choice,
            confidence=confidence,
            latency=time.perf_counter() - start,
        )

    docs: list[DocResult] = []
    wall_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {executor.submit(classify, i): i for i in indices}
        iterator = as_completed(futures)
        if show_progress:
            iterator = tqdm(iterator, total=len(futures), desc=f"jev {model}")
        for future in iterator:
            docs.append(future.result())
    wall_clock = time.perf_counter() - wall_start

    docs.sort(key=lambda d: d.index)

    # Accumulate token usage/cost from the cached client totals.
    usage = {
        "input_tokens": client.usage.input_tokens,
        "output_tokens": client.usage.output_tokens,
        "cost": client.usage.cost,
    }

    y_true = [dataset.y_test[d.index] for d in docs]
    y_pred = [
        name_to_index.get(d.pred_label, -1) if d.pred_label else -1 for d in docs
    ]
    n_errors = sum(1 for d in docs if d.error is not None)

    latencies = [d.latency for d in docs]
    metrics = compute_metrics(y_true, y_pred, dataset.target_names)
    metrics["n_unparseable"] = sum(1 for d in docs if d.pred_label is None)
    metrics["n_errors"] = n_errors

    confusion_matrix = None
    report = ""
    if not n_errors or len(docs) - n_errors > 0:
        confusion_matrix = confusion(y_true, y_pred, dataset.n_classes)
        report = report_text(y_true, y_pred, dataset.target_names)

    throughput = len(docs) / wall_clock if wall_clock else 0.0

    return JevResult(
        name=f"jev:{model.split('/')[-1]}",
        model=model,
        endpoint=endpoint,
        metrics=metrics,
        report=report,
        confusion=confusion_matrix,
        latency=latency_summary(latencies),
        wall_clock_seconds=wall_clock,
        throughput_docs_per_s=throughput,
        n_errors=n_errors,
        n_truncated=client.stats["truncated"],
        usage=usage,
        client_stats=dict(client.stats),
        docs=docs,
    )


def check_determinism(
    dataset: Dataset,
    model: str,
    sample_size: int = 20,
    n_repeats: int = 3,
    concurrency: int = 4,
) -> dict[str, Any]:
    """Repeat a small sample and report choice/confidence stability."""
    client = JevClient(model=model)
    client.probe()
    questions = build_classification_questions(dataset.target_names)

    sample = list(range(min(sample_size, dataset.n_test)))
    records = []
    for i in sample:
        choices, confidences = [], []
        for _ in range(n_repeats):
            body = client.decide(dataset.x_test[i], questions, use_cache=False)
            choice, confidence = parse_choice(body)
            choices.append(choice)
            confidences.append(confidence)
        records.append(
            {
                "index": i,
                "choices": choices,
                "stable_choice": len(set(choices)) == 1,
                "max_confidence_spread": max(confidences) - min(confidences),
            }
        )

    n_stable = sum(1 for r in records if r["stable_choice"])
    spreads = [r["max_confidence_spread"] for r in records]
    return {
        "sample_size": len(records),
        "n_repeats": n_repeats,
        "stable_choice_rate": n_stable / len(records) if records else 0.0,
        "mean_confidence_spread": float(np.mean(spreads)) if spreads else 0.0,
        "max_confidence_spread": float(np.max(spreads)) if spreads else 0.0,
        "records": records,
    }


def format_wall_clock(seconds: float) -> str:
    return format_seconds(seconds)