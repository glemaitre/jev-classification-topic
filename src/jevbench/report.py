"""Serialization and reporting for benchmark results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from .config import PLOTS_DIR, RESULTS_DIR  # noqa: E402
from .timing import jsonable  # noqa: E402

sns.set_theme(style="whitegrid")


# -- serialization ------------------------------------------------------------


def serialize_result(result: Any) -> dict[str, Any]:
    """Turn any of the three result dataclasses into a JSON-friendly dict."""
    payload: dict[str, Any] = {
        "method": result.name,
        "metrics": jsonable(result.metrics),
        "report": result.report,
        "timing": jsonable(result.timing_row()),
    }
    for extra in ("model", "model_name", "endpoint", "device", "embedding_dim",
                  "wall_clock_seconds", "throughput_docs_per_s", "n_errors",
                  "n_truncated", "usage", "client_stats", "latency",
                  "grid_search_seconds", "best_params", "best_cv_score", "scoring"):
        if hasattr(result, extra):
            payload[extra] = jsonable(getattr(result, extra))
    return payload


def save_method_json(payload: dict[str, Any], path: Path | None = None) -> Path:
    path = path or RESULTS_DIR / "metrics.json"
    existing: dict[str, Any] = {}
    if path.exists():
        existing = json.loads(path.read_text())
    existing[payload["method"]] = payload
    path.write_text(json.dumps(existing, indent=2))
    return path


def save_method_artifacts(result: Any, results_dir: Path = RESULTS_DIR) -> dict[str, Path]:
    """Save per-method JSON, confusion matrix, text report and confusion plot."""
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    slug = result.name.replace("/", "_").replace(":", "_")
    paths: dict[str, Path] = {}

    payload = serialize_result(result)
    metrics_path = results_dir / "metrics.json"
    save_method_json(payload, metrics_path)
    paths["metrics"] = metrics_path

    if result.confusion is not None:
        confusion_path = results_dir / f"confusion_{slug}.npy"
        np.save(confusion_path, result.confusion)
        paths["confusion"] = confusion_path
        paths["confusion_plot"] = save_confusion_plot(
            result.confusion, result.metrics, result.name,
            plots_dir / f"confusion_{slug}.png",
        )

    if result.report:
        report_path = results_dir / f"report_{slug}.txt"
        report_path.write_text(result.report)
        paths["report"] = report_path

    if hasattr(result, "docs") and result.docs:
        docs_path = results_dir / f"jev_docs_{slug}.csv"
        pd.DataFrame(
            [
                {
                    "index": d.index,
                    "true_label": d.true_label,
                    "pred_label": d.pred_label,
                    "correct": d.pred_label == d.true_label,
                    "confidence": d.confidence,
                    "latency_s": d.latency,
                    "error": d.error,
                }
                for d in result.docs
            ]
        ).to_csv(docs_path, index=False)
        paths["docs"] = docs_path
        paths["calibration_plot"] = save_confidence_calibration(
            result.docs, plots_dir / f"calibration_{slug}.png"
        )

    return paths


# -- plots --------------------------------------------------------------------


def _short_labels(target_names: list[str]) -> list[str]:
    return [name.replace("comp.", "c.").replace("talk.", "t.").replace("rec.", "r.")
            .replace("sci.", "s.").replace("soc.", "so.").replace("misc.", "m.")
            .replace("alt.", "a.") for name in target_names]


def save_confusion_plot(
    matrix: np.ndarray,
    metrics: dict[str, Any],
    title: str,
    path: Path,
) -> Path:
    names = list(metrics["per_class"].keys())
    labels = _short_labels(names)
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        matrix, annot=False, cmap="Blues", square=True,
        xticklabels=labels, yticklabels=labels, cbar_kws={"shrink": 0.7}, ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(
        f"{title}\naccuracy={metrics['accuracy']:.3f}  macro-F1={metrics['f1_macro']:.3f}"
    )
    plt.xticks(rotation=90, fontsize=6)
    plt.yticks(rotation=0, fontsize=6)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def save_confidence_calibration(docs: list, path: Path, n_bins: int = 10) -> Path:
    confidences = np.array([d.confidence for d in docs if d.pred_label is not None])
    correct = np.array(
        [d.pred_label == d.true_label for d in docs if d.pred_label is not None]
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].hist(confidences, bins=n_bins, range=(0, 1), color="#2E86AB", alpha=0.85)
    axes[0].set_xlabel("Jev confidence")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Confidence distribution")

    bins = np.linspace(0, 1, n_bins + 1)
    bin_idx = np.clip(np.digitize(confidences, bins) - 1, 0, n_bins - 1)
    xs, ys = [], []
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.sum() > 0:
            xs.append(confidences[mask].mean())
            ys.append(correct[mask].mean())
    axes[1].plot([0, 1], [0, 1], "--", color="grey", label="perfect calibration")
    axes[1].plot(xs, ys, "o-", color="#C0392B", label="Jev")
    axes[1].set_xlabel("Mean confidence")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Calibration")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def save_tfidf_heatmap(cv_results: pd.DataFrame, path: Path) -> Path:
    """Heatmap of mean CV score across two vectorizer parameters."""
    frame = cv_results.copy()
    frame["max_features"] = frame["param_vect__max_features"].map(
        {None: "all", 10000: "10k", 50000: "50k"}
    ).fillna(frame["param_vect__max_features"].astype(str))
    frame["ngram"] = frame["param_vect__ngram_range"].astype(str)
    frame["max_df"] = frame["param_vect__max_df"].astype(float)

    fig, axes = plt.subplots(1, frame["ngram"].nunique(), figsize=(12, 5), squeeze=False)
    for ax, (ngram, group) in zip(axes[0], frame.groupby("ngram")):
        pivot = group.pivot_table(
            index="max_df", columns="max_features", values="mean_test_score"
        )
        sns.heatmap(
            pivot, annot=True, fmt=".3f", cmap="viridis", ax=ax,
            cbar=ax is axes[0][-1],
        )
        ax.set_title(f"ngram_range={ngram}")
        ax.set_xlabel("max_features")
        ax.set_ylabel("max_df")
    fig.suptitle("TF-IDF grid search: mean CV score")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def save_timing_bar(timing: pd.DataFrame, path: Path) -> Path:
    """Bar chart comparing training and prediction times across methods."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    train = timing.assign(
        train_s=(
            timing.get("grid_search_seconds", 0)
            + timing.get("fit_median", 0)
            + timing.get("encode_train_median", 0)
        )
    )
    sns.barplot(data=train, x="method", y="train_s", ax=axes[0], color="#1A5276")
    axes[0].set_title("Training / fitting time (s)")
    axes[0].set_ylabel("seconds")
    axes[0].tick_params(axis="x", rotation=20)

    predict = timing.assign(
        predict_s=timing.get("predict_median", 0) + timing.get("encode_test_median", 0)
    )
    sns.barplot(data=predict, x="method", y="predict_s", ax=axes[1], color="#117A65")
    axes[1].set_title("Prediction time per full test set (s)")
    axes[1].set_ylabel("seconds")
    axes[1].tick_params(axis="x", rotation=20)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# -- summary report -----------------------------------------------------------


def build_timing_table(payloads: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for payload in payloads:
        row = {"method": payload["method"]}
        row.update(payload.get("timing", {}))
        for key in ("wall_clock_seconds", "throughput_docs_per_s", "usage"):
            if key in payload:
                row[key] = payload[key]
        rows.append(row)
    return pd.DataFrame(rows)


def build_metrics_table(payloads: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for payload in payloads:
        metrics = payload["metrics"]
        rows.append(
            {
                "method": payload["method"],
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["f1_macro"],
                "weighted_f1": metrics["f1_weighted"],
                "cohen_kappa": metrics["cohen_kappa"],
                "n_samples": metrics["n_samples"],
            }
        )
    return pd.DataFrame(rows)


def build_per_class_table(payloads: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for payload in payloads:
        for label, values in payload["metrics"]["per_class"].items():
            rows.append({"method": payload["method"], "label": label, **values})
    return pd.DataFrame(rows)


def _md_table(frame: pd.DataFrame, float_fmt: str = "{:.4f}") -> str:
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in frame.iterrows():
        cells = []
        for value in row:
            if isinstance(value, float):
                if value != value:  # NaN
                    cells.append("")
                else:
                    cells.append(float_fmt.format(value))
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_summary_report(
    payloads: list[dict[str, Any]],
    dataset_info: dict[str, Any],
    path: Path | None = None,
) -> Path:
    path = path or RESULTS_DIR / "REPORT.md"
    metrics_table = build_metrics_table(payloads)
    timing_table = build_timing_table(payloads)

    lines: list[str] = []
    lines.append("# 20 Newsgroups benchmark — TF-IDF, embeddings, HGB vs Jev\n")
    lines.append(
        f"- Train documents: **{dataset_info['n_train']}**; "
        f"test documents: **{dataset_info['n_test']}**; "
        f"classes: **{dataset_info['n_classes']}**."
    )
    lines.append(
        "- Documents have headers, footers and quotes removed to prevent label leakage."
    )
    lines.append("- Random state: 42. Local timings measured on this machine; "
                 "Jev timings include network latency.\n")

    lines.append("## Quality\n")
    lines.append(_md_table(metrics_table))
    lines.append("")

    timing_cols = [
        "method", "grid_search_seconds", "encode_train_median", "fit_median",
        "encode_test_median", "predict_median", "throughput_docs_per_s",
        "wall_clock_seconds", "cost_usd",
    ]
    timing_cols = [c for c in timing_cols if c in timing_table.columns]
    lines.append("## Time and cost\n")
    lines.append("`fit_*` are seconds to fit/refit the final model on the train split; "
                 "`predict_median` is seconds to predict the full test split. "
                 "Jev has no training and pays network latency per document.\n")
    lines.append(_md_table(timing_table[timing_cols]))
    lines.append("")

    lines.append("## Notes on fairness\n")
    lines.append(
        "- The local methods (TF-IDF, embeddings, HistGradientBoosting) are "
        "**supervised**: they train on the labelled train split. Jev is used "
        "**zero-shot** on the same test set."
    )
    lines.append(
        "- HistGradientBoosting needs dense input, so it runs on TF-IDF reduced by "
        "truncated SVD to 100 components (LSA); the aggressive reduction limits its "
        "accuracy relative to the sparse TF-IDF baseline."
    )
    lines.append(
        "- Jev's confidences are calibrated probabilities; the TF-IDF SVM exposes no "
        "probabilities, so no calibration comparison is made."
    )
    lines.append(
        "- Jev latency depends on OpenRouter load and network conditions; local "
        "methods depend on this machine's CPU/GPU."
    )

    path.write_text("\n".join(lines))
    return path


def save_tables(payloads: list[dict[str, Any]]) -> dict[str, Path]:
    timing = build_timing_table(payloads)
    per_class = build_per_class_table(payloads)
    metrics = build_metrics_table(payloads)
    paths = {
        "timing_csv": RESULTS_DIR / "timing.csv",
        "per_class_csv": RESULTS_DIR / "per_class.csv",
        "metrics_csv": RESULTS_DIR / "metrics_summary.csv",
    }
    timing.to_csv(paths["timing_csv"], index=False)
    per_class.to_csv(paths["per_class_csv"], index=False)
    metrics.to_csv(paths["metrics_csv"], index=False)
    timing_path = save_timing_bar(timing, PLOTS_DIR / "timing_bar.png")
    paths["timing_plot"] = timing_path
    return paths