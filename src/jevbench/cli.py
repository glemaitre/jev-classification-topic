"""Command-line entry point for the benchmark."""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any

from .config import (
    JEV_MODEL_DEFAULT,
    PLOTS_DIR,
    RESULTS_DIR,
    openrouter_api_key,
)
from .data import load_dataset
from .report import (
    save_method_artifacts,
    save_tfidf_heatmap,
    write_summary_report,
    save_tables,
)
from .timing import format_seconds


def _dataset_info(dataset) -> dict[str, Any]:
    return {
        "n_train": dataset.n_train,
        "n_test": dataset.n_test,
        "n_classes": dataset.n_classes,
        "target_names": dataset.target_names,
    }


def _print_header(text: str) -> None:
    print(f"\n{'=' * 70}\n{text}\n{'=' * 70}")


def cmd_data(args: argparse.Namespace) -> int:
    dataset = load_dataset(use_cache=not args.no_cache)
    info = _dataset_info(dataset)
    print(f"train={info['n_train']} test={info['n_test']} classes={info['n_classes']}")
    print("categories:", ", ".join(info["target_names"]))
    return 0


def cmd_tfidf(args: argparse.Namespace) -> int:
    from .tfidf import run_tfidf

    dataset = load_dataset()
    _print_header(f"TF-IDF + LinearSVC grid search (cv={args.cv}, scoring={args.scoring})")
    result = run_tfidf(
        dataset,
        cv=args.cv,
        scoring=args.scoring,
        n_jobs=args.n_jobs,
        n_repeats=args.n_repeats,
        verbose=args.verbose,
    )
    _print_header("Results")
    print(f"best params: {result.best_params}")
    print(f"best CV {result.scoring}: {result.best_cv_score:.4f}")
    print(f"grid search: {format_seconds(result.grid_search_seconds)}")
    print(f"refit: {format_seconds(result.fit_timing['median'])}")
    print(f"predict: {format_seconds(result.predict_timing['median'])} "
          f"({result.throughput_docs_per_s:.0f} docs/s)")
    print(f"test accuracy={result.metrics['accuracy']:.4f} "
          f"macro-F1={result.metrics['f1_macro']:.4f}")
    print(result.report)

    paths = save_method_artifacts(result)
    if result.cv_results is not None:
        paths["heatmap"] = save_tfidf_heatmap(
            result.cv_results, PLOTS_DIR / "tfidf_heatmap.png"
        )
    print("saved:", ", ".join(str(p) for p in paths.values()))
    return 0


def cmd_embed(args: argparse.Namespace) -> int:
    from .embed import run_embed

    dataset = load_dataset()
    _print_header(f"Embeddings + LogisticRegression: {args.model}")
    result = run_embed(
        dataset,
        model_name=args.model,
        batch_size=args.batch_size,
        n_repeats=args.n_repeats,
        use_cache=not args.no_cache,
        C=args.C,
        show_progress=not args.quiet,
    )
    _print_header("Results")
    print(f"device: {result.device}, dim: {result.embedding_dim}")
    if result.train_encode_timing:
        print(f"encode train: {format_seconds(result.train_encode_timing['median'])}")
        print(f"encode test:  {format_seconds(result.test_encode_timing['median'])}")
    else:
        print("encode: loaded cached embeddings (timings unavailable)")
    print(f"fit: {format_seconds(result.fit_timing['median'])}")
    print(f"predict: {format_seconds(result.predict_timing['median'])} "
          f"({result.throughput_docs_per_s:.0f} docs/s)")
    print(f"test accuracy={result.metrics['accuracy']:.4f} "
          f"macro-F1={result.metrics['f1_macro']:.4f}")
    print(result.report)

    paths = save_method_artifacts(result)
    print("saved:", ", ".join(str(p) for p in paths.values()))
    return 0


def cmd_jev(args: argparse.Namespace) -> int:
    from .jev_eval import check_determinism, run_jev

    if not openrouter_api_key():
        print("ERROR: OPENROUTER_API_KEY is not set. Copy .env.example to .env and add "
              "your key.", file=sys.stderr)
        return 2

    dataset = load_dataset()
    _print_header(f"Jev zero-shot classification: {args.model}")
    result = run_jev(
        dataset,
        model=args.model,
        concurrency=args.concurrency,
        limit=args.limit,
        allow_other=args.allow_other,
        use_cache=not getattr(args, "no_cache", False),
    )
    _print_header("Results")
    print(f"endpoint: {result.endpoint}")
    print(f"wall clock: {format_seconds(result.wall_clock_seconds)} "
          f"({result.throughput_docs_per_s:.1f} docs/s, concurrency={args.concurrency})")
    if result.client_stats.get("cache_hits"):
        print(f"cache hits: {result.client_stats['cache_hits']} / {len(result.docs)} "
              f"— timings/cost reflect only the {result.client_stats.get('requests', 0)} "
              f"live requests (use --no-cache to force a full live run)")
    print(f"latency mean={result.latency.get('mean', 0):.3f}s "
          f"p50={result.latency.get('p50', 0):.3f}s "
          f"p95={result.latency.get('p95', 0):.3f}s")
    print(f"tokens in={result.usage['input_tokens']} out={result.usage['output_tokens']} "
          f"cost=${result.usage['cost']:.4f}")
    print(f"errors={result.n_errors} truncated={result.n_truncated}")
    print(f"test accuracy={result.metrics['accuracy']:.4f} "
          f"macro-F1={result.metrics['f1_macro']:.4f}")
    print(result.report)

    paths = save_method_artifacts(result)

    if args.determinism:
        _print_header("Determinism check")
        det = check_determinism(
            dataset, args.model,
            sample_size=args.determinism_size,
            n_repeats=args.determinism_repeats,
            concurrency=args.concurrency,
        )
        det_path = RESULTS_DIR / "jev_determinism.json"
        det_path.write_text(json.dumps(det, indent=2))
        print(f"stable choice rate: {det['stable_choice_rate']:.2%}")
        print(f"mean confidence spread: {det['mean_confidence_spread']:.4f}")
        paths["determinism"] = det_path

    print("saved:", ", ".join(str(p) for p in paths.values()))
    return 0 if result.n_errors == 0 else 1


def cmd_report(args: argparse.Namespace) -> int:
    metrics_path = RESULTS_DIR / "metrics.json"
    if not metrics_path.exists():
        print("No results/metrics.json yet. Run a method first.", file=sys.stderr)
        return 1
    payloads = list(json.loads(metrics_path.read_text()).values())
    dataset = load_dataset()
    paths = save_tables(payloads)
    report_path = write_summary_report(payloads, _dataset_info(dataset))
    paths["report"] = report_path
    _print_header("Summary")
    print(f"wrote {report_path}")
    for name, path in paths.items():
        print(f"  {name}: {path}")
    return 0


def cmd_all(args: argparse.Namespace) -> int:
    exit_code = 0
    if not args.skip_tfidf:
        exit_code |= cmd_tfidf(_tfidf_namespace(args))
    if not args.skip_embed:
        exit_code |= cmd_embed(_embed_namespace(args))

    if not args.skip_jev:
        if openrouter_api_key():
            exit_code |= cmd_jev(_jev_namespace(args))
        else:
            print("\n[skip] Jev: OPENROUTER_API_KEY not set.", file=sys.stderr)

    cmd_report(argparse.Namespace())
    return exit_code


# -- argument parsing ---------------------------------------------------------


def _tfidf_namespace(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        cv=args.cv, scoring=args.scoring, n_jobs=args.n_jobs,
        n_repeats=args.repeats, verbose=args.verbose,
    )


def _embed_namespace(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        model=args.embed_model, batch_size=args.batch_size,
        n_repeats=args.repeats, no_cache=args.no_cache, C=args.C,
        quiet=args.quiet,
    )


def _jev_namespace(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        model=args.jev_model, concurrency=args.concurrency, limit=args.jev_limit,
        allow_other=args.allow_other, determinism=args.determinism,
        determinism_size=args.determinism_size,
        determinism_repeats=args.determinism_repeats,
        no_cache=args.no_cache,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jevbench",
        description="Benchmark TF-IDF vs LM embeddings vs TypeSafe Jev on 20 Newsgroups.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_data = sub.add_parser("data", help="download/cache the dataset and print stats")
    p_data.add_argument("--no-cache", action="store_true")
    p_data.set_defaults(func=cmd_data)

    p_tfidf = sub.add_parser("tfidf", help="run the TF-IDF + LinearSVC grid search")
    p_tfidf.add_argument("--cv", type=int, default=5)
    p_tfidf.add_argument("--scoring", default="accuracy")
    p_tfidf.add_argument("--n-jobs", type=int, default=-1)
    p_tfidf.add_argument("--n-repeats", type=int, default=3)
    p_tfidf.add_argument("--verbose", type=int, default=1)
    p_tfidf.set_defaults(func=cmd_tfidf)

    p_embed = sub.add_parser("embed", help="run the embedding + logistic regression")
    p_embed.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    p_embed.add_argument("--batch-size", type=int, default=64)
    p_embed.add_argument("--n-repeats", type=int, default=3)
    p_embed.add_argument("--C", type=float, default=10.0)
    p_embed.add_argument("--no-cache", action="store_true")
    p_embed.add_argument("--quiet", action="store_true", help="hide encoding progress bars")
    p_embed.set_defaults(func=cmd_embed)

    p_jev = sub.add_parser("jev", help="run zero-shot classification with Jev")
    p_jev.add_argument("--model", default=JEV_MODEL_DEFAULT)
    p_jev.add_argument("--concurrency", type=int, default=8)
    p_jev.add_argument("--limit", type=int, default=None,
                       help="only classify the first N test documents")
    p_jev.add_argument("--allow-other", action="store_true",
                       help="add an 'other' option to the criteria")
    p_jev.add_argument("--determinism", action="store_true")
    p_jev.add_argument("--determinism-size", type=int, default=20)
    p_jev.add_argument("--determinism-repeats", type=int, default=3)
    p_jev.add_argument("--no-cache", action="store_true",
                       help="bypass the response cache and call the API for every doc")
    p_jev.set_defaults(func=cmd_jev)

    p_report = sub.add_parser("report", help="aggregate results and write the report")
    p_report.set_defaults(func=cmd_report)

    p_all = sub.add_parser("all", help="run every available method and report")
    p_all.add_argument("--skip-tfidf", action="store_true")
    p_all.add_argument("--skip-embed", action="store_true")
    p_all.add_argument("--skip-jev", action="store_true")
    p_all.add_argument("--cv", type=int, default=5)
    p_all.add_argument("--scoring", default="accuracy")
    p_all.add_argument("--n-jobs", type=int, default=-1)
    p_all.add_argument("--repeats", type=int, default=3)
    p_all.add_argument("--verbose", type=int, default=1)
    p_all.add_argument("--embed-model",
                       default="sentence-transformers/all-MiniLM-L6-v2")
    p_all.add_argument("--batch-size", type=int, default=64)
    p_all.add_argument("--C", type=float, default=10.0)
    p_all.add_argument("--no-cache", action="store_true")
    p_all.add_argument("--quiet", action="store_true")
    p_all.add_argument("--jev-model", default=JEV_MODEL_DEFAULT)
    p_all.add_argument("--concurrency", type=int, default=8)
    p_all.add_argument("--jev-limit", type=int, default=None)
    p_all.add_argument("--allow-other", action="store_true")
    p_all.add_argument("--determinism", action="store_true")
    p_all.add_argument("--determinism-size", type=int, default=20)
    p_all.add_argument("--determinism-repeats", type=int, default=3)
    p_all.set_defaults(func=cmd_all)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    start = time.perf_counter()
    code = args.func(args)
    print(f"\nTotal: {format_seconds(time.perf_counter() - start)}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())