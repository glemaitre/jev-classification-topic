# Jev benchmark: TF-IDF vs LM embeddings vs TypeSafe Jev

Compare three ways of classifying the full **20 Newsgroups** dataset (all 20
classes, 11,314 train / 7,532 test documents):

1. **TF-IDF + LinearSVC** with grid search — the classic sparse lexical baseline,
   mirroring the scikit-learn example
   [`plot_grid_search_text_feature_extraction.py`](https://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_text_feature_extraction.html).
2. **Sentence embeddings + LogisticRegression** — a language-model text encoder
   (`sentence-transformers/all-MiniLM-L6-v2`) that captures semantics, with a
   linear classifier trained on the frozen embeddings.
3. **Jev** (`typesafe/jev-1.13` via OpenRouter) — TypeSafe's *System One*
   decisions model, used **zero-shot**: each post is sent as `state` with a single
   `choice` question whose criteria are the 20 newsgroup names.

The benchmark reports **quality** (accuracy, macro/weighted F1, Cohen's kappa,
per-class scores, confusion matrices) and **time/cost** (training vs prediction,
throughput, API latency and token cost).

> Jev is a decisions model, not a chat model. It does not generate text; it
> returns typed answers with calibrated probabilities. See
> [OpenRouter's System One docs](https://openrouter.ai/docs/guides/community/typesafe-sdk).

## Setup

Requires [uv](https://docs.astral.sh/uv/) (Python 3.12 is fetched automatically).

```bash
uv sync
cp .env.example .env        # then paste your OpenRouter key
```

Get a key at <https://openrouter.ai/settings/keys>. Jev costs about
**$0.042/M input tokens** and **$0 output**, so a full test-set run is on the
order of a few cents. Verify access (and discover the live endpoint) with:

```bash
uv run python scripts/probe_openrouter.py
```

## Running

```bash
# everything that is available; skips Jev with a message if no key is set
uv run jevbench all

# or one method at a time
uv run jevbench data                 # download/cache the dataset, print stats
uv run jevbench tfidf                # grid search + timings
uv run jevbench embed                # MiniLM embeddings + logistic regression
uv run jevbench jev --concurrency 8  # zero-shot Jev on the full test set
uv run jevbench report               # aggregate results -> REPORT.md + plots
```

Useful options:

| Command | Option | Meaning |
| --- | --- | --- |
| `tfidf` | `--cv 5 --scoring accuracy` | CV folds / scoring (grid from the sklearn example) |
| `embed` | `--model sentence-transformers/all-MiniLM-L6-v2 --C 10` | encoder and regularisation |
| `jev` | `--model typesafe/jev-1.13` | pinned model id (`~typesafe/jev-latest` for the alias) |
| `jev` | `--concurrency 8` | parallel in-flight requests |
| `jev` | `--limit 500` | only classify the first N test documents |
| `jev` | `--determinism` | repeat a sample to check answer stability |

A dry-run of the Jev pipeline uses a mock transport and needs no key:

```bash
uv run pytest -q
```

## Outputs

Written to `results/`:

- `metrics.json` — per-method metrics, timings and usage
- `timing.csv`, `per_class.csv`, `metrics_summary.csv`
- `confusion_<method>.npy` and `plots/confusion_<method>.png`
- `plots/timing_bar.png`, `plots/tfidf_heatmap.png`
- `plots/calibration_<method>.png` — Jev confidence distribution and calibration
- `jev_docs_<method>.csv` — per-document prediction, confidence, latency
- `report_<method>.txt` — full sklearn classification report
- `REPORT.md` — the combined summary

`cache/` holds the dataset, embeddings and raw Jev responses (so re-runs are
free), and is git-ignored along with `results/` and `.env`.

## Notebook

```bash
uv run jupyter lab notebooks/benchmark.ipynb
```

The notebook loads `results/` and renders the quality table, timing table,
confusion matrices, calibration plot and `REPORT.md` inline.

## Methodology and fairness caveats

- Documents are loaded with `remove=("headers", "footers", "quotes")` so the
  newsgroup name cannot leak into the text.
- TF-IDF and embeddings are **supervised** (trained on the labelled train
  split); Jev is **zero-shot** on the same test set. This is the intended
  comparison — a trained lexical baseline and a trained semantic encoder versus a
  decision model used out of the box.
- Local timings are measured with a warm-up run and the median of repeated runs
  on this machine (Apple MPS/CPU). Jev timings include network latency and vary
  with OpenRouter load; per-call latency percentiles are reported separately.
- Repeats: `--n-repeats` (default 3) for local fit/predict; encode timings use the
  same number of full encoding passes.

## Project layout

```
src/jevbench/
  config.py      categories, descriptions, paths, Jev model ids
  data.py        20 Newsgroups loading + parquet cache
  timing.py      timers, repeated runs, latency percentiles
  metrics.py     accuracy/F1/kappa, per-class report, confusion
  tfidf.py       method A
  embed.py       method B
  jev_client.py  OpenRouter System One client (probe, retries, cache)
  jev_eval.py    method C + determinism check
  report.py      tables, plots, REPORT.md
  cli.py         `jevbench` entry point
scripts/probe_openrouter.py
notebooks/benchmark.ipynb
tests/test_jev_offline.py
```