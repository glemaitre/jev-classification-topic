# 20 Newsgroups benchmark — TF-IDF, embeddings, HGB vs Jev

- Train documents: **11314**; test documents: **7532**; classes: **20**.
- Documents have headers, footers and quotes removed to prevent label leakage.
- Random state: 42. Local timings measured on this machine; Jev timings include network latency.

## Quality

| method | accuracy | macro_f1 | weighted_f1 | cohen_kappa | n_samples |
| --- | --- | --- | --- | --- | --- |
| tfidf | 0.6985 | 0.6842 | 0.6946 | 0.6823 | 7532 |
| embed:all-MiniLM-L6-v2 | 0.6612 | 0.6501 | 0.6616 | 0.6431 | 7532 |
| jev:jev-1.13 | 0.7200 | 0.7168 | 0.7316 | 0.7052 | 7532 |
| hgb:lsa100 | 0.5386 | 0.5290 | 0.5404 | 0.5139 | 7532 |

## Time and cost

`fit_*` are seconds to fit/refit the final model on the train split; `predict_median` is seconds to predict the full test split. Jev has no training and pays network latency per document.

| method | grid_search_seconds | encode_train_median | fit_median | encode_test_median | predict_median | throughput_docs_per_s | wall_clock_seconds | cost_usd |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tfidf | 95.7919 | 0.0000 | 6.7660 | 0.0000 | 0.9362 | 8044.9153 |  |  |
| embed:all-MiniLM-L6-v2 | 0.0000 | 27.2805 | 0.3717 | 18.7479 | 0.0014 | 5268820.4821 |  |  |
| jev:jev-1.13 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.2834 | 53.9417 | 139.6323 | 0.3427 |
| hgb:lsa100 | 0.0000 | 13.9498 | 66.0266 | 2.0714 | 0.6800 | 11075.6949 |  |  |

## Notes on fairness

- The local methods (TF-IDF, embeddings, HistGradientBoosting) are **supervised**: they train on the labelled train split. Jev is used **zero-shot** on the same test set.
- HistGradientBoosting needs dense input, so it runs on TF-IDF reduced by truncated SVD to 100 components (LSA); the aggressive reduction limits its accuracy relative to the sparse TF-IDF baseline.
- Jev's confidences are calibrated probabilities; the TF-IDF SVM exposes no probabilities, so no calibration comparison is made.
- Jev latency depends on OpenRouter load and network conditions; local methods depend on this machine's CPU/GPU.