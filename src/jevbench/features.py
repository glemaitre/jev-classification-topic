"""Shared dense feature extraction: TF-IDF followed by truncated SVD (LSA).

HistGradientBoosting cannot consume sparse TF-IDF matrices directly, so this
module reduces them to a dense, low-dimensional representation. The dense
matrices are cached to disk and shared across dense-input methods.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from .config import CACHE_DIR, RANDOM_STATE
from .data import Dataset
from .timing import Timer, TimingResult

VECTORIZER_KWARGS = {
    "sublinear_tf": True,
    "max_df": 0.5,
    "min_df": 2,
    "ngram_range": (1, 2),
}


@dataclass
class LsaFeatures:
    x_train: np.ndarray
    x_test: np.ndarray
    n_components: int
    fit_timing: dict[str, float] = field(default_factory=dict)
    transform_timing: dict[str, float] = field(default_factory=dict)


def _cache_path(n_components: int) -> Path:
    return CACHE_DIR / f"lsa_{n_components}.npz"


def build_lsa_features(
    dataset: Dataset,
    n_components: int = 100,
    n_repeats: int = 3,
    use_cache: bool = True,
) -> LsaFeatures:
    """Fit TF-IDF + SVD on the train split and transform both splits."""
    path = _cache_path(n_components)
    if use_cache and path.exists():
        data = np.load(path)
        return LsaFeatures(
            x_train=data["x_train"],
            x_test=data["x_test"],
            n_components=n_components,
        )

    def fit() -> tuple[TfidfVectorizer, TruncatedSVD]:
        vectorizer = TfidfVectorizer(**VECTORIZER_KWARGS)
        matrix = vectorizer.fit_transform(dataset.x_train)
        svd = TruncatedSVD(n_components=n_components, random_state=RANDOM_STATE)
        svd.fit(matrix)
        return vectorizer, svd

    fit_timing = TimingResult(label="lsa_fit")
    for _ in range(n_repeats):
        with Timer() as timer:
            vectorizer, svd = fit()
        fit_timing.seconds.append(timer.elapsed)

    def transform(texts: list[str]) -> np.ndarray:
        return svd.transform(vectorizer.transform(texts)).astype(np.float32)

    transform_timing = TimingResult(label="lsa_transform")
    x_test = transform(dataset.x_test)
    for _ in range(n_repeats):
        with Timer() as timer:
            x_test = transform(dataset.x_test)
        transform_timing.seconds.append(timer.elapsed)

    x_train = svd.transform(vectorizer.transform(dataset.x_train)).astype(np.float32)

    np.savez_compressed(path, x_train=x_train, x_test=x_test)

    return LsaFeatures(
        x_train=x_train,
        x_test=x_test,
        n_components=n_components,
        fit_timing=fit_timing.as_dict(),
        transform_timing=transform_timing.as_dict(),
    )