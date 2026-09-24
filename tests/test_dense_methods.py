"""Offline tests for the dense method (LSA features + HistGradientBoosting)."""

from __future__ import annotations

import numpy as np
import pytest

from jevbench import features as features_module
from jevbench.data import Dataset
from jevbench.features import build_lsa_features
from jevbench.hgb import run_hgb

VOCAB = [
    "space rocket orbit mars nasa satellite launch astronaut moon",
    "graphics image render pixel shader gpu opengl texture",
    "hockey goal puck ice player nhl team score",
]


def _tiny_dataset() -> Dataset:
    rng = np.random.default_rng(0)
    x_train, y_train = [], []
    for label, phrase in enumerate(VOCAB):
        for _ in range(20):
            words = rng.choice(phrase.split(), size=8, replace=True)
            x_train.append(" ".join(words))
            y_train.append(label)
    x_test = ["rocket orbit mars nasa", "image pixel shader gpu", "puck ice nhl goal"]
    y_test = [0, 1, 2]
    return Dataset(
        x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
        target_names=["sci.space", "comp.graphics", "rec.sport.hockey"],
    )


@pytest.fixture()
def patched_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(
        features_module, "_cache_path", lambda n: tmp_path / f"lsa_{n}.npz"
    )


def test_lsa_features_shapes_and_cache(patched_cache):
    dataset = _tiny_dataset()
    first = build_lsa_features(dataset, n_components=5, n_repeats=1, use_cache=True)
    assert first.x_train.shape == (60, 5)
    assert first.x_test.shape == (3, 5)
    assert first.x_train.dtype == np.float32

    second = build_lsa_features(dataset, n_components=5, n_repeats=1, use_cache=True)
    np.testing.assert_allclose(first.x_train, second.x_train)
    # Cache hit => no timing recorded for the fit.
    assert second.fit_timing == {}


def test_hgb_runs_and_scores(patched_cache):
    dataset = _tiny_dataset()
    result = run_hgb(dataset, n_components=5, n_repeats=1, use_cache=True)
    assert result.name == "hgb:lsa5"
    assert 0.0 <= result.metrics["accuracy"] <= 1.0
    assert result.confusion.shape == (3, 3)
    assert result.report
    row = result.timing_row()
    assert row["method"] == "hgb:lsa5"
    assert "fit_median" in row and "predict_median" in row