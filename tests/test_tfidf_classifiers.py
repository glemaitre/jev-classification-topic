"""Offline tests for the TF-IDF classifiers (linsvc, lr, nb)."""

from __future__ import annotations

import numpy as np
import pytest

from jevbench.data import Dataset
from jevbench.tfidf import CLASSIFIER_FACTORIES, build_pipeline, run_tfidf

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
            x_train.append(" ".join(rng.choice(phrase.split(), size=8, replace=True)))
            y_train.append(label)
    return Dataset(
        x_train=x_train, y_train=y_train,
        x_test=["rocket orbit mars nasa", "image pixel shader gpu", "puck ice nhl goal"],
        y_test=[0, 1, 2],
        target_names=["sci.space", "comp.graphics", "rec.sport.hockey"],
    )


@pytest.mark.parametrize("classifier", sorted(CLASSIFIER_FACTORIES))
def test_tfidf_classifiers_run(classifier):
    dataset = _tiny_dataset()
    grid = {
        "vect__max_df": [1.0],
        "vect__ngram_range": [(1, 1)],
        "clf__C": [1.0],
        "clf__alpha": [1.0],
    }
    # prune the grid to params that exist for the chosen classifier
    grid = {k: v for k, v in grid.items() if _has_param(build_pipeline(classifier), k)}
    result = run_tfidf(
        dataset, classifier=classifier, param_grid=grid, cv=2,
        n_jobs=1, n_repeats=1, verbose=0,
    )
    assert result.name == f"tfidf:{classifier}"
    assert 0.0 <= result.metrics["accuracy"] <= 1.0
    assert result.confusion.shape == (3, 3)
    assert result.report


def _has_param(pipeline, name: str) -> bool:
    step, param = name.split("__", 1)
    if step not in pipeline.named_steps:
        return False
    return hasattr(pipeline.named_steps[step], param)