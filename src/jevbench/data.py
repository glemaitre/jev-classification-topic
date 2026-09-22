"""Dataset loading and caching for 20 Newsgroups."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.datasets import fetch_20newsgroups

from .config import CACHE_DIR, CATEGORIES, CATEGORY_DESCRIPTIONS, RANDOM_STATE

REMOVE = ("headers", "footers", "quotes")


@dataclass
class Dataset:
    """A train/test split of 20 Newsgroups with integer-encoded labels."""

    x_train: list[str]
    y_train: list[int]
    x_test: list[str]
    y_test: list[int]
    target_names: list[str]

    @property
    def n_train(self) -> int:
        return len(self.x_train)

    @property
    def n_test(self) -> int:
        return len(self.x_test)

    @property
    def n_classes(self) -> int:
        return len(self.target_names)


def load_dataset(use_cache: bool = True) -> Dataset:
    """Load the full 20-class 20 Newsgroups dataset.

    Headers, footers and quotes are removed so that the newsgroup name never
    leaks into the document text — otherwise the label would be trivially
    recoverable.
    """
    train_cache = CACHE_DIR / "20ng_train.parquet"
    test_cache = CACHE_DIR / "20ng_test.parquet"

    if use_cache and train_cache.exists() and test_cache.exists():
        train = pd.read_parquet(train_cache)
        test = pd.read_parquet(test_cache)
    else:
        train = _fetch("train")
        test = _fetch("test")
        train.to_parquet(train_cache, index=False)
        test.to_parquet(test_cache, index=False)

    target_names = train.attrs.get("target_names")
    if target_names is None:
        target_names = list(CATEGORIES)

    return Dataset(
        x_train=train["text"].tolist(),
        y_train=train["target"].tolist(),
        x_test=test["text"].tolist(),
        y_test=test["target"].tolist(),
        target_names=list(target_names),
    )


def _fetch(subset: str) -> pd.DataFrame:
    bunch = fetch_20newsgroups(
        subset=subset,
        categories=list(CATEGORIES),
        shuffle=True,
        random_state=RANDOM_STATE,
        remove=REMOVE,
    )
    frame = pd.DataFrame({"text": bunch.data, "target": bunch.target})
    frame.attrs["target_names"] = list(bunch.target_names)
    return frame


def label_descriptions() -> dict[str, str]:
    """Mapping label -> short description used to frame the task."""
    return dict(CATEGORY_DESCRIPTIONS)