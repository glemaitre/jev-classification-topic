"""Timing helpers: warm-up, repeated runs and percentile summaries."""

from __future__ import annotations

import gc
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar

T = TypeVar("T")


class Timer:
    """Context manager measuring wall-clock seconds (``perf_counter``)."""

    def __init__(self) -> None:
        self.elapsed: float = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        self.elapsed = time.perf_counter() - self._start


@dataclass
class TimingResult:
    """Repeated timing measurements of a single operation."""

    label: str
    seconds: list[float] = field(default_factory=list)

    @property
    def mean(self) -> float:
        return statistics.fmean(self.seconds) if self.seconds else 0.0

    @property
    def median(self) -> float:
        return statistics.median(self.seconds) if self.seconds else 0.0

    @property
    def stdev(self) -> float:
        return statistics.stdev(self.seconds) if len(self.seconds) > 1 else 0.0

    @property
    def min(self) -> float:
        return min(self.seconds) if self.seconds else 0.0

    @property
    def max(self) -> float:
        return max(self.seconds) if self.seconds else 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "mean": self.mean,
            "median": self.median,
            "stdev": self.stdev,
            "min": self.min,
            "max": self.max,
            "n_repeats": len(self.seconds),
        }


def repeat(
    operation: Callable[[], T],
    n_repeats: int = 3,
    warmup: bool = True,
) -> tuple[T, TimingResult]:
    """Run ``operation`` ``n_repeats`` times, timing each run.

    Returns the result of the final run and the timing measurements.
    A warm-up run is discarded when ``warmup`` is true (caches, lazy imports,
    thread pools, MPS kernel compilation...).
    """
    if warmup:
        operation()
        gc.collect()

    result: T = None  # type: ignore[assignment]
    timing = TimingResult(label=getattr(operation, "__name__", "operation"))
    for _ in range(n_repeats):
        gc.collect()
        with Timer() as timer:
            result = operation()
        timing.seconds.append(timer.elapsed)
    return result, timing


def latency_summary(latencies: list[float]) -> dict[str, float]:
    """Summary statistics for per-request latencies, in seconds."""
    if not latencies:
        return {}
    ordered = sorted(latencies)

    def pct(q: float) -> float:
        idx = min(int(round(q * (len(ordered) - 1))), len(ordered) - 1)
        return ordered[idx]

    return {
        "count": len(ordered),
        "total": sum(ordered),
        "mean": statistics.fmean(ordered),
        "median": statistics.median(ordered),
        "stdev": statistics.stdev(ordered) if len(ordered) > 1 else 0.0,
        "min": ordered[0],
        "p50": pct(0.50),
        "p90": pct(0.90),
        "p95": pct(0.95),
        "p99": pct(0.99),
        "max": ordered[-1],
    }


def format_seconds(seconds: float) -> str:
    """Human-friendly duration."""
    if seconds < 1:
        return f"{seconds * 1e3:.1f} ms"
    if seconds < 60:
        return f"{seconds:.2f} s"
    minutes, secs = divmod(seconds, 60)
    return f"{int(minutes)} min {secs:.1f} s"


def jsonable(obj: Any) -> Any:
    """Recursively convert numpy scalars / arrays to plain Python types."""
    import numpy as np

    if isinstance(obj, dict):
        return {k: jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj