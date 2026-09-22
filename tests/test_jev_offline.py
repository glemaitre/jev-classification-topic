"""Offline tests for the Jev client and evaluation pipeline.

Uses httpx.MockTransport so no network or API key is required.
"""

from __future__ import annotations

import httpx
import pytest

from jevbench.config import JEV_ENDPOINTS
from jevbench.data import Dataset
from jevbench.jev_client import (
    JevClient,
    build_classification_questions,
    parse_choice,
)
from jevbench.jev_eval import run_jev


def _mock_client(responses: list[dict]) -> httpx.Client:
    """A client whose endpoint returns canned responses in order."""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/chat/completions"):
            return httpx.Response(400, text="use /api/v1/systemone instead")
        index = min(calls["n"], len(responses) - 1)
        calls["n"] += 1
        return httpx.Response(200, json=responses[index])

    return httpx.Client(transport=httpx.MockTransport(handler))


def _choice_response(choice: str, confidence: float = 0.9) -> dict:
    return {
        "model": "typesafe/jev-1.13",
        "answers": {"category": {"type": "choice", "choice": choice,
                                 "confidence": confidence}},
        "usage": {"input_tokens": 100, "output_tokens": 5, "cost": 1e-5},
    }


@pytest.fixture()
def tiny_dataset() -> Dataset:
    names = ["comp.graphics", "sci.space"]
    return Dataset(
        x_train=["train a", "train b"],
        y_train=[0, 1],
        x_test=["gpu rendering", "orbit mars", "shader code", "rocket launch"],
        y_test=[0, 1, 0, 1],
        target_names=names,
    )


def test_probe_and_decide(tmp_path):
    client = JevClient(
        api_key="dummy",
        client=_mock_client([{"answers": {"ok": {"type": "noul", "noul": 0.9}}}]),
        cache_path=tmp_path / "cache.jsonl",
    )
    endpoint = client.probe()
    assert endpoint == JEV_ENDPOINTS[0]

    body = client.decide("some state", build_classification_questions(
        ["comp.graphics", "sci.space"]
    ))
    assert "answers" in body
    assert client.usage.input_tokens == 0  # probe response does not count


def test_cache_avoids_second_request(tmp_path):
    responses = [_choice_response("comp.graphics")]
    mock = _mock_client(responses)
    client = JevClient(api_key="dummy", client=mock, cache_path=tmp_path / "c.jsonl")
    questions = build_classification_questions(["comp.graphics", "sci.space"])
    client.decide("hello", questions)
    client.decide("hello", questions)
    assert client.stats["requests"] == 1
    assert client.stats["cache_hits"] == 1


def test_parse_choice_defensive():
    assert parse_choice({"answers": {"category": {"choice": "x", "confidence": 0.5}}}) == (
        "x",
        0.5,
    )
    assert parse_choice({"answers": {}}) == (None, 0.0)
    assert parse_choice({}) == (None, 0.0)


def test_run_jev_end_to_end(tiny_dataset, tmp_path):
    choices = ["comp.graphics", "sci.space", "comp.graphics", "sci.space"]
    responses = [_choice_response(c) for c in choices]
    # probe first, then one response per doc
    responses = [{"answers": {}}] + responses
    client = JevClient(
        api_key="dummy",
        client=_mock_client(responses),
        cache_path=tmp_path / "c.jsonl",
    )
    result = run_jev(tiny_dataset, model="typesafe/jev-1.13", concurrency=2,
                     show_progress=False, client=client)
    assert result.metrics["accuracy"] == 1.0
    assert result.n_errors == 0
    assert len(result.docs) == 4
    assert result.latency["count"] == 4
    assert result.confusion is not None
    assert result.confusion.shape == (2, 2)