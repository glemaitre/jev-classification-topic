"""OpenRouter client for TypeSafe's Jev (System One / Decisions API).

Jev is a *decisions* model: it does not generate prose. A request carries a
``state`` (the document) and a map of named ``questions``; the response carries
typed ``answers``. For classification we ask a single ``choice`` question whose
``criteria`` are the candidate classes.

Requests are retried with exponential backoff, and successful responses are
cached to a JSONL file keyed by (model, prompt version, state, questions) so a
re-run costs nothing.
"""

from __future__ import annotations

import hashlib
import json
import random
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from .config import (
    CACHE_DIR,
    CATEGORY_DESCRIPTIONS,
    JEV_ENDPOINTS,
    JEV_MAX_STATE_CHARS,
    JEV_MODEL_DEFAULT,
    JEV_PROMPT_VERSION,
    openrouter_api_key,
)

JEV_INSTRUCTIONS = (
    "You are classifying a Usenet newsgroup post. Choose the single newsgroup "
    "that this post was most likely posted to, based on its content and topic."
)

RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}


class JevError(RuntimeError):
    """Raised when a Jev request cannot be completed."""


@dataclass
class JevUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0

    def add(self, other: "JevUsage") -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cost += other.cost


@dataclass
class JevClient:
    """Minimal client for the OpenRouter System One endpoint."""

    api_key: str | None = None
    model: str = JEV_MODEL_DEFAULT
    endpoint: str | None = None
    timeout: float = 60.0
    max_retries: int = 5
    max_state_chars: int = JEV_MAX_STATE_CHARS
    prompt_version: str = JEV_PROMPT_VERSION
    cache_path: Path | None = None
    client: httpx.Client | None = None
    _cache: dict[str, dict] = field(default_factory=dict, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    usage: JevUsage = field(default_factory=JevUsage, init=False)
    stats: dict[str, int] = field(
        default_factory=lambda: {
            "requests": 0,
            "cache_hits": 0,
            "retries": 0,
            "errors": 0,
            "truncated": 0,
        },
        init=False,
    )

    def __post_init__(self) -> None:
        self.api_key = self.api_key or openrouter_api_key()
        if not self.api_key:
            raise JevError(
                "No OpenRouter API key. Set OPENROUTER_API_KEY (see .env.example)."
            )
        self.cache_path = self.cache_path or CACHE_DIR / "jev_responses.jsonl"
        self.client = self.client or httpx.Client(timeout=self.timeout)
        self._load_cache()

    # -- cache ---------------------------------------------------------------

    def _load_cache(self) -> None:
        if self.cache_path and self.cache_path.exists():
            with self.cache_path.open() as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    self._cache[record["key"]] = record["response"]

    def _cache_key(self, state: str, questions: dict) -> str:
        payload = json.dumps(
            {
                "model": self.model,
                "prompt_version": self.prompt_version,
                "state": state,
                "questions": questions,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _store(self, key: str, response: dict) -> None:
        self._cache[key] = response
        if self.cache_path:
            with self._lock:
                with self.cache_path.open("a") as handle:
                    handle.write(json.dumps({"key": key, "response": response}) + "\n")

    # -- endpoint discovery --------------------------------------------------

    def probe(self) -> str:
        """Return a working endpoint, trying the known candidates in order."""
        if self.endpoint:
            return self.endpoint
        payload = {
            "model": self.model,
            "state": "Hello world.",
            "questions": {
                "ok": {"type": "noul", "instructions": "Is this a greeting?"}
            },
        }
        errors: list[str] = []
        for candidate in JEV_ENDPOINTS:
            try:
                response = self.client.post(
                    candidate,
                    headers=self._headers(),
                    json=payload,
                )
            except httpx.HTTPError as exc:  # pragma: no cover - network dependent
                errors.append(f"{candidate}: {exc}")
                continue
            if response.status_code == 200:
                self.endpoint = candidate
                return candidate
            errors.append(f"{candidate}: HTTP {response.status_code} {response.text[:200]}")
        raise JevError("No working System One endpoint found:\n" + "\n".join(errors))

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            http_client_header(): "jevbench/0.1",
        }

    # -- request -------------------------------------------------------------

    def decide(self, state: str, questions: dict, use_cache: bool = True) -> dict:
        """Post one state + questions, with caching and retries."""
        if not state.strip():
            state = "(empty document)"
        if len(state) > self.max_state_chars:
            state = state[: self.max_state_chars]
            self.stats["truncated"] += 1

        key = self._cache_key(state, questions)
        if use_cache and key in self._cache:
            self.stats["cache_hits"] += 1
            return self._cache[key]

        endpoint = self.probe()
        payload = {"model": self.model, "state": state, "questions": questions}

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.post(
                    endpoint, headers=self._headers(), json=payload
                )
            except httpx.HTTPError as exc:
                last_error = exc
                if attempt == self.max_retries:
                    break
                self._sleep_backoff(attempt)
                self.stats["retries"] += 1
                continue

            if response.status_code == 200:
                body = response.json()
                self.stats["requests"] += 1
                usage = body.get("usage") or {}
                self.usage.add(
                    JevUsage(
                        input_tokens=int(usage.get("input_tokens", 0) or 0),
                        output_tokens=int(usage.get("output_tokens", 0) or 0),
                        cost=float(usage.get("cost", 0.0) or 0.0),
                    )
                )
                self._store(key, body)
                return body

            # A 400 telling us to use another endpoint: retry once against it.
            if response.status_code in {400, 404} and attempt == 0:
                moved = _parse_suggested_endpoint(response.text)
                if moved and moved != endpoint:
                    endpoint = moved
                    self.endpoint = moved
                    continue

            last_error = JevError(
                f"HTTP {response.status_code}: {response.text[:300]}"
            )
            if response.status_code not in RETRYABLE_STATUS:
                break
            if attempt == self.max_retries:
                break
            self._sleep_backoff(attempt, response=response)
            self.stats["retries"] += 1

        self.stats["errors"] += 1
        raise JevError(f"Jev request failed for model {self.model}: {last_error}")

    def _sleep_backoff(self, attempt: int, response: httpx.Response | None = None) -> None:
        retry_after = None
        if response is not None:
            retry_after = response.headers.get("Retry-After")
        if retry_after:
            try:
                delay = float(retry_after)
            except ValueError:
                delay = 0.0
        else:
            delay = min(2.0**attempt, 30.0)
        time.sleep(delay * (0.5 + random.random() * 0.5))


def http_client_header() -> str:
    """OpenRouter-optional attribution header name."""
    return "X-Title"


def _parse_suggested_endpoint(text: str) -> str | None:
    """Extract '/api/...' endpoint hints from an error body, if present."""
    import re

    match = re.search(r"(/api/[A-Za-z0-9_/\-]+)", text)
    if not match:
        return None
    path = match.group(1)
    return f"https://openrouter.ai{path}"


def build_classification_questions(
    target_names: list[str],
    descriptions: dict[str, str] | None = None,
    instructions: str = JEV_INSTRUCTIONS,
    allow_other: bool = False,
) -> dict[str, Any]:
    """Build the single ``choice`` question used to classify a document."""
    descriptions = descriptions or CATEGORY_DESCRIPTIONS
    criteria = {name: descriptions.get(name, name) for name in target_names}
    if allow_other:
        criteria["other"] = "None of the above newsgroups fits this post."
    return {
        "category": {
            "type": "choice",
            "instructions": instructions,
            "criteria": criteria,
        }
    }


def parse_choice(body: dict, question: str = "category") -> tuple[str | None, float]:
    """Return ``(choice, confidence)`` for a choice answer, defensively."""
    answer = (body.get("answers") or {}).get(question) or {}
    choice = answer.get("choice") or answer.get("value") or answer.get("label")
    confidence = answer.get("confidence", answer.get("probability", 0.0))
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        confidence = 0.0
    return choice, confidence