#!/usr/bin/env python
"""Probe how OpenRouter serves the TypeSafe Jev model.

Tries the known System One / Decisions endpoints and model-id spellings, prints
which combination works, and shows the raw response. Requires
OPENROUTER_API_KEY in the environment or in a .env file at the project root.

Usage:
    uv run python scripts/probe_openrouter.py [--model typesafe/jev-1.13]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

ENDPOINTS = [
    "https://openrouter.ai/api/v1/systemone",
    "https://openrouter.ai/api/alpha/decisions",
    "https://openrouter.ai/api/v1/chat/completions",
]

MODEL_IDS = [
    "typesafe/jev-1.13",
    "typesafe/jev-1.13-20260917",
    "jev-1.13",
    "jev-latest",
    "~typesafe/jev-latest",
]

SAMPLE_STATE = "I was charged twice for my subscription and want my money back."
SAMPLE_QUESTIONS = {
    "department": {
        "type": "choice",
        "instructions": "Which team should handle this?",
        "criteria": {"billing": "Charges and refunds", "technical": "Bugs and outages"},
    }
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, help="only probe this model id")
    args = parser.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("OPENROUTER_API_KEY is not set (env or .env).", file=sys.stderr)
        return 2

    models = [args.model] if args.model else MODEL_IDS
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    working: list[tuple[str, str]] = []
    with httpx.Client(timeout=60.0) as client:
        for endpoint in ENDPOINTS:
            for model in models:
                payload = {
                    "model": model,
                    "state": SAMPLE_STATE,
                    "questions": SAMPLE_QUESTIONS,
                }
                try:
                    response = client.post(endpoint, headers=headers, json=payload)
                except httpx.HTTPError as exc:
                    print(f"[ERR ] {endpoint} :: {model} -> {exc}")
                    continue
                tag = "OK  " if response.status_code == 200 else "FAIL"
                print(f"[{tag}] {endpoint} :: {model} -> HTTP {response.status_code}")
                if response.status_code == 200:
                    working.append((endpoint, model))
                    print(json.dumps(response.json(), indent=2)[:1500])
                else:
                    print("       ", response.text[:300].replace("\n", " "))

    print("\nWorking combinations:")
    for endpoint, model in working:
        print(f"  endpoint={endpoint}  model={model}")
    if not working:
        print("  none — check your key/model access.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())