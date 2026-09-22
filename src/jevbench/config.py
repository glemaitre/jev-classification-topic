"""Global configuration for the benchmark."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = PROJECT_ROOT / "cache"
RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"

for _directory in (CACHE_DIR, RESULTS_DIR, PLOTS_DIR):
    _directory.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

#: The 20 Newsgroups newsgroup names, in the canonical scikit-learn order.
CATEGORIES: tuple[str, ...] = (
    "alt.atheism",
    "comp.graphics",
    "comp.os.ms-windows.misc",
    "comp.sys.ibm.pc.hardware",
    "comp.sys.mac.hardware",
    "comp.windows.x",
    "misc.forsale",
    "rec.autos",
    "rec.motorcycles",
    "rec.sport.baseball",
    "rec.sport.hockey",
    "sci.crypt",
    "sci.electronics",
    "sci.med",
    "sci.space",
    "soc.religion.christian",
    "talk.politics.guns",
    "talk.politics.mideast",
    "talk.politics.misc",
    "talk.religion.misc",
)

#: Short descriptions used as the ``criteria`` for Jev's ``choice`` question and
#: as the zero-shot target text for embeddings. Kept deliberately terse: they are
#: the only label information the model sees.
CATEGORY_DESCRIPTIONS: dict[str, str] = {
    "alt.atheism": "Atheism, critiques of religion, and debates about belief in God.",
    "comp.graphics": "Computer graphics: rendering, image formats, 3D, visualization.",
    "comp.os.ms-windows.misc": "Microsoft Windows operating systems, general topics.",
    "comp.sys.ibm.pc.hardware": "IBM PC and compatible hardware: motherboards, drives, upgrades.",
    "comp.sys.mac.hardware": "Apple Macintosh hardware: Macs, peripherals, upgrades.",
    "comp.windows.x": "The X Window System: X11 servers, clients, window managers.",
    "misc.forsale": "Items offered for sale, wanted ads, and classifieds.",
    "rec.autos": "Automobiles: makes, models, repair, driving.",
    "rec.motorcycles": "Motorcycles: bikes, riding, repair, safety.",
    "rec.sport.baseball": "Baseball: teams, players, games, statistics.",
    "rec.sport.hockey": "Ice hockey: teams, players, games, statistics.",
    "sci.crypt": "Cryptography, encryption, ciphers, and computer security.",
    "sci.electronics": "Electronics: circuits, components, design, repair.",
    "sci.med": "Medicine, health, diseases, and medical practice.",
    "sci.space": "Space exploration, astronomy, satellites, and space science.",
    "soc.religion.christian": "Christianity, theology, the Bible, and church matters.",
    "talk.politics.guns": "Gun control, firearms laws, and the politics of guns.",
    "talk.politics.mideast": "Middle East politics, conflicts, and regional affairs.",
    "talk.politics.misc": "Miscellaneous political discussion and debate.",
    "talk.religion.misc": "Religion in general, interfaith topics, and religious debate.",
}

# ---- Jev / OpenRouter -------------------------------------------------------

OPENROUTER_API_KEY_ENV = "OPENROUTER_API_KEY"

#: The documented System One endpoint. The probe script verifies it and falls
#: back to the alternative used by some public examples.
JEV_ENDPOINTS: tuple[str, ...] = (
    "https://openrouter.ai/api/v1/systemone",
    "https://openrouter.ai/api/alpha/decisions",
)

#: Pinned model id by default so runs are reproducible; the ``~typesafe/jev-latest``
#: alias can be selected with ``--jev-model``.
JEV_MODEL_DEFAULT = "typesafe/jev-1.13"
JEV_MODEL_LATEST_ALIAS = "~typesafe/jev-latest"

#: Max characters of a document sent as ``state``. Longer posts are truncated;
#: the truncation rate is reported.
JEV_MAX_STATE_CHARS = 6000

#: prompt version — bump when the question wording changes so the response cache
#: is invalidated.
JEV_PROMPT_VERSION = "v1"


def openrouter_api_key() -> str | None:
    """Return the OpenRouter API key from the environment, if any."""
    key = os.environ.get(OPENROUTER_API_KEY_ENV, "")
    return key or None