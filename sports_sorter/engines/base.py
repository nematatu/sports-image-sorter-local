"""Engine interface and registry.

An *engine* is anything that can look at an image and return a probability-like
score per configured label. Engines are deliberately narrow: they receive a
path plus the config, and they return a :class:`Prediction`. Everything else
(persistence, resume, export, review) is engine-agnostic, which is what makes
the comparison in ``docs/engines.md`` possible without touching the pipeline.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Prediction:
    """Result of one engine on one image."""

    label_id: str | None
    confidence: float | None = None
    scores: dict[str, float] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)
    status: str = "ok"  # ok | error | skipped
    error: str | None = None

    def top(self, n: int = 3) -> list[tuple[str, float]]:
        return sorted(self.scores.items(), key=lambda kv: kv[1], reverse=True)[:n]

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


class Engine:
    """Base class for classification engines."""

    name: str = "base"
    version: str = "1"
    description: str = ""
    #: True when the engine sends image data to a third party.
    sends_images_offsite: bool = False
    #: True when the engine needs an API key / paid account.
    requires_credentials: bool = False
    #: False when the engine can run fully offline.
    requires_network: bool = False

    def __init__(self, config, **options: Any):
        self.config = config
        self.options = options

    # -- capabilities --------------------------------------------------------
    def availability(self) -> tuple[bool, str]:
        """Return ``(usable, reason)``; ``reason`` is shown by ``engines``."""
        return True, "ok"

    # -- work ----------------------------------------------------------------
    def prepare(self) -> None:
        """Load weights, build prompt embeddings, compile helpers..."""

    def classify(self, path: str | Path) -> Prediction:  # pragma: no cover - interface
        raise NotImplementedError

    def close(self) -> None:
        """Release resources."""

    # -- helpers -------------------------------------------------------------
    def _error(self, exc: BaseException) -> Prediction:
        return Prediction(label_id=None, confidence=None, status="error", error=f"{type(exc).__name__}: {exc}")

    def empty_scores(self) -> dict[str, float]:
        return {lab.id: 0.0 for lab in self.config.labels}


_REGISTRY: dict[str, str] = {
    # name -> "module:ClassName" (imported lazily so an engine's optional
    # dependency never breaks the rest of the CLI).
    "apple_vision": "sports_sorter.engines.apple_vision:AppleVisionEngine",
    "ocr_text": "sports_sorter.engines.apple_vision:AppleVisionTextEngine",
    "clip": "sports_sorter.engines.local_hf:ClipZeroShotEngine",
    "siglip2": "sports_sorter.engines.local_hf:Siglip2ZeroShotEngine",
    "sports100_vit": "sports_sorter.engines.local_hf:Sports100VitEngine",
    "sportsnet7": "sports_sorter.engines.local_hf:SportsNet7Engine",
    "gemini": "sports_sorter.engines.external:GeminiVisionEngine",
    "openai": "sports_sorter.engines.external:OpenAIVisionEngine",
    "jev": "sports_sorter.engines.external:TypeSafeJevEngine",
    "fusion": "sports_sorter.engines.fusion:FusionEngine",
    "fake": "sports_sorter.engines.base:FakeEngine",
}


def register(name: str, target: str) -> None:
    """Register an engine under ``name`` (``"module:ClassName"``)."""
    _REGISTRY[name] = target


def available_engines() -> list[str]:
    return sorted(_REGISTRY)


def get_engine(name: str, config, **options: Any) -> Engine:
    if name not in _REGISTRY:
        raise KeyError(f"unknown engine {name!r}; known engines: {', '.join(available_engines())}")
    module_name, _, class_name = _REGISTRY[name].partition(":")
    import importlib

    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    return cls(config, **options)


class FakeEngine(Engine):
    """Deterministic engine used by tests and by ``--dry-run`` smoke checks.

    It never reads pixels: it derives a label from the SHA-256 of the path so
    that tests are fast and reproducible. Real runs must use a real engine.
    """

    name = "fake"
    version = "1"
    description = "Deterministic stub engine (tests / smoke checks only)"

    def classify(self, path: str | Path) -> Prediction:
        import hashlib

        labels = [lab.id for lab in self.config.labels]
        digest = hashlib.sha256(str(path).encode("utf-8")).hexdigest()
        idx = int(digest[:8], 16) % (len(labels) + 1)
        if idx == len(labels):
            return Prediction(label_id="unclassified", confidence=0.5, raw={"fake": True})
        label = labels[idx]
        return Prediction(label_id=label, confidence=0.9, scores={label: 0.9}, raw={"fake": True})
