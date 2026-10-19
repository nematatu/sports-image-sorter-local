"""Apple Vision framework engines (macOS only, fully offline, zero cost).

Two engines live here:

``apple_vision``
    ``VNClassifyImageRequest`` - Apple's bundled ~1300-class classifier. It is
    fast, free and on-device, but its label set is generic: it knows
    "volleyball" and "tennis" but has no concept of "badminton", and it happily
    labels a portrait "person". Useful as a weak signal and as a sanity check,
    not as the primary classifier.

``ocr_text``
    ``VNRecognizeTextRequest`` (Japanese + English) used as a *content* signal:
    scoreboards, banners and venue signage often spell out the sport
    (バレーボール, 卓球, ...). It never looks at the filename or EXIF, and it
    abstains when no keyword is present. On photos without overlays it simply
    produces nothing, which is why it is weighted separately in the fusion.

The Swift helper is compiled once and cached next to the database; if a
compiler is not available the engine reports itself as unavailable instead of
failing the run.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Iterator

from .base import Engine, Prediction

HELPER_SOURCE = Path(__file__).resolve().parent.parent / "tools" / "vision_helper.swift"


def helper_cache_dir() -> Path:
    base = os.environ.get("SPORTS_SORTER_CACHE")
    if base:
        d = Path(base).expanduser()
    else:
        d = Path.home() / ".cache" / "sports-image-sorter"
    d.mkdir(parents=True, exist_ok=True)
    return d


class _VisionHelperMixin:
    """Compile/run the Swift helper that talks to the Vision framework."""

    def _helper_path(self) -> Path | None:
        cache = helper_cache_dir()
        binary = cache / "vision_helper"
        source_mtime = HELPER_SOURCE.stat().st_mtime if HELPER_SOURCE.exists() else 0
        if binary.exists() and binary.stat().st_mtime >= source_mtime:
            return binary
        compiler = shutil.which("swiftc")
        if not compiler:
            return None
        try:
            subprocess.run(
                [compiler, "-O", "-o", str(binary), str(HELPER_SOURCE)],
                check=True,
                capture_output=True,
                timeout=300,
            )
        except Exception:
            return None
        return binary

    def run_helper(self, paths: Iterable[str | Path]) -> list[dict[str, Any]]:
        binary = self._helper_path()
        if binary is None:
            raise RuntimeError("swiftc is not available; cannot use Apple Vision engines")
        cmd = [str(binary), *[str(p) for p in paths]]
        proc = subprocess.run(cmd, capture_output=True, timeout=600)
        if proc.returncode != 0:
            raise RuntimeError(f"vision_helper failed: {proc.stderr.decode('utf-8', 'replace')[:400]}")
        return json.loads(proc.stdout.decode("utf-8") or "[]")


class AppleVisionEngine(_VisionHelperMixin, Engine):
    name = "apple_vision"
    version = "vnclassifyimage-v1"
    description = "Apple VNClassifyImageRequest (on-device, free, generic label set)"

    def availability(self) -> tuple[bool, str]:
        if sys.platform != "darwin":
            return False, "macOS only"
        if self._helper_path() is None:
            return False, "swiftc not found (install Xcode Command Line Tools)"
        return True, "on-device, no network, no cost"

    def classify(self, path: str | Path) -> Prediction:
        try:
            return next(self.classify_many([path]))[1]
        except StopIteration:  # pragma: no cover
            return self._error(RuntimeError("no result"))

    def classify_many(self, paths: Iterable[str | Path]) -> Iterator[tuple[str, Prediction]]:
        paths = list(paths)
        try:
            entries = self.run_helper(paths)
        except Exception as exc:
            for p in paths:
                yield str(p), self._error(exc)
            return
        by_path = {e.get("path"): e for e in entries}
        for p in paths:
            entry = by_path.get(str(p), {})
            if entry.get("error"):
                yield str(p), self._error(RuntimeError(entry["error"]))
                continue
            scores: dict[str, float] = {}
            unmapped_best = 0.0
            labels = entry.get("classifications", []) or []
            for item in labels:
                label_id = self.config.resolve(str(item.get("id", "")))
                conf = float(item.get("confidence", 0.0))
                if label_id:
                    scores[label_id] = max(scores.get(label_id, 0.0), conf)
                else:
                    unmapped_best = max(unmapped_best, conf)
            raw = {"engine": self.name, "top_labels": labels[:8], "unmapped_best": unmapped_best}
            if not scores:
                # Nothing in Apple's taxonomy maps to ours: be honest. A zero
                # confidence keeps this "no opinion" from acting as a strong
                # abstain vote in fusion.
                return_pred = Prediction(label_id="unclassified", confidence=0.0, scores=scores, raw=raw)
            else:
                label = max(scores, key=scores.get)
                return_pred = Prediction(label_id=label, confidence=scores[label], scores=scores, raw=raw)
            yield str(p), return_pred


class AppleVisionTextEngine(_VisionHelperMixin, Engine):
    """OCR-only engine: reads the sport off scoreboards / banners / signage."""

    name = "ocr_text"
    version = "vnrecognizetext-v1"
    description = "Apple Vision OCR keyword matcher (scoreboards, banners, signage)"

    def availability(self) -> tuple[bool, str]:
        if sys.platform != "darwin":
            return False, "macOS only"
        if self._helper_path() is None:
            return False, "swiftc not found (install Xcode Command Line Tools)"
        return True, "on-device OCR, no network, no cost"

    def _keywords(self) -> dict[str, list[str]]:
        table: dict[str, list[str]] = {}
        for lab in self.config.labels:
            words = [lab.name_ja, lab.name_en, lab.id.replace("_", " ")]
            words += list(lab.aliases)
            words += list(lab.ocr_keywords)
            table[lab.id] = [w.lower() for w in words if w and len(w) >= 2]
        return table

    def classify(self, path: str | Path) -> Prediction:
        try:
            return next(self.classify_many([path]))[1]
        except StopIteration:  # pragma: no cover
            return self._error(RuntimeError("no result"))

    def classify_many(self, paths: Iterable[str | Path]) -> Iterator[tuple[str, Prediction]]:
        paths = list(paths)
        table = self._keywords()
        try:
            entries = self.run_helper(paths)
        except Exception as exc:
            for p in paths:
                yield str(p), self._error(exc)
            return
        by_path = {e.get("path"): e for e in entries}
        for p in paths:
            entry = by_path.get(str(p), {})
            text = " \n ".join(entry.get("text", []) or []).lower()
            if not text.strip():
                yield str(p), Prediction(label_id=None, confidence=None, raw={"engine": self.name, "text": []})
                continue
            hits: dict[str, float] = {}
            matched_words: dict[str, list[str]] = {}
            for label_id, words in table.items():
                for word in words:
                    if word and word in text:
                        hits[label_id] = max(hits.get(label_id, 0.0), 1.0 if len(word) >= 4 else 0.8)
                        matched_words.setdefault(label_id, []).append(word)
            raw = {"engine": self.name, "text": entry.get("text", [])[:6], "matched": matched_words}
            if not hits:
                yield str(p), Prediction(label_id=None, confidence=None, raw=raw)
                continue
            label = max(hits, key=hits.get)
            yield str(p), Prediction(label_id=label, confidence=hits[label], scores=hits, raw=raw)
