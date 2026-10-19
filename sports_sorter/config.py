"""Configuration: sport taxonomy, zero-shot prompts, engine weights.

The taxonomy is data, not code. Everything about *which* sports exist, how they
are named on disk, and how they are described to a zero-shot model lives in a
JSON file so that users can adapt the tool to their own library without
touching Python.

``DEFAULT_CONFIG`` below is the taxonomy used for the reference run in
``docs/evaluation.md`` (badminton / volleyball / table tennis are the sports
that actually appear in the reference image set; the rest is a sensible
starting point for a general sports library).
"""

from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

DEFAULT_CONFIG: dict[str, Any] = {
    "version": 1,
    # Directory name used for images the tool refuses to commit to a sport.
    "unclassified_dir": "未分類",
    # Directory name for "a sport, but not one of the configured ones".
    "other_dir": "その他の競技",
    # A prediction below this confidence is downgraded to ``unclassified``.
    "min_confidence": 0.35,
    # When the two best labels are closer than this, treat the image as
    # ambiguous instead of guessing (see engines/fusion.py).
    "ambiguity_margin": 0.08,
    "labels": [
        {
            "id": "badminton",
            "name_ja": "バドミントン",
            "name_en": "badminton",
            "aliases": ["badminton", "バドミントン"],
            "ocr_keywords": ["バドミントン", "バド", "シャトル", "BWF", "badminton"],
            "prompts": [
                "a photo of a badminton player swinging a racket at a shuttlecock",
                "a badminton court with a high net and shuttlecock in play",
                "a close-up of a badminton racket and feather shuttlecock",
                "a woman playing badminton in an indoor sports hall",
            ],
            "evidence": ["shuttlecock", "thin badminton racket", "high narrow net"],
        },
        {
            "id": "volleyball",
            "name_ja": "バレーボール",
            "name_en": "volleyball",
            "aliases": ["volleyball", "バレーボール", "volley ball"],
            "ocr_keywords": ["バレーボール", "バレー", "Vリーグ", "volleyball"],
            "prompts": [
                "a photo of volleyball players at the net with knee pads",
                "a women's volleyball team in jerseys on an indoor court",
                "a volleyball player spiking the ball over a volleyball net",
                "a volleyball court with a net and players diving",
            ],
            "evidence": ["knee pads", "volleyball net", "team jersey + shorts uniform"],
        },
        {
            "id": "table_tennis",
            "name_ja": "卓球",
            "name_en": "table tennis",
            "aliases": ["table tennis", "table_tennis", "ping pong", "卓球"],
            "ocr_keywords": ["卓球", "table tennis", "ping pong", "Tリーグ"],
            "prompts": [
                "a photo of a table tennis player hitting a ball with a ping pong paddle",
                "a blue table tennis table with a net and paddles",
                "a table tennis match in an indoor hall",
            ],
            "evidence": ["table with net", "paddle", "small ball", "indoor hall"],
        },
        {
            "id": "tennis",
            "name_ja": "テニス",
            "name_en": "tennis",
            "aliases": ["tennis", "テニス"],
            "ocr_keywords": ["テニス", "ATP", "WTA", "tennis"],
            "prompts": [
                "a photo of a tennis player swinging a tennis racket on a tennis court",
                "a tennis court with white lines and a tennis ball",
                "a woman serving in a tennis match",
            ],
            "evidence": ["tennis court", "tennis racket", "yellow tennis ball"],
        },
        {
            "id": "basketball",
            "name_ja": "バスケットボール",
            "name_en": "basketball",
            "aliases": ["basketball", "バスケットボール", "basket ball"],
            "ocr_keywords": ["バスケットボール", "バスケ", "B.LEAGUE", "NBA", "basketball"],
            "prompts": [
                "a photo of a basketball player dribbling a basketball on a court",
                "a basketball hoop and backboard in a gymnasium",
                "a women's basketball game with players in jersey and shorts",
            ],
            "evidence": ["hoop and backboard", "orange basketball", "wooden gym floor"],
        },
        {
            "id": "track_field",
            "name_ja": "陸上競技",
            "name_en": "athletics",
            "aliases": ["track and field", "track_field", "athletics", "running", "陸上競技"],
            "ocr_keywords": ["陸上競技", "陸上", "トラック", "マラソン", "athletics"],
            "prompts": [
                "a photo of runners sprinting on a red athletics track",
                "an athlete doing the long jump into a sand pit",
                "a track and field stadium with lane markings",
            ],
            "evidence": ["red running track with lanes", "starting blocks", "sand pit"],
        },
        {
            "id": "swimming",
            "name_ja": "水泳",
            "name_en": "swimming",
            "aliases": ["swimming", "swim", "水泳"],
            "ocr_keywords": ["水泳", "競泳", "swimming"],
            "prompts": [
                "a photo of a swimmer swimming freestyle in a swimming pool",
                "swimmers in a pool with lane ropes and starting blocks",
                "a swimming race seen from above",
            ],
            "evidence": ["pool water", "lane ropes", "goggles and swim cap"],
        },
    ],
    # Labels that must never be produced by a classifier vote but can be
    # assigned by a human or by the fusion fallback.
    "fallback_labels": [
        {"id": "other", "name_ja": "その他の競技", "name_en": "other sport", "aliases": ["other"]},
        {"id": "unclassified", "name_ja": "未分類", "name_en": "unclassified", "aliases": ["unclassified", "unknown"]},
    ],
    # Negative/abstain descriptions for zero-shot engines: if one of these
    # scores higher than every sport, the image is left unclassified. This is
    # how "portraits, medal ceremonies, casual clothes" are handled *by the
    # model* rather than after the fact.
    "abstain_prompts": [
        "a portrait photograph of a person looking at the camera",
        "a close-up of a person's face",
        "an award ceremony with people wearing medals",
        "people hugging and celebrating after a match",
        "a group photo of people standing indoors",
        "an indoor photo with no sports equipment visible",
    ],
    "engines": {
        # Weights used by the `fusion` engine. Any engine not listed gets 0.0.
        # Broad zero-shot models get the most weight; narrow supervised
        # fine-tunes are advisory only (see docs/evaluation.md for why, and for
        # the pre-registered weights that were tested before this revision).
        "weights": {
            "sports100_vit": 0.25,
            "sportsnet7": 0.25,
            "clip": 0.8,
            "siglip2": 1.0,
            "apple_vision": 0.2,
            "ocr_text": 0.5,
        },
        "default": ["siglip2", "clip", "sports100_vit", "sportsnet7", "apple_vision", "ocr_text"],
    },
    # Extensions the scanner is allowed to open.
    "extensions": [".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif", ".tif", ".tiff", ".bmp", ".gif"],
    "preprocess": {"max_side": 1024, "thumbnail_side": 384},
}


@dataclass(frozen=True)
class Label:
    """One sport (or fallback bucket)."""

    id: str
    name_ja: str
    name_en: str = ""
    aliases: tuple[str, ...] = ()
    prompts: tuple[str, ...] = ()
    evidence: tuple[str, ...] = ()
    ocr_keywords: tuple[str, ...] = ()
    is_fallback: bool = False

    @property
    def dir_name(self) -> str:
        return self.name_ja

    def matches(self, text: str) -> bool:
        """True when ``text`` (an engine's class name) refers to this label."""
        t = (text or "").strip().lower().replace("_", " ").replace("-", " ")
        for cand in (self.id, self.name_ja, self.name_en, *self.aliases):
            c = (cand or "").strip().lower().replace("_", " ").replace("-", " ")
            if c and (t == c or c in t.split("/") or t in c.split("/")):
                return True
        return False


@dataclass
class Config:
    """Runtime configuration (taxonomy + prompts + engine settings)."""

    raw: dict[str, Any] = field(default_factory=lambda: copy.deepcopy(DEFAULT_CONFIG))
    source: str | None = None

    # -- labels --------------------------------------------------------------
    @property
    def labels(self) -> list[Label]:
        out: list[Label] = []
        for entry in self.raw.get("labels", []):
            out.append(
                Label(
                    id=entry["id"],
                    name_ja=entry.get("name_ja", entry["id"]),
                    name_en=entry.get("name_en", ""),
                    aliases=tuple(entry.get("aliases", ())),
                    prompts=tuple(entry.get("prompts", ())),
                    evidence=tuple(entry.get("evidence", ())),
                    ocr_keywords=tuple(entry.get("ocr_keywords", ())),
                )
            )
        return out

    @property
    def fallback_labels(self) -> list[Label]:
        out: list[Label] = []
        for entry in self.raw.get("fallback_labels", []):
            out.append(
                Label(
                    id=entry["id"],
                    name_ja=entry.get("name_ja", entry["id"]),
                    name_en=entry.get("name_en", ""),
                    aliases=tuple(entry.get("aliases", ())),
                    is_fallback=True,
                )
            )
        return out

    @property
    def all_labels(self) -> list[Label]:
        return self.labels + self.fallback_labels

    @property
    def abstain_prompts(self) -> list[str]:
        return list(self.raw.get("abstain_prompts", []))

    @property
    def min_confidence(self) -> float:
        return float(self.raw.get("min_confidence", 0.35))

    @property
    def ambiguity_margin(self) -> float:
        return float(self.raw.get("ambiguity_margin", 0.08))

    @property
    def extensions(self) -> set[str]:
        return {e.lower() for e in self.raw.get("extensions", DEFAULT_CONFIG["extensions"])}

    @property
    def preprocess(self) -> dict[str, Any]:
        return dict(self.raw.get("preprocess", DEFAULT_CONFIG["preprocess"]))

    @property
    def engine_weights(self) -> dict[str, float]:
        return dict(self.raw.get("engines", {}).get("weights", {}))

    @property
    def default_engines(self) -> list[str]:
        return list(self.raw.get("engines", {}).get("default", []))

    # -- helpers -------------------------------------------------------------
    def label(self, label_id: str) -> Label | None:
        for lab in self.all_labels:
            if lab.id == label_id:
                return lab
        return None

    def resolve(self, text: str) -> str | None:
        """Map an arbitrary engine class name onto a configured label id."""
        for lab in self.all_labels:
            if lab.matches(text):
                return lab.id
        return None

    def dir_name(self, label_id: str) -> str:
        lab = self.label(label_id)
        if lab is not None:
            return lab.dir_name
        if label_id == "other":
            return self.raw.get("other_dir", "その他の競技")
        if label_id == "unclassified":
            return self.raw.get("unclassified_dir", "未分類")
        return label_id

    def to_json(self) -> str:
        return json.dumps(self.raw, ensure_ascii=False, indent=2, sort_keys=True)


def load_config(path: str | os.PathLike[str] | None = None) -> Config:
    """Load a taxonomy config, falling back to :data:`DEFAULT_CONFIG`."""
    if path is None:
        env = os.environ.get("SPORTS_SORTER_CONFIG")
        if env:
            path = env
    if path is None:
        return Config()
    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"config not found: {p}")
    data = json.loads(p.read_text(encoding="utf-8"))
    merged = _deep_merge(copy.deepcopy(DEFAULT_CONFIG), data)
    return Config(raw=merged, source=str(p))


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def config_hash(cfg: Config, extra: Iterable[str] = ()) -> str:
    """Stable hash of the config, used to invalidate cached predictions."""
    import hashlib

    h = hashlib.sha256()
    h.update(cfg.to_json().encode("utf-8"))
    for item in sorted(extra):
        h.update(str(item).encode("utf-8"))
    return h.hexdigest()[:16]


def prediction_config_hash(cfg: Config, engine: str, version: str, options: dict | None = None) -> str:
    """Hash of everything an *engine's* output depends on.

    Fusion weights and thresholds are deliberately excluded: changing how
    votes are combined must not force a costly re-run of every model.
    """
    import hashlib

    payload = {
        "labels": [
            {
                "id": lab.id,
                "name": lab.name_ja,
                "aliases": list(lab.aliases),
                "prompts": list(lab.prompts),
                "ocr_keywords": list(lab.ocr_keywords),
            }
            for lab in cfg.all_labels
        ],
        "abstain_prompts": cfg.abstain_prompts,
        "preprocess": cfg.preprocess,
        "engine": engine,
        "version": version,
        "options": options or {},
    }
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:16]
