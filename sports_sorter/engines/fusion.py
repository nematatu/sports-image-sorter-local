"""Fusion of multiple engines into one final decision.

The rules are intentionally conservative. A wrong folder is worse than an
``未分類`` that a human can review, so the fusion:

* gives every engine a configurable weight (``engines.weights``),
* treats each engine's *top* label as one vote weighted by its confidence,
* keeps "unclassified" votes in a separate abstain bucket,
* refuses to decide when the abstain bucket wins, when the winner is below
  ``min_confidence``, or when the top two labels are within
  ``ambiguity_margin`` of each other.

The function is pure (it takes stored rows and returns a Prediction) so the
whole decision can be unit-tested without models or disks.
"""

from __future__ import annotations

import json
from typing import Any, Iterable, Mapping

from .base import Engine, Prediction


def _as_row(row: Any) -> dict[str, Any]:
    """Accept sqlite3.Row, dict-like or Prediction."""
    if isinstance(row, Prediction):
        return {
            "engine": getattr(row, "engine", ""),
            "label_id": row.label_id,
            "confidence": row.confidence,
            "scores": row.scores,
            "status": row.status,
        }
    if isinstance(row, Mapping):
        data = dict(row)
    else:  # sqlite3.Row
        data = {k: row[k] for k in row.keys()}
    scores = data.get("scores")
    if isinstance(scores, str):
        try:
            data["scores"] = json.loads(scores or "{}")
        except json.JSONDecodeError:
            data["scores"] = {}
    return data


def fuse(
    rows: Iterable[Any],
    config,
    *,
    weights: Mapping[str, float] | None = None,
    min_confidence: float | None = None,
    ambiguity_margin: float | None = None,
) -> Prediction:
    """Combine engine rows into one final :class:`Prediction`."""
    weights = dict(weights or config.engine_weights)
    min_confidence = config.min_confidence if min_confidence is None else min_confidence
    ambiguity_margin = config.ambiguity_margin if ambiguity_margin is None else ambiguity_margin

    votes: dict[str, float] = {}
    abstain = 0.0
    used: list[dict[str, Any]] = []
    for raw_row in rows:
        row = _as_row(raw_row)
        engine = str(row.get("engine") or "")
        if row.get("status", "ok") != "ok":
            continue
        label = row.get("label_id")
        if not label:
            continue
        weight = float(weights.get(engine, 0.0))
        if weight <= 0:
            continue
        conf = row.get("confidence")
        conf = 0.5 if conf is None else max(0.0, min(1.0, float(conf)))
        contribution = weight * conf
        used.append({"engine": engine, "label": label, "confidence": conf, "weight": weight})
        if label == "unclassified":
            abstain += contribution
        else:
            votes[label] = votes.get(label, 0.0) + contribution

    if not votes:
        return Prediction(
            label_id="unclassified",
            confidence=None,
            scores={},
            raw={"fusion": "no usable votes", "votes": used, "abstain": abstain},
        )

    total = sum(votes.values()) + abstain
    shares = {label: value / total for label, value in votes.items()}
    ranked = sorted(shares.items(), key=lambda kv: kv[1], reverse=True)
    best_label, best_share = ranked[0]
    second_share = ranked[1][1] if len(ranked) > 1 else 0.0
    abstain_share = abstain / total
    sport_share = sum(votes.values()) / total
    # Effective confidence blends *agreement* (share of votes) with the
    # engines' own confidence, so a lone 10%-confident vote cannot pass a 50%
    # threshold just because it is the only vote.
    weight_total = sum(v["weight"] for v in used) or 1.0
    mean_confidence = sum(v["weight"] * v["confidence"] for v in used) / weight_total
    effective = best_share * mean_confidence

    reason = None
    final_label = best_label
    if abstain_share >= sport_share:
        final_label = "unclassified"
        reason = "abstain won"
    elif best_share - second_share < ambiguity_margin:
        final_label = "unclassified"
        reason = f"ambiguous: {best_label} {best_share:.3f} vs {ranked[1][0]} {second_share:.3f}"
    elif effective < min_confidence:
        final_label = "unclassified"
        reason = f"confidence {effective:.3f} < {min_confidence}"

    confidence = effective if final_label != "unclassified" else max(abstain_share, 1.0 - sport_share, 0.0)
    return Prediction(
        label_id=final_label,
        confidence=round(float(confidence), 6),
        scores={k: round(v, 6) for k, v in shares.items()},
        raw={
            "fusion": reason or "ok",
            "votes": used,
            "abstain_share": abstain_share,
            "vote_share": best_share,
            "mean_confidence": mean_confidence,
            "margin": best_share - second_share,
            "runner_up": ranked[1][0] if len(ranked) > 1 else None,
        },
    )


def finalize_store(store, config, *, engines: Iterable[str] | None = None, weights: Mapping[str, float] | None = None) -> dict[int, Prediction]:
    """Compute and persist a ``fusion`` prediction for every image in the DB."""
    engine_list = list(engines) if engines else None
    pmap = store.prediction_map(engine_list)
    fhash = _fusion_hash(config)
    run_id = store.start_run("fusion", "rules-v1", fhash, {"weights": dict(weights or config.engine_weights)})
    # Fusion output is derived data: drop rows from an older config/weight set
    # so that "the newest fusion row" is never ambiguous.
    store.conn.execute("DELETE FROM predictions WHERE engine='fusion' AND config_hash<>?", (fhash,))
    store.conn.commit()
    out: dict[int, Prediction] = {}
    for image_id in store.image_ids():
        # Never let a previous fusion result vote for itself.
        rows = [row for name, row in pmap.get(image_id, {}).items() if name != "fusion"]
        pred = fuse(rows, config, weights=weights)
        out[image_id] = pred
        store.record_prediction(
            image_id=image_id,
            run_id=run_id,
            engine="fusion",
            config_hash=fhash,
            label_id=pred.label_id,
            confidence=pred.confidence,
            scores=pred.scores,
            raw=pred.raw,
            status=pred.status,
            error=pred.error,
        )
    store.finish_run(run_id, "finished")
    return out


def _fusion_hash(config) -> str:
    import hashlib

    h = hashlib.sha256()
    h.update(config.to_json().encode("utf-8"))
    h.update(b"fusion-v1")
    return h.hexdigest()[:16]


class FusionEngine(Engine):
    """Engine-shaped wrapper around :func:`fuse` for tests and single images.

    The pipeline uses :func:`finalize_store` (it needs the whole image set);
    this class exists so that the fusion rules can be exercised through the
    same interface as every other engine.
    """

    name = "fusion"
    version = "rules-v1"
    description = "Weighted vote + abstain/ambiguity rules over other engines"

    def __init__(self, config, *, sources: Mapping[str, Prediction] | None = None, **options):
        super().__init__(config, **options)
        self.sources = dict(sources or {})

    def classify(self, path):
        rows = []
        for engine, pred in self.sources.items():
            rows.append(
                {
                    "engine": engine,
                    "label_id": pred.label_id,
                    "confidence": pred.confidence,
                    "scores": pred.scores,
                    "status": pred.status,
                }
            )
        return fuse(rows, self.config)
