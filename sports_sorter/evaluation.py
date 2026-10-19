"""Accuracy evaluation against human labels.

Only rows in the ``labels`` table count as ground truth. Engine predictions are
never used as truth: if there is no human label, ``coverage`` is reported and
accuracy is explicitly ``None`` instead of a made-up number.

Metrics: accuracy, macro/weighted F1 and a confusion matrix over the configured
labels plus ``unclassified``. The evaluator also returns the list of
misclassifications so a human can inspect *why* a model failed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterable

from .config import Config


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    """Wilson score interval for a binomial proportion (95% by default).

    With a 23-image sample the interval is wide, which is exactly the point:
    it keeps a single number from looking more certain than it is.
    """
    if n <= 0:
        return None
    phat = k / n
    denom = 1 + z * z / n
    centre = phat + z * z / (2 * n)
    margin = z * ((phat * (1 - phat) / n + z * z / (4 * n * n)) ** 0.5)
    return (max(0.0, (centre - margin) / denom), min(1.0, (centre + margin) / denom))


@dataclass
class EvalResult:
    engine: str
    images_total: int = 0
    labeled: int = 0
    evaluated: int = 0
    correct: int = 0
    errors: int = 0
    confusion: dict[str, dict[str, int]] = field(default_factory=dict)
    per_label: dict[str, dict[str, float]] = field(default_factory=dict)
    misclassified: list[dict[str, Any]] = field(default_factory=list)
    ambiguous_correct: int = 0

    @property
    def accuracy(self) -> float | None:
        return self.correct / self.evaluated if self.evaluated else None

    @property
    def coverage(self) -> float:
        return self.evaluated / self.labeled if self.labeled else 0.0

    @property
    def macro_f1(self) -> float | None:
        if not self.per_label:
            return None
        return sum(v["f1"] for v in self.per_label.values()) / len(self.per_label)

    @property
    def accuracy_ci95(self) -> tuple[float, float] | None:
        return wilson_ci(self.correct, self.evaluated)

    def to_dict(self) -> dict[str, Any]:
        return {
            "engine": self.engine,
            "images_total": self.images_total,
            "labeled": self.labeled,
            "evaluated": self.evaluated,
            "correct": self.correct,
            "accuracy": self.accuracy,
            "accuracy_ci95": self.accuracy_ci95,
            "coverage": self.coverage,
            "macro_f1": self.macro_f1,
            "errors": self.errors,
            "confusion": self.confusion,
            "per_label": self.per_label,
            "misclassified": self.misclassified,
        }


def evaluate(
    store,
    config: Config,
    engine: str = "fusion",
    *,
    config_hash: str | None = None,
    limit: int | None = None,
) -> EvalResult:
    """Compare ``engine``'s predictions with human labels."""
    gold = store.labels()
    result = EvalResult(engine=engine)
    images = store.images()
    result.images_total = len(images)
    result.labeled = len(gold)
    labels = [lab.id for lab in config.labels] + ["other", "unclassified"]
    for lab in labels:
        result.confusion[lab] = {other: 0 for other in labels}

    for image in images:
        image_id = int(image["id"])
        truth = gold.get(image_id)
        if not truth:
            continue
        row = store.prediction(image_id, engine, config_hash)
        if row is None:
            continue
        if row["status"] != "ok" or not row["label_id"]:
            result.errors += 1
            continue
        predicted = row["label_id"]
        result.evaluated += 1
        if predicted not in result.confusion:
            result.confusion[predicted] = {other: 0 for other in labels}
        if truth not in result.confusion:
            result.confusion[truth] = {other: 0 for other in labels}
        result.confusion[truth][predicted] = result.confusion[truth].get(predicted, 0) + 1
        if predicted == truth:
            result.correct += 1
        else:
            result.misclassified.append(
                {
                    "image_id": image_id,
                    "rel_path": image["rel_path"],
                    "truth": truth,
                    "predicted": predicted,
                    "confidence": row["confidence"],
                }
            )
            if limit and len(result.misclassified) >= limit:
                result.misclassified = result.misclassified[:limit]

    # Per-label precision / recall / F1.
    for lab in labels:
        tp = result.confusion.get(lab, {}).get(lab, 0)
        fp = sum(result.confusion.get(other, {}).get(lab, 0) for other in labels if other != lab)
        fn = sum(result.confusion.get(lab, {}).get(other, 0) for other in labels if other != lab)
        if tp + fp + fn == 0:
            continue
        precision, recall, f1 = _prf(tp, fp, fn)
        result.per_label[lab] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": tp + fn,
        }
    return result


def compare_engines(store, config: Config, engines: Iterable[str] | None = None) -> list[EvalResult]:
    names = list(engines) if engines else store.engines_seen()
    return [evaluate(store, config, engine) for engine in names]


def format_eval(result: EvalResult) -> str:
    lines = [f"engine: {result.engine}"]
    if result.accuracy is None:
        lines.append("accuracy: n/a (no human-labeled images with predictions)")
    else:
        ci = result.accuracy_ci95
        ci_text = f" [95% CI {ci[0]:.3f}-{ci[1]:.3f}]" if ci else ""
        lines.append(
            f"accuracy: {result.accuracy:.4f} ({result.correct}/{result.evaluated}){ci_text}, "
            f"macro-F1: {result.macro_f1:.4f}, coverage: {result.coverage:.1%}"
        )
    if result.per_label:
        lines.append("label                 P       R       F1      support")
        for lab, m in sorted(result.per_label.items()):
            lines.append(f"{lab:<20} {m['precision']:.3f}   {m['recall']:.3f}   {m['f1']:.3f}   {m['support']}")
    if result.misclassified:
        lines.append("misclassified:")
        for item in result.misclassified:
            lines.append(f"  #{item['image_id']} {item['rel_path']}: truth={item['truth']} predicted={item['predicted']} ({item['confidence']})")
    return "\n".join(lines)


def result_json(result: EvalResult) -> str:
    return json.dumps(result.to_dict(), ensure_ascii=False, indent=2)
