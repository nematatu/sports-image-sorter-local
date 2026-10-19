"""Run reports (JSON / Markdown) that are safe to share.

By default the report contains no absolute paths and no filenames, only
aggregates, so it can be pasted into an issue. ``--include-paths`` is opt-in and
prints a warning, because filenames can themselves be personal data.
"""

from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from typing import Any

from .config import Config
from .images import hamming
from .evaluation import evaluate


def _duplicates(store, *, include_paths: bool = False) -> dict[str, Any]:
    rows = store.images()
    by_hash: dict[str, list[str]] = defaultdict(list)
    for r in rows:
        if r["sha256"]:
            by_hash[r["sha256"]].append(r["rel_path"])
    exact = {h: files for h, files in by_hash.items() if len(files) > 1}
    near = []
    usable = [r for r in rows if r["phash"]]
    for i in range(len(usable)):
        for j in range(i + 1, len(usable)):
            d = hamming(usable[i]["phash"], usable[j]["phash"])
            if d <= 6 and usable[i]["sha256"] != usable[j]["sha256"]:
                near.append({"a": usable[i]["rel_path"], "b": usable[j]["rel_path"], "distance": d})
    out: dict[str, Any] = {
        "exact_groups": len(exact),
        "exact_files": sum(len(v) for v in exact.values()),
        "near_pairs": len(near),
    }
    if include_paths:
        out["exact"] = exact
        out["near"] = near[:50]
    return out


def _latest_counts(store, engines: list[str]) -> dict[str, dict[str, int]]:
    """Count one (newest) prediction per image and engine.

    ``counts_by_label`` is a raw SQL aggregate, so a database that has seen a
    model re-run after a taxonomy fix contains several rows per image. A report
    must describe the current state, not every historical run, hence this
    projection over ``prediction_map`` (which keeps the newest row per engine).
    """
    counts: dict[str, dict[str, int]] = {engine: {} for engine in engines}
    for predictions in store.prediction_map(engines).values():
        for engine, row in predictions.items():
            label = row["label_id"] or "unclassified"
            counts[engine][label] = counts[engine].get(label, 0) + 1
    return counts


def build_report(store, config: Config, *, include_paths: bool = False) -> dict[str, Any]:
    images = store.images()
    errors = [r for r in images if r["error"]]
    engines = store.engines_seen()
    by_engine = _latest_counts(store, engines)
    human = store.labels()
    evals = {}
    for engine in ["fusion"] + [e for e in store.engines_seen() if e != "fusion"]:
        result = evaluate(store, config, engine)
        if result.labeled:
            payload = result.to_dict()
            if not include_paths:
                for item in payload["misclassified"]:
                    item.pop("rel_path", None)
            evals[engine] = payload
    exports = store.exports()
    export_ok = sum(1 for e in exports if e["verified"])
    report: dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "images": {
            "total": len(images),
            "decode_errors": len(errors),
            "human_labeled": len(human),
            "by_format": dict(Counter((r["format"] or "unknown") for r in images)),
        },
        "counts_by_engine": by_engine,
        "human_labels": dict(Counter(human.values())),
        "evaluation": evals,
        "duplicates": _duplicates(store, include_paths=include_paths),
        "exports": {"recorded": len(exports), "verified": export_ok},
        "labels": {lab.id: lab.name_ja for lab in config.all_labels},
    }
    if include_paths:
        report["images"]["error_files"] = [r["rel_path"] for r in errors]
        report["paths_note"] = "This report contains filenames and must not be published."
    return report


def report_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Sports image sorting report")
    lines.append("")
    lines.append(f"Generated: {report['generated_at']}")
    lines.append("")
    images = report["images"]
    lines.append("## Images")
    lines.append("")
    lines.append(f"- Total: **{images['total']}**")
    lines.append(f"- Decode errors: {images['decode_errors']}")
    lines.append(f"- Human-labeled (ground truth): {images['human_labeled']}")
    if images.get("by_format"):
        lines.append("- Formats: " + ", ".join(f"{k} {v}" for k, v in sorted(images["by_format"].items())))
    lines.append("")
    lines.append("## Counts by engine")
    lines.append("")
    engines = report.get("counts_by_engine", {})
    label_ids = sorted({lab for counts in engines.values() for lab in counts})
    if engines:
        lines.append("| engine | " + " | ".join(label_ids) + " |")
        lines.append("|---|" + "---|" * len(label_ids))
        for engine, counts in engines.items():
            lines.append("| " + engine + " | " + " | ".join(str(counts.get(l, 0)) for l in label_ids) + " |")
    else:
        lines.append("_No predictions yet._")
    lines.append("")
    lines.append("## Evaluation against human labels")
    lines.append("")
    evals = report.get("evaluation", {})
    if evals:
        lines.append("| engine | accuracy | macro F1 | coverage | evaluated |")
        lines.append("|---|---|---|---|---|")
        for engine, e in evals.items():
            acc = "n/a" if e["accuracy"] is None else f"{e['accuracy']:.3f}"
            f1 = "n/a" if e["macro_f1"] is None else f"{e['macro_f1']:.3f}"
            lines.append(f"| {engine} | {acc} | {f1} | {e['coverage']:.1%} | {e['evaluated']}/{e['labeled']} |")
    else:
        lines.append("_No human labels yet. Accuracy cannot be computed._")
    lines.append("")
    dup = report.get("duplicates", {})
    lines.append("## Duplicates")
    lines.append("")
    lines.append(f"- Exact duplicate groups: {dup.get('exact_groups', 0)} ({dup.get('exact_files', 0)} files)")
    lines.append(f"- Near-duplicate pairs (dHash <= 6): {dup.get('near_pairs', 0)}")
    lines.append("")
    exports = report.get("exports", {})
    lines.append("## Export")
    lines.append("")
    lines.append(f"- Recorded: {exports.get('recorded', 0)}")
    lines.append(f"- Verified (hash-matched): {exports.get('verified', 0)}")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- Filenames and EXIF are never used as classification signals.")
    lines.append("- `未分類` is a valid outcome; ambiguous images are not forced into a sport.")
    if "paths_note" in report:
        lines.append("- " + report["paths_note"])
    lines.append("")
    return "\n".join(lines)
