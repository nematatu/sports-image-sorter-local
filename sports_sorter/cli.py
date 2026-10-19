"""Command line interface.

Examples
--------
    sports-sorter scan      --input ~/Pictures/sports
    sports-sorter classify  --input ~/Pictures/sports --engine siglip2 --engine sports100_vit
    sports-sorter finalize
    sports-sorter evaluate  --engine siglip2
    sports-sorter export    --out ~/Pictures/sports-sorted
    sports-sorter verify
    sports-sorter review

Every command that touches the input directory goes through
:mod:`sports_sorter.safety`; every write is exclusive (no overwrites).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Iterable, Sequence

from . import __version__
from .config import DEFAULT_CONFIG, Config, config_hash, load_config, prediction_config_hash
from .db import Store
from .engines import available_engines, get_engine
from .evaluation import compare_engines, evaluate, format_eval, result_json
from .exporter import export_images, plan_export, verify_exports, verify_originals
from .images import find_exact_duplicates, find_near_duplicates, scan_directory
from .report import build_report, report_markdown
from .safety import SafetyError, assert_input_allowed, assert_safe_output

DEFAULT_DB = Path("data") / "sports.db"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _cfg(args) -> Config:
    return load_config(getattr(args, "config", None))


def _db_path(args) -> Path:
    return Path(getattr(args, "db", None) or os.environ.get("SPORTS_SORTER_DB") or DEFAULT_DB).expanduser()


def _open_store(args) -> Store:
    return Store(_db_path(args))


def _input_path(args, store: Store, *, required: bool = True) -> Path | None:
    raw = getattr(args, "input", None) or store.get_meta("input_dir")
    if not raw:
        if required:
            raise SystemExit("error: --input is required (or run `scan` first to remember it)")
        return None
    return assert_input_allowed(raw, allow_blocked=getattr(args, "i_know_what_i_am_doing", False))


def _flatten_engines(values: Iterable[str] | None) -> list[str]:
    out: list[str] = []
    for value in values or []:
        for item in str(value).split(","):
            item = item.strip()
            if item and item not in out:
                out.append(item)
    return out


def _print_counts(counts: dict[str, int], cfg: Config) -> None:
    total = sum(counts.values())
    for label_id, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"  {cfg.dir_name(label_id):<12} {n:>5}")
    print(f"  {'total':<12} {total:>5}")


def _progress(done: int, total: int, label: str, ok: int, errors: int) -> None:
    if total and (done == total or done % 10 == 0 or done == 1):
        print(f"\r  {label}: {done}/{total} (ok {ok}, errors {errors})", end="", flush=True)
        if done == total:
            print()


def _scan(store: Store, cfg: Config, input_path: Path, *, recursive: bool) -> int:
    print(f"scanning {input_path} ...")
    records = scan_directory(input_path, recursive=recursive, extensions=cfg.extensions)
    for rec in records:
        store.upsert_image(rec)
    store.set_meta("input_dir", str(input_path))
    store.set_meta("input_scanned_at", time.time())
    errors = [r for r in records if r.error]
    formats: dict[str, int] = {}
    for r in records:
        formats[r.format or "unknown"] = formats.get(r.format or "unknown", 0) + 1
    print(f"  images: {len(records)}  decode errors: {len(errors)}")
    print("  formats: " + ", ".join(f"{k} {v}" for k, v in sorted(formats.items())))
    return len(records)


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #
def cmd_scan(args) -> int:
    cfg = _cfg(args)
    with _open_store(args) as store:
        input_path = _input_path(args, store)
        _scan(store, cfg, input_path, recursive=args.recursive)
    return 0


def cmd_classify(args) -> int:
    cfg = _cfg(args)
    engines = _flatten_engines(args.engine) or cfg.default_engines
    if not engines:
        raise SystemExit("error: no engines configured (use --engine or engines.default)")
    with _open_store(args) as store:
        input_path = _input_path(args, store)
        if not args.no_scan:
            _scan(store, cfg, input_path, recursive=args.recursive)
        images = store.images()
        if args.limit:
            images = images[: args.limit]
        usable = [img for img in images if not img["error"] or args.include_errors]
        print(f"candidates: {len(usable)} / {len(images)}")
        for name in engines:
            try:
                engine = get_engine(name, cfg, device=args.device, batch_size=args.batch_size, crops=args.crops)
            except Exception as exc:
                print(f"[{name}] cannot load: {exc}")
                continue
            ok, reason = engine.availability()
            if not ok:
                print(f"[{name}] unavailable: {reason}")
                continue
            print(f"[{name}] {engine.description or ''}")
            options = {"device": args.device, "batch_size": args.batch_size, "crops": args.crops}
            chash = prediction_config_hash(cfg, name, engine.version, options)
            if args.force:
                pending = list(usable)
                skipped = 0
            else:
                pending = []
                skipped = 0
                for img in usable:
                    row = store.prediction(int(img["id"]), name, chash)
                    if row is not None and row["status"] == "ok" and not args.include_errors:
                        skipped += 1
                    else:
                        pending.append(img)
            print(f"  pending: {len(pending)}  cached: {skipped}")
            if args.dry_run:
                for img in pending[:20]:
                    print(f"    would classify: {img['rel_path']}")
                if len(pending) > 20:
                    print(f"    ... and {len(pending) - 20} more")
                continue
            if not pending:
                continue
            try:
                engine.prepare()
            except Exception as exc:
                print(f"  prepare failed: {type(exc).__name__}: {exc}")
                continue
            run_id = store.start_run(name, engine.version, chash, options)
            paths = [str(img["abs_path"]) for img in pending]
            by_path = {str(img["abs_path"]): img for img in pending}
            ok_count = err_count = 0
            started = time.time()
            classify_many = getattr(engine, "classify_many", None)
            try:
                if classify_many is not None:
                    results = classify_many(paths)
                else:
                    def _one(paths=paths):
                        for p in paths:
                            yield p, engine.classify(p)

                    results = _one()
                for done, (path, pred) in enumerate(results, start=1):
                    img = by_path.get(str(path))
                    if img is None:
                        continue
                    if pred.status == "ok":
                        ok_count += 1
                    else:
                        err_count += 1
                    store.record_prediction(
                        image_id=int(img["id"]),
                        run_id=run_id,
                        engine=name,
                        config_hash=chash,
                        label_id=pred.label_id,
                        confidence=pred.confidence,
                        scores=pred.scores,
                        raw=pred.raw,
                        status=pred.status,
                        error=pred.error,
                    )
                    _progress(done, len(paths), name, ok_count, err_count)
            except KeyboardInterrupt:
                store.finish_run(run_id, "interrupted")
                print(f"\n[{name}] interrupted; progress saved. Re-run to resume.")
                return 130
            finally:
                engine.close()
            elapsed = time.time() - started
            store.finish_run(run_id, "finished", f"ok={ok_count} errors={err_count} elapsed={elapsed:.1f}s")
            print(f"[{name}] done in {elapsed:.1f}s (ok {ok_count}, errors {err_count})")
    return 0


def cmd_finalize(args) -> int:
    from .engines.fusion import finalize_store

    cfg = _cfg(args)
    weights = cfg.engine_weights
    if args.exclude:
        for name in _flatten_engines(args.exclude):
            weights.pop(name, None)
    with _open_store(args) as store:
        if not store.engines_seen():
            raise SystemExit("error: no predictions yet; run `classify` first")
        predictions = finalize_store(store, cfg, weights=weights)
        counts: dict[str, int] = {}
        for pred in predictions.values():
            counts[pred.label_id or "unclassified"] = counts.get(pred.label_id or "unclassified", 0) + 1
        print(f"finalized {len(predictions)} images")
        _print_counts(counts, cfg)
    return 0


def cmd_export(args) -> int:
    cfg = _cfg(args)
    with _open_store(args) as store:
        input_path = _input_path(args, store)
        if args.engine and not store.prediction_map([args.engine]):
            raise SystemExit(f"error: no predictions for engine {args.engine!r}")
        report = export_images(
            store,
            cfg,
            input_path,
            args.out,
            engine=args.engine,
            dry_run=args.dry_run,
            include_unclassified=not args.exclude_unclassified,
            verify=not args.no_verify,
        )
        label = "planned" if args.dry_run else "copied"
        print(f"{label}: {report.copied if not args.dry_run else len([i for i in report.items if i.action == 'copy'])}")
        print(f"skipped: {report.skipped}  errors: {report.errors}")
        for lab, n in sorted(report.per_label.items(), key=lambda kv: -kv[1]):
            print(f"  {cfg.dir_name(lab):<12} {n:>5}")
        for item in report.items:
            if item.action == "error":
                print(f"  ERROR {item.source}: {item.reason}")
        if args.json:
            Path(args.json).write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"report: {args.json}")
    return 0 if report.errors == 0 else 1


def cmd_verify(args) -> int:
    with _open_store(args) as store:
        result = verify_exports(store)
        print("export verification:")
        print(f"  checked {result.checked}, ok {result.ok}, missing dest {result.missing_dest}, "
              f"modified dest {result.modified_dest}, modified source {result.modified_source}, "
              f"missing source {result.missing_source}")
        for problem in result.problems[:20]:
            print(f"  PROBLEM {problem}")
        original_problems = []
        input_dir = args.input or store.get_meta("input_dir")
        if input_dir:
            original_problems = verify_originals(store, input_dir)
            print(f"originals re-hashed: {len(original_problems)} changed/missing")
            for p in original_problems[:20]:
                print(f"  PROBLEM {p}")
        if args.json:
            payload = result.to_dict()
            payload["originals"] = original_problems
            Path(args.json).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return 0 if result.ok_all and not original_problems else 1


def cmd_evaluate(args) -> int:
    cfg = _cfg(args)
    with _open_store(args) as store:
        if args.all:
            results = compare_engines(store, cfg)
            for result in results:
                print(format_eval(result))
                print()
            payload = [r.to_dict() for r in results]
        else:
            result = evaluate(store, cfg, args.engine or "fusion")
            print(format_eval(result))
            payload = result.to_dict()
        if args.json:
            Path(args.json).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"wrote {args.json}")
    return 0


def cmd_compare(args) -> int:
    cfg = _cfg(args)
    with _open_store(args) as store:
        engines = _flatten_engines(args.engine) or store.engines_seen()
        rows = []
        for engine in engines:
            result = evaluate(store, cfg, engine)
            rows.append(result)
        header = f"{'engine':<18} {'acc':>7} {'macroF1':>8} {'coverage':>9} {'n':>5}"
        print(header)
        print("-" * len(header))
        for r in rows:
            acc = "n/a" if r.accuracy is None else f"{r.accuracy:.3f}"
            f1 = "n/a" if r.macro_f1 is None else f"{r.macro_f1:.3f}"
            print(f"{r.engine:<18} {acc:>7} {f1:>8} {r.coverage:>8.1%} {r.evaluated:>5}")
        if args.json:
            Path(args.json).write_text(json.dumps([r.to_dict() for r in rows], ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


def cmd_engines(args) -> int:
    cfg = _cfg(args)
    rows = []
    for name in available_engines():
        try:
            engine = get_engine(name, cfg)
            ok, reason = engine.availability()
            rows.append(
                {
                    "name": name,
                    "version": engine.version,
                    "available": ok,
                    "reason": reason,
                    "offsite": engine.sends_images_offsite,
                    "description": engine.description,
                }
            )
        except Exception as exc:
            rows.append({"name": name, "available": False, "reason": str(exc), "offsite": False, "description": ""})
    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
        return 0
    for row in rows:
        mark = "OK " if row["available"] else "-- "
        offsite = " [SENDS PHOTOS OFF-DEVICE]" if row["offsite"] else ""
        print(f"{mark}{row['name']:<14} {row.get('version', ''):<32}{offsite}")
        print(f"    {row.get('description', '')}")
        print(f"    {row.get('reason', '')}")
    return 0


def cmd_dedupe(args) -> int:
    with _open_store(args) as store:
        rows = store.images()
        exact = find_exact_duplicates(rows)
        near = find_near_duplicates(rows, distance=args.distance)
        print(f"exact duplicate groups: {len(exact)}")
        for digest, group in exact.items():
            print(f"  {digest[:12]}: " + ", ".join(str(g["rel_path"]) for g in group))
        print(f"near-duplicate pairs (dHash <= {args.distance}): {len(near)}")
        for a, b, dist in near[:50]:
            print(f"  {a['rel_path']} <-> {b['rel_path']} (distance {dist})")
        if args.json:
            Path(args.json).write_text(
                json.dumps(
                    {
                        "exact": {k: [str(g["rel_path"]) for g in v] for k, v in exact.items()},
                        "near": [{"a": str(a["rel_path"]), "b": str(b["rel_path"]), "distance": d} for a, b, d in near],
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
    return 0


def _resolve_image_id(store: Store, token: str) -> int:
    if token.isdigit():
        return int(token)
    row = store.image_by_rel_path(token)
    if row is None:
        # allow the absolute path too, as a convenience
        for image in store.images():
            if image["abs_path"] == token:
                return int(image["id"])
        raise SystemExit(f"error: image not found: {token}")
    return int(row["id"])


def cmd_label(args) -> int:
    cfg = _cfg(args)
    valid = {lab.id for lab in cfg.all_labels}
    with _open_store(args) as store:
        if args.list:
            for lab in cfg.all_labels:
                print(f"{lab.id:<14} {lab.name_ja}")
            return 0
        if args.csv:
            count = 0
            with open(args.csv, newline="", encoding="utf-8-sig") as fh:
                reader = csv.DictReader(fh)
                if not reader.fieldnames or not {"rel_path", "label"}.issubset(set(reader.fieldnames)):
                    raise SystemExit("error: CSV must have columns: rel_path,label")
                pairs = []
                for row in reader:
                    label = row["label"].strip()
                    if label not in valid:
                        raise SystemExit(f"error: unknown label {label!r} in CSV")
                    pairs.append((_resolve_image_id(store, row["rel_path"].strip()), label))
                count = store.set_labels(pairs)
            print(f"labels written: {count}")
            return 0
        if not args.image or not args.label:
            raise SystemExit("error: use --image ID --label LABEL, or --csv FILE, or --list")
        if args.label not in valid:
            raise SystemExit(f"error: unknown label {args.label!r}; valid: {', '.join(sorted(valid))}")
        image_id = _resolve_image_id(store, str(args.image))
        if args.clear:
            store.clear_label(image_id)
            print(f"cleared label for image {image_id}")
        else:
            store.set_label(image_id, args.label, note=args.note)
            print(f"image {image_id} -> {args.label}")
    return 0


def cmd_review(args) -> int:
    from .review_server import serve

    cfg = _cfg(args)
    store = _open_store(args)
    input_path = _input_path(args, store, required=False)
    serve(store, cfg, input_dir=input_path, host=args.host, port=args.port, open_browser=args.open)
    return 0


def cmd_report(args) -> int:
    cfg = _cfg(args)
    with _open_store(args) as store:
        report = build_report(store, cfg, include_paths=args.include_paths)
        text = report_markdown(report)
        if args.json:
            Path(args.json).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        if args.out:
            Path(args.out).write_text(text, encoding="utf-8")
            print(f"wrote {args.out}")
        else:
            print(text)
    return 0


def cmd_init_config(args) -> int:
    path = Path(args.out)
    if path.exists() and not args.force:
        raise SystemExit(f"error: {path} exists (use --force)")
    if args.preset == "default":
        data = DEFAULT_CONFIG
    else:
        data = load_config(args.preset).raw
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {path}")
    return 0


# --------------------------------------------------------------------------- #
# parser
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sports-sorter",
        description="Classify sports photos into per-sport folders, locally and safely.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--db", help=f"SQLite database (default: {DEFAULT_DB})")
    parser.add_argument("--config", help="taxonomy/prompt JSON (default: built-in)")
    parser.add_argument("--input", help="input directory (remembered after the first scan)")
    parser.add_argument("--i-know-what-i-am-doing", action="store_true", help="allow a normally blocked input path")
    parser.add_argument("-q", "--quiet", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("scan", help="index images (read-only)")
    p.add_argument("--recursive", action="store_true")
    p.set_defaults(func=cmd_scan)

    p = sub.add_parser("classify", help="run one or more engines")
    p.add_argument("--engine", action="append", help="engine name; repeat or comma-separate")
    p.add_argument("--device", default=os.environ.get("SPORTS_SORTER_DEVICE", "auto"), choices=["auto", "cpu", "mps", "cuda"])
    p.add_argument("--crops", default="center", choices=["none", "center", "five"])
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--limit", type=int)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force", action="store_true", help="recompute even if cached")
    p.add_argument("--no-scan", action="store_true", help="do not re-hash the input tree")
    p.add_argument("--recursive", action="store_true")
    p.add_argument("--include-errors", action="store_true", help="retry files that failed earlier")
    p.set_defaults(func=cmd_classify)

    p = sub.add_parser("finalize", help="fuse engine predictions into the final label")
    p.add_argument("--exclude", action="append", help="engine(s) to exclude from fusion")
    p.set_defaults(func=cmd_finalize)

    p = sub.add_parser("export", help="copy images into per-sport folders (never overwrites)")
    p.add_argument("--out", required=True)
    p.add_argument("--engine", help="fallback engine when no human/fusion label exists")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--exclude-unclassified", action="store_true")
    p.add_argument("--no-verify", action="store_true")
    p.add_argument("--json")
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("verify", help="verify exports and originals by SHA-256")
    p.add_argument("--json")
    p.set_defaults(func=cmd_verify)

    p = sub.add_parser("evaluate", help="accuracy against human labels")
    p.add_argument("--engine", default="fusion")
    p.add_argument("--all", action="store_true")
    p.add_argument("--json")
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("compare", help="compare all engines against human labels")
    p.add_argument("--engine", action="append")
    p.add_argument("--json")
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("engines", help="list engines and their availability")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_engines)

    p = sub.add_parser("dedupe", help="exact and near-duplicate report")
    p.add_argument("--distance", type=int, default=6)
    p.add_argument("--json")
    p.set_defaults(func=cmd_dedupe)

    p = sub.add_parser("label", help="set human ground-truth labels (never overwritten by models)")
    p.add_argument("--image", help="image id or relative path")
    p.add_argument("--label")
    p.add_argument("--csv", help="CSV with rel_path,label")
    p.add_argument("--list", action="store_true")
    p.add_argument("--clear", action="store_true")
    p.add_argument("--note")
    p.set_defaults(func=cmd_label)

    p = sub.add_parser("review", help="start the local review UI")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--open", action="store_true")
    p.set_defaults(func=cmd_review)

    p = sub.add_parser("report", help="aggregate report (safe to share by default)")
    p.add_argument("--json")
    p.add_argument("--out")
    p.add_argument("--include-paths", action="store_true")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("init-config", help="write an editable taxonomy JSON")
    p.add_argument("--out", default="config.json")
    p.add_argument("--preset", default="default", help="'default' or a path to an existing config")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_init_config)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except SafetyError as exc:
        print(f"safety: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130
    except BrokenPipeError:  # pragma: no cover
        return 0
    except SystemExit:
        raise
    except Exception as exc:
        if os.environ.get("SPORTS_SORTER_DEBUG"):
            raise
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
