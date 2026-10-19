"""Copy classified images into per-sport folders, without ever overwriting.

Guarantees enforced here (and tested in ``tests/test_exporter.py``):

* the source is opened read-only and only ``shutil.copyfileobj``-style reads
  happen, so the original cannot be modified;
* a destination file is created with ``O_CREAT | O_EXCL``; if the name exists,
  ``name__1.ext`` / ``name__2.ext`` ... is chosen instead. An existing file is
  never truncated or replaced;
* every copied file is re-hashed and compared with the source hash before the
  export is recorded;
* the ledger in SQLite makes exports resumable and verifiable later.
"""

from __future__ import annotations

import errno
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from .config import Config
from .images import sha256_file
from .safety import assert_safe_output


@dataclass
class ExportItem:
    image_id: int
    source: Path
    dest: Path
    label_id: str
    sha256: str
    bytes: int
    action: str  # copy | skip | error
    reason: str = ""


@dataclass
class ExportReport:
    out_dir: Path
    items: list[ExportItem] = field(default_factory=list)
    copied: int = 0
    skipped: int = 0
    errors: int = 0
    bytes_copied: int = 0
    per_label: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "out_dir": str(self.out_dir),
            "copied": self.copied,
            "skipped": self.skipped,
            "errors": self.errors,
            "bytes_copied": self.bytes_copied,
            "per_label": self.per_label,
            "items": [
                {
                    "image_id": i.image_id,
                    "source": str(i.source),
                    "dest": str(i.dest),
                    "label": i.label_id,
                    "sha256": i.sha256,
                    "bytes": i.bytes,
                    "action": i.action,
                    "reason": i.reason,
                }
                for i in self.items
            ],
        }


def resolve_label(store, config: Config, image_id: int, *, engine: str | None = None) -> tuple[str | None, str]:
    """Human label wins, then the fusion result, then ``engine``'s prediction."""
    human = store.label_row(image_id)
    if human:
        return human["label_id"], "human"
    fused = store.prediction(image_id, "fusion")
    if fused and fused["label_id"]:
        return fused["label_id"], "fusion"
    if engine:
        row = store.prediction(image_id, engine)
        if row and row["label_id"]:
            return row["label_id"], engine
    return None, "none"


def _unique_destination(directory: Path, filename: str, reserved: set[Path] | None = None) -> Path:
    """Return a path that does not exist yet (no overwrite, ever)."""
    reserved = reserved or set()
    stem = Path(filename).stem
    suffix = Path(filename).suffix
    candidate = directory / filename
    counter = 1
    while candidate.exists() or candidate in reserved:
        candidate = directory / f"{stem}__{counter}{suffix}"
        counter += 1
        if counter > 10_000:  # pragma: no cover - pathological directory
            raise RuntimeError(f"cannot find a free filename for {filename} in {directory}")
    return candidate


def _copy_exclusive(source: Path, dest: Path) -> None:
    """Copy ``source`` to ``dest`` without ever overwriting ``dest``."""
    fd = os.open(dest, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(fd, "wb") as out, open(source, "rb") as src:
            shutil.copyfileobj(src, out, length=1 << 20)
            out.flush()
            os.fsync(out.fileno())
    except BaseException:
        # Do not leave a half-written file that looks like a successful copy,
        # including when the user interrupts the copy with Ctrl-C.
        try:
            dest.unlink()
        except OSError:
            pass
        raise
    shutil.copystat(source, dest, follow_symlinks=False)


def prepare_directories(config: Config, out_dir: Path) -> dict[str, Path]:
    dirs: dict[str, Path] = {}
    for lab in config.labels:
        d = out_dir / lab.dir_name
        d.mkdir(parents=True, exist_ok=True)
        dirs[lab.id] = d
    other = out_dir / config.raw.get("other_dir", "その他の競技")
    other.mkdir(parents=True, exist_ok=True)
    dirs["other"] = other
    unclassified = out_dir / config.raw.get("unclassified_dir", "未分類")
    unclassified.mkdir(parents=True, exist_ok=True)
    dirs["unclassified"] = unclassified
    return dirs


def plan_export(
    store,
    config: Config,
    input_dir: str | Path,
    out_dir: str | Path,
    *,
    engine: str | None = None,
    include_unclassified: bool = True,
) -> ExportReport:
    input_dir = Path(input_dir).expanduser().resolve()
    out_path = Path(out_dir).expanduser()
    report = ExportReport(out_dir=out_path)
    dirs = prepare_directories(config, out_path)
    reserved: set[Path] = set()
    for image in store.images():
        image_id = int(image["id"])
        label, source_kind = resolve_label(store, config, image_id, engine=engine)
        if label is None:
            report.errors += 1
            report.items.append(
                ExportItem(image_id, Path(image["abs_path"]), Path(), "", "", int(image["size"] or 0), "error", "no label")
            )
            continue
        if label == "unclassified" and not include_unclassified:
            report.skipped += 1
            continue
        directory = dirs.get(label)
        if directory is None:
            # Unknown label in the DB (e.g. taxonomy changed): keep it visible.
            directory = out_path / "未分類"
            directory.mkdir(parents=True, exist_ok=True)
            label = "unclassified"
        src = Path(image["abs_path"])
        if not src.exists():
            report.errors += 1
            report.items.append(ExportItem(image_id, src, Path(), label, image["sha256"], 0, "error", "source missing"))
            continue
        dest = _unique_destination(directory, src.name, reserved)
        reserved.add(dest)
        report.items.append(ExportItem(image_id, src, dest, label, image["sha256"], int(image["size"] or 0), "copy"))
        report.per_label[label] = report.per_label.get(label, 0) + 1
    return report


def export_images(
    store,
    config: Config,
    input_dir: str | Path,
    out_dir: str | Path,
    *,
    engine: str | None = None,
    dry_run: bool = False,
    include_unclassified: bool = True,
    verify: bool = True,
) -> ExportReport:
    input_path = Path(input_dir).expanduser().resolve()
    out_path = assert_safe_output(input_path, out_dir)
    report = plan_export(store, config, input_path, out_path, engine=engine, include_unclassified=include_unclassified)
    if dry_run:
        for item in report.items:
            if item.action == "copy":
                report.skipped += 1
        return report

    for item in report.items:
        if item.action != "copy":
            continue
        existing = store.conn.execute("SELECT * FROM exports WHERE image_id=?", (item.image_id,)).fetchone()
        if (
            existing
            and existing["label_id"] == item.label_id
            and Path(existing["dest_path"]).exists()
            and existing["sha256"] == sha256_file(item.source)
        ):
            item.action = "skip"
            item.reason = "already exported (hash matches)"
            report.skipped += 1
            continue
        try:
            _copy_exclusive(item.source, item.dest)
            digest = sha256_file(item.dest)
            if verify and digest != sha256_file(item.source):
                item.dest.unlink(missing_ok=True)
                raise RuntimeError("hash mismatch after copy; destination removed")
            item.action = "copy"
            report.copied += 1
            report.bytes_copied += item.dest.stat().st_size
            store.record_export(
                image_id=item.image_id,
                label_id=item.label_id,
                dest_path=str(item.dest),
                bytes_=item.dest.stat().st_size,
                sha256=digest,
            )
        except OSError as exc:
            item.action = "error"
            item.reason = f"{type(exc).__name__}: {exc}"
            report.errors += 1
            if exc.errno == errno.EEXIST:  # extremely unlikely race
                item.reason = "destination appeared while copying; skipped"
                report.errors -= 1
                report.skipped += 1
    return report


@dataclass
class VerifyResult:
    checked: int = 0
    ok: int = 0
    missing_dest: int = 0
    modified_dest: int = 0
    modified_source: int = 0
    missing_source: int = 0
    problems: list[dict] = field(default_factory=list)

    @property
    def ok_all(self) -> bool:
        return not (self.missing_dest or self.modified_dest or self.modified_source or self.missing_source)

    def to_dict(self) -> dict:
        return {
            "checked": self.checked,
            "ok": self.ok,
            "missing_dest": self.missing_dest,
            "modified_dest": self.modified_dest,
            "modified_source": self.modified_source,
            "missing_source": self.missing_source,
            "all_ok": self.ok_all,
            "problems": self.problems,
        }


def verify_exports(store, *, check_sources: bool = True) -> VerifyResult:
    """Verify every recorded export against the original's stored SHA-256."""
    result = VerifyResult()
    for row in store.exports():
        result.checked += 1
        src = Path(store.image(int(row["image_id"]))["abs_path"]) if store.image(int(row["image_id"])) else None
        dest = Path(row["dest_path"])
        if not dest.exists():
            result.missing_dest += 1
            result.problems.append({"image_id": row["image_id"], "problem": "destination missing", "dest": str(dest)})
            continue
        dest_hash = sha256_file(dest)
        if dest_hash != row["sha256"]:
            result.modified_dest += 1
            result.problems.append({"image_id": row["image_id"], "problem": "destination hash changed", "dest": str(dest)})
            continue
        if check_sources:
            if src is None or not src.exists():
                result.missing_source += 1
                result.problems.append({"image_id": row["image_id"], "problem": "source missing", "source": str(src)})
                continue
            if sha256_file(src) != row["sha256"]:
                result.modified_source += 1
                result.problems.append({"image_id": row["image_id"], "problem": "ORIGINAL changed", "source": str(src)})
                continue
        result.ok += 1
        store.mark_export_verified(int(row["image_id"]), True)
    return result


def verify_originals(store, input_dir: str | Path) -> list[dict]:
    """Re-hash the input directory and report files that changed on disk."""
    input_path = Path(input_dir).expanduser().resolve()
    problems: list[dict] = []
    for image in store.images():
        path = input_path / image["rel_path"]
        if not path.exists():
            problems.append({"rel_path": image["rel_path"], "problem": "missing"})
            continue
        if sha256_file(path) != image["sha256"]:
            problems.append({"rel_path": image["rel_path"], "problem": "changed"})
    return problems
