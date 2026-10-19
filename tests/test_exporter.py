import hashlib
from pathlib import Path

import pytest

from sports_sorter.exporter import (
    _copy_exclusive,
    _unique_destination,
    export_images,
    plan_export,
    verify_exports,
    verify_originals,
)
from sports_sorter.images import find_exact_duplicates, scan_directory


def _prepare(store, cfg, image_dir, labels=None):
    records = scan_directory(image_dir, extensions=cfg.extensions)
    ids = [store.upsert_image(r) for r in records]
    for i, image_id in enumerate(ids):
        store.set_label(image_id, (labels or ["volleyball"] * len(ids))[i])
    return records, ids


def test_dry_run_copies_nothing(store, cfg, image_dir, tmp_path):
    _prepare(store, cfg, image_dir)
    out = tmp_path / "sorted"
    report = export_images(store, cfg, image_dir, out, dry_run=True)
    assert report.copied == 0
    assert not any(p.is_file() for p in out.rglob("*"))


def test_export_copies_and_never_modifies_originals(store, cfg, image_dir, tmp_path):
    records, _ = _prepare(store, cfg, image_dir)
    before = {r.rel_path: hashlib.sha256(Path(r.abs_path).read_bytes()).hexdigest() for r in records}
    out = tmp_path / "sorted"
    report = export_images(store, cfg, image_dir, out)
    assert report.copied == len(records)
    assert report.errors == 0
    exports = {int(e["image_id"]): e for e in store.exports()}
    for rec in records:
        image_id = int(store.image_by_rel_path(rec.rel_path)["id"])
        dest = Path(exports[image_id]["dest_path"])
        assert dest.read_bytes() == Path(rec.abs_path).read_bytes()
    after = {r.rel_path: hashlib.sha256(Path(r.abs_path).read_bytes()).hexdigest() for r in records}
    assert before == after
    # Output dirs for all configured sports exist, even when empty.
    for lab in cfg.labels:
        assert (out / lab.dir_name).is_dir()
    assert (out / "未分類").is_dir()


def test_export_is_resumable_and_does_not_duplicate(store, cfg, image_dir, tmp_path):
    _prepare(store, cfg, image_dir)
    out = tmp_path / "sorted"
    first = export_images(store, cfg, image_dir, out)
    second = export_images(store, cfg, image_dir, out)
    assert first.copied == 6
    assert second.copied == 0
    assert second.skipped == 6
    assert len(list(out.rglob("*.png"))) == 6


def test_existing_file_is_never_overwritten(store, cfg, image_dir, tmp_path):
    _prepare(store, cfg, image_dir)
    out = tmp_path / "sorted"
    target = out / "バレーボール" / "img_0.png"
    target.parent.mkdir(parents=True)
    target.write_text("user file that must survive", encoding="utf-8")
    report = export_images(store, cfg, image_dir, out)
    assert target.read_text(encoding="utf-8") == "user file that must survive"
    assert (target.parent / "img_0__1.png").exists()
    assert report.copied == 6


def test_collision_naming_is_stable(tmp_path):
    d = tmp_path / "d"
    d.mkdir()
    (d / "a.png").write_text("x")
    p1 = _unique_destination(d, "a.png")
    assert p1.name == "a__1.png"
    p1.write_text("y")
    p2 = _unique_destination(d, "a.png")
    assert p2.name == "a__2.png"


def test_copy_exclusive_refuses_existing(tmp_path):
    src = tmp_path / "src.txt"
    src.write_text("hello")
    dest = tmp_path / "dest.txt"
    dest.write_text("keep me")
    with pytest.raises(FileExistsError):
        _copy_exclusive(src, dest)
    assert dest.read_text() == "keep me"


def test_copy_exclusive_removes_partial_file_on_interrupt(tmp_path, monkeypatch):
    import shutil

    src = tmp_path / "src.bin"
    src.write_bytes(b"x" * 1024)
    dest = tmp_path / "dest.bin"

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(shutil, "copyfileobj", interrupted)
    with pytest.raises(KeyboardInterrupt):
        _copy_exclusive(src, dest)
    assert not dest.exists()


def test_verify_detects_destination_tampering(store, cfg, image_dir, tmp_path):
    _prepare(store, cfg, image_dir)
    out = tmp_path / "sorted"
    export_images(store, cfg, image_dir, out)
    good = verify_exports(store)
    assert good.ok_all and good.checked == 6
    dest = Path(store.exports()[0]["dest_path"])
    dest.write_bytes(b"tampered")
    bad = verify_exports(store)
    assert not bad.ok_all
    assert bad.modified_dest == 1


def test_verify_detects_original_change(store, cfg, image_dir, tmp_path):
    records, _ = _prepare(store, cfg, image_dir)
    out = tmp_path / "sorted"
    export_images(store, cfg, image_dir, out)
    Path(records[0].abs_path).write_bytes(b"original changed")
    problems = verify_originals(store, image_dir)
    assert any(p["problem"] == "changed" for p in problems)
    assert not verify_exports(store).ok_all


def test_plan_export_without_labels_reports_error(store, cfg, image_dir, tmp_path):
    records = scan_directory(image_dir, extensions=cfg.extensions)
    for rec in records:
        store.upsert_image(rec)
    report = export_images(store, cfg, image_dir, tmp_path / "out", dry_run=True)
    assert report.errors == len(records)


def test_safety_refuses_output_inside_input(store, cfg, image_dir):
    _prepare(store, cfg, image_dir)
    from sports_sorter.safety import SafetyError

    with pytest.raises(SafetyError):
        export_images(store, cfg, image_dir, image_dir / "nested")
