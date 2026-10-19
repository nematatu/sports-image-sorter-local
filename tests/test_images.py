from pathlib import Path

from PIL import Image

from sports_sorter.images import (
    dhash,
    find_exact_duplicates,
    find_near_duplicates,
    hamming,
    load_prepared,
    scan_directory,
    sha256_file,
)
from tests.conftest import draw_synthetic


def test_scan_reports_dimensions_hash_and_phash(image_dir, cfg):
    records = scan_directory(image_dir, extensions=cfg.extensions)
    assert len(records) == 6
    for rec in records:
        assert rec.error is None
        assert rec.sha256
        assert rec.phash
        assert rec.width == 320 and rec.height == 240
        assert rec.format == "png"


def test_scan_does_not_modify_input(image_dir, cfg):
    before = {p.name: sha256_file(p) for p in image_dir.iterdir()}
    scan_directory(image_dir, extensions=cfg.extensions)
    after = {p.name: sha256_file(p) for p in image_dir.iterdir()}
    assert before == after


def test_exact_duplicates_are_detected(image_dir, cfg):
    records = scan_directory(image_dir, extensions=cfg.extensions)
    groups = find_exact_duplicates(records)
    assert len(groups) == 1
    names = {r.rel_path for group in groups.values() for r in group}
    assert names == {"img_0.png", "dup.png"}


def test_near_duplicates_are_detected(tmp_path, cfg):
    a = draw_synthetic(tmp_path / "a.png", seed=1, size=(400, 300))
    img = Image.open(a).resize((200, 150), Image.LANCZOS)
    b = tmp_path / "b.png"
    img.save(b)
    draw_synthetic(tmp_path / "c.png", seed=4, size=(400, 300))
    records = scan_directory(tmp_path, extensions=cfg.extensions)
    pairs = find_near_duplicates(records, distance=10)
    assert any({p[0].rel_path, p[1].rel_path} == {"a.png", "b.png"} for p in pairs)
    d = hamming(dhash(Image.open(a)), dhash(Image.open(b)))
    assert d <= 10


def test_load_prepared_downscales_and_is_independent(tmp_path):
    p = draw_synthetic(tmp_path / "big.png", size=(1600, 1200))
    img = load_prepared(p, max_side=400)
    assert max(img.size) == 400
    img.putpixel((0, 0), (1, 2, 3))  # in-memory only
    assert Image.open(p).size == (1600, 1200)


def test_corrupt_file_is_reported_not_dropped(tmp_path):
    bad = tmp_path / "broken.jpg"
    bad.write_bytes(b"not an image")
    records = scan_directory(tmp_path)
    assert len(records) == 1
    assert records[0].error is not None


def test_hash_is_stable(image_dir):
    p = image_dir / "img_1.png"
    assert sha256_file(p) == sha256_file(p)
    assert len(sha256_file(p)) == 64


def test_non_image_files_are_ignored(tmp_path):
    (tmp_path / "notes.txt").write_text("hello", encoding="utf-8")
    draw_synthetic(tmp_path / "x.png")
    records = scan_directory(tmp_path)
    assert [r.rel_path for r in records] == ["x.png"]
