"""Shared test fixtures. All images are synthetic (Pillow-drawn shapes)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sports_sorter.config import load_config  # noqa: E402
from sports_sorter.db import Store  # noqa: E402
from sports_sorter.images import scan_directory  # noqa: E402


def draw_synthetic(path: Path, seed: int = 0, size=(320, 240)) -> Path:
    """Draw a deterministic, obviously-synthetic image (no real people)."""
    img = Image.new("RGB", size, (30 + seed * 7 % 200, 80, 160))
    d = ImageDraw.Draw(img)
    d.rectangle([10, 10, size[0] - 10, size[1] - 10], outline=(255, 255, 255), width=3)
    d.ellipse([40 + seed, 50, 140 + seed, 150], fill=(240, 200, 40))
    d.line([0, size[1] - 40, size[0], 20], fill=(255, 80, 80), width=5)
    d.text((12, size[1] - 24), f"synthetic-{seed}", fill=(255, 255, 255))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    return path


@pytest.fixture()
def cfg():
    return load_config(None)


@pytest.fixture()
def image_dir(tmp_path):
    root = tmp_path / "images"
    for i in range(5):
        draw_synthetic(root / f"img_{i}.png", seed=i)
    # exact duplicate of img_0 under a different name
    (root / "dup.png").write_bytes((root / "img_0.png").read_bytes())
    return root


@pytest.fixture()
def store(tmp_path):
    s = Store(tmp_path / "test.db")
    yield s
    s.close()


@pytest.fixture()
def populated(store, cfg, image_dir):
    records = scan_directory(image_dir, extensions=cfg.extensions)
    ids = [store.upsert_image(r) for r in records]
    store.set_meta("input_dir", str(image_dir))
    return store, records, ids
