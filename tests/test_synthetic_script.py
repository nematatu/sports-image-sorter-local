import subprocess
import sys
from pathlib import Path

from PIL import Image


def _run(*args: str):
    root = Path(__file__).resolve().parents[1]
    return subprocess.run(
        [sys.executable, str(root / "scripts" / "gen_synthetic_images.py"), *args],
        capture_output=True,
        text=True,
        check=False,
    )


def test_synthetic_generator_writes_images_without_overwriting(tmp_path):
    out = tmp_path / "synthetic"
    result = _run("--out", str(out), "--per-sport", "1", "--width", "240", "--height", "180")
    assert result.returncode == 0, result.stderr
    images = sorted(out.glob("*.png"))
    assert len(images) == 4
    assert all(Image.open(p).size == (240, 180) for p in images)

    before = {p.name: p.read_bytes() for p in images}
    second = _run("--out", str(out), "--per-sport", "1", "--width", "240", "--height", "180")
    assert second.returncode == 0
    assert "skipped 4" in second.stdout
    assert before == {p.name: p.read_bytes() for p in images}
