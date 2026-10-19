"""Image discovery, hashing, perceptual hashing and preprocessing.

Nothing in this module writes to the scanned directory. Files are opened
read-only, EXIF orientation is honoured so that the model and the review UI see
the same pixels a human would (orientation is a *technical* transform, never a
classification signal), and the filename/EXIF never leaks into a prediction.
"""

from __future__ import annotations

import hashlib
import io
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

from PIL import Image, ImageOps

try:  # optional, enables HEIC/HEIF on macOS builds of Pillow
    import pillow_heif  # type: ignore

    pillow_heif.register_heif_opener()
except Exception:  # pragma: no cover - optional dependency
    pass

Image.MAX_IMAGE_PIXELS = 400_000_000  # be permissive but not unbounded

DEFAULT_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".heic",
    ".heif",
    ".tif",
    ".tiff",
    ".bmp",
    ".gif",
}


@dataclass
class ImageRecord:
    """One candidate image on disk."""

    rel_path: str
    abs_path: str
    size: int
    sha256: str
    width: int = 0
    height: int = 0
    format: str = ""
    phash: str = ""
    mtime: float = 0.0
    error: str | None = None
    extra: dict = field(default_factory=dict)

    @property
    def ext(self) -> str:
        return Path(self.rel_path).suffix.lower()


def sha256_file(path: str | os.PathLike[str], chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def iter_image_paths(
    root: str | os.PathLike[str],
    *,
    recursive: bool = False,
    extensions: Iterable[str] | None = None,
) -> Iterator[Path]:
    """Yield image paths under ``root`` (sorted, deterministic)."""
    exts = {e.lower() for e in (extensions or DEFAULT_EXTENSIONS)}
    base = Path(root)
    if recursive:
        candidates = (p for p in base.rglob("*") if p.is_file())
    else:
        candidates = (p for p in base.iterdir() if p.is_file())
    for p in sorted(candidates, key=lambda x: str(x)):
        if p.name.startswith("."):
            continue
        if p.suffix.lower() in exts:
            yield p


def _read_exif_orientation(path: Path) -> int:
    try:
        with Image.open(path) as im:
            exif = im.getexif()
            return int(exif.get(0x0112, 1) or 1)
    except Exception:
        return 1


def load_prepared(
    path: str | os.PathLike[str],
    *,
    max_side: int = 1024,
    mode: str = "RGB",
) -> Image.Image:
    """Open an image read-only, apply EXIF orientation and downscale.

    Returns a new in-memory image; the file is closed before returning so no
    handle is kept on the user's originals.
    """
    p = Path(path)
    with open(p, "rb") as fh:  # explicit rb: never writable
        data = fh.read()
    with Image.open(io.BytesIO(data)) as im:
        im = ImageOps.exif_transpose(im)
        if im.mode != mode:
            im = im.convert(mode)
        if max_side and max(im.size) > max_side:
            ratio = max_side / float(max(im.size))
            new_size = (max(1, int(im.width * ratio)), max(1, int(im.height * ratio)))
            im = im.resize(new_size, Image.LANCZOS)
        return im.copy()


def dhash(image: Image.Image, hash_size: int = 8) -> str:
    """Difference hash (perceptual) used for near-duplicate detection."""
    im = image.convert("L").resize((hash_size + 1, hash_size), Image.LANCZOS)
    # L-mode bytes are one byte per pixel, row-major: no deprecated getdata().
    pixels = im.tobytes()
    bits = []
    for row in range(hash_size):
        for col in range(hash_size):
            left = pixels[row * (hash_size + 1) + col]
            right = pixels[row * (hash_size + 1) + col + 1]
            bits.append("1" if left > right else "0")
    return "{:0{}x}".format(int("".join(bits), 2), (hash_size * hash_size) // 4)


def hamming(a: str, b: str) -> int:
    """Hamming distance between two hex-encoded hashes."""
    ia, ib = int(a, 16), int(b, 16)
    return bin(ia ^ ib).count("1")


def scan_directory(
    root: str | os.PathLike[str],
    *,
    recursive: bool = False,
    extensions: Iterable[str] | None = None,
    compute_phash: bool = True,
    phash_side: int = 384,
    progress=None,
) -> list[ImageRecord]:
    """Hash and describe every image under ``root``.

    A file that cannot be decoded is still returned with ``error`` set, so the
    caller can report it instead of silently dropping it.
    """
    records: list[ImageRecord] = []
    base = Path(root)
    for p in iter_image_paths(base, recursive=recursive, extensions=extensions):
        rel = str(p.relative_to(base))
        try:
            st = p.stat()
        except OSError as exc:
            records.append(ImageRecord(rel, str(p), 0, "", error=f"stat failed: {exc}"))
            continue
        try:
            digest = sha256_file(p)
        except OSError as exc:
            records.append(ImageRecord(rel, str(p), st.st_size, "", error=f"read failed: {exc}", mtime=st.st_mtime))
            continue
        rec = ImageRecord(rel_path=rel, abs_path=str(p), size=st.st_size, sha256=digest, mtime=st.st_mtime)
        try:
            with open(p, "rb") as fh:
                with Image.open(fh) as im:
                    rec.width, rec.height = im.size
                    rec.format = (im.format or "").lower()
                    orientation = int(im.getexif().get(0x0112, 1) or 1)
        except Exception as exc:
            rec.error = f"decode failed: {exc}"
        else:
            if compute_phash:
                try:
                    img = load_prepared(p, max_side=phash_side)
                    rec.phash = dhash(img)
                except Exception as exc:  # pragma: no cover - defensive
                    rec.error = rec.error or f"phash failed: {exc}"
            # Orientation is stored for reporting only. It is never a feature.
            rec.extra["orientation"] = orientation
        records.append(rec)
        if progress is not None:
            progress(rec)
    return records


def _field(obj, name: str):
    if isinstance(obj, dict):
        return obj.get(name)
    try:
        return obj[name]
    except (TypeError, IndexError):
        return getattr(obj, name, None)


def find_exact_duplicates(records: Iterable[ImageRecord]) -> dict[str, list]:
    groups: dict[str, list] = {}
    for rec in records:
        if _field(rec, "error") or not _field(rec, "sha256"):
            continue
        groups.setdefault(_field(rec, "sha256"), []).append(rec)
    return {k: v for k, v in groups.items() if len(v) > 1}


def find_near_duplicates(records: Iterable[ImageRecord], *, distance: int = 6) -> list[tuple]:
    """O(n^2) near-duplicate search on dHash. Fine for tens of thousands."""
    usable = [r for r in records if _field(r, "phash") and not _field(r, "error")]
    out: list[tuple] = []
    for i in range(len(usable)):
        for j in range(i + 1, len(usable)):
            d = hamming(_field(usable[i], "phash"), _field(usable[j], "phash"))
            if d <= distance:
                out.append((usable[i], usable[j], d))
    return out
