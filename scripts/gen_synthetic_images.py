#!/usr/bin/env python3
"""Generate deterministic synthetic images for a *pipeline* smoke test.

These drawings are deliberately simple and contain no real people. They are
useful for exercising scan -> classify -> finalize -> export (for example with
the ``fake`` engine) and for checking that the review UI can render files.

They are **not** a benchmark: a model's accuracy on coloured geometric shapes
says nothing about its accuracy on real sports photos. The reference evaluation
uses a private, manually labelled photo set that is not distributed with this
repository.

Examples
--------
    python scripts/gen_synthetic_images.py --out data/synthetic
    python scripts/gen_synthetic_images.py --out data/synthetic --per-sport 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from PIL import Image, ImageDraw
except ImportError:  # pragma: no cover - friendly message for a missing core dep
    print("Pillow is required: pip install pillow", file=sys.stderr)
    raise SystemExit(2)

SPORTS = ("badminton", "volleyball", "table_tennis", "portrait")
COLORS = {
    "badminton": ((18, 92, 62), (236, 244, 238)),
    "volleyball": ((194, 108, 52), (250, 246, 232)),
    "table_tennis": ((35, 68, 132), (238, 242, 250)),
    "portrait": ((92, 92, 112), (232, 232, 240)),
}


def _player(draw: ImageDraw.ImageDraw, x: int, y: int, color: tuple[int, int, int], scale: float = 1.0):
    """A stick figure; no identity, no face detail."""
    head = max(6, int(12 * scale))
    draw.ellipse([x - head, y - head * 3, x + head, y - head], fill=color)
    draw.line([x, y - head, x, y + int(30 * scale)], fill=color, width=max(3, int(5 * scale)))
    draw.line([x, y - head // 2, x - int(24 * scale), y + int(8 * scale)], fill=color, width=max(3, int(5 * scale)))
    draw.line([x, y - head // 2, x + int(24 * scale), y + int(8 * scale)], fill=color, width=max(3, int(5 * scale)))
    draw.line([x, y + int(30 * scale), x - int(18 * scale), y + int(62 * scale)], fill=color, width=max(3, int(5 * scale)))
    draw.line([x, y + int(30 * scale), x + int(18 * scale), y + int(62 * scale)], fill=color, width=max(3, int(5 * scale)))


def _badminton(draw: ImageDraw.ImageDraw, w: int, h: int, rng):
    line = COLORS["badminton"][1]
    draw.rectangle([int(w * 0.05), int(h * 0.08), int(w * 0.95), int(h * 0.92)], outline=line, width=3)
    draw.line([int(w * 0.12), int(h * 0.72), int(w * 0.88), int(h * 0.52)], fill=line, width=2)
    draw.line([int(w * 0.12), int(h * 0.34), int(w * 0.88), int(h * 0.30)], fill=line, width=2)
    draw.line([int(w * 0.20), int(h * 0.12), int(w * 0.20), int(h * 0.88)], fill=line, width=5)
    _player(draw, int(w * 0.36), int(h * 0.49), (245, 220, 120), 1.1)
    sx, sy = int(w * 0.66), int(h * 0.31)
    draw.ellipse([sx, sy, sx + 12, sy + 12], fill=(255, 255, 255))
    for dx in (-8, 0, 8):
        draw.line([sx + 6, sy + 3, sx + 6 + dx, sy - 20], fill=(255, 255, 255), width=2)


def _volleyball(draw: ImageDraw.ImageDraw, w: int, h: int, rng):
    line = COLORS["volleyball"][1]
    draw.rectangle([int(w * 0.04), int(h * 0.10), int(w * 0.96), int(h * 0.92)], outline=line, width=3)
    draw.line([int(w * 0.50), int(h * 0.10), int(w * 0.50), int(h * 0.92)], fill=(235, 235, 235), width=2)
    draw.line([int(w * 0.10), int(h * 0.66), int(w * 0.90), int(h * 0.58)], fill=line, width=2)
    draw.line([int(w * 0.10), int(h * 0.26), int(w * 0.90), int(h * 0.24)], fill=(245, 245, 245), width=5)
    draw.rectangle([int(w * 0.08), int(h * 0.12), int(w * 0.92), int(h * 0.16)], fill=(245, 245, 245))
    _player(draw, int(w * 0.30), int(h * 0.47), (250, 235, 190), 1.0)
    _player(draw, int(w * 0.70), int(h * 0.49), (250, 235, 190), 1.0)
    cx, cy, r = int(w * 0.51), int(h * 0.24), max(12, int(min(w, h) * 0.06))
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(250, 250, 250), outline=(190, 190, 190), width=2)
    draw.arc([cx - r, cy - r, cx + r, cy + r], 20, 160, fill=(190, 190, 190), width=2)
    draw.arc([cx - r, cy - r, cx + r, cy + r], 200, 340, fill=(190, 190, 190), width=2)


def _table_tennis(draw: ImageDraw.ImageDraw, w: int, h: int, rng):
    line = COLORS["table_tennis"][1]
    draw.polygon(
        [
            (int(w * 0.08), int(h * 0.55)),
            (int(w * 0.68), int(h * 0.48)),
            (int(w * 0.92), int(h * 0.72)),
            (int(w * 0.30), int(h * 0.83)),
        ],
        fill=(48, 110, 190),
        outline=line,
    )
    draw.line([int(w * 0.32), int(h * 0.47), int(w * 0.53), int(h * 0.78)], fill=(245, 245, 245), width=5)
    draw.line([int(w * 0.08), int(h * 0.56), int(w * 0.30), int(h * 0.84)], fill=line, width=2)
    # Paddle and ball.
    px, py = int(w * 0.68), int(h * 0.30)
    draw.ellipse([px - 26, py - 34, px + 26, py + 18], fill=(190, 45, 45), outline=(255, 220, 180), width=2)
    draw.line([px + 18, py + 10, px + 46, py + 54], fill=(235, 205, 165), width=7)
    draw.ellipse([int(w * 0.79), int(h * 0.22), int(w * 0.79) + 12, int(h * 0.22) + 12], fill=(255, 245, 200))


def _portrait(draw: ImageDraw.ImageDraw, w: int, h: int, rng):
    # No equipment, no venue, no sport-specific cue: a valid "unclassified" case.
    draw.rectangle([0, 0, w, h], fill=(92, 92, 112))
    draw.ellipse([int(w * 0.30), int(h * 0.18), int(w * 0.70), int(h * 0.62)], fill=(214, 184, 154))
    draw.ellipse([int(w * 0.38), int(h * 0.34), int(w * 0.44), int(h * 0.40)], fill=(45, 45, 55))
    draw.ellipse([int(w * 0.56), int(h * 0.34), int(w * 0.62), int(h * 0.40)], fill=(45, 45, 55))
    draw.arc([int(w * 0.42), int(h * 0.40), int(w * 0.58), int(h * 0.54)], 20, 160, fill=(120, 70, 70), width=3)
    draw.polygon(
        [(int(w * 0.18), h), (int(w * 0.32), int(h * 0.60)), (int(w * 0.68), int(h * 0.60)), (int(w * 0.82), h)],
        fill=(44, 56, 78),
    )


DRAWERS = {
    "badminton": _badminton,
    "volleyball": _volleyball,
    "table_tennis": _table_tennis,
    "portrait": _portrait,
}


def generate(out_dir: Path, *, per_sport: int, width: int, height: int, seed: int) -> list[Path]:
    import random

    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for sport in SPORTS:
        for index in range(per_sport):
            rng = random.Random(f"{seed}:{sport}:{index}")
            image = Image.new("RGB", (width, height), COLORS[sport][0])
            draw = ImageDraw.Draw(image)
            # Small deterministic variation so files are not byte-identical.
            for _ in range(12):
                x = rng.randrange(width)
                y = rng.randrange(height)
                r = rng.randrange(2, 7)
                draw.ellipse([x - r, y - r, x + r, y + r], fill=COLORS[sport][1])
            DRAWERS[sport](draw, width, height, rng)
            path = out_dir / f"synthetic_{sport}_{index + 1:02d}.png"
            if path.exists():
                continue
            image.save(path)
            written.append(path)
    return written


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="data/synthetic", help="output directory (default: data/synthetic)")
    parser.add_argument("--per-sport", type=int, default=1, help="images per sport/portrait category")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--seed", type=int, default=20261020)
    parser.add_argument("--force", action="store_true", help="overwrite generated files from a previous run")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.per_sport < 1:
        raise SystemExit("--per-sport must be >= 1")
    if args.width < 64 or args.height < 64:
        raise SystemExit("--width and --height must be >= 64")
    out = Path(args.out).expanduser()
    if args.force:
        for sport in SPORTS:
            for index in range(args.per_sport):
                (out / f"synthetic_{sport}_{index + 1:02d}.png").unlink(missing_ok=True)
    written = generate(out, per_sport=args.per_sport, width=args.width, height=args.height, seed=args.seed)
    skipped = args.per_sport * len(SPORTS) - len(written)
    print(f"wrote {len(written)} image(s) to {out}")
    if skipped:
        print(f"skipped {skipped} existing file(s); use --force to regenerate")
    print("synthetic images are for pipeline smoke tests, not accuracy benchmarks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
