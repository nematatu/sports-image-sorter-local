"""Safety guards.

The tool is designed to be run against a photo library that the user cares
about, so the defaults are deliberately paranoid:

* Directories that are never read or written unless the caller explicitly
  overrides the guard (``~/Downloads``, ``~/Library``, system paths, ...).
* The output directory may not live inside the input directory, so an export
  can never race the scanner or pollute the source tree.
* Originals are only ever opened read-only; every write goes through a helper
  that refuses to overwrite an existing file.

These checks are cheap and run before any I/O, which makes the "never touch the
originals" promise testable (see ``tests/test_safety.py``).
"""

from __future__ import annotations

import os
from pathlib import Path


class SafetyError(RuntimeError):
    """Raised when an operation would violate a safety rule."""


# Paths that are blocked by default. ``~`` is expanded at import time.
BLOCKED_INPUT_PATTERNS: tuple[str, ...] = (
    "~/Downloads",
    "~/Library",
    "~/Movies",
    "~/Music",
    "~/Pictures/Photos Library.photoslibrary",
    "/System",
    "/Library",
    "/private/var/db",
    "/usr",
    "/bin",
    "/sbin",
    "/etc",
)


def _norm(path: str | os.PathLike[str]) -> Path:
    p = Path(path).expanduser()
    try:
        return p.resolve()
    except OSError:  # pragma: no cover - resolve of a missing path
        return p.absolute()


def blocked_paths() -> list[Path]:
    return [_norm(p) for p in BLOCKED_INPUT_PATTERNS]


def is_within(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def assert_input_allowed(path: str | os.PathLike[str], *, allow_blocked: bool = False) -> Path:
    """Validate a directory the tool is about to *read*.

    Raises :class:`SafetyError` for blocked locations unless the caller passes
    ``allow_blocked=True`` (which the CLI only does when the user typed the
    explicit ``--i-know-what-i-am-doing`` flag).
    """
    p = _norm(path)
    if not p.exists():
        raise SafetyError(f"input directory does not exist: {p}")
    if not p.is_dir():
        raise SafetyError(f"input path is not a directory: {p}")
    if not os.access(p, os.R_OK):
        raise SafetyError(f"input directory is not readable: {p}")
    if not allow_blocked:
        for blocked in blocked_paths():
            if p == blocked or is_within(p, blocked):
                raise SafetyError(
                    f"refusing to read {p}: it is inside the protected location {blocked}. "
                    "Pass --i-know-what-i-am-doing to override (documented in docs/safety.md)."
                )
    return p


def assert_safe_output(input_dir: str | os.PathLike[str], output_dir: str | os.PathLike[str]) -> Path:
    """Validate an export destination.

    The destination must not be the input directory, a parent of it, or inside
    it, and must not be a blocked location. The directory is created if needed.
    """
    src = _norm(input_dir)
    dst = _norm(output_dir)
    if dst == src:
        raise SafetyError("output directory must differ from the input directory")
    if is_within(src, dst):
        raise SafetyError(f"output directory {dst} contains the input directory {src}")
    if is_within(dst, src):
        raise SafetyError(f"output directory {dst} is inside the input directory {src}")
    for blocked in blocked_paths():
        if dst == blocked or is_within(dst, blocked):
            raise SafetyError(f"refusing to write into protected location {blocked}")
    dst.mkdir(parents=True, exist_ok=True)
    return dst


def ensure_within_root(path: Path, root: Path) -> Path:
    """Ensure ``path`` stays inside ``root`` (used when serving thumbnails)."""
    p = _norm(path)
    r = _norm(root)
    if not is_within(p, r) and p != r:
        raise SafetyError(f"path escapes root: {p} not under {r}")
    return p
