import pytest

import sports_sorter.safety as safety
from sports_sorter.safety import (
    SafetyError,
    assert_input_allowed,
    assert_safe_output,
    ensure_within_root,
    is_within,
)


def test_blocked_input_requires_explicit_override(tmp_path, monkeypatch):
    blocked = tmp_path / "Downloads"
    blocked.mkdir()
    monkeypatch.setattr(safety, "BLOCKED_INPUT_PATTERNS", (str(blocked),))

    with pytest.raises(SafetyError, match="protected location"):
        assert_input_allowed(blocked)

    assert assert_input_allowed(blocked, allow_blocked=True) == blocked.resolve()


def test_output_cannot_wrap_or_nest_inside_input(tmp_path):
    source = tmp_path / "source"
    source.mkdir()

    with pytest.raises(SafetyError, match="contains the input"):
        assert_safe_output(source, tmp_path)
    with pytest.raises(SafetyError, match="inside the input"):
        assert_safe_output(source, source / "nested")
    with pytest.raises(SafetyError, match="must differ"):
        assert_safe_output(source, source)

    out = assert_safe_output(source, tmp_path / "out")
    assert out.is_dir()


def test_output_blocked_location_is_refused(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    blocked = tmp_path / "Library"
    blocked.mkdir()
    monkeypatch.setattr(safety, "BLOCKED_INPUT_PATTERNS", (str(blocked),))

    with pytest.raises(SafetyError, match="protected location"):
        assert_safe_output(source, blocked / "out")


def test_ensure_within_root_blocks_traversal(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    inside = root / "a" / "b.jpg"
    inside.parent.mkdir()
    inside.write_bytes(b"x")
    outside = tmp_path / "outside.jpg"
    outside.write_bytes(b"x")

    assert ensure_within_root(inside, root) == inside.resolve()
    with pytest.raises(SafetyError, match="escapes root"):
        ensure_within_root(outside, root)
    assert is_within(inside, root)
    assert not is_within(outside, root)
