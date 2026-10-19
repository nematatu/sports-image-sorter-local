from pathlib import Path

import pytest

from sports_sorter.config import DEFAULT_CONFIG, config_hash, load_config
from sports_sorter.safety import SafetyError, assert_input_allowed, assert_safe_output, is_within


def test_default_taxonomy_has_expected_labels():
    cfg = load_config(None)
    ids = {lab.id for lab in cfg.labels}
    assert {"badminton", "volleyball", "table_tennis", "tennis", "basketball", "track_field", "swimming"} <= ids
    assert {lab.id for lab in cfg.fallback_labels} == {"other", "unclassified"}


def test_resolve_aliases():
    cfg = load_config(None)
    assert cfg.resolve("volley ball") == "volleyball"
    assert cfg.resolve("table_tennis") == "table_tennis"
    assert cfg.resolve("バレーボール") == "volleyball"
    assert cfg.resolve("badminton") == "badminton"
    assert cfg.resolve("curling") is None


def test_dir_names_are_japanese():
    cfg = load_config(None)
    assert cfg.dir_name("volleyball") == "バレーボール"
    assert cfg.dir_name("unclassified") == "未分類"
    assert cfg.dir_name("other") == "その他の競技"


def test_custom_config_merges_and_hashes(tmp_path):
    p = tmp_path / "cfg.json"
    p.write_text('{"min_confidence": 0.9, "labels": [{"id": "curling", "name_ja": "カーリング", "prompts": ["curling"]}]}', encoding="utf-8")
    cfg = load_config(p)
    assert cfg.min_confidence == 0.9
    # A partial label list replaces the default list, other keys survive.
    assert [lab.id for lab in cfg.labels] == ["curling"]
    assert cfg.raw["ambiguity_margin"] == DEFAULT_CONFIG["ambiguity_margin"]
    assert config_hash(cfg) != config_hash(load_config(None))


def test_prompts_are_non_empty_for_every_sport():
    cfg = load_config(None)
    for lab in cfg.labels:
        assert lab.prompts, lab.id


def test_safety_blocks_downloads(tmp_path):
    downloads = Path.home() / "Downloads"
    if downloads.exists():
        with pytest.raises(SafetyError):
            assert_input_allowed(downloads)
    # An ordinary temp dir is fine.
    assert assert_input_allowed(tmp_path) == tmp_path.resolve()


def test_safety_output_must_not_be_inside_input(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    with pytest.raises(SafetyError):
        assert_safe_output(src, src)
    with pytest.raises(SafetyError):
        assert_safe_output(src, src / "out")
    with pytest.raises(SafetyError):
        assert_safe_output(src / "child", src)


def test_safety_output_creates_directory(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    out = assert_safe_output(src, tmp_path / "out")
    assert out.is_dir()
    assert is_within(out, tmp_path)
