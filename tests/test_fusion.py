import json

from sports_sorter.engines.base import Prediction
from sports_sorter.engines.fusion import fuse


def _row(engine, label, conf):
    return {"engine": engine, "label_id": label, "confidence": conf, "scores": {label: conf}, "status": "ok"}


def test_weighted_vote_prefers_stronger_engine(cfg):
    pred = fuse([_row("siglip2", "volleyball", 0.9), _row("sportsnet7", "tennis", 0.5)], cfg)
    assert pred.label_id == "volleyball"


def test_agreement_increases_confidence(cfg):
    pred = fuse([_row("siglip2", "volleyball", 0.8), _row("clip", "volleyball", 0.8)], cfg)
    assert pred.label_id == "volleyball"
    assert pred.confidence > 0.6


def test_abstain_wins_when_models_are_unsure(cfg):
    pred = fuse(
        [
            _row("siglip2", "unclassified", 0.95),
            _row("clip", "unclassified", 0.8),
            _row("sports100_vit", "volleyball", 0.2),
        ],
        cfg,
    )
    assert pred.label_id == "unclassified"


def test_ambiguity_margin_forces_unclassified(cfg):
    pred = fuse(
        [_row("siglip2", "volleyball", 0.55), _row("clip", "badminton", 0.54)],
        cfg,
        ambiguity_margin=0.2,
    )
    assert pred.label_id == "unclassified"
    assert "ambiguous" in pred.raw["fusion"]


def test_low_confidence_forces_unclassified(cfg):
    pred = fuse([_row("apple_vision", "tennis", 0.1)], cfg, min_confidence=0.5)
    assert pred.label_id == "unclassified"


def test_disabled_engine_is_ignored(cfg):
    pred = fuse([_row("siglip2", "tennis", 0.9), _row("gemini", "volleyball", 0.99)], cfg)
    assert pred.label_id == "tennis"


def test_error_rows_are_ignored(cfg):
    rows = [
        {"engine": "siglip2", "label_id": "tennis", "confidence": 0.9, "status": "error"},
        _row("clip", "badminton", 0.7),
    ]
    assert fuse(rows, cfg).label_id == "badminton"


def test_no_votes_is_unclassified(cfg):
    pred = fuse([], cfg)
    assert pred.label_id == "unclassified"
    assert pred.confidence is None


def test_sqlite_row_with_json_scores(store, image_dir, cfg):
    from sports_sorter.images import scan_directory

    rec = scan_directory(image_dir, extensions=cfg.extensions)[0]
    iid = store.upsert_image(rec)
    run = store.start_run("x", "1", "h")
    store.record_prediction(image_id=iid, run_id=run, engine="clip", config_hash="h", label_id="swimming", confidence=0.8)
    row = store.prediction(iid, "clip")
    assert fuse([row], cfg).label_id == "swimming"


def test_fusion_engine_wrapper(cfg):
    from sports_sorter.engines.fusion import FusionEngine

    engine = FusionEngine(
        cfg,
        sources={
            "siglip2": Prediction(label_id="badminton", confidence=0.9),
            "clip": Prediction(label_id="badminton", confidence=0.7),
        },
    )
    assert engine.classify("ignored").label_id == "badminton"


def test_finalize_store_is_idempotent(store, image_dir, cfg):
    from sports_sorter.engines.fusion import finalize_store
    from sports_sorter.images import scan_directory

    recs = scan_directory(image_dir, extensions=cfg.extensions)
    ids = [store.upsert_image(r) for r in recs]
    run = store.start_run("clip", "1", "h")
    for iid in ids:
        store.record_prediction(image_id=iid, run_id=run, engine="clip", config_hash="h", label_id="volleyball", confidence=0.8)
    first = finalize_store(store, cfg)
    second = finalize_store(store, cfg)
    assert {i: p.label_id for i, p in first.items()} == {i: p.label_id for i, p in second.items()}
    assert all(p.label_id == "volleyball" for p in second.values())
    # fusion rows exist and are unique per image
    rows = store.conn.execute("SELECT COUNT(*) AS n FROM predictions WHERE engine='fusion'").fetchone()
    assert rows["n"] == len(ids)
