import time

from sports_sorter.db import Store
from sports_sorter.images import scan_directory


def test_upsert_is_idempotent(store, image_dir, cfg):
    records = scan_directory(image_dir, extensions=cfg.extensions)
    ids1 = [store.upsert_image(r) for r in records]
    ids2 = [store.upsert_image(r) for r in records]
    assert ids1 == ids2
    assert len(store.images()) == len(records)


def test_prediction_upsert_and_prediction_map(store, image_dir, cfg):
    records = scan_directory(image_dir, extensions=cfg.extensions)
    image_id = store.upsert_image(records[0])
    run = store.start_run("fake", "1", "hash1")
    store.record_prediction(image_id=image_id, run_id=run, engine="fake", config_hash="hash1", label_id="volleyball", confidence=0.8)
    store.record_prediction(image_id=image_id, run_id=run, engine="fake", config_hash="hash1", label_id="tennis", confidence=0.9)
    row = store.prediction(image_id, "fake", "hash1")
    assert row["label_id"] == "tennis"
    assert len(store.prediction_map() [image_id]) == 1


def test_new_config_hash_creates_second_row(store, image_dir, cfg):
    records = scan_directory(image_dir, extensions=cfg.extensions)
    image_id = store.upsert_image(records[0])
    run = store.start_run("fake", "1", "h")
    store.record_prediction(image_id=image_id, run_id=run, engine="fake", config_hash="h1", label_id="tennis", confidence=0.5)
    store.record_prediction(image_id=image_id, run_id=run, engine="fake", config_hash="h2", label_id="volleyball", confidence=0.5)
    assert len(store.prediction_map()[image_id]) == 1  # one per engine
    assert store.prediction(image_id, "fake", "h1")["label_id"] == "tennis"
    assert store.prediction(image_id, "fake", "h2")["label_id"] == "volleyball"


def test_human_labels_are_separate_from_predictions(store, image_dir, cfg):
    records = scan_directory(image_dir, extensions=cfg.extensions)
    image_id = store.upsert_image(records[0])
    run = store.start_run("fake", "1", "h")
    store.record_prediction(image_id=image_id, run_id=run, engine="fake", config_hash="h", label_id="tennis", confidence=0.9)
    store.set_label(image_id, "volleyball", source="human")
    assert store.labels()[image_id] == "volleyball"
    # Updating predictions must never overwrite the human label.
    store.record_prediction(image_id=image_id, run_id=run, engine="fake", config_hash="h", label_id="badminton", confidence=0.9)
    assert store.labels()[image_id] == "volleyball"


def test_bulk_labels(store, image_dir, cfg):
    records = scan_directory(image_dir, extensions=cfg.extensions)
    ids = [store.upsert_image(r) for r in records]
    assert store.set_labels([(i, "unclassified") for i in ids]) == len(ids)
    assert set(store.labels().values()) == {"unclassified"}


def test_export_ledger_requires_existing_image(store, image_dir, cfg):
    records = scan_directory(image_dir, extensions=cfg.extensions)
    image_id = store.upsert_image(records[0])
    store.record_export(image_id=image_id, label_id="tennis", dest_path="/tmp/x", bytes_=1, sha256="abc")
    assert len(store.exports()) == 1
