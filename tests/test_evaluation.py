from sports_sorter.evaluation import evaluate, format_eval
from sports_sorter.images import scan_directory


def _setup(store, cfg, image_dir, pairs):
    recs = scan_directory(image_dir, extensions=cfg.extensions)
    ids = [store.upsert_image(r) for r in recs]
    run = store.start_run("fake", "1", "h")
    for iid, (truth, predicted) in zip(ids, pairs):
        store.set_label(iid, truth)
        store.record_prediction(image_id=iid, run_id=run, engine="fake", config_hash="h", label_id=predicted, confidence=0.8)
    return ids


def test_accuracy_and_confusion(store, cfg, image_dir):
    # 6 images: 4 correct, 2 wrong
    pairs = [("volleyball", "volleyball"), ("volleyball", "volleyball"), ("badminton", "badminton"),
             ("badminton", "tennis"), ("unclassified", "unclassified"), ("table_tennis", "tennis")]
    _setup(store, cfg, image_dir, pairs)
    result = evaluate(store, cfg, "fake")
    assert result.evaluated == 6
    assert result.correct == 4
    assert abs(result.accuracy - 4 / 6) < 1e-9
    assert result.confusion["badminton"]["tennis"] == 1
    assert result.per_label["volleyball"]["f1"] == 1.0
    assert len(result.misclassified) == 2
    assert "accuracy" in format_eval(result)


def test_no_labels_means_no_accuracy(store, cfg, image_dir):
    recs = scan_directory(image_dir, extensions=cfg.extensions)
    for rec in recs:
        store.upsert_image(rec)
    run = store.start_run("fake", "1", "h")
    store.record_prediction(image_id=1, run_id=run, engine="fake", config_hash="h", label_id="tennis", confidence=0.9)
    result = evaluate(store, cfg, "fake")
    assert result.labeled == 0
    assert result.accuracy is None
    assert "n/a" in format_eval(result)


def test_missing_prediction_reduces_coverage(store, cfg, image_dir):
    recs = scan_directory(image_dir, extensions=cfg.extensions)
    ids = [store.upsert_image(r) for r in recs]
    for iid in ids:
        store.set_label(iid, "volleyball")
    run = store.start_run("fake", "1", "h")
    store.record_prediction(image_id=ids[0], run_id=run, engine="fake", config_hash="h", label_id="volleyball", confidence=0.9)
    result = evaluate(store, cfg, "fake")
    assert result.labeled == 6
    assert result.evaluated == 1
    assert abs(result.coverage - 1 / 6) < 1e-9
