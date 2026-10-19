import json

from sports_sorter.report import build_report, report_markdown


def test_report_counts_latest_prediction_per_engine(store, cfg, image_dir):
    from sports_sorter.images import scan_directory

    records = scan_directory(image_dir, extensions=cfg.extensions)
    ids = [store.upsert_image(rec) for rec in records]
    store.set_labels([(i, "volleyball") for i in ids])

    old = store.start_run("fake", "1", "old")
    for image_id in ids:
        store.record_prediction(
            image_id=image_id,
            run_id=old,
            engine="fake",
            config_hash="old",
            label_id="badminton",
            confidence=0.9,
        )
    new = store.start_run("fake", "2", "new")
    for image_id in ids:
        store.record_prediction(
            image_id=image_id,
            run_id=new,
            engine="fake",
            config_hash="new",
            label_id="volleyball",
            confidence=0.8,
        )

    report = build_report(store, cfg)

    # One vote per image, not one per historical run.
    assert report["counts_by_engine"]["fake"] == {"volleyball": len(ids)}
    assert report["images"]["human_labeled"] == len(ids)
    assert report["human_labels"] == {"volleyball": len(ids)}
    assert report["evaluation"]["fake"]["evaluated"] == len(ids)

    text = report_markdown(report)
    assert f"Human-labeled (ground truth): {len(ids)}" in text
    assert "| fake | 6 |" in text


def test_report_is_path_free_by_default(store, cfg, image_dir):
    from sports_sorter.images import scan_directory

    records = scan_directory(image_dir, extensions=cfg.extensions)
    ids = [store.upsert_image(rec) for rec in records]
    store.set_labels([(i, "volleyball") for i in ids])
    run = store.start_run("fake", "1", "h")
    for image_id in ids:
        store.record_prediction(
            image_id=image_id,
            run_id=run,
            engine="fake",
            config_hash="h",
            label_id="badminton",
            confidence=0.9,
        )

    payload = json.dumps(build_report(store, cfg), ensure_ascii=False)
    assert str(image_dir) not in payload
    assert "img_0.png" not in payload
    assert "paths_note" not in payload
    assert '"exact"' not in payload
    assert '"near"' not in payload
    assert '"rel_path"' not in payload

    with_paths = build_report(store, cfg, include_paths=True)
    assert with_paths["paths_note"]
    assert with_paths["evaluation"]["fake"]["misclassified"][0]["rel_path"]
    assert with_paths["duplicates"]["exact"]
