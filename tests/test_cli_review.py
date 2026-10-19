import json
import sqlite3
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

from sports_sorter.cli import main
from sports_sorter.review_server import ReviewState, get_thumbnail, make_handler


def test_cli_end_to_end(tmp_path, image_dir, capsys):
    db = tmp_path / "data" / "sports.db"
    out = tmp_path / "sorted"
    argv_common = ["--db", str(db), "--input", str(image_dir)]

    assert main([*argv_common, "scan"]) == 0
    assert main([*argv_common, "classify", "--engine", "fake", "--no-scan"]) == 0
    assert main(["--db", str(db), "finalize"]) == 0
    assert main(["--db", str(db), "label", "--image", "img_0.png", "--label", "volleyball"]) == 0
    assert main(["--db", str(db), "export", "--out", str(out)]) == 0

    assert (out / "バレーボール" / "img_0.png").exists()
    assert len(list(out.rglob("*.png"))) == 6
    # All configured sport directories exist.
    assert (out / "バドミントン").is_dir()
    assert (out / "未分類").is_dir()

    # Re-running export is a no-op (resume, no duplicates).
    assert main(["--db", str(db), "export", "--out", str(out)]) == 0
    assert len(list(out.rglob("*.png"))) == 6

    assert main(["--db", str(db), "verify"]) == 0
    assert main(["--db", str(db), "report", "--json", str(tmp_path / "report.json")]) == 0
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["images"]["total"] == 6
    assert report["exports"]["verified"] == 6


def test_cli_dry_run_does_not_create_predictions(tmp_path, image_dir):
    db = tmp_path / "sports.db"
    assert main(["--db", str(db), "--input", str(image_dir), "scan"]) == 0
    assert main(["--db", str(db), "classify", "--engine", "fake", "--dry-run", "--no-scan"]) == 0
    con = sqlite3.connect(db)
    assert con.execute("SELECT COUNT(*) FROM predictions").fetchone()[0] == 0
    con.close()


def test_cli_unknown_label_is_rejected(tmp_path, image_dir):
    import pytest

    db = tmp_path / "sports.db"
    assert main(["--db", str(db), "--input", str(image_dir), "scan"]) == 0
    with pytest.raises(SystemExit):
        main(["--db", str(db), "label", "--image", "img_0.png", "--label", "curling"])
    con = sqlite3.connect(db)
    assert con.execute("SELECT COUNT(*) FROM labels").fetchone()[0] == 0
    con.close()


def test_review_server_api(tmp_path, image_dir, cfg):
    from sports_sorter.db import Store
    from sports_sorter.images import scan_directory

    store = Store(tmp_path / "review.db")
    for rec in scan_directory(image_dir, extensions=cfg.extensions):
        store.upsert_image(rec)
    run = store.start_run("fake", "1", "h")
    store.record_prediction(image_id=1, run_id=run, engine="fake", config_hash="h", label_id="volleyball", confidence=0.9)

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(ReviewState(store, cfg, image_dir)))
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/summary") as resp:
            summary = json.loads(resp.read())
        assert summary["total"] == 6
        assert any(lab["id"] == "unclassified" for lab in summary["labels"])

        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/state?page=1&page_size=10") as resp:
            payload = json.loads(resp.read())
        assert payload["total"] == 6

        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/thumb/1?size=120") as resp:
            assert resp.headers["Content-Type"] == "image/jpeg"
            assert len(resp.read()) > 100

        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/label",
            data=json.dumps({"image_id": 1, "label_id": "badminton"}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req) as resp:
            assert json.loads(resp.read())["ok"] is True
        assert store.label_row(1)["label_id"] == "badminton"

        # Unknown label is rejected.
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/label",
            data=json.dumps({"image_id": 1, "label_id": "nope"}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            urllib.request.urlopen(req)
            raise AssertionError("expected HTTP 400")
        except urllib.error.HTTPError as exc:
            assert exc.code == 400
    finally:
        httpd.shutdown()
        httpd.server_close()
        store.close()


def test_thumbnail_is_generated_once(tmp_path, image_dir):
    from sports_sorter.images import scan_directory

    rec = scan_directory(image_dir, extensions={".png"})[0]
    first = get_thumbnail(999999, __import__("pathlib").Path(rec.abs_path), 120)
    second = get_thumbnail(999999, __import__("pathlib").Path(rec.abs_path), 120)
    assert first == second
    assert first.exists()
