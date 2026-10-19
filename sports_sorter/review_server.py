"""Lightweight local review UI (stdlib only, no build step, no CDN).

The server is read-mostly and binds to ``127.0.0.1`` by default. It exposes
only images that are already in the SQLite database (looked up by primary key),
so it can never be tricked into serving an arbitrary file. Thumbnails are
generated on demand into a cache directory and are never copied into the repo.

Features
--------
* paginated thumbnail grid (works with thousands of images),
* filter by final label / unclassified / disagreement / not-yet-reviewed,
* inline correction and bulk correction (writes the human ``labels`` table),
* side-by-side engine verdicts so disagreements are obvious.
"""

from __future__ import annotations

import json
import mimetypes
import os
import threading
import webbrowser
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from PIL import Image

from .config import Config
from .images import load_prepared
from .safety import SafetyError

WEB_DIR = Path(__file__).resolve().parent / "web"
THUMB_CACHE = Path(os.environ.get("SPORTS_SORTER_CACHE", "data/cache")).expanduser() / "thumbs"
THUMB_CACHE.mkdir(parents=True, exist_ok=True)

_thumb_lock = threading.Lock()


def final_label(store, config: Config, image_id: int, engine: str | None = None) -> tuple[str | None, float | None, str]:
    human = store.label_row(image_id)
    if human:
        return human["label_id"], None, "human"
    fused = store.prediction(image_id, "fusion")
    if fused and fused["label_id"]:
        return fused["label_id"], fused["confidence"], "fusion"
    if engine:
        row = store.prediction(image_id, engine)
        if row and row["label_id"]:
            return row["label_id"], row["confidence"], engine
    return None, None, "none"


def _thumb_path(image_id: int, size: int, source: Path) -> Path:
    stamp = int(source.stat().st_mtime) if source.exists() else 0
    return THUMB_CACHE / f"{image_id}_{size}_{stamp}.jpg"


def get_thumbnail(image_id: int, source: Path, size: int = 240) -> Path:
    size = max(96, min(640, int(size)))
    target = _thumb_path(image_id, size, source)
    if target.exists():
        return target
    with _thumb_lock:
        if target.exists():
            return target
        image = load_prepared(source, max_side=max(size * 2, 512))
        image.thumbnail((size, size), Image.LANCZOS)
        tmp = target.with_suffix(".tmp")
        image.convert("RGB").save(tmp, format="JPEG", quality=82)
        tmp.replace(target)
    return target


class ReviewState:
    def __init__(self, store, config: Config, input_dir: Path | None):
        self.store = store
        self.config = config
        self.input_dir = input_dir

    def summary(self) -> dict[str, Any]:
        images = self.store.images()
        human = self.store.labels()
        fused = {int(r["image_id"]): r for r in self.store.conn.execute("SELECT * FROM predictions WHERE engine='fusion'")}
        counts = Counter()
        for image in images:
            iid = int(image["id"])
            label = human.get(iid) or (fused.get(iid)["label_id"] if iid in fused else None) or "unclassified"
            counts[label] += 1
        return {
            "total": len(images),
            "errors": sum(1 for i in images if i["error"]),
            "reviewed": len(human),
            "counts": dict(counts),
            "labels": [{"id": l.id, "name_ja": l.name_ja} for l in self.config.all_labels],
            "engines": self.store.engines_seen(),
        }

    def rows(self, *, label: str | None, page: int, page_size: int, only: str | None, engine: str | None, q: str | None):
        images = self.store.images()
        human = self.store.labels()
        pmap = self.store.prediction_map()
        out = []
        for image in images:
            iid = int(image["id"])
            preds = pmap.get(iid, {})
            fused = preds.get("fusion")
            picked_label, confidence, source = final_label(self.store, self.config, iid, engine)
            votes = {
                name: {"label": row["label_id"], "confidence": row["confidence"], "status": row["status"]}
                for name, row in preds.items()
                if name not in {"fusion"}
            }
            distinct = {v["label"] for v in votes.values() if v["label"]}
            disagreement = len(distinct) > 1
            if q and q.lower() not in image["rel_path"].lower():
                continue
            if only == "disagreement" and not disagreement:
                continue
            if only == "unreviewed" and iid in human:
                continue
            if only == "unclassified" and picked_label != "unclassified":
                continue
            if label and picked_label != label:
                continue
            out.append(
                {
                    "id": iid,
                    "rel_path": image["rel_path"],
                    "width": image["width"],
                    "height": image["height"],
                    "error": image["error"],
                    "phash": image["phash"],
                    "sha256": image["sha256"][:12],
                    "label": picked_label,
                    "confidence": confidence,
                    "source": source,
                    "human": human.get(iid),
                    "disagreement": disagreement,
                    "votes": votes,
                    "thumb": f"/api/thumb/{iid}?size=240",
                }
            )
        total = len(out)
        start = max(0, (page - 1) * page_size)
        return {"total": total, "page": page, "page_size": page_size, "items": out[start : start + page_size]}


def make_handler(state: ReviewState):
    class Handler(BaseHTTPRequestHandler):
        server_version = "SportsSorterReview/0.1"

        # -- utilities ------------------------------------------------------
        def log_message(self, fmt, *args):  # keep the terminal quiet
            if os.environ.get("SPORTS_SORTER_DEBUG"):
                super().log_message(fmt, *args)

        def _json(self, payload: Any, status: int = 200) -> None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def _file(self, path: Path, content_type: str | None = None) -> None:
            if not path.exists() or not path.is_file():
                self._json({"error": "not found"}, 404)
                return
            data = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", content_type or mimetypes.guess_type(path.name)[0] or "application/octet-stream")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(data)

        def _check_origin(self) -> bool:
            # Same-origin only: a malicious site must not be able to POST labels.
            origin = self.headers.get("Origin")
            if origin and urlparse(origin).netloc not in {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}:
                self._json({"error": "origin not allowed"}, 403)
                return False
            return True

        def _body(self) -> dict:
            length = int(self.headers.get("Content-Length") or 0)
            if length > 2_000_000:
                raise ValueError("body too large")
            raw = self.rfile.read(length) if length else b"{}"
            return json.loads(raw.decode("utf-8") or "{}")

        # -- GET ------------------------------------------------------------
        def do_GET(self):
            parsed = urlparse(self.path)
            query = parse_qs(parsed.query)
            if parsed.path in {"/", "/index.html"}:
                self._file(WEB_DIR / "index.html", "text/html; charset=utf-8")
                return
            if parsed.path.startswith("/static/"):
                name = parsed.path.rsplit("/", 1)[-1]
                if name not in {"app.js", "style.css"}:
                    self._json({"error": "not found"}, 404)
                    return
                self._file(WEB_DIR / name)
                return
            if parsed.path == "/api/summary":
                self._json(state.summary())
                return
            if parsed.path == "/api/state":
                try:
                    page = max(1, int(query.get("page", ["1"])[0]))
                    page_size = max(1, min(200, int(query.get("page_size", ["60"])[0])))
                except ValueError:
                    page, page_size = 1, 60
                payload = state.rows(
                    label=query.get("label", [None])[0] or None,
                    page=page,
                    page_size=page_size,
                    only=query.get("only", [None])[0] or None,
                    engine=query.get("engine", [None])[0] or None,
                    q=query.get("q", [None])[0] or None,
                )
                self._json(payload)
                return
            if parsed.path.startswith("/api/thumb/"):
                try:
                    image_id = int(parsed.path.rsplit("/", 1)[-1])
                    size = int(query.get("size", ["240"])[0])
                except ValueError:
                    self._json({"error": "bad id"}, 400)
                    return
                image = state.store.image(image_id)
                if image is None:
                    self._json({"error": "not found"}, 404)
                    return
                source = Path(image["abs_path"])
                try:
                    thumb = get_thumbnail(image_id, source, size)
                except Exception as exc:
                    self._json({"error": f"thumbnail failed: {exc}"}, 500)
                    return
                self._file(thumb, "image/jpeg")
                return
            if parsed.path.startswith("/api/full/"):
                try:
                    image_id = int(parsed.path.rsplit("/", 1)[-1])
                except ValueError:
                    self._json({"error": "bad id"}, 400)
                    return
                image = state.store.image(image_id)
                if image is None:
                    self._json({"error": "not found"}, 404)
                    return
                source = Path(image["abs_path"])
                try:
                    preview = load_prepared(source, max_side=1600)
                    import io

                    buf = io.BytesIO()
                    preview.convert("RGB").save(buf, format="JPEG", quality=88)
                    data = buf.getvalue()
                except Exception as exc:
                    self._json({"error": str(exc)}, 500)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(data)
                return
            self._json({"error": "not found"}, 404)

        # -- POST -----------------------------------------------------------
        def do_POST(self):
            if not self._check_origin():
                return
            parsed = urlparse(self.path)
            try:
                body = self._body()
            except Exception as exc:
                self._json({"error": f"bad body: {exc}"}, 400)
                return
            valid = {lab.id for lab in state.config.all_labels}
            if parsed.path == "/api/label":
                label = body.get("label_id")
                if label not in valid:
                    self._json({"error": "unknown label"}, 400)
                    return
                image_id = int(body.get("image_id"))
                if state.store.image(image_id) is None:
                    self._json({"error": "unknown image"}, 404)
                    return
                state.store.set_label(image_id, label, note=body.get("note"))
                self._json({"ok": True, "image_id": image_id, "label": label})
                return
            if parsed.path == "/api/bulk":
                label = body.get("label_id")
                if label not in valid:
                    self._json({"error": "unknown label"}, 400)
                    return
                ids = [int(i) for i in body.get("image_ids", [])]
                pairs = [(i, label) for i in ids if state.store.image(i) is not None]
                count = state.store.set_labels(pairs)
                self._json({"ok": True, "updated": count})
                return
            if parsed.path == "/api/clear":
                image_id = int(body.get("image_id"))
                state.store.clear_label(image_id)
                self._json({"ok": True})
                return
            self._json({"error": "not found"}, 404)

    return Handler


def serve(store, config: Config, *, input_dir: Path | None, host: str = "127.0.0.1", port: int = 8765, open_browser: bool = False):
    state = ReviewState(store, config, input_dir)
    handler = make_handler(state)
    httpd = ThreadingHTTPServer((host, port), handler)
    url = f"http://{host}:{port}/"
    print(f"review UI: {url}")
    print("  Ctrl-C to stop")
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        httpd.server_close()
