"""SQLite persistence: images, prediction runs, human labels and exports.

The schema is intentionally boring and append-friendly so that every step of
the pipeline can be interrupted (Ctrl-C, crash, power loss) and resumed:

* ``images``      - one row per file discovered, keyed by relative path.
* ``runs``        - one row per ``(engine, config)`` execution.
* ``predictions`` - one row per ``(image, run)``; unique constraint makes
                    re-running an engine idempotent.
* ``labels``      - human ground truth / corrections ("gold" labels). Written
                    only by the ``label`` command and the review UI, never by
                    an engine, so model output can never be mistaken for truth.
* ``exports``     - copy ledger, used by ``verify`` to prove that the export is
                    complete and byte-identical to the originals.
"""

from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS images (
    id          INTEGER PRIMARY KEY,
    rel_path    TEXT NOT NULL UNIQUE,
    abs_path    TEXT NOT NULL,
    size        INTEGER NOT NULL DEFAULT 0,
    sha256      TEXT NOT NULL DEFAULT '',
    phash       TEXT NOT NULL DEFAULT '',
    width       INTEGER NOT NULL DEFAULT 0,
    height      INTEGER NOT NULL DEFAULT 0,
    format      TEXT NOT NULL DEFAULT '',
    mtime       REAL NOT NULL DEFAULT 0,
    error       TEXT,
    extra       TEXT NOT NULL DEFAULT '{}',
    added_at    REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
    id            INTEGER PRIMARY KEY,
    engine        TEXT NOT NULL,
    engine_version TEXT NOT NULL DEFAULT '',
    config_hash   TEXT NOT NULL DEFAULT '',
    options       TEXT NOT NULL DEFAULT '{}',
    started_at    REAL NOT NULL,
    finished_at   REAL,
    status        TEXT NOT NULL DEFAULT 'running',
    notes         TEXT
);

CREATE TABLE IF NOT EXISTS predictions (
    id          INTEGER PRIMARY KEY,
    image_id    INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
    run_id      INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    engine      TEXT NOT NULL,
    config_hash TEXT NOT NULL DEFAULT '',
    label_id    TEXT,
    confidence  REAL,
    scores      TEXT NOT NULL DEFAULT '{}',
    raw         TEXT NOT NULL DEFAULT '{}',
    status      TEXT NOT NULL DEFAULT 'ok',
    error       TEXT,
    duration_ms INTEGER,
    created_at  REAL NOT NULL,
    UNIQUE (image_id, engine, config_hash)
);

CREATE INDEX IF NOT EXISTS idx_predictions_image ON predictions(image_id);
CREATE INDEX IF NOT EXISTS idx_predictions_engine ON predictions(engine);

CREATE TABLE IF NOT EXISTS labels (
    image_id   INTEGER PRIMARY KEY REFERENCES images(id) ON DELETE CASCADE,
    label_id   TEXT NOT NULL,
    source     TEXT NOT NULL DEFAULT 'human',
    note       TEXT,
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS exports (
    id          INTEGER PRIMARY KEY,
    image_id    INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
    run_id      INTEGER,
    label_id    TEXT NOT NULL,
    dest_path   TEXT NOT NULL,
    bytes       INTEGER NOT NULL DEFAULT 0,
    sha256      TEXT NOT NULL DEFAULT '',
    verified    INTEGER NOT NULL DEFAULT 0,
    copied_at   REAL NOT NULL,
    UNIQUE (image_id)
);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


class Store:
    """Thin wrapper around a SQLite database."""

    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path), timeout=30.0, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    # -- lifecycle -----------------------------------------------------------
    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        try:
            yield self.conn
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    # -- meta ----------------------------------------------------------------
    def set_meta(self, key: str, value: Any) -> None:
        self.conn.execute(
            "INSERT INTO meta(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, json.dumps(value, ensure_ascii=False) if not isinstance(value, str) else value),
        )
        self.conn.commit()

    def get_meta(self, key: str, default: Any = None) -> Any:
        row = self.conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        if row is None:
            return default
        try:
            return json.loads(row["value"])
        except json.JSONDecodeError:
            return row["value"]

    # -- images --------------------------------------------------------------
    def upsert_image(self, rec) -> int:
        now = time.time()
        cur = self.conn.execute(
            """
            INSERT INTO images(rel_path, abs_path, size, sha256, phash, width, height, format, mtime, error, extra, added_at)
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(rel_path) DO UPDATE SET
              abs_path=excluded.abs_path, size=excluded.size, sha256=excluded.sha256,
              phash=excluded.phash, width=excluded.width, height=excluded.height,
              format=excluded.format, mtime=excluded.mtime, error=excluded.error, extra=excluded.extra
            """,
            (
                rec.rel_path,
                rec.abs_path,
                rec.size,
                rec.sha256,
                rec.phash,
                rec.width,
                rec.height,
                rec.format,
                rec.mtime,
                rec.error,
                json.dumps(getattr(rec, "extra", {}) or {}, ensure_ascii=False),
                now,
            ),
        )
        self.conn.commit()
        row = self.conn.execute("SELECT id FROM images WHERE rel_path=?", (rec.rel_path,)).fetchone()
        return int(row["id"])

    def images(self, *, order_by: str = "rel_path") -> list[sqlite3.Row]:
        allowed = {"rel_path", "id", "size", "width", "height"}
        col = order_by if order_by in allowed else "rel_path"
        return list(self.conn.execute(f"SELECT * FROM images ORDER BY {col}"))

    def image(self, image_id: int) -> sqlite3.Row | None:
        return self.conn.execute("SELECT * FROM images WHERE id=?", (image_id,)).fetchone()

    def image_by_rel_path(self, rel_path: str) -> sqlite3.Row | None:
        return self.conn.execute("SELECT * FROM images WHERE rel_path=?", (rel_path,)).fetchone()

    def image_ids(self) -> list[int]:
        return [int(r["id"]) for r in self.conn.execute("SELECT id FROM images ORDER BY rel_path")]

    # -- runs ----------------------------------------------------------------
    def start_run(self, engine: str, version: str, config_hash: str, options: dict | None = None) -> int:
        cur = self.conn.execute(
            "INSERT INTO runs(engine, engine_version, config_hash, options, started_at, status) VALUES(?,?,?,?,?,?)",
            (engine, version, config_hash, json.dumps(options or {}, ensure_ascii=False), time.time(), "running"),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def finish_run(self, run_id: int, status: str = "finished", notes: str | None = None) -> None:
        self.conn.execute(
            "UPDATE runs SET finished_at=?, status=?, notes=? WHERE id=?",
            (time.time(), status, notes, run_id),
        )
        self.conn.commit()

    # -- predictions ---------------------------------------------------------
    def record_prediction(
        self,
        *,
        image_id: int,
        run_id: int,
        engine: str,
        config_hash: str,
        label_id: str | None,
        confidence: float | None,
        scores: dict[str, float] | None = None,
        raw: dict | None = None,
        status: str = "ok",
        error: str | None = None,
        duration_ms: int | None = None,
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO predictions(image_id, run_id, engine, config_hash, label_id, confidence, scores, raw, status, error, duration_ms, created_at)
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(image_id, engine, config_hash) DO UPDATE SET
              run_id=excluded.run_id, label_id=excluded.label_id, confidence=excluded.confidence,
              scores=excluded.scores, raw=excluded.raw, status=excluded.status, error=excluded.error,
              duration_ms=excluded.duration_ms, created_at=excluded.created_at
            """,
            (
                image_id,
                run_id,
                engine,
                config_hash,
                label_id,
                confidence,
                json.dumps(scores or {}, ensure_ascii=False),
                json.dumps(raw or {}, ensure_ascii=False),
                status,
                error,
                duration_ms,
                time.time(),
            ),
        )
        self.conn.commit()

    def prediction(self, image_id: int, engine: str, config_hash: str | None = None) -> sqlite3.Row | None:
        if config_hash:
            return self.conn.execute(
                "SELECT * FROM predictions WHERE image_id=? AND engine=? AND config_hash=?",
                (image_id, engine, config_hash),
            ).fetchone()
        return self.conn.execute(
            "SELECT * FROM predictions WHERE image_id=? AND engine=? ORDER BY created_at DESC, id DESC LIMIT 1",
            (image_id, engine),
        ).fetchone()

    def prediction_map(self, engines: Sequence[str] | None = None) -> dict[int, dict[str, sqlite3.Row]]:
        """``{image_id: {engine: row}}`` for the newest prediction per engine."""
        sql = "SELECT * FROM predictions"
        params: list[Any] = []
        if engines:
            sql += " WHERE engine IN (%s)" % ",".join("?" * len(engines))
            params.extend(engines)
        sql += " ORDER BY created_at ASC, id ASC"
        out: dict[int, dict[str, sqlite3.Row]] = {}
        for row in self.conn.execute(sql, params):
            out.setdefault(int(row["image_id"]), {})[row["engine"]] = row
        return out

    def engines_seen(self) -> list[str]:
        return [r["engine"] for r in self.conn.execute("SELECT DISTINCT engine FROM predictions ORDER BY engine")]

    # -- human labels --------------------------------------------------------
    def set_label(self, image_id: int, label_id: str, *, source: str = "human", note: str | None = None) -> None:
        self.conn.execute(
            """
            INSERT INTO labels(image_id, label_id, source, note, updated_at) VALUES(?,?,?,?,?)
            ON CONFLICT(image_id) DO UPDATE SET label_id=excluded.label_id, source=excluded.source,
              note=excluded.note, updated_at=excluded.updated_at
            """,
            (image_id, label_id, source, note, time.time()),
        )
        self.conn.commit()

    def set_labels(self, pairs: Iterable[tuple[int, str]], *, source: str = "human") -> int:
        rows = [(int(i), str(l), source, time.time()) for i, l in pairs]
        with self.transaction() as conn:
            conn.executemany(
                """
                INSERT INTO labels(image_id, label_id, source, updated_at) VALUES(?,?,?,?)
                ON CONFLICT(image_id) DO UPDATE SET label_id=excluded.label_id,
                  source=excluded.source, updated_at=excluded.updated_at
                """,
                rows,
            )
        return len(rows)

    def clear_label(self, image_id: int) -> None:
        self.conn.execute("DELETE FROM labels WHERE image_id=?", (image_id,))
        self.conn.commit()

    def labels(self) -> dict[int, str]:
        return {int(r["image_id"]): r["label_id"] for r in self.conn.execute("SELECT image_id, label_id FROM labels")}

    def label_row(self, image_id: int) -> sqlite3.Row | None:
        return self.conn.execute("SELECT * FROM labels WHERE image_id=?", (image_id,)).fetchone()

    # -- exports -------------------------------------------------------------
    def record_export(
        self, *, image_id: int, label_id: str, dest_path: str, bytes_: int, sha256: str, run_id: int | None = None
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO exports(image_id, run_id, label_id, dest_path, bytes, sha256, verified, copied_at)
            VALUES(?,?,?,?,?,?,0,?)
            ON CONFLICT(image_id) DO UPDATE SET run_id=excluded.run_id, label_id=excluded.label_id,
              dest_path=excluded.dest_path, bytes=excluded.bytes, sha256=excluded.sha256, copied_at=excluded.copied_at
            """,
            (image_id, run_id, label_id, dest_path, bytes_, sha256, time.time()),
        )
        self.conn.commit()

    def exports(self) -> list[sqlite3.Row]:
        return list(self.conn.execute("SELECT * FROM exports ORDER BY dest_path"))

    def mark_export_verified(self, image_id: int, ok: bool) -> None:
        self.conn.execute("UPDATE exports SET verified=? WHERE image_id=?", (1 if ok else 0, image_id))
        self.conn.commit()

    # -- stats ---------------------------------------------------------------
    def counts_by_label(self, engine: str | None = None, config_hash: str | None = None) -> dict[str, int]:
        if engine:
            sql = "SELECT label_id, COUNT(*) AS n FROM predictions WHERE engine=?"
            params: list[Any] = [engine]
            if config_hash:
                sql += " AND config_hash=?"
                params.append(config_hash)
            sql += " GROUP BY label_id"
            rows = self.conn.execute(sql, params)
        else:
            rows = self.conn.execute("SELECT label_id, COUNT(*) AS n FROM labels GROUP BY label_id")
        return {(r["label_id"] or "unclassified"): int(r["n"]) for r in rows}
