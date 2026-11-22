"""SQLite log: ingestion manifest, query log, eval runs. Read by the dashboard."""

from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS manifest (
    path        TEXT PRIMARY KEY,
    sha         TEXT NOT NULL,
    paper_id    TEXT,
    status      TEXT NOT NULL,          -- ok | failed
    error       TEXT,
    n_chunks    INTEGER,
    parse_secs  REAL,
    ingested_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS ingest_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL, n_new INTEGER, n_skipped INTEGER, n_failed INTEGER,
    n_removed INTEGER, total_secs REAL, config_hash TEXT
);
CREATE TABLE IF NOT EXISTS queries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL, source TEXT,                  -- cli | ui | eval
    question TEXT, rewritten TEXT, paper_filter TEXT,
    retrieval_ms REAL, rerank_ms REAL, rewrite_ms REAL, ttft_ms REAL, total_ms REAL,
    gen_tokens INTEGER, tokens_per_sec REAL, n_chunks INTEGER,
    answer TEXT, sources TEXT, cited TEXT, config_hash TEXT
);
CREATE TABLE IF NOT EXISTS eval_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL, label TEXT, config_hash TEXT, config TEXT,
    n_questions INTEGER, hit_rate REAL, mrr REAL,
    faithfulness REAL, correctness REAL, refusal_accuracy REAL,
    avg_total_ms REAL
);
CREATE TABLE IF NOT EXISTS eval_results (
    run_id INTEGER, qid TEXT, question TEXT, answerable INTEGER,
    hit INTEGER, rank INTEGER, faithfulness REAL, correctness REAL,
    refused INTEGER, answer TEXT, judge_notes TEXT, total_ms REAL
);
"""


class DB:
    def __init__(self, path: Path):
        self.path = path
        with self.conn() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def conn(self):
        c = sqlite3.connect(self.path)
        c.row_factory = sqlite3.Row
        try:
            yield c
            c.commit()
        finally:
            c.close()

    # ---- manifest ------------------------------------------------------
    def manifest(self) -> dict[str, sqlite3.Row]:
        with self.conn() as c:
            return {r["path"]: r for r in c.execute("SELECT * FROM manifest")}

    def upsert_manifest(self, **row) -> None:
        row.setdefault("ingested_at", time.time())
        cols = ", ".join(row)
        qs = ", ".join("?" for _ in row)
        with self.conn() as c:
            c.execute(f"INSERT OR REPLACE INTO manifest ({cols}) VALUES ({qs})", list(row.values()))

    def delete_manifest(self, path: str) -> None:
        with self.conn() as c:
            c.execute("DELETE FROM manifest WHERE path = ?", (path,))

    # ---- generic insert -----------------------------------------------
    def insert(self, table: str, **row) -> int:
        row = {k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in row.items()}
        cols = ", ".join(row)
        qs = ", ".join("?" for _ in row)
        with self.conn() as c:
            cur = c.execute(f"INSERT INTO {table} ({cols}) VALUES ({qs})", list(row.values()))
            return cur.lastrowid

    def query(self, sql: str, params: tuple = ()) -> list[dict]:
        with self.conn() as c:
            return [dict(r) for r in c.execute(sql, params)]
