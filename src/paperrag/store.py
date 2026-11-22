"""LanceDB-backed storage: a `chunks` table and a `papers` table (one row per paper).

Both tables have a vector column (dense search) and a full-text index (BM25).
Fusion of the two result lists happens in retrieval.py, not here.
"""

from __future__ import annotations

import lancedb

from .config import Config

CHUNKS = "chunks"
PAPERS = "papers"


def _quote(s: str) -> str:
    return "'" + s.replace("'", "''") + "'"


def paper_filter(paper_ids: list[str] | None) -> str | None:
    if not paper_ids:
        return None
    return f"paper_id IN ({', '.join(_quote(p) for p in paper_ids)})"


class Store:
    def __init__(self, cfg: Config):
        self.db = lancedb.connect(str(cfg.index_dir))

    # ---- tables --------------------------------------------------------
    def _table(self, name: str):
        return self.db.open_table(name) if name in self.db.list_tables().tables else None

    @property
    def chunks(self):
        return self._table(CHUNKS)

    @property
    def papers(self):
        return self._table(PAPERS)

    # ---- writes --------------------------------------------------------
    def add(self, name: str, rows: list[dict]) -> None:
        if not rows:
            return
        t = self._table(name)
        if t is None:
            self.db.create_table(name, data=rows)
        else:
            t.add(rows)

    def delete_paper(self, paper_id: str) -> None:
        for t in (self.chunks, self.papers):
            if t is not None:
                t.delete(f"paper_id = {_quote(paper_id)}")

    def rebuild_fts(self) -> None:
        """(Re)build BM25 indexes. Cheap at a few-hundred-paper scale."""
        if self.chunks is not None and self.chunks.count_rows():
            self.chunks.create_fts_index("embed_text", replace=True)
        if self.papers is not None and self.papers.count_rows():
            self.papers.create_fts_index("embed_text", replace=True)

    # ---- reads ---------------------------------------------------------
    def vector_search(self, name: str, vec: list[float], limit: int, where: str | None = None):
        t = self._table(name)
        if t is None:
            return []
        q = t.search(vec, query_type="vector").limit(limit)
        if where:
            q = q.where(where, prefilter=True)
        return q.to_list()

    def text_search(self, name: str, text: str, limit: int, where: str | None = None):
        t = self._table(name)
        if t is None or not text.strip():
            return []
        q = t.search(text, query_type="fts").limit(limit)
        if where:
            q = q.where(where, prefilter=True)
        try:
            return q.to_list()
        except Exception:
            # FTS index missing or query syntax tripped the parser: degrade to vector-only.
            return []

    def list_papers(self) -> list[dict]:
        t = self.papers
        if t is None:
            return []
        cols = ["paper_id", "title", "authors", "year", "filename", "n_chunks"]
        return sorted(t.search().select(cols).limit(100_000).to_list(), key=lambda r: r["title"])

    def chunk_count(self) -> int:
        return self.chunks.count_rows() if self.chunks is not None else 0
