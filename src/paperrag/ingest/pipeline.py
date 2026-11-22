"""Incremental ingestion: hash -> parse (cached) -> chunk -> embed -> store."""

from __future__ import annotations

import hashlib
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from .. import llm
from ..config import Config
from ..db import DB
from ..store import CHUNKS, PAPERS, Store
from .chunk import chunk_blocks
from .parse import PaperMeta, convert, extract_meta, to_blocks


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


@dataclass
class IngestReport:
    new: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    secs: float = 0.0


def _embed_header(meta: PaperMeta, section: str) -> str:
    # Contextual header: prepended for embedding/BM25 so chunks "know" their paper/section.
    return f"{meta.title}\n{section}".strip()


def ingest_one(cfg: Config, store: Store, pdf: Path, sha: str) -> int:
    paper_id = sha[:12]
    t0 = time.perf_counter()
    doc = convert(pdf, cfg.root / "parsed", sha, cfg.ingest.formula_enrichment)
    parse_secs = time.perf_counter() - t0
    blocks = to_blocks(doc)
    meta = extract_meta(blocks, pdf)
    chunks = chunk_blocks(blocks, cfg.ingest.chunk_target_tokens,
                          cfg.ingest.chunk_max_tokens, cfg.ingest.chunk_overlap_tokens)
    if not chunks:
        raise ValueError("no text extracted (scanned PDF? OCR is disabled)")

    embed_texts = [f"{_embed_header(meta, c.section)}\n\n{c.text}" for c in chunks]
    vecs = llm.embed(cfg, embed_texts)
    paper_text = f"{meta.title}\n\n{meta.abstract}"
    paper_vec = llm.embed(cfg, [paper_text])[0]

    store.delete_paper(paper_id)  # idempotent re-ingest
    store.add(CHUNKS, [
        {
            "chunk_id": f"{paper_id}:{c.index:04d}", "paper_id": paper_id,
            "paper_title": meta.title, "kind": c.kind, "section": c.section,
            "page_start": c.page_start, "page_end": c.page_end,
            "token_count": c.token_count, "text": c.text, "embed_text": et, "vector": v,
        }
        for c, et, v in zip(chunks, embed_texts, vecs)
    ])
    store.add(PAPERS, [{
        "paper_id": paper_id, "title": meta.title, "authors": ", ".join(meta.authors),
        "year": meta.year or 0, "abstract": meta.abstract, "filename": pdf.name,
        "n_chunks": len(chunks), "embed_text": paper_text, "vector": paper_vec,
    }])
    return len(chunks), parse_secs


def ingest_folder(cfg: Config, folder: Path | None = None, force: bool = False,
                  progress: Callable[[str], None] = print) -> IngestReport:
    folder = folder or cfg.papers_dir
    store, db = Store(cfg), DB(cfg.db_path)
    manifest = db.manifest()
    report = IngestReport()
    t0 = time.perf_counter()

    pdfs = sorted(folder.glob("*.pdf"))
    present = {str(p.resolve()) for p in pdfs}

    # Files that disappeared from the folder -> drop from index.
    for path, row in manifest.items():
        if Path(path).parent == folder.resolve() and path not in present:
            if row["paper_id"]:
                store.delete_paper(row["paper_id"])
            db.delete_manifest(path)
            report.removed.append(Path(path).name)
            progress(f"removed  {Path(path).name}")

    for i, pdf in enumerate(pdfs, 1):
        key = str(pdf.resolve())
        sha = file_sha(pdf)
        prev = manifest.get(key)
        if not force and prev and prev["sha"] == sha and prev["status"] == "ok":
            report.skipped.append(pdf.name)
            continue
        if prev and prev["paper_id"] and prev["sha"] != sha:
            store.delete_paper(prev["paper_id"])  # file changed: drop old version
        progress(f"[{i}/{len(pdfs)}] parsing {pdf.name} ...")
        try:
            n, parse_secs = ingest_one(cfg, store, pdf, sha)
            db.upsert_manifest(path=key, sha=sha, paper_id=sha[:12], status="ok",
                               error=None, n_chunks=n, parse_secs=parse_secs)
            report.new.append(pdf.name)
            progress(f"         ok: {n} chunks ({parse_secs:.0f}s parse)")
        except Exception as e:  # keep going; record the failure for the dashboard
            db.upsert_manifest(path=key, sha=sha, paper_id=None, status="failed",
                               error=f"{type(e).__name__}: {e}", n_chunks=0, parse_secs=None)
            report.failed.append((pdf.name, str(e)))
            progress(f"         FAILED: {e}")
            traceback.print_exc()

    if report.new or report.removed:
        store.rebuild_fts()
    report.secs = time.perf_counter() - t0
    db.insert("ingest_runs", ts=time.time(), n_new=len(report.new),
              n_skipped=len(report.skipped), n_failed=len(report.failed),
              n_removed=len(report.removed), total_secs=report.secs, config_hash=cfg.hash())
    return report
