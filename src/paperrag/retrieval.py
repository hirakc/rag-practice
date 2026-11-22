"""Hybrid retrieval: BM25 + dense vectors, fused with Reciprocal Rank Fusion,
then optionally reranked by a cross-encoder."""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field

from . import llm
from .config import Config
from .store import CHUNKS, PAPERS, Store, paper_filter


def rrf(ranked_lists: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion: score(d) = sum over lists of 1 / (k + rank).

    Ranks are 1-based. Items missing from a list contribute nothing for it.
    Ties are broken by first appearance, so the result is deterministic.
    """
    scores: dict[str, float] = defaultdict(float)
    first_seen: dict[str, int] = {}
    for lst in ranked_lists:
        for rank, key in enumerate(lst, start=1):
            scores[key] += 1.0 / (k + rank)
            first_seen.setdefault(key, len(first_seen))
    return sorted(scores.items(), key=lambda kv: (-kv[1], first_seen[kv[0]]))


@dataclass
class Retrieved:
    chunks: list[dict]
    retrieval_ms: float = 0.0
    rerank_ms: float = 0.0
    candidates: list[dict] = field(default_factory=list)  # pre-rerank, fused order


def _hybrid(cfg: Config, store: Store, table: str, query: str, key: str,
            where: str | None, limit: int) -> list[dict]:
    qvec = llm.embed(cfg, [query])[0]
    dense = store.vector_search(table, qvec, limit, where)
    lists = [[r[key] for r in dense]]
    rows = {r[key]: r for r in dense}
    if cfg.retrieval.hybrid:
        sparse = store.text_search(table, query, limit, where)
        lists.append([r[key] for r in sparse])
        for r in sparse:
            rows.setdefault(r[key], r)
    fused = rrf(lists, k=cfg.retrieval.rrf_k)
    out = []
    for k_, score in fused[:limit]:
        row = dict(rows[k_])
        row.pop("vector", None)
        row["rrf_score"] = score
        out.append(row)
    return out


def retrieve(cfg: Config, store: Store, query: str,
             paper_ids: list[str] | None = None) -> Retrieved:
    t0 = time.perf_counter()
    cands = _hybrid(cfg, store, CHUNKS, query, "chunk_id",
                    paper_filter(paper_ids), cfg.retrieval.candidates)
    t1 = time.perf_counter()
    res = Retrieved(chunks=cands[: cfg.retrieval.top_k], candidates=cands,
                    retrieval_ms=(t1 - t0) * 1000)
    if cfg.retrieval.use_reranker and cands:
        scores = llm.rerank(cfg, query, [c["embed_text"] for c in cands])
        for c, s in zip(cands, scores):
            c["rerank_score"] = s
        res.chunks = sorted(cands, key=lambda c: -c["rerank_score"])[: cfg.retrieval.top_k]
        res.rerank_ms = (time.perf_counter() - t1) * 1000
    return res


def find_papers(cfg: Config, store: Store, query: str, limit: int = 10) -> list[dict]:
    """Rank papers by fusing (a) the title+abstract index and
    (b) how highly each paper's chunks rank for the query."""
    paper_hits = _hybrid(cfg, store, PAPERS, query, "paper_id", None, limit * 2)
    chunk_hits = _hybrid(cfg, store, CHUNKS, query, "chunk_id", None, cfg.retrieval.candidates * 2)
    by_chunk: list[str] = []
    for c in chunk_hits:  # first occurrence order = best chunk rank per paper
        if c["paper_id"] not in by_chunk:
            by_chunk.append(c["paper_id"])
    fused = rrf([[p["paper_id"] for p in paper_hits], by_chunk], k=cfg.retrieval.rrf_k)
    meta = {p["paper_id"]: p for p in store.list_papers()}
    best_chunk = {}
    for c in chunk_hits:
        best_chunk.setdefault(c["paper_id"], c)
    out = []
    for pid, score in fused[:limit]:
        if pid not in meta:
            continue
        row = dict(meta[pid], score=score)
        if pid in best_chunk:
            bc = best_chunk[pid]
            row["evidence"] = {"section": bc["section"], "page": bc["page_start"],
                               "text": bc["text"][:300]}
        out.append(row)
    return out
