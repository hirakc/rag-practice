"""Download a starter corpus of open-access arXiv papers on RAG / retrieval.

Writes <papers_dir>/<arxiv_id>.pdf plus a sidecar <arxiv_id>.meta.json with
title/authors/year/abstract, which ingestion prefers over heuristics.

    uv run python scripts/fetch_arxiv.py            # default list
    uv run python scripts/fetch_arxiv.py 2005.11401 # specific ids
"""

from __future__ import annotations

import json
import sys
import time

import arxiv
import requests

from paperrag.config import load_config

STARTER_IDS = [
    "2005.11401",  # RAG (Lewis et al.)
    "2004.04906",  # Dense Passage Retrieval
    "2002.08909",  # REALM
    "2007.01282",  # Fusion-in-Decoder
    "2112.04426",  # RETRO
    "2208.03299",  # Atlas
    "2301.12652",  # REPLUG
    "2310.11511",  # Self-RAG
    "2401.15884",  # Corrective RAG
    "2212.10496",  # HyDE
    "2307.03172",  # Lost in the Middle
    "2004.12832",  # ColBERT
    "2104.08663",  # BEIR
    "2112.09118",  # Contriever
    "2212.03533",  # E5
    "2402.03216",  # BGE M3-Embedding
    "2309.15217",  # RAGAS
    "2312.10997",  # RAG survey
    "2404.16130",  # GraphRAG
]


def main(ids: list[str]) -> None:
    cfg = load_config()
    out = cfg.papers_dir
    client = arxiv.Client(delay_seconds=3.0)
    for r in client.results(arxiv.Search(id_list=ids, max_results=len(ids))):
        short = r.get_short_id().split("v")[0]
        pdf_path = out / f"{short}.pdf"
        meta = {
            "title": " ".join(r.title.split()),
            "authors": [a.name for a in r.authors],
            "year": r.published.year,
            "abstract": " ".join(r.summary.split()),
            "source": f"arXiv:{short}",
        }
        (out / f"{short}.meta.json").write_text(json.dumps(meta, indent=2))
        if pdf_path.exists():
            print(f"skip  {short}  {meta['title'][:70]}")
            continue
        resp = requests.get(f"https://arxiv.org/pdf/{short}", timeout=60)
        resp.raise_for_status()
        pdf_path.write_bytes(resp.content)
        print(f"saved {short}  {meta['title'][:70]}")
        time.sleep(3)  # be polite to arXiv


if __name__ == "__main__":
    main(sys.argv[1:] or STARTER_IDS)
