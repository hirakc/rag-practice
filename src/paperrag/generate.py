"""Prompt assembly, follow-up rewriting, streaming answers, and citation parsing."""

from __future__ import annotations

import re
import time
from collections.abc import Iterator
from dataclasses import dataclass, field

from . import llm
from .config import Config
from .db import DB
from .retrieval import retrieve
from .store import Store

NOT_FOUND = "I couldn't find this in your papers."

SYSTEM_PROMPTS = {
    "v1": f"""You are a research assistant answering questions about the user's research papers.

Rules:
- Use ONLY the numbered sources provided. Do not use outside knowledge.
- Cite every factual claim with its source number in square brackets, e.g. [1] or [2][3].
- If the sources do not contain the answer, reply exactly: "{NOT_FOUND}"
  You may then add one sentence on what the sources do cover.
- Be concise and precise. Preserve numbers, metric names and equations as written.
- Tables are given in Markdown; read them carefully before quoting values.""",
    # v2: closes the "answer the covered half, then use world knowledge for the rest" leak.
    "v2": f"""You are a research assistant answering questions about the user's research papers.

Rules:
- Use ONLY the numbered sources provided. Never use outside or general knowledge,
  even for facts you are sure of and even for parts of the question the sources ignore.
- Cite every factual claim with its source number in square brackets, e.g. [1] or [2][3].
  A sentence without a citation must not contain a factual claim.
- If the sources do not contain the answer at all, reply exactly: "{NOT_FOUND}"
  You may then add one sentence on what the sources do cover.
- If the sources answer only part of the question, answer that part with citations and
  say explicitly that the rest is not covered by your papers. Do not answer the rest.
- Be concise and precise. Preserve numbers, metric names and equations as written.
- Tables are given in Markdown; read them carefully before quoting values.""",
}
# v3 = v2 system prompt + the key rules repeated *after* the sources, where the model
# attends most. v2 alone still answered "capital of France" from world knowledge.
SYSTEM_PROMPTS["v3"] = SYSTEM_PROMPTS["v2"]

REMINDERS = {
    "v3": """Before answering, check each part of the question against the sources above.
- Answer only the parts the sources support, with [n] citations.
- For any part the sources do not cover, write that it is not covered by your papers.
  Do NOT answer it from general knowledge, even if you know the answer.
- If no part is covered, reply exactly: "I couldn't find this in your papers.\"""",
}

REWRITE_PROMPT = """Rewrite the user's latest message as a standalone search question,
resolving pronouns and references using the conversation. Output only the question.

Conversation:
{history}

Latest message: {question}

Standalone question:"""


# ---- citations -----------------------------------------------------------
_CITE_RE = re.compile(r"\[(\d+(?:\s*[-–,]\s*\d+)*)\]")


def parse_citations(text: str, n_sources: int) -> list[int]:
    """Return sorted, de-duplicated 1-based source numbers cited in `text`.

    Handles [1], [1, 3], [2-4] and [1][2]. Ignores numbers outside 1..n_sources.
    """
    cited: set[int] = set()
    for group in _CITE_RE.findall(text):
        for part in re.split(r"\s*,\s*", group):
            if re.search(r"[-–]", part):
                a, b = (int(x) for x in re.split(r"\s*[-–]\s*", part))
                cited.update(range(min(a, b), max(a, b) + 1))
            else:
                cited.add(int(part))
    return sorted(c for c in cited if 1 <= c <= n_sources)


def is_refusal(text: str) -> bool:
    return "couldn't find this in your papers" in text.lower().replace("’", "'")


# ---- prompt assembly -----------------------------------------------------
def source_label(c: dict) -> str:
    pages = (f"p. {c['page_start']}" if c["page_start"] == c["page_end"]
             else f"pp. {c['page_start']}-{c['page_end']}")
    sec = f", §{c['section']}" if c.get("section") else ""
    return f"{c['paper_title']}{sec}, {pages}"


def build_messages(cfg: Config, question: str, chunks: list[dict]) -> list[dict]:
    blocks = []
    for i, c in enumerate(chunks, start=1):
        kind = " (table)" if c["kind"] == "table" else ""
        blocks.append(f"[{i}] {source_label(c)}{kind}\n{c['text']}")
    user = "Sources:\n\n" + "\n\n---\n\n".join(blocks) + f"\n\nQuestion: {question}"
    if cfg.generation.prompt_version in REMINDERS:
        user += "\n\n" + REMINDERS[cfg.generation.prompt_version]
    return [
        {"role": "system", "content": SYSTEM_PROMPTS[cfg.generation.prompt_version]},
        {"role": "user", "content": user},
    ]


def rewrite_followup(cfg: Config, question: str, history: list[dict]) -> str:
    if not history:
        return question
    convo = "\n".join(f"{m['role']}: {m['content'][:600]}" for m in history[-6:])
    out = llm.chat_once(cfg, [{"role": "user", "content": REWRITE_PROMPT.format(
        history=convo, question=question)}]).strip().strip('"')
    return out or question


# ---- end-to-end answer ---------------------------------------------------
@dataclass
class Answer:
    question: str
    rewritten: str = ""
    text: str = ""
    sources: list[dict] = field(default_factory=list)
    cited: list[int] = field(default_factory=list)
    timings: dict = field(default_factory=dict)
    stats: llm.GenStats = field(default_factory=llm.GenStats)


def answer_stream(cfg: Config, store: Store, question: str,
                  history: list[dict] | None = None,
                  paper_ids: list[str] | None = None,
                  db: DB | None = None, source: str = "cli") -> Iterator[tuple[str, Answer]]:
    """Yield (token, Answer-so-far). The final Answer has sources, citations and timings."""
    t0 = time.perf_counter()
    ans = Answer(question=question)

    tr = time.perf_counter()
    ans.rewritten = rewrite_followup(cfg, question, history or [])
    rewrite_ms = (time.perf_counter() - tr) * 1000

    ret = retrieve(cfg, store, ans.rewritten, paper_ids)
    ans.sources = ret.chunks

    if not ret.chunks:
        ans.text = NOT_FOUND + " (The index is empty or no chunks matched.)"
        yield ans.text, ans
    else:
        for tok in llm.chat_stream(cfg, build_messages(cfg, ans.rewritten, ret.chunks), ans.stats):
            ans.text += tok
            yield tok, ans

    ans.cited = parse_citations(ans.text, len(ans.sources))
    ans.timings = {
        "rewrite_ms": rewrite_ms,
        "retrieval_ms": ret.retrieval_ms,
        "rerank_ms": ret.rerank_ms,
        "ttft_ms": ans.stats.ttft_ms,
        "total_ms": (time.perf_counter() - t0) * 1000,
    }
    if db is not None:
        db.insert(
            "queries", ts=time.time(), source=source, question=question,
            rewritten=ans.rewritten, paper_filter=",".join(paper_ids or []),
            retrieval_ms=ret.retrieval_ms, rerank_ms=ret.rerank_ms, rewrite_ms=rewrite_ms,
            ttft_ms=ans.stats.ttft_ms, total_ms=ans.timings["total_ms"],
            gen_tokens=ans.stats.gen_tokens, tokens_per_sec=ans.stats.tokens_per_sec,
            n_chunks=len(ans.sources), answer=ans.text,
            sources=[{"chunk_id": c["chunk_id"], "label": source_label(c)} for c in ans.sources],
            cited=ans.cited, config_hash=cfg.hash(),
        )
    yield "", ans


def answer(cfg: Config, store: Store, question: str, **kw) -> Answer:
    final = None
    for _, final in answer_stream(cfg, store, question, **kw):
        pass
    return final
