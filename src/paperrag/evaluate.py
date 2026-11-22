"""Eval harness: retrieval hit-rate/MRR (deterministic) + LLM-judge scores.

Evidence is given as (paper filename, verbatim quote) rather than chunk ids,
so the same eval set stays valid when chunk size/overlap change.
"""

from __future__ import annotations

import re
import time
from pathlib import Path

import yaml

from . import llm
from .config import Config
from .db import DB
from .generate import answer, is_refusal
from .store import Store

JUDGE_PROMPT = """You are grading an answer produced by a retrieval-augmented system.

Question: {question}

Reference answer: {reference}

Sources the system was given:
{sources}

System answer:
{answer}

Score two things from 1 (worst) to 5 (best):
- faithfulness: are ALL claims in the system answer supported by the given sources?
  (Ignore whether they are correct; only whether the sources back them up.)
- correctness: does the system answer agree with the reference answer on the key facts?

Respond in JSON."""

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "faithfulness": {"type": "integer", "minimum": 1, "maximum": 5},
        "correctness": {"type": "integer", "minimum": 1, "maximum": 5},
        "notes": {"type": "string"},
    },
    "required": ["faithfulness", "correctness", "notes"],
}


def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def evidence_rank(chunks: list[dict], evidence: list[dict], paper_ids: dict[str, str]) -> int | None:
    """1-based rank of the first retrieved chunk that contains any evidence quote."""
    wanted = [(paper_ids.get(e["paper"]), normalize(e["quote"])) for e in evidence]
    for rank, c in enumerate(chunks, start=1):
        text = normalize(c["text"])
        if any(pid == c["paper_id"] and q in text for pid, q in wanted if pid):
            return rank
    return None


def judge(cfg: Config, question: str, reference: str, ans_text: str, sources: list[dict]) -> dict:
    import json

    src = "\n\n".join(f"[{i}] {c['text'][:1500]}" for i, c in enumerate(sources, 1))
    raw = llm.chat_once(cfg, [{"role": "user", "content": JUDGE_PROMPT.format(
        question=question, reference=reference, sources=src, answer=ans_text)}],
        model=cfg.models.judge, format=JUDGE_SCHEMA)
    try:
        d = json.loads(raw)
        return {"faithfulness": (int(d["faithfulness"]) - 1) / 4,
                "correctness": (int(d["correctness"]) - 1) / 4, "notes": d.get("notes", "")}
    except (ValueError, KeyError, TypeError):
        return {"faithfulness": None, "correctness": None, "notes": f"unparseable: {raw[:200]}"}


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def run_eval(cfg: Config, questions_path: Path, label: str = "", use_judge: bool = True,
             progress=print) -> dict:
    items = yaml.safe_load(Path(questions_path).read_text())["questions"]
    store, db = Store(cfg), DB(cfg.db_path)
    llm.warmup(cfg)
    paper_ids = {p["filename"]: p["paper_id"] for p in store.list_papers()}
    missing = {e["paper"] for q in items for e in q.get("evidence", [])} - paper_ids.keys()
    if missing:
        progress(f"warning: evidence papers not indexed: {sorted(missing)}")

    run_id = db.insert("eval_runs", ts=time.time(), label=label or cfg.hash(),
                       config_hash=cfg.hash(), config=cfg.snapshot(), n_questions=len(items))
    rows = []
    for i, q in enumerate(items, 1):
        answerable = q.get("answerable", True)
        a = answer(cfg, store, q["question"], db=db, source="eval")
        refused = is_refusal(a.text)
        rank = evidence_rank(a.sources, q.get("evidence", []), paper_ids) if answerable else None
        j = (judge(cfg, q["question"], q.get("answer", ""), a.text, a.sources)
             if use_judge and answerable else {"faithfulness": None, "correctness": None, "notes": ""})
        row = dict(run_id=run_id, qid=q["id"], question=q["question"], answerable=int(answerable),
                   hit=int(rank is not None) if answerable else None, rank=rank,
                   faithfulness=j["faithfulness"], correctness=j["correctness"],
                   refused=int(refused), answer=a.text, judge_notes=j["notes"],
                   total_ms=a.timings["total_ms"])
        db.insert("eval_results", **row)
        rows.append(row)
        progress(f"[{i}/{len(items)}] {q['id']}: hit={row['hit']} rank={rank} "
                 f"faith={j['faithfulness']} corr={j['correctness']} refused={refused}")

    ans_rows = [r for r in rows if r["answerable"]]
    summary = {
        "run_id": run_id,
        "hit_rate": _mean([r["hit"] for r in ans_rows]),
        "mrr": _mean([1 / r["rank"] if r["rank"] else 0.0 for r in ans_rows]),
        "faithfulness": _mean([r["faithfulness"] for r in ans_rows]),
        "correctness": _mean([r["correctness"] for r in ans_rows]),
        # right call = answered when answerable, refused when not
        "refusal_accuracy": _mean([int(bool(r["refused"]) != bool(r["answerable"])) for r in rows]),
        "avg_total_ms": _mean([r["total_ms"] for r in rows]),
    }
    with db.conn() as c:
        c.execute("""UPDATE eval_runs SET hit_rate=?, mrr=?, faithfulness=?, correctness=?,
                     refusal_accuracy=?, avg_total_ms=? WHERE id=?""",
                  (summary["hit_rate"], summary["mrr"], summary["faithfulness"],
                   summary["correctness"], summary["refusal_accuracy"],
                   summary["avg_total_ms"], run_id))
    return summary
