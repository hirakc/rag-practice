"""Gradio app: Chat · Papers (upload + ingest) · Metrics dashboard."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import gradio as gr
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from . import llm
from .config import Config
from .db import DB
from .generate import answer_stream, source_label
from .retrieval import find_papers
from .store import Store


# ---- helpers ---------------------------------------------------------------
def _text(content) -> str:
    """Gradio may hand back message content as a string or a list of parts."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return str(content or "")


def _paper_choices(store: Store) -> list[tuple[str, str]]:
    return [(f"{p['title'][:70]} ({p['year'] or '?'})", p["paper_id"]) for p in store.list_papers()]


def _sources_md(sources: list[dict], cited: list[int]) -> str:
    if not sources:
        return ""
    out = ["### Sources  \n<small>★ = cited in the answer</small>"]
    for i, c in enumerate(sources, 1):
        star = "★ " if i in cited else ""
        score = f" · rerank {c['rerank_score']:.2f}" if "rerank_score" in c else ""
        out.append(f"<details><summary>{star}<b>[{i}]</b> {source_label(c)}"
                   f"{' (table)' if c['kind'] == 'table' else ''}{score}</summary>\n\n"
                   f"{c['text']}\n\n</details>")
    return "\n\n".join(out)


def _empty_fig(msg: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=msg, showarrow=False, x=0.5, y=0.5, xref="paper", yref="paper")
    fig.update_layout(xaxis_visible=False, yaxis_visible=False, height=300)
    return fig


# ---- metrics -----------------------------------------------------------------
def _metrics(cfg: Config, store: Store, db: DB):
    # Corpus
    man = pd.DataFrame(db.query("SELECT * FROM manifest"))
    n_ok = int((man.status == "ok").sum()) if len(man) else 0
    n_fail = int((man.status == "failed").sum()) if len(man) else 0
    chunks = store.chunks
    tok = (chunks.search().select(["token_count", "kind"]).limit(1_000_000).to_pandas()
           if chunks is not None else pd.DataFrame(columns=["token_count", "kind"]))
    corpus_md = (f"**Papers:** {n_ok} indexed · {n_fail} failed  \n"
                 f"**Chunks:** {len(tok)} ({int((tok.kind == 'table').sum()) if len(tok) else 0} tables) · "
                 f"avg {tok.token_count.mean() if len(tok) else 0:.0f} tokens  \n"
                 f"**Avg parse time:** {man.parse_secs.mean() if len(man) and man.parse_secs.notna().any() else 0:.1f}s/paper")
    failed = (man[man.status == "failed"][["path", "error"]].assign(path=lambda d: d.path.map(lambda p: Path(p).name))
              if n_fail else pd.DataFrame(columns=["path", "error"]))
    runs_ing = pd.DataFrame(db.query(
        "SELECT datetime(ts,'unixepoch','localtime') AS time, n_new, n_skipped, n_failed, n_removed, "
        "round(total_secs,1) AS secs FROM ingest_runs ORDER BY ts DESC LIMIT 20"))

    # Eval quality
    ev = pd.DataFrame(db.query("SELECT * FROM eval_runs WHERE hit_rate IS NOT NULL ORDER BY ts"))
    if len(ev):
        ev["run"] = ev.id.astype(str) + ": " + ev.label
        metrics = ["hit_rate", "mrr", "faithfulness", "correctness", "refusal_accuracy"]
        long = ev.melt(id_vars="run", value_vars=metrics, var_name="metric", value_name="score")
        quality_fig = px.bar(long, x="metric", y="score", color="run", barmode="group",
                             range_y=[0, 1], title="Eval runs: quality")

        def cfg_cols(c):
            c = json.loads(c)
            return pd.Series({"chat": c["models"]["chat"], "hybrid": c["retrieval"]["hybrid"],
                              "rerank": c["retrieval"]["use_reranker"], "top_k": c["retrieval"]["top_k"],
                              "chunk": c["ingest"]["chunk_target_tokens"],
                              "prompt": c["generation"]["prompt_version"]})
        ev_table = pd.concat([ev[["run"]], ev.config.apply(cfg_cols),
                              ev[metrics + ["avg_total_ms"]].round(3)], axis=1).iloc[::-1]
    else:
        quality_fig = _empty_fig("No eval runs yet — run `uv run paperrag eval --label baseline`")
        ev_table = pd.DataFrame()

    # Performance
    q = pd.DataFrame(db.query("SELECT * FROM queries ORDER BY ts"))
    if len(q):
        q["time"] = pd.to_datetime(q.ts, unit="s")
        q["generation_ms"] = (q.total_ms - q.rewrite_ms - q.retrieval_ms - q.rerank_ms).clip(lower=0)
        stages = ["rewrite_ms", "retrieval_ms", "rerank_ms", "generation_ms"]
        by_cfg = q.groupby("config_hash")[stages].mean().reset_index().melt(
            id_vars="config_hash", var_name="stage", value_name="ms")
        latency_fig = px.bar(by_cfg, x="config_hash", y="ms", color="stage",
                             title="Avg latency breakdown per config")
        trend_fig = px.scatter(q, x="time", y="total_ms", color="source",
                               hover_data=["question"], title="Total latency per query")
        tps_fig = px.line(q[q.tokens_per_sec > 0], x="time", y="tokens_per_sec",
                          markers=True, title="Generation speed (tokens/s)")
        perf_md = (f"**Queries logged:** {len(q)} · median total {q.total_ms.median() / 1000:.1f}s · "
                   f"median first-token {q.ttft_ms.median():.0f}ms · "
                   f"median {q.tokens_per_sec.median():.1f} tok/s")
    else:
        latency_fig = trend_fig = tps_fig = _empty_fig("No queries yet")
        perf_md = "No queries yet."

    return (corpus_md, failed, runs_ing, quality_fig, ev_table,
            perf_md, latency_fig, trend_fig, tps_fig)


# ---- app -------------------------------------------------------------------
def build_app(cfg: Config) -> gr.Blocks:
    store, db = Store(cfg), DB(cfg.db_path)
    llm.warmup(cfg)

    def chat(message, history, mode, paper_ids):
        history = history or []
        history = history + [{"role": "user", "content": message}]
        if mode == "Find papers":
            hits = find_papers(cfg, store, message)
            lines = [f"**{i}. {p['title']}** ({p['year'] or '?'}) — `{p['paper_id']}`"
                     + (f"\n   > §{p['evidence']['section']}, p. {p['evidence']['page']}: "
                        f"{p['evidence']['text'][:200]}…" if p.get("evidence") else "")
                     for i, p in enumerate(hits, 1)]
            history.append({"role": "assistant", "content": "\n\n".join(lines) or "No papers found."})
            yield history, "", ""
            return

        prior = [{"role": m["role"], "content": _text(m["content"])} for m in history[:-1]]
        history.append({"role": "assistant", "content": ""})
        final = None
        for tok, final in answer_stream(cfg, store, message, history=prior,
                                        paper_ids=paper_ids or None, db=db, source="ui"):
            history[-1]["content"] = final.text
            yield history, "", ""
        t = final.timings
        rewritten = (f"🔎 searched for: *{final.rewritten}*  \n"
                     if final.rewritten.strip() != message.strip() else "")
        info = (f"{rewritten}<small>retrieval {t['retrieval_ms']:.0f}ms · rerank {t['rerank_ms']:.0f}ms · "
                f"first token {t['ttft_ms']:.0f}ms · total {t['total_ms'] / 1000:.1f}s · "
                f"{final.stats.tokens_per_sec:.1f} tok/s</small>")
        yield history, _sources_md(final.sources, final.cited), info

    def upload_and_ingest(files):
        from .ingest.pipeline import ingest_folder

        log: list[str] = []
        for f in files or []:
            src = Path(f if isinstance(f, str) else f.name)
            shutil.copy(src, cfg.papers_dir / src.name)
            log.append(f"saved {src.name}")
        yield "\n".join(log + ["ingesting…"]), gr.update()
        report = ingest_folder(cfg, progress=log.append)
        log.append(f"\nnew {len(report.new)} · skipped {len(report.skipped)} · "
                   f"failed {len(report.failed)} · removed {len(report.removed)} · {report.secs:.0f}s")
        yield "\n".join(log), gr.update(choices=_paper_choices(store))

    def papers_table():
        rows = store.list_papers()
        return pd.DataFrame(rows)[["paper_id", "title", "authors", "year", "n_chunks", "filename"]] \
            if rows else pd.DataFrame()

    with gr.Blocks(title="PaperRAG", fill_height=True) as app:
        gr.Markdown("# 📚 PaperRAG — local Q&A over your research papers")
        with gr.Tab("Chat"):
            with gr.Row():
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(height=520, label="Conversation", buttons=["copy"],
                                         latex_delimiters=[{"left": "$$", "right": "$$", "display": True},
                                                           {"left": "$", "right": "$", "display": False}])
                    info = gr.Markdown()
                    msg = gr.Textbox(placeholder="Ask about your papers…", show_label=False, submit_btn=True)
                    with gr.Row():
                        mode = gr.Radio(["Ask", "Find papers"], value="Ask", label="Mode")
                        clear = gr.Button("New conversation", variant="secondary")
                with gr.Column(scale=2):
                    paper_sel = gr.Dropdown(choices=_paper_choices(store), multiselect=True,
                                            label="Restrict to papers (empty = all)")
                    sources = gr.Markdown()
            msg.submit(chat, [msg, chatbot, mode, paper_sel], [chatbot, sources, info]) \
               .then(lambda: "", None, msg)
            clear.click(lambda: ([], "", ""), None, [chatbot, sources, info])

        with gr.Tab("Papers"):
            with gr.Row():
                with gr.Column(scale=1):
                    files = gr.File(file_count="multiple", file_types=[".pdf"], label="Upload PDFs")
                    ingest_btn = gr.Button("Upload & ingest", variant="primary")
                    ingest_log = gr.Textbox(label="Ingestion log", lines=14)
                with gr.Column(scale=2):
                    table = gr.Dataframe(papers_table, label="Indexed papers", wrap=True)
            ingest_btn.click(upload_and_ingest, files, [ingest_log, paper_sel]) \
                      .then(papers_table, None, table)

        with gr.Tab("Metrics") as metrics_tab:
            refresh = gr.Button("Refresh", size="sm")
            gr.Markdown("## Answer quality (eval set)")
            quality_fig = gr.Plot()
            ev_table = gr.Dataframe(label="Eval runs (newest first)", wrap=True)
            gr.Markdown("## Performance")
            perf_md = gr.Markdown()
            with gr.Row():
                latency_fig, trend_fig = gr.Plot(), gr.Plot()
            tps_fig = gr.Plot()
            gr.Markdown("## Corpus")
            corpus_md = gr.Markdown()
            with gr.Row():
                ingest_runs = gr.Dataframe(label="Ingestion runs")
                failed = gr.Dataframe(label="Failed files")
            outs = [corpus_md, failed, ingest_runs, quality_fig, ev_table,
                    perf_md, latency_fig, trend_fig, tps_fig]
            refresh.click(lambda: _metrics(cfg, store, db), None, outs)
            metrics_tab.select(lambda: _metrics(cfg, store, db), None, outs)

    return app
