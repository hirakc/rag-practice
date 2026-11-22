"""Command-line interface:  uv run paperrag --help"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import typer

from .config import load_config

app = typer.Typer(add_completion=False, help="Local RAG over your research papers.")
ConfigOpt = Annotated[Optional[Path], typer.Option("--config", "-c", help="Path to config.yaml")]


@app.command()
def ingest(folder: Annotated[Optional[Path], typer.Argument()] = None,
           force: Annotated[bool, typer.Option(help="Re-chunk/re-embed everything (uses parse cache)")] = False,
           config: ConfigOpt = None):
    """Ingest PDFs from data/papers (or FOLDER). Incremental by file hash."""
    from .ingest.pipeline import ingest_folder

    r = ingest_folder(load_config(config), folder, force=force)
    typer.echo(f"\nnew {len(r.new)} | skipped {len(r.skipped)} | failed {len(r.failed)} "
               f"| removed {len(r.removed)} | {r.secs:.0f}s")


@app.command()
def ask(question: str,
        paper: Annotated[Optional[list[str]], typer.Option("--paper", "-p", help="Restrict to paper id(s)")] = None,
        show_sources: Annotated[bool, typer.Option("--sources/--no-sources")] = True,
        config: ConfigOpt = None):
    """Ask a question; answer streams with [n] citations."""
    from .db import DB
    from .generate import answer_stream, source_label
    from .store import Store

    cfg = load_config(config)
    final = None
    for tok, final in answer_stream(cfg, Store(cfg), question, paper_ids=paper,
                                    db=DB(cfg.db_path), source="cli"):
        typer.echo(tok, nl=False)
    typer.echo("\n")
    if show_sources and final.sources:
        for i, c in enumerate(final.sources, 1):
            mark = "*" if i in final.cited else " "
            typer.echo(f" {mark}[{i}] {source_label(c)}")
    t = final.timings
    typer.echo(f"\nretrieval {t['retrieval_ms']:.0f}ms · rerank {t['rerank_ms']:.0f}ms · "
               f"first token {t['ttft_ms']:.0f}ms · total {t['total_ms'] / 1000:.1f}s · "
               f"{final.stats.tokens_per_sec:.1f} tok/s")


@app.command()
def find(query: str, limit: int = 10, config: ConfigOpt = None):
    """Find which of your papers discuss a topic."""
    from .retrieval import find_papers
    from .store import Store

    cfg = load_config(config)
    for i, p in enumerate(find_papers(cfg, Store(cfg), query, limit), 1):
        typer.echo(f"{i:2}. [{p['paper_id']}] {p['title']} ({p['year'] or '?'})")
        if ev := p.get("evidence"):
            typer.echo(f"      §{ev['section']}, p. {ev['page']}: {ev['text'][:160]}...")


@app.command()
def papers(config: ConfigOpt = None):
    """List indexed papers and their ids."""
    from .store import Store

    for p in Store(load_config(config)).list_papers():
        typer.echo(f"[{p['paper_id']}] {p['title'][:80]} ({p['year'] or '?'}) - {p['n_chunks']} chunks")


@app.command(name="eval")
def eval_cmd(label: Annotated[str, typer.Option(help="Name for this run on the dashboard")] = "",
             questions: Path = Path("eval/questions.yaml"),
             judge: Annotated[bool, typer.Option("--judge/--no-judge")] = True,
             config: ConfigOpt = None):
    """Run the eval set and log metrics for the dashboard."""
    from .evaluate import run_eval

    s = run_eval(load_config(config), questions, label=label, use_judge=judge)
    typer.echo("\n" + "  ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                                for k, v in s.items()))


@app.command()
def ui(port: int = 7860, config: ConfigOpt = None):
    """Launch the Gradio app (chat, upload, metrics)."""
    from .ui import build_app

    build_app(load_config(config)).launch(server_port=port)


if __name__ == "__main__":
    app()
