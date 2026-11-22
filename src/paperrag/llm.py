"""Thin wrappers around Ollama (embeddings + chat) and the cross-encoder reranker."""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass
from functools import lru_cache

import ollama

from .config import Config


@lru_cache(maxsize=4)
def _client(host: str) -> ollama.Client:
    return ollama.Client(host=host)


def client(cfg: Config) -> ollama.Client:
    return _client(cfg.models.ollama_host)


# ---- embeddings --------------------------------------------------------
def embed(cfg: Config, texts: list[str]) -> list[list[float]]:
    out: list[list[float]] = []
    bs = cfg.ingest.embed_batch_size
    for i in range(0, len(texts), bs):
        resp = client(cfg).embed(model=cfg.models.embed, input=texts[i : i + bs], truncate=True)
        out.extend(resp.embeddings)
    return out


# ---- chat ----------------------------------------------------------------
@dataclass
class GenStats:
    ttft_ms: float = 0.0
    total_ms: float = 0.0
    gen_tokens: int = 0
    tokens_per_sec: float = 0.0


def _chat(cfg: Config, model: str, messages: list[dict], **kw):
    """Call chat, retrying without `think` for models that don't support it."""
    think = cfg.models.think
    try:
        return client(cfg).chat(model=model, messages=messages, think=think, **kw)
    except ollama.ResponseError as e:
        if think is not None and "think" in str(e).lower():
            return client(cfg).chat(model=model, messages=messages, **kw)
        raise


def chat_once(cfg: Config, messages: list[dict], model: str | None = None,
              temperature: float = 0.0, format=None) -> str:
    resp = _chat(cfg, model or cfg.models.chat, messages, stream=False, format=format,
                 options={"temperature": temperature, "num_ctx": cfg.generation.num_ctx})
    return resp.message.content


def chat_stream(cfg: Config, messages: list[dict], stats: GenStats) -> Iterator[str]:
    """Yield content tokens; fill `stats` as a side effect."""
    t0 = time.perf_counter()
    stream = _chat(cfg, cfg.models.chat, messages, stream=True,
                   options={"temperature": cfg.generation.temperature,
                            "num_ctx": cfg.generation.num_ctx})
    for part in stream:
        tok = part.message.content or ""
        if tok and not stats.ttft_ms:
            stats.ttft_ms = (time.perf_counter() - t0) * 1000
        if part.done:
            stats.gen_tokens = part.eval_count or 0
            if part.eval_duration:
                stats.tokens_per_sec = stats.gen_tokens / (part.eval_duration / 1e9)
        if tok:
            yield tok
    stats.total_ms = (time.perf_counter() - t0) * 1000


# ---- reranker ------------------------------------------------------------
@lru_cache(maxsize=1)
def _cross_encoder(name: str):
    import torch
    from sentence_transformers import CrossEncoder

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    kwargs = {"torch_dtype": torch.float16} if device == "mps" else {}
    return CrossEncoder(name, device=device, max_length=1024, model_kwargs=kwargs)


def warmup(cfg: Config) -> None:
    """Load the reranker up front so the first query's timings aren't skewed."""
    if cfg.retrieval.use_reranker:
        rerank(cfg, "warmup", ["warmup"])


def rerank(cfg: Config, query: str, docs: list[str]) -> list[float]:
    if not docs:
        return []
    model = _cross_encoder(cfg.models.reranker)
    return [float(s) for s in model.predict([(query, d) for d in docs], batch_size=16)]
