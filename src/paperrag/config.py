"""Typed configuration loaded from config.yaml."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "config.yaml"


class Models(BaseModel):
    ollama_host: str = "http://localhost:11434"
    chat: str = "qwen3:14b"
    embed: str = "bge-m3"
    judge: str = "qwen3:14b"
    reranker: str = "BAAI/bge-reranker-v2-m3"
    think: bool | None = False


class Ingest(BaseModel):
    chunk_target_tokens: int = 650
    chunk_max_tokens: int = 800
    chunk_overlap_tokens: int = 80
    formula_enrichment: bool = True
    embed_batch_size: int = 32


class Retrieval(BaseModel):
    hybrid: bool = True
    candidates: int = 30
    top_k: int = 6
    rrf_k: int = 60
    use_reranker: bool = True


class Generation(BaseModel):
    temperature: float = 0.1
    num_ctx: int = 8192
    prompt_version: str = "v1"


class Config(BaseModel):
    data_dir: Path = Path("data")
    models: Models = Models()
    ingest: Ingest = Ingest()
    retrieval: Retrieval = Retrieval()
    generation: Generation = Generation()

    # ---- derived paths -------------------------------------------------
    @property
    def root(self) -> Path:
        return self.data_dir if self.data_dir.is_absolute() else ROOT / self.data_dir

    @property
    def papers_dir(self) -> Path:
        return self.root / "papers"

    @property
    def index_dir(self) -> Path:
        return self.root / "index"

    @property
    def db_path(self) -> Path:
        return self.root / "paperrag.db"

    def ensure_dirs(self) -> None:
        for p in (self.papers_dir, self.index_dir):
            p.mkdir(parents=True, exist_ok=True)

    # ---- experiment tracking ------------------------------------------
    def snapshot(self) -> dict:
        return json.loads(self.model_dump_json())

    def hash(self) -> str:
        blob = json.dumps(self.snapshot(), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:10]


def load_config(path: Path | str | None = None) -> Config:
    path = Path(path) if path else DEFAULT_CONFIG
    data = yaml.safe_load(path.read_text()) if path.exists() else {}
    cfg = Config.model_validate(data or {})
    cfg.ensure_dirs()
    return cfg
