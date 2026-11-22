"""PDF -> (PaperMeta, list[Block]) using Docling.

Docling output is cached as JSON per file hash, so re-chunking with new
settings never has to re-parse a PDF.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path

from docling_core.types.doc import DoclingDocument

from .chunk import Block

_NUMBERED_RE = re.compile(r"^\s*((?:\d+|[A-Z])(?:\.\d+)*)\.?\s+\S")
_YEAR_RE = re.compile(r"\b(19[89]\d|20[0-4]\d)\b")
_ARXIV_RE = re.compile(r"arXiv:(\d{2})(\d{2})\.\d{4,5}")


@dataclass
class PaperMeta:
    title: str = ""
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    abstract: str = ""
    source: str = ""


def heading_level(text: str) -> int:
    """Infer depth from numbering: '2' -> 1, '2.1' -> 2, 'A.3' -> 2. Unnumbered -> 1."""
    m = _NUMBERED_RE.match(text)
    return m.group(1).count(".") + 1 if m else 1


@lru_cache(maxsize=2)
def _converter(formula_enrichment: bool):
    from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    opts = PdfPipelineOptions(
        do_formula_enrichment=formula_enrichment,
        do_ocr=False,  # arXiv-style PDFs have a text layer; OCR is slow
        accelerator_options=AcceleratorOptions(device=AcceleratorDevice.AUTO),
    )
    return DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})


def convert(pdf: Path, cache_dir: Path, sha: str, formula_enrichment: bool) -> DoclingDocument:
    cache = cache_dir / f"{sha}-f{int(formula_enrichment)}.json"  # parse settings in key
    if cache.exists():
        return DoclingDocument.load_from_json(cache)
    doc = _converter(formula_enrichment).convert(pdf).document
    cache_dir.mkdir(parents=True, exist_ok=True)
    doc.save_as_json(cache)
    return doc


def to_blocks(doc: DoclingDocument) -> list[Block]:
    # Table captions are already included in table markdown; don't repeat them as text.
    table_captions = {cap.cref for t in doc.tables for cap in t.captions}
    blocks: list[Block] = []
    for item, _ in doc.iterate_items():
        label = item.label.value
        page = item.prov[0].page_no if getattr(item, "prov", None) else 0
        if label in ("section_header", "title"):
            blocks.append(Block("heading", item.text, page, heading_level(item.text)))
        elif label == "table":
            md = item.export_to_markdown(doc).strip()
            if md:
                blocks.append(Block("table", md, page))
        elif label == "formula":
            tex = (item.text or "").strip()
            blocks.append(Block("formula", f"$$ {tex} $$" if tex else (item.orig or "").strip(), page))
        elif label in ("text", "paragraph", "list_item", "footnote", "code", "caption"):
            if label == "caption" and item.self_ref in table_captions:
                continue
            blocks.append(Block("list" if label == "list_item" else "text", item.text, page))
        # pictures: skipped in v1 (no vision model)
    return blocks


def extract_meta(blocks: list[Block], pdf: Path) -> PaperMeta:
    """Prefer a sidecar <name>.meta.json (written by fetch_arxiv); else heuristics."""
    sidecar = pdf.with_suffix(".meta.json")
    if sidecar.exists():
        d = json.loads(sidecar.read_text())
        return PaperMeta(**{k: d.get(k) for k in asdict(PaperMeta()) if k in d})

    meta = PaperMeta(source=pdf.name)
    headings = [i for i, b in enumerate(blocks) if b.kind == "heading"]
    first = headings[0] if headings else None
    if first is not None and blocks[first].page <= 1:
        meta.title = blocks[first].text.strip()
    abstract_idx = next((i for i in headings if blocks[i].text.strip().lower() == "abstract"), None)
    if abstract_idx is not None:
        parts = []
        for b in blocks[abstract_idx + 1:]:
            if b.kind == "heading":
                break
            parts.append(b.text)
        meta.abstract = " ".join(parts)
    else:
        txt = next((b.text for b in blocks if b.text.lower().startswith("abstract")), "")
        meta.abstract = txt[len("abstract"):].lstrip(" .:—-")
    if first is not None:
        end = abstract_idx if abstract_idx is not None else first + 4
        meta.authors = [b.text.strip() for b in blocks[first + 1:end] if b.kind == "text"][:3]
    page1 = " ".join(b.text for b in blocks if b.page <= 1)
    if m := _ARXIV_RE.search(page1):
        meta.year = 2000 + int(m.group(1))
    elif m := _YEAR_RE.search(page1):
        meta.year = int(m.group(1))
    meta.title = meta.title or pdf.stem
    return meta
