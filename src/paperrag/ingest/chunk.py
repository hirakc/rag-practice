"""Section-aware chunking.

Input: a flat, ordered list of `Block`s (what the parser produces).
Output: `Chunk`s that never cross a section boundary, carry their section
path and page range, keep tables as standalone chunks, and skip the
reference list.

This module is pure (no Docling, no I/O) so it is easy to unit-test.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

BlockKind = Literal["heading", "text", "list", "formula", "table", "caption"]

_REFERENCES_RE = re.compile(
    r"^\s*([\dIVX]+\.?\s*)?(references|bibliography|works cited|literature cited)\s*$",
    re.IGNORECASE,
)
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])")


def count_tokens(text: str) -> int:
    """Cheap token estimate (~1.3 tokens per whitespace word for English/BPE)."""
    return int(len(text.split()) * 1.3) + 1


@dataclass
class Block:
    kind: BlockKind
    text: str
    page: int
    level: int = 1  # heading depth; only meaningful for kind == "heading"


@dataclass
class Chunk:
    index: int
    kind: Literal["text", "table"]
    section: str
    text: str
    page_start: int
    page_end: int
    token_count: int = field(init=False)

    def __post_init__(self) -> None:
        self.token_count = count_tokens(self.text)


def is_references_heading(text: str) -> bool:
    return bool(_REFERENCES_RE.match(text))


def _split_oversized(text: str, max_tokens: int) -> list[str]:
    """Split one long block on sentence boundaries into pieces <= max_tokens."""
    pieces, cur = [], ""
    for sent in _SENTENCE_RE.split(text):
        candidate = f"{cur} {sent}".strip()
        if cur and count_tokens(candidate) > max_tokens:
            pieces.append(cur)
            cur = sent
        else:
            cur = candidate
    if cur:
        pieces.append(cur)
    return pieces


def chunk_blocks(
    blocks: list[Block],
    target_tokens: int = 650,
    max_tokens: int = 800,
    overlap_tokens: int = 80,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    headings: list[tuple[int, str]] = []  # stack of (level, title)
    buf: list[Block] = []
    fresh = False  # does buf contain anything beyond carried-over overlap?
    in_references = False

    def section_path() -> str:
        return " > ".join(t for _, t in headings)

    def emit(kind: str, parts: list[Block]) -> None:
        text = "\n\n".join(b.text for b in parts).strip()
        if not text:
            return
        chunks.append(
            Chunk(
                index=len(chunks),
                kind=kind,  # type: ignore[arg-type]
                section=section_path(),
                text=text,
                page_start=min(b.page for b in parts),
                page_end=max(b.page for b in parts),
            )
        )

    def overlap_tail(parts: list[Block]) -> list[Block]:
        """Trailing blocks (whole paragraphs) totalling <= overlap_tokens."""
        tail, total = [], 0
        for b in reversed(parts):
            t = count_tokens(b.text)
            if total + t > overlap_tokens:
                break
            tail.insert(0, b)
            total += t
        return tail

    def flush() -> None:
        nonlocal buf, fresh
        if fresh:
            emit("text", buf)
        buf, fresh = [], False

    def roll() -> None:
        """Emit buf and keep its tail as overlap for the next chunk."""
        nonlocal buf, fresh
        carry = overlap_tail(buf)
        emit("text", buf)
        buf, fresh = carry, False

    for block in blocks:
        if block.kind == "heading":
            flush()
            in_references = is_references_heading(block.text)
            while headings and headings[-1][0] >= block.level:
                headings.pop()
            headings.append((block.level, block.text.strip()))
            continue

        if in_references or not block.text.strip():
            continue

        if block.kind == "table":
            emit("table", [block])
            continue

        # Oversized single block: split on sentences first.
        pieces = (
            [Block(block.kind, p, block.page) for p in _split_oversized(block.text, max_tokens)]
            if count_tokens(block.text) > max_tokens
            else [block]
        )
        for piece in pieces:
            buf_tokens = sum(count_tokens(b.text) for b in buf)
            if fresh and buf_tokens + count_tokens(piece.text) > max_tokens:
                roll()
            # carried overlap + this piece would still exceed max: drop the overlap
            if sum(count_tokens(b.text) for b in buf) + count_tokens(piece.text) > max_tokens:
                buf = []
            buf.append(piece)
            fresh = True
            if sum(count_tokens(b.text) for b in buf) >= target_tokens:
                roll()

    flush()
    return chunks
