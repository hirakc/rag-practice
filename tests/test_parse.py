from pathlib import Path

from paperrag.ingest.chunk import Block
from paperrag.ingest.parse import extract_meta, heading_level


def test_heading_level_from_numbering():
    assert heading_level("2 Methods") == 1
    assert heading_level("2.1 Models") == 2
    assert heading_level("3.2.1 Details") == 3
    assert heading_level("A Implementation Details") == 1
    assert heading_level("A.3 More") == 2
    assert heading_level("Abstract") == 1
    assert heading_level("Broader Impact") == 1


def test_meta_heuristics_without_sidecar(tmp_path: Path):
    blocks = [
        Block("heading", "A Great Paper", 1),
        Block("text", "Ada Lovelace, Alan Turing", 1),
        Block("heading", "Abstract", 1),
        Block("text", "We propose things.", 1),
        Block("text", "arXiv:2005.11401v4 [cs.CL] 12 Apr 2021", 1),
        Block("heading", "1 Introduction", 1),
    ]
    m = extract_meta(blocks, tmp_path / "x.pdf")
    assert m.title == "A Great Paper"
    assert m.authors == ["Ada Lovelace, Alan Turing"]
    assert m.abstract.startswith("We propose things.")
    assert m.year == 2020


def test_meta_prefers_sidecar(tmp_path: Path):
    pdf = tmp_path / "p.pdf"
    (tmp_path / "p.meta.json").write_text('{"title": "T", "authors": ["A"], "year": 2019, "abstract": "x"}')
    m = extract_meta([], pdf)
    assert (m.title, m.authors, m.year) == ("T", ["A"], 2019)
