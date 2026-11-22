from paperrag.ingest.chunk import Block, chunk_blocks, count_tokens, is_references_heading


def para(n_words: int, page: int = 1, word: str = "w") -> Block:
    return Block("text", " ".join([word] * n_words) + ".", page)


def test_chunks_do_not_cross_sections_and_carry_section_path():
    blocks = [
        Block("heading", "1 Introduction", 1, level=1),
        para(50),
        Block("heading", "2 Method", 2, level=1),
        Block("heading", "2.1 Encoder", 2, level=2),
        para(50, page=2),
    ]
    chunks = chunk_blocks(blocks)
    assert [c.section for c in chunks] == ["1 Introduction", "2 Method > 2.1 Encoder"]
    assert chunks[1].page_start == 2


def test_sibling_heading_replaces_previous_at_same_level():
    blocks = [
        Block("heading", "2 Method", 1, level=1),
        Block("heading", "2.1 A", 1, level=2),
        para(10),
        Block("heading", "2.2 B", 1, level=2),
        para(10),
    ]
    assert [c.section for c in chunk_blocks(blocks)] == ["2 Method > 2.1 A", "2 Method > 2.2 B"]


def test_long_section_splits_under_max_with_overlap():
    blocks = [Block("heading", "S", 1)] + [para(100, page=i // 3 + 1) for i in range(20)]
    chunks = chunk_blocks(blocks, target_tokens=400, max_tokens=500, overlap_tokens=140)
    assert len(chunks) > 1
    assert all(c.token_count <= 500 + 5 for c in chunks)
    # overlap: last paragraph of chunk i reappears at start of chunk i+1
    first_last_para = chunks[0].text.split("\n\n")[-1]
    assert chunks[1].text.startswith(first_last_para)


def test_tables_are_standalone_chunks():
    blocks = [
        Block("heading", "4 Results", 3),
        para(20, page=3),
        Block("table", "Table 1: scores\n| a | b |\n|---|---|\n| 1 | 2 |", 3),
        para(20, page=3),
    ]
    chunks = chunk_blocks(blocks)
    kinds = [c.kind for c in chunks]
    assert kinds.count("table") == 1
    table = next(c for c in chunks if c.kind == "table")
    assert table.text.startswith("Table 1") and table.section == "4 Results"


def test_references_section_is_skipped_but_appendix_is_kept():
    blocks = [
        Block("heading", "5 Conclusion", 8),
        para(10, page=8),
        Block("heading", "References", 9),
        Block("text", "[1] Smith et al. 2020. A paper.", 9),
        Block("heading", "A Appendix", 10),
        para(10, page=10, word="appendix"),
    ]
    chunks = chunk_blocks(blocks)
    assert [c.section for c in chunks] == ["5 Conclusion", "A Appendix"]
    assert not any("Smith" in c.text for c in chunks)


def test_oversized_single_paragraph_is_split_on_sentences():
    long = " ".join(f"Sentence number {i} is here." for i in range(300))
    chunks = chunk_blocks([Block("heading", "S", 1), Block("text", long, 1)], 300, 400, 0)
    assert len(chunks) > 1
    assert all(c.token_count <= 410 for c in chunks)


def test_references_heading_variants():
    for h in ["References", "REFERENCES", "7 References", "VII. Bibliography"]:
        assert is_references_heading(h)
    assert not is_references_heading("Referenced Work in Retrieval")


def test_count_tokens_is_monotonic():
    assert count_tokens("a b c") < count_tokens("a b c d e f")
