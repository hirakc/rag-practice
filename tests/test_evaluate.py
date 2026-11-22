from paperrag.evaluate import evidence_rank, normalize


def chunk(pid, text):
    return {"paper_id": pid, "text": text}


def test_normalize_ignores_case_punctuation_and_whitespace():
    assert normalize("Top-K  retrieval,\nK=5!") == "top k retrieval k 5"


def test_evidence_rank_finds_first_matching_chunk_in_right_paper():
    chunks = [chunk("p2", "We use DPR as retriever."),       # right quote, wrong paper
              chunk("p1", "Unrelated."),
              chunk("p1", "We use  DPR as the\nretriever.")]
    ev = [{"paper": "a.pdf", "quote": "DPR as the retriever"}]
    assert evidence_rank(chunks, ev, {"a.pdf": "p1", "b.pdf": "p2"}) == 3


def test_evidence_rank_none_when_missing_or_paper_unindexed():
    chunks = [chunk("p1", "nothing here")]
    assert evidence_rank(chunks, [{"paper": "a.pdf", "quote": "DPR"}], {"a.pdf": "p1"}) is None
    assert evidence_rank(chunks, [{"paper": "zzz.pdf", "quote": "nothing"}], {}) is None
