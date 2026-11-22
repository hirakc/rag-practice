from paperrag.retrieval import rrf
from paperrag.store import paper_filter


def test_rrf_rewards_agreement_between_lists():
    fused = dict(rrf([["a", "b", "c"], ["b", "a", "d"]], k=60))
    assert fused["a"] == fused["b"]  # symmetric ranks -> same score
    assert fused["a"] > fused["c"] and fused["a"] > fused["d"]


def test_rrf_item_in_both_lists_beats_top_of_one_list():
    order = [k for k, _ in rrf([["x", "shared"], ["y", "shared"]], k=1)]
    # x: 1/2 ; shared: 1/3 + 1/3 = 0.667 -> shared wins with small k
    assert order[0] == "shared"


def test_rrf_is_deterministic_on_ties():
    assert [k for k, _ in rrf([["a"], ["b"]])] == ["a", "b"]


def test_rrf_empty():
    assert rrf([]) == [] and rrf([[], []]) == []


def test_paper_filter_quotes_and_escapes():
    assert paper_filter(None) is None
    assert paper_filter(["ab", "c'd"]) == "paper_id IN ('ab', 'c''d')"
