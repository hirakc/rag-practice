from paperrag.generate import is_refusal, parse_citations, source_label


def test_parse_simple_and_adjacent_citations():
    assert parse_citations("A [1]. B [2][3].", 5) == [1, 2, 3]


def test_parse_lists_and_ranges():
    assert parse_citations("X [1, 4] and Y [2-3] and Z [5–6]", 6) == [1, 2, 3, 4, 5, 6]


def test_out_of_range_and_non_citations_ignored():
    assert parse_citations("see [9] and [0] and [a] and [1]", 3) == [1]


def test_duplicates_collapse():
    assert parse_citations("[2] ... [2] ... [2]", 3) == [2]


def test_refusal_detection_handles_curly_apostrophe():
    assert is_refusal("I couldn’t find this in your papers.")
    assert not is_refusal("The answer is 42 [1].")


def test_source_label_single_and_range_pages():
    c = {"paper_title": "RAG", "section": "4 Results", "page_start": 7, "page_end": 7}
    assert source_label(c) == "RAG, §4 Results, p. 7"
    c2 = {**c, "page_end": 8, "section": ""}
    assert source_label(c2) == "RAG, pp. 7-8"
