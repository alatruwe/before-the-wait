"""Lexicon tests ported from frictionmap.

Source: https://github.com/alatruwe/frictionmap/blob/6bc1552d335eab1434d93519644689f8e560e456/tests/test_clusters.py
Commit: 6bc1552d335eab1434d93519644689f8e560e456

Test bodies are unchanged; only the import differs. Not ported: the six
detect_excerpts/Corpus cluster tests and the word-index round-trip test,
which exercise code that lexicon.py does not include.
"""
from __future__ import annotations

from before_the_wait.lexicon import (
    _LEXICON,
    find_markers,
    strip_code_fences,
)


def test_find_markers_returns_sorted_positions():
    text = "actually wait hmm"
    markers = find_markers(text)
    positions = [m[0] for m in markers]
    assert positions == sorted(positions)


def test_find_markers_case_insensitive():
    markers = find_markers("Wait and ACTUALLY and hmm")
    assert len(markers) == 3


def test_lexicon_locked():
    assert _LEXICON == (
        "actually", "wait", "hmm", "no,",
        "let me reconsider", "on second thought", "scratch that",
        "hold on", "reconsidering", "i was wrong", "let me think",
        "now I'm realizing", "however",
    )


def test_word_boundary_no_substring_let_me():
    assert find_markers("delete me from the list") == []


def test_word_boundary_no_substring_no_comma():
    assert find_markers("anno, the year was 1066") == []


def test_dropped_markers_no_longer_matched():
    # "but" and "let me check" were calibrated out — must not re-enter
    # the lexicon as a regression.
    assert find_markers("but it might still work") == []
    assert find_markers("let me check the docs first") == []


def test_let_me_think_matches():
    markers = find_markers("let me think about this carefully")
    assert len(markers) == 1
    assert markers[0][2].lower() == "let me think"


def test_however_matches():
    markers = find_markers("However, the schema is wrong")
    assert len(markers) == 1
    assert markers[0][2].lower() == "however"


def test_no_comma_disambiguator_matches():
    markers = find_markers("no, that won't work")
    assert len(markers) == 1
    assert markers[0][2].lower() == "no,"


def test_no_problem_does_not_match():
    assert find_markers("no problem here, all good") == []


def test_multi_word_literal_whitespace():
    # Tab between "let me" and "reconsider" → no match.
    assert find_markers("let me\treconsider that") == []


def test_multi_word_literal_whitespace_double_space():
    # Two spaces between words → no match (literal single space only).
    assert find_markers("let me  reconsider that") == []


def test_longest_first_alternation():
    # "let me reconsider" should win over "let me think" prefix; both are
    # in the lexicon but the longer match should be preferred at this offset.
    markers = find_markers("let me reconsider this approach")
    matched = [m[2].lower() for m in markers]
    assert matched == ["let me reconsider"]


def test_now_im_realizing_case_insensitive():
    markers = find_markers("Now I'm Realizing this is wrong")
    assert len(markers) == 1


def test_marker_inside_fence_filtered():
    text = "outside wait\n```\ninside wait inside\n```\nafter"
    markers = find_markers(text)
    matched_offsets = [m[0] for m in markers]
    # The "wait" inside the fence must be filtered. The two outside
    # ("outside wait" and the implicit "wait" prefix of "wait inside"
    # is INSIDE the fence, so excluded) — only the very first remains.
    assert len(markers) == 1
    assert text[markers[0][0]:markers[0][1]].lower() == "wait"
    # Sanity: the surviving offset is before any fence.
    assert matched_offsets[0] < text.index("```")


def test_marker_outside_fence_on_same_line_hits():
    # "actually" sits on the same line as the closing fence, after the
    # third backtick — must hit and offsets must point at original text.
    text = "```\ncode\n``` actually here"
    markers = find_markers(text)
    assert len(markers) == 1
    start, end, marker = markers[0]
    assert text[start:end] == "actually"
    assert marker.lower() == "actually"


def test_strip_code_fences_removes_fenced_block():
    text = "before\n```\nfenced wait\n```\nafter"
    stripped = strip_code_fences(text)
    assert "fenced" not in stripped
    assert "before" in stripped and "after" in stripped
