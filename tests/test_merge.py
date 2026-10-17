"""D10/D11 tests: looped-trace exclusion and hit -> onset merging."""
from __future__ import annotations

import pytest

from before_the_wait.label import Onset, exclude_looped, merge_onsets


def hit(token_index: int, same_para: bool | None) -> Onset:
    return Onset(
        dataset="t", index=0, token_index=token_index, marker="wait",
        char_start=token_index * 4, prev_token_text=".", prev_token_kind="period",
        paragraph_initial=not same_para, onset_mid_token=False,
        tokens_since_prev_hit=None, tokens_to_next_hit=None,
        same_paragraph_as_prev_hit=same_para,
    )


# --- paragraph rule (D11 primary) ---

def test_paragraph_first_hit_is_always_an_onset():
    hits = [hit(10, None)]
    assert merge_onsets(hits, "paragraph") == [(hits[0], None)]


def test_paragraph_same_paragraph_hits_merge():
    hits = [hit(10, None), hit(13, True), hit(17, True)]
    merged = merge_onsets(hits, "paragraph")
    assert [m for _, m in merged] == [None, 10, 10]


def test_paragraph_merges_into_first_hit_not_previous():
    """Three hits in one paragraph all point at the paragraph's first hit."""
    hits = [hit(10, None), hit(13, True), hit(50, True)]
    assert [m for _, m in merge_onsets(hits, "paragraph")] == [None, 10, 10]


def test_paragraph_new_paragraph_starts_a_new_onset():
    hits = [hit(10, None), hit(13, True), hit(40, False), hit(44, True)]
    assert [m for _, m in merge_onsets(hits, "paragraph")] == [None, 10, None, 40]


def test_paragraph_all_distinct_paragraphs_all_onsets():
    hits = [hit(10, None), hit(40, False), hit(90, False)]
    assert all(m is None for _, m in merge_onsets(hits, "paragraph"))


# --- window rule (parked; counted once) ---

def test_window_merges_at_exactly_n():
    """The fixed wording is difference <= n, so a gap of exactly n merges."""
    hits = [hit(100, None), hit(110, False)]
    assert [m for _, m in merge_onsets(hits, "window", n=10)] == [None, 100]


def test_window_keeps_gap_of_n_plus_one():
    hits = [hit(100, None), hit(111, False)]
    assert [m for _, m in merge_onsets(hits, "window", n=10)] == [None, None]


def test_window_measures_against_previous_hit_not_previous_onset():
    """Merged hits still reset the reference point — 'merged or not'."""
    hits = [hit(100, None), hit(108, False), hit(116, False)]
    # 108 merges into 100; 116 is 8 from the previous HIT (108), so it merges
    # too — into 100, the current onset. Against the previous ONSET it would
    # be 16 away and would count as an onset.
    assert [m for _, m in merge_onsets(hits, "window", n=10)] == [None, 100, 100]


def test_window_ignores_paragraph_structure():
    hits = [hit(100, None), hit(105, False)]
    assert [m for _, m in merge_onsets(hits, "window", n=10)] == [None, 100]


def test_window_n_zero_keeps_everything():
    hits = [hit(10, None), hit(11, True), hit(12, True)]
    assert all(m is None for _, m in merge_onsets(hits, "window", n=0))


def test_unknown_rule_raises():
    with pytest.raises(ValueError):
        merge_onsets([hit(1, None)], "nonsense")


# --- D10 ---

def test_exclude_looped_drops_only_looped():
    keys = [("gsm8k", 1), ("gsm8k", 2), ("math500", 3)]
    assert exclude_looped(keys, {("gsm8k", 2)}) == [("gsm8k", 1), ("math500", 3)]


def test_exclude_looped_empty_set_is_identity():
    keys = [("gsm8k", 1), ("math500", 3)]
    assert exclude_looped(keys, set()) == keys
