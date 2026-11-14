"""Tests for scripts/build_examples.py (handoff 09): split, candidate rule, matching.

Hand-built traces only; no tokenizer, no trace files.
"""
from __future__ import annotations

import importlib.util
import random
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("build_examples", REPO / "scripts" / "build_examples.py")
be = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(be)


# ---------------------------------------------------------------- split

def test_split_sizes_disjoint_and_reproducible():
    kept = list(range(1000, 1292))
    s = be.make_split(kept)
    assert [len(s[k]) for k in be.SPLITS] == [146, 44, 102]
    allx = s["train"] + s["validation"] + s["test"]
    assert sorted(allx) == kept and len(set(allx)) == 292
    order = sorted(kept)
    random.Random(20261019).shuffle(order)
    assert s["train"] == order[:146] and s["test"] == order[190:]


# ---------------------------------------------------------------- candidate rule

def test_candidates_quiet_zone_edges_and_spans():
    # reasoning span [0, 30), lookahead 5, hits at 12 (span 12..14) and 20 (span 20..21)
    c = be.negative_candidates(0, 30, [12, 20], [(12, 14), (20, 21)], lookahead=5)
    # t needs t+5 < 30 -> t <= 24; no hit in t+1..t+5; t not in [12,14) or [20,21)
    # t=6: window 7..11 clean -> ok; t=7: window 8..12 contains 12 -> no
    assert 6 in c and 7 not in c
    # 12 and 13 are inside a marker span; 14 sees 20 within 5? 15..19 -> clean at 14
    assert 12 not in c and 13 not in c and 14 in c
    assert 15 not in c and 19 not in c  # 20 within t+1..t+5
    assert 20 not in c  # in a span
    assert 21 in c and 24 in c and 25 not in c  # 25 + 5 = 30 not < 30
    assert c == [0, 1, 2, 3, 4, 5, 6, 14, 21, 22, 23, 24]


def test_candidates_respect_reasoning_start():
    c = be.negative_candidates(10, 30, [], [], lookahead=5)
    assert c[0] == 10 and c[-1] == 24


def test_quiet_after_counts_merged_hits_too():
    # any hit counts, not only onsets: the caller passes every hit index
    assert be.quiet_after(10, [3, 16], lookahead=5) is True
    assert be.quiet_after(10, [3, 15], lookahead=5) is False
    assert be.quiet_after(10, [10], lookahead=5) is True  # t itself is not in t+1..


# ---------------------------------------------------------------- matching

def _pos(split, tr, t, kind="word"):
    return {"split": split, "trace_index": tr, "t": t, "kind": kind, "onset": t + 1, "d": 1}


def test_matching_same_trace_first():
    split_of = {1: "train", 2: "train"}
    cands = {(1, "word"): [10, 11, 12, 13], (2, "word"): [50, 51]}
    kept, negs, dropped = be.match_set([_pos("train", 1, 5)], cands, split_of)
    assert len(kept) == 1 and not dropped
    assert all(n["same_trace"] and n["trace_index"] == 1 for n in negs) and len(negs) == 2


def test_matching_cross_trace_fallback_same_split_only():
    split_of = {1: "train", 2: "train", 3: "test"}
    cands = {(1, "word"): [10], (2, "word"): [50, 51], (3, "word"): [90, 91, 92]}
    kept, negs, dropped = be.match_set([_pos("train", 1, 5)], cands, split_of)
    assert not dropped and len(negs) == 2
    same = [n for n in negs if n["same_trace"]]
    cross = [n for n in negs if not n["same_trace"]]
    assert [n["t"] for n in same] == [10]
    assert len(cross) == 1 and cross[0]["trace_index"] == 2  # never trace 3 (test split)


def test_matching_no_reuse_and_kind_respected():
    split_of = {1: "train"}
    cands = {(1, "word"): [10, 11, 12, 13], (1, "digit"): [20, 21]}
    ps = [_pos("train", 1, 5), _pos("train", 1, 6), _pos("train", 1, 7, kind="digit")]
    kept, negs, dropped = be.match_set(ps, cands, split_of)
    assert not dropped
    keys = [(n["trace_index"], n["t"]) for n in negs]
    assert len(keys) == len(set(keys)) == 6
    assert sorted(n["t"] for n in negs if n["kind"] == "word") == [10, 11, 12, 13]
    assert sorted(n["t"] for n in negs if n["kind"] == "digit") == [20, 21]


def test_matching_d18_drop_returns_partial_draw_to_pool():
    # 3 'word' candidates in the split: the first positive takes 2; the second
    # can get only 1 -> dropped, and that 1 goes back; the third positive is a
    # 'digit' and is unaffected.
    split_of = {1: "train", 2: "train"}
    cands = {(1, "word"): [10, 11], (2, "word"): [50], (2, "digit"): [60, 61]}
    ps = [_pos("train", 1, 5), _pos("train", 2, 40), _pos("train", 2, 41, kind="digit")]
    kept, negs, dropped = be.match_set(ps, cands, split_of)
    # positive (1,5) takes 10, 11; positive (2,40) gets only 50 -> dropped
    assert [(p["trace_index"], p["t"]) for p in dropped] == [(2, 40)]
    assert {(p["trace_index"], p["t"]) for p in kept} == {(1, 5), (2, 41)}
    assert 50 not in [n["t"] for n in negs]  # returned, never recorded
    assert len(negs) == 4


def test_matching_is_deterministic_and_fresh_per_call():
    split_of = {1: "train", 2: "train"}
    cands = {(1, "word"): list(range(100, 140)), (2, "word"): list(range(200, 240))}
    ps = [_pos("train", 1, t) for t in range(5)] + [_pos("train", 2, t) for t in range(5)]
    a = be.match_set(ps, cands, split_of)
    b = be.match_set(ps, cands, split_of)
    assert a == b
    assert cands[(1, "word")] == list(range(100, 140))  # input not consumed


def test_clean_onsets_strict():
    gaps = {(1, 10): 5, (1, 50): 6, (2, 7): 41}
    assert be.clean_onsets(gaps, 5) == {(1, 50), (2, 7)}
    assert be.clean_onsets(gaps, 40) == {(2, 7)}
