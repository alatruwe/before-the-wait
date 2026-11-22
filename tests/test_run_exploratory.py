"""Tests for scripts/run_exploratory.py and scripts/plot_exploratory_traces.py (step 14,
exploratory). Synthetic data only, except the seed draw on the committed split ids."""
from __future__ import annotations

import importlib.util
import json
import random
import re
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rx = _load("rx", "run_exploratory.py")
pt = _load("pt", "plot_exploratory_traces.py")
import run_rival as rr  # noqa: E402


def _pos(split, tr, t, d):
    return {"split": split, "label": 1, "trace_index": tr, "t": t, "d": d,
            "pos_trace_index": None, "pos_t": None, "same_trace": ""}


def _neg(split, tr, t, ptr, pt_, same):
    return {"split": split, "label": 0, "trace_index": tr, "t": t, "d": None,
            "pos_trace_index": ptr, "pos_t": pt_, "same_trace": same}


def test_exact_distance_gives_each_negative_its_positives_d():
    rows = [_pos("test", 1, 10, 7), _pos("test", 1, 11, 6), _pos("train", 1, 10, 9),
            _neg("test", 1, 500, 1, 11, "True"), _neg("test", 2, 40, 1, 10, "False"),
            _neg("train", 1, 600, 1, 10, "True")]
    assert rx.exact_distance(rows).tolist() == [7, 6, 9, 6, 7, 9]
    with pytest.raises(AssertionError, match="orphan"):
        rx.exact_distance(rows + [_neg("test", 3, 5, 9, 9, "False")])
    with pytest.raises(AssertionError, match="duplicate"):
        rx.exact_distance(rows + [_pos("test", 1, 10, 7)])


def test_same_trace_filter_drops_only_cross_trace_negatives():
    rows = [_pos("test", 1, 10, 7), _neg("test", 1, 500, 1, 10, "True"),
            _neg("test", 4, 50, 1, 10, "False"), _pos("test", 2, 3, 8)]
    assert rx.same_trace_mask(rows).tolist() == [True, True, False, True]
    with pytest.raises(AssertionError):
        rx.same_trace_mask([_neg("test", 1, 1, 1, 1, "")])


def test_group_blocks_are_d21s_columns():
    import scipy.sparse as sp

    rng = np.random.default_rng(0)
    n = 9
    T = sp.random(n, 12, density=0.3, random_state=1, format="csr")
    Z = rng.normal(size=(n, 6 + rr.HIDDEN))
    r = rr.dense_factors(Z)
    f_text, f = rr.multipliers(r, rr.text_r(T))
    full = rr.assemble(T, Z, f_text, f).toarray()
    blocks = [rx.group_block("text", T, Z, f_text, f).toarray()] + \
        [rx.group_block(g, T, Z, f_text, f) for g in ("entropy", "output", "position")]
    assert np.array_equal(np.hstack(blocks), full)
    assert f["output"] == 1.0


def test_group_choosers_tie_rules():
    text = [{"auroc": 0.8, "window": w, "n": n, "C": C}
            for w in (16, 8) for n in (3, 2) for C in (1e-2, 1e-4)]
    best = min(text, key=rx.text_key)
    assert (best["window"], best["n"], best["C"]) == (8, 2, 1e-4)
    dense = [{"auroc": 0.7, "C": 1.0}, {"auroc": 0.7, "C": 1e-3}, {"auroc": 0.6, "C": 1e-4}]
    assert min(dense, key=rx.dense_key)["C"] == 1e-3
    assert rx.at_edge(1e-4) and rx.at_edge(1.0) and not rx.at_edge(1e-2)


def test_trace_draw_reproduces_from_seed():
    ids = list(range(200, 300))
    assert pt.draw_traces(ids) == random.Random(20261021).sample(sorted(ids), 10)
    assert pt.draw_traces(list(reversed(ids))) == pt.draw_traces(ids)  # sorted first
    test_ids = json.loads((REPO / "results" / "split.json").read_text())["test"]
    d = pt.draw_traces(test_ids)
    assert len(set(d)) == 10 and set(d) <= set(test_ids)


def test_verdict_word_scanner():
    for bad in ("this band is positive", "a null result", "inconclusive", "Claim level 2",
                "the verdict was", "Positive."):
        assert rx.verdict_words(bad), bad
    for ok in ("positives and negatives", "positive positions", "positive position",
               "positive examples", rx.DISCLAIMER, "n_pos", "results/verdicts.csv"):
        assert rx.verdict_words(ok) == [], ok
    assert rx.html_own_text("a <!--quoted-->null<!--/quoted--> b") == "a  b"


def test_html_is_self_contained_and_marks_onsets():
    scores = {b: np.linspace(-1, 1, 6) for b in ("2-5", "6-10", "11-20", "21-40")}
    layers = {"2-5": 19, "6-10": 18, "11-20": 19, "21-40": 19}
    svg = pt.svg_plot(0, scores, [3], layers)
    body = pt.text_html(["a", "b", "c", "Wait", "x", "y"], 0, {3}, {5})
    out = pt.page(1, 42, "prob <b>", svg, body, layers, 1, "check.")
    assert not re.search(r"(src|href)=\"https?://", out) and "<script" not in out
    assert '<mark title="onset, t=3">Wait</mark>' in out
    assert "onset t=3" in out and 'title="merged hit, t=5"' in out
    assert "prob &lt;b&gt;" in out and "Exploratory" in out
    assert rx.verdict_words(rx.html_own_text(out)) == []


def test_paragraph_gap():
    text = "abc.\n\nWait x\n\n\n\nHmm"
    assert rx.paragraph_gap(text, 0, 6) == 1          # token before the break
    assert rx.paragraph_gap(text, 4, 6) == 1          # token holding the break
    assert rx.paragraph_gap(text, 6, 6) == 0          # inside the onset's paragraph
    assert rx.paragraph_gap(text, 7, 17) == 2         # empty paragraph counts (D23 pieces)
    with pytest.raises(AssertionError):
        rx.paragraph_gap(text, 8, 6)


def test_draws_are_d22s():
    keys, draws = rx.test_keys_and_draws()
    assert len(keys) == 102 and draws.shape == (10_000, 102)


def test_best_c_per_layer_reads_committed_csv():
    best = rx.best_c_per_layer(rx.PROBE_CSV, "6-10")
    assert sorted(best) == list(range(1, 28))
    assert best[18]["C"] == 1e-3 and best[18]["auroc_str"] == "0.852655"
    g = rx.best_c_per_layer(rx.GATE_CSV, None)
    assert sorted(g) == list(range(1, 28))
