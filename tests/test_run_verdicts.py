"""Tests for scripts/run_verdicts.py (handoff 13): resampling unit, pairing, cross-trace
negatives, percentiles, verdict boundaries, control band, loader, rehearsal lock,
single-layer read. Fixtures only; no real data."""
from __future__ import annotations

import csv
import gzip
import importlib.util
import io
import json
import random
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
_spec = importlib.util.spec_from_file_location("run_verdicts", REPO / "scripts" / "run_verdicts.py")
rv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rv)
import extract_activations as ea  # noqa: E402
import run_gate as rg  # noqa: E402

HEADER = ["distance_set", "split", "label", "trace_index", "t", "kind", "onset", "d",
          "pos_trace_index", "pos_t", "same_trace"]


def test_loader_keeps_named_splits_and_drops_gate(tmp_path):
    p = tmp_path / "e.csv.gz"
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(HEADER)
    w.writerows([["2-5", "train", 1, 1, 10, "word", 13, 3, "", "", ""],
                 ["2-5", "test", 1, 3, 30, "word", 33, 3, "", "", ""],
                 ["gate", "test", 1, 2, 40, "word", 41, 1, "", "", ""],
                 ["21-40", "validation", 0, 2, 20, "word", "", "", 2, 50, True]])
    with gzip.open(p, "wt", newline="") as fh:
        fh.write(buf.getvalue())
    out = rv.load_rows(p, {"validation"})
    assert [(b, r["split"]) for b in rv.BANDS for r in out[b]] == [("21-40", "validation")]
    out = rv.load_rows(p, {"train", "test"})
    assert [(b, r["t"]) for b in rv.BANDS for r in out[b]] == [("2-5", 10), ("2-5", 30)]


def test_draws_match_g4_order():
    keys = list(range(7))
    d = rv.resample_draws(len(keys), n=3, seed=5)
    rng = random.Random(5)
    assert d.tolist() == [[rng.randrange(7) for _ in keys] for _ in range(3)]


def test_resampling_unit_is_the_problem_and_pairing():
    # problem 10: two examples; problem 20: one; problem 30: no examples at all
    y = np.array([1, 0, 1, 0])
    prob = np.array([10, 10, 20, 20])
    s_p = np.array([0.9, 0.1, 0.8, 0.2])
    s_r = np.array([0.1, 0.9, 0.2, 0.8])
    keys = [10, 20, 30]
    draws = np.array([[0, 0, 2], [1, 1, 1], [2, 2, 2], [0, 1, 2]])
    a_p, a_r = rv.paired_resamples(y, s_p, s_r, prob, keys, draws)
    # draw 0: problem 10 twice -> examples [0,1,0,1]; both readers on the same examples
    assert a_p[0] == rg.auroc(y[[0, 1, 0, 1]], s_p[[0, 1, 0, 1]]) == 1.0
    assert a_r[0] == rg.auroc(y[[0, 1, 0, 1]], s_r[[0, 1, 0, 1]]) == 0.0
    assert a_p[1] == 1.0 and a_r[1] == 0.0
    assert np.isnan(a_p[2]) and np.isnan(a_r[2])          # empty resample: dropped
    assert a_p[3] == 1.0


def test_cross_trace_negative_follows_its_own_trace():
    # positive in trace 1; its matched negative sits in trace 2 (row trace_index = 2)
    rows = [{"label": 1, "trace_index": 1}, {"label": 0, "trace_index": 2},
            {"label": 0, "trace_index": 1}]
    y, prob = rv.labels(rows), rv.problems(rows)
    s = np.array([0.9, 0.5, 0.1])
    keys = [1, 2]
    a_p, _ = rv.paired_resamples(y, s, s, prob, keys, np.array([[0, 0]]))  # only trace 1
    assert a_p[0] == rg.auroc(np.array([1, 0, 1, 0]), np.array([0.9, 0.1, 0.9, 0.1]))
    a_p, _ = rv.paired_resamples(y, s, s, prob, keys, np.array([[1, 1]]))  # only trace 2
    assert np.isnan(a_p[0])                                # the negative came alone


def test_interval_percentiles():
    v = np.arange(1, 10001, dtype=float)
    lo, hi = rv.interval(v)
    assert (lo, hi) == tuple(np.percentile(v, [0.625, 99.375]))
    assert lo == pytest.approx(1 + 0.00625 * 9999) and hi == pytest.approx(1 + 0.99375 * 9999)
    lo2, hi2 = rv.interval(np.concatenate([v, [np.nan]]))  # NaN (dropped) excluded
    assert (lo2, hi2) == (lo, hi)


def test_verdict_boundaries():
    d = 0.05
    assert rv.verdict(0.001, 0.05, d) == "positive"        # upper == δ counts
    assert rv.verdict(0.0, 0.05, d) == "inconclusive"      # lower == 0 is not > 0
    assert rv.verdict(0.001, 0.0499999, d) == "null"       # just under δ
    assert rv.verdict(-0.2, 0.0499999, d) == "null"
    assert rv.verdict(-0.01, 0.2, d) == "inconclusive"
    assert rv.verdict(0.06, 0.10, d) == "positive"


def test_control_and_chance_bands_inclusive():
    assert rv.control_passes(0.45) and rv.control_passes(0.55) and rv.control_passes(0.5)
    assert not rv.control_passes(0.4499) and not rv.control_passes(0.5501)
    assert rv.near_chance(0.45) and not rv.near_chance(0.56)


def test_only_the_chosen_layer_is_read(tmp_path, monkeypatch):
    hidden = np.full((2, 28, rv.rp.HIDDEN), np.nan, dtype=np.float16)
    hidden[:, 18, :] = np.array([[1.0], [2.0]], dtype=np.float16)  # layer 19
    data = ea.npz_bytes({"hidden": hidden, "t": np.array([3, 8], np.int32)})
    (tmp_path / "gsm8k_0007.npz").write_bytes(data)
    monkeypatch.setattr(rv, "ACT_DIR", tmp_path)
    sha = rg.sha256_file(tmp_path / "gsm8k_0007.npz")
    X = rv.gather_layer([{"trace_index": 7, "t": 8}, {"trace_index": 7, "t": 3}], 19,
                        {7: sha}, set())
    assert np.isfinite(X).all() and X[0, 0] == 2 and X[1, 0] == 1


def test_run_refuses_without_passing_rehearsal(tmp_path, monkeypatch):
    rec = tmp_path / "rehearsal.json"
    monkeypatch.setattr(rv, "REHEARSAL", rec)
    with pytest.raises(AssertionError, match="no rehearsal"):
        rv.require_rehearsal()
    rec.write_text(json.dumps({"pass": False, "script_sha256": rg.sha256_file(rv.SELF)}))
    with pytest.raises(AssertionError, match="did not pass"):
        rv.require_rehearsal()
    rec.write_text(json.dumps({"pass": True, "script_sha256": "0" * 64}))
    with pytest.raises(AssertionError, match="changed"):
        rv.require_rehearsal()
    rec.write_text(json.dumps({"pass": True, "script_sha256": rg.sha256_file(rv.SELF)}))
    assert rv.require_rehearsal()["pass"] is True


def test_lifts_csv_size_guard():
    rows = [{"band": b, "_lifts": np.full(rv.N_BOOT, -0.012345678901234567)} for b in rv.BANDS]
    assert len(rv.lifts_csv(rows)) < rv.MAX_CSV_BYTES
