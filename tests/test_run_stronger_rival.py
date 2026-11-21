"""Tests for scripts/run_stronger_rival.py (stronger rival, stage B). Synthetic data only."""
from __future__ import annotations

import csv
import gzip
import importlib.util
import io
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
_spec = importlib.util.spec_from_file_location("rs", REPO / "scripts" / "run_stronger_rival.py")
rs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rs)
import extract_activations as ea  # noqa: E402

HEADER = ["distance_set", "split", "label", "trace_index", "t", "kind", "onset", "d",
          "pos_trace_index", "pos_t", "same_trace"]


def test_loader_keeps_only_band_6_10_train_and_validation(tmp_path):
    p = tmp_path / "e.csv.gz"
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(HEADER)
    w.writerows([["6-10", "train", 1, 1, 10, "", "", "", "", "", ""],
                 ["6-10", "test", 1, 3, 30, "", "", "", "", "", ""],
                 ["2-5", "train", 1, 1, 11, "", "", "", "", "", ""],
                 ["gate", "validation", 1, 2, 40, "", "", "", "", "", ""],
                 ["6-10", "validation", 0, 2, 20, "", "", "", "", "", ""]])
    with gzip.open(p, "wt", newline="") as fh:
        fh.write(buf.getvalue())
    assert [(r["split"], r["t"]) for r in rs.load_rows(p)] == [("train", 10), ("validation", 20)]


def _act(tmp_path, tr, ts):
    hidden = np.zeros((len(ts), rs.N_LAYERS_7B, rs.HIDDEN_7B), np.float16)
    for k in range(len(ts)):
        hidden[k, :, 0] = np.arange(1, rs.N_LAYERS_7B + 1) + 100 * k
    data = ea.npz_bytes({"hidden": hidden, "t": np.array(ts, np.int32)})
    p = tmp_path / f"gsm8k_{tr:04d}.npz"
    p.write_bytes(data)
    return ea.sha256_file(p)


def test_gather_is_layer_major_and_refuses_test_traces(tmp_path):
    man = {1: _act(tmp_path, 1, [4, 9]), 2: _act(tmp_path, 2, [5])}
    rows = [{"trace_index": 1, "t": 9}, {"trace_index": 2, "t": 5}, {"trace_index": 1, "t": 4}]
    X = rs.gather_all_layers(rows, tmp_path, man, forbidden={7})
    assert X.shape == (28, 3, 3584)
    assert X[0, 0, 0] == 101 and X[27, 0, 0] == 128 and X[4, 1, 0] == 5 and X[0, 2, 0] == 1
    with pytest.raises(AssertionError, match="forbidden"):
        rs.gather_all_layers(rows, tmp_path, man, forbidden={2})
    man[2] = "0" * 64
    with pytest.raises(AssertionError, match="sha256"):
        rs.gather_all_layers(rows, tmp_path, man, forbidden=set())


def test_frozen_probe_hash_guard(tmp_path, monkeypatch):
    rs.check_frozen_probe()  # the real frozen probe passes
    fake = tmp_path / "band_6-10.npz"
    fake.write_bytes(b"not the probe")
    monkeypatch.setattr(rs, "PROBE", fake)
    with pytest.raises(AssertionError, match="frozen"):
        rs.check_frozen_probe()


# ---------------------------------------------------------------- stage C (written, not run on test)

def test_loader_test_split_only_band_6_10(tmp_path):
    p = tmp_path / "e.csv.gz"
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(HEADER)
    w.writerows([["6-10", "train", 1, 1, 10, "", "", "", "", "", ""],
                 ["6-10", "test", 1, 3, 30, "", "", "", "", "", ""],
                 ["2-5", "test", 1, 3, 31, "", "", "", "", "", ""],
                 ["6-10", "validation", 0, 2, 20, "", "", "", "", "", ""]])
    with gzip.open(p, "wt", newline="") as fh:
        fh.write(buf.getvalue())
    assert [(r["split"], r["t"]) for r in rs.load_rows(p, splits=("validation",))] == \
        [("validation", 20)]
    assert [(r["split"], r["t"]) for r in rs.load_rows(p, splits=("test",))] == [("test", 30)]


def test_s8_stops_on_mismatch():
    rs.s8_check(0.8683752768186852)  # the value on file passes
    for bad in (0.8683752768186851, 0.8684, 0.5):
        with pytest.raises(SystemExit, match="S8"):
            rs.s8_check(bad)


def test_draws_are_step_13s():
    rv = rs._rv()
    d = rs.draws_for_test()
    assert np.array_equal(d, rv.resample_draws(102))
    assert d.shape == (10_000, 102)


def test_lift_uses_d22_pairing_and_verdicts():
    rv = rs._rv()
    rng = np.random.default_rng(0)
    prob = np.repeat(np.arange(10), 6)
    y = np.tile([1, 0, 0, 1, 0, 0], 10)
    s_p = y + rng.normal(0, 0.5, len(y))
    s_r = y + rng.normal(0, 1.5, len(y))
    keys = list(range(10)) + [99]                  # a key with no examples contributes none
    draws = rv.resample_draws(len(keys), n=500)
    out = rs.lift_and_interval(y, s_p, s_r, prob, keys, draws)
    a_p, a_r = rv.paired_resamples(y, s_p, s_r, prob, keys, draws)
    assert (out["interval_low"], out["interval_high"]) == rv.interval(a_p - a_r)
    assert out["lift"] == rg_auroc(y, s_p) - rg_auroc(y, s_r)
    assert out["verdict_0.05"] == rv.verdict(out["interval_low"], out["interval_high"], 0.05)
    assert out["claim_level_2"] == (out["verdict_0.05"] == "positive")


def rg_auroc(y, s):
    import run_gate as rg
    return rg.auroc(y, s)


def test_single_layer_7b_read_and_forbidden(tmp_path, monkeypatch):
    hidden = np.full((2, rs.N_LAYERS_7B, rs.HIDDEN_7B), np.nan, np.float16)
    hidden[:, 18, :] = np.array([[1.0], [2.0]], np.float16)   # layer 19 only finite
    data = ea.npz_bytes({"hidden": hidden, "t": np.array([3, 8], np.int32)})
    (tmp_path / "gsm8k_0007.npz").write_bytes(data)
    monkeypatch.setattr(rs, "ACT_7B", tmp_path)
    man = {7: ea.sha256_file(tmp_path / "gsm8k_0007.npz")}
    X = rs.gather_layer_7b([{"trace_index": 7, "t": 8}, {"trace_index": 7, "t": 3}], 19, man, set())
    assert np.isfinite(X).all() and X[0, 0] == 2 and X[1, 0] == 1
    with pytest.raises(AssertionError, match="forbidden"):
        rs.gather_layer_7b([{"trace_index": 7, "t": 8}], 19, man, {7})


def test_test_mode_refuses_without_rehearsal(tmp_path, monkeypatch):
    import json
    rec = tmp_path / "r.json"
    monkeypatch.setattr(rs, "REHEARSAL_JSON", rec)
    with pytest.raises(AssertionError, match="no rehearsal"):
        rs.require_rehearsal()
    rec.write_text(json.dumps({"pass": True, "script_sha256": "0" * 64}))
    with pytest.raises(AssertionError, match="changed"):
        rs.require_rehearsal()
    rec.write_text(json.dumps({"pass": False, "script_sha256": ea.sha256_file(rs.SELF)}))
    with pytest.raises(AssertionError, match="did not pass"):
        rs.require_rehearsal()


def test_score_both_refuses_test_traces():
    with pytest.raises(AssertionError, match="forbidden"):
        rs.score_both([{"trace_index": 5, "t": 1, "label": 1, "split": "validation"}], {5})
