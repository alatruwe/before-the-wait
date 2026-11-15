"""Tests for scripts/run_probes.py (handoff 11b): loader boundary, layer-28 exclusion,
train-only standardization, saved-probe round trip. Fixtures only; no real data."""
from __future__ import annotations

import csv
import gzip
import importlib.util
import io
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
_spec = importlib.util.spec_from_file_location("run_probes", REPO / "scripts" / "run_probes.py")
rp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rp)
import extract_activations as ea  # noqa: E402

HEADER = ["distance_set", "split", "label", "trace_index", "t", "kind", "onset", "d",
          "pos_trace_index", "pos_t", "same_trace"]


def test_loader_drops_test_and_gate_rows(tmp_path):
    p = tmp_path / "e.csv.gz"
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(HEADER)
    w.writerows([
        ["2-5", "train", 1, 1, 10, "word", 13, 3, "", "", ""],
        ["21-40", "validation", 0, 2, 20, "word", "", "", 2, 50, True],
        ["6-10", "test", 1, 3, 30, "word", 38, 8, "", "", ""],     # test: dropped
        ["gate", "train", 1, 1, 40, "word", 41, 1, "", "", ""],    # gate: dropped
        ["gate", "test", 0, 3, 50, "word", "", "", 3, 9, True],    # both: dropped
        ["11-20", "train", 0, 1, 60, "word", "", "", 1, 75, True],
    ])
    with gzip.open(p, "wt", newline="") as fh:
        fh.write(buf.getvalue())
    out = rp.load_band_rows(p)
    got = {b: [(r["split"], r["trace_index"], r["t"], r["label"]) for r in rs]
           for b, rs in out.items()}
    assert got == {"2-5": [("train", 1, 10, 1)], "6-10": [], "11-20": [("train", 1, 60, 0)],
                   "21-40": [("validation", 2, 20, 0)]}


def test_gather_reads_layers_1_to_27_only(tmp_path):
    hidden = np.zeros((3, 28, rp.HIDDEN), dtype=np.float16)
    for k in range(27):
        hidden[:, k, :] = k + 1                      # layer k+1
    hidden[:, 27, :] = np.nan                        # layer 28 must never appear
    hidden[:, :, 0] += np.arange(3)[:, None]         # row marker
    arrays = {"hidden": hidden, "t": np.array([4, 9, 15], dtype=np.int32),
              "entropy": np.zeros(20, np.float32), "prompt_entropy": np.zeros(5, np.float32)}
    data = ea.npz_bytes(arrays)
    (tmp_path / "gsm8k_0007.npz").write_bytes(data)
    import hashlib

    manifest = {7: hashlib.sha256(data).hexdigest()}
    rows = [{"trace_index": 7, "t": 15}, {"trace_index": 7, "t": 4}]
    X = rp.gather_layers_1_27(rows, tmp_path, set(), manifest)
    assert X.shape == (27, 2, rp.HIDDEN) and np.isfinite(X).all()
    assert X[0, 0, 1] == 1 and X[26, 0, 1] == 27      # layer order
    assert X[5, 0, 0] == 6 + 2 and X[5, 1, 0] == 6 + 0  # row order follows rows


def test_gather_rejects_hash_mismatch(tmp_path):
    arrays = {"hidden": np.zeros((1, 28, rp.HIDDEN), np.float16), "t": np.array([1], np.int32),
              "entropy": np.zeros(2, np.float32), "prompt_entropy": np.zeros(1, np.float32)}
    (tmp_path / "gsm8k_0001.npz").write_bytes(ea.npz_bytes(arrays))
    try:
        rp.gather_layers_1_27([{"trace_index": 1, "t": 1}], tmp_path, set(), {1: "0" * 64})
    except AssertionError:
        return
    raise AssertionError("hash mismatch not caught")


def test_zstats_from_train_only():
    rng = np.random.default_rng(0)
    tr = rng.normal(3.0, 2.0, size=(300, 5)).astype(np.float16)
    mu, sd = rp.zstats(tr)
    assert mu.dtype == np.float64 and sd.dtype == np.float64
    assert np.allclose(mu, tr.astype(np.float64).mean(axis=0), atol=1e-6)
    z = rp.apply_z(tr, mu, sd)
    assert np.allclose(z.mean(axis=0), 0, atol=1e-5) and np.allclose(z.std(axis=0), 1, atol=1e-4)


def test_fit_layer_and_saved_probe_round_trip(tmp_path):
    rng = np.random.default_rng(1)
    n, dim = 120, 8
    y = (rng.random(n) < 0.33).astype(int)
    X = (rng.normal(size=(n, dim)) + y[:, None] * 0.8).astype(np.float16)
    Xva = rng.normal(size=(40, dim)).astype(np.float16)
    yva = (rng.random(40) < 0.33).astype(int)
    yva[:2] = [0, 1]
    fits, _ = rp.fit_layer("2-5", 3, X, y, Xva, yva)
    assert [f["C"] for f in fits] == rp.C_GRID and all(f["converged"] for f in fits)
    mu, sd = rp.zstats(X)
    assert all(np.array_equal(f["mu"], mu) for f in fits)  # train statistics, not validation
    best = fits[2]
    a = ea.npz_bytes(rp.probe_arrays(best))
    assert a == ea.npz_bytes(rp.probe_arrays(best))  # reproducible bytes
    p = tmp_path / "band.npz"
    p.write_bytes(a)
    with np.load(p) as z:
        s = rp.probe_scores(Xva, z["mean"], z["sd"], z["coef"], z["intercept"][0])
        assert int(z["layer"][0]) == 3 and float(z["C"][0]) == best["C"]
    assert rp.rg.auroc(yva, s) == best["auroc"]
