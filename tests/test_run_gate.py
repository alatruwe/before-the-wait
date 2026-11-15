"""Tests for scripts/run_gate.py (handoff 11a): loader boundary, train-only standardization,
G3 tie rule, bootstrap determinism. Fixtures only; no real data, no model."""
from __future__ import annotations

import csv
import gzip
import importlib.util
import io
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("run_gate", REPO / "scripts" / "run_gate.py")
rg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rg)

HEADER = ["distance_set", "split", "label", "trace_index", "t", "kind", "onset", "d",
          "pos_trace_index", "pos_t", "same_trace"]


def _write(path, rows):
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(HEADER)
    w.writerows(rows)
    with gzip.open(path, "wt", newline="") as fh:
        fh.write(buf.getvalue())


def test_loader_drops_test_and_band_rows(tmp_path):
    p = tmp_path / "e.csv.gz"
    _write(p, [
        ["gate", "train", 1, 1, 10, "word", 11, 1, "", "", ""],
        ["gate", "validation", 0, 2, 20, "word", "", "", 2, 30, True],
        ["gate", "test", 1, 3, 30, "word", 31, 1, "", "", ""],      # test: dropped
        ["2-5", "train", 1, 1, 40, "word", 43, 3, "", "", ""],      # band: dropped
        ["21-40", "validation", 0, 2, 50, "word", "", "", 2, 9, True],  # band: dropped
        ["11-20", "test", 1, 3, 60, "word", 75, 15, "", "", ""],    # both: dropped
    ])
    rows = rg.load_gate_rows(p)
    assert [(r["split"], r["trace_index"], r["t"], r["label"]) for r in rows] == [
        ("train", 1, 10, 1), ("validation", 2, 20, 0)]
    assert not any(r["split"] == "test" for r in rows)


def test_standardization_uses_train_statistics_only():
    rng = np.random.default_rng(0)
    tr = rng.normal(5.0, 2.0, size=(200, 4))
    va = rng.normal(-100.0, 50.0, size=(50, 4))
    A, B = rg.standardize(tr, va)
    assert np.allclose(A.mean(axis=0), 0, atol=1e-5) and np.allclose(A.std(axis=0), 1, atol=1e-5)
    mu, sd = tr.mean(axis=0), tr.std(axis=0)
    assert np.allclose(B, (va - mu) / sd, atol=1e-4)
    # changing validation does not change the transform of train
    A2, _ = rg.standardize(tr, va * 7 + 3)
    assert np.array_equal(A, A2)


def test_standardization_constant_dimension():
    tr = np.ones((10, 2))
    A, B = rg.standardize(tr, np.full((3, 2), 2.0))
    assert np.all(A == 0) and np.all(B == 1)


def test_best_fit_ties_lowest_layer_then_smallest_c():
    fits = [{"layer": 5, "C": 0.1, "auroc": 0.9}, {"layer": 3, "C": 1.0, "auroc": 0.9},
            {"layer": 3, "C": 0.01, "auroc": 0.9}, {"layer": 9, "C": 1e-4, "auroc": 0.89}]
    assert rg.best_fit(fits) == {"layer": 3, "C": 0.01, "auroc": 0.9}


def test_bootstrap_deterministic_and_by_problem():
    rng = np.random.default_rng(1)
    y = np.array([1, 0, 0] * 30)
    s = rng.normal(size=90) + y
    prob = np.repeat(np.arange(30), 3)
    a = rg.bootstrap_by_problem(y, s, prob, n=200)
    b = rg.bootstrap_by_problem(y, s, prob, n=200)
    assert a == b and a[0] < rg.auroc(y, s) < a[1]
