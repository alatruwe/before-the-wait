"""Tests for scripts/export_wave.py (handoff viz-wave-data, exploratory). Synthetic data."""
from __future__ import annotations

import importlib.util
import json
import random
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("xw", REPO / "scripts" / "export_wave.py")
xw = importlib.util.module_from_spec(spec)
spec.loader.exec_module(xw)


def _pos(tr, onset, d):
    return {"label": 1, "trace_index": tr, "t": onset - d, "onset": onset,
            "pos_trace_index": None, "pos_t": None}


def _neg(tr, t, ptr, pt):
    return {"label": 0, "trace_index": tr, "t": t, "onset": None,
            "pos_trace_index": ptr, "pos_t": pt}


def _set(rows, d, per_layer):
    """per_layer: one score array over rows, repeated for all 27 layers."""
    return (rows, np.array(d), [np.array(per_layer, dtype=float)] * 27)


def _toy():
    # event A = (1, 100) has positives at d 6 and 7; event B = (2, 50) only at d 6
    rows = [_pos(1, 100, 6), _pos(1, 100, 7), _pos(2, 50, 6),
            _neg(1, 10, 1, 94), _neg(3, 5, 1, 94),   # both matched to A, d 6
            _neg(4, 9, 1, 93),                       # matched to A, d 7
            _neg(5, 9, 2, 44)]                       # matched to B, d 6
    scores = [5.0, 7.0, 9.0, 1.0, 3.0, 2.0, 4.0]
    events = [(1, 100), (2, 50)]
    return events, {"6-10": _set(rows, [6, 7, 6, 6, 6, 7, 6], scores)}


def test_missing_distance_is_nan_in_both_grids():
    events, per = _toy()
    pos_raw, neg_raw, pos_z, neg_z, n, *_ = xw.build_grids(events, per)
    assert pos_raw.shape == (2, 27, 40)
    for g in (pos_raw, neg_raw, pos_z, neg_z):
        assert np.isnan(g[1, 0, 6])            # B has no d = 7
        assert np.isnan(g[0, 0, 0])            # no gate set given: d = 1 empty
    assert n[0, 5] == 2 and n[0, 6] == 1 and n[0, 0] == 0


def test_negative_lands_in_its_positives_cell():
    events, per = _toy()
    pos_raw, neg_raw, *_ = xw.build_grids(events, per)
    assert pos_raw[0, 3, 5] == 5.0 and pos_raw[0, 3, 6] == 7.0 and pos_raw[1, 3, 5] == 9.0
    assert neg_raw[0, 3, 5] == 2.0             # mean of 1 and 3
    assert neg_raw[0, 3, 6] == 2.0 and neg_raw[1, 3, 5] == 4.0


def test_standardization_uses_negatives_only():
    events, per = _toy()
    _, _, pos_z, neg_z, _, neg_mean, neg_sd = xw.build_grids(events, per)
    negs = np.array([1.0, 3.0, 2.0, 4.0])
    si = xw.SETS.index("6-10")
    assert neg_mean[si, 0] == negs.mean() and neg_sd[si, 0] == negs.std()
    assert np.isnan(neg_mean[xw.SETS.index("gate"), 0])
    assert np.isclose(pos_z[0, 0, 5], (5.0 - negs.mean()) / negs.std())
    assert np.isclose(neg_z[0, 0, 5], ((1 - 2.5) / negs.std() + (3 - 2.5) / negs.std()) / 2)


def test_sample_draw_reproduces_from_seed():
    assert xw.draw_sample(806) == random.Random(20261122).sample(range(806), 200)
    assert xw.draw_sample(806)[:3] == xw.draw_sample(806)[:3]


def test_sample_json_has_nulls_and_note():
    events, per = _toy()
    _, _, pos_z, neg_z, n, *_ = xw.build_grids(events, per)
    obj = json.loads(xw.sample_json(events, pos_z, neg_z, n, [1, 0], 2))
    assert obj["events"][0]["trace_index"] == 2 and obj["events"][0]["pos_z"][0][6] is None
    assert "Exploratory" in obj["note"] and obj["seams"] == [5.5, 10.5, 20.5]
