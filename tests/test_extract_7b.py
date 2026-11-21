"""Tests for scripts/extract_7b.py (stronger rival, stage A). Synthetic data only; no model."""
from __future__ import annotations

import csv
import gzip
import importlib.util
import io
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
_spec = importlib.util.spec_from_file_location("x7", REPO / "scripts" / "extract_7b.py")
x7 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(x7)

HEADER = ["distance_set", "split", "label", "trace_index", "t", "kind", "onset", "d",
          "pos_trace_index", "pos_t", "same_trace"]


def test_band_filter_reads_no_split_or_label(tmp_path):
    p = tmp_path / "e.csv.gz"
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(HEADER)
    # split and label hold junk: if either were parsed, int() or a split filter would act
    w.writerows([["6-10", "JUNK", "not-an-int", 3, 40, "", "", "", "", "", ""],
                 ["6-10", "test", "?", 3, 12, "", "", "", "", "", ""],
                 ["6-10", "train", "?", 1, 7, "", "", "", "", "", ""],
                 ["2-5", "train", 1, 1, 8, "", "", "", "", "", ""],
                 ["gate", "test", 1, 2, 9, "", "", "", "", "", ""],
                 ["6-10", "validation", 0, 3, 40, "", "", "", "", "", ""]])
    with gzip.open(p, "wt", newline="") as fh:
        fh.write(buf.getvalue())
    assert x7.load_band_positions(p) == {1: [7], 3: [12, 40]}


class Tok:
    eos_token_id = 151643

    def __init__(self, table):
        self.table = table

    def decode(self, ids, skip_special_tokens=False):
        return "".join(self.table.get(i, f"<{i}>") for i in ids)


def test_compat_flags_a_planted_mismatch():
    traces = [{"index": 0, "prompt_token_ids": [1, 2], "output_token_ids": [3, 4]}]
    a = Tok({1: "a", 2: "b", 3: "c", 4: "d"})
    assert x7.compat_check(a, Tok(dict(a.table)), traces)["ids_decoding_differently"] == []
    r = x7.compat_check(a, Tok({**a.table, 3: "C"}), traces)
    assert r["ids_decoding_differently"] == [3] and r["traces_full_decode_differs"] == [0]
    assert r["distinct_ids"] == 5  # 4 ids + EOS


class FakeModel:
    """Tiny stand-in: 3 'layers', hidden 4, logits = lm_head(last)."""
    device = torch.device("cpu")

    def __init__(self, poison=None):
        self.w = torch.arange(20, dtype=torch.float32).reshape(4, 5) / 10
        self.poison = poison

    def lm_head(self, h):
        return h @ self.w

    def __call__(self, ids, output_hidden_states, use_cache):
        L = ids.shape[1]
        hs = tuple(torch.full((1, L, 4), float(k)) + torch.arange(L)[None, :, None]
                   for k in range(4))
        if self.poison is not None:
            hs[self.poison][0, 0, 0] = float("inf")
        return SimpleNamespace(hidden_states=hs, logits=self.lm_head(hs[-1])[0][None])


def _trace():
    return {"index": 5, "prompt_token_ids": [9, 9], "output_token_ids": [3, 4, 5, 6],
            "eos_token_stripped": True}


def test_extract_saves_entries_1_to_last_at_P_plus_t():
    arrays, checks = x7.extract_trace(FakeModel(), _trace(), [1, 3])
    h = arrays["hidden"]
    assert h.shape == (2, 3, 4) and h.dtype == np.float16           # entries 1..3, not 0
    assert h[0, 0, 0] == 1 + 3 and h[1, 2, 0] == 3 + 5              # value k + (P + t)
    assert checks["wiring_max_abs_diff"] == 0.0 and checks["finite"]
    assert arrays["t"].tolist() == [1, 3]


def test_finite_check_catches_inf_anywhere():
    for k in (1, 3):  # a saved layer, at a position that is not saved
        _, checks = x7.extract_trace(FakeModel(poison=k), _trace(), [2])
        assert not checks["finite"]
    _, checks = x7.extract_trace(FakeModel(poison=0), _trace(), [2])  # embeddings: not saved
    assert checks["finite"]


def test_id_check_rejects_bad_position():
    tr = _trace()
    with pytest.raises(AssertionError):
        x7.extract_trace(FakeModel(), tr, [4])  # t beyond the generated ids
