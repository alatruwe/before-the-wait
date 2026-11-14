"""Tests for scripts/extract_activations.py (handoff 10): offsets, layer index, entropy,
replay, container. A tiny fake causal LM only; no test loads the real model."""
from __future__ import annotations

import csv
import gzip
import importlib.util
import io
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("extract_activations",
                                               REPO / "scripts" / "extract_activations.py")
ea = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ea)

from before_the_wait.lens import sampling_probability  # noqa: E402

V = 32  # vocabulary size = hidden size in the fake


class FakeLM:
    """hidden_states[0] = onehot(id) (embeddings), [k] = (k + 1) * onehot(id) for
    k = 1..n-1, and the last entry = onehot(next id) ("already normed"). Logits =
    lm_head(last) = 10 * last, so row i predicts ids[i + 1]: replay is perfect
    only if the code reads token j at row j - 1."""

    def __init__(self, n_layers: int = 4):
        self.n_layers = n_layers
        self.device = torch.device("cpu")
        self.lm_head = lambda h: 10.0 * h
        self.model = SimpleNamespace(norm=lambda h: 2.0 * h + 1.0)

    def __call__(self, ids, output_hidden_states=True, use_cache=False):
        x = torch.nn.functional.one_hot(ids[0], V).float()
        nxt = torch.zeros_like(x)
        nxt[:-1] = x[1:]
        hs = [x[None]] + [((k + 1) * x)[None] for k in range(1, self.n_layers)] + [nxt[None]]
        return SimpleNamespace(hidden_states=tuple(hs), logits=self.lm_head(nxt[None])[0][None])


def _trace(prompt, out, stripped):
    return {"index": 7, "prompt_token_ids": prompt, "output_token_ids": out,
            "eos_token_stripped": stripped}


EOS = 31


@pytest.mark.parametrize("stripped", [True, False])
def test_offset_layers_entropy_and_replay(stripped):
    prompt = [1, 2, 3, 4]
    out = [5, 6, 7, 8, 9, 10, 11]
    ts = [0, 2, 6]
    m = FakeLM(n_layers=4)
    arrays, checks = ea.extract_trace(m, _trace(prompt, out, stripped), EOS, ts, chunk=3)
    h = arrays["hidden"]
    assert h.shape == (3, 4, V)  # entry 0 dropped: 4 layers saved
    for r, t in enumerate(ts):
        tok = out[t]
        # saved layers 1..3 are (k + 1) * onehot(token at P + t), in order
        for k in range(1, 4):
            assert h[r, k - 1].argmax() == tok and h[r, k - 1, tok] == k + 1
        # saved layer 4 is the last entry, unnormed
        nxt = out[t + 1] if t + 1 < len(out) else (EOS if stripped else None)
        if nxt is not None:
            assert h[r, 3].argmax() == nxt and h[r, 3, nxt] == 1.0
    assert list(arrays["t"]) == ts and arrays["t"].dtype == np.int32
    # entropy index: entry i from the logits at P + i; prompt entries at input i
    ids = prompt + out + ([EOS] if stripped else [])
    full = m(torch.tensor([ids])).logits[0]
    ent = ea.entropy_rows(full).numpy()
    assert np.allclose(arrays["entropy"], ent[len(prompt):len(prompt) + len(out)])
    assert np.allclose(arrays["prompt_entropy"], ent[:len(prompt)])
    assert arrays["entropy"].shape == (len(out),) and arrays["prompt_entropy"].shape == (4,)
    # replay: token j read at row P + j - 1 -> all sampleable; EOS included when restored
    assert checks["replay_n"] == len(out) + (1 if stripped else 0)
    assert checks["replay_share"] == 1.0 and checks["replay_n_zero"] == 0
    assert checks["wiring_max_abs_diff"] == 0.0 and checks["norm_twice_max_abs_diff"] > 0


def test_offset_guard_rejects_out_of_range_t():
    with pytest.raises(AssertionError):
        ea.extract_trace(FakeLM(), _trace([1, 2], [5, 6], False), EOS, [2])


def _warped_prob(row, tok, temperature=0.6, top_p=0.95):
    """Reference: transformers' processors on one row, as generate() applied them."""
    from transformers.generation.logits_process import TemperatureLogitsWarper, TopPLogitsWarper

    x = row.float()[None].clone()
    x = TopPLogitsWarper(top_p)(None, TemperatureLogitsWarper(temperature)(None, x))
    return float(torch.softmax(x, dim=-1)[0, tok])


def test_replay_rows_match_transformers_warpers_row_by_row():
    g = torch.Generator().manual_seed(0)
    logits = torch.randn(40, 50, generator=g) * 3
    logits[0, :10] = 2.0  # ties at the top
    logits[1] = 0.0       # all tied
    toks = torch.randint(0, 50, (40,), generator=g)
    toks[2] = int(logits[2].argmax())
    got = ea.replay_prob_rows(logits, toks, temperature=0.6, top_p=0.95)
    for i in range(40):
        assert abs(float(got[i]) - _warped_prob(logits[i], int(toks[i]))) < 1e-6, i
    assert float(got[2]) > 0


def test_replay_keeps_the_token_that_crosses_top_p():
    # tempered probs about 0.62 / 0.38: the runner-up crosses 0.95 and is sampleable.
    # lens.sampling_probability scores it 0 (the defect found in handoff 10).
    raw = torch.tensor([0.57, 0.43]).log()
    row = torch.full((8,), -30.0)
    row[:2] = raw
    p = float(ea.replay_prob_rows(row[None], torch.tensor([1]), temperature=0.6, top_p=0.95)[0])
    assert p > 0.3 and abs(p - _warped_prob(row, 1)) < 1e-6
    assert sampling_probability(row, 1, temperature=0.6, top_p=0.95) == 0.0


def test_entropy_nats_temperature_one():
    row = torch.zeros(1, 8)
    assert abs(float(ea.entropy_rows(row)[0]) - np.log(8)) < 1e-6


def test_npz_bytes_deterministic_and_loadable(tmp_path):
    arrays = {"hidden": np.arange(24, dtype=np.float16).reshape(2, 3, 4),
              "t": np.array([3, 9], dtype=np.int32),
              "entropy": np.ones(5, dtype=np.float32),
              "prompt_entropy": np.zeros(2, dtype=np.float32)}
    a, b = ea.npz_bytes(arrays), ea.npz_bytes(arrays)
    assert a == b
    p = tmp_path / "x.npz"
    ea.write_atomic(p, a)
    back = ea.read_npz(p)
    assert sorted(back) == sorted(arrays)
    for k in arrays:
        assert back[k].dtype == arrays[k].dtype and np.array_equal(back[k], arrays[k])


def test_load_positions_reads_trace_index_and_t_only(tmp_path):
    p = tmp_path / "e.csv.gz"
    buf = io.StringIO()
    w = csv.writer(buf)
    # other columns hold values the loader could not parse if it touched them
    w.writerow(["distance_set", "split", "label", "trace_index", "t", "kind"])
    w.writerows([["?", "?", "?", 5, 9, "?"], ["?", "?", "?", 5, 3, "?"],
                 ["?", "?", "?", 5, 9, "?"], ["?", "?", "?", 2, 0, "?"]])
    with gzip.open(p, "wt", newline="") as fh:
        fh.write(buf.getvalue())
    assert ea.load_positions(p) == {2: [0], 5: [3, 9]}


def test_verify_choice_from_metadata_only():
    pos = {1: [0], 4: [0, 1, 2], 6: [0], 9: [0, 1]}
    assert ea.verify_choice(pos) == [1, 6, 4]
