"""Tests for scripts/run_rival.py (handoff 12): loader boundary, layer-28-only read,
causal-only features, train-only fitting, paragraph-break rule, ties, save/reload.
Fixtures only; no real data, no model, no tokenizer."""
from __future__ import annotations

import csv
import gzip
import hashlib
import importlib.util
import io
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
_spec = importlib.util.spec_from_file_location("run_rival", REPO / "scripts" / "run_rival.py")
rr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rr)
import extract_activations as ea  # noqa: E402

HEADER = ["distance_set", "split", "label", "trace_index", "t", "kind", "onset", "d",
          "pos_trace_index", "pos_t", "same_trace"]


# ---------------------------------------------------------------- boundaries

def test_loader_drops_test_and_gate_rows(tmp_path):
    p = tmp_path / "e.csv.gz"
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(HEADER)
    w.writerows([["2-5", "train", 1, 1, 10, "word", 13, 3, "", "", ""],
                 ["2-5", "test", 1, 3, 30, "word", 33, 3, "", "", ""],
                 ["gate", "validation", 1, 2, 40, "word", 41, 1, "", "", ""],
                 ["21-40", "validation", 0, 2, 20, "word", "", "", 2, 50, True]])
    with gzip.open(p, "wt", newline="") as fh:
        fh.write(buf.getvalue())
    out = rr.rp.load_band_rows(p)  # the 11b loader, reused (plan Q1)
    assert [(r["split"], r["t"]) for b in rr.BANDS for r in out[b]] == [
        ("train", 10), ("validation", 20)]


def test_only_layer_28_is_read(tmp_path):
    hidden = np.full((2, 28, rr.HIDDEN), np.nan, dtype=np.float16)  # layers 1-27: NaN
    hidden[:, 27, :] = np.array([[1.0], [2.0]], dtype=np.float16)
    data = ea.npz_bytes({"hidden": hidden, "t": np.array([3, 8], np.int32)})
    (tmp_path / "x.npz").write_bytes(data)
    mm = rr.rp.memmap_npz_member(tmp_path / "x.npz", "hidden")
    out = np.asarray(mm[np.array([1, 0]), rr.OUT_LAYER_INDEX, :])
    assert np.isfinite(out).all() and out[0, 0] == 2 and out[1, 0] == 1


def test_no_probe_output_path_in_script():
    src = (REPO / "scripts" / "run_rival.py").read_text()
    for forbidden in ("probes.md", "probe_layers", '"probes"', "probes/"):
        assert forbidden not in src, forbidden


# ---------------------------------------------------------------- features

def _ctx(ids, P, breaks, onsets, entropy, prompt_entropy, out_rows):
    return rr.TraceCtx(ids, P, breaks, onsets, np.asarray(entropy, np.float32),
                       np.asarray(prompt_entropy, np.float32), out_rows)


def _fixture():
    P = 20
    n = 60
    ids = list(range(1000, 1000 + P + n))
    ent = np.linspace(0.1, 3.0, n)
    pent = np.linspace(5.0, 6.0, P)
    out = {t: np.full(rr.HIDDEN, t, np.float16) for t in range(n)}
    return ids, P, [5, 30, 70], [12, 40], ent, pent, out


def _features(ctx, t):
    return ctx.dense(t), [ctx.doc(t, w) for w in rr.WINDOWS]


def test_causal_only_changes_after_t_change_nothing():
    ids, P, breaks, onsets, ent, pent, out = _fixture()
    t = 33
    i = P + t
    base = _features(_ctx(ids, P, breaks, onsets, ent, pent, out), t)
    ids2 = ids[:i + 1] + [7] * (len(ids) - i - 1)          # every token after t
    ent2 = np.concatenate([ent[:t + 1], np.full(len(ent) - t - 1, 99.0)])  # entropy after t
    breaks2 = [b for b in breaks if b <= i] + [i + 1, i + 5]  # breaks after t
    onsets2 = [o for o in onsets if o < t] + [t + 1, t + 9]   # onsets after t
    out2 = dict(out)
    for k in range(t + 1, 60):
        out2[k] = np.full(rr.HIDDEN, -1, np.float16)
    new = _features(_ctx(ids2, P, breaks2, onsets2, ent2, pent, out2), t)
    assert np.array_equal(base[0], new[0]) and base[1] == new[1]
    # and the other direction: a token before t does matter
    ids3 = list(ids)
    ids3[i - 1] = 5
    assert _features(_ctx(ids3, P, breaks, onsets, ent, pent, out), t)[1] != base[1]


def test_entropy_window_excludes_t_and_reaches_prompt():
    ids, P, breaks, onsets, ent, pent, out = _fixture()
    e_t, m, mx = rr.entropy_feats(np.asarray(ent, np.float32), np.asarray(pent, np.float32), 3, P)
    past = list(pent[P - 13:]) + list(ent[:3])  # t-16 ... t-1 with t = 3
    assert len(past) == 16
    assert abs(e_t - ent[3]) < 1e-6 and abs(m - np.mean(past)) < 1e-5 and abs(mx - max(past)) < 1e-6


def test_position_features():
    P = 20
    assert rr.since_onset([12, 40], 33, P) == 21
    assert rr.since_onset([12, 40], 12, P) == P + 12      # strictly before t
    assert rr.since_onset([], 5, P) == P + 5
    assert rr.since_break([5, 30, 70], P + 33, P, 33) == P + 33 - 30
    assert rr.since_break([5, 53], 53, P, 33) == 0          # break ends in token t
    assert rr.since_break([], P + 3, P, 3) == P + 3


def test_text_windows():
    ids = list(range(100))
    assert rr.text_doc(ids, 10, 8) == " ".join(map(str, range(3, 11)))
    assert rr.text_doc(ids, 3, 8) == "0 1 2 3"
    assert rr.text_doc(ids, 10, "all") == " ".join(map(str, range(11)))


def test_break_tokens_rule():
    # tokens: "a" ".\n\n" "b" "\n" "\n" "c" "\n\n\n" "d"
    toks = ["a", ".\n\n", "b", "\n", "\n", "c", "\n\n\n", "d"]
    text = "".join(toks)
    offs = [0]
    for s in toks:
        offs.append(offs[-1] + len(s))
    assert rr.break_tokens(text, offs) == [1, 4, 6]
    assert rr.break_tokens("abc", [0, 1, 2, 3]) == []


# ---------------------------------------------------------------- fitting

def _toy_band(seed=0, val_shift=0.0):
    rng = np.random.default_rng(seed)
    rows, ctx = [], {}
    for tr, split in ((1, "train"), (2, "train"), (3, "validation")):
        P, n = 10, 80
        ids = list(rng.integers(0, 30, size=P + n))
        ent = rng.random(n) + (val_shift if split == "validation" else 0)
        out = {t: (rng.normal(size=rr.HIDDEN) + (val_shift if split == "validation" else 0)
                   ).astype(np.float16) for t in range(n)}
        ctx[tr] = _ctx(ids, P, [15, 50], [30], ent, rng.random(P), out)
        for t in range(20, 70, 2):
            rows.append({"split": split, "label": int(t % 3 == 0), "trace_index": tr, "t": t})
    return rows, ctx


def test_train_only_fitting_ignores_validation():
    rows, ctx = _toy_band()
    rows2, ctx2 = _toy_band()
    # replace validation entirely: different ids, entropy and states
    _, ctxv = _toy_band(seed=9, val_shift=5.0)
    ctx2[3] = ctxv[3]
    B1, B2 = rr.Band("2-5", rows, ctx), rr.Band("2-5", rows2, ctx2)
    assert np.array_equal(B1.mu, B2.mu) and np.array_equal(B1.sd, B2.sd) and B1.r == B2.r
    v1, T1, _ = B1.text(8, 2)
    v2, T2, _ = B2.text(8, 2)
    assert v1.vocabulary_ == v2.vocabulary_ and np.array_equal(v1.idf_, v2.idf_)
    assert rr.text_r(T1) == rr.text_r(T2)


def test_ties_and_nonconverged_eligible():
    rs = [{"auroc": 0.8, "window": "all", "n": 2, "C": 1e-4, "converged": True},
          {"auroc": 0.8, "window": 16, "n": 3, "C": 1e-4, "converged": True},
          {"auroc": 0.8, "window": 16, "n": 2, "C": 1e-2, "converged": True},
          {"auroc": 0.8, "window": 16, "n": 2, "C": 1e-3, "converged": False}]
    assert rr.chosen_key(rs[3]) == min(map(rr.chosen_key, rs))  # not dropped for non-convergence


def test_save_reload_reproduces_scores(tmp_path):
    rows, ctx = _toy_band()
    B = rr.Band("2-5", rows, ctx)
    res = rr.run_cell(B, 16, 2, tmp_path / "jl", n_jobs=1, c_values=[1e-2])
    best = {**res["fits"][0], "window": 16, "n": 2}
    vec, _, _ = B.text(16, 2)
    a = ea.npz_bytes(rr.rival_arrays(B, vec, best, res))
    assert a == ea.npz_bytes(rr.rival_arrays(B, vec, best, res))
    p = tmp_path / "band.npz"
    p.write_bytes(a)
    s = rr.score_saved(p, B)
    assert rr.rg.auroc(B.y[B.va], s) == best["auroc"]
    assert hashlib.sha256(p.read_bytes()).hexdigest() == hashlib.sha256(a).hexdigest()
