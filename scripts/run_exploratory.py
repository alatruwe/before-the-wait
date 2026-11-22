"""Step 14: the registered exploratory analyses (pre-registration §13, D25). EXPLORATORY.

No verdicts. Nothing here is chosen or tuned on test; every setting was fixed on
validation or is pinned in D25 (X1-X8). No frozen object (probes/, rival/, rival_7b/) is
refitted or written; their hashes are checked at the start and end of every mode.
Plan: handoffs/14-exploratory-plan.md as approved.

    .venv/bin/python scripts/run_exploratory.py --a6      # accuracy (run_traces.csv only)
    .venv/bin/python scripts/run_exploratory.py --a5      # paragraphs (labels and text only)
    .venv/bin/python scripts/run_exploratory.py --a7      # same-trace negatives (test)
    .venv/bin/python scripts/run_exploratory.py --a1      # exact-distance curve (test)
    .venv/bin/python scripts/run_exploratory.py --a2-time-one
    .venv/bin/python scripts/run_exploratory.py --a2      # groups alone (validation, then test)
    .venv/bin/python scripts/plot_exploratory_traces.py   # analysis 3
    .venv/bin/python scripts/run_exploratory.py --summary # summary.md, manifest.csv

Outputs under results/exploratory/; scratch under activations/exploratory/ (gitignored).
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import re
import shutil
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402

import extract_activations as ea  # noqa: E402  (sets HF_HOME; npz_bytes, write_atomic)
import run_gate as rg  # noqa: E402
import run_probes as rp  # noqa: E402
import run_rival as rr  # noqa: E402
import run_verdicts as rv  # noqa: E402

EXAMPLES_PATH = REPO_ROOT / "examples" / "examples.csv.gz"
OUT = REPO_ROOT / "results" / "exploratory"
STATE = REPO_ROOT / "activations" / "exploratory"          # gitignored via activations/
A2_CACHE = STATE / "a2_cache"
A2_READERS = STATE / "a2_readers"
JOBLIB_TMP = REPO_ROOT / "activations" / "joblib_tmp"
VERDICTS_CSV = REPO_ROOT / "results" / "verdicts.csv"
PROBE_CSV = REPO_ROOT / "results" / "probe_layers.csv"
GATE_CSV = REPO_ROOT / "results" / "gate_layers.csv"
RUN_TRACES = REPO_ROOT / "results" / "run_traces.csv"
SPLIT_JSON = REPO_ROOT / "results" / "split.json"
ONSETS_PATH = REPO_ROOT / "results" / "run_onsets_paragraph.csv"
TRACES_DIR = REPO_ROOT / "traces" / "run"

BANDS = rp.BANDS
SPLITS = ("train", "validation", "test")
LAYERS = rp.PROBE_LAYERS
C_GRID = rp.C_GRID
SEAMS = (5.5, 10.5, 20.5)
BAND_RANGE = {"2-5": (2, 5), "6-10": (6, 10), "11-20": (11, 20), "21-40": (21, 40)}
DRAWS_SHA = "1c267812f3760abd36a060fbb2ae72dd3c6fbb671b45e690dccca684ee72716b"  # D22, step 13
READER_7B = REPO_ROOT / "rival_7b" / "band_6-10.npz"
READER_7B_SHA = "a2ac7a37eb54f731fcb20635d9f400eb483129b24258da19a8befd849783e850"
CROSS_TRACE_TEST = {"2-5": 28, "6-10": 68, "11-20": 641, "21-40": 2254}  # examples_summary e
GROUPS = ("text", "entropy", "output", "position")
DENSE = {g: (lo, hi) for g, lo, hi in rr.DENSE_GROUPS}
N_JOBS = 7
CONTACT_DATE = "2026-11-21"


# ---------------------------------------------------------------- frozen objects

def frozen_expected() -> dict[str, str]:
    h = {f"probes/band_{b}.npz": rv.PROBE_SHA[b] for b in BANDS}
    h.update({f"rival/band_{b}.npz": rv.RIVAL_SHA[b] for b in BANDS})
    h["rival_7b/band_6-10.npz"] = READER_7B_SHA
    h["examples/examples.csv.gz"] = rg.EXAMPLES_SHA256
    return h


def frozen_check(mode: str, when: str) -> dict[str, str]:
    actual = {k: rg.sha256_file(REPO_ROOT / k) for k in frozen_expected()}
    STATE.mkdir(parents=True, exist_ok=True)
    rec = {"mode": mode, "when": when, "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "match": actual == frozen_expected(), "sha256": actual}
    with open(STATE / "frozen_checks.jsonl", "a") as fh:
        fh.write(json.dumps(rec, sort_keys=True) + "\n")
    assert rec["match"], f"frozen object changed ({mode}, {when})"
    return actual


# ---------------------------------------------------------------- rows

def load_rows(splits, sets) -> dict[str, list[dict]]:
    """Rows of the named distance sets and splits, per set, in file order. Other rows are
    dropped before any other field is parsed."""
    out: dict[str, list[dict]] = {s: [] for s in sets}
    with gzip.open(EXAMPLES_PATH, "rt", newline="") as fh:
        for r in csv.DictReader(fh):
            if r["distance_set"] not in out or r["split"] not in splits:
                continue
            out[r["distance_set"]].append({
                "split": r["split"], "label": int(r["label"]),
                "trace_index": int(r["trace_index"]), "t": int(r["t"]),
                "onset": int(r["onset"]) if r["onset"] else None,
                "d": int(r["d"]) if r["d"] else None,
                "pos_trace_index": int(r["pos_trace_index"]) if r["pos_trace_index"] else None,
                "pos_t": int(r["pos_t"]) if r["pos_t"] else None,
                "same_trace": r["same_trace"]})
    for rows in out.values():
        assert all(r["split"] in splits for r in rows)
    return out


def exact_distance(rows: list[dict]) -> np.ndarray:
    """d per row: a positive's own d; a negative gets the d of the positive it was matched
    to (pos_trace_index, pos_t, same split). Raises on an orphan or ambiguous positive."""
    pos: dict[tuple, int] = {}
    for r in rows:
        if r["label"] == 1:
            key = (r["split"], r["trace_index"], r["t"])
            assert key not in pos, f"duplicate positive {key}"
            assert r["d"] is not None
            pos[key] = r["d"]
    out = []
    for r in rows:
        if r["label"] == 1:
            out.append(r["d"])
        else:
            key = (r["split"], r["pos_trace_index"], r["pos_t"])
            assert key in pos, f"orphan negative {key}"
            out.append(pos[key])
    return np.array(out, dtype=np.int64)


def same_trace_mask(rows: list[dict]) -> np.ndarray:
    """True for every positive and every negative drawn from its positive's own trace."""
    m = []
    for r in rows:
        if r["label"] == 1:
            m.append(True)
        else:
            assert r["same_trace"] in ("True", "False"), r["same_trace"]
            m.append(r["same_trace"] == "True")
    return np.array(m, dtype=bool)


def verdicts_row(band: str) -> dict[str, str]:
    with open(VERDICTS_CSV, newline="") as fh:
        for r in csv.DictReader(fh):
            if r["band"] == band:
                return r
    raise KeyError(band)


def quoted(band: str) -> dict[str, str]:
    """Confirmatory numbers quoted from results/verdicts.csv (no verdict column)."""
    r = verdicts_row(band)
    return {k: r[k] for k in ("probe_auroc", "rival_auroc", "lift", "interval_low",
                              "interval_high")}


def test_keys_and_draws() -> tuple[list[int], np.ndarray]:
    keys = rv.split_problems("test")
    assert len(keys) == 102
    draws = rv.resample_draws(len(keys))
    assert hashlib.sha256(draws.tobytes()).hexdigest() == DRAWS_SHA, "draws differ from D22"
    return keys, draws


def lift_interval(y, s_p, s_r, prob, keys, draws) -> dict:
    a_p, a_r = rv.paired_resamples(y, s_p, s_r, prob, keys, draws)
    lo, hi = rv.interval(a_p - a_r)
    return {"probe_auroc": rg.auroc(y, s_p), "other_auroc": rg.auroc(y, s_r),
            "lift": rg.auroc(y, s_p) - rg.auroc(y, s_r), "interval_low": lo,
            "interval_high": hi, "dropped": int(np.isnan(a_p).sum())}


def tokenizer():
    from transformers import AutoTokenizer

    from before_the_wait.generate import MODEL_ID, MODEL_REVISION

    return AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)


def write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)


def save_record(name: str, rec: dict) -> None:
    STATE.mkdir(parents=True, exist_ok=True)
    ea.write_atomic(STATE / f"{name}.json",
                    (json.dumps(rec, indent=1, sort_keys=True, default=float) + "\n").encode())


def f6(x) -> str:
    return "" if x is None else f"{x:.6f}"


# ---------------------------------------------------------------- analysis 6

def a6() -> None:
    split = json.loads(SPLIT_JSON.read_text())
    where = {i: s for s in SPLITS for i in split[s]}
    rows = []
    with open(RUN_TRACES, newline="") as fh:
        for r in csv.DictReader(fh):
            if r["looped"] == "False":
                rows.append(r)
    assert len(rows) == 292 and all(int(r["index"]) in where for r in rows)
    out = []
    for name in ("all",) + SPLITS:
        g = [r for r in rows if name == "all" or where[int(r["index"])] == name]
        n_ok = sum(r["correct"] == "True" for r in g)
        cap = sum(r["finish_reason"] == "length" for r in g)
        out.append([name, len(g), n_ok, f"{n_ok / len(g):.4f}", cap])
    write_csv(OUT / "a6_accuracy.csv", ["split", "kept_traces", "correct", "accuracy",
                                        "at_length_cap"], out)
    save_record("a6", {"rows": out})
    print("a6 done", out, flush=True)


# ---------------------------------------------------------------- analysis 5

def paragraph_gap(output_text: str, c_t: int, c_onset: int) -> int:
    """Blank lines between the start of token t and the onset's first character (D11 rule:
    paragraphs are the pieces between "\\n\\n")."""
    assert c_t <= c_onset, (c_t, c_onset)
    return output_text[c_t:c_onset].count("\n\n")


def a5() -> None:
    from before_the_wait.align import prefix_lengths

    rows_by = load_rows(SPLITS, BANDS)
    onset_char: dict[tuple[int, int], int] = {}
    with open(ONSETS_PATH, newline="") as fh:
        for r in csv.DictReader(fh):
            if r["merged_into"] == "":
                onset_char[(int(r["index"]), int(r["token_index"]))] = int(r["char_start"])
    tok = tokenizer()
    want = sorted({r["trace_index"] for rows in rows_by.values() for r in rows})
    ctx = {}
    t0 = time.perf_counter()
    for tr in want:
        trace = json.loads((TRACES_DIR / f"gsm8k_{tr:04d}.json").read_text())
        out_ids = list(trace["output_token_ids"])
        P = len(trace["prompt_token_ids"])
        ids = list(trace["prompt_token_ids"]) + out_ids
        offs = prefix_lengths(tok, ids)
        text = tok.decode(ids, skip_special_tokens=False)
        assert len(text) == offs[-1], tr
        out_offs = prefix_lengths(tok, out_ids)
        assert tok.decode(out_ids, skip_special_tokens=False) == trace["output_text"], tr
        ctx[tr] = {"P": P, "breaks": rr.break_tokens(text, offs), "out_offs": out_offs,
                   "text": trace["output_text"]}
    print(f"a5: aligned {len(ctx)} traces in {time.perf_counter() - t0:.0f}s", flush=True)
    out = []
    for b in BANDS:
        for sp in ("all",) + SPLITS:
            rows = [r for r in rows_by[b] if sp == "all" or r["split"] == sp]
            gaps, sb_pos, sb_neg = [], [], []
            for r in rows:
                c = ctx[r["trace_index"]]
                sb = rr.since_break(c["breaks"], c["P"] + r["t"], c["P"], r["t"])
                if r["label"] == 1:
                    co = onset_char[(r["trace_index"], r["onset"])]
                    gaps.append(paragraph_gap(c["text"], c["out_offs"][r["t"]], co))
                    sb_pos.append(sb)
                else:
                    sb_neg.append(sb)
            g = np.array(gaps)
            n = len(g)
            own, prev, early = int((g == 0).sum()), int((g == 1).sum()), int((g >= 2).sum())
            out.append([b, sp, n, own, f"{own / n:.4f}", prev, f"{prev / n:.4f}", early,
                        f"{early / n:.4f}", f"{np.median(sb_pos):g}", len(sb_neg),
                        f"{np.median(sb_neg):g}"])
    write_csv(OUT / "a5_paragraphs.csv",
              ["band", "split", "n_pos", "own_paragraph", "own_share", "previous_paragraph",
               "previous_share", "earlier", "earlier_share", "median_since_break_pos",
               "n_neg", "median_since_break_neg"], out)
    save_record("a5", {"rows": out})
    print("a5 done", flush=True)


# ---------------------------------------------------------------- shared test scoring

def frozen_test_scores(rows_by: dict[str, list[dict]]):
    """Frozen probe and frozen rival decision functions on each band's rows."""
    manifest = rg.load_manifest()
    verified: set[int] = set()
    tok = tokenizer()
    ctx = rr.build_contexts([r for b in BANDS for r in rows_by[b]], tok, manifest)
    res = {}
    for b in BANDS:
        rows = rows_by[b]
        L = int(rv.probe_meta(b)["layer"][0])
        s_p = rv.probe_scores_for(b, rv.gather_layer(rows, L, manifest, verified))
        s_r = rv.rival_scores_for(b, rows, ctx)
        res[b] = {"y": rv.labels(rows), "s_p": s_p, "s_r": s_r, "prob": rv.problems(rows)}
    return res, ctx


def check_confirmatory(b: str, y, s_p, s_r) -> None:
    """Wiring: the full test set reproduces verdicts.csv exactly (not written as a number)."""
    q = verdicts_row(b)
    a_p, a_r = rg.auroc(y, s_p), rg.auroc(y, s_r)
    assert repr(a_p) == q["probe_auroc"], (b, repr(a_p), q["probe_auroc"])
    assert repr(a_r) == q["rival_auroc"], (b, repr(a_r), q["rival_auroc"])
    assert repr(a_p - a_r) == q["lift"], (b, repr(a_p - a_r), q["lift"])


# ---------------------------------------------------------------- analysis 7

def a7() -> None:
    from joblib import Parallel, delayed

    keys, draws = test_keys_and_draws()
    rows_by = load_rows({"test"}, BANDS)
    res, _ = frozen_test_scores(rows_by)
    jobs, meta = [], {}
    for b in BANDS:
        r = res[b]
        check_confirmatory(b, r["y"], r["s_p"], r["s_r"])
        m = same_trace_mask(rows_by[b])
        removed = int((~m).sum())
        assert removed == CROSS_TRACE_TEST[b], (b, removed)
        meta[b] = {"removed": removed, "n_pos": int((r["y"][m] == 1).sum()),
                   "n_neg": int((r["y"][m] == 0).sum()),
                   "n_problems": len(set(r["prob"][m].tolist()))}
        jobs.append((b, r["y"][m], r["s_p"][m], r["s_r"][m], r["prob"][m]))
    JOBLIB_TMP.mkdir(parents=True, exist_ok=True)
    outs = Parallel(n_jobs=4, backend="loky", temp_folder=str(JOBLIB_TMP))(
        delayed(lift_interval)(y, sp, sr, pr, keys, draws) for _, y, sp, sr, pr in jobs)
    shutil.rmtree(JOBLIB_TMP, ignore_errors=True)
    rows = []
    for (b, *_), o in zip(jobs, outs):
        q = quoted(b)
        rows.append([b, meta[b]["removed"], meta[b]["n_pos"], meta[b]["n_neg"],
                     meta[b]["n_problems"], f6(o["probe_auroc"]), f6(o["other_auroc"]),
                     f6(o["lift"]), f6(o["interval_low"]), f6(o["interval_high"]),
                     o["dropped"], q["lift"], q["interval_low"], q["interval_high"]])
    write_csv(OUT / "a7_same_trace.csv",
              ["band", "negatives_removed", "n_pos", "n_neg", "n_problems", "probe_auroc",
               "rival_auroc", "lift", "interval_low_98.75", "interval_high_98.75",
               "dropped_resamples", "confirmatory_lift_quoted",
               "confirmatory_interval_low_quoted", "confirmatory_interval_high_quoted"], rows)
    save_record("a7", {"meta": meta, "draws_sha256": DRAWS_SHA,
                       "full_set_reproduces_verdicts_csv": True,
                       "results": {b: o for (b, *_), o in zip(jobs, outs)}})
    print("a7 done", flush=True)


# ---------------------------------------------------------------- analysis 1

def best_c_per_layer(path: Path, band: str | None) -> dict[int, dict]:
    """Validation-best C per layer from a committed layer CSV; ties at the printed six
    decimals are returned so the caller can refit them (none expected)."""
    by: dict[int, list[dict]] = {}
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            if band is not None and r["band"] != band:
                continue
            by.setdefault(int(r["layer"]), []).append(
                {"C": float(r["C"]), "auroc_str": r["validation_auroc"],
                 "auroc": float(r["validation_auroc"]), "converged": r["converged"]})
    out = {}
    for L, rs in by.items():
        assert len(rs) == 5
        top = max(r["auroc"] for r in rs)
        tied = sorted(r["C"] for r in rs if r["auroc"] == top)
        best = min(rs, key=lambda r: (-r["auroc"], r["C"]))
        out[L] = {**best, "tied": tied}
    return out


def fit_single(kind: str, Xtr, ytr, Xva, yva, Xte, C: float) -> dict:
    """One fit at one C with the band recipe (rp: zstats, apply_z, float64 scores) or the
    gate recipe (rg: standardize, decision_function), single-threaded."""
    import warnings

    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=1):
        if kind == "band":
            mu, sd = rp.zstats(Xtr)
            A = rp.apply_z(Xtr, mu, sd)
        else:
            A, B, T = rg.standardize(Xtr.astype(np.float32), Xva.astype(np.float32),
                                     Xte.astype(np.float32))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", ConvergenceWarning)
            m = LogisticRegression(penalty="l2", C=C, solver="lbfgs", max_iter=rp.MAX_ITER,
                                   fit_intercept=True)
            m.fit(A, ytr)
        warned = any(issubclass(x.category, ConvergenceWarning) for x in w)
        coef, b0 = m.coef_[0].astype(np.float64), float(m.intercept_[0])
        if kind == "band":
            s_va = rp.probe_scores(Xva, mu, sd, coef, b0)
            s_te = rp.probe_scores(Xte, mu, sd, coef, b0)
        else:
            s_va, s_te = m.decision_function(B), m.decision_function(T)
    n_iter = int(m.n_iter_[0])
    return {"C": C, "val_auroc": rg.auroc(yva, s_va), "s_te": s_te, "coef": coef,
            "intercept": b0, "converged": (not warned) and n_iter < rp.MAX_ITER,
            "n_iter": n_iter}


def cell_aurocs(y, s, d, dists) -> dict[int, tuple]:
    out = {}
    for k in dists:
        m = d == k
        yy = y[m]
        out[k] = (rg.auroc(yy, s[m]) if 0 < yy.sum() < len(yy) else None,
                  int((yy == 1).sum()), int((yy == 0).sum()))
    return out


def a1_set(name, kind, rows, best, manifest, verified):
    from joblib import Parallel, delayed

    t0 = time.perf_counter()
    X = rp.gather_layers_1_27(rows, rp.ACT_DIR, verified, manifest)
    split = np.array([r["split"] for r in rows])
    y = rv.labels(rows)
    tr, va, te = split == "train", split == "validation", split == "test"
    print(f"[a1 {name}] gathered {X.shape} in {time.perf_counter() - t0:.0f}s", flush=True)
    jobs = []
    for L in LAYERS:
        for C in best[L]["tied"]:
            jobs.append((L, C))
    JOBLIB_TMP.mkdir(parents=True, exist_ok=True)
    t1 = time.perf_counter()
    fits = Parallel(n_jobs=N_JOBS, backend="loky", temp_folder=str(JOBLIB_TMP))(
        delayed(fit_single)(kind, X[L - 1][tr], y[tr], X[L - 1][va], y[va], X[L - 1][te], C)
        for L, C in jobs)
    shutil.rmtree(JOBLIB_TMP, ignore_errors=True)
    print(f"[a1 {name}] {len(fits)} refits in {time.perf_counter() - t1:.0f}s", flush=True)
    del X
    by_layer: dict[int, dict] = {}
    refit_rows = []
    for (L, C), f in zip(jobs, fits):
        f["layer"] = L
        csv_ok = (f"{f['val_auroc']:.6f}" == best[L]["auroc_str"]) if C == best[L]["C"] else None
        refit_rows.append([name, L, f"{C:g}", f6(f["val_auroc"]), csv_ok, f["converged"],
                           f["n_iter"], len(best[L]["tied"]) > 1])
        cur = by_layer.get(L)
        if cur is None or (-f["val_auroc"], f["C"]) < (-cur["val_auroc"], cur["C"]):
            by_layer[L] = f
    return by_layer, refit_rows, y[te], np.array([r["trace_index"] for r in rows])[te], \
        [r for r, k in zip(rows, te) if k]


def a1() -> None:
    import matplotlib
    matplotlib.use("Agg")

    t_start = time.perf_counter()
    manifest = rg.load_manifest()
    verified: set[int] = set()
    all_rows = load_rows(SPLITS, BANDS + ["gate"])
    refits, cells, wiring = [], [], {}
    # gate (d = 1), no rival
    gbest = best_c_per_layer(GATE_CSV, None)
    g_by, g_ref, g_y, _, g_te_rows = a1_set("gate", "gate", all_rows["gate"], gbest, manifest,
                                           verified)
    refits += g_ref
    g_d = exact_distance(g_te_rows)
    assert set(g_d.tolist()) == {1}
    for L in LAYERS:
        f = g_by[L]
        a = rg.auroc(g_y, f["s_te"])
        cells.append(["gate", L, f"{f['C']:g}", 1, int((g_y == 1).sum()), int((g_y == 0).sum()),
                      len({r["trace_index"] for r in g_te_rows}), f6(a), "", ""])
    gate_choice = min(LAYERS, key=lambda L: (-g_by[L]["val_auroc"], L, g_by[L]["C"]))
    # band rival scores on the test rows (frozen, file-only rebuild)
    te_rows = {b: [r for r in all_rows[b] if r["split"] == "test"] for b in BANDS}
    tok = tokenizer()
    ctx = rr.build_contexts([r for b in BANDS for r in te_rows[b]], tok, manifest)
    curve = {"gate": {1: None}}
    for b in BANDS:
        best = best_c_per_layer(PROBE_CSV, b)
        by, ref, y, prob, rows_te = a1_set(b, "band", all_rows[b], best, manifest, verified)
        refits += ref
        assert rows_te == te_rows[b]
        d = exact_distance(rows_te)
        lo, hi = BAND_RANGE[b]
        dists = list(range(lo, hi + 1))
        assert set(d.tolist()) == set(dists), (b, sorted(set(d.tolist())))
        s_r = rv.rival_scores_for(b, rows_te, ctx)
        rc = cell_aurocs(y, s_r, d, dists)
        # wiring: the refit at the frozen layer and C is the frozen probe
        z = rv.probe_meta(b)
        Lf, Cf = int(z["layer"][0]), float(z["C"][0])
        f = by[Lf]
        assert f["C"] == Cf, (b, f["C"], Cf)
        same = bool(np.array_equal(f["coef"], z["coef"]) and f["intercept"] == float(z["intercept"][0]))
        a_all = rg.auroc(y, f["s_te"])
        q = verdicts_row(b)
        wiring[b] = {"layer": Lf, "C": Cf, "coef_identical": same,
                     "test_auroc_repr": repr(a_all), "verdicts_csv": q["probe_auroc"],
                     "equal": repr(a_all) == q["probe_auroc"],
                     "rival_equal": repr(rg.auroc(y, s_r)) == q["rival_auroc"]}
        assert same and wiring[b]["equal"] and wiring[b]["rival_equal"], wiring[b]
        for L in LAYERS:
            pc = cell_aurocs(y, by[L]["s_te"], d, dists)
            for k in dists:
                n_prob = len(set(prob[d == k].tolist()))
                lift = pc[k][0] - rc[k][0]
                cells.append([b, L, f"{by[L]['C']:g}", k, pc[k][1], pc[k][2], n_prob,
                              f6(pc[k][0]), f6(rc[k][0]), f6(lift)])
        curve[b] = {"layer": Lf}
        print(f"[a1 {b}] done", flush=True)
    assert all(r[4] is not False for r in refits), [r for r in refits if r[4] is False]
    write_csv(OUT / "a1_refits.csv", ["set", "layer", "C", "validation_auroc",
                                      "matches_committed_csv", "converged", "n_iter",
                                      "tied_at_six_decimals"], refits)
    write_csv(OUT / "a1_exact_distance.csv",
              ["set", "layer", "C", "d", "n_pos", "n_neg", "n_problems", "probe_auroc",
               "rival_auroc", "lift"], cells)
    a1_figures(cells, gate_choice, {b: curve[b]["layer"] for b in BANDS})
    save_record("a1", {"wiring": wiring, "n_refits": len(refits),
                       "not_converged": sum(not r[5] for r in refits),
                       "ties": sum(bool(r[7]) for r in refits),
                       "gate_layer_for_curve": gate_choice,
                       "gate_C_for_curve": g_by[gate_choice]["C"],
                       "seconds": round(time.perf_counter() - t_start, 1)})
    print(f"a1 done in {time.perf_counter() - t_start:.0f}s", flush=True)


def a1_figures(cells, gate_layer: int, frozen_layer: dict[str, int]) -> None:
    import matplotlib.pyplot as plt

    for b in BANDS:
        lo, hi = BAND_RANGE[b]
        dists = list(range(lo, hi + 1))
        A = np.full((len(LAYERS), len(dists)), np.nan)
        Lf = np.full_like(A, np.nan)
        for r in cells:
            if r[0] == b:
                A[r[1] - 1, r[3] - lo] = float(r[7])
                Lf[r[1] - 1, r[3] - lo] = float(r[9])
        fig, axes = plt.subplots(1, 2, figsize=(4 + 0.35 * len(dists) * 2, 6.2), dpi=110)
        for ax, M, title, cmap, kw in (
                (axes[0], A, "probe AUROC", "viridis", {}),
                (axes[1], Lf, "lift over the frozen rival", "RdBu_r",
                 {"vmin": -np.nanmax(np.abs(Lf)), "vmax": np.nanmax(np.abs(Lf))})):
            im = ax.imshow(M, aspect="auto", origin="lower", cmap=cmap,
                           extent=(lo - 0.5, hi + 0.5, 0.5, 27.5), **kw)
            ax.axhline(frozen_layer[b], color="k", lw=0.6, ls=":")
            ax.set_xlabel("exact distance d (tokens before the onset)")
            ax.set_ylabel("layer")
            ax.set_title(title, fontsize=10)
            fig.colorbar(im, ax=ax, fraction=0.046)
        fig.suptitle(f"Exploratory: band {b} probes per layer, test split, by exact distance\n"
                     f"(dotted: frozen layer {frozen_layer[b]}; no verdict)", fontsize=10)
        fig.tight_layout()
        fig.savefig(OUT / f"a1_heatmap_{b}.png")
        plt.close(fig)
    # curve across d = 1..40
    fig, ax = plt.subplots(figsize=(9, 4.6), dpi=110)
    colors = {"gate": "#555555", "2-5": "#1f77b4", "6-10": "#d62728", "11-20": "#2ca02c",
              "21-40": "#9467bd"}
    for name, L in [("gate", gate_layer)] + [(b, frozen_layer[b]) for b in BANDS]:
        pts = sorted((r[3], float(r[7]), r[8]) for r in cells if r[0] == name and r[1] == L)
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-", ms=3, color=colors[name],
                label=f"{name} probe (layer {L})")
        if name != "gate":
            ax.plot([p[0] for p in pts], [float(p[2]) for p in pts], "--", color=colors[name],
                    lw=1, label=f"{name} frozen rival")
    for s in SEAMS:
        ax.axvline(s, color="k", lw=0.6, ls=":")
    ax.set_xlabel("exact distance d")
    ax.set_ylabel("AUROC (test split)")
    ax.set_title("Exploratory: AUROC by exact distance. Dotted lines are band seams, where one "
                 "probe hands over to the next;\na jump at a seam may be the handover. No verdict.",
                 fontsize=9)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "a1_curve.png")
    plt.close(fig)


# ---------------------------------------------------------------- analysis 2

def group_block(group: str, T, Z: np.ndarray, f_text: float, f: dict[str, float]):
    """The group's own columns of rr.assemble, with D21's multiplier (Q2 ruling)."""
    import scipy.sparse as sp

    if group == "text":
        return sp.csr_matrix(T * f_text, dtype=np.float64)
    lo, hi = DENSE[group]
    return Z[:, lo:hi] * f[group]


def text_key(r: dict):
    return rr.chosen_key(r)


def dense_key(r: dict):
    return (-r["auroc"], r["C"])


def at_edge(C: float) -> bool:
    return C in (C_GRID[0], C_GRID[-1])


def a2_cell_path(band, window, n) -> Path:
    return A2_CACHE / f"{rr.cell_name(band, window, n)}.npz"


def a2_text_cell(B: "rr.Band", window, n, n_jobs=5, c_values=C_GRID) -> dict:
    from joblib import Parallel, delayed

    t0 = time.perf_counter()
    vec, Ttr, Tva = B.text(window, n)
    t_vec = time.perf_counter() - t0
    r_text = rr.text_r(Ttr)
    f_text, f = rr.multipliers(B.r, r_text)
    Xtr = group_block("text", Ttr, None, f_text, f)
    Xva = group_block("text", Tva, None, f_text, f)
    JOBLIB_TMP.mkdir(parents=True, exist_ok=True)
    fits = Parallel(n_jobs=min(n_jobs, len(c_values)), backend="loky",
                    temp_folder=str(JOBLIB_TMP))(
        delayed(rr.fit_one)(Xtr, B.y[B.tr], Xva, B.y[B.va], C) for C in c_values)
    shutil.rmtree(JOBLIB_TMP, ignore_errors=True)
    return {"band": B.band, "window": window, "n": n, "vocab": len(vec.vocabulary_),
            "r_text": r_text, "f_text": f_text, "vectorize_seconds": t_vec,
            "wall_seconds": time.perf_counter() - t0, "fits": fits}


def a2_save_cell(res: dict) -> None:
    arrays = {f"coef_{k}": f["coef"] for k, f in enumerate(res["fits"])}
    meta = {k: v for k, v in res.items() if k != "fits"}
    meta["fits"] = [{k: v for k, v in f.items() if k != "coef"} for f in res["fits"]]
    arrays["meta"] = np.frombuffer(json.dumps(meta, sort_keys=True).encode(), dtype=np.uint8)
    A2_CACHE.mkdir(parents=True, exist_ok=True)
    ea.write_atomic(a2_cell_path(res["band"], res["window"], res["n"]), ea.npz_bytes(arrays))


def a2_load_cell(band, window, n) -> dict | None:
    p = a2_cell_path(band, window, n)
    if not p.exists():
        return None
    with np.load(p, allow_pickle=False) as z:
        meta = json.loads(z["meta"].tobytes().decode())
        for k, f in enumerate(meta["fits"]):
            f["coef"] = z[f"coef_{k}"]
    return meta


def a2_setup_trainval():
    """Train and validation rows only, through the 11b loader, exactly as step 12."""
    assert rg.sha256_file(EXAMPLES_PATH) == rg.EXAMPLES_SHA256
    rows_by = rp.load_band_rows(EXAMPLES_PATH)
    rp.check_counts(rows_by)
    tok = tokenizer()
    manifest = rg.load_manifest()
    ctx = rr.build_contexts([r for b in BANDS for r in rows_by[b]], tok, manifest)
    return rows_by, ctx, tok, manifest


def a2_time_one() -> None:
    frozen_check("a2-time-one", "before")
    rows_by, ctx, _, _ = a2_setup_trainval()
    B = rr.Band("21-40", rows_by["21-40"], ctx)
    res = a2_text_cell(B, "all", 3, n_jobs=1, c_values=[1.0])
    f = res["fits"][0]
    one = res["vectorize_seconds"] + f["seconds"]
    # step 12: heaviest single cell fit 435.9 s + vectorize 49.9 s; whole grid 4,312.5 s
    ratio = 4312.5 / (435.9 + 49.9)
    proj = ratio * one
    rec = {"vectorize_s": res["vectorize_seconds"], "fit_s": f["seconds"], "n_iter": f["n_iter"],
           "converged": f["converged"], "vocab": res["vocab"], "ratio_from_step_12": ratio,
           "projected_grid_s": proj, "stop_line_s": 7200, "over_stop_line": proj > 7200,
           "peak_rss_gb": rr.peak_rss_gb()}
    save_record("a2_time_one", rec)
    frozen_check("a2-time-one", "after")
    print("A2 TIMING", json.dumps(rec), flush=True)


def reader_arrays(group, cfg, vec, B, f_text, f, fit) -> dict[str, np.ndarray]:
    arr = {"group": np.array([GROUPS.index(group)], np.int64),
           "C": np.array([fit["C"]], np.float64),
           "coef": np.asarray(fit["coef"], np.float64),
           "intercept": np.array([fit["intercept"]], np.float64)}
    if group == "text":
        arr.update({"window": np.array([0 if cfg["window"] == "all" else int(cfg["window"])],
                                       np.int64),
                    "ngram_max": np.array([cfg["n"]], np.int64),
                    "terms": np.array(sorted(vec.vocabulary_, key=vec.vocabulary_.get), dtype=str),
                    "idf": np.asarray(vec.idf_, np.float64),
                    "factor": np.array([f_text], np.float64)})
    else:
        lo, hi = DENSE[group]
        arr.update({"dense_mean": B.mu[lo:hi], "dense_sd": B.sd[lo:hi],
                    "factor": np.array([f[group]], np.float64)})
    return arr


def reader_scores(path: Path, rows, ctx) -> np.ndarray:
    """Rebuild a group reader from its file alone and score rows."""
    from threadpoolctl import threadpool_limits

    with np.load(path, allow_pickle=False) as z:
        group = GROUPS[int(z["group"][0])]
        if group == "text":
            w = int(z["window"][0])
            window = "all" if w == 0 else w
            terms = [str(x) for x in z["terms"]]
            vec = rr.make_vectorizer(int(z["ngram_max"][0]),
                                     vocabulary={tm: k for k, tm in enumerate(terms)})
            vec.idf_ = z["idf"]
            X = vec.transform(ctx[r["trace_index"]].doc(r["t"], window) for r in rows)
            X = X * float(z["factor"][0])
        else:
            lo, hi = DENSE[group]
            D = np.stack([ctx[r["trace_index"]].dense(r["t"])[lo:hi] for r in rows])
            X = (D - z["dense_mean"]) / z["dense_sd"] * float(z["factor"][0])
        with threadpool_limits(limits=1):
            return rr.scores(X, z["coef"], z["intercept"][0])


def a2() -> None:
    from joblib import Parallel, delayed

    t_start = time.perf_counter()
    A2_CACHE.mkdir(parents=True, exist_ok=True)
    rows_by, ctx, tok, manifest = a2_setup_trainval()
    grid, choices, bands = [], {}, {}
    for b in BANDS:
        B = rr.Band(b, rows_by[b], ctx)
        bands[b] = B
        # wiring: dense multipliers equal the frozen rival's (same train rows, same D21 code)
        _, fdense = rr.multipliers(B.r, 1.0)
        with np.load(rv.RIVAL[b], allow_pickle=False) as z:
            fac = z["factors"]
        assert [fdense["entropy"], fdense["output"], fdense["position"]] == fac[1:].tolist(), b
        cells = {}
        for window in rr.WINDOWS:
            for n in rr.NGRAMS:
                c = a2_load_cell(b, window, n)
                if c is None:
                    res = a2_text_cell(B, window, n)
                    a2_save_cell(res)
                    c = a2_load_cell(b, window, n)
                    fmax = max(f["seconds"] for f in res["fits"])
                    print(f"[a2 {rr.cell_name(b, window, n)}] vocab {res['vocab']:,} wall "
                          f"{res['wall_seconds']:.0f}s slowest fit {fmax:.0f}s", flush=True)
                else:
                    print(f"[a2 {rr.cell_name(b, window, n)}] cached", flush=True)
                cells[(window, n)] = c
        flat = [{**f, "window": w, "n": n, "vocab": c["vocab"]}
                for (w, n), c in cells.items() for f in c["fits"]]
        assert len(flat) == 60
        bt = min(flat, key=text_key)
        for f in flat:
            grid.append([b, "text", f["window"], f"1-{f['n']}", f"{f['C']:g}", f6(f["auroc"]),
                         f["converged"], f["n_iter"], f["vocab"]])
        choices[(b, "text")] = {"window": bt["window"], "n": bt["n"], "C": bt["C"],
                                "auroc": bt["auroc"], "converged": bt["converged"],
                                "coef": bt["coef"], "intercept": bt["intercept"]}
        # dense groups: C only
        JOBLIB_TMP.mkdir(parents=True, exist_ok=True)
        jobs = [(g, C) for g in GROUPS[1:] for C in C_GRID]
        fits = Parallel(n_jobs=5, backend="loky", temp_folder=str(JOBLIB_TMP))(
            delayed(rr.fit_one)(group_block(g, None, B.Ztr, 0.0, fdense), B.y[B.tr],
                                group_block(g, None, B.Zva, 0.0, fdense), B.y[B.va], C)
            for g, C in jobs)
        shutil.rmtree(JOBLIB_TMP, ignore_errors=True)
        for g in GROUPS[1:]:
            gf = [f for (gg, _), f in zip(jobs, fits) if gg == g]
            for f in gf:
                grid.append([b, g, "", "", f"{f['C']:g}", f6(f["auroc"]), f["converged"],
                             f["n_iter"], ""])
            best = min(gf, key=dense_key)
            choices[(b, g)] = {"C": best["C"], "auroc": best["auroc"],
                               "converged": best["converged"], "coef": best["coef"],
                               "intercept": best["intercept"]}
        print(f"[a2 {b}] validation choices made", flush=True)
    assert len(grid) == 300
    write_csv(OUT / "a2_grid.csv", ["band", "group", "window", "ngram_range", "C",
                                    "validation_auroc", "converged", "n_iter",
                                    "vocabulary_size"], grid)
    # save chosen readers, then the choice record, before any test row is read
    A2_READERS.mkdir(parents=True, exist_ok=True)
    rec_choice = {}
    for b in BANDS:
        B = bands[b]
        for g in GROUPS:
            ch = choices[(b, g)]
            vec, f_text = None, 0.0
            _, f = rr.multipliers(B.r, 1.0)
            if g == "text":
                cell = a2_load_cell(b, ch["window"], ch["n"])
                vec, Ttr, Tva = B.text(ch["window"], ch["n"])
                assert len(vec.vocabulary_) == cell["vocab"]
                f_text, f = rr.multipliers(B.r, cell["r_text"])
            data = ea.npz_bytes(reader_arrays(g, ch, vec, B, f_text, f,
                                              {"C": ch["C"], "coef": ch["coef"],
                                               "intercept": ch["intercept"]}))
            p = A2_READERS / f"band_{b}_{g}.npz"
            ea.write_atomic(p, data)
            # reload check on validation: file-only rebuild reproduces the chosen AUROC
            va_rows = [r for r in rows_by[b] if r["split"] == "validation"]
            s = reader_scores(p, va_rows, ctx)
            assert rg.auroc(rv.labels(va_rows), s) == ch["auroc"], (b, g)
            rec_choice[f"{b}/{g}"] = {k: v for k, v in ch.items() if k not in ("coef", "intercept")}
            rec_choice[f"{b}/{g}"]["reader_sha256"] = hashlib.sha256(data).hexdigest()
            rec_choice[f"{b}/{g}"]["reader_bytes"] = len(data)
    blob = json.dumps(rec_choice, sort_keys=True, default=str).encode()
    save_record("a2_choices", {"choices": rec_choice, "sha256": hashlib.sha256(blob).hexdigest(),
                               "written_before_test": time.strftime("%Y-%m-%dT%H:%M:%S")})
    print("a2: choices recorded; now reading test rows", flush=True)
    del bands, ctx

    # ---- test
    keys, draws = test_keys_and_draws()
    te = load_rows({"test"}, BANDS)
    res, tctx = frozen_test_scores(te)
    jobs, meta = [], []
    for b in BANDS:
        r = res[b]
        check_confirmatory(b, r["y"], r["s_p"], r["s_r"])
        for g in GROUPS:
            s_g = reader_scores(A2_READERS / f"band_{b}_{g}.npz", te[b], tctx)
            jobs.append((r["y"], r["s_p"], s_g, r["prob"]))
            meta.append((b, g))
    JOBLIB_TMP.mkdir(parents=True, exist_ok=True)
    outs = Parallel(n_jobs=6, backend="loky", temp_folder=str(JOBLIB_TMP))(
        delayed(lift_interval)(y, sp, sg, pr, keys, draws) for y, sp, sg, pr in jobs)
    shutil.rmtree(JOBLIB_TMP, ignore_errors=True)
    rows = []
    for (b, g), o in zip(meta, outs):
        ch = rec_choice[f"{b}/{g}"]
        rows.append([b, g, ch.get("window", ""), f"1-{ch['n']}" if "n" in ch else "",
                     f"{ch['C']:g}", at_edge(ch["C"]), ch["converged"], f6(ch["auroc"]),
                     f6(o["other_auroc"]), f6(o["lift"]), f6(o["interval_low"]),
                     f6(o["interval_high"]), o["dropped"], quoted(b)["rival_auroc"],
                     ch["reader_sha256"]])
    write_csv(OUT / "a2_groups.csv",
              ["band", "group", "window", "ngram_range", "C", "C_at_grid_edge", "converged",
               "validation_auroc", "test_auroc", "probe_lift", "interval_low_98.75",
               "interval_high_98.75", "dropped_resamples", "combined_rival_auroc_quoted",
               "reader_sha256"], rows)
    save_record("a2", {"n_fits": len(grid), "not_converged": sum(not r[6] for r in grid),
                       "draws_sha256": DRAWS_SHA,
                       "results": {f"{b}/{g}": o for (b, g), o in zip(meta, outs)},
                       "seconds": round(time.perf_counter() - t_start, 1)})
    print(f"a2 done in {time.perf_counter() - t_start:.0f}s", flush=True)


# ---------------------------------------------------------------- verdict words

_WORD = re.compile(r"\b(inconclusive|null|claim level|positive)\b(\s+(position|example)s?)?",
                   re.IGNORECASE)
DISCLAIMER = "No verdict is applied to any number here."


def verdict_words(text: str) -> list[str]:
    """Verdict words in our own text (Q1). 'positives', 'positive position(s)' and
    'positive example(s)' are the label's name and allowed; 'verdict' only in DISCLAIMER."""
    hits = [m.group(0) for m in _WORD.finditer(text) if not (
        m.group(1).lower() == "positive" and m.group(2))]
    rest = text.replace(DISCLAIMER, "")
    hits += re.findall(r"\bverdicts?\b(?!\.csv)", rest, flags=re.IGNORECASE)
    return hits


def html_own_text(html: str) -> str:
    """The HTML minus the quoted model and problem text (data, not our labelling)."""
    return re.sub(r"<!--quoted-->.*?<!--/quoted-->", "", html, flags=re.DOTALL)


# ---------------------------------------------------------------- summary

def read_csv(name: str) -> list[dict]:
    with open(OUT / name, newline="") as fh:
        return list(csv.DictReader(fh))


def load_record(name: str) -> dict:
    return json.loads((STATE / f"{name}.json").read_text())


def r4(x: str) -> str:
    return "" if x in ("", None) else f"{float(x):.4f}"


def signed(x: str) -> str:
    return "" if x in ("", None) else f"{float(x):+.4f}"


def summary() -> None:
    a1r, a2r, a3r, a7r = (load_record(n) for n in ("a1", "a2", "a3", "a7"))
    checks = [json.loads(x) for x in (STATE / "frozen_checks.jsonl").read_text().splitlines()]
    L: list[str] = []
    a = L.append
    a("# Exploratory analyses (pre-registration §13, D25)")
    a("")
    a("Everything in this file is **exploratory**. It feeds no registered decision rule and does "
      "not change, replace or qualify any confirmatory number; confirmatory values appear only "
      "as quotes from `results/verdicts.csv`, marked \"quoted\". Nothing was chosen or tuned on "
      "the test split: every setting was fixed on validation or pinned in D25 (X1–X8). "
      + DISCLAIMER)
    a("")
    a("Computed by `scripts/run_exploratory.py` (analyses 1, 2, 5, 6, 7) and "
      "`scripts/plot_exploratory_traces.py` (analysis 3); plan `handoffs/14-exploratory-plan.md`"
      " as approved.")
    a("")
    a("## Data contact (exploratory)")
    a("")
    a("| Date | Analysis | What was read | What was not |")
    a("|---|---|---|---|")
    rows_dc = [
        ("6", "`results/run_traces.csv` (correct, looped, finish_reason) and `results/split.json`.",
         "Any example row, activation or score."),
        ("5", "Band rows of all three splits (labels, onset, d, matched positive example); token ids "
         "and text of the 292 kept traces; paragraph-rule onsets. Tokenizer only.",
         "Any activation, score or model."),
        ("7", "Band test rows with `same_trace`; saved states at each frozen probe's layer and "
         "the rival's features (layer 28, entropy, ids, onsets) for the test traces; the frozen "
         "probes and rivals.", "Train or validation rows; any refit."),
        ("1", "Band and gate rows of train, validation and test; saved layers 1–27 at their "
         "positions; rival features for the band test rows; `probe_layers.csv`, "
         "`gate_layers.csv`. **The gate's test rows were read here for the first time** "
         "(the gate ran on validation only, §11).", "Layer 28 as a probe input; any new C."),
        ("2", "Band rows of train and validation (fits and choices), then, after the choices "
         "were recorded, band test rows; rival features; frozen probe scores on test.",
         "Any test row before the choices were recorded."),
        ("3", "`results/split.json` (test ids), 10 test trace files, the 1.5B model "
         "(teacher-forced over the stored ids), the four frozen probes, the saved activations "
         "of those 10 traces.", "Any other trace; any refit."),
    ]
    for an, what, nott in rows_dc:
        a(f"| {CONTACT_DATE} | {an} | {what} | {nott} |")
    a("")
    a("Frozen objects (`probes/`, `rival/`, `rival_7b/`, `examples/examples.csv.gz`): sha256 "
      f"checked before and after every run, {len(checks)} checks, all equal to the recorded "
      f"values: {all(c['match'] for c in checks)}.")
    a("")

    # 1
    a("## 1. Exact-distance curve")
    a("")
    a("Exploratory: AUROC of the per-layer band probes (each layer at its validation-best C, "
      "refitted on train) on the band's test examples at each exact distance d, and the lift "
      "over the band's frozen rival on the same slice; d = 1 uses the gate probes and has no "
      "lift.")
    a("")
    a(f"- Refits: {a1r['n_refits']} (27 gate, 108 band), not converged: "
      f"{a1r['not_converged']}, C tied at six decimals in the committed CSV: {a1r['ties']}. "
      "Each refit's validation AUROC equals its committed CSV row (`a1_refits.csv`).")
    a("- A negative is counted at the distance of the positive example it was matched to.")
    a("- Wiring check: at each band's frozen layer and C the refit's coefficients equal the "
      "frozen probe's, and its all-distances test AUROC equals `results/verdicts.csv` exactly:")
    a("")
    a("| band | frozen layer | C | coefficients identical | test AUROC equals verdicts.csv |")
    a("|---|---|---|---|---|")
    for b in BANDS:
        w = a1r["wiring"][b]
        a(f"| {b} | {w['layer']} | {w['C']:g} | {w['coef_identical']} | {w['equal']} |")
    a("")
    cells = read_csv("a1_exact_distance.csv")
    gl = a1r["gate_layer_for_curve"]
    a(f"At each band's frozen layer (gate: its validation-best layer {gl}, C "
      f"{a1r['gate_C_for_curve']:g}); every layer is in `a1_exact_distance.csv` and the heat "
      "maps:")
    a("")
    a("| d | set | layer | n_pos | n_neg | probe AUROC | rival AUROC | lift |")
    a("|---|---|---|---|---|---|---|---|")
    fl = {b: a1r["wiring"][b]["layer"] for b in BANDS}
    fl["gate"] = gl
    for r in sorted((r for r in cells if int(r["layer"]) == fl[r["set"]]),
                    key=lambda r: int(r["d"])):
        a(f"| {r['d']} | {r['set']} | {r['layer']} | {r['n_pos']} | {r['n_neg']} | "
          f"{r4(r['probe_auroc'])} | {r4(r['rival_auroc'])} | {signed(r['lift'])} |")
    a("")
    a("The band seams (5→6, 10→11, 20→21) are where one band's probe hands over to the next "
      "(and the rival changes with it); a jump in either curve at a seam may be the handover, "
      "not a change at that distance. Point estimates only; no interval.")
    a("")
    a("Figures: `a1_curve.png` (AUROC by d at the layers above, seams dotted) and "
      "`a1_heatmap_<band>.png` (layer × d, AUROC and lift).")
    a("")

    # 2
    g = read_csv("a2_groups.csv")
    a("## 2. Each rival group alone")
    a("")
    a("Exploratory: one reader per band and feature group (text; visible uncertainty = "
      "entropy; output state = layer 28; position), D21's recipe and scaling, chosen on "
      "validation, scored on test; the probe's lift over each with a 98.75% paired interval "
      "(D22's procedure and draws).")
    a("")
    a(f"- Fits: {a2r['n_fits']} (text: 60 per band over the registered window × n-gram × C "
      "grid, the combined rival's tie rule; entropy, output state, position: 5 C values each, "
      f"ties to the smaller C). Not converged: {a2r['not_converged']} (`a2_grid.csv`).")
    a("- Scaling (Q2 ruling): each group keeps D21's multiplier (r_out / r_g), so its columns "
      "are exactly its columns in the combined rival. Alone, the multiplier only rescales the "
      "input, and that scale interacts with C: the same C regularizes a rescaled group more or "
      "less strongly.")
    a("- Choices were written and hashed before any test row was read "
      f"(`activations/exploratory/a2_choices.json`). Draw sha256 `{a2r['draws_sha256'][:12]}…`, "
      "equal to step 13's.")
    a("")
    a("| band | group | window | n-grams | C | C at grid edge | converged | validation AUROC | "
      "test AUROC | probe lift | 98.75% interval | combined rival AUROC (confirmatory, quoted) |")
    a("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in g:
        a(f"| {r['band']} | {r['group']} | {r['window']} | {r['ngram_range']} | {r['C']} | "
          f"{r['C_at_grid_edge']} | {r['converged']} | {r4(r['validation_auroc'])} | "
          f"{r4(r['test_auroc'])} | {signed(r['probe_lift'])} | "
          f"{signed(r['interval_low_98.75'])} to {signed(r['interval_high_98.75'])} | "
          f"{r4(r['combined_rival_auroc_quoted'])} |")
    a("")

    # 3
    a("## 3. Per-trace plots")
    a("")
    a("Exploratory: the four frozen band probes' decision function at every reasoning token "
      "of 10 test traces (`random.Random(20261021).sample(sorted(test problems), 10)`), from a "
      "teacher-forced pass of the 1.5B over the stored ids, with the text alongside.")
    a("")
    a(f"- Traces, in draw order: {', '.join(str(x) for x in a3r['draw'])}.")
    a(f"- Check: at every saved example position of these traces the new score was compared "
      f"with the score from the saved activation; largest absolute difference "
      f"{a3r['max_abs_score_diff']:.3g} (tolerance {a3r['tolerance']:g}; `a3_check.csv`).")
    a("")
    a("| file | trace | reasoning tokens | saved positions compared |")
    a("|---|---|---|---|")
    for t in a3r["traces"]:
        a(f"| `traces/trace_{t['draw']:02d}.html` | {t['trace_index']} | "
          f"{t['reasoning_tokens']} | {t['positions']} |")
    a("")

    # 4
    a("## 4. The add-on (§12)")
    a("")
    a("Exploratory: not run. It needs at least 20 hits in each group; the audit found 13 real "
      "hits (`results/audit.md`), so it is not runnable (D25 X5).")
    a("")

    # 5
    a("## 5. Where each band falls relative to paragraph breaks")
    a("")
    a("Exploratory: for each band, the share of positives in the onset's own paragraph, the "
      "paragraph before it, or earlier, and the median tokens since the last paragraph break "
      "for positives and their matched negatives; all splits, labels and text only.")
    a("")
    a("- Paragraphs (Q5): the blank lines (`\\n\\n`) between the start of token t and the "
      "onset's first character, counted as D11's onset rule counts them (empty pieces "
      "included); 0 = own paragraph, 1 = previous, 2 or more = earlier. Tokens since the last "
      "break: the rival's position feature (R4, `run_rival.since_break`).")
    a("")
    a("| band | split | n_pos | own paragraph | previous | earlier | median since break, "
      "positives | median since break, negatives |")
    a("|---|---|---|---|---|---|---|---|")
    for r in read_csv("a5_paragraphs.csv"):
        a(f"| {r['band']} | {r['split']} | {r['n_pos']} | {r['own_share']} | "
          f"{r['previous_share']} | {r['earlier_share']} | {r['median_since_break_pos']} | "
          f"{r['median_since_break_neg']} |")
    a("")

    # 6
    a("## 6. Accuracy, as context")
    a("")
    a("Exploratory: accuracy of the 292 kept traces, overall and per split, with the number "
      "that hit the 4,096-token length cap; nothing is related to accuracy.")
    a("")
    a("| split | kept traces | correct | accuracy | at length cap |")
    a("|---|---|---|---|---|")
    for r in read_csv("a6_accuracy.csv"):
        a(f"| {r['split']} | {r['kept_traces']} | {r['correct']} | {r['accuracy']} | "
          f"{r['at_length_cap']} |")
    a("")

    # 7
    a("## 7. Same-trace-only negatives")
    a("")
    a("Exploratory: the frozen probe and frozen rival of each band scored on the band's test "
      "examples with every cross-trace negative removed, no refit and no rematching; lift and "
      "98.75% paired interval (D22's draws), next to the confirmatory values.")
    a("")
    a("- Removed counts equal `results/examples_summary.md` table e (test column). Before "
      "filtering, the same code reproduced `verdicts.csv`'s probe AUROC, rival AUROC and lift "
      "exactly.")
    a("")
    a("| band | negatives removed | n_pos | n_neg | probe AUROC | rival AUROC | lift | 98.75% "
      "interval | confirmatory lift (quoted) | confirmatory interval (quoted) |")
    a("|---|---|---|---|---|---|---|---|---|---|")
    for r in read_csv("a7_same_trace.csv"):
        a(f"| {r['band']} | {r['negatives_removed']} | {r['n_pos']} | {r['n_neg']} | "
          f"{r4(r['probe_auroc'])} | {r4(r['rival_auroc'])} | {signed(r['lift'])} | "
          f"{signed(r['interval_low_98.75'])} to {signed(r['interval_high_98.75'])} | "
          f"{signed(r['confirmatory_lift_quoted'])} | "
          f"{signed(r['confirmatory_interval_low_quoted'])} to "
          f"{signed(r['confirmatory_interval_high_quoted'])} |")
    a("")
    a("## Files")
    a("")
    a("All under `results/exploratory/`; sha256 and bytes in `manifest.csv`.")
    a("")
    (OUT / "summary.md").write_text("\n".join(L))

    # verdict-word scan over every output (own text only in the HTML: Q1)
    bad = {}
    for p in sorted(OUT.rglob("*")):
        if p.suffix in (".md", ".csv", ".html"):
            txt = p.read_text()
            if p.suffix == ".html":
                txt = html_own_text(txt)
            hits = verdict_words(txt)
            if hits:
                bad[str(p.relative_to(OUT))] = hits
    assert not bad, f"verdict words in outputs: {bad}"
    # manifest
    man = []
    for p in sorted(OUT.rglob("*")):
        if p.is_file() and p.name != "manifest.csv":
            assert p.stat().st_size < 1_000_000, p
            man.append([str(p.relative_to(OUT)), p.stat().st_size, rg.sha256_file(p)])
    write_csv(OUT / "manifest.csv", ["file", "bytes", "sha256"], man)
    total = sum(m[1] for m in man) + (OUT / "manifest.csv").stat().st_size
    save_record("summary", {"files": len(man) + 1, "total_bytes": total, "verdict_words": bad})
    print(f"summary written; {len(man) + 1} files, {total:,} bytes; no verdict words", flush=True)


# ---------------------------------------------------------------- main

MODES = {"--a6": a6, "--a5": a5, "--a7": a7, "--a1": a1, "--a2": a2}


def main() -> None:
    args = sys.argv[1:]
    if args == ["--a2-time-one"]:
        a2_time_one()
        return
    if args == ["--summary"]:
        summary()
        return
    assert len(args) == 1 and args[0] in MODES, f"one of {sorted(MODES)} or --summary"
    mode = args[0].lstrip("-")
    t0 = time.perf_counter()
    frozen_check(mode, "before")
    MODES[args[0]]()
    frozen_check(mode, "after")
    print(f"{mode}: frozen objects unchanged; {time.perf_counter() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
