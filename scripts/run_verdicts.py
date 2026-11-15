"""Step 13: verdicts, probe vs combined rival on the test split, per band (§9, §10, §11, §14).

Scores the four frozen probes (probes/) and the four frozen rivals (rival/). Nothing is
chosen, fitted or tuned here except the registered shuffled-labels control (§9, V4).
Pins D22 (V1-V6); plan handoffs/13-verdicts-plan.md as approved.

    .venv/bin/python scripts/run_verdicts.py --rehearse   # validation only; no test row read
    .venv/bin/python scripts/run_verdicts.py --run        # control, then verdicts, no pause
    .venv/bin/python scripts/run_verdicts.py --run --out activations/verdicts_rerun

--rehearse reads validation band rows only and must reproduce the eight validation
AUROCs and their G4 intervals; it records a pass with this file's sha256. --run refuses
to start without that record, or if this file changed since. --run stage A loads train
and test band rows and the chosen probe layer only, and runs the shuffled-labels
control; if any band is outside 0.5 +/- 0.05 it stops before any real test score and
before any rival test feature is built. Stage B scores probe and rival on test, runs the
10,000 paired resamples over the 102 test problems, and writes the three result files.
Gate rows are never kept. No per-example test score is written to disk. Prints no AUROC.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import os
import random
import sys
import time
import warnings
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ["HF_HOME"] = str(REPO_ROOT / ".hf_cache")
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402

import extract_activations as ea  # noqa: E402  (write_atomic)
import run_gate as rg  # noqa: E402  (sha256_file, auroc, load_manifest, EXAMPLES_SHA256)
import run_probes as rp  # noqa: E402  (memmap_npz_member, zstats, apply_z, probe_scores, EXPECTED)
import run_rival as rr  # noqa: E402  (build_contexts, Band.docs, score_saved, load_cell)

EXAMPLES_PATH = REPO_ROOT / "examples" / "examples.csv.gz"
ACT_DIR = REPO_ROOT / "activations" / "run"
SPLIT_JSON = REPO_ROOT / "results" / "split.json"
PROBE_CSV = REPO_ROOT / "results" / "probe_layers.csv"
RIVAL_CSV = REPO_ROOT / "results" / "rival_grid.csv"
STATE_DIR = REPO_ROOT / "activations" / "verdicts"        # gitignored via activations/
REHEARSAL = STATE_DIR / "rehearsal.json"
CONTROL_STOP = STATE_DIR / "control_stop.json"
JOBLIB_TMP = REPO_ROOT / "activations" / "joblib_tmp"
RESULTS = REPO_ROOT / "results"
SELF = Path(__file__).resolve()

BANDS = ["2-5", "6-10", "11-20", "21-40"]
SEED = 20261019                      # V2, V4 (the registered split seed)
N_BOOT = 10_000                      # §10, §18
PCT = (0.625, 99.375)                # V1: 98.75% percentile interval
G4_N, G4_PCT = 2000, (2.5, 97.5)     # the validation intervals of 11b / 12 (Q5)
DELTA = 0.05                         # §11
SECONDARY = (0.03, 0.08)             # §11, secondary
M_MIN = 555                          # §11
MAX_ITER = 10_000
CONTACT_DATE = "2026-11-15"
MAX_CSV_BYTES = 1_000_000            # CLAUDE.md (Q8)

PROBE = {b: REPO_ROOT / "probes" / f"band_{b}.npz" for b in BANDS}
RIVAL = {b: REPO_ROOT / "rival" / f"band_{b}.npz" for b in BANDS}
PROBE_SHA = {  # results/probes.md table d
    "2-5": "02bcbe5f8184c1301f822b08c3afc939d05b2bc832e438a54f189c3a8719ccea",
    "6-10": "bf9bd8708a4abcd6ff9a0643288cd86341d70dab9955b21cd9950b9a16133dfa",
    "11-20": "95cff87fc495e4547dfd21c166539aeb96558e261aa00b0253bd69dbbe863739",
    "21-40": "02075f7c4f62e34fbcfe7ec7d4585efbaf6e21130eff994fa72a54fa6f1164b9",
}
RIVAL_SHA = {  # results/rival.md table e
    "2-5": "89bd053af2d5ab068a4037a14b916f9d25a46aa0486d4d6673c87ea4e045552b",
    "6-10": "111c5ec5bd5f8426e5fc9b76a943a259c4594411d6e0c7ea8b97290bd981af56",
    "11-20": "3f1a617acf8e05ef87c808cf5647232b4c0e8b565dd2760724ee45de3fd5dd06",
    "21-40": "4ea2e1b88940f84872cc3e1d8c92207076c9941f631838332c310c487bd743c2",
}
# Validation AUROC and G4 95% interval as printed (probes.md d, rival.md e).
PROBE_VAL = {"2-5": ("0.9177", "0.9031", "0.9313"), "6-10": ("0.8527", "0.8283", "0.8754"),
             "11-20": ("0.7732", "0.7410", "0.8031"), "21-40": ("0.6968", "0.6562", "0.7361")}
RIVAL_VAL = {"2-5": ("0.8993", "0.8824", "0.9150"), "6-10": ("0.8350", "0.8095", "0.8603"),
             "11-20": ("0.7645", "0.7345", "0.7963"), "21-40": ("0.6824", "0.6529", "0.7136")}
# Test counts: examples_summary.md table c (positives, negatives) and table d (onsets).
TEST_COUNTS = {"2-5": (3191, 6382), "6-10": (3861, 7722), "11-20": (7453, 14906),
               "21-40": (13907, 27814)}
TEST_ONSETS = {"2-5": 794, "6-10": 762, "11-20": 726, "21-40": 661}


# ---------------------------------------------------------------- loading

def load_rows(path: Path, splits: set[str]) -> dict[str, list[dict]]:
    """Band rows of the named splits only, per band, in file order. Gate rows and rows of
    other splits are dropped before any other field is parsed."""
    out: dict[str, list[dict]] = {b: [] for b in BANDS}
    with gzip.open(path, "rt", newline="") as fh:
        for r in csv.DictReader(fh):
            if r["distance_set"] not in BANDS or r["split"] not in splits:
                continue
            out[r["distance_set"]].append({
                "split": r["split"], "label": int(r["label"]),
                "trace_index": int(r["trace_index"]), "t": int(r["t"])})
    for rows in out.values():
        assert all(r["split"] in splits for r in rows)
    return out


def split_problems(split: str) -> list[int]:
    return sorted(json.loads(SPLIT_JSON.read_text())[split])


def gather_layer(rows: list[dict], layer: int, manifest: dict[int, str],
                 verified: set[int]) -> np.ndarray:
    """[n_rows, 1536] float16, one saved layer (1-28) only, in row order."""
    X = np.empty((len(rows), rp.HIDDEN), dtype=np.float16)
    by_trace: dict[int, list[int]] = {}
    for i, r in enumerate(rows):
        by_trace.setdefault(r["trace_index"], []).append(i)
    for tr, idxs in sorted(by_trace.items()):
        path = ACT_DIR / f"gsm8k_{tr:04d}.npz"
        if tr not in verified:
            assert rg.sha256_file(path) == manifest[tr], f"sha256 mismatch {path.name}"
            verified.add(tr)
        with np.load(path, allow_pickle=False) as z:
            ts = z["t"]
        want = np.array([rows[i]["t"] for i in idxs])
        pos = np.searchsorted(ts, want)
        assert np.all(ts[pos] == want), tr
        hidden = rp.memmap_npz_member(path, "hidden")
        assert hidden.shape[1] == 28
        X[idxs] = np.asarray(hidden[pos, layer - 1, :])  # this layer only
        del hidden
    return X


def labels(rows) -> np.ndarray:
    return np.array([r["label"] for r in rows])


def problems(rows) -> np.ndarray:
    return np.array([r["trace_index"] for r in rows])


# ---------------------------------------------------------------- frozen objects

def probe_meta(band: str) -> dict:
    with np.load(PROBE[band], allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def rival_config(band: str) -> dict:
    with np.load(RIVAL[band], allow_pickle=False) as z:
        w = int(z["window"][0])
        return {"window": "all" if w == 0 else w, "n": int(z["ngram_max"][0]),
                "C": float(z["C"][0])}


def probe_scores_for(band: str, X: np.ndarray) -> np.ndarray:
    """Frozen probe's decision function on its own layer's states X."""
    from threadpoolctl import threadpool_limits

    z = probe_meta(band)
    with threadpool_limits(limits=1):
        return rp.probe_scores(X, z["mean"], z["sd"], z["coef"], z["intercept"][0])


class Examples:
    """Duck-types the Band that rr.score_saved reads: every row is scored."""
    docs = rr.Band.docs

    def __init__(self, rows, ctx):
        self.rows, self.ctx = rows, ctx
        self.va = np.arange(len(rows))


def rival_scores_for(band: str, rows, ctx) -> np.ndarray:
    return rr.score_saved(RIVAL[band], Examples(rows, ctx))  # file-only rebuild, as in 12


def frozen_hashes() -> dict[str, str]:
    h = {f"probe {b}": rg.sha256_file(PROBE[b]) for b in BANDS}
    h.update({f"rival {b}": rg.sha256_file(RIVAL[b]) for b in BANDS})
    return h


def check_frozen() -> dict[str, str]:
    h = frozen_hashes()
    for b in BANDS:
        assert h[f"probe {b}"] == PROBE_SHA[b], f"probe {b} sha256 mismatch"
        assert h[f"rival {b}"] == RIVAL_SHA[b], f"rival {b} sha256 mismatch"
    assert rg.sha256_file(EXAMPLES_PATH) == rg.EXAMPLES_SHA256
    return h


# ---------------------------------------------------------------- statistics (V1-V3, V6)

def resample_draws(n_problems: int, n: int = N_BOOT, seed: int = SEED) -> np.ndarray:
    """[n, n_problems] indices into the sorted problem list. Same generator and draw order
    as G4 (rg.bootstrap_by_problem). Depends only on the seed and the problem count."""
    rng = random.Random(seed)
    return np.array([[rng.randrange(n_problems) for _ in range(n_problems)] for _ in range(n)],
                    dtype=np.int64)


def paired_resamples(y, s_p, s_r, problem, keys, draws):
    """Whole problems with replacement; both scores on the same resampled examples (V3).
    A key with no examples contributes none. A single-class resample is dropped (NaN)."""
    y, s_p, s_r, problem = map(np.asarray, (y, s_p, s_r, problem))
    assert set(problem.tolist()) <= set(keys)
    members = [np.flatnonzero(problem == k) for k in keys]
    a_p = np.full(len(draws), np.nan)
    a_r = np.full(len(draws), np.nan)
    for b, d in enumerate(draws):
        idx = np.concatenate([members[j] for j in d])
        yy = y[idx]
        if len(yy) == 0 or yy.min() == yy.max():
            continue
        a_p[b] = rg.auroc(yy, s_p[idx])
        a_r[b] = rg.auroc(yy, s_r[idx])
    return a_p, a_r


def interval(values, pct=PCT) -> tuple[float, float]:
    v = np.asarray(values)
    lo, hi = np.percentile(v[~np.isnan(v)], list(pct))  # numpy default "linear"
    return float(lo), float(hi)


def verdict(lo: float, hi: float, delta: float) -> str:
    """§11 / V6, on unrounded values."""
    pos = lo > 0 and hi >= delta
    null = hi < delta
    inc = lo <= 0 and hi >= delta
    assert pos + null + inc == 1
    return "positive" if pos else ("null" if null else "inconclusive")


def control_passes(a: float) -> bool:
    return 0.45 <= a <= 0.55  # §9: 0.5 ± 0.05, inclusive; literal bounds, no float subtraction


def near_chance(a: float) -> bool:
    return 0.45 <= a <= 0.55  # §14: within 0.05 of 0.5, inclusive


# ---------------------------------------------------------------- shuffled-labels control (V4)

def shuffled_control(band, Xtr, ytr, Xte, yte) -> dict:
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from threadpoolctl import threadpool_limits

    z = probe_meta(band)
    C = float(z["C"][0])
    with threadpool_limits(limits=1):
        mu, sd = rp.zstats(Xtr)
        same = bool(np.array_equal(mu, z["mean"]) and np.array_equal(sd, z["sd"]))
        y_perm = ytr[np.random.default_rng(SEED).permutation(len(ytr))]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", ConvergenceWarning)
            m = LogisticRegression(penalty="l2", C=C, solver="lbfgs", max_iter=MAX_ITER,
                                   fit_intercept=True)
            m.fit(rp.apply_z(Xtr, mu, sd), y_perm)
        warned = any(issubclass(x.category, ConvergenceWarning) for x in w)
        n_iter = int(m.n_iter_[0])
        s = rp.probe_scores(Xte, mu, sd, m.coef_[0].astype(np.float64), float(m.intercept_[0]))
    return {"band": band, "auroc": rg.auroc(yte, s), "same_standardization": same,
            "converged": (not warned) and n_iter < MAX_ITER, "n_iter": n_iter,
            "permuted_changed": int((y_perm != ytr).sum())}


# ---------------------------------------------------------------- rehearsal

def csv_value(path: Path, match: dict) -> str:
    with open(path, newline="") as fh:
        hits = [r["validation_auroc"] for r in csv.DictReader(fh)
                if all((float(r[k]) == v) if isinstance(v, float) else (r[k] == str(v))
                       for k, v in match.items())]
    assert len(hits) == 1, (path.name, match, hits)
    return hits[0]


def rehearse() -> None:
    from transformers import AutoTokenizer

    from before_the_wait.generate import MODEL_ID, MODEL_REVISION

    t0 = time.perf_counter()
    check_frozen()
    rows_by_band = load_rows(EXAMPLES_PATH, {"validation"})
    for b in BANDS:
        y = labels(rows_by_band[b])
        assert (int((y == 1).sum()), int((y == 0).sum())) == rp.EXPECTED[b]["validation"], b
    keys = split_problems("validation")
    draws = resample_draws(len(keys), n=G4_N)
    manifest = rg.load_manifest()
    verified: set[int] = set()
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    ctx = rr.build_contexts([r for b in BANDS for r in rows_by_band[b]], tok, manifest)
    checks, ok = [], True
    for b in BANDS:
        rows = rows_by_band[b]
        y, prob = labels(rows), problems(rows)
        z = probe_meta(b)
        L, Cp = int(z["layer"][0]), float(z["C"][0])
        s_p = probe_scores_for(b, gather_layer(rows, L, manifest, verified))
        s_r = rival_scores_for(b, rows, ctx)
        a_p, a_r = rg.auroc(y, s_p), rg.auroc(y, s_r)
        bp, br = paired_resamples(y, s_p, s_r, prob, keys, draws)
        cfg = rival_config(b)
        cell = rr.load_cell(b, cfg["window"], cfg["n"])
        cached = [f["auroc"] for f in cell["fits"] if f["C"] == cfg["C"]]
        assert len(cached) == 1
        for who, a, boot, md, csv6, exact in (
                ("probe", a_p, bp, PROBE_VAL[b],
                 csv_value(PROBE_CSV, {"band": b, "layer": L, "C": Cp}), None),
                ("rival", a_r, br, RIVAL_VAL[b],
                 csv_value(RIVAL_CSV, {"band": b, "window": cfg["window"],
                                       "ngram_range": f"1-{cfg['n']}", "C": cfg["C"]}),
                 cached[0])):
            lo, hi = interval(boot, G4_PCT)
            c = {"band": b, "who": who, "auroc": a, "md": md[0], "csv": csv6,
                 "md_ok": f"{a:.4f}" == md[0], "csv_ok": f"{a:.6f}" == csv6,
                 "exact_ok": True if exact is None else (a == exact),
                 "interval": [lo, hi], "interval_ok": (f"{lo:.4f}", f"{hi:.4f}") == md[1:],
                 "dropped": int(np.isnan(boot).sum())}
            c["pass"] = c["md_ok"] and c["csv_ok"] and c["exact_ok"] and c["interval_ok"]
            ok &= c["pass"]
            checks.append(c)
    assert frozen_hashes() == {**{f"probe {b}": PROBE_SHA[b] for b in BANDS},
                               **{f"rival {b}": RIVAL_SHA[b] for b in BANDS}}
    rec = {"pass": bool(ok), "script_sha256": rg.sha256_file(SELF), "checks": checks,
           "seconds": round(time.perf_counter() - t0, 1)}
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    ea.write_atomic(REHEARSAL, (json.dumps(rec, indent=1, sort_keys=True) + "\n").encode())
    print("rehearsal", "PASS" if ok else "FAIL", f"({rec['seconds']}s)", flush=True)
    if not ok:
        sys.exit(1)


# ---------------------------------------------------------------- run

def require_rehearsal() -> dict:
    assert REHEARSAL.exists(), "no rehearsal record: run --rehearse first"
    rec = json.loads(REHEARSAL.read_text())
    assert rec["pass"] is True, "rehearsal did not pass"
    assert rec["script_sha256"] == rg.sha256_file(SELF), "script changed since the rehearsal"
    return rec


def _band_boot(y, s_p, s_r, prob, keys, draws):
    a_p, a_r = paired_resamples(y, s_p, s_r, prob, keys, draws)
    return a_p - a_r


def run(out_dir: Path) -> None:
    from joblib import Parallel, delayed
    from transformers import AutoTokenizer

    from before_the_wait.generate import MODEL_ID, MODEL_REVISION

    t0 = time.perf_counter()
    rec = require_rehearsal()
    check_frozen()
    keys = split_problems("test")
    assert len(keys) == 102
    draws = resample_draws(len(keys))                       # before any test row is read
    draws_sha = hashlib.sha256(draws.tobytes()).hexdigest()
    JOBLIB_TMP.mkdir(parents=True, exist_ok=True)
    manifest = rg.load_manifest()
    verified: set[int] = set()

    # ---- stage A: shuffled-labels control (train rows, test rows, chosen layer only)
    print("stage A: shuffled-labels control", flush=True)
    rows_by_band = load_rows(EXAMPLES_PATH, {"train", "test"})
    tr = {b: [r for r in rows_by_band[b] if r["split"] == "train"] for b in BANDS}
    te = {b: [r for r in rows_by_band[b] if r["split"] == "test"] for b in BANDS}
    for b in BANDS:
        ytr, yte = labels(tr[b]), labels(te[b])
        assert (int((ytr == 1).sum()), int((ytr == 0).sum())) == rp.EXPECTED[b]["train"], b
        assert (int((yte == 1).sum()), int((yte == 0).sum())) == TEST_COUNTS[b], b
        assert set(problems(te[b]).tolist()) <= set(keys), b
    layer = {b: int(probe_meta(b)["layer"][0]) for b in BANDS}
    Xtr = {b: gather_layer(tr[b], layer[b], manifest, verified) for b in BANDS}
    Xte = {b: gather_layer(te[b], layer[b], manifest, verified) for b in BANDS}
    ctrl = Parallel(n_jobs=4, backend="loky", temp_folder=str(JOBLIB_TMP))(
        delayed(shuffled_control)(b, Xtr[b], labels(tr[b]), Xte[b], labels(te[b]))
        for b in BANDS)
    ctrl = {c["band"]: c for c in ctrl}
    del Xtr
    for b in BANDS:
        assert ctrl[b]["same_standardization"], b
    if not all(control_passes(ctrl[b]["auroc"]) for b in BANDS):
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        ea.write_atomic(CONTROL_STOP, (json.dumps(ctrl, indent=1, sort_keys=True) + "\n").encode())
        print("stage A: control FAIL; stopped before any real test score "
              f"(values in {CONTROL_STOP.relative_to(REPO_ROOT)})", flush=True)
        sys.exit(2)
    print("stage A: control pass (all four bands)", flush=True)

    # ---- stage B: probe and rival on test, paired bootstrap
    print("stage B: verdicts", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    ctx = rr.build_contexts([r for b in BANDS for r in te[b]], tok, manifest)
    res = {}
    for b in BANDS:
        y = labels(te[b])
        s_p = probe_scores_for(b, Xte[b])
        s_r = rival_scores_for(b, te[b], ctx)
        res[b] = {"y": y, "s_p": s_p, "s_r": s_r, "prob": problems(te[b])}
    lifts = Parallel(n_jobs=4, backend="loky", temp_folder=str(JOBLIB_TMP))(
        delayed(_band_boot)(res[b]["y"], res[b]["s_p"], res[b]["s_r"], res[b]["prob"],
                            keys, draws) for b in BANDS)
    rows_out = []
    for b, L in zip(BANDS, lifts):
        r = res[b]
        p, rv = rg.auroc(r["y"], r["s_p"]), rg.auroc(r["y"], r["s_r"])
        lo, hi = interval(L)
        rows_out.append({
            "band": b, "n_problems": len(set(r["prob"].tolist())), "onsets": TEST_ONSETS[b],
            "positives": int((r["y"] == 1).sum()), "negatives": int((r["y"] == 0).sum()),
            "probe_auroc": p, "rival_auroc": rv, "lift": p - rv,
            "interval_low": lo, "interval_high": hi,
            "verdict_0.05": verdict(lo, hi, DELTA),
            "verdict_0.03": verdict(lo, hi, SECONDARY[0]),
            "verdict_0.08": verdict(lo, hi, SECONDARY[1]),
            "shuffled_labels_auroc": ctrl[b]["auroc"],
            "both_near_chance": near_chance(p) and near_chance(rv),
            "dropped_resamples": int(np.isnan(L).sum()),
            "_lifts": L})
    after = frozen_hashes()
    assert after == {**{f"probe {b}": PROBE_SHA[b] for b in BANDS},
                     **{f"rival {b}": RIVAL_SHA[b] for b in BANDS}}, "frozen files changed"
    write_outputs(out_dir, rows_out, ctrl, rec, draws_sha, layer)
    print(f"done; outputs in {out_dir.relative_to(REPO_ROOT)} "
          f"({time.perf_counter() - t0:.0f}s)", flush=True)


# ---------------------------------------------------------------- outputs

CSV_COLS = ["band", "n_problems", "onsets", "positives", "negatives", "probe_auroc",
            "rival_auroc", "lift", "interval_low", "interval_high", "verdict_0.05",
            "verdict_0.03", "verdict_0.08", "shuffled_labels_auroc", "both_near_chance",
            "dropped_resamples"]


def fmt(v) -> str:
    return repr(float(v)) if isinstance(v, (float, np.floating)) else str(v)


def verdicts_csv(rows_out) -> bytes:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(CSV_COLS)
    for r in rows_out:
        w.writerow([fmt(r[c]) for c in CSV_COLS])
    return buf.getvalue().encode()


def lifts_csv(rows_out) -> bytes:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["resample"] + [f"lift_{r['band']}" for r in rows_out])
    for k in range(N_BOOT):
        w.writerow([k] + ["" if np.isnan(r["_lifts"][k]) else repr(float(r["_lifts"][k]))
                          for r in rows_out])
    data = buf.getvalue().encode()
    assert len(data) < MAX_CSV_BYTES, f"bootstrap_lifts.csv would be {len(data)} bytes"
    return data


def verdicts_md(rows_out, ctrl, rec, draws_sha, layer) -> bytes:
    import scipy
    import sklearn

    f4 = lambda v: f"{v:.4f}"  # noqa: E731
    L = ["# Verdicts: probe vs combined rival on test, per band", "",
         "Computed by `scripts/run_verdicts.py`. §9, §10, §11, §14; D22 (V1–V6); plan "
         "`handoffs/13-verdicts-plan.md` as approved. Printed to 4 decimals; "
         "`results/verdicts.csv` holds the unrounded values.", "",
         "## a. Data contact", "",
         "| Date | What was touched | What was not |", "|---|---|---|",
         f"| {CONTACT_DATE} | The test split of the four bands (2-5, 6-10, 11-20, 21-40): labels, "
         "activations at each band's chosen probe layer and at layer 28, entropy and prompt "
         "entropy, token ids and paragraph-rule onsets. First and only scoring. Train rows of "
         "the four bands (labels and chosen layer) for the shuffled-labels control; validation "
         "rows for the rehearsal. | The gate rows; any other layer; any exploratory analysis "
         "(§13); the hand audit and add-on (§12). |", "",
         "## b. Frozen objects scored", "",
         "| band | probe layer | probe C | probe sha256 | rival window | rival n-grams | rival C "
         "| rival sha256 |", "|---|---|---|---|---|---|---|---|"]
    for b in BANDS:
        cfg = rival_config(b)
        L.append(f"| {b} | {layer[b]} | {float(probe_meta(b)['C'][0]):g} | `{PROBE_SHA[b]}` | "
                 f"{cfg['window']} | 1-{cfg['n']} | {cfg['C']:g} | `{RIVAL_SHA[b]}` |")
    L += ["", "Both sets of files were byte-identical (sha256) before and after the run.", "",
          "## c. Rehearsal on validation", "",
          "The scoring path of this script, run on the validation examples before any test row "
          "was read (`--rehearse`). Reproduced = equal to the printed table value (4 decimals), "
          "to the grid CSV (6 decimals) and, for rivals, exactly equal as a float to the "
          "unrounded value in the step-12 cell cache; and the G4 95% interval (2,000 resamples, "
          "same draws) reproduced to 4 decimals.", "",
          "| band | reader | validation AUROC | table | grid CSV | 95% interval | reproduced |",
          "|---|---|---|---|---|---|---|"]
    for c in rec["checks"]:
        L.append(f"| {c['band']} | {c['who']} | {f4(c['auroc'])} | {c['md']} | {c['csv']} | "
                 f"{f4(c['interval'][0])}–{f4(c['interval'][1])} | "
                 f"{'yes' if c['pass'] else 'NO'} |")
    L += ["", "## d. Shuffled-labels control (§9)", "",
          "Train labels permuted (numpy `default_rng(20261019).permutation`, per band); probe "
          "refitted at the chosen layer and C with the frozen probe's standardization (train "
          "mean and SD, recomputed and identical); scored on the band's test examples with "
          "their true labels. Pass: 0.45 ≤ AUROC ≤ 0.55.", "",
          "| band | AUROC | converged | iterations | pass |", "|---|---|---|---|---|"]
    for b in BANDS:
        c = ctrl[b]
        L.append(f"| {b} | {f4(c['auroc'])} | {c['converged']} | {c['n_iter']} | "
                 f"{'pass' if control_passes(c['auroc']) else 'FAIL'} |")
    L += ["", "## e. Verdicts at δ = 0.05", "",
          "Lift = probe AUROC − rival AUROC on the band's test examples. Interval: 98.75%, "
          "paired bootstrap over whole test problems, 10,000 resamples (V1–V3).", "",
          "| band | probe AUROC | rival AUROC | lift | δ | 98.75% interval | verdict |",
          "|---|---|---|---|---|---|---|"]
    for r in rows_out:
        L.append(f"| {r['band']} | {f4(r['probe_auroc'])} | {f4(r['rival_auroc'])} | "
                 f"{r['lift']:+.4f} | {DELTA} | {r['interval_low']:+.4f} to "
                 f"{r['interval_high']:+.4f} | {r['verdict_0.05']} |")
    pos = [r["band"] for r in rows_out if r["verdict_0.05"] == "positive"]
    L += ["", f"Bands with a positive verdict at δ = 0.05 (§11, Claim): "
          f"{', '.join(pos) if pos else 'none'}.", "",
          "## f. Secondary verdicts (secondary; never the headline)", "",
          "| band | verdict at δ = 0.03 | verdict at δ = 0.08 |", "|---|---|---|"]
    for r in rows_out:
        L.append(f"| {r['band']} | {r['verdict_0.03']} | {r['verdict_0.08']} |")
    L += ["", "## g. Counts and the both-near-chance flag", "",
          "Onsets: clean test onsets from `results/examples_summary.md` table d. Problems: test "
          "problems with at least one example in the band (of 102 resampled).", "",
          "| band | onsets | positives | negatives | problems | ≥ m = 555 | both within 0.05 "
          "of 0.5 (§14) |", "|---|---|---|---|---|---|---|"]
    for r in rows_out:
        L.append(f"| {r['band']} | {r['onsets']} | {r['positives']} | {r['negatives']} | "
                 f"{r['n_problems']} | {'yes' if r['onsets'] >= M_MIN else 'no'} | "
                 f"{'yes' if r['both_near_chance'] else 'no'} |")
    below = [r["band"] for r in rows_out if r["onsets"] < M_MIN]
    L += ["", ("All four bands are at or above m = 555 test onsets, so §11's below-minimum "
               "clause does not apply." if not below else
               f"Below m = 555: {', '.join(below)} (§11: positive or inconclusive only)."), "",
          "## h. V1–V6 as implemented", "",
          "- V1: interval = `np.percentile(lifts, [0.625, 99.375])` (linear), dropped resamples "
          "excluded.",
          "- V2: `random.Random(20261019)`, fresh per band, `randrange(102)` × 102 per resample "
          "over the sorted test problem ids of `results/split.json` (the G4 draw order). The "
          "10,000 × 102 draw matrix was built before any test row was read; all bands share it. "
          f"sha256 (int64 bytes): `{draws_sha}`.",
          "- V3: an example's problem is its own trace (`trace_index`), so a cross-trace "
          "negative travels with its own trace. A drawn problem with no examples in the band "
          "contributes none (Q2 ruling). Probe and rival are scored on the same resampled "
          "indices. Dropped (single-class) resamples per band: "
          + ", ".join(f"{r['band']} {r['dropped_resamples']}" for r in rows_out) + ".",
          "- V4: see d. Permutation `np.random.default_rng(20261019).permutation(n_train)` over "
          "the band's train rows in `examples.csv.gz` order; same solver settings as the probes "
          "(L2, lbfgs, max_iter 10,000, single-threaded BLAS).",
          "- V5: `--rehearse` (validation) passed and recorded this script's sha256; `--run` "
          "checked it, ran the control on all four bands, and computed no real test score "
          "until all four passed.",
          "- V6: verdicts on unrounded values: positive = lower > 0 and upper ≥ δ; null = upper "
          "< δ; inconclusive = lower ≤ 0 and upper ≥ δ (exhaustive, asserted).",
          f"- numpy {np.__version__}, scikit-learn {sklearn.__version__}, scipy "
          f"{scipy.__version__}.", ""]
    return "\n".join(L).encode()


def write_outputs(out_dir: Path, rows_out, ctrl, rec, draws_sha, layer) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {"verdicts.csv": verdicts_csv(rows_out),
             "bootstrap_lifts.csv": lifts_csv(rows_out),
             "verdicts.md": verdicts_md(rows_out, ctrl, rec, draws_sha, layer)}
    for name, data in files.items():
        ea.write_atomic(out_dir / name, data)


def main() -> None:
    if "--rehearse" in sys.argv:
        rehearse()
    elif "--run" in sys.argv:
        out = RESULTS
        if "--out" in sys.argv:
            out = (REPO_ROOT / sys.argv[sys.argv.index("--out") + 1]).resolve()
            assert (REPO_ROOT / "activations") in out.parents, "--out must be under activations/"
        run(out)
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
