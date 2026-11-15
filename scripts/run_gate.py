"""Step 11a: the gate at distance 1, on validation (pre-registration §11 Gate, §7, §18).

The first step that reads labels and produces a score. It reads the gate rows
(distance_set == "gate") of the train and validation splits only. Test rows and
every band row are dropped by the loader before anything else, and asserted absent;
test activations are never opened.

    .venv/bin/python scripts/run_gate.py

Writes results/gate_layers.csv (135 fits) and results/gate.md. Deterministic:
single-threaded BLAS in every fit, fixed bootstrap seed.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
import random
import sys
import time
import warnings
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ["HF_HOME"] = str(REPO_ROOT / ".hf_cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402

EXAMPLES_PATH = REPO_ROOT / "examples" / "examples.csv.gz"
EXAMPLES_SHA256 = "d05d940f4ac4396f236c5bd970b1268677fe3ba9ea8009b8835073bdb34668f1"
ACT_DIR = REPO_ROOT / "activations" / "run"
MANIFEST_PATH = REPO_ROOT / "results" / "extraction_manifest.csv"
ONSETS_PATH = REPO_ROOT / "results" / "run_onsets_paragraph.csv"
TRACES_DIR = REPO_ROOT / "traces" / "run"
LAYERS_CSV = REPO_ROOT / "results" / "gate_layers.csv"
GATE_MD = REPO_ROOT / "results" / "gate.md"

ALLOWED_SPLITS = ("train", "validation")
EXPECTED = {"train": (1251, 2502), "validation": (462, 924)}
N_KEPT = 292
PROBE_LAYERS = list(range(1, 28))  # §7: layers 1-27; layer 28 is the rival's
C_GRID = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]  # §18
MAX_ITER = 10_000
FLOOR = 0.75
BELOW = 0.10
ABOVE = 0.03
BOOT_N = 2000
BOOT_SEED = 20261019
CONTACT_DATE = "2026-11-14"
N_JOBS = 7


# ---------------------------------------------------------------- loading

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_gate_rows(path: Path) -> list[dict]:
    """Gate rows of train and validation only. Test rows and band rows are
    discarded on read, before any other field is parsed, and asserted absent."""
    rows = []
    with gzip.open(path, "rt", newline="") as fh:
        for r in csv.DictReader(fh):
            if r["split"] not in ALLOWED_SPLITS or r["distance_set"] != "gate":
                continue
            rows.append({"split": r["split"], "label": int(r["label"]),
                         "trace_index": int(r["trace_index"]), "t": int(r["t"])})
    assert rows and all(r["split"] in ALLOWED_SPLITS for r in rows)
    assert not any(r["split"] == "test" for r in rows)
    return rows


def marker_start_ids(onsets_csv: Path, traces_dir: Path) -> list[int]:
    """G1: distinct generated ids at paragraph-rule onsets of the kept traces (as c3)."""
    by_trace: dict[int, list[int]] = {}
    with open(onsets_csv, newline="") as fh:
        for r in csv.DictReader(fh):
            if r["merged_into"] == "":
                by_trace.setdefault(int(r["index"]), []).append(int(r["token_index"]))
    ids = set()
    for idx, tis in by_trace.items():
        out = json.loads((traces_dir / f"gsm8k_{idx:04d}.json").read_text())["output_token_ids"]
        ids.update(out[t] for t in tis)
    assert len(by_trace) == N_KEPT and sum(len(v) for v in by_trace.values()) == 2519
    return sorted(ids)


def load_manifest() -> dict[int, str]:
    with open(MANIFEST_PATH, newline="") as fh:
        return {int(r["trace_index"]): r["sha256"] for r in csv.DictReader(fh)}


def gather_states(rows: list[dict], act_dir: Path, manifest: dict[int, str]) -> np.ndarray:
    """[n_rows, 28, 1536] float16, in row order. Opens only the traces in rows."""
    out = np.empty((len(rows), 28, 1536), dtype=np.float16)
    by_trace: dict[int, list[int]] = {}
    for i, r in enumerate(rows):
        by_trace.setdefault(r["trace_index"], []).append(i)
    for tr, idxs in sorted(by_trace.items()):
        path = act_dir / f"gsm8k_{tr:04d}.npz"
        assert sha256_file(path) == manifest[tr], f"sha256 mismatch {path.name}"
        with np.load(path, allow_pickle=False) as z:
            ts, hidden = z["t"], z["hidden"]
            want = np.array([rows[i]["t"] for i in idxs])
            pos = np.searchsorted(ts, want)
            assert np.all(ts[pos] == want), tr
            out[idxs] = hidden[pos]
    return out


# ---------------------------------------------------------------- scoring

def standardize(train: np.ndarray, *others: np.ndarray):
    """z-score per dimension with mean and SD from train only (§18)."""
    mu = train.mean(axis=0, dtype=np.float64)
    sd = train.std(axis=0, dtype=np.float64)
    sd = np.where(sd > 0, sd, 1.0)
    f = lambda x: ((x - mu) / sd).astype(np.float32)  # noqa: E731
    return (f(train),) + tuple(f(o) for o in others)


def auroc(y: np.ndarray, s: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, s))


def bootstrap_by_problem(y, s, problem, n=BOOT_N, seed=BOOT_SEED) -> tuple[float, float]:
    """G4: 95% interval, resampling whole validation problems with replacement."""
    y, s, problem = np.asarray(y), np.asarray(s), np.asarray(problem)
    keys = sorted(set(problem.tolist()))
    members = {k: np.flatnonzero(problem == k) for k in keys}
    rng = random.Random(seed)
    vals = []
    for _ in range(n):
        idx = np.concatenate([members[keys[rng.randrange(len(keys))]] for _ in keys])
        yy = y[idx]
        if yy.min() != yy.max():
            vals.append(auroc(yy, s[idx]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def fit_layer(layer: int, Xtr: np.ndarray, ytr: np.ndarray, Xva: np.ndarray,
              yva: np.ndarray) -> tuple[list[dict], float]:
    """Five fits (C grid) for one layer, single-threaded. Returns rows and seconds."""
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from threadpoolctl import threadpool_limits

    t0 = time.perf_counter()
    out = []
    with threadpool_limits(limits=1):
        A, B = standardize(Xtr.astype(np.float32), Xva.astype(np.float32))
        for C in C_GRID:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always", ConvergenceWarning)
                m = LogisticRegression(penalty="l2", C=C, solver="lbfgs",
                                       max_iter=MAX_ITER, fit_intercept=True)
                m.fit(A, ytr)
            warned = any(issubclass(x.category, ConvergenceWarning) for x in w)
            converged = (not warned) and int(m.n_iter_[0]) < MAX_ITER
            out.append({"layer": layer, "C": C,
                        "auroc": auroc(yva, m.decision_function(B)),
                        "converged": converged, "n_iter": int(m.n_iter_[0])})
    return out, time.perf_counter() - t0


def best_fit(rows: list[dict]) -> dict:
    """G3: highest validation AUROC; ties -> lowest layer, then smallest C."""
    return min(rows, key=lambda r: (-r["auroc"], r["layer"], r["C"]))


def fit_best_scores(layer, C, Xtr, ytr, Xva):
    from sklearn.linear_model import LogisticRegression
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=1):
        A, B = standardize(Xtr.astype(np.float32), Xva.astype(np.float32))
        m = LogisticRegression(penalty="l2", C=C, solver="lbfgs", max_iter=MAX_ITER)
        m.fit(A, ytr)
        return m.decision_function(B)


# ---------------------------------------------------------------- main

def main() -> None:
    t_start = time.perf_counter()
    actual = sha256_file(EXAMPLES_PATH)
    assert actual == EXAMPLES_SHA256, f"examples.csv.gz sha256 mismatch: {actual}"
    rows = load_gate_rows(EXAMPLES_PATH)
    counts = {}
    for sp in ALLOWED_SPLITS:
        rs = [r for r in rows if r["split"] == sp]
        counts[sp] = (sum(r["label"] == 1 for r in rs), sum(r["label"] == 0 for r in rs),
                      len({r["trace_index"] for r in rs}))
        assert counts[sp][:2] == EXPECTED[sp], (sp, counts[sp])
    manifest = load_manifest()

    ids = marker_start_ids(ONSETS_PATH, TRACES_DIR)
    from before_the_wait.generate import load_model_and_tokenizer

    tokenizer, model = load_model_and_tokenizer()  # weights only; no forward pass
    decoded = [tokenizer.decode([i]) for i in ids]

    t0 = time.perf_counter()
    states = gather_states(rows, ACT_DIR, manifest)
    print(f"gathered {states.shape} in {time.perf_counter() - t0:.1f}s", flush=True)
    split = np.array([r["split"] for r in rows])
    y = np.array([r["label"] for r in rows])
    prob = np.array([r["trace_index"] for r in rows])
    tr, va = split == "train", split == "validation"

    # §11 clause 1: the model's own odds, from layer 28 through lm_head (already normed)
    import torch

    h28 = torch.from_numpy(states[va, 27].copy()).to(model.device)
    with torch.inference_mode():
        logits = model.lm_head(h28)
        p = torch.softmax(logits.float(), dim=-1)[:, ids].sum(dim=-1)
    odds = p.to("cpu").numpy().astype(np.float64)
    odds_auc = auroc(y[va], odds)
    odds_ci = bootstrap_by_problem(y[va], odds, prob[va])
    clause1 = odds_auc >= FLOOR
    print(f"model odds AUROC {odds_auc:.4f} {odds_ci}", flush=True)
    del model

    # §11 clauses 2-3: the probe, layers 1-27 x C grid
    from joblib import Parallel, delayed

    t0 = time.perf_counter()
    res = Parallel(n_jobs=N_JOBS, backend="loky")(
        delayed(fit_layer)(L, states[tr, L - 1], y[tr], states[va, L - 1], y[va])
        for L in PROBE_LAYERS)
    fits = [r for rs, _ in res for r in rs]
    layer_secs = [s for _, s in res]
    print(f"135 fits in {time.perf_counter() - t0:.1f}s wall; one layer (5 fits, 1 thread): "
          f"median {np.median(layer_secs):.1f}s, max {max(layer_secs):.1f}s", flush=True)
    assert len(fits) == 135
    best = best_fit(fits)
    scores = fit_best_scores(best["layer"], best["C"], states[tr, best["layer"] - 1], y[tr],
                             states[va, best["layer"] - 1])
    assert abs(auroc(y[va], scores) - best["auroc"]) < 1e-12
    best_ci = bootstrap_by_problem(y[va], scores, prob[va])
    lo_b, hi_b = odds_auc - BELOW, odds_auc + ABOVE
    clause2 = best["auroc"] >= lo_b
    clause3 = best["auroc"] <= hi_b

    with open(LAYERS_CSV, "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["layer", "C", "validation_auroc", "converged", "n_iter"])
        for r in sorted(fits, key=lambda r: (r["layer"], r["C"])):
            w.writerow([r["layer"], f"{r['C']:g}", f"{r['auroc']:.6f}", r["converged"], r["n_iter"]])

    failed = [n for n, ok in (("clause 1", clause1), ("clause 2", clause2),
                              ("clause 3", clause3)) if not ok]
    L: list[str] = []
    a = L.append
    a("# Gate at distance 1 (validation)")
    a("")
    a("Computed by `scripts/run_gate.py`. §11 Gate; probe per §7 and §18; G1–G4 from handoff 11a.")
    a("")
    a("## a. Data contact")
    a("")
    a("| Date | What was touched | What was not |")
    a("|---|---|---|")
    a(f"| {CONTACT_DATE} | Gate examples (distance 1) of the train and validation splits: "
      "labels and saved activations (layers 1-28); the model's lm_head weights (no forward "
      f"pass); the onset token ids of the {N_KEPT} kept traces. | The test split (no row "
      "loaded, no activation opened); any distance > 1; any rival feature or entropy value. |")
    a("")
    a("## b. Marker-start id set (G1)")
    a("")
    a(f"Distinct generated token ids at the 2,519 paragraph-rule onsets of the {N_KEPT} kept "
      f"traces: **{len(ids)} ids**.")
    a("")
    a("| id | decoded |")
    a("|---|---|")
    for i, d in zip(ids, decoded):
        a(f"| {i} | `{json.dumps(d)[1:-1]}` |")
    a("")
    a("## c. Counts")
    a("")
    a("| split | positives | negatives | problems |")
    a("|---|---|---|---|")
    for sp in ALLOWED_SPLITS:
        a(f"| {sp} | {counts[sp][0]} | {counts[sp][1]} | {counts[sp][2]} |")
    a("")
    a("## d. The model's own odds (clause 1)")
    a("")
    a("Score = summed probability over the id set, softmax(lm_head(layer-28 state)) in float32, "
      "temperature 1, no top-p; the layer-28 state already carries the final norm. "
      f"Interval: 95%, {BOOT_N:,} resamples of whole validation problems, seed {BOOT_SEED}.")
    a("")
    a(f"- Model-odds AUROC (validation): **{odds_auc:.4f}** (95% interval "
      f"{odds_ci[0]:.4f}–{odds_ci[1]:.4f}).")
    a(f"- Clause 1, AUROC ≥ {FLOOR}: **{'pass' if clause1 else 'FAIL'}**.")
    a("")
    a("## e. Probe per layer (best C)")
    a("")
    a(f"Logistic regression, L2, lbfgs, max_iter {MAX_ITER:,}, z-scored with train statistics; "
      "fit on train, scored on validation by the decision function. All 135 fits are in "
      f"`results/gate_layers.csv`; fits not converged: {sum(not r['converged'] for r in fits)}.")
    a("")
    a("| layer | best C | validation AUROC |")
    a("|---|---|---|")
    for Lyr in PROBE_LAYERS:
        b = best_fit([r for r in fits if r["layer"] == Lyr])
        a(f"| {Lyr} | {b['C']:g} | {b['auroc']:.4f} |")
    a("")
    a("## f. Best layer (clauses 2 and 3)")
    a("")
    a(f"- Best (layer, C) by G3: layer **{best['layer']}**, C = {best['C']:g}; validation AUROC "
      f"**{best['auroc']:.4f}** (95% interval {best_ci[0]:.4f}–{best_ci[1]:.4f}).")
    a(f"- Clause 2, probe ≥ model odds − {BELOW} = {lo_b:.4f}: **{'pass' if clause2 else 'FAIL'}**.")
    a(f"- Clause 3, probe ≤ model odds + {ABOVE} = {hi_b:.4f}: **{'pass' if clause3 else 'FAIL'}**.")
    a("")
    a("## g. Verdict")
    a("")
    a("**GATE PASSES**" if not failed else f"**GATE FAILS** ({', '.join(failed)})")
    a("")
    GATE_MD.write_text("\n".join(L))
    print("\n".join(L))
    print(f"total {time.perf_counter() - t_start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
