"""Step 11b: band probes, one per layer, layer chosen on validation (pre-registration §7, §18).

Reads the four bands' rows of the train and validation splits only. Test rows and
gate rows are dropped by the loader before anything else and asserted absent; test
activations are never opened. Layer 28 (the rival's, §7, D13) is never read into an
array: each file's `hidden` member is memory-mapped and sliced to layers 1-27.

    .venv/bin/python scripts/run_probes.py               # full run
    .venv/bin/python scripts/run_probes.py --time-one    # time one layer of band 21-40

Writes results/probe_layers.csv (540 fits), results/probes.md and
probes/band_<lo>-<hi>.npz (the chosen probe per band, fitted on train only, B1).
Deterministic: single-threaded BLAS in every fit and score, fixed bootstrap seed.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import os
import shutil
import sys
import time
import warnings
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import numpy as np  # noqa: E402

import extract_activations as ea  # noqa: E402  (npz_bytes, write_atomic)
import run_gate as rg  # noqa: E402  (sha256_file, auroc, bootstrap_by_problem, best_fit)

EXAMPLES_PATH = REPO_ROOT / "examples" / "examples.csv.gz"
EXAMPLES_SHA256 = rg.EXAMPLES_SHA256
ACT_DIR = REPO_ROOT / "activations" / "run"
JOBLIB_TMP = REPO_ROOT / "activations" / "joblib_tmp"  # gitignored via activations/
PROBES_DIR = REPO_ROOT / "probes"
LAYERS_CSV = REPO_ROOT / "results" / "probe_layers.csv"
PROBES_MD = REPO_ROOT / "results" / "probes.md"

BANDS = ["2-5", "6-10", "11-20", "21-40"]
ALLOWED_SPLITS = ("train", "validation")
EXPECTED = {
    "2-5": {"train": (4955, 9910), "validation": (1820, 3640)},
    "6-10": {"train": (6028, 12056), "validation": (2164, 4328)},
    "11-20": {"train": (11623, 23246), "validation": (4134, 8268)},
    "21-40": {"train": (21656, 43312), "validation": (7776, 15552)},
}
PROBE_LAYERS = list(range(1, 28))  # §7; layer 28 is not loaded
N_PROBE_LAYERS = len(PROBE_LAYERS)
HIDDEN = 1536
C_GRID = rg.C_GRID
MAX_ITER = 10_000
CONTACT_DATE = "2026-11-14"
N_JOBS = 7


# ---------------------------------------------------------------- loading

def load_band_rows(path: Path) -> dict[str, list[dict]]:
    """Band rows of train and validation only, per band. Test rows and gate rows are
    discarded on read, before any other field is parsed, and asserted absent."""
    out: dict[str, list[dict]] = {b: [] for b in BANDS}
    with gzip.open(path, "rt", newline="") as fh:
        for r in csv.DictReader(fh):
            if r["split"] not in ALLOWED_SPLITS or r["distance_set"] not in BANDS:
                continue
            out[r["distance_set"]].append({
                "split": r["split"], "label": int(r["label"]),
                "trace_index": int(r["trace_index"]), "t": int(r["t"])})
    for b, rows in out.items():
        assert all(r["split"] in ALLOWED_SPLITS for r in rows)
        assert not any(r["split"] == "test" for r in rows)
    return out


def memmap_npz_member(path: Path, name: str) -> np.memmap:
    """Read-only memmap of one member of a stored (uncompressed) .npz."""
    with zipfile.ZipFile(path) as zf:
        info = zf.getinfo(f"{name}.npy")
        assert info.compress_type == zipfile.ZIP_STORED
        offset = info.header_offset
    with open(path, "rb") as fh:
        fh.seek(offset)
        local = fh.read(30)
        assert local[:4] == b"PK\x03\x04"
        n_name = int.from_bytes(local[26:28], "little")
        n_extra = int.from_bytes(local[28:30], "little")
        fh.seek(offset + 30 + n_name + n_extra)
        version = np.lib.format.read_magic(fh)
        read_header = (np.lib.format.read_array_header_1_0 if version == (1, 0)
                       else np.lib.format.read_array_header_2_0)
        shape, fortran, dtype = read_header(fh)
        assert not fortran
        data_offset = fh.tell()
    return np.memmap(path, dtype=dtype, mode="r", offset=data_offset, shape=shape)


def gather_layers_1_27(rows: list[dict], act_dir: Path, verified: set[int],
                       manifest: dict[int, str]) -> np.ndarray:
    """[27, n_rows, 1536] float16 (layer-major), layers 1-27 only, in row order."""
    X = np.empty((N_PROBE_LAYERS, len(rows), HIDDEN), dtype=np.float16)
    by_trace: dict[int, list[int]] = {}
    for i, r in enumerate(rows):
        by_trace.setdefault(r["trace_index"], []).append(i)
    for tr, idxs in sorted(by_trace.items()):
        path = act_dir / f"gsm8k_{tr:04d}.npz"
        if tr not in verified:
            assert rg.sha256_file(path) == manifest[tr], f"sha256 mismatch {path.name}"
            verified.add(tr)
        with np.load(path, allow_pickle=False) as z:
            ts = z["t"]
        want = np.array([rows[i]["t"] for i in idxs])
        pos = np.searchsorted(ts, want)
        assert np.all(ts[pos] == want), tr
        hidden = memmap_npz_member(path, "hidden")
        assert hidden.shape[1] == 28
        block = np.asarray(hidden[pos, :N_PROBE_LAYERS, :])  # layers 1-27 only
        X[:, idxs, :] = block.transpose(1, 0, 2)
        del hidden
    return X


# ---------------------------------------------------------------- probe math

def zstats(train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-dimension mean and SD (float64) from the band's train examples only (§18)."""
    x = train.astype(np.float32)
    mu = x.mean(axis=0, dtype=np.float64)
    sd = x.std(axis=0, dtype=np.float64)
    return mu, np.where(sd > 0, sd, 1.0)


def apply_z(x: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return ((x.astype(np.float32) - mu) / sd).astype(np.float32)


def probe_scores(x: np.ndarray, mu, sd, coef, intercept) -> np.ndarray:
    """Decision function of a saved probe: z(x) . coef + intercept, in float64."""
    return apply_z(x, mu, sd).astype(np.float64) @ np.asarray(coef, np.float64) + float(intercept)


def fit_layer(band: str, layer: int, Xtr: np.ndarray, ytr: np.ndarray, Xva: np.ndarray,
              yva: np.ndarray) -> tuple[list[dict], float]:
    """Five fits (C grid) for one layer, single-threaded."""
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from threadpoolctl import threadpool_limits

    t0 = time.perf_counter()
    out = []
    with threadpool_limits(limits=1):
        mu, sd = zstats(Xtr)
        A = apply_z(Xtr, mu, sd)
        for C in C_GRID:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always", ConvergenceWarning)
                m = LogisticRegression(penalty="l2", C=C, solver="lbfgs",
                                       max_iter=MAX_ITER, fit_intercept=True)
                m.fit(A, ytr)
            warned = any(issubclass(x.category, ConvergenceWarning) for x in w)
            n_iter = int(m.n_iter_[0])
            coef = m.coef_[0].astype(np.float64)
            intercept = float(m.intercept_[0])
            s = probe_scores(Xva, mu, sd, coef, intercept)
            out.append({"band": band, "layer": layer, "C": C, "auroc": rg.auroc(yva, s),
                        "converged": (not warned) and n_iter < MAX_ITER, "n_iter": n_iter,
                        "coef": coef, "intercept": intercept, "mu": mu, "sd": sd})
    return out, time.perf_counter() - t0


def probe_arrays(fit: dict) -> dict[str, np.ndarray]:
    return {"layer": np.array([fit["layer"]], dtype=np.int64),
            "C": np.array([fit["C"]], dtype=np.float64),
            "coef": np.asarray(fit["coef"], dtype=np.float64),
            "intercept": np.array([fit["intercept"]], dtype=np.float64),
            "mean": np.asarray(fit["mu"], dtype=np.float64),
            "sd": np.asarray(fit["sd"], dtype=np.float64)}


# ---------------------------------------------------------------- main

def band_data(band, rows_by_band, verified, manifest):
    rows = rows_by_band[band]
    t0 = time.perf_counter()
    X = gather_layers_1_27(rows, ACT_DIR, verified, manifest)
    split = np.array([r["split"] for r in rows])
    y = np.array([r["label"] for r in rows])
    prob = np.array([r["trace_index"] for r in rows])
    print(f"[{band}] gathered {X.shape} ({X.nbytes / 1e9:.2f} GB) in "
          f"{time.perf_counter() - t0:.1f}s", flush=True)
    return X, split, y, prob


def check_counts(rows_by_band) -> dict:
    counts = {}
    for b in BANDS:
        for sp in ALLOWED_SPLITS:
            rs = [r for r in rows_by_band[b] if r["split"] == sp]
            c = (sum(r["label"] == 1 for r in rs), sum(r["label"] == 0 for r in rs),
                 len({r["trace_index"] for r in rs}))
            assert c[:2] == EXPECTED[b][sp], (b, sp, c)
            counts[(b, sp)] = c
    return counts


def main() -> None:
    from joblib import Parallel, delayed

    t_start = time.perf_counter()
    actual = rg.sha256_file(EXAMPLES_PATH)
    assert actual == EXAMPLES_SHA256, f"examples.csv.gz sha256 mismatch: {actual}"
    rows_by_band = load_band_rows(EXAMPLES_PATH)
    counts = check_counts(rows_by_band)
    manifest = rg.load_manifest()
    verified: set[int] = set()
    JOBLIB_TMP.mkdir(parents=True, exist_ok=True)

    if "--time-one" in sys.argv:
        X, split, y, _ = band_data("21-40", rows_by_band, verified, manifest)
        tr, va = split == "train", split == "validation"
        L = 14
        _, secs = fit_layer("21-40", L, X[L - 1][tr], y[tr], X[L - 1][va], y[va])
        print(f"one layer of 21-40 (layer {L}, 5 fits, 1 thread): {secs:.1f}s; "
              f"estimate all 540 fits on {N_JOBS} workers: see report", flush=True)
        shutil.rmtree(JOBLIB_TMP, ignore_errors=True)
        return

    fits: list[dict] = []
    chosen: dict[str, dict] = {}
    cis: dict[str, tuple[float, float]] = {}
    shas: dict[str, str] = {}
    timing: dict[str, list[float]] = {}
    PROBES_DIR.mkdir(exist_ok=True)
    for band in BANDS:
        X, split, y, prob = band_data(band, rows_by_band, verified, manifest)
        tr, va = split == "train", split == "validation"
        t0 = time.perf_counter()
        res = Parallel(n_jobs=N_JOBS, backend="loky", temp_folder=str(JOBLIB_TMP))(
            delayed(fit_layer)(band, L, X[L - 1][tr], y[tr], X[L - 1][va], y[va])
            for L in PROBE_LAYERS)
        band_fits = [r for rs, _ in res for r in rs]
        timing[band] = [s for _, s in res]
        print(f"[{band}] 135 fits in {time.perf_counter() - t0:.1f}s wall; one layer "
              f"median {np.median(timing[band]):.1f}s, max {max(timing[band]):.1f}s", flush=True)
        assert len(band_fits) == 135
        best = rg.best_fit(band_fits)
        # B2: save the chosen probe, then reload it and re-score validation
        path = PROBES_DIR / f"band_{band}.npz"
        data = ea.npz_bytes(probe_arrays(best))
        ea.write_atomic(path, data)
        shas[band] = hashlib.sha256(data).hexdigest()
        from threadpoolctl import threadpool_limits

        with np.load(path, allow_pickle=False) as z, threadpool_limits(limits=1):
            Lc = int(z["layer"][0])
            s = probe_scores(X[Lc - 1][va], z["mean"], z["sd"], z["coef"], z["intercept"][0])
        assert rg.auroc(y[va], s) == best["auroc"], (band, rg.auroc(y[va], s), best["auroc"])
        cis[band] = rg.bootstrap_by_problem(y[va], s, prob[va])
        chosen[band] = best
        fits.extend(band_fits)
        del X, res
        shutil.rmtree(JOBLIB_TMP, ignore_errors=True)
        JOBLIB_TMP.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(JOBLIB_TMP, ignore_errors=True)
    assert len(fits) == 540

    with open(LAYERS_CSV, "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["band", "layer", "C", "validation_auroc", "converged", "n_iter"])
        for r in fits:
            w.writerow([r["band"], r["layer"], f"{r['C']:g}", f"{r['auroc']:.6f}",
                        r["converged"], r["n_iter"]])

    L: list[str] = []
    a = L.append
    a("# Band probes: per layer, layer chosen on validation")
    a("")
    a("Computed by `scripts/run_probes.py`. §7 and §18; G2–G4, B1 and B2 from handoff 11b.")
    a("")
    a("## a. Data contact")
    a("")
    a("| Date | What was touched | What was not |")
    a("|---|---|---|")
    a(f"| {CONTACT_DATE} | Band examples (2-5, 6-10, 11-20, 21-40) of the train and validation "
      "splits: labels and saved activations of layers 1-27. | The test split (no row loaded, "
      "no activation opened); layer 28; any rival feature or entropy value; the gate rows. |")
    a("")
    a("## b. Counts")
    a("")
    a("| band | split | positives | negatives | problems |")
    a("|---|---|---|---|---|")
    for b in BANDS:
        for sp in ALLOWED_SPLITS:
            c = counts[(b, sp)]
            a(f"| {b} | {sp} | {c[0]} | {c[1]} | {c[2]} |")
    a("")
    a("## c. Per band, per layer: best C and its validation AUROC")
    a("")
    a(f"Logistic regression, L2, lbfgs, max_iter {MAX_ITER:,}, z-scored with the band's train "
      "statistics; fit on train, scored on validation by the decision function. All 540 fits "
      f"are in `results/probe_layers.csv`; fits not converged: "
      f"{sum(not r['converged'] for r in fits)}.")
    a("")
    a("| layer | " + " | ".join(f"{b}: C | {b}: AUROC" for b in BANDS) + " |")
    a("|---|" + "---|---|" * len(BANDS))
    for Lyr in PROBE_LAYERS:
        cells = []
        for b in BANDS:
            bf = rg.best_fit([r for r in fits if r["band"] == b and r["layer"] == Lyr])
            cells += [f"{bf['C']:g}", f"{bf['auroc']:.4f}"]
        a(f"| {Lyr} | " + " | ".join(cells) + " |")
    a("")
    a("## d. Chosen probe per band")
    a("")
    a("G3: highest validation AUROC over layers 1-27 × C; ties to the lowest layer, then the "
      "smallest C. Fitted on train only (B1). Interval: 95%, 2,000 resamples of whole "
      "validation problems, seed 20261019 (G4).")
    a("")
    a("| band | layer | C | validation AUROC | 95% interval | probe file sha256 |")
    a("|---|---|---|---|---|---|")
    for b in BANDS:
        c = chosen[b]
        a(f"| {b} | {c['layer']} | {c['C']:g} | {c['auroc']:.4f} | "
          f"{cis[b][0]:.4f}–{cis[b][1]:.4f} | `{shas[b]}` |")
    a("")
    a("## e. Test split")
    a("")
    a("Test split not read. The chosen probes are frozen as of this commit.")
    a("")
    PROBES_MD.write_text("\n".join(L))
    print("\n".join(L))
    print(f"total {time.perf_counter() - t_start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
