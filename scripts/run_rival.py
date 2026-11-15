"""Step 12: the combined outside rival per band, chosen on validation (pre-registration §8).

One logistic regression per band on four feature groups together (text, visible
uncertainty, output state, position), fitted on train and chosen on validation over
the registered grid: window {8, 16, 32, 64, 128, all} x n-grams {(1,2), (1,3)} x C
{1e-4 ... 1}. Pins D13, D19, D20, D21 (R1-R6); plan handoffs/12-rival-plan.md.

Reads the four bands' rows of the train and validation splits only, through the 11b
loader (same examples as the probes, §6). Test rows and gate rows are dropped before
anything else and asserted absent; test activations are never opened. Of the saved
activations only layer 28 is read (memory-mapped slice); layers 1-27 never enter an
array. No probe output is opened.

    .venv/bin/python scripts/run_rival.py --time-one     # heaviest cell, one fit (C = 1)
    caffeinate -i nohup .venv/bin/python scripts/run_rival.py > activations/rival_cache/run.out 2>&1 &

Resumable: each (band, window, n-gram range) cell, all five C values, is cached
atomically under activations/rival_cache/ (gitignored). After the 48 cells, the
chosen rival per band is refitted identically, saved to rival/, reloaded and
re-scored, and results/rival_grid.csv and results/rival.md are written.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import resource
import shutil
import sys
import time
import warnings
from bisect import bisect_left, bisect_right
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402

import extract_activations as ea  # noqa: E402  (npz_bytes, write_atomic)
import run_gate as rg  # noqa: E402  (sha256_file, auroc, bootstrap_by_problem, load_manifest)
import run_probes as rp  # noqa: E402  (load_band_rows, check_counts, memmap_npz_member)

EXAMPLES_PATH = REPO_ROOT / "examples" / "examples.csv.gz"
ACT_DIR = REPO_ROOT / "activations" / "run"
# gitignored via activations/; RIVAL_CACHE_DIR (inside the repo) gives a fresh rerun
CACHE_DIR = Path(os.environ.get("RIVAL_CACHE_DIR", REPO_ROOT / "activations" / "rival_cache"))
assert REPO_ROOT / "activations" in CACHE_DIR.resolve().parents, CACHE_DIR
JOBLIB_TMP = REPO_ROOT / "activations" / "joblib_tmp"   # gitignored via activations/
TRACES_DIR = REPO_ROOT / "traces" / "run"
ONSETS_PATH = REPO_ROOT / "results" / "run_onsets_paragraph.csv"
RIVAL_DIR = REPO_ROOT / "rival"
GRID_CSV = REPO_ROOT / "results" / "rival_grid.csv"
RIVAL_MD = REPO_ROOT / "results" / "rival.md"
PROGRESS = CACHE_DIR / "progress.log"

BANDS = rp.BANDS
WINDOWS = [8, 16, 32, 64, 128, "all"]
NGRAMS = [2, 3]  # ngram_range (1, n)
C_GRID = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
MAX_ITER = 10_000
ENT_PAST = 16
HIDDEN = 1536
OUT_LAYER_INDEX = 27  # layer 28 in the saved [n, 28, 1536] array
CONTACT_DATE = "2026-11-14"
N_JOBS = 5
DENSE_GROUPS = (("entropy", 0, 3), ("output", 3, 3 + HIDDEN), ("position", 3 + HIDDEN, 6 + HIDDEN))


# ---------------------------------------------------------------- per-trace context

def break_tokens(text: str, offsets: list[int]) -> list[int]:
    """Input indices of tokens in which a "\\n\\n" ends (its second \\n), ascending.

    `offsets` is align.prefix_lengths: token k covers characters [off[k], off[k+1]).
    Occurrences are found overlapping, so "\\n\\n\\n" yields two.
    """
    out = set()
    c = text.find("\n\n")
    while c != -1:
        end = c + 1
        k = bisect_right(offsets, end) - 1
        out.add(min(k, len(offsets) - 2))
        c = text.find("\n\n", c + 1)
    return sorted(out)


def since_break(breaks: list[int], i: int, P: int, t: int) -> int:
    k = bisect_right(breaks, i)
    return i - breaks[k - 1] if k else P + t


def since_onset(onsets: list[int], t: int, P: int) -> int:
    """onsets: generated-token indices of paragraph-rule onsets, ascending."""
    k = bisect_left(onsets, t)  # strictly before t
    return t - onsets[k - 1] if k else P + t


def entropy_feats(entropy: np.ndarray, prompt_entropy: np.ndarray, t: int, P: int):
    past = [float(entropy[k]) if k >= 0 else float(prompt_entropy[P + k])
            for k in range(t - ENT_PAST, t)]
    return float(entropy[t]), float(np.mean(past)), float(np.max(past))


def text_doc(ids: list[int], i: int, window) -> str:
    lo = 0 if window == "all" else max(0, i - int(window) + 1)
    return " ".join(map(str, ids[lo:i + 1]))


class TraceCtx:
    """Everything a feature at (trace, t) may read: ids up to t, entropy up to t,
    prompt entropy, breaks, onsets, layer-28 state."""

    def __init__(self, ids, P, breaks, onsets, entropy, prompt_entropy, out_rows):
        self.ids, self.P, self.breaks, self.onsets = ids, P, breaks, onsets
        self.entropy, self.prompt_entropy, self.out_rows = entropy, prompt_entropy, out_rows

    def dense(self, t: int) -> np.ndarray:
        i = self.P + t
        e = entropy_feats(self.entropy, self.prompt_entropy, t, self.P)
        pos = (t, since_onset(self.onsets, t, self.P), since_break(self.breaks, i, self.P, t))
        return np.concatenate([np.array(e, np.float64),
                               self.out_rows[t].astype(np.float64),
                               np.array(pos, np.float64)])

    def doc(self, t: int, window) -> str:
        return text_doc(self.ids, self.P + t, window)


def build_contexts(rows_all: list[dict], tokenizer, manifest) -> dict[int, TraceCtx]:
    from before_the_wait.align import prefix_lengths

    onsets: dict[int, list[int]] = {}
    with open(ONSETS_PATH, newline="") as fh:
        for r in csv.DictReader(fh):
            if r["merged_into"] == "":
                onsets.setdefault(int(r["index"]), []).append(int(r["token_index"]))
    want: dict[int, set[int]] = {}
    for r in rows_all:
        want.setdefault(r["trace_index"], set()).add(r["t"])
    ctx = {}
    for tr in sorted(want):
        trace = json.loads((TRACES_DIR / f"gsm8k_{tr:04d}.json").read_text())
        ids = list(trace["prompt_token_ids"]) + list(trace["output_token_ids"])
        P = len(trace["prompt_token_ids"])
        offs = prefix_lengths(tokenizer, ids)
        text = tokenizer.decode(ids, skip_special_tokens=False)
        assert len(text) == offs[-1], tr
        path = ACT_DIR / f"gsm8k_{tr:04d}.npz"
        assert rg.sha256_file(path) == manifest[tr], f"sha256 mismatch {path.name}"
        with np.load(path, allow_pickle=False) as z:
            ts, ent, pent = z["t"], z["entropy"], z["prompt_entropy"]
        assert len(ent) == len(trace["output_token_ids"]) and len(pent) == P
        ts_want = np.array(sorted(want[tr]))
        pos = np.searchsorted(ts, ts_want)
        assert np.all(ts[pos] == ts_want), tr
        hidden = rp.memmap_npz_member(path, "hidden")
        out = np.asarray(hidden[pos, OUT_LAYER_INDEX, :])  # layer 28 only
        del hidden
        ctx[tr] = TraceCtx(ids, P, break_tokens(text, offs), sorted(onsets.get(tr, [])),
                           ent, pent, dict(zip(ts_want.tolist(), out)))
    return ctx


# ---------------------------------------------------------------- matrices (train-only fitting)

def make_vectorizer(n: int, vocabulary=None):
    from sklearn.feature_extraction.text import TfidfVectorizer

    # R2: library defaults plus min_df = 2; ids are split on whitespace (plan Q2).
    return TfidfVectorizer(ngram_range=(1, n), min_df=2, analyzer="word",
                           tokenizer=str.split, token_pattern=None, lowercase=False,
                           vocabulary=vocabulary)


def dense_stats(D_train: np.ndarray):
    mu = D_train.mean(axis=0)
    sd = D_train.std(axis=0)
    return mu, np.where(sd > 0, sd, 1.0)


def group_r(Z: np.ndarray, lo: int, hi: int) -> float:
    return float(np.sqrt(np.mean(np.sum(Z[:, lo:hi] ** 2, axis=1))))


def text_r(T) -> float:
    return float(np.sqrt(np.asarray(T.multiply(T).sum(axis=1)).ravel().mean()))


def dense_factors(Z_train: np.ndarray) -> dict[str, float]:
    r = {g: group_r(Z_train, lo, hi) for g, lo, hi in DENSE_GROUPS}
    return {f"r_{g}": v for g, v in r.items()}


def assemble(T, Z, f_text: float, f: dict[str, float]):
    import scipy.sparse as sp

    Zs = Z.copy()
    for g, lo, hi in DENSE_GROUPS:
        Zs[:, lo:hi] *= f[g]
    return sp.hstack([T * f_text, sp.csr_matrix(Zs)], format="csr", dtype=np.float64)


def multipliers(r: dict[str, float], r_text: float) -> tuple[float, dict[str, float]]:
    r_out = r["r_output"]
    f = {g: (r_out / r[f"r_{g}"] if r[f"r_{g}"] > 0 else 0.0) for g, _, _ in DENSE_GROUPS}
    return (r_out / r_text if r_text > 0 else 0.0), f


def scores(X, coef, intercept) -> np.ndarray:
    return np.asarray(X @ np.asarray(coef, np.float64)).ravel() + float(intercept)


def fit_one(X, y, Xva, yva, C: float) -> dict:
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from threadpoolctl import threadpool_limits

    t0 = time.perf_counter()
    with threadpool_limits(limits=1):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", ConvergenceWarning)
            m = LogisticRegression(penalty="l2", C=C, solver="lbfgs", max_iter=MAX_ITER,
                                   fit_intercept=True)
            m.fit(X, y)
        warned = any(issubclass(x.category, ConvergenceWarning) for x in w)
        n_iter = int(m.n_iter_[0])
        coef = m.coef_[0].astype(np.float64)
        b = float(m.intercept_[0])
        s = scores(Xva, coef, b)
    return {"C": C, "auroc": rg.auroc(yva, s), "converged": (not warned) and n_iter < MAX_ITER,
            "n_iter": n_iter, "coef": coef, "intercept": b,
            "seconds": time.perf_counter() - t0}


def window_rank(w) -> int:
    return WINDOWS.index(w)


def chosen_key(r: dict):
    """R5 ties: smaller window, then (1,2), then smaller C. Non-converged fits stay eligible."""
    return (-r["auroc"], window_rank(r["window"]), r["n"], r["C"])


# ---------------------------------------------------------------- band data

class Band:
    def __init__(self, band, rows, ctx):
        self.band = band
        self.rows = rows
        self.split = np.array([r["split"] for r in rows])
        self.y = np.array([r["label"] for r in rows])
        self.prob = np.array([r["trace_index"] for r in rows])
        self.tr = np.flatnonzero(self.split == "train")
        self.va = np.flatnonzero(self.split == "validation")
        self.ctx = ctx
        D = np.stack([ctx[r["trace_index"]].dense(r["t"]) for r in rows])
        self.mu, self.sd = dense_stats(D[self.tr])  # train only
        Z = (D - self.mu) / self.sd
        self.Ztr, self.Zva = Z[self.tr], Z[self.va]
        self.r = dense_factors(self.Ztr)

    def docs(self, idx, window):
        rows = self.rows
        return (self.ctx[rows[k]["trace_index"]].doc(rows[k]["t"], window) for k in idx)

    def text(self, window, n, vocabulary=None, idf=None):
        vec = make_vectorizer(n, vocabulary)
        if vocabulary is None:
            Ttr = vec.fit_transform(self.docs(self.tr, window))  # train only
        else:
            vec.idf_ = idf
            Ttr = vec.transform(self.docs(self.tr, window))
        Tva = vec.transform(self.docs(self.va, window))
        return vec, Ttr, Tva


def cell_name(band, window, n) -> str:
    return f"{band}_w{window}_n{n}"


def run_cell(B: Band, window, n, temp_folder, n_jobs=N_JOBS, c_values=C_GRID) -> dict:
    from joblib import Parallel, delayed

    t0 = time.perf_counter()
    vec, Ttr, Tva = B.text(window, n)
    t_vec = time.perf_counter() - t0
    r_text = text_r(Ttr)
    f_text, f = multipliers(B.r, r_text)
    Xtr = assemble(Ttr, B.Ztr, f_text, f)
    Xva = assemble(Tva, B.Zva, f_text, f)
    zero_tr = int((Ttr.getnnz(axis=1) == 0).sum())
    zero_va = int((Tva.getnnz(axis=1) == 0).sum())
    ytr, yva = B.y[B.tr], B.y[B.va]
    fits = Parallel(n_jobs=min(n_jobs, len(c_values)), backend="loky",
                    temp_folder=str(temp_folder))(
        delayed(fit_one)(Xtr, ytr, Xva, yva, C) for C in c_values)
    return {"band": B.band, "window": window, "n": n, "vocab": len(vec.vocabulary_),
            "r_text": r_text, "f_text": f_text, "zero_train": zero_tr, "zero_val": zero_va,
            "nnz_train": int(Xtr.nnz), "vectorize_seconds": t_vec,
            "wall_seconds": time.perf_counter() - t0, "fits": fits}


# ---------------------------------------------------------------- cache

def save_cell(res: dict) -> None:
    arrays = {f"coef_{k}": f["coef"] for k, f in enumerate(res["fits"])}
    meta = {k: v for k, v in res.items() if k != "fits"}
    meta["fits"] = [{k: v for k, v in f.items() if k != "coef"} for f in res["fits"]]
    arrays["meta"] = np.frombuffer(json.dumps(meta, sort_keys=True).encode(), dtype=np.uint8)
    ea.write_atomic(CACHE_DIR / f"{cell_name(res['band'], res['window'], res['n'])}.npz",
                    ea.npz_bytes(arrays))


def load_cell(band, window, n) -> dict | None:
    p = CACHE_DIR / f"{cell_name(band, window, n)}.npz"
    if not p.exists():
        return None
    with np.load(p, allow_pickle=False) as z:
        meta = json.loads(z["meta"].tobytes().decode())
        for k, f in enumerate(meta["fits"]):
            f["coef"] = z[f"coef_{k}"]
    return meta


# ---------------------------------------------------------------- frozen rival

def rival_arrays(B: Band, vec, best: dict, cell: dict) -> dict[str, np.ndarray]:
    terms = sorted(vec.vocabulary_, key=vec.vocabulary_.get)
    f_text, f = multipliers(B.r, cell["r_text"])
    return {
        "window": np.array([0 if best["window"] == "all" else int(best["window"])], np.int64),
        "ngram_max": np.array([best["n"]], np.int64),
        "C": np.array([best["C"]], np.float64),
        "terms": np.array(terms, dtype=str),
        "idf": np.asarray(vec.idf_, np.float64),
        "dense_mean": B.mu, "dense_sd": B.sd,
        "factors": np.array([f_text, f["entropy"], f["output"], f["position"]], np.float64),
        "coef": np.asarray(best["coef"], np.float64),
        "intercept": np.array([best["intercept"]], np.float64),
    }


def score_saved(path: Path, B: Band) -> np.ndarray:
    """Rebuild the rival from its file alone and score the band's validation examples."""
    from threadpoolctl import threadpool_limits

    with np.load(path, allow_pickle=False) as z:
        w = int(z["window"][0])
        window = "all" if w == 0 else w
        n = int(z["ngram_max"][0])
        terms = [str(x) for x in z["terms"]]
        vec = make_vectorizer(n, vocabulary={tm: k for k, tm in enumerate(terms)})
        vec.idf_ = z["idf"]
        Tva = vec.transform(B.docs(B.va, window))
        D = np.stack([B.ctx[B.rows[k]["trace_index"]].dense(B.rows[k]["t"]) for k in B.va])
        Zva = (D - z["dense_mean"]) / z["dense_sd"]
        fa = z["factors"]
        X = assemble(Tva, Zva, float(fa[0]),
                     {"entropy": float(fa[1]), "output": float(fa[2]), "position": float(fa[3])})
        with threadpool_limits(limits=1):
            return scores(X, z["coef"], z["intercept"][0])


# ---------------------------------------------------------------- main

def setup():
    from transformers import AutoTokenizer

    from before_the_wait.generate import MODEL_ID, MODEL_REVISION

    assert rg.sha256_file(EXAMPLES_PATH) == rg.EXAMPLES_SHA256
    rows_by_band = rp.load_band_rows(EXAMPLES_PATH)  # 11b loader: same examples (§6)
    counts = rp.check_counts(rows_by_band)           # 11b counts as constants (plan Q1)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    manifest = rg.load_manifest()
    rows_all = [r for b in BANDS for r in rows_by_band[b]]
    t0 = time.perf_counter()
    ctx = build_contexts(rows_all, tokenizer, manifest)
    print(f"contexts for {len(ctx)} traces in {time.perf_counter() - t0:.1f}s", flush=True)
    return rows_by_band, counts, ctx


def peak_rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9  # bytes on macOS


def time_one() -> None:
    rows_by_band, _, ctx = setup()
    B = Band("21-40", rows_by_band["21-40"], ctx)
    JOBLIB_TMP.mkdir(parents=True, exist_ok=True)
    res = run_cell(B, "all", 3, JOBLIB_TMP, n_jobs=1, c_values=[1.0])
    f = res["fits"][0]
    print(f"TIMING 21-40 / all / (1,3) / C=1: vectorize {res['vectorize_seconds']:.1f}s, "
          f"fit {f['seconds']:.1f}s, n_iter {f['n_iter']}, converged {f['converged']}, "
          f"vocab {res['vocab']:,}, nnz train {res['nnz_train']:,}, "
          f"peak RSS (main) {peak_rss_gb():.1f} GB", flush=True)
    shutil.rmtree(JOBLIB_TMP, ignore_errors=True)


def main() -> None:
    if "--time-one" in sys.argv:
        time_one()
        return
    t_start = time.perf_counter()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    rows_by_band, counts, ctx = setup()
    if not PROGRESS.exists():
        PROGRESS.write_text("cell\tvocab\tvectorize_s\twall_s\tfit_s_max\tall_converged\n")
    cells: dict[tuple, dict] = {}
    bands: dict[str, Band] = {}
    for band in BANDS:
        B = Band(band, rows_by_band[band], ctx)
        bands[band] = B
        for window in WINDOWS:
            for n in NGRAMS:
                cached = load_cell(band, window, n)
                if cached is not None:
                    cells[(band, window, n)] = cached
                    print(f"[{cell_name(band, window, n)}] cached", flush=True)
                    continue
                JOBLIB_TMP.mkdir(parents=True, exist_ok=True)
                res = run_cell(B, window, n, JOBLIB_TMP)
                shutil.rmtree(JOBLIB_TMP, ignore_errors=True)
                save_cell(res)
                cells[(band, window, n)] = load_cell(band, window, n)
                fmax = max(f["seconds"] for f in res["fits"])
                allc = all(f["converged"] for f in res["fits"])
                with open(PROGRESS, "a") as fh:
                    fh.write(f"{cell_name(band, window, n)}\t{res['vocab']}\t"
                             f"{res['vectorize_seconds']:.1f}\t{res['wall_seconds']:.1f}\t"
                             f"{fmax:.1f}\t{allc}\n")
                print(f"[{cell_name(band, window, n)}] vocab {res['vocab']:,} vectorize "
                      f"{res['vectorize_seconds']:.1f}s wall {res['wall_seconds']:.1f}s "
                      f"slowest fit {fmax:.1f}s converged {allc}", flush=True)

    # selection, freezing, reload check
    grid_rows, chosen, shas, cis, chosen_cell = [], {}, {}, {}, {}
    RIVAL_DIR.mkdir(exist_ok=True)
    for band in BANDS:
        B = bands[band]
        flat = []
        for (b, window, n), cell in cells.items():
            if b != band:
                continue
            for f in cell["fits"]:
                flat.append({**f, "band": band, "window": window, "n": n, "vocab": cell["vocab"]})
        assert len(flat) == 60
        grid_rows += flat
        best = min(flat, key=chosen_key)
        cell = cells[(band, best["window"], best["n"])]
        vec, _, _ = B.text(best["window"], best["n"])  # identical refit of the vectorizer
        assert len(vec.vocabulary_) == cell["vocab"]
        path = RIVAL_DIR / f"band_{band}.npz"
        data = ea.npz_bytes(rival_arrays(B, vec, best, cell))
        ea.write_atomic(path, data)
        shas[band] = hashlib.sha256(data).hexdigest()
        s = score_saved(path, B)
        assert rg.auroc(B.y[B.va], s) == best["auroc"], (band, rg.auroc(B.y[B.va], s), best["auroc"])
        cis[band] = rg.bootstrap_by_problem(B.y[B.va], s, B.prob[B.va])
        chosen[band], chosen_cell[band] = best, cell
        print(f"[{band}] frozen {path.name} ({len(data):,} bytes)", flush=True)
    assert len(grid_rows) == 240

    write_grid(grid_rows)
    write_md(counts, bands, cells, chosen, shas, cis, {b: (RIVAL_DIR / f"band_{b}.npz").stat().st_size for b in BANDS})
    print(RIVAL_MD.read_text())
    print(f"total {time.perf_counter() - t_start:.1f}s", flush=True)


def wlabel(w) -> str:
    return str(w)


def write_grid(grid_rows) -> None:
    order = {b: k for k, b in enumerate(BANDS)}
    with open(GRID_CSV, "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["band", "window", "ngram_range", "C", "validation_auroc", "converged",
                    "n_iter", "vocabulary_size"])
        for r in sorted(grid_rows, key=lambda r: (order[r["band"]], window_rank(r["window"]),
                                                  r["n"], r["C"])):
            w.writerow([r["band"], wlabel(r["window"]), f"1-{r['n']}", f"{r['C']:g}",
                        f"{r['auroc']:.6f}", r["converged"], r["n_iter"], r["vocab"]])


def write_md(counts, bands, cells, chosen, shas, cis, sizes) -> None:
    import scipy
    import sklearn

    L: list[str] = []
    a = L.append
    a("# Outside rival: combined, per band, chosen on validation")
    a("")
    a("Computed by `scripts/run_rival.py`. §8, D13, D19, D20, D21 (R1–R6); plan "
      "`handoffs/12-rival-plan.md` as approved.")
    a("")
    a("## a. Data contact")
    a("")
    a("| Date | What was touched | What was not |")
    a("|---|---|---|")
    a(f"| {CONTACT_DATE} | Band examples (2-5, 6-10, 11-20, 21-40) of the train and validation "
      "splits: labels, layer-28 state, entropy and prompt entropy, token ids, paragraph-rule "
      "onsets. | The test split (no row loaded, no activation opened); layers 1-27; any probe "
      "output; the gate rows. |")
    a("")
    a("## b. Counts")
    a("")
    a("| band | split | positives | negatives | problems |")
    a("|---|---|---|---|---|")
    for b in BANDS:
        for sp in rp.ALLOWED_SPLITS:
            c = counts[(b, sp)]
            a(f"| {b} | {sp} | {c[0]} | {c[1]} | {c[2]} |")
    a("")
    a("## c. Definitions as implemented")
    a("")
    a(f"scikit-learn {sklearn.__version__}, scipy {scipy.__version__}, numpy {np.__version__}. "
      "At an example (trace, t): P = prompt length, i = P + t, ids = prompt ids + generated ids.")
    a("")
    a("- **Text (R1, R2).** Window w ∈ {8, 16, 32, 64, 128}: ids[i − w + 1 … i]; \"all\": "
      "ids[0 … i]. Each window is written as its token ids separated by single spaces and "
      "**split on whitespace** (`tokenizer=str.split, token_pattern=None, lowercase=False`), so "
      "every id is one token (the default token pattern would drop one-character ids). "
      "`TfidfVectorizer(ngram_range=(1, n), min_df=2)`, all other settings default (l2 norm, "
      "smoothed IDF, raw counts); vocabulary and IDF fitted on the band's train examples.")
    a("- **Entropy (D19, D20).** Entropy at t (raw softmax, nats); mean and max over t−16 … t−1, "
      "from the prompt entropy where that range reaches before generation; t excluded.")
    a("- **Output state (D13).** The saved layer-28 state at t (final norm applied), 1,536 columns.")
    a("- **Position (R4).** t; t minus the last paragraph-rule onset before t, else P + t; i minus "
      "the input index of the token in which the last \"\\n\\n\" at or before i ends (located with "
      "`align.prefix_lengths`, the incremental decode the onsets use), else P + t.")
    a("- **Scaling (R3).** Entropy, output-state and position columns z-scored with the band's "
      "train mean and SD (a zero-variance column gets SD 1). Group g multiplied by r_out / r_g, "
      "r_g = sqrt(mean over train rows of the squared row norm of g). One sparse float64 matrix: "
      "text, entropy, output state, position.")
    a(f"- **Classifier (R5).** Logistic regression, L2, lbfgs, max_iter {MAX_ITER:,}, intercept, "
      "no class weights; C ∈ {1e-4, 1e-3, 1e-2, 1e-1, 1}; scored by the decision function. Chosen: "
      "highest validation AUROC; ties to the smaller window, then (1,2), then the smaller C. A "
      "fit that did not converge stays eligible and is flagged.")
    a("- **Frozen (R6).** The chosen rival is the one fitted on train, saved per band and "
      "reloaded from its file alone to reproduce its validation AUROC exactly.")
    a("")
    a("## d. Group factors and multipliers")
    a("")
    a("Dense groups (per band, from the train examples):")
    a("")
    a("| band | r_entropy | × entropy | r_output | × output | r_position | × position |")
    a("|---|---|---|---|---|---|---|")
    for b in BANDS:
        r = bands[b].r
        _, f = multipliers(r, 1.0)
        a(f"| {b} | {r['r_entropy']:.4f} | {f['entropy']:.4f} | {r['r_output']:.4f} | "
          f"{f['output']:.4f} | {r['r_position']:.4f} | {f['position']:.4f} |")
    a("")
    a("Text group, per text configuration (chosen configuration marked):")
    a("")
    a("| band | window | n-grams | vocabulary | r_text | × text | all-zero text rows (train / val) | chosen |")
    a("|---|---|---|---|---|---|---|---|")
    for b in BANDS:
        for wnd in WINDOWS:
            for n in NGRAMS:
                c = cells[(b, wnd, n)]
                mark = "yes" if (chosen[b]["window"], chosen[b]["n"]) == (wnd, n) else ""
                a(f"| {b} | {wlabel(wnd)} | 1-{n} | {c['vocab']:,} | {c['r_text']:.4f} | "
                  f"{c['f_text']:.4f} | {c['zero_train']} / {c['zero_val']} | {mark} |")
    a("")
    a("## e. Chosen rival per band")
    a("")
    a("Interval: 95%, 2,000 resamples of whole validation problems, seed 20261019 (G4). "
      "Fits not converged in the grid: "
      f"{sum(not f['converged'] for c in cells.values() for f in c['fits'])} of 240 "
      "(`results/rival_grid.csv`).")
    a("")
    a("| band | window | n-grams | C | validation AUROC | 95% interval | converged | file bytes | sha256 |")
    a("|---|---|---|---|---|---|---|---|---|")
    for b in BANDS:
        c = chosen[b]
        a(f"| {b} | {wlabel(c['window'])} | 1-{c['n']} | {c['C']:g} | {c['auroc']:.4f} | "
          f"{cis[b][0]:.4f}–{cis[b][1]:.4f} | {c['converged']} | {sizes[b]:,} | `{shas[b]}` |")
    a("")
    a("## f. Test split")
    a("")
    a("Test split not read. The chosen rivals are frozen as of this commit.")
    a("")
    RIVAL_MD.write_text("\n".join(L))


if __name__ == "__main__":
    main()
