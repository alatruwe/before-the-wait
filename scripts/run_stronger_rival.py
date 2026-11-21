"""Stronger rival, stage B: the 7B reader for band 6-10, fit and chosen on validation (§8, D24).

One L2 logistic regression per 7B layer (hidden_states entries 1-28) per C, with the
probe's recipe (§7, §18; run_probes.fit_layer), on the band 6-10 train examples; layer and
C chosen by validation AUROC with the probe's tie rule. The chosen reader is fitted on
train, saved to rival_7b/band_6-10.npz, hashed, reloaded and re-scored on validation.
Plan handoffs/stronger-rival-plan.md as approved.

    .venv/bin/python scripts/run_stronger_rival.py --fit        # stage B
    .venv/bin/python scripts/run_stronger_rival.py --rehearse   # stage C rehearsal, validation
    .venv/bin/python scripts/run_stronger_rival.py --test       # stage C, test, once

--fit reads band 6-10 rows of the train and validation splits only: test rows, gate rows
and every other band are dropped before any other field is parsed, and asserted absent.
It opens only the 7B activation files of train and validation traces (asserted against
results/split.json). The frozen 1.5B probe is hash-checked at start and end.

--rehearse reads band 6-10 validation rows only, scores the frozen probe and the frozen
7B reader through the same functions --test uses, and must reproduce both validation
AUROCs on file; it records a pass with this file's sha256. --test refuses to start
without that pass or if this file changed, builds the step-13 draw matrix before any test
row is read, scores the probe, stops unless its test AUROC reproduces
results/verdicts.csv exactly (S8), then scores the 7B reader and writes the verdict
(D22, §11). No AUROC is printed; no per-example test score is written.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import os
import shutil
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ["HF_HOME"] = str(REPO_ROOT / ".hf_cache")
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402

import extract_activations as ea  # noqa: E402  (npz_bytes, write_atomic, sha256_file)
import run_gate as rg  # noqa: E402  (auroc, best_fit, bootstrap_by_problem, EXAMPLES_SHA256)
import run_probes as rp  # noqa: E402  (memmap_npz_member, fit_layer, probe_arrays, ...)

BAND = "6-10"
ALLOWED_SPLITS = ("train", "validation")
EXAMPLES_PATH = REPO_ROOT / "examples" / "examples.csv.gz"
SPLIT_JSON = REPO_ROOT / "results" / "split.json"
ACT_7B = REPO_ROOT / "activations" / "run_7b"
STAGE_B_JSON = ACT_7B / "stage_b.json"
JOBLIB_TMP = REPO_ROOT / "activations" / "joblib_tmp"
MANIFEST_7B = REPO_ROOT / "results" / "stronger_rival_extraction_manifest.csv"
LAYERS_CSV = REPO_ROOT / "results" / "stronger_rival_layers.csv"
MD = REPO_ROOT / "results" / "stronger_rival.md"
READER = REPO_ROOT / "rival_7b" / "band_6-10.npz"
PROBE = REPO_ROOT / "probes" / "band_6-10.npz"
PROBE_SHA = "bf9bd8708a4abcd6ff9a0643288cd86341d70dab9955b21cd9950b9a16133dfa"
EXPECTED = {"train": (6028, 12056), "validation": (2164, 4328)}
N_LAYERS_7B = 28
HIDDEN_7B = 3584
LAYERS = list(range(1, N_LAYERS_7B + 1))
C_GRID = rp.C_GRID
N_JOBS = 6
CONTACT_DATE = "2026-11-21"
RIVAL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
RIVAL_REV = "916b56a44061fd5cd7d6a8fb632557ed4f724f60"


# ---------------------------------------------------------------- loading

def load_rows(path: Path, splits=ALLOWED_SPLITS, band: str = BAND) -> list[dict]:
    """Rows of one band and the named splits only. Every other row is dropped before any
    other field is parsed."""
    rows = []
    with gzip.open(path, "rt", newline="") as fh:
        for r in csv.DictReader(fh):
            if r["distance_set"] != band or r["split"] not in splits:
                continue
            rows.append({"split": r["split"], "label": int(r["label"]),
                         "trace_index": int(r["trace_index"]), "t": int(r["t"])})
    assert all(r["split"] in splits for r in rows)
    return rows


def check_frozen_probe() -> None:
    assert ea.sha256_file(PROBE) == PROBE_SHA, "frozen 1.5B probe changed"


def load_manifest_7b() -> dict[int, str]:
    with open(MANIFEST_7B, newline="") as fh:
        return {int(r["trace_index"]): r["sha256"] for r in csv.DictReader(fh)}


def gather_all_layers(rows: list[dict], act_dir: Path, manifest: dict[int, str],
                      forbidden: set[int]) -> np.ndarray:
    """[28, n_rows, 3584] float16 (layer-major), in row order. Refuses forbidden traces."""
    X = np.empty((N_LAYERS_7B, len(rows), HIDDEN_7B), dtype=np.float16)
    by_trace: dict[int, list[int]] = {}
    for i, r in enumerate(rows):
        by_trace.setdefault(r["trace_index"], []).append(i)
    assert not (set(by_trace) & forbidden), "a forbidden (test) trace would be opened"
    for tr, idxs in sorted(by_trace.items()):
        path = act_dir / f"gsm8k_{tr:04d}.npz"
        assert ea.sha256_file(path) == manifest[tr], f"sha256 mismatch {path.name}"
        with np.load(path, allow_pickle=False) as z:
            ts = z["t"]
        want = np.array([rows[i]["t"] for i in idxs])
        pos = np.searchsorted(ts, want)
        assert np.all(ts[pos] == want), tr
        hidden = rp.memmap_npz_member(path, "hidden")
        assert hidden.shape[1:] == (N_LAYERS_7B, HIDDEN_7B)
        X[:, idxs, :] = np.asarray(hidden[pos]).transpose(1, 0, 2)
        del hidden
    return X


# ---------------------------------------------------------------- stage B

def fit() -> None:
    from joblib import Parallel, delayed
    from threadpoolctl import threadpool_limits

    t_start = time.perf_counter()
    check_frozen_probe()
    assert ea.sha256_file(EXAMPLES_PATH) == rg.EXAMPLES_SHA256
    rows = load_rows(EXAMPLES_PATH)
    split = np.array([r["split"] for r in rows])
    y = np.array([r["label"] for r in rows])
    prob = np.array([r["trace_index"] for r in rows])
    counts = {}
    for sp in ALLOWED_SPLITS:
        m = split == sp
        counts[sp] = (int((y[m] == 1).sum()), int((y[m] == 0).sum()), len(set(prob[m].tolist())))
        assert counts[sp][:2] == EXPECTED[sp], (sp, counts[sp])
    test_traces = set(json.loads(SPLIT_JSON.read_text())["test"])
    X = gather_all_layers(rows, ACT_7B, load_manifest_7b(), test_traces)
    tr, va = split == "train", split == "validation"
    JOBLIB_TMP.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    res = Parallel(n_jobs=N_JOBS, backend="loky", temp_folder=str(JOBLIB_TMP))(
        delayed(rp.fit_layer)(BAND, L, X[L - 1][tr], y[tr], X[L - 1][va], y[va])
        for L in LAYERS)
    fits = [r for rs, _ in res for r in rs]
    secs = [s for _, s in res]
    fit_wall = time.perf_counter() - t0
    assert len(fits) == len(LAYERS) * len(C_GRID)
    best = rg.best_fit(fits)
    data = ea.npz_bytes(rp.probe_arrays(best))
    READER.parent.mkdir(exist_ok=True)
    ea.write_atomic(READER, data)
    sha = hashlib.sha256(data).hexdigest()
    with np.load(READER, allow_pickle=False) as z, threadpool_limits(limits=1):
        Lc = int(z["layer"][0])
        s = rp.probe_scores(X[Lc - 1][va], z["mean"], z["sd"], z["coef"], z["intercept"][0])
    re_auroc = rg.auroc(y[va], s)
    assert re_auroc == best["auroc"], "reloaded reader does not reproduce its validation AUROC"
    ci = rg.bootstrap_by_problem(y[va], s, prob[va])
    shutil.rmtree(JOBLIB_TMP, ignore_errors=True)
    check_frozen_probe()
    ea.write_atomic(STAGE_B_JSON, (json.dumps(
        {"layer": best["layer"], "C": best["C"], "validation_auroc": best["auroc"],
         "sha256": sha}, indent=1) + "\n").encode())
    write_csv(fits)
    write_md(counts, fits, best, sha, ci, len(data))
    print(f"stage B done: {len(fits)} fits ({sum(not f['converged'] for f in fits)} not "
          f"converged), fit wall {fit_wall:.0f}s, layer median {np.median(secs):.1f}s, "
          f"total {time.perf_counter() - t_start:.0f}s", flush=True)


# ---------------------------------------------------------------- outputs

def write_csv(fits) -> None:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["band", "layer", "C", "validation_auroc", "converged", "n_iter"])
    for r in fits:
        w.writerow([BAND, r["layer"], f"{r['C']:g}", f"{r['auroc']:.6f}", r["converged"],
                    r["n_iter"]])
    ea.write_atomic(LAYERS_CSV, buf.getvalue().encode())


def stage_a_facts() -> dict:
    j = lambda p: json.loads(p.read_text())  # noqa: E731
    with open(MANIFEST_7B, newline="") as fh:
        man = list(csv.DictReader(fh))
    return {"compat": j(ACT_7B / "compat.json"), "verify": j(ACT_7B / "verify.json"),
            "traces": len(man), "positions": sum(int(r["n_positions"]) for r in man),
            "bytes": sum(int(r["file_bytes"]) for r in man),
            "wiring_max": max(float(r["wiring_max_abs_diff"]) for r in man),
            "all_finite": all(r["finite"] == "True" for r in man),
            "layers": {int(r["n_layers"]) for r in man}, "hidden": {int(r["hidden_size"]) for r in man}}


def write_md(counts, fits, best, sha, ci, nbytes) -> None:
    import sklearn

    A = stage_a_facts()
    c = A["compat"]
    L = ["# Stronger rival: the 7B reader for band 6-10", "",
         "Computed by `scripts/extract_7b.py` (stage A) and `scripts/run_stronger_rival.py` "
         "(stage B). §8, D24; plan `handoffs/stronger-rival-plan.md` as approved. Band 6-10 "
         "only.", "",
         "## a. Data contact", "",
         "| Date | What was touched | What was not |", "|---|---|---|",
         f"| {CONTACT_DATE} | Stage A: the token ids of the 292 kept run traces (tokenizer check); "
         "the 7B's hidden states at every band 6-10 position, extracted without reading split or "
         "label (only the distance_set, trace_index and t columns of the examples file). Stage "
         "B: band 6-10 examples of the train and validation splits: labels and the 7B states of "
         "those traces. | The test split (no row loaded, no test-trace activation opened); every "
         "other band; the gate rows; the frozen 1.5B probe (hash-checked only). |", "",
         "## b. Stage A: model and extraction", "",
         f"- Model `{RIVAL_ID}` at revision `{RIVAL_REV}`, float16 on MPS (stored as bfloat16), "
         f"{sorted(A['layers'])[0]} layers, hidden size {sorted(A['hidden'])[0]}. Input: "
         "`lens.build_input_ids` (the stored prompt and generated ids, EOS 151643 restored "
         "where stripped), never re-tokenized. Saved: `hidden_states` entries 1-28 (entry 28 "
         "post-norm; entry 0, the embeddings, not saved), float16, one stored zip per trace in "
         "`activations/run_7b/` (gitignored).",
         f"- Token-id compatibility (S4), tokenizer files only, before any weight was loaded: "
         f"{c['traces']} traces, {c['distinct_ids']:,} distinct ids (prompt, output, EOS); ids "
         f"decoding differently: {len(c['ids_decoding_differently'])}; traces whose full decode "
         f"differs: {len(c['traces_full_decode_differs'])}; EOS ids {c['eos_a']} / {c['eos_b']}.",
         f"- Saved: {A['traces']} traces, {A['positions']:,} positions (equal to the band 6-10 "
         f"set of the examples file), {A['bytes'] / 1e9:.2f} GB. Per-trace sha256 in "
         "`results/stronger_rival_extraction_manifest.csv`.",
         f"- Checks: id at every saved position equals `output_token_ids[t]`; max |lm_head(entry "
         f"28) − logits| over all traces = {A['wiring_max']}; every hidden state and logit finite: "
         f"{A['all_finite']}; fresh-process re-extraction of "
         f"{len(A['verify'])} traces byte-identical: "
         f"{all(v['bytes_identical'] for v in A['verify'])}.", "",
         "## c. Stage B: counts", "",
         "| split | positives | negatives | problems |", "|---|---|---|---|"]
    for sp in ALLOWED_SPLITS:
        p, n, pr = counts[sp]
        L.append(f"| {sp} | {p} | {n} | {pr} |")
    nc = sum(not f["converged"] for f in fits)
    L += ["", "## d. Settings", "",
          f"The probe's recipe (§7, §18): per layer, z-score per dimension with the train mean "
          f"and SD; logistic regression, L2, lbfgs, max_iter {rp.MAX_ITER:,}, intercept, no class "
          f"weights; C ∈ {{{', '.join(f'{x:g}' for x in C_GRID)}}}; fitted on train, scored on "
          "validation by the decision function, single-threaded BLAS (`run_probes.fit_layer`). "
          f"Chosen: highest validation AUROC over the 28 layers × 5 C; ties to the lowest layer, "
          f"then the smallest C (`run_gate.best_fit`). {len(fits)} fits in "
          f"`results/stronger_rival_layers.csv`; not converged: {nc} (eligible, flagged). "
          f"scikit-learn {sklearn.__version__}, numpy {np.__version__}.", "",
          "## e. Per layer: best C and its validation AUROC", "",
          "| layer | C | validation AUROC | converged |", "|---|---|---|---|"]
    for Lyr in LAYERS:
        bf = rg.best_fit([f for f in fits if f["layer"] == Lyr])
        L.append(f"| {Lyr} | {bf['C']:g} | {bf['auroc']:.4f} | {bf['converged']} |")
    edge = best["C"] in (min(C_GRID), max(C_GRID))
    L += ["", "## f. Chosen 7B reader", "",
          "Fitted on train only; saved, hashed, reloaded from its file and re-scored on "
          "validation (equal as a float). Interval: 95%, 2,000 resamples of whole validation "
          "problems, seed 20261019 (as in `results/probes.md`).", "",
          "| band | layer | C | C at a grid edge | validation AUROC | 95% interval | file bytes | "
          "sha256 |", "|---|---|---|---|---|---|---|---|",
          f"| {BAND} | {best['layer']} | {best['C']:g} | {'yes' if edge else 'no'} | "
          f"{best['auroc']:.4f} | {ci[0]:.4f}–{ci[1]:.4f} | {nbytes:,} | `{sha}` |", "",
          "Selection result: the best of 140 settings on validation. No test number exists yet; "
          "the frozen 1.5B probe was not read in this stage.", ""]
    ea.write_atomic(MD, "\n".join(L).encode())


# ---------------------------------------------------------------- stage C (D22 / S7, S8)

REHEARSAL_JSON = ACT_7B / "stage_c_rehearsal.json"
RESULT_CSV = REPO_ROOT / "results" / "stronger_rival.csv"
PROBE_CSV = REPO_ROOT / "results" / "probe_layers.csv"
SELF = Path(__file__).resolve()
PROBE_VAL_MD = "0.8527"                  # results/probes.md table d, band 6-10
PROBE_TEST_REPR = "0.8683752768186852"   # results/verdicts.csv, band 6-10 (S8)
DRAWS_SHA = "1c267812f3760abd36a060fbb2ae72dd3c6fbb671b45e690dccca684ee72716b"  # verdicts.md §h
TEST_COUNTS = (3861, 7722)               # examples_summary.md table c, band 6-10 test
TEST_ONSETS = 762                        # examples_summary.md table d
DELTA, SECONDARY = 0.05, (0.03, 0.08)


def _rv():
    import run_verdicts as rv  # D22 functions: resample_draws, paired_resamples, interval, ...

    return rv


def reader_meta() -> dict:
    with np.load(READER, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def check_frozen_both() -> str:
    check_frozen_probe()
    want = json.loads(STAGE_B_JSON.read_text())["sha256"]
    got = ea.sha256_file(READER)
    assert got == want, "frozen 7B reader changed"
    return got


def gather_layer_7b(rows: list[dict], layer: int, manifest: dict[int, str],
                    forbidden: set[int]) -> np.ndarray:
    """[n_rows, 3584] float16, one 7B layer (1-28) only, in row order."""
    X = np.empty((len(rows), HIDDEN_7B), dtype=np.float16)
    by_trace: dict[int, list[int]] = {}
    for i, r in enumerate(rows):
        by_trace.setdefault(r["trace_index"], []).append(i)
    assert not (set(by_trace) & forbidden), "a forbidden trace would be opened"
    for tr, idxs in sorted(by_trace.items()):
        path = ACT_7B / f"gsm8k_{tr:04d}.npz"
        assert ea.sha256_file(path) == manifest[tr], f"sha256 mismatch {path.name}"
        with np.load(path, allow_pickle=False) as z:
            ts = z["t"]
        want = np.array([rows[i]["t"] for i in idxs])
        pos = np.searchsorted(ts, want)
        assert np.all(ts[pos] == want), tr
        hidden = rp.memmap_npz_member(path, "hidden")
        X[idxs] = np.asarray(hidden[pos, layer - 1, :])  # this layer only
        del hidden
    return X


def reader_scores(X: np.ndarray) -> np.ndarray:
    from threadpoolctl import threadpool_limits

    z = reader_meta()
    with threadpool_limits(limits=1):
        return rp.probe_scores(X, z["mean"], z["sd"], z["coef"], z["intercept"][0])


def score_both(rows: list[dict], forbidden: set[int]) -> tuple[np.ndarray, np.ndarray]:
    """Frozen 1.5B probe (its own layer, via run_verdicts) and frozen 7B reader (its layer)."""
    rv = _rv()
    assert not ({r["trace_index"] for r in rows} & forbidden), "a forbidden trace would be read"
    p_layer = int(rv.probe_meta(BAND)["layer"][0])
    s_p = rv.probe_scores_for(BAND, rv.gather_layer(rows, p_layer, rg.load_manifest(), set()))
    s_r = reader_scores(gather_layer_7b(rows, int(reader_meta()["layer"][0]),
                                        load_manifest_7b(), forbidden))
    return s_p, s_r


def s8_check(a: float) -> None:
    if repr(float(a)) != PROBE_TEST_REPR:
        sys.exit("STOP (S8): the frozen probe's test AUROC does not reproduce "
                 "results/verdicts.csv; the lift is not computed")


def draws_for_test() -> np.ndarray:
    rv = _rv()
    draws = rv.resample_draws(len(rv.split_problems("test")))
    assert hashlib.sha256(draws.tobytes()).hexdigest() == DRAWS_SHA, "draws differ from step 13"
    return draws


def lift_and_interval(y, s_p, s_r, prob, keys, draws) -> dict:
    rv = _rv()
    a_p, a_r = rv.paired_resamples(y, s_p, s_r, prob, keys, draws)
    lifts = a_p - a_r
    lo, hi = rv.interval(lifts)
    p, r = rg.auroc(y, s_p), rg.auroc(y, s_r)
    out = {"probe_auroc": p, "reader_auroc": r, "lift": p - r, "interval_low": lo,
           "interval_high": hi, "dropped_resamples": int(np.isnan(lifts).sum())}
    for d in (DELTA,) + SECONDARY:
        out[f"verdict_{d}"] = rv.verdict(lo, hi, d)
    out["claim_level_2"] = out[f"verdict_{DELTA}"] == "positive"
    return out


def rehearse() -> None:
    """Validation only: reproduce both validation AUROCs on file; record a pass."""
    rv = _rv()
    t0 = time.perf_counter()
    reader_sha = check_frozen_both()
    rows = load_rows(EXAMPLES_PATH, splits=("validation",))
    y = np.array([r["label"] for r in rows])
    assert (int((y == 1).sum()), int((y == 0).sum())) == EXPECTED["validation"]
    forbidden = set(json.loads(SPLIT_JSON.read_text())["test"])
    s_p, s_r = score_both(rows, forbidden)
    a_p, a_r = rg.auroc(y, s_p), rg.auroc(y, s_r)
    pz, rz = rv.probe_meta(BAND), reader_meta()
    b = json.loads(STAGE_B_JSON.read_text())
    probe_csv = rv.csv_value(PROBE_CSV, {"band": BAND, "layer": int(pz["layer"][0]),
                                          "C": float(pz["C"][0])})
    reader_csv = rv.csv_value(LAYERS_CSV, {"band": BAND, "layer": int(rz["layer"][0]),
                                            "C": float(rz["C"][0])})
    checks = {"probe_md_ok": f"{a_p:.4f}" == PROBE_VAL_MD, "probe_csv_ok": f"{a_p:.6f}" == probe_csv,
              "reader_csv_ok": f"{a_r:.6f}" == reader_csv,
              "reader_exact_ok": a_r == b["validation_auroc"]}
    ok = all(checks.values())
    check_frozen_both()
    rec = {"pass": ok, "script_sha256": ea.sha256_file(SELF), "checks": checks,
           "probe": {"layer": int(pz["layer"][0]), "C": float(pz["C"][0]), "auroc": a_p,
                     "md": PROBE_VAL_MD, "csv": probe_csv},
           "reader": {"layer": int(rz["layer"][0]), "C": float(rz["C"][0]), "auroc": a_r,
                      "csv": reader_csv, "sha256": reader_sha},
           "n_validation": len(rows), "seconds": round(time.perf_counter() - t0, 1)}
    ea.write_atomic(REHEARSAL_JSON, (json.dumps(rec, indent=1, sort_keys=True) + "\n").encode())
    print("rehearsal", "PASS" if ok else "FAIL", f"({rec['seconds']}s)", flush=True)
    if not ok:
        sys.exit(1)


def require_rehearsal() -> dict:
    assert REHEARSAL_JSON.exists(), "no rehearsal record: run --rehearse first"
    rec = json.loads(REHEARSAL_JSON.read_text())
    assert rec["pass"] is True, "rehearsal did not pass"
    assert rec["script_sha256"] == ea.sha256_file(SELF), "script changed since the rehearsal"
    return rec


def score_test_once() -> None:
    """The one test scoring of band 6-10 (S7, S8). Runs to the files with no pause."""
    rv = _rv()
    rec = require_rehearsal()
    assert "## g." not in MD.read_text(), "stage C has already been run"
    assert not RESULT_CSV.exists(), "stage C has already been run"
    before = {"probe": ea.sha256_file(PROBE), "reader": check_frozen_both()}
    keys = rv.split_problems("test")
    draws = draws_for_test()                                   # before any test row is read
    rows = load_rows(EXAMPLES_PATH, splits=("test",))
    y = np.array([r["label"] for r in rows])
    prob = np.array([r["trace_index"] for r in rows])
    assert (int((y == 1).sum()), int((y == 0).sum())) == TEST_COUNTS
    assert set(prob.tolist()) <= set(keys)
    rv_p = rv.probe_meta(BAND)
    s_p = rv.probe_scores_for(BAND, rv.gather_layer(rows, int(rv_p["layer"][0]),
                                                    rg.load_manifest(), set()))
    s8_check(rg.auroc(y, s_p))                             # before the 7B is scored
    s_r = reader_scores(gather_layer_7b(rows, int(reader_meta()["layer"][0]),
                                        load_manifest_7b(), set()))
    res = lift_and_interval(y, s_p, s_r, prob, keys, draws)
    res.update(band=BAND, n_problems=len(set(prob.tolist())), onsets=TEST_ONSETS,
               positives=int((y == 1).sum()), negatives=int((y == 0).sum()))
    after = {"probe": ea.sha256_file(PROBE), "reader": check_frozen_both()}
    assert before == after, "a frozen file changed"
    write_stage_c(res, rec, before, after)
    print("stage C done; results in results/stronger_rival.md and .csv", flush=True)


C_COLS = ["band", "n_problems", "onsets", "positives", "negatives", "probe_auroc",
          "reader_auroc", "lift", "interval_low", "interval_high", "verdict_0.05",
          "verdict_0.03", "verdict_0.08", "claim_level_2", "dropped_resamples"]


def write_stage_c(res, rec, before, after) -> None:
    rv = _rv()
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(C_COLS)
    w.writerow([rv.fmt(res[c]) for c in C_COLS])
    ea.write_atomic(RESULT_CSV, buf.getvalue().encode())
    f4 = lambda v: f"{v:.4f}"  # noqa: E731
    L = ["## g. Data contact: test", "",
         "| Date | What was touched | What was not |", "|---|---|---|",
         f"| {CONTACT_DATE} | The band 6-10 test examples: labels, the 1.5B activations at the "
         "frozen probe's layer and the 7B activations at the chosen reader's layer. Second "
         "scoring of these examples for the probe (§8, D24), first for the 7B reader. | Every "
         "other band; the gate rows; every other layer. |", "",
         "## h. Rehearsal on validation", "",
         f"Before any test row was read: probe validation AUROC {f4(rec['probe']['auroc'])} "
         f"(table {rec['probe']['md']}, CSV {rec['probe']['csv']}); 7B reader "
         f"{f4(rec['reader']['auroc'])} (CSV {rec['reader']['csv']}, exactly equal to stage B). "
         "Reproduced: yes. On test, the probe's AUROC reproduced `results/verdicts.csv` "
         f"exactly (`{PROBE_TEST_REPR}`, S8).", "",
         "## i. Verdict at δ = 0.05 (claim level 2)", "",
         "Lift = frozen 1.5B probe AUROC − 7B reader AUROC on the band 6-10 test examples. "
         "Interval: 98.75%, paired bootstrap over the 102 test problems, 10,000 resamples, "
         "`random.Random(20261019)`, the same draws as step 13 (D22).", "",
         "| band | probe AUROC | 7B reader AUROC | lift | δ | 98.75% interval | verdict | "
         "claim level 2 |", "|---|---|---|---|---|---|---|---|",
         f"| {BAND} | {f4(res['probe_auroc'])} | {f4(res['reader_auroc'])} | "
         f"{res['lift']:+.4f} | {DELTA} | {res['interval_low']:+.4f} to "
         f"{res['interval_high']:+.4f} | {res['verdict_0.05']} | "
         f"{'holds' if res['claim_level_2'] else 'does not hold'} |", "",
         "## j. Secondary verdicts, counts and frozen files", "",
         f"- Secondary (never the headline): δ = 0.03 {res['verdict_0.03']}; δ = 0.08 "
         f"{res['verdict_0.08']}.",
         f"- Counts: onsets {res['onsets']}, positives {res['positives']}, negatives "
         f"{res['negatives']}, problems with examples {res['n_problems']} of 102; dropped "
         f"resamples {res['dropped_resamples']}.",
         f"- Frozen probe sha256 before / after: `{before['probe']}` / `{after['probe']}`.",
         f"- Frozen 7B reader sha256 before / after: `{before['reader']}` / `{after['reader']}`.",
         ""]
    ea.write_atomic(MD, MD.read_bytes() + ("\n".join(L)).encode())


def main() -> None:
    if "--fit" in sys.argv:
        fit()
    elif "--rehearse" in sys.argv:
        rehearse()
    elif "--test" in sys.argv:
        score_test_once()
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
