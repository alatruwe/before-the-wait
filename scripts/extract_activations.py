"""Step 10: hidden states at the example positions (pre-registration §4, §7, §8).

Teacher-forced replay of the 292 kept traces through the model, using the stored
token ids. Saves, per trace, layers 1-28 at that trace's example positions, the
per-token entropy of the generated tokens, and the entropy at the prompt
positions (added on approval, plan Q2). Nothing else.

Label-blind by construction: from examples/examples.csv.gz only the trace_index
and t columns are read. No label, split, band or distance is read anywhere, and
nothing here compares one group of positions with another.

Run from the repo root:

    .venv/bin/python scripts/extract_activations.py --limit 3     # timing; counts toward the run
    caffeinate -i nohup .venv/bin/python scripts/extract_activations.py > activations/run/run.out 2>&1 &
    .venv/bin/python scripts/extract_activations.py --verify      # fresh process, 3 traces (check e)
    .venv/bin/python scripts/extract_activations.py --summarize   # no model: checks c, d, f; summary

Re-running skips traces whose file and check record already exist.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import os
import shutil
import statistics
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ["HF_HOME"] = str(REPO_ROOT / ".hf_cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402

EXAMPLES_PATH = REPO_ROOT / "examples" / "examples.csv.gz"
EXAMPLES_SHA256 = "d05d940f4ac4396f236c5bd970b1268677fe3ba9ea8009b8835073bdb34668f1"
TRACES_DIR = REPO_ROOT / "traces" / "run"
OUT_DIR = REPO_ROOT / "activations" / "run"
CHECKS_PATH = OUT_DIR / "checks.jsonl"
PROGRESS_PATH = OUT_DIR / "progress.log"
VERIFY_PATH = OUT_DIR / "verify.json"
SUMMARY_PATH = REPO_ROOT / "results" / "extraction_summary.md"
MANIFEST_PATH = REPO_ROOT / "results" / "extraction_manifest.csv"

N_TRACES = 292
N_POSITIONS = 206_989
N_LAYERS = 28
HIDDEN = 1536
TEMPERATURE = 0.6
TOP_P = 0.95
REPLAY_FLOOR = 0.99
CHUNK = 512
HIST_BINS = 10_000  # replay-probability histogram, for the pooled median
ZIP_DATE = (1980, 1, 1, 0, 0, 0)
CONTACT_DATE = "2026-11-14"
PROGRESS_FIELDS = ["i", "trace_index", "seq_len", "n_positions", "seconds",
                   "wiring_max_abs_diff", "replay_share"]


# ---------------------------------------------------------------- pure helpers

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_positions(path: Path) -> dict[int, list[int]]:
    """Unique sorted t per trace. Reads the trace_index and t columns only."""
    out: dict[int, set[int]] = {}
    with gzip.open(path, "rt", newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        i_tr, i_t = header.index("trace_index"), header.index("t")
        for row in reader:
            out.setdefault(int(row[i_tr]), set()).add(int(row[i_t]))
    return {k: sorted(v) for k, v in sorted(out.items())}


def input_index(prompt_len: int, t: int) -> int:
    """Generated position t sits at input index P + t."""
    return prompt_len + t


def stack_layers(hidden_states: tuple) -> Any:
    """hidden_states[1..] -> [L, n_layers, H]; entry 0 (embeddings) dropped."""
    import torch

    return torch.stack([h[0] for h in hidden_states[1:]], dim=1)


def entropy_rows(logits: Any) -> Any:
    """Entropy in nats of softmax(float32 logits), row-wise (E1: T 1, no top-p)."""
    import torch

    logp = torch.log_softmax(logits.float(), dim=-1)
    return -(logp.exp() * logp).sum(dim=-1)


def replay_prob_rows(logits: Any, token_ids: Any, *, temperature: float, top_p: float) -> Any:
    """Probability of each token under the processors generate() applied.

    transformers' TemperatureLogitsWarper then TopPLogitsWarper (generation used
    no top-k), softmax of the processed logits, on CPU in float32. Not
    lens.sampling_probability: that function drops the token whose cumulative
    mass crosses top_p (ruling on handoff 10, Q1, option A; src/ untouched).
    """
    import torch
    from transformers.generation.logits_process import (
        TemperatureLogitsWarper,
        TopPLogitsWarper,
    )

    x = logits.detach().to("cpu", torch.float32).clone()
    x = TemperatureLogitsWarper(temperature)(None, x)
    x = TopPLogitsWarper(top_p)(None, x)
    probs = torch.softmax(x, dim=-1)
    return torch.gather(probs, -1, token_ids.to("cpu")[:, None])[:, 0]


def npz_bytes(arrays: dict[str, np.ndarray]) -> bytes:
    """A stored (uncompressed) .npz with fixed timestamps: same arrays, same bytes."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_STORED) as zf:
        for name, arr in arrays.items():
            member = io.BytesIO()
            np.lib.format.write_array(member, np.ascontiguousarray(arr), allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=ZIP_DATE)
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o644 << 16
            zf.writestr(info, member.getvalue())
    return buf.getvalue()


def write_atomic(path: Path, data: bytes) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def append_jsonl(path: Path, record: dict) -> None:
    with open(path, "a") as fh:
        fh.write(json.dumps(record) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def load_checks() -> dict[int, dict]:
    if not CHECKS_PATH.exists():
        return {}
    out = {}
    for line in CHECKS_PATH.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            out[r["trace_index"]] = r
    return out


def trace_path(idx: int) -> Path:
    return TRACES_DIR / f"gsm8k_{idx:04d}.json"


def act_path(idx: int, out_dir: Path = OUT_DIR) -> Path:
    return out_dir / f"gsm8k_{idx:04d}.npz"


# ---------------------------------------------------------------- one trace

def extract_trace(model: Any, trace: dict, eos_id: int, ts: list[int],
                  *, chunk: int = CHUNK) -> tuple[dict[str, np.ndarray], dict]:
    """Forward pass, gather, entropy, wiring and replay checks for one trace."""
    import torch

    from before_the_wait.lens import build_input_ids

    ids = build_input_ids(trace, eos_id)
    P = len(trace["prompt_token_ids"])
    out_ids = trace["output_token_ids"]
    n = len(out_ids)
    L = len(ids)
    assert L == P + n + (1 if trace["eos_token_stripped"] else 0), (L, P, n)
    assert ts == sorted(set(ts)) and ts and 0 <= ts[0] and ts[-1] < n
    for t in ts:
        assert ids[input_index(P, t)] == out_ids[t], t

    ids_t = torch.tensor([ids], device=model.device)
    with torch.inference_mode():
        out = model(ids_t, output_hidden_states=True, use_cache=False)
        hs = out.hidden_states
        n_layers = len(hs) - 1
        if getattr(model, "_btw_real", False):
            assert n_layers == N_LAYERS, len(hs)
        logits = out.logits[0]

        # a. wiring: lm_head over the last entry reproduces the logits exactly;
        # applying the norm again does not.
        direct = model.lm_head(hs[-1][0])
        wiring = float((direct.float() - logits.float()).abs().max())
        del direct
        rows = slice(0, min(chunk, L))
        normed = model.lm_head(model.model.norm(hs[-1][0][rows]))
        norm_twice = float((normed.float() - logits[rows].float()).abs().max())
        del normed

        # gather layers 1..28 at P + t
        idx = torch.tensor([input_index(P, t) for t in ts], device=model.device)
        hidden = torch.stack([h[0].index_select(0, idx) for h in hs[1:]], dim=1)
        hidden_np = hidden.to("cpu").numpy().astype(np.float16, copy=False)
        del hs, hidden

        # entropy at every input position (prompt and generated)
        ent = torch.cat([entropy_rows(logits[s:s + chunk]) for s in range(0, L, chunk)])
        ent_np = ent.to("cpu").numpy().astype(np.float32)
        prompt_entropy = ent_np[:P].copy()
        entropy = ent_np[P:P + n].copy()

        # b. replay: token j (generated ids, plus the restored EOS) under the
        # generation settings at the preceding position P + j - 1
        targets = ids[P:]
        tgt = torch.tensor(targets, device=model.device)
        pr = []
        for s in range(0, len(targets), chunk):
            rows_l = logits[P - 1 + s:P - 1 + s + len(targets[s:s + chunk])]
            pr.append(replay_prob_rows(rows_l, tgt[s:s + chunk],
                                        temperature=TEMPERATURE, top_p=TOP_P))
        replay = torch.cat(pr).to("cpu").numpy().astype(np.float64)
        del logits, out

    assert hidden_np.shape == (len(ts), n_layers, hidden_np.shape[2])
    hist = np.bincount(np.minimum((replay * HIST_BINS).astype(np.int64), HIST_BINS - 1),
                       minlength=HIST_BINS)
    abs_sum = np.abs(hidden_np.astype(np.float32)).sum(axis=0).sum(axis=-1, dtype=np.float64)
    arrays = {
        "hidden": hidden_np,
        "t": np.asarray(ts, dtype=np.int32),
        "entropy": entropy,
        "prompt_entropy": prompt_entropy,
    }
    checks = {
        "trace_index": int(trace["index"]),
        "prompt_len": P, "n_generated": n, "seq_len": L,
        "eos_restored": bool(trace["eos_token_stripped"]),
        "n_positions": len(ts),
        "wiring_max_abs_diff": wiring,
        "norm_twice_max_abs_diff": norm_twice,
        "replay_n": int(len(replay)),
        "replay_n_zero": int((replay <= 0).sum()),
        "replay_share": float((replay > 0).mean()),
        "replay_median": float(np.median(replay)),
        "replay_hist": {str(i): int(c) for i, c in enumerate(hist) if c},
        "hidden_finite": bool(np.isfinite(hidden_np).all()),
        "entropy_finite": bool(np.isfinite(entropy).all() and np.isfinite(prompt_entropy).all()),
        "layer_abs_sum": [float(x) for x in abs_sum],
    }
    return arrays, checks


def read_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def kept_with_positions() -> dict[int, list[int]]:
    actual = sha256_file(EXAMPLES_PATH) if EXAMPLES_PATH.exists() else None
    if actual is None:
        sys.exit("examples/examples.csv.gz missing: rebuild with scripts/build_examples.py")
    if actual != EXAMPLES_SHA256:
        sys.exit(f"examples.csv.gz sha256 mismatch: {actual}")
    pos = load_positions(EXAMPLES_PATH)
    assert len(pos) == N_TRACES, len(pos)
    assert sum(len(v) for v in pos.values()) == N_POSITIONS
    return pos


def load_model():
    from before_the_wait.generate import MODEL_ID, MODEL_REVISION, load_model_and_tokenizer

    print(f"loading {MODEL_ID} @ {MODEL_REVISION}", flush=True)
    tokenizer, model = load_model_and_tokenizer()
    model._btw_real = True
    assert model.config.num_hidden_layers == N_LAYERS
    assert model.config.hidden_size == HIDDEN
    print(f"dtype {model.dtype}, device {model.device}", flush=True)
    return int(tokenizer.eos_token_id), model


# ---------------------------------------------------------------- modes

def run(limit: int | None) -> None:
    pos = kept_with_positions()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    eos_id, model = load_model()
    done = load_checks()
    if not PROGRESS_PATH.exists():
        PROGRESS_PATH.write_text("\t".join(PROGRESS_FIELDS) + "\n")
    order = list(pos)
    if limit is not None:
        order = order[:limit]
    start = time.perf_counter()
    n_new = 0
    for i, idx in enumerate(order):
        path = act_path(idx)
        if path.exists() and idx in done:
            print(f"[{i + 1:3d}/{len(order)}] {path.name} exists, skipping", flush=True)
            continue
        trace = json.loads(trace_path(idx).read_text())
        t0 = time.perf_counter()
        arrays, checks = extract_trace(model, trace, eos_id, pos[idx])
        if checks["wiring_max_abs_diff"] != 0.0:
            sys.exit(f"STOP: wiring check failed on {idx}: {checks['wiring_max_abs_diff']}")
        if not checks["norm_twice_max_abs_diff"] > 0.0:
            sys.exit(f"STOP: re-normed logits equal the logits on {idx}")
        if checks["replay_share"] < REPLAY_FLOOR:
            sys.exit(f"STOP: replay share {checks['replay_share']:.4f} < 0.99 on {idx}")
        if not (checks["hidden_finite"] and checks["entropy_finite"]):
            sys.exit(f"STOP: non-finite values on {idx}")
        data = npz_bytes(arrays)
        write_atomic(path, data)
        checks["sha256"] = hashlib.sha256(data).hexdigest()
        checks["file_bytes"] = len(data)
        checks["seconds"] = round(time.perf_counter() - t0, 3)
        append_jsonl(CHECKS_PATH, checks)
        with open(PROGRESS_PATH, "a") as fh:
            fh.write("\t".join(str(x) for x in [
                i, idx, checks["seq_len"], checks["n_positions"], checks["seconds"],
                checks["wiring_max_abs_diff"], f"{checks['replay_share']:.6f}"]) + "\n")
        n_new += 1
        print(f"[{i + 1:3d}/{len(order)}] gsm8k_{idx:04d} L={checks['seq_len']} "
              f"pos={checks['n_positions']} {checks['seconds']:.1f}s "
              f"wiring={checks['wiring_max_abs_diff']} replay={checks['replay_share']:.4f}",
              flush=True)
    print(f"done; {n_new} new traces in {time.perf_counter() - start:.1f}s", flush=True)


def verify_choice(pos: dict[int, list[int]]) -> list[int]:
    kept = sorted(pos)
    most = min(kept, key=lambda k: (-len(pos[k]), k))
    out = []
    for k in (kept[0], kept[len(kept) // 2], most):
        if k not in out:
            out.append(k)
    return out


def verify() -> None:
    pos = kept_with_positions()
    eos_id, model = load_model()
    results = []
    for idx in verify_choice(pos):
        saved = read_npz(act_path(idx))
        trace = json.loads(trace_path(idx).read_text())
        arrays, _ = extract_trace(model, trace, eos_id, pos[idx])
        diffs = {k: float(np.abs(arrays[k].astype(np.float64) - saved[k].astype(np.float64)).max())
                 for k in arrays}
        same_bytes = hashlib.sha256(npz_bytes(arrays)).hexdigest() == sha256_file(act_path(idx))
        results.append({"trace_index": idx, "n_positions": len(pos[idx]),
                        "max_abs_diff": diffs, "bytes_identical": same_bytes})
        print(json.dumps(results[-1]), flush=True)
    VERIFY_PATH.write_text(json.dumps(results, indent=1) + "\n")


def hist_median(hist: np.ndarray) -> float:
    """Lower edge of the bin holding the median (resolution 1 / HIST_BINS)."""
    c = np.cumsum(hist)
    return float(np.searchsorted(c, (c[-1] + 1) // 2) / HIST_BINS)


def summarize() -> None:
    pos = kept_with_positions()
    checks = load_checks()
    files = sorted(OUT_DIR.glob("gsm8k_*.npz"))
    assert len(files) == N_TRACES and sorted(checks) == list(pos), (len(files), len(checks))

    saved_pairs: set[tuple[int, int]] = set()
    layer_abs = np.zeros(N_LAYERS, dtype=np.float64)
    n_saved = 0
    manifest = []
    shapes_ok = finite_ok = True
    for idx in pos:
        path = act_path(idx)
        z = read_npz(path)
        c = checks[idx]
        h, t, e, pe = z["hidden"], z["t"], z["entropy"], z["prompt_entropy"]
        shapes_ok &= (sorted(z) == ["entropy", "hidden", "prompt_entropy", "t"]
                      and h.dtype == np.float16 and t.dtype == np.int32
                      and e.dtype == np.float32 and pe.dtype == np.float32
                      and h.shape == (len(t), N_LAYERS, HIDDEN)
                      and e.shape == (c["n_generated"],) and pe.shape == (c["prompt_len"],)
                      and bool(np.all(np.diff(t) > 0)))
        finite_ok &= bool(np.isfinite(h).all() and np.isfinite(e).all() and np.isfinite(pe).all())
        layer_abs += np.abs(h.astype(np.float32)).sum(axis=0).sum(axis=-1, dtype=np.float64)
        n_saved += len(t)
        saved_pairs.update((idx, int(x)) for x in t)
        sha = sha256_file(path)
        assert sha == c["sha256"], idx
        manifest.append((idx, len(t), c["n_generated"], sha))
    expected = {(k, t) for k, v in pos.items() for t in v}
    coverage_ok = saved_pairs == expected and n_saved == N_POSITIONS
    layer_mean = layer_abs / (n_saved * HIDDEN)
    run_mean = np.sum([c["layer_abs_sum"] for c in checks.values()], axis=0) / (n_saved * HIDDEN)

    hist = np.zeros(HIST_BINS, dtype=np.int64)
    for c in checks.values():
        for k, v in c["replay_hist"].items():
            hist[int(k)] += v
    rn = sum(c["replay_n"] for c in checks.values())
    rz = sum(c["replay_n_zero"] for c in checks.values())
    shares = [c["replay_share"] for c in checks.values()]
    wiring = [c["wiring_max_abs_diff"] for c in checks.values()]
    norm2 = [c["norm_twice_max_abs_diff"] for c in checks.values()]
    verify_res = json.loads(VERIFY_PATH.read_text()) if VERIFY_PATH.exists() else None
    du = sum(p.stat().st_size for p in OUT_DIR.iterdir() if p.is_file())
    free = shutil.disk_usage(OUT_DIR).free
    secs = sum(c["seconds"] for c in checks.values())

    with open(MANIFEST_PATH, "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["trace_index", "n_positions", "n_generated_tokens", "sha256"])
        w.writerows(manifest)

    L: list[str] = []
    a = L.append
    a("# Extraction summary: layers 1-28 at the example positions")
    a("")
    a("Computed by `scripts/extract_activations.py` from `traces/run/` and the "
      "`trace_index`, `t` columns of `examples/examples.csv.gz` "
      f"(sha256 `{EXAMPLES_SHA256}`). Arrays are in `activations/run/` (not in git); "
      "`results/extraction_manifest.csv` pins each file by sha256.")
    a("")
    a("## Data contact")
    a("")
    a("| Date | What was touched | What was not |")
    a("|---|---|---|")
    a(f"| {CONTACT_DATE} | Teacher-forced pass over the token ids of the {N_TRACES} kept traces; "
      f"hidden states (layers 1-28) saved at the {N_POSITIONS:,} unique example positions, all "
      "three splits; next-token logits used for per-token entropy and for the label-blind wiring "
      "and replay checks. | No label, split or band read; no probe, no rival, no score; no "
      "statistic comparing positives with negatives or one split with another. |")
    a("")
    a("## Settings and indexing")
    a("")
    a("- Model, revision and dtype from `generate.load_model_and_tokenizer` (fp16, MPS, offline).")
    a("- Input = `lens.build_input_ids(trace, eos_id)`: prompt ids + generated ids, plus the "
      "EOS that generation stripped. Never re-tokenized.")
    a("- Example t is input index `P + t` (P = prompt length); the id there was asserted equal "
      "to `output_token_ids[t]` at every saved position.")
    a("- `hidden_states` has 29 entries; entry 0 (embeddings) is not saved; entries 1-28 are "
      "layers 1-28. Entry 28 already carries the final norm and is saved as is.")
    a("- Per trace, `gsm8k_XXXX.npz` (stored zip, fixed timestamps): `hidden` float16 "
      "[n_pos, 28, 1536]; `t` int32 ascending; `entropy` float32 [n generated], entry i from "
      "the logits at P + i; `prompt_entropy` float32 [P], entry i from the logits at input i "
      "(added on approval). Entropy: softmax at temperature 1, no top-p, float32, nats (E1).")
    a("")
    a("## a. Wiring (every trace)")
    a("")
    a(f"- max |lm_head(entry 28) − logits| over all positions: max over traces = {max(wiring)}; "
      f"traces at exactly 0.0: {sum(x == 0.0 for x in wiring)} / {N_TRACES}.")
    a(f"- Seam: max |lm_head(norm(entry 28)) − logits| (first {CHUNK} positions) is nonzero on "
      f"{sum(x > 0 for x in norm2)} / {N_TRACES} traces (min {min(norm2):.3f}, "
      f"max {max(norm2):.3f}); the norm is not applied twice.")
    a("")
    a("## b. Replay (every trace)")
    a("")
    a(f"Probability of each generated token (and the restored EOS) under T {TEMPERATURE}, "
      f"top-p {TOP_P} at the preceding position, through transformers' "
      "`TemperatureLogitsWarper` and `TopPLogitsWarper` (the processors generation applied; "
      "no top-k), on CPU in float32.")
    a("")
    a("The first run stopped on trace 1 (gsm8k_0000, share 0.9596), because "
      "`lens.sampling_probability` drops the token whose cumulative mass crosses top-p, so "
      "tokens generation could sample scored 0; under transformers' own warpers that trace "
      "replays at 1.0, and check b now uses those warpers, with `src/` unchanged and the 0.99 "
      "floor as written. c3 Part B's informational tempered numbers "
      "(`sampling_tempered_median`, `sampling_tempered_p5`) came from the faulty function; "
      "the c3 gate used raw probabilities and is unaffected.")
    a("")
    a("| quantity | value |")
    a("|---|---|")
    a(f"| tokens checked | {rn:,} |")
    a(f"| tokens with p > 0 | {rn - rz:,} ({(rn - rz) / rn:.6f}) |")
    a(f"| tokens with p = 0 | {rz:,} |")
    a(f"| traces at 100% | {sum(s == 1.0 for s in shares)} / {N_TRACES} |")
    a(f"| lowest per-trace share | {min(shares):.6f} (floor 0.99) |")
    a(f"| traces below 0.99 | {sum(s < REPLAY_FLOOR for s in shares)} |")
    a(f"| pooled median p (bin width 1e-4, lower edge) | {hist_median(hist):.4f} |")
    a(f"| median of per-trace medians | {statistics.median(c['replay_median'] for c in checks.values()):.6f} |")
    a("")
    a("## c. Coverage")
    a("")
    a(f"- Trace files: {len(files)} (expected {N_TRACES}).")
    a(f"- Saved positions: {n_saved:,} (expected {N_POSITIONS:,}).")
    a(f"- Saved (trace_index, t) set equals the set from examples.csv.gz: {coverage_ok}.")
    a("")
    a("## d. Values")
    a("")
    a(f"- All finite (hidden, entropy, prompt_entropy): {finite_ok}.")
    a(f"- Dtypes and shapes as specified, t strictly ascending: {shapes_ok}.")
    a(f"- Per-layer mean |value|, pooled over all {n_saved:,} saved positions and 1,536 "
      "dimensions (recomputed from the files; equals the in-run accumulation: "
      f"{bool(np.allclose(layer_mean, run_mean, rtol=1e-9))}):")
    a("")
    a("| layer | mean abs |")
    a("|---|---|")
    for k, v in enumerate(layer_mean, start=1):
        a(f"| {k} | {v:.4f} |")
    a("")
    a("## e. Determinism (fresh process)")
    a("")
    if verify_res is None:
        a("Not run yet.")
    else:
        a("| trace | positions | hidden | entropy | prompt_entropy | t | file bytes identical |")
        a("|---|---|---|---|---|---|---|")
        for r in verify_res:
            d = r["max_abs_diff"]
            a(f"| gsm8k_{r['trace_index']:04d} | {r['n_positions']} | {d['hidden']} | "
              f"{d['entropy']} | {d['prompt_entropy']} | {d['t']} | {r['bytes_identical']} |")
        a("")
        a("Traces: smallest kept index, median kept index, most positions (from metadata only).")
    a("")
    a("## f. Disk")
    a("")
    a(f"- `activations/run/`: {du / 1e9:.2f} GB ({du:,} bytes).")
    a(f"- Free after: {free / 2**30:.1f} GiB.")
    a(f"- Extraction time (sum over traces): {secs / 60:.1f} min.")
    a("")
    SUMMARY_PATH.write_text("\n".join(L))
    print(SUMMARY_PATH.read_text())
    assert coverage_ok and shapes_ok and finite_ok
    assert all(x == 0.0 for x in wiring) and min(shares) >= REPLAY_FLOOR


def main() -> None:
    if "--summarize" in sys.argv:
        summarize()
    elif "--verify" in sys.argv:
        verify()
    else:
        limit = None
        if "--limit" in sys.argv:
            limit = int(sys.argv[sys.argv.index("--limit") + 1])
        run(limit)


if __name__ == "__main__":
    main()
