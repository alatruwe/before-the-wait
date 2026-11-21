"""Stronger rival, stage A: 7B hidden states at the band 6-10 positions (§8, D24; label-blind).

DeepSeek-R1-Distill-Qwen-7B reads the 1.5B's stored token ids (never re-tokenized) and
its hidden states (entries 1-28 of hidden_states; entry 28 post-norm) are saved at
every band 6-10 position. Plan handoffs/stronger-rival-plan.md as approved.

    .venv/bin/python scripts/extract_7b.py --compat      # S4, tokenizer files only, offline
    .venv/bin/python scripts/extract_7b.py --download    # the only network step
    .venv/bin/python scripts/extract_7b.py --limit 3     # timed
    caffeinate -i nohup .venv/bin/python scripts/extract_7b.py > activations/run_7b/run.out 2>&1 &
    .venv/bin/python scripts/extract_7b.py --verify      # fresh process, 3 traces
    .venv/bin/python scripts/extract_7b.py --summarize   # manifest

Reads from examples/examples.csv.gz only the columns distance_set, trace_index and t
(rows of set 6-10); the split and label columns are never parsed. Checks per trace:
ids at the saved positions, lm_head(entry 28) == logits exactly, all hidden states and
logits finite (Q5; stop), saved position set == the band 6-10 set. Resumable.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ["HF_HOME"] = str(REPO_ROOT / ".hf_cache")
_offline = "0" if "--download" in sys.argv else "1"
os.environ["HF_HUB_OFFLINE"] = _offline
os.environ["TRANSFORMERS_OFFLINE"] = _offline
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402

import extract_activations as ea  # noqa: E402  (input_index, npz_bytes, write_atomic, ...)

RIVAL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
RIVAL_REV = "916b56a44061fd5cd7d6a8fb632557ed4f724f60"
BAND = "6-10"
EXAMPLES_PATH = ea.EXAMPLES_PATH
SPLIT_JSON = REPO_ROOT / "results" / "split.json"
TRACES_DIR = ea.TRACES_DIR
OUT_DIR = REPO_ROOT / "activations" / "run_7b"
CHECKS_PATH = OUT_DIR / "checks.jsonl"
PROGRESS_PATH = OUT_DIR / "progress.log"
VERIFY_PATH = OUT_DIR / "verify.json"
COMPAT_PATH = OUT_DIR / "compat.json"
DOWNLOAD_PATH = OUT_DIR / "download.json"
MANIFEST_PATH = REPO_ROOT / "results" / "stronger_rival_extraction_manifest.csv"
N_KEPT = 292
EOS_ID = 151643
SHARDS = {"model-00001-of-000002.safetensors": 8_606_596_466,
          "model-00002-of-000002.safetensors": 6_624_675_384}
ALLOW = ["*.safetensors", "model.safetensors.index.json", "config.json",
         "generation_config.json", "tokenizer*"]
PROGRESS_FIELDS = ["i", "trace_index", "seq_len", "n_positions", "seconds", "wiring_max_abs_diff"]


# ---------------------------------------------------------------- positions (label-blind)

def load_band_positions(path: Path, band: str = BAND) -> dict[int, list[int]]:
    """Unique sorted t per trace for rows of one distance set. Reads three columns only."""
    out: dict[int, set[int]] = {}
    with gzip.open(path, "rt", newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        i_set, i_tr, i_t = (header.index(c) for c in ("distance_set", "trace_index", "t"))
        for row in reader:
            if row[i_set] == band:
                out.setdefault(int(row[i_tr]), set()).add(int(row[i_t]))
    return {k: sorted(v) for k, v in sorted(out.items())}


def kept_traces() -> list[int]:
    s = json.loads(SPLIT_JSON.read_text())
    kept = sorted(s["train"] + s["validation"] + s["test"])
    assert len(kept) == N_KEPT
    return kept


def band_positions() -> dict[int, list[int]]:
    assert ea.sha256_file(EXAMPLES_PATH) == ea.EXAMPLES_SHA256
    return load_band_positions(EXAMPLES_PATH)


def act_path(idx: int, out_dir: Path = OUT_DIR) -> Path:
    return out_dir / f"gsm8k_{idx:04d}.npz"


def load_trace(idx: int) -> dict:
    return json.loads(ea.trace_path(idx).read_text())


# ---------------------------------------------------------------- S4: token-id compatibility

def snapshot_dir() -> Path:
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(RIVAL_ID, revision=RIVAL_REV, allow_patterns=ALLOW,
                                  local_files_only=(_offline == "1")))


def compat_check(tok_a, tok_b, traces: list[dict]) -> dict:
    ids: set[int] = set()
    full_diff = []
    for tr in traces:
        seq = list(tr["prompt_token_ids"]) + list(tr["output_token_ids"])
        ids.update(seq)
        if (tok_a.decode(seq, skip_special_tokens=False)
                != tok_b.decode(seq, skip_special_tokens=False)):
            full_diff.append(int(tr["index"]))
    ids.add(EOS_ID)
    bad = sorted(i for i in ids if tok_a.decode([i]) != tok_b.decode([i]))
    return {"traces": len(traces), "distinct_ids": len(ids), "ids_decoding_differently": bad,
            "traces_full_decode_differs": full_diff,
            "eos_a": tok_a.eos_token_id, "eos_b": tok_b.eos_token_id}


def run_compat() -> None:
    from transformers import AutoTokenizer

    from before_the_wait.generate import MODEL_ID, MODEL_REVISION

    tok_a = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    tok_b = AutoTokenizer.from_pretrained(RIVAL_ID, revision=RIVAL_REV)
    res = compat_check(tok_a, tok_b, [load_trace(i) for i in kept_traces()])
    res["ok"] = (not res["ids_decoding_differently"] and not res["traces_full_decode_differs"]
                 and res["eos_a"] == res["eos_b"] == EOS_ID)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ea.write_atomic(COMPAT_PATH, (json.dumps(res, indent=1) + "\n").encode())
    print(f"compat: traces {res['traces']}, distinct ids {res['distinct_ids']}, "
          f"differ {len(res['ids_decoding_differently'])}, full-decode differ "
          f"{len(res['traces_full_decode_differs'])}, ok={res['ok']}", flush=True)
    if not res["ok"]:
        sys.exit("STOP: token-id compatibility failed")


# ---------------------------------------------------------------- download

def run_download() -> None:
    import shutil

    assert json.loads(COMPAT_PATH.read_text())["ok"], "run --compat first"
    free = shutil.disk_usage(REPO_ROOT).free
    print(f"free disk before download: {free / 2**30:.1f} GiB", flush=True)
    t0 = time.perf_counter()
    d = snapshot_dir()
    sizes = {n: (d / n).stat().st_size for n in SHARDS}
    assert sizes == SHARDS, sizes
    rec = {"free_before_gib": round(free / 2**30, 1), "seconds": round(time.perf_counter() - t0),
           "shards": sizes, "free_after_gib": round(shutil.disk_usage(REPO_ROOT).free / 2**30, 1)}
    ea.write_atomic(DOWNLOAD_PATH, (json.dumps(rec, indent=1) + "\n").encode())
    print(json.dumps(rec), flush=True)


# ---------------------------------------------------------------- one trace

def extract_trace(model, trace: dict, ts: list[int]) -> tuple[dict, dict]:
    import torch

    from before_the_wait.lens import build_input_ids

    ids = build_input_ids(trace, EOS_ID)
    P = len(trace["prompt_token_ids"])
    out_ids = trace["output_token_ids"]
    n = len(out_ids)
    assert len(ids) == P + n + (1 if trace["eos_token_stripped"] else 0)
    assert ts == sorted(set(ts)) and ts and 0 <= ts[0] and ts[-1] < n
    for t in ts:
        assert ids[ea.input_index(P, t)] == out_ids[t], t  # check a
    ids_t = torch.tensor([ids], device=model.device)
    with torch.inference_mode():
        out = model(ids_t, output_hidden_states=True, use_cache=False)
        hs = out.hidden_states
        logits = out.logits[0]
        wiring = float((model.lm_head(hs[-1][0]).float() - logits.float()).abs().max())  # b
        finite = bool(torch.isfinite(logits).all()) and all(
            bool(torch.isfinite(h).all()) for h in hs[1:])                               # e
        idx = torch.tensor([ea.input_index(P, t) for t in ts], device=model.device)
        hidden = torch.stack([h[0].index_select(0, idx) for h in hs[1:]], dim=1)
        hidden_np = hidden.to("cpu").numpy().astype(np.float16, copy=False)
        del out, hs, logits, hidden
    finite = finite and bool(np.isfinite(hidden_np).all())
    arrays = {"hidden": hidden_np, "t": np.asarray(ts, dtype=np.int32)}
    checks = {"trace_index": int(trace["index"]), "prompt_len": P, "seq_len": len(ids),
              "n_positions": len(ts), "n_layers": int(hidden_np.shape[1]),
              "hidden_size": int(hidden_np.shape[2]), "wiring_max_abs_diff": wiring,
              "finite": finite}
    return arrays, checks


def load_model():
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(RIVAL_ID, revision=RIVAL_REV,
                                                 dtype=torch.float16).to("mps")
    model.eval()
    cfg = model.config
    print(f"loaded {RIVAL_ID} @ {RIVAL_REV}: dtype {model.dtype}, device {model.device}, "
          f"layers {cfg.num_hidden_layers}, hidden {cfg.hidden_size}", flush=True)
    return model


def load_checks() -> dict[int, dict]:
    if not CHECKS_PATH.exists():
        return {}
    return {r["trace_index"]: r for r in map(json.loads, CHECKS_PATH.read_text().splitlines())}


def run(limit: int | None) -> None:
    assert json.loads(COMPAT_PATH.read_text())["ok"], "run --compat first"
    pos = band_positions()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model = load_model()
    done = load_checks()
    if not PROGRESS_PATH.exists():
        PROGRESS_PATH.write_text("\t".join(PROGRESS_FIELDS) + "\n")
    order = list(pos)[:limit] if limit else list(pos)
    start, n_new = time.perf_counter(), 0
    for i, idx in enumerate(order):
        path = act_path(idx)
        if path.exists() and idx in done:
            continue
        t0 = time.perf_counter()
        arrays, checks = extract_trace(model, load_trace(idx), pos[idx])
        if checks["wiring_max_abs_diff"] != 0.0:
            sys.exit(f"STOP: wiring check failed on {idx}: {checks['wiring_max_abs_diff']}")
        if not checks["finite"]:
            sys.exit(f"STOP: non-finite hidden state or logit on {idx}")
        data = ea.npz_bytes(arrays)
        ea.write_atomic(path, data)
        checks.update(sha256=hashlib.sha256(data).hexdigest(), file_bytes=len(data),
                      seconds=round(time.perf_counter() - t0, 3))
        ea.append_jsonl(CHECKS_PATH, checks)
        with open(PROGRESS_PATH, "a") as fh:
            fh.write("\t".join(str(checks.get(k, i)) for k in PROGRESS_FIELDS) + "\n")
        n_new += 1
        print(f"[{i + 1:3d}/{len(order)}] gsm8k_{idx:04d} L={checks['seq_len']} "
              f"pos={checks['n_positions']} {checks['seconds']:.1f}s", flush=True)
    print(f"done; {n_new} new traces in {time.perf_counter() - start:.1f}s", flush=True)


def verify() -> None:
    pos = band_positions()
    model = load_model()
    res = []
    for idx in ea.verify_choice(pos):
        arrays, _ = extract_trace(model, load_trace(idx), pos[idx])
        same = hashlib.sha256(ea.npz_bytes(arrays)).hexdigest() == ea.sha256_file(act_path(idx))
        res.append({"trace_index": idx, "n_positions": len(pos[idx]), "bytes_identical": same})
        print(json.dumps(res[-1]), flush=True)
    ea.write_atomic(VERIFY_PATH, (json.dumps(res, indent=1) + "\n").encode())


def summarize() -> None:
    pos = band_positions()
    checks = load_checks()
    assert set(checks) == set(pos), "traces missing"
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    cols = ["trace_index", "seq_len", "n_positions", "n_layers", "hidden_size",
            "wiring_max_abs_diff", "finite", "file_bytes", "sha256"]
    w.writerow(cols)
    n_pos = 0
    for idx in sorted(pos):
        c = checks[idx]
        assert ea.sha256_file(act_path(idx)) == c["sha256"], idx
        with np.load(act_path(idx), allow_pickle=False) as z:
            assert z["t"].tolist() == pos[idx], idx                                # check c
        assert c["wiring_max_abs_diff"] == 0.0 and c["finite"]
        n_pos += c["n_positions"]
        w.writerow([c[k] for k in cols])
    ea.write_atomic(MANIFEST_PATH, buf.getvalue().encode())
    print(f"manifest: {len(pos)} traces, {n_pos} positions, "
          f"{sum(c['file_bytes'] for c in checks.values()) / 1e9:.2f} GB", flush=True)


def main() -> None:
    a = sys.argv
    if "--compat" in a:
        run_compat()
    elif "--download" in a:
        run_download()
    elif "--verify" in a:
        verify()
    elif "--summarize" in a:
        summarize()
    else:
        run(int(a[a.index("--limit") + 1]) if "--limit" in a else None)


if __name__ == "__main__":
    main()
