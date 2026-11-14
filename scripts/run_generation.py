"""Step D: generate the run's 320 GSM8K traces (pre-registration §4).

Run from the repo root, detached and under caffeinate:

    caffeinate -i nohup .venv/bin/python scripts/run_generation.py > traces/run/run.out 2>&1 &

    .venv/bin/python scripts/run_generation.py --draw-only   # write/check draw.json, no model

Re-running skips problems whose trace file already exists, so an interrupted
run resumes. Generation only: text and token ids, no hidden states. Settings
come from before_the_wait.generate, unchanged from the pilot.
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ["HF_HOME"] = str(REPO_ROOT / ".hf_cache")
# Everything is cached; the run never needs the network.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
sys.path.insert(0, str(REPO_ROOT / "src"))

PILOT_DIR = REPO_ROOT / "traces" / "pilot"
OUT_DIR = REPO_ROOT / "traces" / "run"
DRAW_PATH = OUT_DIR / "draw.json"
PROGRESS_PATH = OUT_DIR / "progress.log"

GSM8K_TEST_ROWS = 1319
N_TRACES = 320  # §4, computed 2026-10-19
DRAW_SEED = 20261018  # §18, problem sample
DRAW_CALL = (
    "random.Random(20261018).sample("
    "sorted(set(range(1319)) - set(pilot_idx)), 320)"
)
PROGRESS_FIELDS = [
    "i", "index", "seed", "new_tokens", "seconds", "tok_per_s",
    "finish_reason", "reasoning_complete",
]


def pilot_indices() -> list[int]:
    idx = sorted(int(p.stem.split("_")[1]) for p in PILOT_DIR.glob("gsm8k_*.json"))
    assert len(idx) == 15, f"expected 15 pilot gsm8k traces, found {len(idx)}"
    return idx


def compute_draw(pilot_idx: list[int]) -> list[int]:
    return random.Random(DRAW_SEED).sample(
        sorted(set(range(GSM8K_TEST_ROWS)) - set(pilot_idx)), N_TRACES
    )


def write_or_check_draw() -> list[int]:
    pilot_idx = pilot_indices()
    draw = compute_draw(pilot_idx)
    assert len(draw) == N_TRACES and not set(draw) & set(pilot_idx)
    record = {
        "indices": draw,
        "seed": DRAW_SEED,
        "excluded_pilot_indices": pilot_idx,
        "call": DRAW_CALL,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if DRAW_PATH.exists():
        on_disk = json.loads(DRAW_PATH.read_text())
        assert on_disk == record, "draw.json on disk differs from the recomputed draw"
    else:
        DRAW_PATH.write_text(json.dumps(record, indent=1) + "\n")
    return draw


def main() -> None:
    draw = write_or_check_draw()
    print(f"draw: {len(draw)} indices, first 5 {draw[:5]}")
    if "--draw-only" in sys.argv:
        return

    from datasets import load_dataset

    from before_the_wait.generate import (
        MODEL_ID,
        MODEL_REVISION,
        SEED_BASE,
        generate_trace,
        gsm8k_gold,
        load_model_and_tokenizer,
    )

    print(f"loading {MODEL_ID} @ {MODEL_REVISION}", flush=True)
    tokenizer, model = load_model_and_tokenizer()
    print(f"dtype {model.dtype}, device {model.device}", flush=True)

    gsm = load_dataset("openai/gsm8k", "main", split="test")
    assert len(gsm) == GSM8K_TEST_ROWS, len(gsm)

    if not PROGRESS_PATH.exists():
        PROGRESS_PATH.write_text("\t".join(PROGRESS_FIELDS) + "\n")

    run_start = time.perf_counter()
    for i, di in enumerate(draw):
        path = OUT_DIR / f"gsm8k_{di:04d}.json"
        if path.exists():
            print(f"[{i + 1:3d}/{N_TRACES}] {path.name} exists, skipping", flush=True)
            continue
        seed = SEED_BASE + i
        row = gsm[di]
        trace = generate_trace(
            tokenizer, model,
            dataset="gsm8k", index=di, problem=row["question"],
            gold_answer=gsm8k_gold(row["answer"]), seed=seed,
        )
        # Write via a temp file so a kill mid-write never leaves a partial trace
        # that the resume logic would then skip.
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(trace.to_dict(), ensure_ascii=False, indent=1))
        tmp.replace(path)
        fields = [
            i, di, seed, trace.new_tokens, f"{trace.seconds:.1f}",
            f"{trace.tokens_per_second:.1f}", trace.finish_reason,
            trace.reasoning_complete,
        ]
        with PROGRESS_PATH.open("a") as f:
            f.write("\t".join(str(x) for x in fields) + "\n")
        print(
            f"[{i + 1:3d}/{N_TRACES}] {path.name}  seed={seed}  {trace.new_tokens:4d} tok  "
            f"{trace.seconds:6.1f}s  {trace.tokens_per_second:5.1f} tok/s  "
            f"{trace.finish_reason:6s}  complete={trace.reasoning_complete}",
            flush=True,
        )

    print(f"\ndone; total {time.perf_counter() - run_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
