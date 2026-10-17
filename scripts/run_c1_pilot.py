"""c1 pilot: generate 30 traces (15 GSM8K + 15 MATH500) and save them.

Run from the repo root, under caffeinate so the machine stays awake:

    caffeinate -i .venv/bin/python scripts/run_c1_pilot.py

Re-running skips problems whose trace file already exists, so an interrupted
run resumes. Generation only — the analysis is a separate script so an
analysis bug never costs a regeneration.
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
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch  # noqa: E402
from datasets import load_dataset  # noqa: E402

from before_the_wait.generate import (  # noqa: E402
    MODEL_ID,
    MODEL_REVISION,
    SEED_BASE,
    THINK_CLOSE,
    THINK_OPEN,
    build_prompt,
    generate_trace,
    gsm8k_gold,
    load_model_and_tokenizer,
)

OUT_DIR = REPO_ROOT / "traces" / "pilot"
N_PER_DATASET = 15
DRAW_SEED = 20261016


def draw_indices(n_rows: int, k: int = N_PER_DATASET) -> list[int]:
    return random.Random(DRAW_SEED).sample(range(n_rows), k)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"loading {MODEL_ID} @ {MODEL_REVISION}")
    tokenizer, model = load_model_and_tokenizer()
    print(f"dtype {model.dtype}, device {model.device}")

    # Steer 4: record the special-token ids we depend on.
    think_ids = tokenizer.encode(THINK_OPEN, add_special_tokens=False)
    close_ids = tokenizer.encode(THINK_CLOSE, add_special_tokens=False)
    print(f"<think> ids = {think_ids}, </think> ids = {close_ids}")
    print(f"eos_token_id = {tokenizer.eos_token_id} ({tokenizer.eos_token!r})")

    print("loading datasets")
    gsm = load_dataset("openai/gsm8k", "main", split="test")
    math500 = load_dataset("HuggingFaceH4/MATH-500", split="test")
    print(f"gsm8k test rows = {len(gsm)}, math500 test rows = {len(math500)}")

    gsm_idx = draw_indices(len(gsm))
    math_idx = draw_indices(len(math500))
    print(f"gsm8k draw  = {gsm_idx}")
    print(f"math500 draw = {math_idx}")

    jobs: list[tuple[str, int, str, str]] = []
    for di in gsm_idx:
        row = gsm[di]
        jobs.append(("gsm8k", di, row["problem"] if "problem" in row else row["question"],
                     gsm8k_gold(row["answer"])))
    for di in math_idx:
        row = math500[di]
        jobs.append(("math500", di, row["problem"], str(row["answer"])))

    # Steer 5: determinism check before the full run.
    determinism_check(tokenizer, model, jobs[0])

    # The prompt suffix, recorded exactly (steer 4 / acceptance criterion).
    sample_prompt = build_prompt(tokenizer, jobs[0][2])
    print(f"\nPROMPT SUFFIX: {sample_prompt[-60:]!r}")
    print(f"template emits <think>: {sample_prompt.rstrip().endswith(THINK_OPEN)}\n")

    run_start = time.perf_counter()
    for i, (dataset, di, problem, gold) in enumerate(jobs):
        path = OUT_DIR / f"{dataset}_{di:04d}.json"
        if path.exists():
            print(f"[{i + 1:2d}/30] {path.name} exists, skipping")
            continue
        seed = SEED_BASE + i
        t0 = time.perf_counter()
        trace = generate_trace(
            tokenizer, model,
            dataset=dataset, index=di, problem=problem, gold_answer=gold, seed=seed,
        )
        path.write_text(json.dumps(trace.to_dict(), ensure_ascii=False, indent=1))
        print(
            f"[{i + 1:2d}/30] {path.name}  seed={seed}  {trace.new_tokens:4d} tok  "
            f"{trace.seconds:6.1f}s  {trace.tokens_per_second:5.1f} tok/s  "
            f"{trace.finish_reason:6s}  complete={trace.reasoning_complete}  "
            f"correct={trace.correct}  ({time.perf_counter() - t0:.0f}s)"
        )

    print(f"\ntotal {time.perf_counter() - run_start:.0f}s")


def determinism_check(tokenizer, model, job) -> None:
    """Generate one problem twice with the same seed; report either way."""
    dataset, di, problem, gold = job
    print("\n--- determinism check (same seed, twice, 128 tokens) ---")
    runs = []
    for _ in range(2):
        t = generate_trace(
            tokenizer, model,
            dataset=dataset, index=di, problem=problem, gold_answer=gold,
            seed=SEED_BASE, max_new_tokens=128,
        )
        runs.append(t.output_token_ids)
    identical = runs[0] == runs[1]
    print(f"identical token ids: {identical}")
    if not identical:
        first_diff = next(
            (k for k, (a, b) in enumerate(zip(runs[0], runs[1])) if a != b),
            min(len(runs[0]), len(runs[1])),
        )
        print(f"lengths {len(runs[0])} vs {len(runs[1])}, first difference at token {first_diff}")
    print("--- end determinism check ---\n")
    (REPO_ROOT / "traces" / "pilot" / "_determinism.json").write_text(
        json.dumps({"identical": identical, "lengths": [len(r) for r in runs]}, indent=1)
    )


if __name__ == "__main__":
    main()
