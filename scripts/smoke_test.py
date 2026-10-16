"""Environment smoke test: MPS, fp16 model load, one generation, one hidden-states pass.

Run from the repo root:  .venv/bin/python scripts/smoke_test.py
"""
from __future__ import annotations

import os
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
# Keep every Hugging Face file inside the repo (gitignored). Must be set
# before huggingface_hub / transformers are imported.
os.environ["HF_HOME"] = str(REPO_ROOT / ".hf_cache")

import torch  # noqa: E402
import transformers  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
MODEL_REVISION = "ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562"
PROMPT = "What is 17 + 25?"
MAX_NEW_TOKENS = 512
# Sampling settings from the model card.
TEMPERATURE = 0.6
TOP_P = 0.95
SEED = 0


def main() -> None:
    mps_available = torch.backends.mps.is_available()
    print(f"MPS available = {mps_available}")
    assert mps_available, "MPS backend is not available"
    print(f"torch {torch.__version__}, transformers {transformers.__version__}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, dtype=torch.float16
    ).to("mps")
    cfg = model.config
    print(f"model {MODEL_ID} @ {MODEL_REVISION}")
    print(f"dtype {model.dtype}, device {model.device}, "
          f"layers {cfg.num_hidden_layers}, hidden size {cfg.hidden_size}")

    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPT}], tokenize=False, add_generation_prompt=True
    )
    # The chat template already starts with BOS.
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to("mps")
    seq_len = inputs["input_ids"].shape[1]
    print(f"prompt: {text!r} ({seq_len} tokens)")

    # Hidden-states pass first: it also compiles the Metal kernels, so that
    # one-off cost doesn't count against the tokens/second figure below.
    with torch.inference_mode():
        hidden_states = model(**inputs, output_hidden_states=True).hidden_states
    print(f"\n{len(hidden_states)} hidden-state tensors "
          f"(embeddings + {cfg.num_hidden_layers} layers):")
    for i, h in enumerate(hidden_states):
        print(f"  [{i:2d}] {list(h.shape)} {h.dtype}")
    assert len(hidden_states) == cfg.num_hidden_layers + 1
    assert all(tuple(h.shape) == (1, seq_len, cfg.hidden_size) for h in hidden_states)
    # fp16 overflows past 65504.
    all_finite = all(torch.isfinite(h).all().item() for h in hidden_states)
    max_abs = max(h.abs().max().item() for h in hidden_states)
    print(f"all finite = {all_finite}, max |h| = {max_abs:.1f}")

    torch.manual_seed(SEED)
    torch.mps.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )
    torch.mps.synchronize()
    elapsed = time.perf_counter() - start
    new_tokens = output.shape[1] - seq_len
    print(f"\ngenerated {new_tokens} new tokens in {elapsed:.2f} s "
          f"= {new_tokens / elapsed:.1f} tokens/second")
    print("--- completion ---")
    print(tokenizer.decode(output[0, seq_len:]))


if __name__ == "__main__":
    main()
