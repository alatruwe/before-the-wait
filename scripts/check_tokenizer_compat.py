"""Check that the 7B rival reads the pilot token ids as the same strings.

    .venv/bin/python scripts/check_tokenizer_compat.py

§8's conditional stronger rival, DeepSeek-R1-Distill-Qwen-7B, reads the 1.5B
model's generated token ids unchanged. That is only sound if every id decodes
to the same string under both tokenizers. This decodes every distinct id in the
30 pilot traces (prompt and output) under each and lists any mismatch.

Only tokenizer files are downloaded for the 7B (no weights), into .hf_cache/.
Writes results/tokenizer_compat.md.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ["HF_HOME"] = str(REPO_ROOT / ".hf_cache")
sys.path.insert(0, str(REPO_ROOT / "src"))

from huggingface_hub import HfApi, snapshot_download  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from before_the_wait.generate import MODEL_ID, MODEL_REVISION  # noqa: E402

RIVAL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
TOKENIZER_FILES = ["tokenizer.json", "tokenizer_config.json", "config.json",
                   "special_tokens_map.json"]
TRACE_DIR = REPO_ROOT / "traces" / "pilot"
OUT = REPO_ROOT / "results" / "tokenizer_compat.md"


def main() -> None:
    rival_rev = HfApi().model_info(RIVAL_ID).sha
    rival_dir = snapshot_download(RIVAL_ID, revision=rival_rev, allow_patterns=TOKENIZER_FILES)
    weights = [p.name for p in Path(rival_dir).iterdir()
               if p.suffix in (".safetensors", ".bin")]
    assert not weights, f"weights downloaded: {weights}"

    small = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    rival = AutoTokenizer.from_pretrained(rival_dir)

    traces = [json.loads(p.read_text()) for p in sorted(TRACE_DIR.glob("*.json"))
              if not p.name.startswith("_")]
    ids = sorted({i for t in traces for i in t["prompt_token_ids"] + t["output_token_ids"]})

    mismatches = []
    for i in ids:
        a = small.decode([i], skip_special_tokens=False)
        b = rival.decode([i], skip_special_tokens=False)
        ta = small.convert_ids_to_tokens(i)
        tb = rival.convert_ids_to_tokens(i)
        if a != b or ta != tb:
            mismatches.append((i, ta, tb, a, b))

    seq_mismatch = [
        f"{t['dataset']}_{t['index']}" for t in traces
        if small.decode(t["prompt_token_ids"] + t["output_token_ids"], skip_special_tokens=False)
        != rival.decode(t["prompt_token_ids"] + t["output_token_ids"], skip_special_tokens=False)
    ]

    same_vocab = small.get_vocab() == rival.get_vocab()

    doc = [
        "# Tokenizer compatibility: 1.5B vs. 7B rival",
        "",
        "Computed by `scripts/check_tokenizer_compat.py`. Tokenizer files only; no 7B weights.",
        "",
        "| | |",
        "|---|---|",
        f"| Model | `{MODEL_ID}` @ `{MODEL_REVISION}` |",
        f"| Rival | `{RIVAL_ID}` @ `{rival_rev}` |",
        f"| Vocabulary size (len) | {len(small)} vs. {len(rival)} |",
        f"| Pilot traces | {len(traces)} (prompt + output ids) |",
        f"| Distinct token ids checked | {len(ids)} |",
        f"| Ids decoding differently | **{len(mismatches)}** |",
        f"| Traces whose full decode differs | **{len(seq_mismatch)}** |",
        f"| Full token-to-id maps identical (beyond the pilot) | {'yes' if same_vocab else '**no**'} |",
        "",
    ]
    if mismatches:
        doc += ["| id | token (1.5B) | token (7B) | decoded (1.5B) | decoded (7B) |",
                "|---|---|---|---|---|"]
        doc += [f"| {i} | `{ta!r}` | `{tb!r}` | `{a!r}` | `{b!r}` |"
                for i, ta, tb, a, b in mismatches]
        doc.append("")
    if seq_mismatch:
        doc += ["Traces whose full decode differs: " + ", ".join(seq_mismatch), ""]
    OUT.write_text("\n".join(doc) + "\n")
    print("\n".join(doc))


if __name__ == "__main__":
    main()
