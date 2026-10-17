"""Trace generation: load the pinned model, sample one trace per problem.

Shared by the c1 pilot and the full step-D run, so the generation settings live
here rather than in the pilot driver.

Text and token ids only. Nothing in this module touches hidden states — the
pre-registration freeze rule allows generation, counting and the model's own
next-token probabilities at distance 1, and this is the generation half.
"""
from __future__ import annotations

import os
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
# Keep every Hugging Face file inside the repo (gitignored). Must be set before
# huggingface_hub / transformers are imported.
os.environ["HF_HOME"] = str(REPO_ROOT / ".hf_cache")

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
MODEL_REVISION = "ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562"

# Fixed by the handoff. D uses the same ones.
TEMPERATURE = 0.6
TOP_P = 0.95
MAX_NEW_TOKENS = 4096
SEED_BASE = 20261016

USER_TEMPLATE = "{problem}\nPlease reason step by step, and put your final answer within \\boxed{{}}."

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"


@dataclass
class Trace:
    dataset: str
    index: int
    problem: str
    gold_answer: str

    prompt_text: str
    prompt_token_ids: list[int]

    output_text: str
    output_token_ids: list[int]

    finish_reason: str  # "eos" | "length"
    eos_token_stripped: bool
    new_tokens: int
    seconds: float
    tokens_per_second: float

    reasoning_char_span: list[int]
    reasoning_token_span: list[int]
    reasoning_complete: bool

    boxed_answer: str | None
    correct: bool

    seed: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_model_and_tokenizer() -> tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, dtype=torch.float16
    ).to("mps")
    model.eval()
    return tokenizer, model


def build_prompt(tokenizer: Any, problem: str) -> str:
    """The tokenizer's own chat template, add_generation_prompt=True."""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": USER_TEMPLATE.format(problem=problem)}],
        tokenize=False,
        add_generation_prompt=True,
    )


def eos_ids(tokenizer: Any, model: Any) -> set[int]:
    out: set[int] = set()
    for source in (tokenizer.eos_token_id, model.generation_config.eos_token_id):
        if source is None:
            continue
        if isinstance(source, int):
            out.add(source)
        else:
            out.update(int(x) for x in source)
    return out


def find_last_boxed(text: str) -> str | None:
    """Content of the last \\boxed{...}, brace-matched so \\frac{1}{2} survives."""
    start = text.rfind("\\boxed{")
    if start == -1:
        return None
    i = start + len("\\boxed{")
    depth = 1
    while i < len(text):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start + len("\\boxed{") : i]
        i += 1
    return None  # unterminated (truncated generation)


def normalize_answer(s: str) -> str:
    """Strip whitespace and $ — the handoff's match rule, nothing cleverer."""
    return re.sub(r"\s+", "", s.replace("$", ""))


def gsm8k_gold(answer_field: str) -> str:
    """GSM8K gold is the number after ####."""
    return answer_field.split("####")[-1].strip()


def generate_trace(
    tokenizer: Any,
    model: Any,
    *,
    dataset: str,
    index: int,
    problem: str,
    gold_answer: str,
    seed: int,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> Trace:
    prompt_text = build_prompt(tokenizer, problem)
    # The chat template already starts with BOS.
    inputs = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).to("mps")
    prompt_len = int(inputs["input_ids"].shape[1])

    torch.manual_seed(seed)
    torch.mps.manual_seed(seed)
    torch.mps.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )
    torch.mps.synchronize()
    seconds = time.perf_counter() - start

    produced = [int(t) for t in output[0, prompt_len:]]
    stops = eos_ids(tokenizer, model)

    # Steer 4: keep special tokens so </think> survives, strip only a trailing EOS.
    if produced and produced[-1] in stops:
        finish_reason = "eos"
        eos_token_stripped = True
        ids = produced[:-1]
    else:
        finish_reason = "length"
        eos_token_stripped = False
        ids = produced

    output_text = tokenizer.decode(ids, skip_special_tokens=False)
    new_tokens = len(ids)

    char_span, token_span, complete = reasoning_spans(tokenizer, prompt_text, output_text, ids)

    boxed = find_last_boxed(output_text)
    correct = boxed is not None and normalize_answer(boxed) == normalize_answer(gold_answer)

    return Trace(
        dataset=dataset,
        index=index,
        problem=problem,
        gold_answer=gold_answer,
        prompt_text=prompt_text,
        prompt_token_ids=[int(t) for t in inputs["input_ids"][0]],
        output_text=output_text,
        output_token_ids=ids,
        finish_reason=finish_reason,
        eos_token_stripped=eos_token_stripped,
        new_tokens=new_tokens,
        seconds=seconds,
        tokens_per_second=(new_tokens / seconds) if seconds > 0 else 0.0,
        reasoning_char_span=list(char_span),
        reasoning_token_span=list(token_span),
        reasoning_complete=complete,
        boxed_answer=boxed,
        correct=correct,
        seed=seed,
    )


def reasoning_spans(
    tokenizer: Any, prompt_text: str, output_text: str, ids: list[int]
) -> tuple[tuple[int, int], tuple[int, int], bool]:
    """Reasoning is between the opening <think> and </think>.

    If the chat template already emitted <think> (this model's does), the
    reasoning starts at char 0 of the output.
    """
    from before_the_wait.align import prefix_lengths, token_index_for_char

    if prompt_text.rstrip().endswith(THINK_OPEN):
        char_start = 0
    elif THINK_OPEN in output_text:
        char_start = output_text.index(THINK_OPEN) + len(THINK_OPEN)
    else:
        char_start = 0

    close = output_text.find(THINK_CLOSE)
    complete = close != -1
    char_end = close if complete else len(output_text)

    lens = prefix_lengths(tokenizer, ids)
    tok_start = token_index_for_char(lens, char_start) if char_start < len(output_text) else 0
    # End is exclusive: the first token whose span starts at or after char_end.
    tok_end = token_index_for_char(lens, char_end) if char_end < len(output_text) else len(ids)
    return (char_start, char_end), (tok_start, tok_end), complete
