# before-the-wait

**Question.** In DeepSeek-R1-Distill-Qwen-1.5B solving GSM8K problems, do the model's internals forecast a "wait" further ahead than everything visible from outside: the text so far, the model's next-token entropy (how unsure it is about the next token), and position? If so, at which distances?

The test compares linear probes on the residual stream (simple classifiers reading the model's internal activations) against a baseline: one classifier trained on all of those outside signals together. Each of four bands gets its own verdict: 2–5, 6–10, 11–20 and 21–40 tokens before the "wait".

**Why it matters.** Reading a reasoning model's chain of thought is a leading proposal for monitoring it. That only works if the text keeps up with the process behind it; this project tests that for one behavior in one small model.

**What's known, what's new.** On this same model, hidden states just before a reflection word like "wait" separate from matched tokens ([Zhu et al., 2025](https://arxiv.org/abs/2506.12217)). Steering along linear directions can induce backtracking ([Venhoff et al., 2025](https://arxiv.org/abs/2506.18167); [Ward et al., 2025](https://arxiv.org/abs/2507.12638)). To our knowledge, no one has tested how far ahead the internals forecast a "wait" against a text baseline; that is the contribution.

**What the outcomes would mean.** A positive band means the internals run ahead of the text at that distance. A null means the text, next-token entropy and position already carry what the internals know about an upcoming "wait". Both are results; the design can produce a real null.

## Status

**Pre-registration: public draft, open for feedback.**

- [`methodology/pre-registration.md`](methodology/pre-registration.md): the full design.
- [`methodology/decisions.md`](methodology/decisions.md): each design decision, with its date and reason.

The design freezes at a tagged commit, `prereg-freeze`. After that, changes go only to the pre-registration's deviations log. No traces have been generated yet; data contact before freeze is logged in §15 of the pre-registration.

Feedback is welcome: please open an issue. The window opens when the design post goes up on LessWrong and closes seven days later; the dates will be recorded here and in the pre-registration at posting.

## Scope

- One model: `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`, run locally on an Apple M4 Pro (MPS).
- One dataset: GSM8K, with MATH500 as a pre-specified fallback.
- Labels: the FrictionMap v1 marker lexicon (13 markers), ported verbatim from [frictionmap](https://github.com/alatruwe/frictionmap). On this model it is probably mostly "wait": [Zhu et al.](https://arxiv.org/abs/2506.12217) report "wait" as 97.2% of hits on their own 17-keyword list (DeepSeek-R1-Distill-Qwen-1.5B on MATH500, their Appendix B). That list shares only "wait" with the FrictionMap lexicon, so the share for these labels is measured in the pilot and reported.
- Claims are about this model only.

## Layout

| Path | Contents |
|---|---|
| `methodology/` | Pre-registration and design decisions |
| `src/before_the_wait/` | Package code. `lexicon.py` is the marker lexicon, copied verbatim from frictionmap at a pinned commit |
| `scripts/` | Runnable scripts. `smoke_test.py` checks the environment |
| `tests/` | `pytest` tests |
| `notes/` | Working notes. `environment.md` records the machine, package versions and pinned hashes |
| `notebooks/`, `results/` | Analysis notebooks and results (empty for now) |
| `traces/` | Generated reasoning traces (empty for now) |

Model weights, the Hugging Face cache and activations are never committed.

## Setup

Tested with Python 3.14.6.

```bash
python3.14 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pytest
.venv/bin/python scripts/smoke_test.py
```

On its first run, the smoke test downloads the pinned model revision (about 3.6 GB) into `.hf_cache/` inside the repo. Other code that uses the Hugging Face libraries must set `HF_HOME` to that folder too; otherwise files go to `~/.cache/huggingface`.

[`notes/environment.md`](notes/environment.md) has the exact package versions, the model revision and the frictionmap commit.

## About

- **Origin:** this project grew out of [FrictionMap](https://github.com/alatruwe/frictionmap), a tool that maps where AI coding sessions hit friction, partly from re-evaluation markers in the model's thinking. Its marker lexicon motivated the question and supplies the labels here. FrictionMap's pre-registered judge study is written up in [Validating an LLM judge for reasoning traces](https://thecognitivestack.substack.com/p/validating-an-llm-judge-for-reasoning).
- **Author:** Adeline Latruwe.
- **AI disclosure:** Claude (Anthropic) was the planning partner, and Claude Code wrote the setup code and drafted this README. The research decisions are Adeline's.
- **License:** [MIT](LICENSE).
