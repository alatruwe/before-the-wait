# before-the-wait

**Question.** In DeepSeek-R1-Distill-Qwen-1.5B solving GSM8K problems, does a linear readout of the model's middle layers forecast a "wait" better than a stated outside reader: the text so far, the model's output distribution, and position? If so, at which distances?

The residual stream at a token is a function of the tokens up to it, so the inside cannot hold information the text lacks; the test is whether a simple readout of the model's own computation beats a specified reader of the same text.

The test compares linear probes on the residual stream (simple classifiers reading the model's internal activations) against a baseline: one classifier trained on all of those outside signals together. Each of four bands gets its own verdict: 2–5, 6–10, 11–20 and 21–40 tokens before the "wait".

**Why it matters.** Reading a reasoning model's chain of thought is a leading proposal for monitoring it. That only works if the text keeps up with the process behind it; this project tests that for one behavior in one small model.

**What's known, what's new.** On this same model, hidden states just before a reflection word like "wait" separate from matched tokens ([Zhu et al., 2025](https://arxiv.org/abs/2506.12217)). Steering along linear directions can induce backtracking ([Venhoff et al., 2025](https://arxiv.org/abs/2506.18167); [Ward et al., 2025](https://arxiv.org/abs/2507.12638)). Hidden states are known to carry information about tokens several steps ahead ([Pal et al., 2023](https://arxiv.org/abs/2311.04897)); whether that is pre-caching or breadcrumbs is open ([Wu, Morris & Levine, 2024](https://arxiv.org/abs/2404.00859)). To our knowledge, no one has tested how far ahead the internals forecast a "wait" against a text baseline; that is the contribution.

**What the outcomes would mean.** A positive band means a simple readout of the model's middle layers beats the stated outside reader at that distance. A null means that reader, given the text, the model's output distribution and position, gets as much about an upcoming "wait" as the readout does. If the readout and the reader are both near chance in a band, neither forecasts at that distance, and the null says nothing about the inside. An inconclusive band means the interval includes both zero and δ: the data cannot tell. All of these are results; the design can produce a real null.

## Status

**Pre-registration frozen at tag `prereg-freeze` (2026-10-24). Confirmatory run complete: the test split was scored once, on 2026-11-15.**

At 6–10 tokens before a "wait", the probe forecast it better than the outside reader, by +0.045 AUROC against a pre-registered bar of 0.05: a positive verdict, with the estimate just under the bar. At 2–5 tokens the probe also leads, by too little to count. At 11–20 and 21–40 tokens the result is inconclusive.

| Band (tokens before the "wait") | Probe AUROC | Rival AUROC | Lift | 98.75% interval | Verdict at δ = 0.05 |
|---|---|---|---|---|---|
| 2–5 | 0.931 | 0.905 | +0.026 | +0.011 to +0.045 | null |
| 6–10 | 0.868 | 0.823 | +0.045 | +0.022 to +0.076 | positive |
| 11–20 | 0.801 | 0.770 | +0.031 | −0.002 to +0.068 | inconclusive |
| 21–40 | 0.730 | 0.709 | +0.021 | −0.019 to +0.067 | inconclusive |

Full table, counts, controls and secondary verdicts: [`results/verdicts.md`](results/verdicts.md).

No deviation is logged in §16. Procedural choices made after the freeze are in [`methodology/decisions.md`](methodology/decisions.md) (D15–D22), each dated before the data it concerns.

**Still to come**
- The hand audit and its add-on (§12) and the stronger rival for band 6–10 (§8). Both were conditional on a positive band.
- The exploratory analyses (§13).
- The write-up.

**Documents**
- [`methodology/pre-registration.md`](methodology/pre-registration.md): the full design, frozen.
- [`methodology/decisions.md`](methodology/decisions.md): each decision, with its date and reason.
- [`results/count_calculations.md`](results/count_calculations.md): the pilot-derived numbers, N = 320 traces and m = 555 onsets per band.
- Results, in the order they were produced: [`run_summary.md`](results/run_summary.md), [`examples_summary.md`](results/examples_summary.md), [`extraction_summary.md`](results/extraction_summary.md), [`gate.md`](results/gate.md), [`probes.md`](results/probes.md), [`rival.md`](results/rival.md), [`verdicts.md`](results/verdicts.md).

Issues are open.

## Scope

- One model: `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`, run locally on an Apple M4 Pro (MPS).
- One dataset: GSM8K, with MATH500 as a pre-specified fallback.
- Labels: the FrictionMap v1 marker lexicon (13 markers), ported verbatim from [frictionmap](https://github.com/alatruwe/frictionmap). On this model it is mostly "wait": 64.9% of pilot hits and 72% of onsets under the paragraph rule ([results/c1_summary.md](results/c1_summary.md)). In the run itself, "wait" is 71.3% of onsets (1,796 of 2,519; [results/run_summary.md](results/run_summary.md)). For comparison, [Zhu et al.](https://arxiv.org/abs/2506.12217) report "wait" as 97.2% of hits on their own 17-keyword list, which shares only "wait" with this lexicon.
- Claims are about this model only.

## Layout

| Path | Contents |
|---|---|
| `methodology/` | Pre-registration (frozen) and design decisions |
| `src/before_the_wait/` | Package code. `lexicon.py` is the marker lexicon, copied verbatim from frictionmap at a pinned commit |
| `scripts/` | One script per step: generation, counts, example building, extraction, gate, probes, rival, verdicts. `smoke_test.py` checks the environment |
| `tests/` | `pytest` tests |
| `notes/` | Working notes. `environment.md` records the machine, package versions and pinned hashes |
| `notebooks/` | Analysis notebooks |
| `results/` | Pilot results (`c1_*`, `c3_*`, `count_calculations.md`) and the run's results: one summary per step, with the CSVs behind them |
| `probes/` | The four frozen probes, one per band |
| `traces/` | Generated reasoning traces: `traces/pilot/` holds the 30 pilot traces, `traces/run/` the 320 traces of the run |

Model weights, the Hugging Face cache and the activations (about 18 GB) are never committed. The example set and the four frozen rivals are not committed either: the scripts rebuild them, and their sha256 hashes are in `results/examples_summary.md` and `results/rival.md`.

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
- **AI disclosure:** Claude (Anthropic) was the planning partner. It wrote the task specification for each step and proposed the procedural details the pre-registration left open; Adeline reviewed and approved each one before the data it concerns (D15–D22). Claude Code wrote the code for every step. This README was drafted by Claude and edited by Adeline. The research question, the design decisions and the reading of the results are Adeline's.
- **License:** [MIT](LICENSE).
