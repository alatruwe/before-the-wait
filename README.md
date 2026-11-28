# before-the-wait

**Question.** In DeepSeek-R1-Distill-Qwen-1.5B solving GSM8K problems, does a linear readout of the model's middle layers forecast a "wait" better than a stated outside reader: the text so far, the model's output distribution, and position? If so, at which distances?

The residual stream at a token is a function of the tokens up to it, so the inside cannot hold information the text lacks; the test is whether a simple readout of the model's own computation beats a specified reader of the same text.

The test compares linear probes on the residual stream (simple classifiers reading the model's internal activations) against a baseline: one classifier trained on all of those outside signals together. Each of four bands gets its own verdict: 2–5, 6–10, 11–20 and 21–40 tokens before the "wait".

**Why it matters.** Reading a reasoning model's chain of thought is a leading proposal for monitoring it. That only works if the text keeps up with the process behind it; this project tests that for one behavior in one small model.

**What's known, what's new.** On this same model, hidden states just before a reflection word like "wait" separate from matched tokens ([Zhu et al., 2025](https://arxiv.org/abs/2506.12217)). Steering along linear directions can induce backtracking ([Venhoff et al., 2025](https://arxiv.org/abs/2506.18167); [Ward et al., 2025](https://arxiv.org/abs/2507.12638)). Hidden states are known to carry information about tokens several steps ahead ([Pal et al., 2023](https://arxiv.org/abs/2311.04897)); whether that is pre-caching or breadcrumbs is open ([Wu, Morris & Levine, 2024](https://arxiv.org/abs/2404.00859)). To our knowledge, no one has tested how far ahead the internals forecast a "wait" against a text baseline; that is the contribution.

## Status

Every registered step has run. The write-up is in progress: a draft of the report is at [`report/before-the-wait.md`](report/before-the-wait.md). Issues are open.

## Results

**Reading a verdict.** Each band's verdict comes from the 98.75% interval on the lift (probe AUROC minus rival AUROC), against a pre-registered bar of δ = 0.05. Positive: the interval lies above zero and reaches δ. Null: the interval lies entirely below δ. Inconclusive: the interval includes both zero and δ, and the data cannot tell.

A positive band means a simple readout of the model's middle layers beats the stated outside reader at that distance. A null means that reader, given the text, the model's output distribution and position, gets as much about an upcoming "wait" as the readout does. If the readout and the reader are both near chance in a band, neither forecasts at that distance, and the null says nothing about the inside. All of these are results; the design can produce a real null.

**Against the outside reader.** At 6–10 tokens before a "wait", the probe forecast it better than the outside reader, by +0.045 AUROC against the bar of 0.05: a positive verdict, with the estimate just under the bar. At 2–5 tokens the probe also leads, by too little to count. At 11–20 and 21–40 tokens the result is inconclusive.

| Band (tokens before the "wait") | Probe AUROC | Rival AUROC | Lift | 98.75% interval | Verdict at δ = 0.05 |
|---|---|---|---|---|---|
| 2–5 | 0.931 | 0.905 | +0.026 | +0.011 to +0.045 | null |
| 6–10 | 0.868 | 0.823 | +0.045 | +0.022 to +0.076 | positive |
| 11–20 | 0.801 | 0.770 | +0.031 | −0.002 to +0.068 | inconclusive |
| 21–40 | 0.730 | 0.709 | +0.021 | −0.019 to +0.067 | inconclusive |

Full table, counts, controls and secondary verdicts: [`results/verdicts.md`](results/verdicts.md).

**Against a stronger reader.** The positive band triggered a second, registered test (§8): the same kind of probe on a larger sibling model, DeepSeek-R1-Distill-Qwen-7B, reading the same tokens. At 6–10 tokens the two are level: lift −0.002, 98.75% interval −0.010 to +0.006, a null. The probe beats the hand-built outside reader and does not beat this one. Details: [`results/stronger_rival.md`](results/stronger_rival.md).

**What the label is.** A hand audit of 100 test onsets (§12) asked whether the trace changes what it held just before the marker. In 13 it did: precision 0.13, 95% interval 0.05 to 0.22. The probe forecasts the marker, and the marker is rarely an overturn. The registered follow-up comparing real and false markers needed 20 real ones and was not run. Details: [`results/audit.md`](results/audit.md).

**Exploratory analyses** (§13, no verdicts): [`results/exploratory/summary.md`](results/exploratory/summary.md).

## Record

- **Frozen before the data.** The pre-registration is frozen at tag `prereg-freeze` (2026-10-24): [`methodology/pre-registration.md`](methodology/pre-registration.md). The confirmatory verdicts were scored on 2026-11-15; the hand audit, the stronger reader and the exploratory analyses ran on 2026-11-21.
- **Deviations.** None is logged in §16.
- **Choices after the freeze.** Procedural choices the pre-registration left open are in [`methodology/decisions.md`](methodology/decisions.md) (D15–D25), each dated before the data it concerns.
- **Test split.** Scored once for the verdicts, once more for band 6–10 in the stronger-reader test, and read again for the exploratory analyses and the illustration data in [`viz/`](viz/); nothing was chosen or tuned on it.
- **Sample size.** [`results/count_calculations.md`](results/count_calculations.md): the pilot-derived numbers, N = 320 traces and m = 555 onsets per band.
- **Step-by-step results**, in the order they were produced: [`run_summary.md`](results/run_summary.md), [`examples_summary.md`](results/examples_summary.md), [`extraction_summary.md`](results/extraction_summary.md), [`gate.md`](results/gate.md), [`probes.md`](results/probes.md), [`rival.md`](results/rival.md), [`verdicts.md`](results/verdicts.md), [`audit.md`](results/audit.md), [`stronger_rival.md`](results/stronger_rival.md), [`exploratory/summary.md`](results/exploratory/summary.md).

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
| `scripts/` | One script per step: generation, counts, example building, extraction, gate, probes, rival, verdicts, audit, stronger reader, exploratory analyses. `smoke_test.py` checks the environment |
| `tests/` | `pytest` tests |
| `notes/` | Working notes. `environment.md` records the machine, package versions and pinned hashes |
| `notebooks/` | Analysis notebooks |
| `results/` | Pilot results (`c1_*`, `c3_*`, `count_calculations.md`) and the run's results: one summary per step, with the CSVs behind them. `results/exploratory/` holds the exploratory analyses |
| `audit/` | The hand audit: the seeded draw, the sheet the labeler read, and the labels |
| `probes/` | The four frozen probes, one per band |
| `rival_7b/` | The frozen 7B reader for band 6–10 |
| `report/` | The report (draft in progress) and its figures; `scripts/make_report_figures.py` redraws them |
| `viz/` | Data for an illustration, from exploratory analysis 1: per-event test scores (a gzipped sample of 200 events) and their means. Not a result |
| `traces/` | Generated reasoning traces: `traces/pilot/` holds the 30 pilot traces, `traces/run/` the 320 traces of the run |

Model weights, the Hugging Face cache and the activations (about 24 GB) are never committed. The example set and the four frozen rivals are not committed either: the scripts rebuild them, and their sha256 hashes are in `results/examples_summary.md` and `results/rival.md`.

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
- **AI disclosure:** Claude (Anthropic) was the planning partner. It wrote the task specification for each step and proposed the procedural details the pre-registration left open; Adeline reviewed and approved each one before the data it concerns (D15–D25). Claude Code wrote the code for every step. This README was drafted by Claude and edited by Adeline. The research question, the design decisions and the reading of the results are Adeline's.
- **License:** [MIT](LICENSE).
