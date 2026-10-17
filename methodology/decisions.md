# Design decisions

Each design decision, with its date and the reason behind it. The rules themselves are in `pre-registration.md`. Decisions made after the public draft opened are also logged in that file's changelog.

| ID | Decision | Date | Reason |
|---|---|---|---|
| F | The question is self-contained: does the model's inside forecast a "wait" earlier than the outside cues do? FrictionMap's marker lexicon is the motivation and the label source, not the object of study. | 2026-10-10 | The claim should match what the test can back. Results are about this model only. |
| D1 | Labels come from the frozen FrictionMap v1 lexicon regex. A hand audit of 100 randomly drawn onsets measures precision only. The audit happens after freeze and before any probe score exists, and never changes training labels. | 2026-10-10 | Gives a precision number for this lexicon on this model. The guard rule stops anyone retraining on the "real" hits after seeing results. |
| D1+ | Exploratory add-on: probe scores before real vs. false hits. Test-split onsets only; at least 20 per group, otherwise logged as not runnable. | 2026-10-10 | Asks whether the inside anticipates the word, or the course-correction behind it. |
| D1− | Other markers this model uses go to an exploratory appendix only. | 2026-10-10 | Picking labels after seeing pilot data would be a fork. |
| D2 | GSM8K is the primary dataset, with MATH500 as the fallback if GSM8K can't reach the minimum event count. | 2026-10-10 | The model is distilled for math, and Zhu et al. studied the same model on math. |
| D3 | A gate at distance 1. Four confirmatory bands (2–5, 6–10, 11–20, 21–40 tokens), each tested on its own with a 4-way correction. Exploratory: an exact-distance curve over 1–40 × all layers, and per-trace plots on 10 seeded traces. Distance is counted in tokens. | 2026-10-11 | Bands pool enough examples per test. Testing each band on its own lets an on/off pattern show. The per-trace plots catch signals tied to moments rather than distances. |
| D4a | Null bar δ = 0.05 AUROC lift (provisional, finalized at freeze). Verdicts are positive, null or inconclusive, by where the interval sits relative to 0 and δ. The full interval is always reported, with secondary verdicts at 0.03 and 0.08. Uncertainty comes from resampling whole problems. Minimum events are computed from pilot counts only. | 2026-10-11 | δ is chosen by what difference would matter. The minimum makes a null reachable rather than only "positive or can't tell." |
| D4b | Primary comparison: the probe vs. one combined outside rival (text + visible uncertainty + position), with the same examples and the same tuning effort. Each rival alone is exploratory. A stronger text rival is built only if a band is positive. | 2026-10-11 | "Outside" means all outside cues together. Beating each cue separately could hide that they jointly explain the signal. |
| D4c-1 | "No wait coming" examples are the same token kind, with no marker in the next 100 tokens (60 as a count-based fallback). Positions 41–100 tokens before an onset are excluded. | 2026-10-11 | Positions right before a "wait" are mostly sentence ends. Matching leaves whether a wait follows as the only difference. |
| D4c-2 | Split by problem, about 50 / 15 / 35. The layer is chosen per band on validation, and test is used once per band. | 2026-10-11 | Picking the best of 28 layers on test data would inflate the score. |
| D4c-3 | The gate passes if the model's own odds reach AUROC ≥ 0.75 and the probe lands within 0.05 of them. A failed gate means fix, log and rerun before anything else. | 2026-10-11 | The model samples its words, so even one token before, the inside knows odds, not outcomes. |
| D5 | Answer correctness is not a hypothesis. Accuracy is reported as context only. | 2026-10-11 | It's a different question, wrong answers are few on GSM8K, and trace length confounds it. |
| D6 | The design is posted publicly for feedback before freeze. Changes are logged with who raised them. Before freeze, only generation, counting, reading traces and the model's own odds at distance 1 touch data. | 2026-10-11 | Feedback can change the design while no analysis has been run. |

| D10 | Looping traces (a 30-token window of generated ids repeated 3+ times verbatim) are excluded. Traces that hit the length cap without looping are kept and flagged. | 2026-10-17 | A verbatim loop produces near-duplicate events that inflate counts and let a probe memorize. Paraphrased circling produces distinct events. Pilot: 5 of 30 loop (4 at the length cap, 1 finished normally); 4 more hit the cap without looping; 9 of 30 affected in all. Looping is not a subset of truncation, so both flags are kept. |

| D11 | Onsets are defined by the paragraph rule: the first hit in a paragraph is the onset, later hits in the same paragraph merge into it. No alternative onset rule is tested in this experiment. | 2026-10-17 | The model delimits its own reasoning steps with blank lines, so the rule is structural and needs no tuned number. Pilot: 192 of 619 within-trace hits share a paragraph with an earlier hit, and nearly half of those are within 3 tokens ("Wait, hold on"). The rule is conservative: it merges some later reconsiderations in the same step, which costs count, not bias. |

## Parked, with revival conditions

| Item | Revival condition |
|---|---|
| Onsets labeled by an LLM judge instead of the lexicon | A positive band |
| Distance counted in reasoning steps instead of tokens | A positive band at 11–20 or 21–40 |
| Incremental comparison (the probe added on top of the rival) | A band where the probe is well above chance but doesn't beat the rival |
| Replication on DeepSeek-R1-Distill-Qwen-7B | A positive result on the 1.5B model |
| Probe score as a predictor of answer correctness | A positive band and at least 50 wrong answers in the test set |
| Window-rule onsets (merge hits within 10 tokens) as an alternative to the paragraph rule | A positive band, and a reader or the writeup needs to know whether the verdict depends on the onset rule |
