# before-the-wait: pre-registration (DRAFT)

**Status:** public DRAFT, open for feedback.
- **Feedback window:** opens when the design post is published on LessWrong, closes seven days later. Dates: [TBD at posting].
- **Freeze:** happens when the window has closed, the pilot-count numbers are set, and no [DECIDE] slot is left. The freeze is a tagged commit (`prereg-freeze`). After it, changes go only to the deviations log (§16).
- **Decision IDs** (F, D1, D2…) refer to `decisions.md`.

**Slot legend**
- **[DECIDE]**: open. Decided before freeze.
- **[FROM PILOT]**: computed by the stated rule from pilot counts only (30 pilot traces). No probe or rival output is involved.
- **No marker:** already decided (see `decisions.md`), or a requirement that follows directly from a decision.

---

## 0. Changelog (draft period)

Changes before freeze, including those prompted by feedback.

| Date | Section | Change | Raised by |
|---|---|---|---|
| 2026-10-16 | §1 | "Effectively 'wait'" reworded: Zhu's 97.2% is over Zhu's own 17-keyword list, not this lexicon; the share for these labels is measured in the pilot. | Claude Code review of handoff 01 |
| 2026-10-16 | §4 | Generation settings recorded as decided (fixed by dispatching the pilot). | Adeline |
| 2026-10-16 | §15 | Pilot data-contact row added. | Adeline |
| 2026-10-17 | §4 | D10: looping traces excluded; truncated-but-not-looping traces kept and flagged. | Adeline |
| 2026-10-17 | §5 | D11: onsets defined by the paragraph rule. | Adeline |
| 2026-10-17 | §4, §5 | Pilot counts corrected before push: 4 truncated-without-looping (not 3); window-rule count stated with its exact definition (503 at ≤ 10, not 534). | Claude Code verification of handoff 03 |
| 2026-10-17 | §5 | Window-rule onsets moved from a pre-declared sensitivity check to parked (not tested in this experiment). | Adeline |
| 2026-10-17 | §11, §15 | Gate: pilot check recorded (model odds 0.962); probe margin set to 0.10 (was 0.05); c3 data-contact row added. | Adeline, after the c3 report |
| 2026-10-17 | §4 | Trace-count rule now requires all four bands to reach the minimum (was three; band 21–40 fell 30 short at N = 143), adds a 20% margin rounded up to ten, and sets the cap at 1,000. N = 190. | Adeline, after the handoff 04 report |
| 2026-10-17 | §6, §11 | Clarified after the count script: the minimum counts onsets, not positions (§3 already said so); band positions outside the reasoning span or inside a marker span are excluded; "largest band" named as 21–40. | Claude Code reading of handoff 04 |
| 2026-10-17 | §4, §6, §11 | The rules that turn pilot counts into the trace count, split, quiet zone, matching kinds, gate-input set and minimum events written in; values to be computed by script. | Planning session (rules stated in chat before the pilot report) |
| 2026-10-17 | §4 | Switch rule now reads "final N (after margin and rounding)" against the cap. | Claude Code reading of handoff 05 |

---

## 1. Question and scope

**Question.** In DeepSeek-R1-Distill-Qwen-1.5B solving GSM8K problems, do the model's internals forecast a "wait" further ahead than everything visible from outside: the text so far, the model's visible uncertainty, and position? If so, at which distances? (F)

**Scope.**
- One model, one dataset.
- The label set is the FrictionMap v1 marker lexicon (13 markers). It is expected to be mostly "wait" on this model: Zhu et al. report "wait" as 97.2% of hits on their own 17-keyword list (this model, MATH500, Appendix B), and that list shares only "wait" with the v1 lexicon. The actual share per marker is counted in the pilot and reported; it does not change the label set. **Pilot (2026-10-16, 30 traces, 649 hits): "wait" 64.9%, "hmm" 12.2%, "let me think" 8.0%, "hold on" 5.5%, "no," 4.8%, "actually" 4.5%, "however" 0.2%; six markers never fired.**
- Claims are about this model only.

**Not tested here:**
- Any closed model's internals.
- Answer correctness (D5).
- Whether "wait" marks genuine reconsideration. The primary label is marker *emission*; the audit in §12 measures the gap.

---

## 2. Hypotheses

**Gate (a wiring check, not a hypothesis).** At distance 1, the pipeline recovers what the model itself predicts (§11). (D3, D4c-3)

**H1–H4 (confirmatory), one per band** b ∈ {2–5, 6–10, 11–20, 21–40} tokens before an onset:
- **H_b:** a linear probe on the residual stream separates "wait coming at distance b" from matched "no wait coming" positions better than the combined outside rival.
- Each band gets its own verdict (§11). (D3, D4b)

**Predictions, recorded before any data, not used in any decision:** [DECIDE]

---

## 3. Units

- **Example:** one token position in a reasoning trace.
- **Event:** one "wait" onset (§5). Minimum counts are in events (D4a).
- **Independence unit:** one problem. Splits and resampling work on whole problems (D4a, D4c-2).

---

## 4. Model, generation, data

**Model.** `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`, revision [pinned at setup], dtype [DECIDE], on an Apple M4 Pro (MPS).

**Generation.** These must be identical to the pilot's.

| Setting | Value |
|---|---|
| Prompt and chat template | No system prompt. User turn: `<problem>\nPlease reason step by step, and put your final answer within \boxed{}.` The tokenizer's chat template with `add_generation_prompt=True`; it appends `<｜Assistant｜><think>\n`, so generation starts inside the reasoning. (Adeline, 2026-10-16) |
| Temperature | 0.6 (Adeline, 2026-10-16) |
| Top-p | 0.95 (Adeline, 2026-10-16) |
| Max new tokens | 4096 (Adeline, 2026-10-16) |
| Samples per problem | 1 (Adeline, 2026-10-16) |
| Generation seed | `torch.manual_seed` and `torch.mps.manual_seed` with 20261016 + i for the i-th problem in draw order. Reproducibility verified in the pilot (identical ids on a repeated problem). (Adeline, 2026-10-16) |

**Activations.**
- Extracted with a teacher-forced pass over the *generated token ids*, reused exactly. The text is never re-tokenized.
- Same weights and dtype as generation.

**Dataset.** GSM8K (D2), split [DECIDE: test / train / both], problems drawn by seeded random sample.
- **Trace count:** N = the smallest integer such that N × r × (1 − ℓ) × o × e_b ≥ m for each of the four bands, where r is the test share (§4 split), ℓ the looped-trace share (pilot: 5/30), o the mean paragraph-rule onsets per non-looped GSM8K trace (pilot: 114/13), e_b the share of GSM8K onsets whose previous onset (or the reasoning start, for a first onset) lies more than the band's upper bound tokens back (pilot gap distribution), and m the minimum events (§11). The final count is that N inflated by 20% and rounded up to the nearest ten, because o is estimated from 13 pilot traces. Capped at 1,000. (Adeline, 2026-10-17: all four bands required, since band 21–40 reached 308 of 338 at the three-band N of 143 and 14 more traces buy its null back; margin and cap set the same day.) **Computed (handoff 04 rerun): N_rule = 157, N = 190 GSM8K traces.**
- **MATH500 switch:** if the final N (after margin and rounding) exceeds the cap, switch to MATH500 and recompute N with its pilot values (o = 175/12, its own e_b). (D2)
- **Exclusions:**
  - pilot problems: [DECIDE: exclude / include]
  - looping traces: **excluded.** A trace loops if any window of 30 consecutive generated token ids inside the reasoning span occurs 3 or more times verbatim. Their hits are near-duplicate examples. (Adeline, 2026-10-17. Pilot: 5 of 30; 4 of those also hit the length cap, 1 finished normally. Looping and truncation are independent flags.)
  - traces that hit the length cap without looping: **kept**, flagged `truncated`; positions without a full 100-token lookahead cannot be negatives (§6). (Adeline, 2026-10-17. Pilot: 4 of 30. In total 9 of 30 pilot traces are looped, truncated, or both.)
  - Every exclusion is counted and reported.
- **Split by problem:** train / validation / test = 50 / 15 / 35, unless N × (1 − ℓ) × o × 0.15 < 100 projected validation onsets, in which case 45 / 20 / 35. Split seed [DECIDE]. (D4c-2)
- **Accuracy:** the `\boxed{}` answer is checked against GSM8K's, per trace. Reported as context only. (D5)

---

## 5. Labels

**Lexicon.** FrictionMap v1 `_LEXICON` (13 markers), verbatim, with the same regex semantics as `src/frictionmap/clusters.py` in github.com/alatruwe/frictionmap at commit [pinned at setup]. Applied to the reasoning text between `<think>` and `</think>`. Frozen; no additions. (D1) Other markers this model uses are counted in an exploratory appendix only.

**Onset token.** The first generated token whose span covers the marker's first character.

**Clustering.** Paragraph rule: the first hit in a paragraph (text between `\n\n` delimiters) is the onset; later hits in the same paragraph merge into it and are not onsets. The unit is the model's own reasoning step; an onset is the step at which it turns into a reconsideration. (Adeline, 2026-10-17.) **Not tested here:** a window rule (a hit is an onset unless the previous hit in the same trace has an onset-token index within 10 of its own, difference ≤ 10) is parked; see `decisions.md`. Pilot counts: 192 of 619 within-trace hits share a paragraph with an earlier hit; on all 30 pilot traces the paragraph rule yields 457 onsets from 649 hits (the parked window rule would yield 503); on the 25 non-looped traces kept under D10, 289 onsets from 385 hits (window rule: 301). Any onset count cited elsewhere states which base it uses.
- Merged markers are not onsets.
- No position before a merged marker can be a negative (§6).

**Distance.** For each token t, d(t) = (index of the next onset) − t. Tokens inside a marker span are excluded.

---

## 6. Examples

| Class | Definition |
|---|---|
| Gate positive | d(t) = 1 |
| Band positive | d(t) ∈ band: 2–5, 6–10, 11–20 or 21–40 (D3) |
| Negative | No marker of any kind (onset or merged) within the next 100 tokens (D4c-1) |
| Excluded (gray zone) | d(t) ∈ 41–100 |
| Excluded | positions outside the reasoning span (a band position before the reasoning start), and positions inside any marker span, onset or merged |

- **Quiet-zone fallback:** 100 tokens, unless for any token kind the pilot count of negative positions (no marker in the next 100 tokens) is smaller than the pilot count of positive positions of that kind in band 21–40; then 60. Computed in the count handoff.
- **Matching:**
  - Each positive is paired with a negative of the same token kind, the kind of the token *at* the position. Candidate kinds: newline, period, comma, other_punct, digit, word, other. A kind stands on its own if at least 20 pilot positive positions (across all bands) fall in it; rarer kinds merge into `other`. Computed in the count handoff.
  - Ratio: [DECIDE]. Source: [DECIDE: same trace when available / any trace in the same split].
- **End of reasoning:** tokens within 100 tokens of the end of reasoning are [DECIDE: negatives if the trace ended naturally / excluded].
- **Same examples for everyone:** the probe and every rival use exactly the same examples.

**Why matching:** positions right before a "wait" are mostly sentence ends and line breaks. With unmatched negatives, any method could win just by detecting the end of a sentence.

---

## 7. Probe

- **Input:** the residual stream (hidden-state output) of each of the 28 layers at token t.
- **Model:** logistic regression with L2 regularization, one per layer per band. Standardization and regularization grid: §17.
- **Training:** each band's positives plus its matched negatives from train problems.
- **Layer selection:** per band, by validation AUROC. The chosen layer is scored on test **once**. (D4c-2)

---

## 8. Outside rival: the primary comparison (D4b)

One logistic regression on the combined features below, trained on the same train examples and tuned on the same validation set, with a grid of comparable size.

| Feature group | Content | Open |
|---|---|---|
| Text | TF-IDF over token-id n-grams of the preceding tokens | n-gram range [DECIDE]; window [DECIDE: fixed / tuned on validation from a stated set] |
| Visible uncertainty | Entropy of the model's next-token distribution at t | Summaries over the previous w tokens [DECIDE: include, which] |
| Position | Token index t; tokens since the previous onset; tokens since the last paragraph break | — |

**Causal-only rule.** Every rival feature uses tokens up to t only. Nothing that depends on later tokens: no trace length, no relative position.

**Exploratory:** each feature group alone against the probe.

**Conditional stronger text rival** (built only for bands with a positive verdict):
- Hidden states of [DECIDE: model] reading the same token ids, with the same probe recipe and layer-selection rule.
- **Claim level 1:** beats the combined rival.
- **Claim level 2:** also beats this one.

---

## 9. Sanity controls

| Control | Expected | Keep? |
|---|---|---|
| Shuffled labels (train labels permuted within each band) | AUROC ≈ 0.5 | [DECIDE] |
| Random direction | — | [DECIDE] |

---

## 10. Metrics

- **Primary:** AUROC per band on test examples. **Lift** = probe AUROC − combined-rival AUROC.
- **Uncertainty:** bootstrap over whole test problems, resample count in §17, as a **98.75% interval** (95% with a 4-band correction). (D3, D4a)
- **Also reported for every band:** counts of onsets, positives, negatives and problems; AUPRC [DECIDE: report / drop; with 1:1 matching the classes are balanced by construction].

---

## 11. Decision rules

### Gate (D4c-3)
Run on [DECIDE: validation / test] problems at d = 1.
1. **The model's own odds separate the classes.** Score = the summed probability over the set of generated token ids that begin a lexicon-marker hit in the experiment's own traces (recomputed from the labels, not fixed from the pilot; the pilot set after D10/D11 was 12 ids). It must reach AUROC ≥ [DECIDE: final floor; provisional 0.75. Pilot check (c3, 2026-10-17): 0.962, 95% interval 0.947–0.977, on 289 paragraph-rule onsets with exact same-kind negatives].
2. **The probe gets close.** Best-layer probe AUROC ≥ (model-odds AUROC − 0.10). (Adeline, 2026-10-17. The clause exists to catch wiring errors, which show as drops of 0.2 or more; with model odds at 0.96 on the pilot, a 0.05 margin would have demanded a 0.91 probe and invited tuning a merely-short probe until it passed.)

**Failure:** stop, fix, log in §16, rerun. Nothing at d > 1 is computed until the gate passes.

### Per band (D4a)
δ = [DECIDE: final value; provisional 0.05; anchors in Appendix A]. "Interval" means the 98.75% interval on the lift.

| Verdict | Condition |
|---|---|
| Positive | Lower end > 0 **and** upper end ≥ δ |
| Null | Upper end < δ (includes "real but under δ") |
| Inconclusive | Lower end ≤ 0 **and** upper end ≥ δ |

- **Minimum events:** m counts **onsets**, not positions: one onset contributes several positive positions to a band (4 in 2–5, 20 in 21–40), and those are near-duplicates that do not count separately. m = the smallest number of test-split onsets per band such that the half-width of the 98.75% interval on the lift is ≤ δ, using the Hanley–McNeil standard error of an AUROC of 0.70 on m positives and m matched negatives for each of the two AUROCs, combined with an assumed correlation of 0.5 between them (so SE_lift = SE of one AUROC). The assumed AUROC and correlation are stated, conservative, and fixed before any probe exists. Computed by script in the count handoff and recorded here.
  - A band below the minimum can only be positive or inconclusive, and the writeup says so.
- **Secondary:** verdicts at δ = 0.03 and 0.08. Reported, never the headline.
- **Claim:** the set of bands with a positive verdict. Every band's verdict and interval are reported.

---

## 12. Hand audit and add-on (D1)

- **Sample:** 100 onsets, simple random draw (seed [DECIDE]) from test-split traces.
- **Timing:** after freeze, before any probe score exists. Labels are committed and hashed before probe scoring.
- **Question:** binary, real vs. false, where a real hit [DECIDE: *overturns* a prior statement / *re-examines* a prior statement].
- **Output:**
  - Precision with a 95% interval.
  - A descriptive side-by-side with the lexicon's earlier precision on Claude coding sessions (FrictionMap). Not a test.
- **Guard:** audit labels never change training or evaluation labels.
- **Add-on (exploratory):**
  - Compares probe scores before real vs. false hits, at [DECIDE: which band(s) or distances].
  - Expected direction, stated in advance: [DECIDE].
  - Runs only if both groups have at least 20 hits. Otherwise it's logged as not runnable.

---

## 13. Exploratory analyses

No verdicts; labeled exploratory everywhere they appear.

1. Exact-distance curve: probe AUROC and lift at every distance 1–40, every layer (D3).
2. Each outside rival group alone vs. the probe (D4b).
3. Per-trace plots: probe score token by token on 10 traces (seed [DECIDE]), with the text alongside. Done after the confirmatory results (D3).
4. The add-on (§12).
5. Where each band falls relative to paragraph breaks.
6. Counts of other markers this model uses.
7. Accuracy, as context (D5).

---

## 14. Outcomes and the null-result path

| Pattern | Reads as |
|---|---|
| Gate fails after fixes | The pipeline can't recover what the model itself predicts. No band results reported; the failure is written up. |
| All bands positive | The inside carries information about an upcoming "wait" that the outside cues lack, as far as 21–40 tokens ahead |
| Near bands positive, far bands null | The inside runs ahead of the outside cues, within the furthest positive band |
| On/off (positive, null, positive) | The signal may be tied to moments, not distances. Per-trace plots are the lead; a follow-up confirms. |
| All null | For this model, at 2–40 tokens, the text plus visible uncertainty plus position carry what the inside carries about an upcoming "wait" |
| Mixed null / inconclusive | Each band reported as is; no pooling |
| All inconclusive | Not enough data to tell; n reported |

**Not allowed after seeing any test result:**
- changing the bands, δ, the layer rule, the label set, the dataset or the rival features
- promoting an exploratory analysis to confirmatory
- adding traces to rescue an inconclusive band

Anything like that is a new, separately pre-registered follow-up.

---

## 15. Data contact before freeze

**Allowed:** generating traces, counting, reading traces by eye, the model's own odds at d = 1.
**Not allowed:** any probe, any rival, anything at d > 1. (D6)

| Date | What was touched | What was seen |
|---|---|---|
| 2026-10-16 | Pilot: 15 GSM8K + 15 MATH500 traces generated with the §4 settings; lexicon hits counted; traces read by eye. Commit `fef6791`. | Counts only (`results/c1_summary.md`): 649 hits, 10.1 per 1k reasoning tokens; 8 traces hit the 4096 cap, 5 loop; "wait" 64.9% of hits; 47% of hits paragraph-initial; 68% of gaps ≥ 20 tokens; negatives pool 25,867 at 100 tokens. No hidden states, no probes. |
| 2026-10-17 | c3: teacher-forced hidden states on the 25 non-looped pilot traces; read at onset − 1 and at matched negatives (no marker in the next 100 tokens) only, plus 50 marker-unrelated positions for the extraction check. Model's own odds and a logit lens per layer. No classifier trained. Commit `ee566de`. | Gate input: model-odds AUROC 0.962 (0.947–0.977) pooled; the marker becomes readable in the model's own vocabulary from layer ~23 of 28; nothing at distances 2–100 was read. |

---

## 16. Deviations log (after freeze)

| Timestamp | Section | Deviation | Reason | Any results seen before it? |
|---|---|---|---|---|
| | | | | |

---

## 17. Technical settings

| Setting | Value |
|---|---|
| Probe input standardization | [DECIDE] |
| Probe regularization grid | [DECIDE] |
| Rival regularization grid | [DECIDE] |
| Bootstrap resamples | [DECIDE] |
| Seeds: problem sample, split, audit, per-trace plots | [DECIDE] |

---

## Appendix A: anchors for δ

| Source | Setting | Number | Use |
|---|---|---|---|
| Hidden Error Awareness (Yuan et al.), Table 5 | Qwen2.5-3B, MATH-500, predicting correctness from the first reasoning step | Probe 0.787 vs. TF-IDF + logistic regression text classifier 0.590 → gap 0.197 | The only published inside-vs-text gap found. Different target, weaker rival than here |
| ReflCtrl (Yan et al.), Table 2 | DS-Llama-8B, GSM8K, step before a reflection step vs. other steps | Entropy d = 0.38 (mean) / 0.50 (max) → AUROC ≈ 0.61 / 0.64 | What one outside cue gets alone |

**Caveat.** HEA gives n = 200 in Table 5 and n = 89 in Table 4 for the same result, and doesn't state the text classifier's evaluation split.

**Justification paragraph:** [DECIDE: written at freeze]

## Appendix B: references

- Zhu et al., arXiv 2506.12217
- Troitskii et al., arXiv 2510.04128
- Venhoff et al., arXiv 2506.18167
- Ward et al., arXiv 2507.12638
- Yan et al. (ReflCtrl), arXiv 2512.13979
- Yuan et al. (Hidden Error Awareness), arXiv 2605.09502
