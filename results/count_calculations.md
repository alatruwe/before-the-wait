# Count calculations for the pre-registration

Computed by `scripts/count_calculations.py` from the pilot results. Every
number below is produced by a rule already written in
`methodology/pre-registration.md`; nothing here is chosen.

## Summary

| Quantity | Value | Rule |
|---|---|---|
| Minimum events m (δ = 0.05) | **338** per band | §11 |
| Trace count from the rule, N_rule | **157** | §4 (all four bands) |
| **Final trace count N** | **190** gsm8k traces | §4 / D12 (×1.2, up to the nearest 10) |
| Binding band | 21-40 | §4 |
| Split | **50 / 15 / 35** | §4 |
| Quiet zone | **100 tokens** | §6 |
| Matching kinds | **newline, period, comma, other_punct, digit, word, other** | §6 |
| Projected generation | **1.2 h** | handoff |

## Inputs

| Symbol | Meaning | Value | Source |
|---|---|---|---|
| δ | null bar | 0.05 (provisional) | §11 |
| r | test share | 0.35 | §4 split |
| ℓ | looped-trace share | 5/30 = 0.1667 | `c1_traces.csv` |
| o (GSM8K) | onsets per non-looped trace | 114/13 = 8.7692 | `c1_onsets_paragraph.csv` |
| o (MATH500) | onsets per non-looped trace | 175/12 = 14.5833 | `c1_onsets_paragraph.csv` |
| margin | D12 inflation on N_rule | ×1.2, rounded up to the nearest 10 | §4 / D12 |
| cap | §4 cap on N | 1000 | §4 / D12 |

## 1. Minimum events m (§11)

> m = the smallest number of test-split onsets per band such that the
> half-width of the 98.75% interval on the lift is ≤ δ, using the
> Hanley–McNeil standard error of an AUROC of 0.70 on m positives and m
> matched negatives ... combined with an assumed correlation of 0.5
> between them (so SE_lift = SE of one AUROC).

- interval 0.9875 two-sided → z = **2.4977**
- assumed AUROC 0.70, fixed before any probe exists

| δ | m | SE at m | half-width z·SE |
|---|---|---|---|
| 0.03 | **936** | 0.01201 | 0.03000 |
| 0.05 (primary) | **338** | 0.02000 | 0.04995 |
| 0.08 | **133** | 0.03192 | 0.07972 |

## 2. Eligibility shares e_b (§4)

> e_b: the share of onsets whose previous onset (or the reasoning start,
> for a first onset) lies more than the band's upper bound tokens back.

**gsm8k** (114 onsets over 13 non-looped traces)

| band | upper bound | onsets with gap > bound | e_b |
|---|---|---|---|
| 2-5 | 5 | 114 / 114 | **1.0000** |
| 6-10 | 10 | 110 / 114 | **0.9649** |
| 11-20 | 20 | 106 / 114 | **0.9298** |
| 21-40 | 40 | 96 / 114 | **0.8421** |

**math500** (175 onsets over 12 non-looped traces)

| band | upper bound | onsets with gap > bound | e_b |
|---|---|---|---|
| 2-5 | 5 | 173 / 175 | **0.9886** |
| 6-10 | 10 | 168 / 175 | **0.9600** |
| 11-20 | 20 | 164 / 175 | **0.9371** |
| 21-40 | 40 | 145 / 175 | **0.8286** |

## 3. Trace count N (§4)

> N = the smallest integer such that N × r × (1 − ℓ) × o × e_b ≥ m for
> each of the four bands ... The final count is that N inflated by 20%
> and rounded up to the nearest ten, because o is estimated from 13
> pilot traces. Capped at 1,000. (D12)

With m = 338, r = 0.35, ℓ = 0.1667, o = 8.7692 (GSM8K):

| band | e_b | onsets per trace in band | N | required by §4 |
|---|---|---|---|---|
| 2-5 | 1.0000 | 2.5577 | 133 | yes |
| 6-10 | 0.9649 | 2.4679 | 137 | yes |
| 11-20 | 0.9298 | 2.3782 | 143 | yes |
| 21-40 | 0.8421 | 2.1538 | 157 | yes |

**Binding band: 21-40 → N_rule = 157 GSM8K traces.**

D12 margin: ceil(1.2 × 157) = 189, rounded up to the nearest 10 → **N = 190**.

Switch rule: N ≤ cap (1000), so the MATH500 switch does not apply.

For the record, the same computation on MATH500 (o = 14.5833) gives N_rule = 96 at binding band 21-40, N = 120.

Projected test-split onsets per band at the final N, against m:

| band | projected test onsets at N | ≥ m? |
|---|---|---|
| 2-5 | 486.0 | yes |
| 6-10 | 468.9 | yes |
| 11-20 | 451.9 | yes |
| 21-40 | 409.2 | yes |

## 4. Split (§4)

> train / validation / test = 50 / 15 / 35, unless N × (1 − ℓ) × o × 0.15
> < 100 projected validation onsets, in which case 45 / 20 / 35.

- N × (1 − ℓ) × o × 0.15 = 190 × 0.8333 × 8.7692 × 0.15 = **208.3** projected validation onsets
- 208.3 ≥ 100 → **50 / 15 / 35**

## 5. Positive positions by token kind (§6)

Tokens at distances 2–40 before each paragraph-rule onset in the 25 non-looped pilot traces, excluding positions inside any marker span (onset or merged) and positions before the reasoning start.

- positions skipped as before the reasoning start: 47
- positions skipped as inside a marker span: 82

| kind | 2-5 | 6-10 | 11-20 | 21-40 | all bands |
|---|---|---|---|---|---|
| `newline` | 28 | 14 | 61 | 117 | **220** |
| `period` | 17 | 39 | 87 | 172 | **315** |
| `comma` | 25 | 108 | 225 | 411 | **769** |
| `other_punct` | 71 | 100 | 211 | 466 | **848** |
| `digit` | 229 | 225 | 463 | 908 | **1825** |
| `word` | 713 | 838 | 1595 | 3183 | **6329** |
| `other` | 68 | 105 | 225 | 438 | **836** |
| **total** | 1151 | 1429 | 2867 | 5695 | **11142** |

## 6. Quiet zone (§6)

> 100 tokens, unless for any token kind the pilot count of negative
> positions (no marker in the next 100 tokens) is smaller than the pilot
> count of positive positions of that kind in the largest band (21-40);
> then 60.

| kind | negatives at 100 | positives in 21-40 | negatives < positives? |
|---|---|---|---|
| `newline` | 559 | 117 | no |
| `period` | 630 | 172 | no |
| `comma` | 1164 | 411 | no |
| `other_punct` | 2044 | 466 | no |
| `digit` | 3838 | 908 | no |
| `word` | 10454 | 3183 | no |
| `other` | 1307 | 438 | no |

Kinds failing the test: none → **quiet zone = 100 tokens**.

## 7. Matching kinds (§6)

> A kind stands on its own if at least 20 pilot positive positions
> (across all bands) fall in it; rarer kinds merge into `other`.

| kind | positives, all bands | ≥ 20? | outcome |
|---|---|---|---|
| `newline` | 220 | yes | stands alone |
| `period` | 315 | yes | stands alone |
| `comma` | 769 | yes | stands alone |
| `other_punct` | 848 | yes | stands alone |
| `digit` | 1825 | yes | stands alone |
| `word` | 6329 | yes | stands alone |
| `other` | 836 | yes | stands alone |

**Final kind list: newline, period, comma, other_punct, digit, word, other.** Merged into `other`: none.

## 8. Generation time projection

- N = 190 traces
- mean new tokens per non-looped gsm8k pilot trace = 1480.5
- at 63 tokens/second → **1.2 hours** (74 minutes)

Using the mean over all 25 non-looped pilot traces (2041.5 tokens) instead gives 1.7 hours.

