# Count calculations for the pre-registration

Computed by `scripts/count_calculations.py` from the pilot results. Every
number below is produced by a rule already written in
`methodology/pre-registration.md`; nothing here is chosen.

## Summary

| Quantity | Value | Rule |
|---|---|---|
| Minimum events m (δ = 0.05) | **555** per band | §11 |
| Trace count from the rule, N_rule | **264** | §4 (all four bands) |
| **Final trace count N** | **320** gsm8k traces | §4 / D12 (×1.2, up to the nearest 10) |
| Binding band | 21-40 | §4 |
| Split | **50 / 15 / 35** | §4 |
| Quiet zone | **100 tokens** | §6 |
| Matching kinds | **newline, period, comma, other_punct, digit, word, other** | §6 |
| Projected generation | **2.6 h** | handoff |
| Projected activation disk | **21.9 GB** | handoff |

## Changes from the 2026-10-17 run

Recomputed 2026-10-19 under the revised positive rule (no lexicon hit of
any kind strictly between a position and its onset; e_b measured from the
previous hit of any kind), m assumptions (AUROC 0.6, correlation
0.3) and 1:2 matching. Before = commit `fed3fa6`.

| Quantity | 2026-10-17 | 2026-10-19 |
|---|---|---|
| m (δ = 0.05) | 338 | **555** |
| N_rule | 157 | **264** |
| N | 190 | **320** |
| Binding band | 21-40 | **21-40** |
| Pilot positive positions, 2-5 | 1151 | **1149** |
| Pilot positive positions, 6-10 | 1429 | **1383** |
| Pilot positive positions, 11-20 | 2867 | **2664** |
| Pilot positive positions, 21-40 | 5695 | **4950** |
| GSM8K onsets with gap > 5 (e_b numerator), 2-5 | 114 | **114** |
| GSM8K onsets with gap > 10 (e_b numerator), 6-10 | 110 | **108** |
| GSM8K onsets with gap > 20 (e_b numerator), 11-20 | 106 | **104** |
| GSM8K onsets with gap > 40 (e_b numerator), 21-40 | 96 | **94** |
| Quiet zone | 100 | **100** |

## Inputs

| Symbol | Meaning | Value | Source |
|---|---|---|---|
| δ | null bar | 0.05 (provisional) | §11 |
| A | assumed AUROC for m | 0.6 | §11 |
| ρ | assumed probe–rival correlation for m | 0.3 | §11 |
| matching | negatives per positive | 1:2 | §6 |
| r | test share | 0.35 | §4 split |
| ℓ | looped-trace share | 5/30 = 0.1667 | `c1_traces.csv` |
| o (GSM8K) | onsets per non-looped trace | 114/13 = 8.7692 | `c1_onsets_paragraph.csv` |
| o (MATH500) | onsets per non-looped trace | 175/12 = 14.5833 | `c1_onsets_paragraph.csv` |
| margin | D12 inflation on N_rule | ×1.2, rounded up to the nearest 10 | §4 / D12 |
| cap | §4 cap on N | 1000 | §4 / D12 |

## 1. Minimum events m (§11)

> m = the smallest number of test-split onsets per band such that the
> half-width of the 98.75% interval on the lift is ≤ δ, using the
> Hanley–McNeil standard error of an AUROC of 0.6 on m positives and m
> matched negatives for each of the two AUROCs, combined with an assumed
> correlation of 0.3 between them.

- interval 0.9875 two-sided → z = **2.4977**
- assumed AUROC 0.6 and correlation 0.3, stated before any probe exists
- SE_lift = √(2(1 − ρ)) × SE_AUROC = √1.4 × SE_AUROC

| δ | m | SE_AUROC at m | SE_lift at m | half-width z·SE_lift |
|---|---|---|---|---|
| 0.03 | **1540** | 0.01015 | 0.01201 | 0.02999 |
| 0.05 (primary) | **555** | 0.01691 | 0.02001 | 0.04998 |
| 0.08 | **217** | 0.02706 | 0.03202 | 0.07998 |

## 2. Eligibility shares e_b (§4)

> e_b: the share of onsets whose previous lexicon hit of any kind, onset or
> merged (or the reasoning start, for a first hit), lies more than the
> band's upper bound tokens back.

**gsm8k** (114 onsets over 13 non-looped traces)

| band | upper bound | onsets with gap > bound | e_b |
|---|---|---|---|
| 2-5 | 5 | 114 / 114 | **1.0000** |
| 6-10 | 10 | 108 / 114 | **0.9474** |
| 11-20 | 20 | 104 / 114 | **0.9123** |
| 21-40 | 40 | 94 / 114 | **0.8246** |

**math500** (175 onsets over 12 non-looped traces)

| band | upper bound | onsets with gap > bound | e_b |
|---|---|---|---|
| 2-5 | 5 | 172 / 175 | **0.9829** |
| 6-10 | 10 | 165 / 175 | **0.9429** |
| 11-20 | 20 | 156 / 175 | **0.8914** |
| 21-40 | 40 | 140 / 175 | **0.8000** |

## 3. Trace count N (§4)

> N = the smallest integer such that N × r × (1 − ℓ) × o × e_b ≥ m for
> each of the four bands ... The final count is that N inflated by 20%
> and rounded up to the nearest ten, because o is estimated from 13
> pilot traces. Capped at 1,000. (D12)

With m = 555, r = 0.35, ℓ = 0.1667, o = 8.7692 (GSM8K):

| band | e_b | onsets per trace in band | N | required by §4 |
|---|---|---|---|---|
| 2-5 | 1.0000 | 2.5577 | 217 | yes |
| 6-10 | 0.9474 | 2.4231 | 230 | yes |
| 11-20 | 0.9123 | 2.3333 | 238 | yes |
| 21-40 | 0.8246 | 2.1090 | 264 | yes |

**Binding band: 21-40 → N_rule = 264 GSM8K traces.**

D12 margin: ceil(1.2 × 264) = 317, rounded up to the nearest 10 → **N = 320**.

Switch rule: N ≤ cap (1000), so the MATH500 switch does not apply.

For the record, the same computation on MATH500 (o = 14.5833) gives N_rule = 164 at binding band 21-40, N = 200.

Projected test-split onsets per band at the final N, against m:

| band | projected test onsets at N | ≥ m? |
|---|---|---|
| 2-5 | 818.5 | yes |
| 6-10 | 775.4 | yes |
| 11-20 | 746.7 | yes |
| 21-40 | 674.9 | yes |

## 4. Split (§4)

> train / validation / test = 50 / 15 / 35, unless N × (1 − ℓ) × o × 0.15
> < 100 projected validation onsets, in which case 45 / 20 / 35.

- N × (1 − ℓ) × o × 0.15 = 320 × 0.8333 × 8.7692 × 0.15 = **350.8** projected validation onsets
- 350.8 ≥ 100 → **50 / 15 / 35**

## 5. Positive positions by token kind (§6)

Tokens at distances 2–40 before each paragraph-rule onset in the 25 non-looped pilot traces, excluding positions before the reasoning start, positions inside any marker span (onset or merged), and positions with a lexicon hit of any kind strictly between them and the onset (so each position belongs only to its nearest onset).

- positions skipped as before the reasoning start: 47
- positions skipped as inside a marker span: 82
- positions skipped as having a hit in between: 996

| kind | 2-5 | 6-10 | 11-20 | 21-40 | all bands |
|---|---|---|---|---|---|
| `newline` | 28 | 10 | 54 | 95 | **187** |
| `period` | 17 | 31 | 82 | 153 | **283** |
| `comma` | 24 | 104 | 206 | 352 | **686** |
| `other_punct` | 71 | 98 | 202 | 420 | **791** |
| `digit` | 229 | 215 | 425 | 798 | **1667** |
| `word` | 712 | 821 | 1486 | 2745 | **5764** |
| `other` | 68 | 104 | 209 | 387 | **768** |
| **total** | 1149 | 1383 | 2664 | 4950 | **10146** |

## 6. Quiet zone (§6)

> 100 tokens, unless for any token kind the pilot count of negative
> positions (no marker in the next 100 tokens) is smaller than twice the
> pilot count of positive positions of that kind in band 21-40
> (1:2 matching); then 60.

| kind | negatives at 100 | positives in 21-40 | 2× positives | negatives < 2× positives? |
|---|---|---|---|---|
| `newline` | 559 | 95 | 190 | no |
| `period` | 630 | 153 | 306 | no |
| `comma` | 1164 | 352 | 704 | no |
| `other_punct` | 2044 | 420 | 840 | no |
| `digit` | 3838 | 798 | 1596 | no |
| `word` | 10454 | 2745 | 5490 | no |
| `other` | 1307 | 387 | 774 | no |

Kinds failing the test: none → **quiet zone = 100 tokens**.

## 7. Matching kinds (§6)

> A kind stands on its own if at least 20 pilot positive positions
> (across all bands) fall in it; rarer kinds merge into `other`.

| kind | positives, all bands | ≥ 20? | outcome |
|---|---|---|---|
| `newline` | 187 | yes | stands alone |
| `period` | 283 | yes | stands alone |
| `comma` | 686 | yes | stands alone |
| `other_punct` | 791 | yes | stands alone |
| `digit` | 1667 | yes | stands alone |
| `word` | 5764 | yes | stands alone |
| `other` | 768 | yes | stands alone |

**Final kind list: newline, period, comma, other_punct, digit, word, other.** Merged into `other`: none.

## 8. Generation time projection

- N = 320 traces
- mean new tokens per non-looped gsm8k pilot trace = 1480.5
- mean new tokens per gsm8k pilot trace, looped included = 1829.3
- at 63 tokens/second → **2.6 hours** (155 minutes)

Looped traces are generated before they are excluded, so the mean includes them. Over non-looped gsm8k traces only (1480.5 tokens) it is 2.1 hours; over all 25 non-looped pilot traces (2041.5 tokens), 2.9 hours.

## 9. Activation disk projection

Hidden states at layers 1–28, width 1,536, fp16: 28 × 1,536 × 2 = 86,016 bytes (~86 KB) per position. Positives projected from the 13 non-looped gsm8k pilot traces under the revised rule, scaled to N × (1 − ℓ) = 266.7 kept traces.

| class | per kept pilot trace | projected at N |
|---|---|---|
| gate positives (d = 1) | 8.77 | 2,338 |
| band 2-5 positives | 35.08 | 9,354 |
| band 6-10 positives | 42.15 | 11,241 |
| band 11-20 positives | 81.23 | 21,662 |
| band 21-40 positives | 150.31 | 40,082 |
| **all positives** | | **84,677** |
| negatives, 1:2 | | **169,354** |
| **total positions** | | **254,031** |

→ **21.9 GB** (20.4 GiB). The negatives figure is an upper bound: it gives every band and the gate its own 2 negatives per positive. §6 allows reuse across bands (never within one), so the real number of distinct negatives is lower.

