# Exploratory analyses (pre-registration §13, D25)

Everything in this file is **exploratory**. It feeds no registered decision rule and does not change, replace or qualify any confirmatory number; confirmatory values appear only as quotes from `results/verdicts.csv`, marked "quoted". Nothing was chosen or tuned on the test split: every setting was fixed on validation or pinned in D25 (X1–X8). No verdict is applied to any number here.

Computed by `scripts/run_exploratory.py` (analyses 1, 2, 5, 6, 7) and `scripts/plot_exploratory_traces.py` (analysis 3); plan `handoffs/14-exploratory-plan.md` as approved.

## Data contact (exploratory)

| Date | Analysis | What was read | What was not |
|---|---|---|---|
| 2026-11-21 | 6 | `results/run_traces.csv` (correct, looped, finish_reason) and `results/split.json`. | Any example row, activation or score. |
| 2026-11-21 | 5 | Band rows of all three splits (labels, onset, d, matched positive example); token ids and text of the 292 kept traces; paragraph-rule onsets. Tokenizer only. | Any activation, score or model. |
| 2026-11-21 | 7 | Band test rows with `same_trace`; saved states at each frozen probe's layer and the rival's features (layer 28, entropy, ids, onsets) for the test traces; the frozen probes and rivals. | Train or validation rows; any refit. |
| 2026-11-21 | 1 | Band and gate rows of train, validation and test; saved layers 1–27 at their positions; rival features for the band test rows; `probe_layers.csv`, `gate_layers.csv`. **The gate's test rows were read here for the first time** (the gate ran on validation only, §11). | Layer 28 as a probe input; any new C. |
| 2026-11-21 | 2 | Band rows of train and validation (fits and choices), then, after the choices were recorded, band test rows; rival features; frozen probe scores on test. | Any test row before the choices were recorded. |
| 2026-11-21 | 3 | `results/split.json` (test ids), 10 test trace files, the 1.5B model (teacher-forced over the stored ids), the four frozen probes, the saved activations of those 10 traces. | Any other trace; any refit. |

Frozen objects (`probes/`, `rival/`, `rival_7b/`, `examples/examples.csv.gz`): sha256 checked before and after every run, 14 checks, all equal to the recorded values: True.

## 1. Exact-distance curve

Exploratory: AUROC of the per-layer band probes (each layer at its validation-best C, refitted on train) on the band's test examples at each exact distance d, and the lift over the band's frozen rival on the same slice; d = 1 uses the gate probes and has no lift.

- Refits: 135 (27 gate, 108 band), not converged: 0, C tied at six decimals in the committed CSV: 0. Each refit's validation AUROC equals its committed CSV row (`a1_refits.csv`).
- A negative is counted at the distance of the positive example it was matched to.
- Wiring check: at each band's frozen layer and C the refit's coefficients equal the frozen probe's, and its all-distances test AUROC equals `results/verdicts.csv` exactly:

| band | frozen layer | C | coefficients identical | test AUROC equals verdicts.csv |
|---|---|---|---|---|
| 2-5 | 19 | 0.001 | True | True |
| 6-10 | 18 | 0.001 | True | True |
| 11-20 | 19 | 0.0001 | True | True |
| 21-40 | 19 | 0.0001 | True | True |

At each band's frozen layer (gate: its validation-best layer 27, C 0.01); every layer is in `a1_exact_distance.csv` and the heat maps:

| d | set | layer | n_pos | n_neg | probe AUROC | rival AUROC | lift |
|---|---|---|---|---|---|---|---|
| 1 | gate | 27 | 806 | 1612 | 0.9656 |  |  |
| 2 | 2-5 | 19 | 802 | 1604 | 0.9461 | 0.9312 | +0.0148 |
| 3 | 2-5 | 19 | 799 | 1598 | 0.9415 | 0.9202 | +0.0213 |
| 4 | 2-5 | 19 | 797 | 1594 | 0.9296 | 0.8942 | +0.0353 |
| 5 | 2-5 | 19 | 793 | 1586 | 0.9070 | 0.8733 | +0.0336 |
| 6 | 6-10 | 18 | 786 | 1572 | 0.8877 | 0.8426 | +0.0451 |
| 7 | 6-10 | 18 | 778 | 1556 | 0.8811 | 0.8394 | +0.0417 |
| 8 | 6-10 | 18 | 771 | 1542 | 0.8747 | 0.8228 | +0.0518 |
| 9 | 6-10 | 18 | 764 | 1528 | 0.8512 | 0.8133 | +0.0380 |
| 10 | 6-10 | 18 | 762 | 1524 | 0.8467 | 0.7978 | +0.0490 |
| 11 | 11-20 | 19 | 761 | 1522 | 0.8334 | 0.7969 | +0.0365 |
| 12 | 11-20 | 19 | 758 | 1516 | 0.8351 | 0.7866 | +0.0485 |
| 13 | 11-20 | 19 | 755 | 1510 | 0.8193 | 0.7831 | +0.0363 |
| 14 | 11-20 | 19 | 751 | 1502 | 0.8004 | 0.7623 | +0.0381 |
| 15 | 11-20 | 19 | 747 | 1494 | 0.8052 | 0.7701 | +0.0351 |
| 16 | 11-20 | 19 | 745 | 1490 | 0.8004 | 0.7773 | +0.0231 |
| 17 | 11-20 | 19 | 740 | 1480 | 0.7871 | 0.7659 | +0.0212 |
| 18 | 11-20 | 19 | 736 | 1472 | 0.7710 | 0.7423 | +0.0287 |
| 19 | 11-20 | 19 | 732 | 1464 | 0.7808 | 0.7582 | +0.0226 |
| 20 | 11-20 | 19 | 728 | 1456 | 0.7726 | 0.7569 | +0.0157 |
| 21 | 21-40 | 19 | 723 | 1446 | 0.7665 | 0.7343 | +0.0322 |
| 22 | 21-40 | 19 | 720 | 1440 | 0.7624 | 0.7266 | +0.0358 |
| 23 | 21-40 | 19 | 717 | 1434 | 0.7627 | 0.7378 | +0.0249 |
| 24 | 21-40 | 19 | 715 | 1430 | 0.7718 | 0.7364 | +0.0354 |
| 25 | 21-40 | 19 | 715 | 1430 | 0.7531 | 0.7109 | +0.0421 |
| 26 | 21-40 | 19 | 714 | 1428 | 0.7471 | 0.7132 | +0.0339 |
| 27 | 21-40 | 19 | 710 | 1420 | 0.7353 | 0.7153 | +0.0201 |
| 28 | 21-40 | 19 | 708 | 1416 | 0.7403 | 0.7058 | +0.0345 |
| 29 | 21-40 | 19 | 705 | 1410 | 0.7450 | 0.7120 | +0.0330 |
| 30 | 21-40 | 19 | 700 | 1400 | 0.7261 | 0.7074 | +0.0187 |
| 31 | 21-40 | 19 | 694 | 1388 | 0.7360 | 0.7175 | +0.0185 |
| 32 | 21-40 | 19 | 690 | 1380 | 0.7237 | 0.7115 | +0.0122 |
| 33 | 21-40 | 19 | 687 | 1374 | 0.7200 | 0.7093 | +0.0107 |
| 34 | 21-40 | 19 | 685 | 1370 | 0.7225 | 0.7031 | +0.0194 |
| 35 | 21-40 | 19 | 682 | 1364 | 0.7148 | 0.7128 | +0.0020 |
| 36 | 21-40 | 19 | 677 | 1354 | 0.7084 | 0.6981 | +0.0103 |
| 37 | 21-40 | 19 | 673 | 1346 | 0.6880 | 0.6844 | +0.0036 |
| 38 | 21-40 | 19 | 670 | 1340 | 0.6815 | 0.6691 | +0.0124 |
| 39 | 21-40 | 19 | 662 | 1324 | 0.6949 | 0.6801 | +0.0148 |
| 40 | 21-40 | 19 | 660 | 1320 | 0.6890 | 0.6834 | +0.0056 |

The band seams (5→6, 10→11, 20→21) are where one band's probe hands over to the next (and the rival changes with it); a jump in either curve at a seam may be the handover, not a change at that distance. Point estimates only; no interval.

Figures: `a1_curve.png` (AUROC by d at the layers above, seams dotted) and `a1_heatmap_<band>.png` (layer × d, AUROC and lift).

## 2. Each rival group alone

Exploratory: one reader per band and feature group (text; visible uncertainty = entropy; output state = layer 28; position), D21's recipe and scaling, chosen on validation, scored on test; the probe's lift over each with a 98.75% paired interval (D22's procedure and draws).

- Fits: 300 (text: 60 per band over the registered window × n-gram × C grid, the combined rival's tie rule; entropy, output state, position: 5 C values each, ties to the smaller C). Not converged: 0 (`a2_grid.csv`).
- Scaling (Q2 ruling): each group keeps D21's multiplier (r_out / r_g), so its columns are exactly its columns in the combined rival. Alone, the multiplier only rescales the input, and that scale interacts with C: the same C regularizes a rescaled group more or less strongly.
- Choices were written and hashed before any test row was read (`activations/exploratory/a2_choices.json`). Draw sha256 `1c267812f376…`, equal to step 13's.

| band | group | window | n-grams | C | C at grid edge | converged | validation AUROC | test AUROC | probe lift | 98.75% interval | combined rival AUROC (confirmatory, quoted) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2-5 | text | 16 | 1-3 | 0.0001 | True | True | 0.7652 | 0.7478 | +0.1828 | +0.1498 to +0.2169 | 0.9046 |
| 2-5 | entropy |  |  | 0.0001 | True | True | 0.5258 | 0.5616 | +0.3690 | +0.3409 to +0.3940 | 0.9046 |
| 2-5 | output |  |  | 0.001 | False | True | 0.8945 | 0.9076 | +0.0229 | +0.0157 to +0.0313 | 0.9046 |
| 2-5 | position |  |  | 0.1 | False | True | 0.7153 | 0.7038 | +0.2268 | +0.1741 to +0.2795 | 0.9046 |
| 6-10 | text | 32 | 1-3 | 0.0001 | True | True | 0.7339 | 0.7212 | +0.1472 | +0.1155 to +0.1772 | 0.8231 |
| 6-10 | entropy |  |  | 0.0001 | True | True | 0.5232 | 0.5387 | +0.3297 | +0.2978 to +0.3570 | 0.8231 |
| 6-10 | output |  |  | 0.001 | False | True | 0.8188 | 0.8276 | +0.0407 | +0.0314 to +0.0512 | 0.8231 |
| 6-10 | position |  |  | 1 | True | True | 0.7024 | 0.6984 | +0.1699 | +0.1272 to +0.2154 | 0.8231 |
| 11-20 | text | 64 | 1-3 | 0.0001 | True | True | 0.7008 | 0.6973 | +0.1035 | +0.0669 to +0.1376 | 0.7701 |
| 11-20 | entropy |  |  | 0.0001 | True | True | 0.5235 | 0.5338 | +0.2670 | +0.2239 to +0.3022 | 0.7701 |
| 11-20 | output |  |  | 0.001 | False | True | 0.7222 | 0.7564 | +0.0444 | +0.0331 to +0.0557 | 0.7701 |
| 11-20 | position |  |  | 0.01 | False | True | 0.6806 | 0.6670 | +0.1338 | +0.0901 to +0.1802 | 0.7701 |
| 21-40 | text | 64 | 1-3 | 1 | True | True | 0.6533 | 0.6539 | +0.0763 | +0.0361 to +0.1182 | 0.7090 |
| 21-40 | entropy |  |  | 0.0001 | True | True | 0.4914 | 0.5257 | +0.2045 | +0.1599 to +0.2500 | 0.7090 |
| 21-40 | output |  |  | 0.0001 | True | True | 0.6573 | 0.6744 | +0.0557 | +0.0423 to +0.0691 | 0.7090 |
| 21-40 | position |  |  | 0.1 | False | True | 0.6210 | 0.6153 | +0.1148 | +0.0649 to +0.1628 | 0.7090 |

## 3. Per-trace plots

Exploratory: the four frozen band probes' decision function at every reasoning token of 10 test traces (`random.Random(20261021).sample(sorted(test problems), 10)`), from a teacher-forced pass of the 1.5B over the stored ids, with the text alongside.

- Traces, in draw order: 805, 599, 881, 5, 590, 609, 460, 1029, 1037, 338.
- Check: at every saved example position of these traces the new score was compared with the score from the saved activation; largest absolute difference 0 (tolerance 0.001; `a3_check.csv`).

| file | trace | reasoning tokens | saved positions compared |
|---|---|---|---|
| `traces/trace_01.html` | 805 | 908 | 442 |
| `traces/trace_02.html` | 599 | 1087 | 581 |
| `traces/trace_03.html` | 881 | 2332 | 1050 |
| `traces/trace_04.html` | 5 | 2783 | 1614 |
| `traces/trace_05.html` | 590 | 3556 | 2028 |
| `traces/trace_06.html` | 609 | 785 | 371 |
| `traces/trace_07.html` | 460 | 679 | 314 |
| `traces/trace_08.html` | 1029 | 1230 | 692 |
| `traces/trace_09.html` | 1037 | 607 | 282 |
| `traces/trace_10.html` | 338 | 595 | 224 |

## 4. The add-on (§12)

Exploratory: not run. It needs at least 20 hits in each group; the audit found 13 real hits (`results/audit.md`), so it is not runnable (D25 X5).

## 5. Where each band falls relative to paragraph breaks

Exploratory: for each band, the share of positives in the onset's own paragraph, the paragraph before it, or earlier, and the median tokens since the last paragraph break for positives and their matched negatives; all splits, labels and text only.

- Paragraphs (Q5): the blank lines (`\n\n`) between the start of token t and the onset's first character, counted as D11's onset rule counts them (empty pieces included); 0 = own paragraph, 1 = previous, 2 or more = earlier. Tokens since the last break: the rival's position feature (R4, `run_rival.since_break`).

| band | split | n_pos | own paragraph | previous | earlier | median since break, positives | median since break, negatives |
|---|---|---|---|---|---|---|---|
| 2-5 | all | 9966 | 0.2657 | 0.7339 | 0.0004 | 40 | 22 |
| 2-5 | train | 4955 | 0.2710 | 0.7282 | 0.0008 | 39 | 23 |
| 2-5 | validation | 1820 | 0.2560 | 0.7440 | 0.0000 | 43.5 | 20 |
| 2-5 | test | 3191 | 0.2629 | 0.7371 | 0.0000 | 37 | 22 |
| 6-10 | all | 12053 | 0.2615 | 0.7234 | 0.0151 | 37 | 22 |
| 6-10 | train | 6028 | 0.2664 | 0.7198 | 0.0138 | 37 | 23 |
| 6-10 | validation | 2164 | 0.2542 | 0.7408 | 0.0051 | 41.5 | 21 |
| 6-10 | test | 3861 | 0.2580 | 0.7192 | 0.0228 | 36 | 23 |
| 11-20 | all | 23210 | 0.2413 | 0.6592 | 0.0995 | 35 | 23 |
| 11-20 | train | 11623 | 0.2434 | 0.6688 | 0.0878 | 34 | 23 |
| 11-20 | validation | 4134 | 0.2397 | 0.6609 | 0.0994 | 38 | 21 |
| 11-20 | test | 7453 | 0.2390 | 0.6434 | 0.1177 | 33 | 23 |
| 21-40 | all | 43339 | 0.1714 | 0.5669 | 0.2617 | 31 | 23 |
| 21-40 | train | 21656 | 0.1731 | 0.5745 | 0.2524 | 32 | 24 |
| 21-40 | validation | 7776 | 0.1707 | 0.5688 | 0.2605 | 33 | 22 |
| 21-40 | test | 13907 | 0.1691 | 0.5540 | 0.2769 | 29 | 23 |

## 6. Accuracy, as context

Exploratory: accuracy of the 292 kept traces, overall and per split, with the number that hit the 4,096-token length cap; nothing is related to accuracy.

| split | kept traces | correct | accuracy | at length cap |
|---|---|---|---|---|
| all | 292 | 241 | 0.8253 | 20 |
| train | 146 | 121 | 0.8288 | 11 |
| validation | 44 | 37 | 0.8409 | 2 |
| test | 102 | 83 | 0.8137 | 7 |

## 7. Same-trace-only negatives

Exploratory: the frozen probe and frozen rival of each band scored on the band's test examples with every cross-trace negative removed, no refit and no rematching; lift and 98.75% paired interval (D22's draws), next to the confirmatory values.

- Removed counts equal `results/examples_summary.md` table e (test column). Before filtering, the same code reproduced `verdicts.csv`'s probe AUROC, rival AUROC and lift exactly.

| band | negatives removed | n_pos | n_neg | probe AUROC | rival AUROC | lift | 98.75% interval | confirmatory lift (quoted) | confirmatory interval (quoted) |
|---|---|---|---|---|---|---|---|---|---|
| 2-5 | 28 | 3191 | 6354 | 0.9308 | 0.9047 | +0.0261 | +0.0106 to +0.0449 | +0.0260 | +0.0105 to +0.0448 |
| 6-10 | 68 | 3861 | 7654 | 0.8682 | 0.8229 | +0.0453 | +0.0217 to +0.0756 | +0.0453 | +0.0216 to +0.0755 |
| 11-20 | 641 | 7453 | 14265 | 0.8006 | 0.7707 | +0.0298 | -0.0034 to +0.0673 | +0.0307 | -0.0019 to +0.0675 |
| 21-40 | 2254 | 13907 | 25560 | 0.7300 | 0.7106 | +0.0194 | -0.0204 to +0.0644 | +0.0212 | -0.0186 to +0.0665 |

## Files

All under `results/exploratory/`; sha256 and bytes in `manifest.csv`.
