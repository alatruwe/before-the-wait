# Verdicts: probe vs combined rival on test, per band

Computed by `scripts/run_verdicts.py`. §9, §10, §11, §14; D22 (V1–V6); plan `handoffs/13-verdicts-plan.md` as approved. Printed to 4 decimals; `results/verdicts.csv` holds the unrounded values.

## a. Data contact

| Date | What was touched | What was not |
|---|---|---|
| 2026-11-15 | The test split of the four bands (2-5, 6-10, 11-20, 21-40): labels, activations at each band's chosen probe layer and at layer 28, entropy and prompt entropy, token ids and paragraph-rule onsets. First and only scoring. Train rows of the four bands (labels and chosen layer) for the shuffled-labels control; validation rows for the rehearsal. | The gate rows; any other layer; any exploratory analysis (§13); the hand audit and add-on (§12). |

## b. Frozen objects scored

| band | probe layer | probe C | probe sha256 | rival window | rival n-grams | rival C | rival sha256 |
|---|---|---|---|---|---|---|---|
| 2-5 | 19 | 0.001 | `02bcbe5f8184c1301f822b08c3afc939d05b2bc832e438a54f189c3a8719ccea` | 16 | 1-3 | 0.001 | `89bd053af2d5ab068a4037a14b916f9d25a46aa0486d4d6673c87ea4e045552b` |
| 6-10 | 18 | 0.001 | `bf9bd8708a4abcd6ff9a0643288cd86341d70dab9955b21cd9950b9a16133dfa` | 32 | 1-3 | 0.0001 | `111c5ec5bd5f8426e5fc9b76a943a259c4594411d6e0c7ea8b97290bd981af56` |
| 11-20 | 19 | 0.0001 | `95cff87fc495e4547dfd21c166539aeb96558e261aa00b0253bd69dbbe863739` | 64 | 1-3 | 0.0001 | `3f1a617acf8e05ef87c808cf5647232b4c0e8b565dd2760724ee45de3fd5dd06` |
| 21-40 | 19 | 0.0001 | `02075f7c4f62e34fbcfe7ec7d4585efbaf6e21130eff994fa72a54fa6f1164b9` | 64 | 1-3 | 0.0001 | `4ea2e1b88940f84872cc3e1d8c92207076c9941f631838332c310c487bd743c2` |

Both sets of files were byte-identical (sha256) before and after the run.

## c. Rehearsal on validation

The scoring path of this script, run on the validation examples before any test row was read (`--rehearse`). Reproduced = equal to the printed table value (4 decimals), to the grid CSV (6 decimals) and, for rivals, exactly equal as a float to the unrounded value in the step-12 cell cache; and the G4 95% interval (2,000 resamples, same draws) reproduced to 4 decimals.

| band | reader | validation AUROC | table | grid CSV | 95% interval | reproduced |
|---|---|---|---|---|---|---|
| 2-5 | probe | 0.9177 | 0.9177 | 0.917655 | 0.9031–0.9313 | yes |
| 2-5 | rival | 0.8993 | 0.8993 | 0.899257 | 0.8824–0.9150 | yes |
| 6-10 | probe | 0.8527 | 0.8527 | 0.852655 | 0.8283–0.8754 | yes |
| 6-10 | rival | 0.8350 | 0.8350 | 0.835002 | 0.8095–0.8603 | yes |
| 11-20 | probe | 0.7732 | 0.7732 | 0.773173 | 0.7410–0.8031 | yes |
| 11-20 | rival | 0.7645 | 0.7645 | 0.764542 | 0.7345–0.7963 | yes |
| 21-40 | probe | 0.6968 | 0.6968 | 0.696802 | 0.6562–0.7361 | yes |
| 21-40 | rival | 0.6824 | 0.6824 | 0.682365 | 0.6529–0.7136 | yes |

## d. Shuffled-labels control (§9)

Train labels permuted (numpy `default_rng(20261019).permutation`, per band); probe refitted at the chosen layer and C with the frozen probe's standardization (train mean and SD, recomputed and identical); scored on the band's test examples with their true labels. Pass: 0.45 ≤ AUROC ≤ 0.55.

| band | AUROC | converged | iterations | pass |
|---|---|---|---|---|
| 2-5 | 0.5143 | True | 28 | pass |
| 6-10 | 0.5383 | True | 28 | pass |
| 11-20 | 0.5048 | True | 15 | pass |
| 21-40 | 0.5119 | True | 16 | pass |

## e. Verdicts at δ = 0.05

Lift = probe AUROC − rival AUROC on the band's test examples. Interval: 98.75%, paired bootstrap over whole test problems, 10,000 resamples (V1–V3).

| band | probe AUROC | rival AUROC | lift | δ | 98.75% interval | verdict |
|---|---|---|---|---|---|---|
| 2-5 | 0.9306 | 0.9046 | +0.0260 | 0.05 | +0.0105 to +0.0448 | null |
| 6-10 | 0.8684 | 0.8231 | +0.0453 | 0.05 | +0.0216 to +0.0755 | positive |
| 11-20 | 0.8008 | 0.7701 | +0.0307 | 0.05 | -0.0019 to +0.0675 | inconclusive |
| 21-40 | 0.7302 | 0.7090 | +0.0212 | 0.05 | -0.0186 to +0.0665 | inconclusive |

Bands with a positive verdict at δ = 0.05 (§11, Claim): 6-10.

## f. Secondary verdicts (secondary; never the headline)

| band | verdict at δ = 0.03 | verdict at δ = 0.08 |
|---|---|---|
| 2-5 | positive | null |
| 6-10 | positive | null |
| 11-20 | inconclusive | null |
| 21-40 | inconclusive | null |

## g. Counts and the both-near-chance flag

Onsets: clean test onsets from `results/examples_summary.md` table d. Problems: test problems with at least one example in the band (of 102 resampled).

| band | onsets | positives | negatives | problems | ≥ m = 555 | both within 0.05 of 0.5 (§14) |
|---|---|---|---|---|---|---|
| 2-5 | 794 | 3191 | 6382 | 102 | yes | no |
| 6-10 | 762 | 3861 | 7722 | 102 | yes | no |
| 11-20 | 726 | 7453 | 14906 | 102 | yes | no |
| 21-40 | 661 | 13907 | 27814 | 102 | yes | no |

All four bands are at or above m = 555 test onsets, so §11's below-minimum clause does not apply.

## h. V1–V6 as implemented

- V1: interval = `np.percentile(lifts, [0.625, 99.375])` (linear), dropped resamples excluded.
- V2: `random.Random(20261019)`, fresh per band, `randrange(102)` × 102 per resample over the sorted test problem ids of `results/split.json` (the G4 draw order). The 10,000 × 102 draw matrix was built before any test row was read; all bands share it. sha256 (int64 bytes): `1c267812f3760abd36a060fbb2ae72dd3c6fbb671b45e690dccca684ee72716b`.
- V3: an example's problem is its own trace (`trace_index`), so a cross-trace negative travels with its own trace. A drawn problem with no examples in the band contributes none (Q2 ruling). Probe and rival are scored on the same resampled indices. Dropped (single-class) resamples per band: 2-5 0, 6-10 0, 11-20 0, 21-40 0.
- V4: see d. Permutation `np.random.default_rng(20261019).permutation(n_train)` over the band's train rows in `examples.csv.gz` order; same solver settings as the probes (L2, lbfgs, max_iter 10,000, single-threaded BLAS).
- V5: `--rehearse` (validation) passed and recorded this script's sha256; `--run` checked it, ran the control on all four bands, and computed no real test score until all four passed.
- V6: verdicts on unrounded values: positive = lower > 0 and upper ≥ δ; null = upper < δ; inconclusive = lower ≤ 0 and upper ≥ δ (exhaustive, asserted).
- numpy 2.5.4, scikit-learn 1.9.1, scipy 1.18.1.
