# Outside rival: combined, per band, chosen on validation

Computed by `scripts/run_rival.py`. §8, D13, D19, D20, D21 (R1–R6); plan `handoffs/12-rival-plan.md` as approved.

## a. Data contact

| Date | What was touched | What was not |
|---|---|---|
| 2026-11-14 | Band examples (2-5, 6-10, 11-20, 21-40) of the train and validation splits: labels, layer-28 state, entropy and prompt entropy, token ids, paragraph-rule onsets. | The test split (no row loaded, no activation opened); layers 1-27; any probe output; the gate rows. |

## b. Counts

| band | split | positives | negatives | problems |
|---|---|---|---|---|
| 2-5 | train | 4955 | 9910 | 146 |
| 2-5 | validation | 1820 | 3640 | 44 |
| 6-10 | train | 6028 | 12056 | 146 |
| 6-10 | validation | 2164 | 4328 | 44 |
| 11-20 | train | 11623 | 23246 | 146 |
| 11-20 | validation | 4134 | 8268 | 44 |
| 21-40 | train | 21656 | 43312 | 146 |
| 21-40 | validation | 7776 | 15552 | 44 |

## c. Definitions as implemented

scikit-learn 1.9.1, scipy 1.18.1, numpy 2.5.4. At an example (trace, t): P = prompt length, i = P + t, ids = prompt ids + generated ids.

- **Text (R1, R2).** Window w ∈ {8, 16, 32, 64, 128}: ids[i − w + 1 … i]; "all": ids[0 … i]. Each window is written as its token ids separated by single spaces and **split on whitespace** (`tokenizer=str.split, token_pattern=None, lowercase=False`), so every id is one token (the default token pattern would drop one-character ids). `TfidfVectorizer(ngram_range=(1, n), min_df=2)`, all other settings default (l2 norm, smoothed IDF, raw counts); vocabulary and IDF fitted on the band's train examples.
- **Entropy (D19, D20).** Entropy at t (raw softmax, nats); mean and max over t−16 … t−1, from the prompt entropy where that range reaches before generation; t excluded.
- **Output state (D13).** The saved layer-28 state at t (final norm applied), 1,536 columns.
- **Position (R4).** t; t minus the last paragraph-rule onset before t, else P + t; i minus the input index of the token in which the last "\n\n" at or before i ends (located with `align.prefix_lengths`, the incremental decode the onsets use), else P + t.
- **Scaling (R3).** Entropy, output-state and position columns z-scored with the band's train mean and SD (a zero-variance column gets SD 1). Group g multiplied by r_out / r_g, r_g = sqrt(mean over train rows of the squared row norm of g). One sparse float64 matrix: text, entropy, output state, position.
- **Classifier (R5).** Logistic regression, L2, lbfgs, max_iter 10,000, intercept, no class weights; C ∈ {1e-4, 1e-3, 1e-2, 1e-1, 1}; scored by the decision function. Chosen: highest validation AUROC; ties to the smaller window, then (1,2), then the smaller C. A fit that did not converge stays eligible and is flagged.
- **Frozen (R6).** The chosen rival is the one fitted on train, saved per band and reloaded from its file alone to reproduce its validation AUROC exactly.

## d. Group factors and multipliers

Dense groups (per band, from the train examples):

| band | r_entropy | × entropy | r_output | × output | r_position | × position |
|---|---|---|---|---|---|---|
| 2-5 | 1.7321 | 22.6274 | 39.1918 | 1.0000 | 1.7321 | 22.6274 |
| 6-10 | 1.7321 | 22.6274 | 39.1918 | 1.0000 | 1.7321 | 22.6274 |
| 11-20 | 1.7321 | 22.6274 | 39.1918 | 1.0000 | 1.7321 | 22.6274 |
| 21-40 | 1.7321 | 22.6274 | 39.1918 | 1.0000 | 1.7321 | 22.6274 |

Text group, per text configuration (chosen configuration marked):

| band | window | n-grams | vocabulary | r_text | × text | all-zero text rows (train / val) | chosen |
|---|---|---|---|---|---|---|---|
| 2-5 | 8 | 1-2 | 10,651 | 1.0000 | 39.1918 | 0 / 0 |  |
| 2-5 | 8 | 1-3 | 24,404 | 1.0000 | 39.1918 | 0 / 0 |  |
| 2-5 | 16 | 1-2 | 15,269 | 1.0000 | 39.1918 | 0 / 0 |  |
| 2-5 | 16 | 1-3 | 39,997 | 1.0000 | 39.1918 | 0 / 0 | yes |
| 2-5 | 32 | 1-2 | 19,970 | 1.0000 | 39.1918 | 0 / 0 |  |
| 2-5 | 32 | 1-3 | 56,438 | 1.0000 | 39.1918 | 0 / 0 |  |
| 2-5 | 64 | 1-2 | 24,385 | 1.0000 | 39.1918 | 0 / 0 |  |
| 2-5 | 64 | 1-3 | 72,261 | 1.0000 | 39.1918 | 0 / 0 |  |
| 2-5 | 128 | 1-2 | 28,179 | 1.0000 | 39.1918 | 0 / 0 |  |
| 2-5 | 128 | 1-3 | 86,417 | 1.0000 | 39.1918 | 0 / 0 |  |
| 2-5 | all | 1-2 | 28,699 | 1.0000 | 39.1918 | 0 / 0 |  |
| 2-5 | all | 1-3 | 88,154 | 1.0000 | 39.1918 | 0 / 0 |  |
| 6-10 | 8 | 1-2 | 11,673 | 1.0000 | 39.1918 | 0 / 0 |  |
| 6-10 | 8 | 1-3 | 27,338 | 1.0000 | 39.1918 | 0 / 0 |  |
| 6-10 | 16 | 1-2 | 16,594 | 1.0000 | 39.1918 | 0 / 0 |  |
| 6-10 | 16 | 1-3 | 44,088 | 1.0000 | 39.1918 | 0 / 0 |  |
| 6-10 | 32 | 1-2 | 20,944 | 1.0000 | 39.1918 | 0 / 0 |  |
| 6-10 | 32 | 1-3 | 59,825 | 1.0000 | 39.1918 | 0 / 0 | yes |
| 6-10 | 64 | 1-2 | 25,046 | 1.0000 | 39.1918 | 0 / 0 |  |
| 6-10 | 64 | 1-3 | 74,664 | 1.0000 | 39.1918 | 0 / 0 |  |
| 6-10 | 128 | 1-2 | 28,353 | 1.0000 | 39.1918 | 0 / 0 |  |
| 6-10 | 128 | 1-3 | 87,087 | 1.0000 | 39.1918 | 0 / 0 |  |
| 6-10 | all | 1-2 | 28,741 | 1.0000 | 39.1918 | 0 / 0 |  |
| 6-10 | all | 1-3 | 88,306 | 1.0000 | 39.1918 | 0 / 0 |  |
| 11-20 | 8 | 1-2 | 15,902 | 1.0000 | 39.1918 | 0 / 0 |  |
| 11-20 | 8 | 1-3 | 40,634 | 1.0000 | 39.1918 | 0 / 0 |  |
| 11-20 | 16 | 1-2 | 20,275 | 1.0000 | 39.1918 | 0 / 0 |  |
| 11-20 | 16 | 1-3 | 56,764 | 1.0000 | 39.1918 | 0 / 0 |  |
| 11-20 | 32 | 1-2 | 23,494 | 1.0000 | 39.1918 | 0 / 0 |  |
| 11-20 | 32 | 1-3 | 68,787 | 1.0000 | 39.1918 | 0 / 0 |  |
| 11-20 | 64 | 1-2 | 26,574 | 1.0000 | 39.1918 | 0 / 0 |  |
| 11-20 | 64 | 1-3 | 80,251 | 1.0000 | 39.1918 | 0 / 0 | yes |
| 11-20 | 128 | 1-2 | 28,682 | 1.0000 | 39.1918 | 0 / 0 |  |
| 11-20 | 128 | 1-3 | 88,231 | 1.0000 | 39.1918 | 0 / 0 |  |
| 11-20 | all | 1-2 | 28,855 | 1.0000 | 39.1918 | 0 / 0 |  |
| 11-20 | all | 1-3 | 88,748 | 1.0000 | 39.1918 | 0 / 0 |  |
| 21-40 | 8 | 1-2 | 20,002 | 1.0000 | 39.1918 | 0 / 0 |  |
| 21-40 | 8 | 1-3 | 55,316 | 1.0000 | 39.1918 | 0 / 0 |  |
| 21-40 | 16 | 1-2 | 22,945 | 1.0000 | 39.1918 | 0 / 0 |  |
| 21-40 | 16 | 1-3 | 66,633 | 1.0000 | 39.1918 | 0 / 0 |  |
| 21-40 | 32 | 1-2 | 25,365 | 1.0000 | 39.1918 | 0 / 0 |  |
| 21-40 | 32 | 1-3 | 75,765 | 1.0000 | 39.1918 | 0 / 0 |  |
| 21-40 | 64 | 1-2 | 27,890 | 1.0000 | 39.1918 | 0 / 0 |  |
| 21-40 | 64 | 1-3 | 85,388 | 1.0000 | 39.1918 | 0 / 0 | yes |
| 21-40 | 128 | 1-2 | 28,820 | 1.0000 | 39.1918 | 0 / 0 |  |
| 21-40 | 128 | 1-3 | 88,676 | 1.0000 | 39.1918 | 0 / 0 |  |
| 21-40 | all | 1-2 | 28,909 | 1.0000 | 39.1918 | 0 / 0 |  |
| 21-40 | all | 1-3 | 88,931 | 1.0000 | 39.1918 | 0 / 0 |  |

## e. Chosen rival per band

Interval: 95%, 2,000 resamples of whole validation problems, seed 20261019 (G4). Fits not converged in the grid: 0 of 240 (`results/rival_grid.csv`).

| band | window | n-grams | C | validation AUROC | 95% interval | converged | file bytes | sha256 |
|---|---|---|---|---|---|---|---|---|
| 2-5 | 16 | 1-3 | 0.001 | 0.8993 | 0.8824–0.9150 | True | 3,719,062 | `89bd053af2d5ab068a4037a14b916f9d25a46aa0486d4d6673c87ea4e045552b` |
| 6-10 | 32 | 1-3 | 0.0001 | 0.8350 | 0.8095–0.8603 | True | 5,543,238 | `111c5ec5bd5f8426e5fc9b76a943a259c4594411d6e0c7ea8b97290bd981af56` |
| 11-20 | 64 | 1-3 | 0.0001 | 0.7645 | 0.7345–0.7963 | True | 7,422,430 | `3f1a617acf8e05ef87c808cf5647232b4c0e8b565dd2760724ee45de3fd5dd06` |
| 21-40 | 64 | 1-3 | 0.0001 | 0.6824 | 0.6529–0.7136 | True | 7,895,034 | `4ea2e1b88940f84872cc3e1d8c92207076c9941f631838332c310c487bd743c2` |

## f. Test split

Test split not read. The chosen rivals are frozen as of this commit.
