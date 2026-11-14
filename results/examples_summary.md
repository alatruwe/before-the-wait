# Examples summary: split, band positives, 1:2 matched negatives

Computed by `scripts/build_examples.py` from `traces/run/`, `results/run_traces.csv` and `results/run_onsets_paragraph.csv`.

## a. Data contact

| Date | What was touched | What was not |
|---|---|---|
| 2026-11-14 | Token ids, text and lexicon hits of the 292 kept traces; example positions selected (split, band positives, matched negatives) and each position's token kind. | No hidden states, no probabilities, no probe, no rival. |

## b. Split (seed 20261019)

| split | problems | onsets |
|---|---|---|
| train | 146 | 1251 |
| validation | 44 | 462 |
| test | 102 | 806 |
| all | 292 | 2519 |

## c. Per distance set × split

Clean onset: previous hit of any kind, or the reasoning start, more than the set's upper bound back (the e_b definition).

| set | split | onsets with ≥ 1 positive | clean onsets | positives | negatives |
|---|---|---|---|---|---|
| gate | train | 1251 | 1251 | 1251 | 2502 |
| gate | validation | 462 | 462 | 462 | 924 |
| gate | test | 806 | 806 | 806 | 1612 |
| 2-5 | train | 1244 | 1236 | 4955 | 9910 |
| 2-5 | validation | 457 | 454 | 1820 | 3640 |
| 2-5 | test | 802 | 794 | 3191 | 6382 |
| 6-10 | train | 1224 | 1191 | 6028 | 12056 |
| 6-10 | validation | 449 | 422 | 2164 | 4328 |
| 6-10 | test | 786 | 762 | 3861 | 7722 |
| 11-20 | train | 1186 | 1137 | 11623 | 23246 |
| 11-20 | validation | 420 | 409 | 4134 | 8268 |
| 11-20 | test | 761 | 726 | 7453 | 14906 |
| 21-40 | train | 1133 | 1021 | 21656 | 43312 |
| 21-40 | validation | 407 | 369 | 7776 | 15552 |
| 21-40 | test | 723 | 661 | 13907 | 27814 |

## d. Test onsets per band against m = 555

Clean test onsets (e_b definition) that keep at least one positive after any D18 drop.

| band | test onsets | ≥ 555? | 08b projection |
|---|---|---|---|
| 2-5 | 794 | yes | 869.4 |
| 6-10 | 762 | yes | 831.2 |
| 11-20 | 726 | yes | 795.2 |
| 21-40 | 661 | yes | 717.8 |

## e. Share of negatives from a different trace

| set | train | validation | test | all |
|---|---|---|---|---|
| gate | 516/2502 = 0.2062 | 222/924 = 0.2403 | 332/1612 = 0.2060 | 1070/5038 = 0.2124 |
| 2-5 | 101/9910 = 0.0102 | 61/3640 = 0.0168 | 28/6382 = 0.0044 | 190/19932 = 0.0095 |
| 6-10 | 62/12056 = 0.0051 | 41/4328 = 0.0095 | 68/7722 = 0.0088 | 171/24106 = 0.0071 |
| 11-20 | 489/23246 = 0.0210 | 382/8268 = 0.0462 | 641/14906 = 0.0430 | 1512/46420 = 0.0326 |
| 21-40 | 3380/43312 = 0.0780 | 1812/15552 = 0.1165 | 2254/27814 = 0.0810 | 7446/86678 = 0.0859 |

## f. Positives and negatives by token kind

| set | `newline` pos / neg | `period` pos / neg | `comma` pos / neg | `other_punct` pos / neg | `digit` pos / neg | `word` pos / neg | `other` pos / neg |
|---|---|---|---|---|---|---|---|
| gate | 1631 / 3262 | 479 / 958 | 51 / 102 | 63 / 126 | 0 / 0 | 295 / 590 | 0 / 0 |
| 2-5 | 240 / 480 | 241 / 482 | 190 / 380 | 349 / 698 | 2436 / 4872 | 5908 / 11816 | 602 / 1204 |
| 6-10 | 116 / 232 | 329 / 658 | 836 / 1672 | 746 / 1492 | 2198 / 4396 | 6842 / 13684 | 986 / 1972 |
| 11-20 | 435 / 870 | 901 / 1802 | 1564 / 3128 | 1255 / 2510 | 4589 / 9178 | 12622 / 25244 | 1844 / 3688 |
| 21-40 | 809 / 1618 | 1568 / 3136 | 2938 / 5876 | 2492 / 4984 | 8836 / 17672 | 23054 / 46108 | 3642 / 7284 |

## g. D18 drops

D18 did not fire.

## h. examples/examples.csv.gz

- sha256 `d05d940f4ac4396f236c5bd970b1268677fe3ba9ea8009b8835073bdb34668f1`
- 273,261 rows (one per example per distance set, header excluded)
