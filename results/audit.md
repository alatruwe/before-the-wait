# Hand audit: precision of the lexicon onsets (§12)

Computed by `scripts/score_audit.py` from `audit/labels.csv` and `audit/draw.json`.

## Data contact

| Date | What was touched | What was not |
|---|---|---|
| 2026-11-21 | `audit/labels.csv` (Adeline's hand labels of 100 test-split onsets) and `audit/draw.json`. | Any probe, rival, activation or score; the examples; any other split. |

## Result

- sha256 of `audit/labels.csv`: `2b85139195452181de38703fda08101d3988877d31b0e45deb38566c5dd604ae`
- Real: 13; false: 87
- Precision: 0.1300
- 95% interval: 0.0543–0.2165 (percentile bootstrap over whole problems, 10,000 resamples, `numpy.random.default_rng(20261020)`)
- Distinct problems: 53
- Flagged `unsure`: 0
- Flagged `mishit`: 0
- Add-on (§12) runnable: no (needs at least 20 real and 20 false; 13 real, 87 false)
