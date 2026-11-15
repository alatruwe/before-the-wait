# Gate at distance 1 (validation)

Computed by `scripts/run_gate.py`. §11 Gate; probe per §7 and §18; G1–G4 from handoff 11a.

## a. Data contact

| Date | What was touched | What was not |
|---|---|---|
| 2026-11-14 | Gate examples (distance 1) of the train and validation splits: labels and saved activations (layers 1-28); the model's lm_head weights (no forward pass); the onset token ids of the 292 kept traces. | The test split (no row loaded, no activation opened); any distance > 1; any rival feature or entropy value. |

## b. Marker-start id set (G1)

Distinct generated token ids at the 2,519 paragraph-rule onsets of the 292 kept traces: **19 ids**.

| id | decoded |
|---|---|
| 902 | ` no` |
| 1077 | ` let` |
| 2308 | ` No` |
| 2753 | `No` |
| 3331 | ` hold` |
| 3520 | ` actually` |
| 3783 | ` wait` |
| 4354 | ` However` |
| 6771 | ` Let` |
| 8580 | ` waiting` |
| 10061 | `Let` |
| 11209 | `However` |
| 13824 | ` Wait` |
| 14190 | `Wait` |
| 47427 | `Hold` |
| 52592 | `Actually` |
| 80022 | `Hmm` |
| 87501 | ` hmm` |
| 88190 | ` Hmm` |

## c. Counts

| split | positives | negatives | problems |
|---|---|---|---|
| train | 1251 | 2502 | 146 |
| validation | 462 | 924 | 44 |

## d. The model's own odds (clause 1)

Score = summed probability over the id set, softmax(lm_head(layer-28 state)) in float32, temperature 1, no top-p; the layer-28 state already carries the final norm. Interval: 95%, 2,000 resamples of whole validation problems, seed 20261019.

- Model-odds AUROC (validation): **0.9462** (95% interval 0.9321–0.9598).
- Clause 1, AUROC ≥ 0.75: **pass**.

## e. Probe per layer (best C)

Logistic regression, L2, lbfgs, max_iter 10,000, z-scored with train statistics; fit on train, scored on validation by the decision function. All 135 fits are in `results/gate_layers.csv`; fits not converged: 0.

| layer | best C | validation AUROC |
|---|---|---|
| 1 | 0.01 | 0.8231 |
| 2 | 0.001 | 0.8505 |
| 3 | 0.001 | 0.8675 |
| 4 | 0.001 | 0.8812 |
| 5 | 0.001 | 0.8828 |
| 6 | 0.001 | 0.8853 |
| 7 | 0.001 | 0.8847 |
| 8 | 0.001 | 0.8946 |
| 9 | 0.001 | 0.9003 |
| 10 | 0.001 | 0.8978 |
| 11 | 0.001 | 0.8994 |
| 12 | 0.001 | 0.9060 |
| 13 | 0.001 | 0.9111 |
| 14 | 0.001 | 0.9159 |
| 15 | 0.001 | 0.9218 |
| 16 | 0.001 | 0.9271 |
| 17 | 0.001 | 0.9323 |
| 18 | 0.001 | 0.9381 |
| 19 | 0.001 | 0.9412 |
| 20 | 0.001 | 0.9462 |
| 21 | 0.01 | 0.9480 |
| 22 | 0.01 | 0.9535 |
| 23 | 0.01 | 0.9527 |
| 24 | 0.01 | 0.9533 |
| 25 | 0.01 | 0.9544 |
| 26 | 0.01 | 0.9570 |
| 27 | 0.01 | 0.9596 |

## f. Best layer (clauses 2 and 3)

- Best (layer, C) by G3: layer **27**, C = 0.01; validation AUROC **0.9596** (95% interval 0.9484–0.9692).
- Clause 2, probe ≥ model odds − 0.1 = 0.8462: **pass**.
- Clause 3, probe ≤ model odds + 0.03 = 0.9762: **pass**.

## g. Verdict

**GATE PASSES**
