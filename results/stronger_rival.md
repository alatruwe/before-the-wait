# Stronger rival: the 7B reader for band 6-10

Computed by `scripts/extract_7b.py` (stage A) and `scripts/run_stronger_rival.py` (stage B). §8, D24; plan `handoffs/stronger-rival-plan.md` as approved. Band 6-10 only.

## a. Data contact

| Date | What was touched | What was not |
|---|---|---|
| 2026-11-21 | Stage A: the token ids of the 292 kept run traces (tokenizer check); the 7B's hidden states at every band 6-10 position, extracted without reading split or label (only the distance_set, trace_index and t columns of the examples file). Stage B: band 6-10 examples of the train and validation splits: labels and the 7B states of those traces. | The test split (no row loaded, no test-trace activation opened); every other band; the gate rows; the frozen 1.5B probe (hash-checked only). |

## b. Stage A: model and extraction

- Model `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` at revision `916b56a44061fd5cd7d6a8fb632557ed4f724f60`, float16 on MPS (stored as bfloat16), 28 layers, hidden size 3584. Input: `lens.build_input_ids` (the stored prompt and generated ids, EOS 151643 restored where stripped), never re-tokenized. Saved: `hidden_states` entries 1-28 (entry 28 post-norm; entry 0, the embeddings, not saved), float16, one stored zip per trace in `activations/run_7b/` (gitignored).
- Token-id compatibility (S4), tokenizer files only, before any weight was loaded: 292 traces, 5,187 distinct ids (prompt, output, EOS); ids decoding differently: 0; traces whose full decode differs: 0; EOS ids 151643 / 151643.
- Saved: 292 traces, 36,159 positions (equal to the band 6-10 set of the examples file), 7.26 GB. Per-trace sha256 in `results/stronger_rival_extraction_manifest.csv`.
- Checks: id at every saved position equals `output_token_ids[t]`; max |lm_head(entry 28) − logits| over all traces = 0.0; every hidden state and logit finite: True; fresh-process re-extraction of 3 traces byte-identical: True.

## c. Stage B: counts

| split | positives | negatives | problems |
|---|---|---|---|
| train | 6028 | 12056 | 146 |
| validation | 2164 | 4328 | 44 |

## d. Settings

The probe's recipe (§7, §18): per layer, z-score per dimension with the train mean and SD; logistic regression, L2, lbfgs, max_iter 10,000, intercept, no class weights; C ∈ {0.0001, 0.001, 0.01, 0.1, 1}; fitted on train, scored on validation by the decision function, single-threaded BLAS (`run_probes.fit_layer`). Chosen: highest validation AUROC over the 28 layers × 5 C; ties to the lowest layer, then the smallest C (`run_gate.best_fit`). 140 fits in `results/stronger_rival_layers.csv`; not converged: 0 (eligible, flagged). scikit-learn 1.9.1, numpy 2.5.4.

## e. Per layer: best C and its validation AUROC

| layer | C | validation AUROC | converged |
|---|---|---|---|
| 1 | 0.001 | 0.6964 | True |
| 2 | 0.001 | 0.7567 | True |
| 3 | 0.001 | 0.7663 | True |
| 4 | 0.001 | 0.7765 | True |
| 5 | 0.001 | 0.7911 | True |
| 6 | 0.001 | 0.7929 | True |
| 7 | 0.0001 | 0.8056 | True |
| 8 | 0.0001 | 0.8087 | True |
| 9 | 0.001 | 0.8093 | True |
| 10 | 0.0001 | 0.8099 | True |
| 11 | 0.0001 | 0.8210 | True |
| 12 | 0.0001 | 0.8179 | True |
| 13 | 0.0001 | 0.8216 | True |
| 14 | 0.0001 | 0.8233 | True |
| 15 | 0.0001 | 0.8260 | True |
| 16 | 0.0001 | 0.8323 | True |
| 17 | 0.0001 | 0.8375 | True |
| 18 | 0.0001 | 0.8441 | True |
| 19 | 0.0001 | 0.8565 | True |
| 20 | 0.0001 | 0.8518 | True |
| 21 | 0.0001 | 0.8505 | True |
| 22 | 0.0001 | 0.8479 | True |
| 23 | 0.0001 | 0.8428 | True |
| 24 | 0.0001 | 0.8399 | True |
| 25 | 0.0001 | 0.8359 | True |
| 26 | 0.0001 | 0.8350 | True |
| 27 | 0.0001 | 0.8305 | True |
| 28 | 0.001 | 0.8219 | True |

## f. Chosen 7B reader

Fitted on train only; saved, hashed, reloaded from its file and re-scored on validation (equal as a float). Interval: 95%, 2,000 resamples of whole validation problems, seed 20261019 (as in `results/probes.md`).

| band | layer | C | C at a grid edge | validation AUROC | 95% interval | file bytes | sha256 |
|---|---|---|---|---|---|---|---|
| 6-10 | 19 | 0.0001 | yes | 0.8565 | 0.8347–0.8774 | 87,384 | `a2ac7a37eb54f731fcb20635d9f400eb483129b24258da19a8befd849783e850` |

Selection result: the best of 140 settings on validation. No test number exists yet; the frozen 1.5B probe was not read in this stage.
## g. Data contact: test

| Date | What was touched | What was not |
|---|---|---|
| 2026-11-21 | The band 6-10 test examples: labels, the 1.5B activations at the frozen probe's layer and the 7B activations at the chosen reader's layer. Second scoring of these examples for the probe (§8, D24), first for the 7B reader. | Every other band; the gate rows; every other layer. |

## h. Rehearsal on validation

Before any test row was read: probe validation AUROC 0.8527 (table 0.8527, CSV 0.852655); 7B reader 0.8565 (CSV 0.856463, exactly equal to stage B). Reproduced: yes. On test, the probe's AUROC reproduced `results/verdicts.csv` exactly (`0.8683752768186852`, S8).

## i. Verdict at δ = 0.05 (claim level 2)

Lift = frozen 1.5B probe AUROC − 7B reader AUROC on the band 6-10 test examples. Interval: 98.75%, paired bootstrap over the 102 test problems, 10,000 resamples, `random.Random(20261019)`, the same draws as step 13 (D22).

| band | probe AUROC | 7B reader AUROC | lift | δ | 98.75% interval | verdict | claim level 2 |
|---|---|---|---|---|---|---|---|
| 6-10 | 0.8684 | 0.8706 | -0.0022 | 0.05 | -0.0100 to +0.0062 | null | does not hold |

## j. Secondary verdicts, counts and frozen files

- Secondary (never the headline): δ = 0.03 null; δ = 0.08 null.
- Counts: onsets 762, positives 3861, negatives 7722, problems with examples 102 of 102; dropped resamples 0.
- Frozen probe sha256 before / after: `bf9bd8708a4abcd6ff9a0643288cd86341d70dab9955b21cd9950b9a16133dfa` / `bf9bd8708a4abcd6ff9a0643288cd86341d70dab9955b21cd9950b9a16133dfa`.
- Frozen 7B reader sha256 before / after: `a2ac7a37eb54f731fcb20635d9f400eb483129b24258da19a8befd849783e850` / `a2ac7a37eb54f731fcb20635d9f400eb483129b24258da19a8befd849783e850`.
