# Extraction summary: layers 1-28 at the example positions

Computed by `scripts/extract_activations.py` from `traces/run/` and the `trace_index`, `t` columns of `examples/examples.csv.gz` (sha256 `d05d940f4ac4396f236c5bd970b1268677fe3ba9ea8009b8835073bdb34668f1`). Arrays are in `activations/run/` (not in git); `results/extraction_manifest.csv` pins each file by sha256.

## Data contact

| Date | What was touched | What was not |
|---|---|---|
| 2026-11-14 | Teacher-forced pass over the token ids of the 292 kept traces; hidden states (layers 1-28) saved at the 206,989 unique example positions, all three splits; next-token logits used for per-token entropy and for the label-blind wiring and replay checks. | No label, split or band read; no probe, no rival, no score; no statistic comparing positives with negatives or one split with another. |

## Settings and indexing

- Model, revision and dtype from `generate.load_model_and_tokenizer` (fp16, MPS, offline).
- Input = `lens.build_input_ids(trace, eos_id)`: prompt ids + generated ids, plus the EOS that generation stripped. Never re-tokenized.
- Example t is input index `P + t` (P = prompt length); the id there was asserted equal to `output_token_ids[t]` at every saved position.
- `hidden_states` has 29 entries; entry 0 (embeddings) is not saved; entries 1-28 are layers 1-28. Entry 28 already carries the final norm and is saved as is.
- Per trace, `gsm8k_XXXX.npz` (stored zip, fixed timestamps): `hidden` float16 [n_pos, 28, 1536]; `t` int32 ascending; `entropy` float32 [n generated], entry i from the logits at P + i; `prompt_entropy` float32 [P], entry i from the logits at input i (added on approval). Entropy: softmax at temperature 1, no top-p, float32, nats (E1).

## a. Wiring (every trace)

- max |lm_head(entry 28) − logits| over all positions: max over traces = 0.0; traces at exactly 0.0: 292 / 292.
- Seam: max |lm_head(norm(entry 28)) − logits| (first 512 positions) is nonzero on 292 / 292 traces (min 13.090, max 22.484); the norm is not applied twice.

## b. Replay (every trace)

Probability of each generated token (and the restored EOS) under T 0.6, top-p 0.95 at the preceding position, through transformers' `TemperatureLogitsWarper` and `TopPLogitsWarper` (the processors generation applied; no top-k), on CPU in float32.

The first run stopped on trace 1 (gsm8k_0000, share 0.9596), because `lens.sampling_probability` drops the token whose cumulative mass crosses top-p, so tokens generation could sample scored 0; under transformers' own warpers that trace replays at 1.0, and check b now uses those warpers, with `src/` unchanged and the 0.99 floor as written. c3 Part B's informational tempered numbers (`sampling_tempered_median`, `sampling_tempered_p5`) came from the faulty function; the c3 gate used raw probabilities and is unaffected.

| quantity | value |
|---|---|
| tokens checked | 467,340 |
| tokens with p > 0 | 467,312 (0.999940) |
| tokens with p = 0 | 28 |
| traces at 100% | 266 / 292 |
| lowest per-trace share | 0.998654 (floor 0.99) |
| traces below 0.99 | 0 |
| pooled median p (bin width 1e-4, lower edge) | 0.9999 |
| median of per-trace medians | 1.000000 |

## c. Coverage

- Trace files: 292 (expected 292).
- Saved positions: 206,989 (expected 206,989).
- Saved (trace_index, t) set equals the set from examples.csv.gz: True.

## d. Values

- All finite (hidden, entropy, prompt_entropy): True.
- Dtypes and shapes as specified, t strictly ascending: True.
- Per-layer mean |value|, pooled over all 206,989 saved positions and 1,536 dimensions (recomputed from the files; equals the in-run accumulation: True):

| layer | mean abs |
|---|---|
| 1 | 0.5010 |
| 2 | 0.6153 |
| 3 | 0.7535 |
| 4 | 0.8609 |
| 5 | 0.9635 |
| 6 | 0.9454 |
| 7 | 1.0149 |
| 8 | 1.0885 |
| 9 | 1.1034 |
| 10 | 1.1651 |
| 11 | 1.2139 |
| 12 | 1.2785 |
| 13 | 1.3531 |
| 14 | 1.4546 |
| 15 | 1.5209 |
| 16 | 1.5989 |
| 17 | 1.7609 |
| 18 | 2.0088 |
| 19 | 2.3012 |
| 20 | 2.6751 |
| 21 | 3.1405 |
| 22 | 3.8171 |
| 23 | 4.5072 |
| 24 | 5.1567 |
| 25 | 6.0744 |
| 26 | 6.7884 |
| 27 | 7.9013 |
| 28 | 1.5082 |

## e. Determinism (fresh process)

| trace | positions | hidden | entropy | prompt_entropy | t | file bytes identical |
|---|---|---|---|---|---|---|
| gsm8k_0000 | 311 | 0.0 | 0.0 | 0.0 | 0.0 | True |
| gsm8k_0653 | 460 | 0.0 | 0.0 | 0.0 | 0.0 | True |
| gsm8k_0606 | 2590 | 0.0 | 0.0 | 0.0 | 0.0 | True |

Traces: smallest kept index, median kept index, most positions (from metadata only).

## f. Disk

- `activations/run/`: 17.81 GB (17,809,034,319 bytes).
- Free after: 321.3 GiB.
- Extraction time (sum over traces): 34.9 min.
