# c3 — gate input at distance 1

Paragraph-rule onsets over 25 non-looped traces (5 excluded for looping): **289**.
Gate-input id set after Part A: **12 distinct token ids** = [902, 1077, 3331, 3520, 3783, 6771, 10061, 13824, 14190, 47427, 80022, 88190]

## AUROC — model's own odds, onset-preceding vs matched negatives

| group | pairs | AUROC | 95% CI (bootstrap over traces) | median odds, onset | median odds, negative |
|---|---|---|---|---|---|
| gsm8k | 114 | **0.959** | [0.935, 0.990] | 0.5504 | 0.003232 |
| math500 | 175 | **0.965** | [0.945, 0.979] | 0.5979 | 0.000293 |
| pooled | 289 | **0.962** | [0.947, 0.977] | 0.5867 | 0.0005179 |

Share of onset-preceding positions where the argmax token is a marker-start id: **76.5%**

The pre-registration's provisional gate floor is AUROC 0.75 (§11). Reported against it; the floor is not changed here.

## Matching check — onsets and negatives drawn, per token kind

| dataset | prev_token_kind | onsets | negatives drawn | shortfall |
|---|---|---|---|---|
| gsm8k | `comma` | 4 | 4 | 0 |
| gsm8k | `newline` | 81 | 81 | 0 |
| gsm8k | `other_punct` | 2 | 2 | 0 |
| gsm8k | `period` | 17 | 17 | 0 |
| gsm8k | `word` | 10 | 10 | 0 |
| math500 | `comma` | 4 | 4 | 0 |
| math500 | `newline` | 115 | 115 | 0 |
| math500 | `other_punct` | 7 | 7 | 0 |
| math500 | `period` | 21 | 21 | 0 |
| math500 | `word` | 28 | 28 | 0 |

## Layer curves (median summed marker probability)

| layer | onset, gsm8k | neg, gsm8k | onset, math500 | neg, math500 |
|---|---|---|---|---|
| 1 | 4.55e-05 | 3.99e-05 | 5.17e-05 | 3.74e-05 |
| 2 | 4.76e-05 | 3.93e-05 | 4.93e-05 | 1.98e-05 |
| 3 | 0.000177 | 0.00013 | 0.000145 | 4.22e-05 |
| 4 | 0.000155 | 0.000111 | 0.000153 | 5e-05 |
| 5 | 0.000123 | 7.04e-05 | 0.000157 | 5.09e-05 |
| 6 | 0.000147 | 0.000115 | 0.000165 | 7.45e-05 |
| 7 | 0.000209 | 0.000139 | 0.000221 | 7.27e-05 |
| 8 | 0.000225 | 0.000134 | 0.000269 | 6.36e-05 |
| 9 | 0.000194 | 0.000133 | 0.000219 | 6.76e-05 |
| 10 | 0.000116 | 9.22e-05 | 0.000117 | 6.67e-05 |
| 11 | 9.98e-05 | 9.74e-05 | 0.000107 | 4.51e-05 |
| 12 | 0.000179 | 0.000185 | 0.000237 | 7.88e-05 |
| 13 | 0.000184 | 0.000192 | 0.00019 | 6.5e-05 |
| 14 | 0.00016 | 0.000163 | 0.000199 | 5.94e-05 |
| 15 | 0.000148 | 0.000113 | 0.000189 | 6.09e-05 |
| 16 | 0.000261 | 0.000176 | 0.00026 | 6.82e-05 |
| 17 | 0.000386 | 0.000176 | 0.000387 | 6.7e-05 |
| 18 | 0.000918 | 0.000373 | 0.00116 | 0.000117 |
| 19 | 0.000956 | 0.00027 | 0.0014 | 0.000104 |
| 20 | 0.000701 | 0.000166 | 0.00242 | 9.16e-05 |
| 21 | 0.000911 | 0.000163 | 0.00321 | 5.74e-05 |
| 22 | 0.00338 | 0.000177 | 0.00471 | 6.87e-05 |
| 23 | 0.0101 | 0.000203 | 0.0153 | 8.23e-05 |
| 24 | 0.0302 | 0.000164 | 0.0669 | 2.59e-05 |
| 25 | 0.0575 | 0.000145 | 0.157 | 2.03e-05 |
| 26 | 0.125 | 5.41e-05 | 0.262 | 2.31e-06 |
| 27 | 0.211 | 0.000225 | 0.394 | 1.28e-05 |
| 28 | 0.55 | 0.00323 | 0.598 | 0.000293 |

## Three sentences

**At which layer does the marker become predictable?** In the last third: taking "predictable" as the first layer where the median onset-preceding curve reaches an absolute 1%, that is layer 23 of 28 on GSM8K and layer 23 on MATH500, and the curve then climbs steeply — on GSM8K from 0.0034 at layer 22 to 0.55 at the final layer — while everything before roughly layer 19 sits at the 1e-4 noise floor.

**Does it differ by dataset?** No: layer 23 against layer 23 on the absolute criterion and layer 25 against 24 on "10% of the final value", so the layer at which the marker surfaces looks like a property of the model, not the dataset. (A ratio-to-negatives criterion would say layer 17 against layer 2, but that is an artifact: at those layers both curves are near 1e-4 and MATH500 simply has a lower negative baseline.)

**How separated are the negatives?** Pooled AUROC is 0.962 (95% CI 0.947-0.977), with median odds 0.587 at onset-preceding positions against 0.000518 at matched negatives — a 1133x gap in the medians.

## Part B — extraction seam test

```json
{
 "trace": "gsm8k_1054",
 "sequence_length": 716,
 "n_hidden_states": 29,
 "n_layers": 28,
 "embedding_index": 0,
 "hidden_state_shape": [
  1,
  716,
  1536
 ],
 "dtype": "torch.float16",
 "max_abs_diff_lm_head_only": 0.0,
 "max_abs_diff_norm_then_lm_head": 23.171875,
 "last_hidden_state_already_normed": true,
 "sampling_raw_median": 0.9981254935264587,
 "sampling_raw_p5": 0.3734600245952606,
 "sampling_tempered_median": 1.0,
 "sampling_tempered_p5": 0.3827747106552124,
 "smoke_positions": [
  100,
  150
 ],
 "smoke_shape": [
  29,
  50,
  1536
 ],
 "smoke_dtype": "torch.float16",
 "smoke_all_finite": true,
 "smoke_marker_hits_in_range": []
}
```
