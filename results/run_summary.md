# Run summary: 320 GSM8K traces, counts

Computed by `scripts/analyze_run.py` from `traces/run/` (`run_traces.csv`, `run_onsets_paragraph.csv`, `traces/run/progress.log`).

## Data contact (D15)

| Date | What was touched | What was not |
|---|---|---|
| 2026-11-13 | 320 GSM8K traces generated with the §4 settings from `prereg-freeze` 8ca3ebe; lexicon hits and onsets counted. | No hidden states, no probe, no rival, no position read at any distance. |

## Traces

| Quantity | Value |
|---|---|
| generated | 320 |
| looped (D10) | 28 (ℓ = 28/320 = 0.0875) |
| truncated (length cap, not looped; §4) | 20 |
| looped and capped | 21 |
| looped, not capped | 7 |
| kept (non-looped) | 292 |
| accuracy on kept traces (context only, D5) | 241/292 = 0.8253 |

## Onsets (D11, paragraph rule), kept traces

| Quantity | Value |
|---|---|
| hits checked by the hard check (all 320 traces) | 4583, 0 failures |
| hits, kept traces | 3344 |
| onsets | 2519 |
| o (onsets per kept trace) | 2519/292 = 8.6267 |
| "wait" share of onsets | 1796/2519 = 0.7130 |

## Per band (D16)

e_b: share of onsets whose previous lexicon hit of any kind, onset or merged
(or the reasoning start, for a first hit), lies more than the band's upper
bound tokens back (`results/count_calculations.md`). Projected test onsets =
0.35 × eligible onsets, against m = 555.

| band | e_b | eligible onsets | projected test onsets | ≥ 555? | flag | pilot e_b | pilot projection |
|---|---|---|---|---|---|---|---|
| 2-5 | 0.9861 (2484/2519) | 2484 | 869.4 | yes | no | 1.0000 | 818.5 |
| 6-10 | 0.9428 (2375/2519) | 2375 | 831.2 | yes | no | 0.9474 | 775.4 |
| 11-20 | 0.9019 (2272/2519) | 2272 | 795.2 | yes | no | 0.9123 | 746.7 |
| 21-40 | 0.8142 (2051/2519) | 2051 | 717.8 | yes | no | 0.8246 | 674.9 |

**Flag: no band flagged.**

## Secondary line (D16)

| Threshold | Run | Pilot | Flag |
|---|---|---|---|
| o < 7.0 | 8.6267 | 8.7692 | no |
| ℓ > 25% | 0.0875 | 0.1667 | no |

## Generation (traces/run/progress.log)

| Quantity | Value |
|---|---|
| traces logged | 320 |
| new tokens, total | 575,778 |
| generation seconds, sum over traces | 9,158.3 |
| wall time (`run.out`, `done; total`) | 9,319 s |
| finish_reason = eos | 279 |
| finish_reason = length | 41 |
| reasoning_complete = False | 39 |
| reasoning_complete = True | 281 |

