# viz/

**Exploratory (pre-registration §13, D25). No verdict is applied to any number here.**

This folder holds data for an illustration. It is not a result and supports no claim. The registered results are in [`results/`](../results/). The analysis these scores come from is exploratory analysis 1 ([`results/exploratory/summary.md`](../results/exploratory/summary.md)).

All scores come from **test problems only**: 806 marker onsets (events) in the 102 test traces.

## What a cell is

A cell is one event, one layer (1–27) and one distance d (1–40 tokens before the onset).

- **`pos`:** the decision-function score of that layer's probe at the position d tokens before the onset.
  - At d = 1 the probe is the gate probe; at other distances it is the band's probe (2–5, 6–10, 11–20, 21–40).
  - These are analysis 1's per-layer probes: refitted on train at each layer's validation-best C, with no new choice.
- **`neg`:** the mean score, from the same probe, of the negatives matched to that positive position. These are positions with no marker ahead, drawn by the registered matching.

## Standardization

The probes are not on one scale, so each one is standardized separately. For each probe (one set and one layer), take the mean and standard deviation (population SD, ddof 0) of its scores on **its own test negatives only**. Both `pos` and `neg` are then expressed as (score − mean) / SD.

A height therefore reads as "SDs above matched no-marker positions". By construction, the average `neg_z` across all of a probe's negatives is 0.

## Seams and missing cells

- The band probes are separate, so the surface has seams at d 5→6, 10→11 and 20→21, plus 1→2, where the gate probe hands over. They are left as they are: nothing is smoothed, interpolated or filled.
- A cell is missing where the onset has no positive example at that distance. Missing cells are stored as the JSON empty value.
- Far distances have fewer events: `n` falls from 806 at d = 1 to 660 at d = 40.

## Files

| File | Content |
|---|---|
| `wave_sample.json.gz` | **Gzipped JSON** (read with `gzip.open` / `zcat`). It holds 200 events drawn with `random.Random(20261122).sample(range(806), 200)`, in draw order. Each event has `trace_index`, `onset`, and `pos_z` and `neg_z` as [27 layers][40 distances], rounded to 2 decimals, with the JSON empty value where missing. It also has `n` for all 806 events, the layer and distance axes, `seams` and a `note`. |
| `wave_mean.csv` | For each layer and d: mean `pos_z`, mean `neg_z` and `n`, over all 806 events. |

## How the data was made

The data was made by `scripts/export_wave.py`. As a check, the AUROC of every (set, layer, d) cell was recomputed from the exported per-example scores. All 1,080 cells equal `results/exploratory/a1_exact_distance.csv` to six decimals.
