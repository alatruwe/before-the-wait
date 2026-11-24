"""Handoff viz-wave-data: export the per-event test scores behind exploratory analysis 1
(pre-registration §13, D25 X2) as data for an illustration. EXPLORATORY; no verdict.

Computes no new statistic. Refits analysis 1's 135 per-layer probes on train with
run_exploratory.a1_set (same code, same validation-best C), scores the test rows, and
exports one grid per test onset (event) x layer 1-27 x distance 1-40. A wiring check
recomputes every (set, layer, d) AUROC from the exported per-example scores and requires
it to equal results/exploratory/a1_exact_distance.csv to six decimals.

    .venv/bin/python scripts/export_wave.py

Writes activations/viz/wave_events.npz (gitignored), viz/wave_sample.json.gz (gzipped
JSON, 200 events, 2 decimals; ruling option a) and viz/wave_mean.csv. Nothing under results/ is written. Frozen hashes checked before/after.
"""
from __future__ import annotations

import csv
import gzip
import json
import math
import random
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import numpy as np  # noqa: E402

import extract_activations as ea  # noqa: E402
import run_exploratory as rx  # noqa: E402
import run_gate as rg  # noqa: E402
import run_verdicts as rv  # noqa: E402

VIZ = REPO_ROOT / "viz"
NPZ = REPO_ROOT / "activations" / "viz" / "wave_events.npz"
A1_CSV = rx.OUT / "a1_exact_distance.csv"
SETS = ["gate"] + list(rx.BANDS)
LAYERS = list(range(1, 28))
DISTS = list(range(1, 41))
SEED = 20261122
N_SAMPLE = 200
SEAMS = [5.5, 10.5, 20.5]
MAX_BYTES = 1_000_000
NOTE = ("Exploratory (pre-registration §13, D25). No verdict is applied to any number here. "
        "Data for an illustration, not a result. Test problems only. Each cell is the "
        "decision-function score of that layer's probe (gate probe for d = 1, the band's "
        "probe otherwise) at the positive position d tokens before a 'wait' onset, standardized by "
        "the mean and SD (ddof 0) of that probe's test negatives; neg_z is the mean "
        "standardized score of the negatives matched to that positive position. Seams at d 5|6, "
        "10|11, 20|21 are probe handovers. A missing cell (JSON empty value) means no positive example at that distance. "
        "Registered results: results/.")


def set_of(d: int) -> str:
    if d == 1:
        return "gate"
    return next(b for b, (lo, hi) in rx.BAND_RANGE.items() if lo <= d <= hi)


def build_grids(events, per_set):
    """per_set[s] = (rows, d, scores[L-1] arrays over rows). Returns raw and z grids
    [E, 27, 40], n [27, 40] and neg_mean/neg_sd [5, 27] (SETS order).

    A positive (trace, onset - d) fills cell (event(trace, onset), L, d); its matched
    negatives (pos_trace_index, pos_t) are averaged into the same cell of the baseline grid.
    Standardization uses each (set, layer) probe's negatives only."""
    E = len(events)
    idx = {e: i for i, e in enumerate(events)}
    shape = (E, len(LAYERS), len(DISTS))
    pos_raw = np.full(shape, np.nan)
    neg_sum = np.zeros(shape)
    neg_cnt = np.zeros(shape, dtype=np.int64)
    pos_z = np.full(shape, np.nan)
    neg_zsum = np.zeros(shape)
    neg_mean = np.full((len(SETS), len(LAYERS)), np.nan)
    neg_sd = np.full((len(SETS), len(LAYERS)), np.nan)
    for si, s in enumerate(SETS):
        if s not in per_set:
            continue
        rows, d, scores = per_set[s]
        lab = np.array([r["label"] for r in rows])
        pos_cell = {}
        for j, r in enumerate(rows):
            if r["label"] == 1:
                key = (r["trace_index"], r["t"])
                assert key not in pos_cell, f"duplicate positive {key}"
                pos_cell[key] = (idx[(r["trace_index"], r["onset"])], d[j] - 1)
        cell = np.array([pos_cell[(r["trace_index"], r["t"])] if r["label"] == 1
                         else pos_cell[(r["pos_trace_index"], r["pos_t"])] for r in rows])
        assert all(set_of(int(k)) == s for k in d)
        for L in LAYERS:
            sc = np.asarray(scores[L - 1], dtype=np.float64)
            mu, sd = float(sc[lab == 0].mean()), float(sc[lab == 0].std())
            neg_mean[si, L - 1], neg_sd[si, L - 1] = mu, sd
            z = (sc - mu) / sd
            p, n = lab == 1, lab == 0
            pos_raw[cell[p, 0], L - 1, cell[p, 1]] = sc[p]
            pos_z[cell[p, 0], L - 1, cell[p, 1]] = z[p]
            np.add.at(neg_sum, (cell[n, 0], L - 1, cell[n, 1]), sc[n])
            np.add.at(neg_zsum, (cell[n, 0], L - 1, cell[n, 1]), z[n])
            np.add.at(neg_cnt, (cell[n, 0], L - 1, cell[n, 1]), 1)
    has = ~np.isnan(pos_raw)
    assert np.all(neg_cnt[has] > 0), "positive without matched negatives"
    assert np.all(neg_cnt[~has] == 0), "negative without its positive"
    with np.errstate(invalid="ignore", divide="ignore"):
        neg_raw = np.where(has, neg_sum / neg_cnt, np.nan)
        neg_z = np.where(has, neg_zsum / neg_cnt, np.nan)
    n = has.sum(axis=0)
    return pos_raw, neg_raw, pos_z, neg_z, n, neg_mean, neg_sd


def draw_sample(E: int) -> list[int]:
    return random.Random(SEED).sample(range(E), min(N_SAMPLE, E))


def nan_to_none(a: np.ndarray, decimals: int):
    return [[None if math.isnan(v) else round(float(v), decimals) for v in row] for row in a]


def sample_json(events, pos_z, neg_z, n, draw, decimals: int) -> str:
    obj = {"note": NOTE, "layers": LAYERS, "distances": DISTS, "seams": SEAMS,
           "seed": SEED, "decimals": decimals, "n_events_total": len(events),
           "n": n.tolist(),
           "events": [{"draw_order": k, "trace_index": int(events[i][0]),
                       "onset": int(events[i][1]),
                       "pos_z": nan_to_none(pos_z[i], decimals),
                       "neg_z": nan_to_none(neg_z[i], decimals)}
                      for k, i in enumerate(draw)]}
    return json.dumps(obj, separators=(",", ":"), allow_nan=False) + "\n"


def wiring_check(per_set) -> int:
    with open(A1_CSV, newline="") as fh:
        ref = {(r["set"], int(r["layer"]), int(r["d"])): r["probe_auroc"]
               for r in csv.DictReader(fh)}
    assert len(ref) == len(LAYERS) * len(DISTS)
    bad = []
    for s, (rows, d, scores) in per_set.items():
        y = rv.labels(rows)
        for L in LAYERS:
            for k in sorted(set(d.tolist())):
                m = d == k
                got = f"{rg.auroc(y[m], np.asarray(scores[L - 1])[m]):.6f}"
                if got != ref[(s, L, k)]:
                    bad.append((s, L, k, got, ref[(s, L, k)]))
    assert not bad, f"wiring check failed in {len(bad)} cells: {bad[:5]}"
    return len(ref)


def main() -> None:
    t0 = time.perf_counter()
    rx.frozen_check("viz-wave", "before")
    manifest = rg.load_manifest()
    verified: set[int] = set()
    all_rows = rx.load_rows(rx.SPLITS, SETS)
    per_set = {}
    for s in SETS:
        best = rx.best_c_per_layer(rx.GATE_CSV if s == "gate" else rx.PROBE_CSV,
                                   None if s == "gate" else s)
        by, _, _, _, te_rows = rx.a1_set(s, "gate" if s == "gate" else "band",
                                         all_rows[s], best, manifest, verified)
        assert all(r["split"] == "test" for r in te_rows)
        per_set[s] = (te_rows, rx.exact_distance(te_rows), [by[L]["s_te"] for L in LAYERS])
    del all_rows
    events = sorted({(r["trace_index"], r["onset"]) for rows, _, _ in per_set.values()
                     for r in rows if r["label"] == 1})
    grids = build_grids(events, per_set)
    pos_raw, neg_raw, pos_z, neg_z, n, neg_mean, neg_sd = grids
    # per-example scores saved alongside, so the wiring check reads the exported data
    ex = {}
    for s, (rows, d, scores) in per_set.items():
        ex[f"{s}_label"] = rv.labels(rows)
        ex[f"{s}_d"] = d
        ex[f"{s}_scores"] = np.stack([np.asarray(x, dtype=np.float64) for x in scores])
    NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(NPZ, event=np.array(events, dtype=np.int64),
                        pos_raw=pos_raw.astype(np.float32), neg_raw=neg_raw.astype(np.float32),
                        pos_z=pos_z.astype(np.float32), neg_z=neg_z.astype(np.float32),
                        n=n, neg_mean=neg_mean, neg_sd=neg_sd, sets=np.array(SETS), **ex)
    z = np.load(NPZ)
    reload = {s: (per_set[s][0], z[f"{s}_d"], list(z[f"{s}_scores"])) for s in SETS}
    n_cells = wiring_check(reload)
    print(f"wiring check: {n_cells} cells match a1_exact_distance.csv", flush=True)

    VIZ.mkdir(exist_ok=True)
    draw = draw_sample(len(events))
    pz, nz = z["pos_z"].astype(np.float64), z["neg_z"].astype(np.float64)
    # Ruling (option a): the specced 200 events at 2 decimals, gzipped JSON. mtime=0 so the
    # bytes are reproducible.
    settings = (2, N_SAMPLE)
    text = sample_json(events, pz, nz, n, draw, settings[0])
    gz = gzip.compress(text.encode(), compresslevel=9, mtime=0)
    assert len(gz) < MAX_BYTES, f"gzipped sample {len(gz)} bytes"
    ea.write_atomic(VIZ / "wave_sample.json.gz", gz)
    with np.errstate(invalid="ignore"), open(VIZ / "wave_mean.csv", "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["layer", "d", "set", "mean_pos_z", "mean_neg_z", "n"])
        for li, L in enumerate(LAYERS):
            for di, k in enumerate(DISTS):
                w.writerow([L, k, set_of(k), f"{np.nanmean(pz[:, li, di]):.6f}",
                            f"{np.nanmean(nz[:, li, di]):.6f}", int(n[li, di])])
    rx.frozen_check("viz-wave", "after")
    rec = {"E": len(events), "cells_per_set": {s: int(n[:, [k - 1 for k in DISTS
                                                             if set_of(k) == s]].sum() // 27)
                                               for s in SETS},
           "wiring_cells": n_cells, "sample_decimals": settings[0], "sample_events": settings[1],
           "sample_json_bytes": len(text.encode()), "sample_gz_bytes": len(gz), "seconds": round(time.perf_counter() - t0, 1)}
    print(json.dumps(rec), flush=True)
    ea.write_atomic(NPZ.parent / "export_record.json", (json.dumps(rec, indent=1) + "\n").encode())


if __name__ == "__main__":
    main()
