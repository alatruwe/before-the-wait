"""Step D counts: the run's 320 GSM8K traces -> results/run_*.csv and run_summary.md.

    .venv/bin/python scripts/analyze_run.py

Counting only. Reads the saved traces and the tokenizer (offline). Never loads
the model, never touches hidden states, draws no band positives or negatives
and makes no split. Labeling, looping and onset logic are reused unchanged
from src/ (label.py, summarize.analyze_trace, counts.gap_to_previous_hit).

If any marker fails the hard check the script writes nothing and exits 1.
Deterministic: rerunning produces byte-identical outputs.
"""
from __future__ import annotations

import csv
import json
import os
import re
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ["HF_HOME"] = str(REPO_ROOT / ".hf_cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, str(REPO_ROOT / "src"))

from before_the_wait.counts import BANDS, gap_to_previous_hit, minimum_events  # noqa: E402

TRACE_DIR = REPO_ROOT / "traces" / "run"
RESULTS = REPO_ROOT / "results"
PROGRESS_LOG = TRACE_DIR / "progress.log"
RUN_OUT = TRACE_DIR / "run.out"

N_EXPECTED = 320
FREEZE_COMMIT = "8ca3ebe"
CONTACT_DATE = "2026-11-13"
TEST_SHARE = 0.35  # §4 split
DELTA = 0.05  # §11 / D14
M = 555  # §11, results/count_calculations.md
O_THRESHOLD = 7.0  # D16 secondary line
LOOP_THRESHOLD = 0.25  # D16 secondary line

# Pilot values from results/count_calculations.md, for comparison (handoff 08b).
PILOT = {
    "loop_share": 0.1667,
    "o": 8.7692,
    "e_b": {"2-5": 1.0000, "6-10": 0.9474, "11-20": 0.9123, "21-40": 0.8246},
    "projected": {"2-5": 818.5, "6-10": 775.4, "11-20": 746.7, "21-40": 674.9},
}


# ---------------------------------------------------------------- pure helpers

def _truthy(v: Any) -> bool:
    """CSV round-trips booleans as 'True'/'False'."""
    return v is True or v == "True"


def trace_tally(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Trace-level counts (§4 exclusions, D10, D5) from run_traces.csv rows.

    truncated = hit the length cap and not looped (§4); looped-and-capped is
    counted separately.
    """
    n = len(rows)
    looped = [r for r in rows if _truthy(r["looped"])]
    kept = [r for r in rows if not _truthy(r["looped"])]
    capped = lambda r: r["finish_reason"] == "length"  # noqa: E731
    correct_kept = sum(1 for r in kept if _truthy(r["correct"]))
    return {
        "n": n,
        "looped": len(looped),
        "loop_share": len(looped) / n if n else 0.0,
        "truncated": sum(1 for r in kept if capped(r)),
        "looped_capped": sum(1 for r in looped if capped(r)),
        "looped_not_capped": sum(1 for r in looped if not capped(r)),
        "kept": len(kept),
        "correct_kept": correct_kept,
        "accuracy_kept": correct_kept / len(kept) if kept else 0.0,
    }


def onset_tally(onset_rows: list[dict[str, Any]], n_kept: int) -> dict[str, Any]:
    """D11 counts from run_onsets_paragraph.csv rows (kept traces only)."""
    hits = len(onset_rows)
    onsets = [r for r in onset_rows if r["merged_into"] in ("", None)]
    wait = sum(1 for r in onsets if r["marker"] == "wait")
    return {
        "hits": hits,
        "onsets": len(onsets),
        "o": len(onsets) / n_kept if n_kept else 0.0,
        "wait_onsets": wait,
        "wait_share": wait / len(onsets) if onsets else 0.0,
    }


def gap_from_onset_row(row: dict[str, Any], reasoning_start: int) -> int:
    """e_b gap read off a CSV row: tokens_since_prev_hit is the distance to the
    previous hit of any kind (onset or merged); blank for a trace's first hit,
    where the reasoning start stands in."""
    v = row["tokens_since_prev_hit"]
    if v in ("", None):
        return int(row["token_index"]) - reasoning_start
    return int(v)


def band_table(gaps: list[int], test_share: float = TEST_SHARE,
               m: int = M) -> list[dict[str, Any]]:
    """Per band: e_b, eligible onsets (gap > upper bound), projected test onsets
    = test_share x eligible, and the D16 flag (projected < m)."""
    out = []
    for name, _lo, hi in BANDS:
        eligible = sum(1 for g in gaps if g > hi)
        projected = test_share * eligible
        out.append({
            "band": name,
            "upper": hi,
            "eligible": eligible,
            "total": len(gaps),
            "e_b": eligible / len(gaps) if gaps else 0.0,
            "projected": projected,
            "flag": projected < m,
        })
    return out


def progress_facts(progress_text: str, run_out_text: str) -> dict[str, Any]:
    """Generation facts from progress.log (tab-separated, header line) and the
    wall time from run.out's closing 'done; total Ns' line."""
    lines = progress_text.splitlines()
    header = lines[0].split("\t")
    rows = [dict(zip(header, ln.split("\t"))) for ln in lines[1:] if ln.strip()]
    wall = re.search(r"done; total (\d+)s", run_out_text)
    return {
        "rows": len(rows),
        "tokens": sum(int(r["new_tokens"]) for r in rows),
        "gen_seconds": sum(float(r["seconds"]) for r in rows),
        "finish": Counter(r["finish_reason"] for r in rows),
        "reasoning_complete": Counter(r["reasoning_complete"] for r in rows),
        "wall_seconds": int(wall.group(1)) if wall else None,
    }


# ------------------------------------------------------------------- summary

def build_run_summary(tt: dict, ot: dict, bands: list[dict], pf: dict,
                      hard_checked: int) -> str:
    L: list[str] = []
    w = L.append
    w("# Run summary: 320 GSM8K traces, counts")
    w("")
    w(f"Computed by `scripts/analyze_run.py` from `traces/run/` "
      f"(`run_traces.csv`, `run_onsets_paragraph.csv`, `traces/run/progress.log`).")
    w("")
    w("## Data contact (D15)")
    w("")
    w("| Date | What was touched | What was not |")
    w("|---|---|---|")
    w(f"| {CONTACT_DATE} | {tt['n']} GSM8K traces generated with the §4 settings from "
      f"`prereg-freeze` {FREEZE_COMMIT}; lexicon hits and onsets counted. | "
      f"No hidden states, no probe, no rival, no position read at any distance. |")
    w("")

    w("## Traces")
    w("")
    w("| Quantity | Value |")
    w("|---|---|")
    w(f"| generated | {tt['n']} |")
    w(f"| looped (D10) | {tt['looped']} (ℓ = {tt['looped']}/{tt['n']} = {tt['loop_share']:.4f}) |")
    w(f"| truncated (length cap, not looped; §4) | {tt['truncated']} |")
    w(f"| looped and capped | {tt['looped_capped']} |")
    w(f"| looped, not capped | {tt['looped_not_capped']} |")
    w(f"| kept (non-looped) | {tt['kept']} |")
    w(f"| accuracy on kept traces (context only, D5) | {tt['correct_kept']}/{tt['kept']} "
      f"= {tt['accuracy_kept']:.4f} |")
    w("")

    w("## Onsets (D11, paragraph rule), kept traces")
    w("")
    w("| Quantity | Value |")
    w("|---|---|")
    w(f"| hits checked by the hard check (all {tt['n']} traces) | {hard_checked}, 0 failures |")
    w(f"| hits, kept traces | {ot['hits']} |")
    w(f"| onsets | {ot['onsets']} |")
    w(f"| o (onsets per kept trace) | {ot['onsets']}/{tt['kept']} = {ot['o']:.4f} |")
    w(f"| \"wait\" share of onsets | {ot['wait_onsets']}/{ot['onsets']} = {ot['wait_share']:.4f} |")
    w("")

    w("## Per band (D16)")
    w("")
    w("e_b: share of onsets whose previous lexicon hit of any kind, onset or merged")
    w("(or the reasoning start, for a first hit), lies more than the band's upper")
    w(f"bound tokens back (`results/count_calculations.md`). Projected test onsets =")
    w(f"{TEST_SHARE} × eligible onsets, against m = {M}.")
    w("")
    w("| band | e_b | eligible onsets | projected test onsets | ≥ 555? | flag | "
      "pilot e_b | pilot projection |")
    w("|---|---|---|---|---|---|---|---|")
    for b in bands:
        w(f"| {b['band']} | {b['e_b']:.4f} ({b['eligible']}/{b['total']}) | {b['eligible']} | "
          f"{b['projected']:.1f} | {'no' if b['flag'] else 'yes'} | "
          f"{'**yes**' if b['flag'] else 'no'} | {PILOT['e_b'][b['band']]:.4f} | "
          f"{PILOT['projected'][b['band']]:.1f} |")
    w("")
    flagged = [b["band"] for b in bands if b["flag"]]
    w(f"**Flag: {'yes, ' + ', '.join(flagged) if flagged else 'no band flagged'}.**")
    w("")

    w("## Secondary line (D16)")
    w("")
    w("| Threshold | Run | Pilot | Flag |")
    w("|---|---|---|---|")
    w(f"| o < {O_THRESHOLD} | {ot['o']:.4f} | {PILOT['o']:.4f} | "
      f"{'**yes**' if ot['o'] < O_THRESHOLD else 'no'} |")
    w(f"| ℓ > {LOOP_THRESHOLD:.0%} | {tt['loop_share']:.4f} | {PILOT['loop_share']:.4f} | "
      f"{'**yes**' if tt['loop_share'] > LOOP_THRESHOLD else 'no'} |")
    w("")

    w("## Generation (traces/run/progress.log)")
    w("")
    w("| Quantity | Value |")
    w("|---|---|")
    w(f"| traces logged | {pf['rows']} |")
    w(f"| new tokens, total | {pf['tokens']:,} |")
    w(f"| generation seconds, sum over traces | {pf['gen_seconds']:,.1f} |")
    wall = pf["wall_seconds"]
    w(f"| wall time (`run.out`, `done; total`) | "
      f"{f'{wall:,} s' if wall is not None else 'not found'} |")
    for k in sorted(pf["finish"]):
        w(f"| finish_reason = {k} | {pf['finish'][k]} |")
    for k in sorted(pf["reasoning_complete"]):
        w(f"| reasoning_complete = {k} | {pf['reasoning_complete'][k]} |")
    w("")
    return "\n".join(L)


# ---------------------------------------------------------------------- main

def main() -> None:
    from transformers import AutoTokenizer

    from before_the_wait.generate import MODEL_ID, MODEL_REVISION
    from before_the_wait.label import merge_onsets
    from before_the_wait.summarize import (
        PARAGRAPH_ONSET_COLUMNS,
        TRACE_COLUMNS,
        analyze_trace,
    )

    assert minimum_events(DELTA) == M, "m no longer matches count_calculations.md"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)

    paths = sorted(TRACE_DIR.glob("gsm8k_*.json"))
    assert len(paths) == N_EXPECTED, f"expected {N_EXPECTED} trace files, found {len(paths)}"
    traces = [json.loads(p.read_text()) for p in paths]
    print(f"{len(paths)} traces")

    results = [analyze_trace(tokenizer, t) for t in traces]
    failures = [f for r in results for f in r.failures]
    hard_checked = sum(len(r.onsets) for r in results) + len(failures)
    print(f"hard check: {hard_checked} hits checked, {len(failures)} failures")
    if failures:
        print("\nSTOPPING: hard-check failures. Nothing written.")
        for f in failures[:20]:
            print(f"  {f.dataset}_{f.index:04d} tok={f.token_index} marker={f.marker!r} "
                  f"char={f.char_start} mid_token={f.mid_token} tail={f.decoded_tail!r}")
        sys.exit(1)

    trace_rows = [r.row for r in results]
    kept = [(r, t) for r, t in zip(results, traces) if not r.row["looped"]]

    onset_rows: list[dict[str, Any]] = []
    gaps: list[int] = []
    for r, t in kept:
        t0 = t["reasoning_token_span"][0]
        hit_idx = [o.token_index for o in r.onsets]
        for o, m in merge_onsets(r.onsets, "paragraph"):
            row = asdict(o)
            row["merged_into"] = "" if m is None else m
            onset_rows.append(row)
            if m is None:
                g = gap_to_previous_hit(o.token_index, hit_idx, t0)
                # Traceability: the same gap must be readable off the CSV row.
                assert g == gap_from_onset_row(row, t0), (t["index"], o.token_index)
                gaps.append(g)

    tt = trace_tally(trace_rows)
    ot = onset_tally(onset_rows, tt["kept"])
    assert ot["onsets"] == len(gaps)
    assert tt["kept"] + tt["looped"] == N_EXPECTED
    bands = band_table(gaps)
    pf = progress_facts(PROGRESS_LOG.read_text(), RUN_OUT.read_text())
    assert pf["rows"] == N_EXPECTED

    RESULTS.mkdir(exist_ok=True)
    with (RESULTS / "run_traces.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=TRACE_COLUMNS)
        w.writeheader()
        for row in trace_rows:
            w.writerow(row)
    with (RESULTS / "run_onsets_paragraph.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=PARAGRAPH_ONSET_COLUMNS)
        w.writeheader()
        for row in onset_rows:
            w.writerow(row)
    summary = build_run_summary(tt, ot, bands, pf, hard_checked)
    (RESULTS / "run_summary.md").write_text(summary + "\n")

    print(summary)
    print(f"\nwrote {RESULTS / 'run_traces.csv'}, run_onsets_paragraph.csv, run_summary.md")


if __name__ == "__main__":
    main()
