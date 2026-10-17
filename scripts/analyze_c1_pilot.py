"""c1 pilot analysis: traces -> results/c1_traces.csv, c1_onsets.csv, c1_summary.md.

    .venv/bin/python scripts/analyze_c1_pilot.py

Reads the saved traces and the tokenizer. Never loads the model, never touches
hidden states. Separate from generation so re-running an analysis costs
seconds, not an hour.

If any marker fails the hard check the script stops without writing results —
the review's instruction is to report failures, not to work around them.
"""
from __future__ import annotations

import csv
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ["HF_HOME"] = str(REPO_ROOT / ".hf_cache")
sys.path.insert(0, str(REPO_ROOT / "src"))

from transformers import AutoTokenizer  # noqa: E402

from before_the_wait.generate import (  # noqa: E402
    MODEL_ID,
    MODEL_REVISION,
    THINK_CLOSE,
    THINK_OPEN,
)
from before_the_wait.label import merge_onsets  # noqa: E402
from before_the_wait.summarize import (  # noqa: E402
    ONSET_COLUMNS,
    PARAGRAPH_ONSET_COLUMNS,
    TRACE_COLUMNS,
    analyze_trace,
    build_addendum,
    build_summary,
)

TRACE_DIR = REPO_ROOT / "traces" / "pilot"
RESULTS = REPO_ROOT / "results"

# The handoff's closing three-sentence answer. Written against the 30-trace
# pilot below; if this script is ever re-run on a different set of traces,
# rewrite it rather than letting it stand.
CONCLUSION = """## The three-sentence answer

**Do markers fire, and how densely?** They fire readily on both datasets —
649 hits across 30 traces, 10.95 per 1k reasoning tokens on GSM8K (273 hits,
mean 18.2 and median 7.0 per trace) and 9.54 per 1k on MATH500 (376 hits, mean
25.1 and median 19.0) — so the kill condition of fewer than ~1 hit per trace in
both datasets is not remotely met, though the gap between mean and median on
GSM8K is a warning that a few long truncated traces carry much of the total.

**Which markers dominate, and what does the lexicon miss?** `wait` is the
overwhelming majority at 64.9% of pooled hits (68.1% GSM8K, 62.5% MATH500),
with `hmm` (12.2%) and `let me think` (8.0%) the only other markers above 5%;
the most frequent paragraph openers the lexicon does not match are `so` (214),
`but` (168) and `alternatively` (60), of which `but` was deliberately
calibrated out of the FrictionMap lexicon and `alternatively` is the clearest
unclaimed re-evaluation candidate.

**Is GSM8K dense enough to be the primary dataset?** Yes on these counts — 273
onsets from 15 problems, 47.3% of pooled hits paragraph-initial and 83.8% of
within-trace gaps at least 5 tokens, which is the separation the distance bands
need — but the median of 7 hits per trace means a per-problem event budget
should be planned off the median rather than the mean, and the three truncated
GSM8K traces (91, 55 and 36 hits) should be checked for looping before their
onsets are counted as ordinary events.
"""


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)

    paths = sorted(p for p in TRACE_DIR.glob("*.json") if not p.name.startswith("_"))
    print(f"{len(paths)} traces")
    traces = [json.loads(p.read_text()) for p in paths]

    results = []
    for t, path in zip(traces, paths):
        r = analyze_trace(tokenizer, t)
        results.append(r)
        print(
            f"  {path.name}: {r.row['marker_count']:3d} hits, "
            f"{r.row['reasoning_tokens']:4d} reasoning tokens, "
            f"{len(r.failures)} hard-check failures"
        )

    failures = [f for r in results for f in r.failures]
    total_hits = sum(len(r.onsets) for r in results) + len(failures)
    print(f"\nhard check: {total_hits} hits checked, {len(failures)} failures")

    if failures:
        print("\nSTOPPING: hard-check failures. No results written.")
        for f in failures[:20]:
            print(f"  {f.dataset}_{f.index:04d} tok={f.token_index} marker={f.marker!r} "
                  f"char={f.char_start} mid_token={f.mid_token} tail={f.decoded_tail!r}")
        (RESULTS / "_hard_check_failures.json").write_text(
            json.dumps([asdict(f) for f in failures], indent=1, ensure_ascii=False)
        )
        sys.exit(1)

    RESULTS.mkdir(exist_ok=True)

    with (RESULTS / "c1_traces.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=TRACE_COLUMNS)
        w.writeheader()
        for r in results:
            w.writerow(r.row)

    with (RESULTS / "c1_onsets.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=ONSET_COLUMNS)
        w.writeheader()
        for r in results:
            for o in r.onsets:
                w.writerow(asdict(o))

    by_dataset: dict[str, list] = {}
    for r in results:
        by_dataset.setdefault(r.row["dataset"], []).append(r)

    meta = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "n_traces": len(results),
        "prompt_suffix": traces[0]["prompt_text"][-40:],
        "think_ids": tokenizer.encode(THINK_OPEN, add_special_tokens=False),
        "close_ids": tokenizer.encode(THINK_CLOSE, add_special_tokens=False),
        "hard_check_total": total_hits,
        "hard_check_failures": len(failures),
    }
    summary = build_summary(results, by_dataset, meta)

    mismatches = [r.mismatch for r in results if r.mismatch]
    lines = [summary, "", "## Answer mismatches (every one, per the review)", ""]
    if mismatches:
        lines += ["| dataset | index | gold | boxed |", "|---|---|---|---|"]
        for m in mismatches:
            lines.append(f"| {m['dataset']} | {m['index']} | `{m['gold']}` | `{m['boxed']}` |")
    else:
        lines.append("None — every trace matched its gold answer.")
    lines += ["", CONCLUSION, ""]

    # --- Part A (handoff 03): D10 exclusion + D11 paragraph-rule onsets ---
    looped = [r for r in results if r.row["looped"]]
    kept = [r for r in results if not r.row["looped"]]
    merged = {
        (r.row["dataset"], r.row["index"]): merge_onsets(r.onsets, "paragraph")
        for r in kept
    }
    window_counts = {
        n: sum(
            1
            for r in kept
            for _, m in merge_onsets(r.onsets, "window", n=n)
            if m is None
        )
        for n in (5, 10, 20)
    }

    with (RESULTS / "c1_onsets_paragraph.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=PARAGRAPH_ONSET_COLUMNS)
        w.writeheader()
        for r in kept:
            for o, m in merged[(r.row["dataset"], r.row["index"])]:
                row = asdict(o)
                row["merged_into"] = "" if m is None else m
                w.writerow(row)

    lines.append(build_addendum(kept, looped, merged, window_counts))

    (RESULTS / "c1_summary.md").write_text("\n".join(lines))
    print(f"\nwrote {RESULTS / 'c1_traces.csv'}")
    print(f"wrote {RESULTS / 'c1_onsets.csv'}")
    print(f"wrote {RESULTS / 'c1_onsets_paragraph.csv'}")
    print(f"wrote {RESULTS / 'c1_summary.md'} (with D10/D11 addendum)")
    n_onsets = sum(1 for k in merged for _, m in merged[k] if m is None)
    print(f"\nD10: {len(looped)} looped traces excluded, {len(kept)} kept")
    print(f"D11: paragraph-rule onsets (non-looped) = {n_onsets}")
    print(f"     window-rule onsets (non-looped)    = {window_counts}")


if __name__ == "__main__":
    main()
