"""Step 9: split, band positives, 1:2 matched negatives (pre-registration §4–§6).

    .venv/bin/python scripts/build_examples.py

Decides WHICH positions are examples. Reads the run's token ids and text, the
lexicon hits and onsets, and the tokenizer (offline) for each token's kind.
No model, no hidden states, no probabilities, no probe, no rival.

Reused unchanged from src/: label.label_trace and merge_onsets (hits and
onsets), label.token_kind, label.negative_pool (cross-check of the candidate
rule), counts.band_positives, counts.marker_token_span, counts.gap_to_previous_hit.

Outputs: results/split.json, examples/examples.csv.gz (gitignored, rebuilt
here), results/examples_summary.md. Deterministic: byte-identical on rerun.
"""
from __future__ import annotations

import bisect
import csv
import gzip
import hashlib
import io
import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ["HF_HOME"] = str(REPO_ROOT / ".hf_cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, str(REPO_ROOT / "src"))

from before_the_wait.counts import (  # noqa: E402
    BANDS,
    band_positives,
    classify_band_position,
    gap_to_previous_hit,
)

TRACE_DIR = REPO_ROOT / "traces" / "run"
RESULTS = REPO_ROOT / "results"
EXAMPLES_DIR = REPO_ROOT / "examples"
EXAMPLES_PATH = EXAMPLES_DIR / "examples.csv.gz"

SPLIT_SEED = 20261019  # §4 / §18
N_KEPT = 292
N_TRAIN, N_VAL = 146, 44  # 50% and 15% of 292, nearest problem; test takes the rest
SPLITS = ("train", "validation", "test")
QUIET = 100  # §6 quiet zone (count_calculations.md §6)
RATIO = 2  # §6 1:2 matching
M = 555  # §11
CONTACT_DATE = "2026-11-14"
PROJECTED_08B = {"2-5": 869.4, "6-10": 831.2, "11-20": 795.2, "21-40": 717.8}

# Distance sets: the gate (d = 1, P3) and the four bands, each matched on its own.
DISTANCE_SETS: list[tuple[str, int, int]] = [("gate", 1, 1)] + list(BANDS)
BAND_NAMES = [b[0] for b in BANDS]

SPLIT_CALLS = [
    "kept = sorted gsm8k indices with looped == False in results/run_traces.csv",
    "order = kept[:]; random.Random(20261019).shuffle(order)",
    "train = order[:146]; validation = order[146:190]; test = order[190:]",
]

EXAMPLE_COLUMNS = [
    "distance_set", "split", "label", "trace_index", "t", "kind",
    "onset", "d", "pos_trace_index", "pos_t", "same_trace",
]


# ---------------------------------------------------------------- pure helpers

def make_split(kept: list[int], seed: int = SPLIT_SEED,
               n_train: int = N_TRAIN, n_val: int = N_VAL) -> dict[str, list[int]]:
    order = sorted(kept)
    random.Random(seed).shuffle(order)
    return {
        "train": order[:n_train],
        "validation": order[n_train:n_train + n_val],
        "test": order[n_train + n_val:],
    }


def negative_candidates(t0: int, t1: int, hit_indices: list[int],
                        spans: list[tuple[int, int]], lookahead: int = QUIET) -> list[int]:
    """§6 negative candidates, ascending: t in the reasoning span [t0, t1),
    t + lookahead also inside it (full lookahead), no hit of any kind (onset or
    merged) at t+1 … t+lookahead, and t not inside any marker span."""
    in_span = set()
    for s, e in spans:
        in_span.update(range(s, e))
    out = []
    for t in range(t0, t1):
        if t + lookahead >= t1:  # same edge as label.negative_pool
            continue
        if t in in_span:
            continue
        if not quiet_after(t, hit_indices, lookahead):
            continue
        out.append(t)
    return out


def quiet_after(t: int, sorted_hits: list[int], lookahead: int = QUIET) -> bool:
    """True if no hit lies at t+1 … t+lookahead (hits sorted ascending)."""
    i = bisect.bisect_right(sorted_hits, t)
    return i == len(sorted_hits) or sorted_hits[i] > t + lookahead


def match_set(positives: list[dict[str, Any]],
              candidates: dict[tuple[int, str], list[int]],
              split_of: dict[int, str],
              seed: int = SPLIT_SEED,
              ratio: int = RATIO) -> tuple[list[dict], list[dict], list[dict]]:
    """§6 matching for ONE distance set, with D18.

    positives: dicts with split, trace_index, t, kind (any other keys carried).
    candidates: (trace_index, kind) -> ascending candidate positions.
    Fresh rng and fresh "used" state per call. Per split (train, validation,
    test), traces ascending, positives ascending t: draw `ratio` negatives of
    the positive's kind, first rng.sample from the same trace's unused
    candidates, then the rest rng.sample from the sorted (trace, t) list of
    unused candidates of that kind in the other traces of the same split. If
    still short (D18) the positive is dropped and anything drawn for it goes
    back to the pool.

    Returns (kept positives, negatives, dropped positives).
    """
    rng = random.Random(seed)
    pool = {k: list(v) for k, v in candidates.items()}  # sorted, consumed in place
    traces_in = defaultdict(list)
    for tr, sp in split_of.items():
        traces_in[sp].append(tr)
    for sp in traces_in:
        traces_in[sp].sort()

    kept, negatives, dropped = [], [], []
    for sp in SPLITS:
        mine = sorted((p for p in positives if p["split"] == sp),
                      key=lambda p: (p["trace_index"], p["t"]))
        for p in mine:
            tr, kind = p["trace_index"], p["kind"]
            same = pool.get((tr, kind), [])
            k_same = min(ratio, len(same))
            drawn = [(tr, x) for x in rng.sample(same, k_same)]
            if k_same < ratio:
                other = [(o, x) for o in traces_in[sp] if o != tr
                         for x in pool.get((o, kind), [])]
                need = ratio - k_same
                if len(other) < need:
                    dropped.append(p)  # drawn negatives were never removed
                    continue
                drawn += rng.sample(other, need)
            for o, x in drawn:
                pool[(o, kind)].remove(x)
                negatives.append({
                    "split": sp, "trace_index": o, "t": x, "kind": kind,
                    "pos_trace_index": tr, "pos_t": p["t"], "same_trace": o == tr,
                })
            kept.append(p)
    return kept, negatives, dropped


def clean_onsets(gaps: dict[tuple[int, int], int], hi: int) -> set[tuple[int, int]]:
    """Onsets (trace, onset t) whose previous hit or reasoning start lies more
    than `hi` tokens back: the e_b definition."""
    return {k for k, g in gaps.items() if g > hi}


def write_gz_csv(path: Path, columns: list[str], rows: list[dict]) -> bytes:
    """Deterministic gzip (mtime 0, no filename) so reruns are byte-identical."""
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=columns, lineterminator="\n")
    w.writeheader()
    for r in rows:
        w.writerow(r)
    raw = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
        gz.write(buf.getvalue().encode())
    data = raw.getvalue()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return data


# ---------------------------------------------------------------------- main

def main() -> None:
    from transformers import AutoTokenizer

    from before_the_wait.align import prefix_lengths
    from before_the_wait.counts import marker_token_span
    from before_the_wait.generate import MODEL_ID, MODEL_REVISION
    from before_the_wait.label import (
        TOKEN_KINDS,
        label_trace,
        merge_onsets,
        negative_pool,
        token_kind,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)

    # ---------- 1. split ----------
    trace_rows = list(csv.DictReader((RESULTS / "run_traces.csv").open()))
    kept_idx = sorted(int(r["index"]) for r in trace_rows if r["looped"] == "False")
    assert len(kept_idx) == N_KEPT, len(kept_idx)
    split = make_split(kept_idx)
    split_of = {i: sp for sp in SPLITS for i in split[sp]}
    assert [len(split[s]) for s in SPLITS] == [146, 44, 102]
    assert sorted(split_of) == kept_idx and len(split_of) == N_KEPT  # disjoint, covers kept

    # CSV hits, for the consistency check against fresh labeling.
    csv_hits: dict[int, list[tuple[int, str, str]]] = defaultdict(list)
    for r in csv.DictReader((RESULTS / "run_onsets_paragraph.csv").open()):
        csv_hits[int(r["index"])].append((int(r["token_index"]), r["marker"], r["merged_into"]))

    kind_cache: dict[int, str] = {}

    def kind_of(tok: int) -> str:
        if tok not in kind_cache:
            kind_cache[tok] = token_kind(tokenizer.decode([tok], skip_special_tokens=False))
        return kind_cache[tok]

    positives: dict[str, list[dict]] = {name: [] for name, _, _ in DISTANCE_SETS}
    candidates: dict[tuple[int, str], list[int]] = {}
    gaps: dict[tuple[int, int], int] = {}
    onsets_by_split: Counter[str] = Counter()
    csv_positive_totals: Counter[str] = Counter()
    n_onsets = 0

    for idx in kept_idx:
        tr = json.loads((TRACE_DIR / f"gsm8k_{idx:04d}.json").read_text())
        ids = tr["output_token_ids"]
        t0, t1 = tr["reasoning_token_span"]
        sp = split_of[idx]

        hits, failures = label_trace(
            tokenizer, dataset=tr["dataset"], index=idx, ids=ids,
            output_text=tr["output_text"], prefix_len=prefix_lengths(tokenizer, ids),
            reasoning_char_span=tuple(tr["reasoning_char_span"]),
            reasoning_token_span=(t0, t1),
        )
        assert not failures, (idx, failures)
        merged = merge_onsets(hits, "paragraph")
        fresh = [(o.token_index, o.marker, "" if m is None else str(m)) for o, m in merged]
        assert fresh == csv_hits[idx], f"hits differ from run_onsets_paragraph.csv in {idx}"

        hit_idx = [o.token_index for o in hits]
        spans = [marker_token_span(tokenizer, ids, o.token_index, o.marker) for o in hits]
        onset_ts = [o.token_index for o, m in merged if m is None]
        n_onsets += len(onset_ts)
        onsets_by_split[sp] += len(onset_ts)

        for ot in onset_ts:
            gaps[(idx, ot)] = gap_to_previous_hit(ot, hit_idx, t0)
            for name, lo, hi in DISTANCE_SETS:
                for t in band_positives(ot, lo, hi, hit_idx, spans, t0):
                    positives[name].append({
                        "split": sp, "trace_index": idx, "t": t, "kind": kind_of(ids[t]),
                        "onset": ot, "d": ot - t,
                    })

        # Independent CSV path for the positive totals.
        c_hit = [h[0] for h in csv_hits[idx]]
        c_spans = [marker_token_span(tokenizer, ids, h[0], h[1]) for h in csv_hits[idx]]
        for h in csv_hits[idx]:
            if h[2] != "":
                continue
            for name, lo, hi in DISTANCE_SETS:
                csv_positive_totals[name] += sum(
                    1 for d in range(lo, hi + 1)
                    if classify_band_position(h[0] - d, h[0], c_hit, c_spans, t0) is None)

        # Negative candidates, cross-checked against label.negative_pool (all hits).
        cand = negative_candidates(t0, t1, hit_idx, spans)
        pool_counts, _ = negative_pool(tokenizer, ids, (t0, t1), set(hit_idx), QUIET)
        in_span = {x for s, e in spans for x in range(s, e)}
        assert hit_idx == sorted(hit_idx)
        no_span_filter = Counter(kind_of(ids[t]) for t in range(t0, t1)
                                 if t + QUIET < t1 and quiet_after(t, hit_idx))
        assert no_span_filter == pool_counts, f"candidate rule differs from negative_pool in {idx}"
        onset_set = set(onset_ts)
        for t in cand:
            assert t0 <= t and t + QUIET < t1 and t not in in_span
            assert not any(t < h <= t + QUIET for h in hit_idx)
            # never a positive, never 41–100 before an onset
            assert not any(0 < o - t <= QUIET for o in onset_set)
            candidates.setdefault((idx, kind_of(ids[t])), []).append(t)

    csv_onsets = sum(1 for v in csv_hits.values() for h in v if h[2] == "")
    assert n_onsets == csv_onsets == 2519, (n_onsets, csv_onsets)
    for name in positives:
        assert len(positives[name]) == csv_positive_totals[name], name
        keys = [(p["trace_index"], p["t"]) for p in positives[name]]
        assert len(keys) == len(set(keys)), f"position in two onsets in {name}"

    # ---------- 4. matching ----------
    rows: list[dict] = []
    stats: dict[str, dict] = {}
    for name, lo, hi in DISTANCE_SETS:
        kept_p, negs, dropped = match_set(positives[name], candidates, split_of)
        # acceptance asserts
        per_pos = Counter((n["pos_trace_index"], n["pos_t"]) for n in negs)
        kind_by_pos = {(p["trace_index"], p["t"]): p["kind"] for p in kept_p}
        assert set(per_pos) == set(kind_by_pos) and all(v == RATIO for v in per_pos.values())
        neg_keys = [(n["trace_index"], n["t"]) for n in negs]
        assert len(neg_keys) == len(set(neg_keys)), f"negative reused in {name}"
        for n in negs:
            assert n["kind"] == kind_by_pos[(n["pos_trace_index"], n["pos_t"])]
            assert split_of[n["trace_index"]] == n["split"] == split_of[n["pos_trace_index"]]
        stats[name] = {"kept": kept_p, "negs": negs, "dropped": dropped,
                       "lo": lo, "hi": hi}
        for p in sorted(kept_p, key=lambda p: (SPLITS.index(p["split"]), p["trace_index"], p["t"])):
            rows.append({"distance_set": name, "split": p["split"], "label": 1,
                         "trace_index": p["trace_index"], "t": p["t"], "kind": p["kind"],
                         "onset": p["onset"], "d": p["d"], "pos_trace_index": "",
                         "pos_t": "", "same_trace": ""})
        for n in negs:
            rows.append({"distance_set": name, "split": n["split"], "label": 0,
                         "trace_index": n["trace_index"], "t": n["t"], "kind": n["kind"],
                         "onset": "", "d": "", "pos_trace_index": n["pos_trace_index"],
                         "pos_t": n["pos_t"], "same_trace": n["same_trace"]})

    # ---------- outputs ----------
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "split.json").write_text(json.dumps({
        "seed": SPLIT_SEED,
        "calls": SPLIT_CALLS,
        "train": split["train"],
        "validation": split["validation"],
        "test": split["test"],
    }, indent=1) + "\n")
    data = write_gz_csv(EXAMPLES_PATH, EXAMPLE_COLUMNS, rows)
    sha = hashlib.sha256(data).hexdigest()

    summary = build_summary(split, onsets_by_split, gaps, stats, TOKEN_KINDS, sha, len(rows))
    (RESULTS / "examples_summary.md").write_text(summary + "\n")
    print(summary)


def build_summary(split, onsets_by_split, gaps, stats, kinds, sha, n_rows) -> str:
    L: list[str] = []
    w = L.append
    w("# Examples summary: split, band positives, 1:2 matched negatives")
    w("")
    w("Computed by `scripts/build_examples.py` from `traces/run/`, "
      "`results/run_traces.csv` and `results/run_onsets_paragraph.csv`.")
    w("")
    w("## a. Data contact")
    w("")
    w("| Date | What was touched | What was not |")
    w("|---|---|---|")
    w(f"| {CONTACT_DATE} | Token ids, text and lexicon hits of the {N_KEPT} kept traces; "
      f"example positions selected (split, band positives, matched negatives) and each "
      f"position's token kind. | No hidden states, no probabilities, no probe, no rival. |")
    w("")

    w("## b. Split (seed 20261019)")
    w("")
    w("| split | problems | onsets |")
    w("|---|---|---|")
    for sp in SPLITS:
        w(f"| {sp} | {len(split[sp])} | {onsets_by_split[sp]} |")
    w(f"| all | {sum(len(split[s]) for s in SPLITS)} | {sum(onsets_by_split.values())} |")
    w("")

    def with_pos(st, sp):
        return {(p["trace_index"], p["onset"]) for p in st["kept"] if p["split"] == sp}

    def clean_in(st, sp):
        return {k for k in clean_onsets(gaps, st["hi"]) if split_of_global[k[0]] == sp}

    split_of_global = {i: sp for sp in SPLITS for i in split[sp]}

    w("## c. Per distance set × split")
    w("")
    w("Clean onset: previous hit of any kind, or the reasoning start, more than the "
      "set's upper bound back (the e_b definition).")
    w("")
    w("| set | split | onsets with ≥ 1 positive | clean onsets | positives | negatives |")
    w("|---|---|---|---|---|---|")
    for name, st in stats.items():
        for sp in SPLITS:
            npos = sum(1 for p in st["kept"] if p["split"] == sp)
            nneg = sum(1 for n in st["negs"] if n["split"] == sp)
            w(f"| {name} | {sp} | {len(with_pos(st, sp))} | {len(clean_in(st, sp))} | "
              f"{npos} | {nneg} |")
    w("")

    w("## d. Test onsets per band against m = 555")
    w("")
    w("Clean test onsets (e_b definition) that keep at least one positive after any D18 drop.")
    w("")
    w("| band | test onsets | ≥ 555? | 08b projection |")
    w("|---|---|---|---|")
    for name in BAND_NAMES:
        st = stats[name]
        n = len(clean_in(st, "test") & with_pos(st, "test"))
        w(f"| {name} | {n} | {'yes' if n >= M else '**no**'} | {PROJECTED_08B[name]} |")
    w("")

    w("## e. Share of negatives from a different trace")
    w("")
    w("| set | " + " | ".join(SPLITS) + " | all |")
    w("|---|" + "---|" * (len(SPLITS) + 1))
    for name, st in stats.items():
        cells = []
        for sp in list(SPLITS) + ["all"]:
            ns = [n for n in st["negs"] if sp == "all" or n["split"] == sp]
            cross = sum(1 for n in ns if not n["same_trace"])
            cells.append(f"{cross}/{len(ns)} = {cross / len(ns):.4f}" if ns else "–")
        w(f"| {name} | " + " | ".join(cells) + " |")
    w("")

    w("## f. Positives and negatives by token kind")
    w("")
    w("| set | " + " | ".join(f"`{k}` pos / neg" for k in kinds) + " |")
    w("|---|" + "---|" * len(kinds))
    for name, st in stats.items():
        pc = Counter(p["kind"] for p in st["kept"])
        nc = Counter(n["kind"] for n in st["negs"])
        w(f"| {name} | " + " | ".join(f"{pc.get(k, 0)} / {nc.get(k, 0)}" for k in kinds) + " |")
    w("")

    w("## g. D18 drops")
    w("")
    any_drop = any(st["dropped"] for st in stats.values())
    if not any_drop:
        w("D18 did not fire.")
    else:
        w("| set | split | kind | positives dropped |")
        w("|---|---|---|---|")
        for name, st in stats.items():
            c = Counter((p["split"], p["kind"]) for p in st["dropped"])
            for (sp, k), n in sorted(c.items(), key=lambda x: (SPLITS.index(x[0][0]), x[0][1])):
                w(f"| {name} | {sp} | `{k}` | {n} |")
        w("")
        w("Onsets that lost all their positives in a set:")
        w("")
        w("| set | split | trace | onset |")
        w("|---|---|---|---|")
        for name, st in stats.items():
            had = {(p["trace_index"], p["onset"], p["split"]) for p in st["dropped"]}
            kept = {(p["trace_index"], p["onset"]) for p in st["kept"]}
            for tr, ot, sp in sorted(had):
                if (tr, ot) not in kept:
                    w(f"| {name} | {sp} | {tr} | {ot} |")
    w("")

    w("## h. examples/examples.csv.gz")
    w("")
    w(f"- sha256 `{sha}`")
    w(f"- {n_rows:,} rows (one per example per distance set, header excluded)")
    return "\n".join(L)


if __name__ == "__main__":
    main()
